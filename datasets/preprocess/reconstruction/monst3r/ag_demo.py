import argparse
import copy
import functools
import math
import os

import gradio
import matplotlib.pyplot as pl
import numpy as np
import torch
import pickle

from tqdm import tqdm

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images, load_prev_video_results, rgb, enlarge_seg_masks
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer

pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


class AgMonst3r:

    def __init__(
            self,
            args,
            datapath,
            ag_root_dir,
    ):
        self.datapath = datapath
        self.frames_path = os.path.join(self.datapath, "frames")
        self.annotations_path = os.path.join(self.datapath, "annotations")
        self.video_list = sorted(os.listdir(self.frames_path))
        self.gt_annotations = sorted(os.listdir(self.annotations_path))
        print("Total number of ground truth annotations: ", len(self.gt_annotations))

        video_id_frame_id_list_pkl_file_path = os.path.join(self.datapath, "4d_video_frame_id_list.pkl")
        if os.path.exists(video_id_frame_id_list_pkl_file_path):
            with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
                self.video_id_frame_id_list = pickle.load(f)
        else:
            assert False, f"Please generate {video_id_frame_id_list_pkl_file_path} first"
        self.ag_root_dir = ag_root_dir
        self.ag_monst3r_root = os.path.join(ag_root_dir, "ag4D", "monst3r")

        # -------------------------------- MONST3R PARAMS --------------------------------
        self.weights_path = args.weights
        self.croco_model = AsymmetricCroCo3DStereo.from_pretrained(self.weights_path).to(args.device)
        self.croco_model.eval()

    def get_3D_model_from_scene(
            self,
            outdir,
            silent,
            scene,
            min_conf_thr=3,
            as_pointcloud=False,
            mask_sky=False,
            clean_depth=False,
            transparent_cams=False,
            cam_size=0.05,
            show_cam=True,
            save_name=None,
            thr_for_init_conf=True
    ):
        """
        extract 3D_model (glb file) from a reconstructed scene
        """
        if scene is None:
            return None
        # post processes
        if clean_depth:
            scene = scene.clean_pointcloud()
        if mask_sky:
            scene = scene.mask_sky()

        # get optimized values from scene
        rgbimg = scene.imgs
        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()
        # 3D pointcloud from depthmap, poses and intrinsics
        pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
        scene.min_conf_thr = min_conf_thr
        scene.thr_for_init_conf = thr_for_init_conf
        msk = to_numpy(scene.get_masks())
        cmap = pl.get_cmap('viridis')
        cam_color = [cmap(i / len(rgbimg))[:3] for i in range(len(rgbimg))]
        cam_color = [(255 * c[0], 255 * c[1], 255 * c[2]) for c in cam_color]
        return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                           transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam,
                                           silent=silent, save_name=save_name,
                                           cam_color=cam_color)

    def video_monst3r_eval(self, video_id, img_path_list, args):
        # Call the function with default parameters
        scene, imgs = self.get_video_reconstructed_scene(
            args=args,
            model=self.croco_model,
            device=args.device,
            silent=args.silent,
            image_size=args.image_size,
            video_id=video_id,
            filelist=img_path_list,
            schedule='linear',
            niter=args.niter,
            scenegraph_type='swinstride',
            winsize=5,
            refid=0,
            temporal_smoothing_weight=0.01,
            translation_weight='1.0',
            shared_focal=True,
            flow_loss_weight=0.01,
            flow_loss_start_iter=0.1,
            flow_loss_threshold=25,
            use_gt_mask=args.use_gt_davis_masks,
            fps=args.fps,
            num_frames=args.num_frames,
        )
        return scene, imgs

    def get_video_reconstructed_scene(
            self,
            args,
            video_id,
            model,
            device,
            silent,
            image_size,
            filelist,
            schedule,
            niter,
            scenegraph_type,
            winsize,
            refid,
            temporal_smoothing_weight,
            translation_weight,
            shared_focal,
            flow_loss_weight,
            flow_loss_start_iter,
            flow_loss_threshold,
            use_gt_mask,
            fps,
            num_frames
    ):
        """
        from a list of images, run dust3r inference, global aligner.
        then run get_3D_model_from_scene
        """
        translation_weight = float(translation_weight)
        video_root_dir = os.path.join(self.ag_monst3r_root, video_id)
        dynamic_mask_root_dir = os.path.join(video_root_dir, "dynamic_masks")
        if not os.path.exists(dynamic_mask_root_dir):
            os.makedirs(dynamic_mask_root_dir)

        print("----------------------------------------------------------------------------")
        print("Step-1: Loading images and running dust3r inference")

        # For global alignment using a sliding window approach
        # If the number of frames in the video is greater than the window size, it will be run multiple times.
        if args.window_wise and args.prev_output_dir is not None:
            print("Using previous video results")
            prev_num_frames = int(args.window_size * args.window_overlap_ratio)
            prev_video_results = load_prev_video_results(
                args.prev_output_dir,
                num_frames=prev_num_frames,
                index=args.prev_output_index
            )
            imgs = load_images(
                filelist,
                size=image_size,
                verbose=not silent,
                dynamic_mask_root=dynamic_mask_root_dir,
                fps=fps,
                num_frames=num_frames,
                imgs=prev_video_results['imgs']
            )
        else:
            prev_video_results = None
            print("NOT using previous video results")
            imgs = load_images(
                filelist,
                size=image_size,
                verbose=not silent,
                dynamic_mask_root=dynamic_mask_root_dir,
                fps=fps,
                num_frames=num_frames
            )

        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
        if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
            scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
        elif scenegraph_type == "oneref":
            scenegraph_type = scenegraph_type + "-" + str(refid)

        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=args.batch_size, verbose=not silent)

        print("----------------------------------------------------------------------------")
        print("Step-2: Running global alignment")

        # TODO YYJ del model
        if len(imgs) > 2:
            mode = GlobalAlignerMode.PointCloudOptimizer
            scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal=shared_focal,
                                   temporal_smoothing_weight=temporal_smoothing_weight,
                                   translation_weight=translation_weight,
                                   flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter,
                                   flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                                   num_total_iter=niter, empty_cache=len(filelist) > 72,
                                   batchify=not (args.not_batchify or args.window_wise),
                                   window_wise=args.window_wise, window_size=args.window_size,
                                   window_overlap_ratio=args.window_overlap_ratio,
                                   prev_video_results=prev_video_results)
        else:
            mode = GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
        lr = 0.01

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            if args.window_wise:
                scene.compute_window_wise_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
            else:
                scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

        if args.window_wise and args.prev_output_dir is not None:
            scene.clean_prev_results()

        # outfile = self.get_3D_model_from_scene(
        #     video_root_dir,
        #     silent,
        #     scene,
        #     min_conf_thr,
        #     as_pointcloud,
        #     mask_sky,
        #     clean_depth,
        #     transparent_cams,
        #     cam_size,
        #     show_cam
        # )

        poses = scene.save_tum_poses(f'{video_root_dir}/pred_traj.txt')
        K = scene.save_intrinsics(f'{video_root_dir}/pred_intrinsics.txt')
        depth_maps = scene.save_depth_maps(video_root_dir)
        dynamic_masks = scene.save_dynamic_masks(video_root_dir)
        conf = scene.save_conf_maps(video_root_dir)
        init_conf = scene.save_init_conf_maps(video_root_dir)
        rgbs = scene.save_rgb_imgs(video_root_dir)
        enlarge_seg_masks(video_root_dir, kernel_size=5 if use_gt_mask else 3)

        # also return rgb, depth and confidence imgs
        # depth is normalized with the max value for all images
        # we apply the jet colormap on the confidence maps
        rgbimg = scene.imgs
        depths = to_numpy(scene.get_depthmaps())
        confs = to_numpy([c for c in scene.im_conf])
        init_confs = to_numpy([c for c in scene.init_conf_maps])
        cmap = pl.get_cmap('jet')
        depths_max = max([d.max() for d in depths])
        depths = [cmap(d / depths_max) for d in depths]
        confs_max = max([d.max() for d in confs])
        confs = [cmap(d / confs_max) for d in confs]
        init_confs_max = max([d.max() for d in init_confs])
        init_confs = [cmap(d / init_confs_max) for d in init_confs]

        imgs = []
        for i in range(len(rgbimg)):
            imgs.append(rgbimg[i])
            imgs.append(rgb(depths[i]))
            imgs.append(rgb(confs[i]))
            imgs.append(rgb(init_confs[i]))

        print("----------------------------------------------------------------------------")
        print("Step-3: Running dynamic mask computation")

        # if two images, and the shape is same, we can compute the dynamic mask
        if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
            motion_mask_thre = 0.35
            error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=args.output_dir,
                                                         motion_mask_thre=motion_mask_thre)
            # imgs.append(rgb(error_map))
            # apply threshold on the error map
            normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
            error_map_max = normalized_error_map.max()
            error_map = cmap(normalized_error_map / error_map_max)
            imgs.append(rgb(error_map))
            binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
            imgs.append(rgb(binary_error_map * 255))

        return scene, imgs

    def generate_data(self, args):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)

            # Check if the video belongs to a specific split
            split = self.get_video_belongs_to_split(video_id)
            if split != args.split:
                print(f"Video {video_id} does not belong to split {args.split}, skipping...")
                continue

            if os.path.exists(video_frames_path) and len(os.listdir(video_frames_path)) == 0:
                print(f"Video {video_id} has no frames, skipping...")
                continue

            img_paths = []
            frame_id_list = self.video_id_frame_id_list[video_id]
            frame_id_list = sorted(list(np.unique(frame_id_list)))

            if len(frame_id_list) > 200:
                print(f"Video {video_id} has more than 250 frames, skipping...")
                continue

            video_skip_counter = 0
            for frame_id in frame_id_list:
                # Check if Monst3r output directory already exists
                ag_monst3r_output_path = os.path.join(self.ag_monst3r_root, video_id)
                if os.path.exists(ag_monst3r_output_path) and len(os.listdir(ag_monst3r_output_path)) > 0:
                    video_skip_counter += 1
                    continue
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            # If the number of frames is larger than 50 then set to automatically execute using windows else use normal
            if len(img_paths) > 50:
                args.window_wise = True
                args.window_size = 50
                args.window_overlap_ratio = 0.5
                args.niter = 100
            else:
                args.window_wise = False
                args.window_size = 100
                args.niter = 200
                args.window_overlap_ratio = 0.5

            try:
                # print(f"Video {video_id} has {len(img_paths)} frames, skipped {video_skip_counter} frames.")
                if len(img_paths) == 0:
                    continue
                else:
                    self.video_monst3r_eval(video_id, img_paths, args)
            except RuntimeError as e:
                print(f"Error processing video {video_id}: {e}")
                # Delete the output directory if it exists
                ag_monst3r_output_path = os.path.join(self.ag_monst3r_root, video_id)
                if os.path.exists(ag_monst3r_output_path):
                    import shutil
                    shutil.rmtree(ag_monst3r_output_path)
                continue
            except Exception as e:
                print(f"Unexpected error processing video {video_id}: {e}")
                # Delete the output directory if it exists
                ag_monst3r_output_path = os.path.join(self.ag_monst3r_root, video_id)
                if os.path.exists(ag_monst3r_output_path):
                    import shutil
                    shutil.rmtree(ag_monst3r_output_path)
                continue

            # Clear the cache
            torch.cuda.empty_cache()
            print(f"Video {video_id} processed successfully.")
            print("-------------------------------------------------------------------------")

        print("Depth estimation completed for all videos.")

    @staticmethod
    def get_video_belongs_to_split(video_id):
        """
        Get the split that the video belongs to based on its ID.
        """
        first_letter = video_id[0]
        if first_letter.isdigit() and int(first_letter) < 5:
            return "04"
        elif first_letter.isdigit() and int(first_letter) >= 5:
            return "59"
        elif first_letter in "ABCD":
            return "AD"
        elif first_letter in "EFGH":
            return "EH"
        elif first_letter in "IJKL":
            return "IL"
        elif first_letter in "MNOP":
            return "MP"
        elif first_letter in "QRST":
            return "QT"
        elif first_letter in "UVWXYZ":
            return "UZ"


def main():
    parser = argparse.ArgumentParser()

    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")

    parser.add_argument("--ag_root_dir", type=str, default='/data/rohith/ag/')
    parser.add_argument("--niter", type=int, default=200, help="number of iterations")
    parser.add_argument("--datapath", type=str, help="Path to input images directory", default='/data/rohith/ag/')
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights",
                        default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--split", type=str, help="path to the model weights")
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt',
                        help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--prev_output_dir", type=str, default=None, help="previous output dir")
    parser.add_argument("--prev_output_index", type=int, default=None, help="previous output video index")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, help="Path to input images directory", default=None)
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False,
                        help='Use ground truth masks for DAVIS')
    parser.add_argument('--not_batchify', action='store_true', default=False,
                        help='Use non batchify mode for global optimization')
    parser.add_argument('--real_time', action='store_true', default=False, help='Realtime mode')
    parser.add_argument('--window_wise', action='store_true', default=False,
                        help='Use window wise mode for optimization')
    parser.add_argument('--window_size', type=int, default=100, help='Window size')
    parser.add_argument('--window_overlap_ratio', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')

    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=200, help='Maximum number of frames for video processing')

    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")

    args = parser.parse_args()

    # Init the AGMonst3r class
    ag_monst3r = AgMonst3r(args, args.ag_root_dir, args.datapath)

    ag_monst3r.generate_data(args)


if __name__ == "__main__":
    main()
