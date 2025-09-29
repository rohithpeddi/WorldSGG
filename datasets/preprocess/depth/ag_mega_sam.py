import argparse
import glob
import os
from pathlib import Path
import cv2
import kornia

from Depth_Anything.depth_anything.dpt import DPT_DINOv2
from Depth_Anything.depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
import imageio
import numpy as np
import torch
import pickle
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from PIL import Image

from UniDepth.unidepth.models import UniDepthV2

import sys

from cvd_opt.core.raft import RAFT
from cvd_opt.core.utils.utils import InputPadder
from cvd_opt.geometry_utils import NormalGenerator

sys.path.append("base/droid_slam")

import glob
from lietorch import SE3
from droid import Droid


def image_stream(
        image_list,
        mono_disp_list,
        scene_name,
        use_depth=False,
        aligns=None,
        K=None,
        stride=1,
):
    """image generator."""
    del scene_name, stride

    fx, fy, cx, cy = (
        K[0, 0],
        K[1, 1],
        K[0, 2],
        K[1, 2],
    )  # np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()

    for t, (image_file) in enumerate(image_list):
        image = cv2.imread(image_file)
        # depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 5000.
        # depth = np.float32(np.load(depth_file)) / 300.0
        # depth =  1. / pt_data["depth"]

        mono_disp = mono_disp_list[t]
        # mono_disp = np.float32(np.load(disp_file)) #/ 300.0
        depth = np.clip(
            1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
            1e-4,
            1e4,
        )
        depth[depth < 1e-2] = 0.0

        # breakpoint()
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]

        # if t == 4 or t == 29:
        # imageio.imwrite("debug/camel_%d.png"%t, image[..., ::-1])

        image = torch.as_tensor(image).permute(2, 0, 1)
        # print("image ", image.shape)
        # breakpoint()

        depth = torch.as_tensor(depth)
        depth = F.interpolate(
            depth[None, None], (h1, w1), mode="nearest-exact"
        ).squeeze()
        depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

        mask = torch.ones_like(depth)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        if use_depth:
            yield t, image[None], depth, intrinsics, mask
        else:
            yield t, image[None], intrinsics, mask


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return res


def resize_flow(flow, img_h, img_w):
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)
    return flow


class AgMegaSam:

    def __init__(self, datapath):
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

        # ------ Depth Anything parameters ------
        self.margin_width = 50
        self.caption_height = 60

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2

        self._depth_anything_root = os.path.join(self.datapath, 'ag4D', "mega_sam", "depth_anything")

        # ------- UniDepth parameters -------
        self._unidepth_root = os.path.join(self.datapath, 'ag4D', "mega_sam", "unidepth")
        self.LONG_DIM = 640

        # ------- Camera tracking parameters -------
        self._camera_tracking_root = os.path.join(self.datapath, 'ag4D', "mega_sam", "camera_tracking")

        self.mono_img_mismatch_counter = 0
        self.metric_mono_mismatch_counter = 0
        self.metric_img_mismatch_counter = 0

        self._flow_root = os.path.join(self.datapath, 'ag4D', "mega_sam", "preprocess_flow")
        self._flow_estimation_model = None
        self._flow_model = None

        # ------------------ CVD parameters ----------------
        self.ALPHA_MOTION = 0.25
        self.RESIZE_FACTOR = 0.5

        self._cvd_opt_root = os.path.join(self.datapath, 'ag4D', "mega_sam", "cvd_opt")
        os.makedirs(self._cvd_opt_root, exist_ok=True)

    # -------------------------- DEPTH ANYTHING --------------------------

    def _load_depth_anything_model(self, args):
        if args.encoder == 'vits':
            self.depth_anything = DPT_DINOv2(
                encoder='vits',
                features=64,
                out_channels=[48, 96, 192, 384],
                localhub=args.localhub,
            ).cuda()
        elif args.encoder == 'vitb':
            self.depth_anything = DPT_DINOv2(
                encoder='vitb',
                features=128,
                out_channels=[96, 192, 384, 768],
                localhub=args.localhub,
            ).cuda()
        else:
            self.depth_anything = DPT_DINOv2(
                encoder='vitl',
                features=256,
                out_channels=[256, 512, 1024, 1024],
                localhub=args.localhub,
            ).cuda()

        total_params = sum(param.numel() for param in self.depth_anything.parameters())
        print('Total parameters: {:.2f}M'.format(total_params / 1e6))
        self.depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'), strict=True)
        self.depth_anything.eval()

        # ------ Initialize transformations ------
        self._depth_anything_transforms = Compose([
            Resize(
                width=768,
                height=768,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='upper_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def video_depth_anything_estimation(self, video_id, img_paths):
        output_dir = os.path.join(self._depth_anything_root, video_id)

        if os.path.exists(output_dir):
            if len(os.listdir(output_dir)) == len(img_paths):
                print(f"Depth estimation already completed for {video_id}. Skipping...")
                return

        os.makedirs(output_dir, exist_ok=True)

        for filename in tqdm(img_paths, desc=f"Processing {video_id}"):
            raw_image = cv2.imread(filename)[..., :3]
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image.shape[:2]

            image = self._depth_anything_transforms({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).cuda()

            # start = timer()
            with torch.no_grad():
                depth = self.depth_anything(image)
            # end = timer()

            depth = F.interpolate(
                depth[None], (h, w), mode='bilinear', align_corners=False
            )[0, 0]
            depth_npy = np.float32(depth.cpu().numpy())

            np.save(
                os.path.join(output_dir, filename.split('/')[-1][:-4] + '.npy'),
                depth_npy,
            )

    def run_ag_depth_anything_estimation(self):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = self.video_id_frame_id_list[video_id]
            video_skip_counter = 0
            for frame_id in frame_id_list:
                # Check if depth anything output already exists
                depth_anything_output_path = os.path.join(self._depth_anything_root, video_id, f"{frame_id:06d}.npy")
                if os.path.exists(depth_anything_output_path):
                    video_skip_counter += 1
                    continue
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            # print(f"Video {video_id} has {len(img_paths)} frames, skipped {video_skip_counter} frames.")
            if len(img_paths) == 0:
                continue
            else:
                self.video_depth_anything_estimation(video_id, img_paths)
        print("Depth estimation completed for all videos.")

    # -------------------------- UNIDEPTH --------------------------

    def _load_unidepth_model(self, args):
        self._unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14",
                                                          revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._unidepth_model = self._unidepth_model.to(device)
        self._unidepth_model.eval()

    def video_unidepth_estimation(self, video_id, img_paths):
        output_dir = os.path.join(self._unidepth_root, video_id)

        if os.path.exists(output_dir):
            if len(os.listdir(output_dir)) == len(img_paths):
                print(f"Depth estimation already completed for {video_id}. Skipping...")
                return
            else:
                # Remove the existing directory if it is not empty and not complete
                os.rmdir(output_dir)
                print(f"Removing incomplete directory for {video_id}...")

        os.makedirs(output_dir, exist_ok=True)

        fovs = []
        for img_path in tqdm(img_paths):
            rgb = np.array(Image.open(img_path))[..., :3]
            if rgb.shape[1] > rgb.shape[0]:
                final_w, final_h = self.LONG_DIM, int(
                    round(self.LONG_DIM * rgb.shape[0] / rgb.shape[1])
                )
            else:
                final_w, final_h = (
                    int(round(self.LONG_DIM * rgb.shape[1] / rgb.shape[0])),
                    self.LONG_DIM,
                )
            rgb = cv2.resize(rgb, (final_w, final_h), cv2.INTER_AREA)  # .transpose(2, 0, 1)

            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
            # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
            # predict
            predictions = self._unidepth_model.infer(rgb_torch)
            fov_ = np.rad2deg(
                2
                * np.arctan(
                    predictions["depth"].shape[-1]
                    / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())
                )
            )
            depth = predictions["depth"][0, 0].cpu().numpy()
            # print(fov_)
            fovs.append(fov_)
            # breakpoint()
            np.savez(
                os.path.join(output_dir, img_path.split("/")[-1][:-4] + ".npz"),
                depth=np.float32(depth),
                fov=fov_,
            )

    def run_ag_unidepth_estimation(self):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = self.video_id_frame_id_list[video_id]
            for frame_id in frame_id_list:
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            self.video_unidepth_estimation(video_id, img_paths)

        print("Depth estimation completed for all videos.")

    # -------------------------- CAMERA TRACKING --------------------------
    def save_full_reconstruction(
            self, droid, full_traj, rgb_list, senor_depth_list, motion_prob, video_id
    ):
        """Save full reconstruction."""
        output_dir = os.path.join(self._camera_tracking_root, "reconstructions", video_id)
        os.makedirs(output_dir, exist_ok=True)

        t = full_traj.shape[0]
        images = np.array(rgb_list[:t])  # droid.video.images[:t].cpu().numpy()
        disps = 1.0 / (np.array(senor_depth_list[:t]) + 1e-6)

        poses = full_traj  # .cpu().numpy()
        intrinsics = droid.video.intrinsics[:t].cpu().numpy()


        np.save(f"{output_dir}/images.npy", images)
        np.save(f"{output_dir}/disps.npy", disps)
        np.save(f"{output_dir}/poses.npy", poses)
        np.save(f"{output_dir}/intrinsics.npy", intrinsics * 8.0)
        np.save(f"{output_dir}/motion_prob.npy", motion_prob)

        intrinsics = intrinsics[0] * 8.0
        poses_th = torch.as_tensor(poses, device="cpu")
        cam_c2w = SE3(poses_th).inv().matrix().numpy()

        K = np.eye(3)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]
        print("K ", K)
        print("img_data ", images.shape)
        print("disp_data ", disps.shape)

        max_frames = min(1000, images.shape[0])
        print(f"outputs/{video_id}_droid.npz")
        os.makedirs(f"{output_dir}/outputs", exist_ok=True)

        np.savez(
            f"{output_dir}/outputs/{video_id}_droid.npz",
            images=np.uint8(images[:max_frames, ::-1, ...].transpose(0, 2, 3, 1)),
            depths=np.float32(1.0 / disps[:max_frames, ...]),
            intrinsic=K,
            cam_c2w=cam_c2w[:max_frames],
        )

    def video_camera_tracking_estimation(self, video_id, image_list, args):
        tstamps = []
        rgb_list = []
        senor_depth_list = []

        # NOTE Mono is inverse depth, but metric-depth is depth!
        mono_disp_paths = sorted(
            glob.glob(
                os.path.join("%s/%s" % (args.mono_depth_path, video_id), "*.npy")
            )
        )
        metric_depth_paths = sorted(
            glob.glob(
                os.path.join("%s/%s" % (args.metric_depth_path, video_id), "*.npz")
            )
        )

        img_0 = cv2.imread(image_list[0])
        scales = []
        shifts = []
        mono_disp_list = []
        fovs = []
        for t, (mono_disp_file, metric_depth_file) in enumerate(
                zip(mono_disp_paths, metric_depth_paths)
        ):
            da_disp = np.float32(np.load(mono_disp_file))  # / 300.0
            uni_data = np.load(metric_depth_file)
            metric_depth = uni_data["depth"]

            fovs.append(uni_data["fov"])

            da_disp = cv2.resize(
                da_disp,
                (metric_depth.shape[1], metric_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            mono_disp_list.append(da_disp)
            gt_disp = 1.0 / (metric_depth + 1e-8)

            # avoid some bug from UniDepth
            valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
            gt_disp[valid_mask] = 1e-2

            # avoid cases sky dominate entire video
            sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
            if sky_ratio > 0.5:
                non_sky_mask = da_disp > 0.01
                gt_disp_ms = (
                        gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
                )
                da_disp_ms = (
                        da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
                )
                scale = np.median(gt_disp_ms / da_disp_ms)
                shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
            else:
                gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
                da_disp_ms = da_disp - np.median(da_disp) + 1e-8
                scale = np.median(gt_disp_ms / da_disp_ms)
                shift = np.median(gt_disp - scale * da_disp)

            gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
            da_disp_ms = da_disp - np.median(da_disp) + 1e-8

            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp - scale * da_disp)

            scales.append(scale)
            shifts.append(shift)

        print("************** UNIDEPTH FOV ", np.median(fovs))
        ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))
        K = np.eye(3)
        K[0, 0] = (
                ff * 1.0
        )  # pp_intrinsic[0]  * (img_0.shape[1] / (pp_intrinsic[1] * 2))
        K[1, 1] = (
                ff * 1.0
        )  # pp_intrinsic[0]  * (img_0.shape[0] / (pp_intrinsic[2] * 2))
        K[0, 2] = (
                img_0.shape[1] / 2.0
        )  # pp_intrinsic[1]) * (img_0.shape[1] / (pp_intrinsic[1] * 2))
        K[1, 2] = (
                img_0.shape[0] / 2.0
        )  # (pp_intrinsic[2]) * (img_0.shape[0] / (pp_intrinsic[2] * 2))

        ss_product = np.array(scales) * np.array(shifts)
        med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))

        align_scale = scales[med_idx]  # np.median(np.array(scales))
        align_shift = shifts[med_idx]  # np.median(np.array(shifts))
        normalize_scale = (np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98)/2.0)
        aligns = (align_scale, align_shift, normalize_scale)

        for t, image, depth, intrinsics, mask in tqdm(
                image_stream(
                    image_list,
                    mono_disp_list,
                    video_id,
                    use_depth=True,
                    aligns=aligns,
                    K=K,
                )
        ):
            rgb_list.append(image[0])
            senor_depth_list.append(depth)
            # breakpoint()
            if t == 0:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(args)

            droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

        # last frame
        droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

        traj_est, depth_est, motion_prob = droid.terminate(
            image_stream(
                image_list,
                mono_disp_list,
                video_id,
                use_depth=True,
                aligns=aligns,
                K=K,
            ),
            _opt_intr=True,
            full_ba=True,
            scene_name=video_id,
        )

        self.save_full_reconstruction(
            droid,
            traj_est,
            rgb_list,
            senor_depth_list,
            motion_prob,
            video_id,
        )

    def run_camera_tracking(self, args):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = self.video_id_frame_id_list[video_id]
            frame_id_list = sorted(np.unique(frame_id_list))
            for frame_id in frame_id_list:
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            self.video_camera_tracking_estimation(video_id, img_paths, args)

        print("Camera tracking completed for all videos.")
        print("Mono image mismatch count: ", self.mono_img_mismatch_counter)
        print("Metric image mismatch count: ", self.metric_img_mismatch_counter)
        print("Metric and Mono image mismatch count: ", self.metric_mono_mismatch_counter)


    # --------------------------- RUN OPTICAL FLOW ---------------------------

    def _load_flow_estimation_model(self, args):
        self._flow_estimation_model = torch.nn.DataParallel(RAFT(args))
        self._flow_estimation_model.load_state_dict(torch.load(args.model))
        print(f'Loaded checkpoint at {args.model}')
        self._flow_model = self._flow_estimation_model.module
        self._flow_model.cuda()
        self._flow_model.eval()

    def video_preprocess_flow(self, video_id, image_list, args):

        img_data = []
        for t, (image_file) in tqdm(enumerate(image_list)):
            image = cv2.imread(image_file)[..., ::-1]  # rgb
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            image = cv2.resize(image, (w1, h1))
            image = image[: h1 - h1 % 8, : w1 - w1 % 8].transpose(2, 0, 1)
            img_data.append(image)

        img_data = np.array(img_data)

        flows_low = []

        flows_high = []
        flow_masks_high = []

        flow_init = None
        flows_arr_low_bwd = {}
        flows_arr_low_fwd = {}

        ii = []
        jj = []
        flows_arr_up = []
        masks_arr_up = []

        for step in [1, 2, 4, 8, 15]:
            flows_arr_low = []
            for i in tqdm(range(max(0, -step), img_data.shape[0] - max(0, step))):
                image1 = (
                    torch.as_tensor(np.ascontiguousarray(img_data[i: i + 1]))
                    .float()
                    .cuda()
                )
                image2 = (
                    torch.as_tensor(
                        np.ascontiguousarray(img_data[i + step: i + step + 1])
                    )
                    .float()
                    .cuda()
                )

                ii.append(i)
                jj.append(i + step)

                with torch.no_grad():
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)
                    if np.abs(step) > 1:
                        flow_init = np.stack(
                            [flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]], axis=0
                        )
                        flow_init = (
                            torch.as_tensor(np.ascontiguousarray(flow_init))
                            .float()
                            .cuda()
                            .permute(0, 3, 1, 2)
                        )
                    else:
                        flow_init = None

                    flow_low, flow_up, _ = self._flow_model(
                        torch.cat([image1, image2], dim=0),
                        torch.cat([image2, image1], dim=0),
                        iters=22,
                        test_mode=True,
                        flow_init=flow_init,
                    )

                    flow_low_fwd = flow_low[0].cpu().numpy().transpose(1, 2, 0)
                    flow_low_bwd = flow_low[1].cpu().numpy().transpose(1, 2, 0)

                    flow_up_fwd = resize_flow(
                        flow_up[0].cpu().numpy().transpose(1, 2, 0),
                        flow_up.shape[-2] // 2,
                        flow_up.shape[-1] // 2,
                    )
                    flow_up_bwd = resize_flow(
                        flow_up[1].cpu().numpy().transpose(1, 2, 0),
                        flow_up.shape[-2] // 2,
                        flow_up.shape[-1] // 2,
                    )

                    bwd2fwd_flow = warp_flow(flow_up_bwd, flow_up_fwd)
                    fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
                    fwd_mask_up = fwd_lr_error < 1.0

                    # flows_arr_low.append(flow_low_fwd)
                    flows_arr_low_bwd[i + step] = flow_low_bwd
                    flows_arr_low_fwd[i] = flow_low_fwd

                    # masks_arr_low.append(fwd_mask_low)
                    flows_arr_up.append(flow_up_fwd)
                    masks_arr_up.append(fwd_mask_up)

        iijj = np.stack((ii, jj), axis=0)
        flows_high = np.array(flows_arr_up).transpose(0, 3, 1, 2)
        flow_masks_high = np.array(masks_arr_up)[:, None, ...]

        video_flow_dir = os.path.join(self._flow_root, video_id)
        os.makedirs(video_flow_dir, exist_ok=True)

        np.save(os.path.join(video_flow_dir, "flows.npy"), np.float16(flows_high))
        np.save(os.path.join(video_flow_dir, "flow_masks.npy"), flow_masks_high)
        np.save(os.path.join(video_flow_dir, "ii-jj.npy"), iijj)


    def run_flow_estimation(self, args):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = self.video_id_frame_id_list[video_id]
            frame_id_list = sorted(np.unique(frame_id_list))
            for frame_id in frame_id_list:
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            self.video_preprocess_flow(video_id, img_paths, args)

        print("Flow estimation completed for all videos.")


    # --------------------------- RUN CONSISTENT VIDEO DEPTH ESTIMATION ---------------------------
    @staticmethod
    def gradient_loss(gt, pred, u):
        """Gradient loss."""
        del u
        diff = pred - gt
        v_gradient = torch.abs(
            diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1]
        )  # * mask_v
        h_gradient = torch.abs(
            diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:]
        )  # * mask_h

        pred_grad = torch.abs(
            pred[..., 0:-2, 1:-1] - (pred[..., 2:, 1:-1])
        ) + torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
        gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - (gt[..., 2:, 1:-1])) + torch.abs(
            gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:]
        )

        grad_diff = torch.abs(pred_grad - gt_grad)
        nearby_mask = (torch.exp(gt[..., 1:-1, 1:-1]) > 1.0).float().detach()
        # weight = (1. - torch.exp(-(grad_diff * 5.)).detach())
        weight = 1.0 - torch.exp(-(grad_diff * 5.0)).detach()
        weight *= nearby_mask

        g_loss = torch.mean(h_gradient * weight) + torch.mean(v_gradient * weight)
        return g_loss

    @staticmethod
    def si_loss(gt, pred):
        log_gt = torch.log(torch.clamp(gt, 1e-3, 1e3)).view(gt.shape[0], -1)
        log_pred = torch.log(torch.clamp(pred, 1e-3, 1e3)).view(pred.shape[0], -1)
        log_diff = log_gt - log_pred
        num_pixels = gt.shape[-2] * gt.shape[-1]
        data_loss = torch.sum(log_diff ** 2, dim=-1) / num_pixels - torch.sum(
            log_diff, dim=-1
        ) ** 2 / (num_pixels ** 2)
        return torch.mean(data_loss)

    @staticmethod
    def sobel_fg_alpha(disp, mode="sobel", beta=10.0):
        sobel_grad = kornia.filters.spatial_gradient(
            disp, mode=mode, normalized=False
        )
        sobel_mag = torch.sqrt(
            sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2
        )
        alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

        return alpha

    def consistency_loss(
            self,
            cam_c2w,
            K,
            K_inv,
            disp_data,
            init_disp,
            uncertainty,
            flows,
            flow_masks,
            ii,
            jj,
            compute_normals,
            fg_alpha,
            w_ratio=1.0,
            w_flow=0.2,
            w_si=1.0,
            w_grad=2.0,
            w_normal=4.0,
    ):
        """Consistency loss."""
        _, H, W = disp_data.shape
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W)  # .repeat(B ,1 ,1 ,1)
        yy = yy.view(1, 1, H, W)  # .repeat(B ,1 ,1 ,1)
        grid = (
            torch.cat((xx, yy), 1).float().cuda().permute(0, 2, 3, 1)
        )  # [None, ...]

        loss_flow = 0.0  # flow reprojection loss
        loss_d_ratio = 0.0  # depth consistency loss

        flows_step = flows.permute(0, 2, 3, 1)
        flow_masks_step = flow_masks.permute(0, 2, 3, 1).squeeze(-1)

        cam_1to2 = torch.bmm(
            torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
            torch.index_select(cam_c2w, dim=0, index=ii),
        )

        # warp disp from target time
        pixel_locations = grid + flows_step
        resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
        normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

        disp_sampled = torch.nn.functional.grid_sample(
            torch.index_select(disp_data, dim=0, index=jj)[:, None, ...],
            normalized_pixel_locations,
            align_corners=True,
        )

        uu = torch.index_select(uncertainty, dim=0, index=ii).squeeze(1)

        grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(
            -1
        )
        # depth of reference view
        ref_depth = 1.0 / torch.clamp(
            torch.index_select(disp_data, dim=0, index=ii), 1e-3, 1e3
        )

        pts_3d_ref = ref_depth[..., None, None] * (K_inv[None, None, None] @ grid_h)
        rot = cam_1to2[:, None, None, :3, :3]
        trans = cam_1to2[:, None, None, :3, 3:4]

        pts_3d_tgt = (rot @ pts_3d_ref) + trans  # [:, None, None, :, None]
        depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
        disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

        # flow consistency loss
        pts_2D_tgt = K[None, None, None] @ pts_3d_tgt

        flow_masks_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1)
        pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / torch.clamp(
            pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3
        )

        disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
        disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

        ratio = torch.maximum(
            disp_sampled.squeeze() / disp_tgt.squeeze(),
            disp_tgt.squeeze() / disp_sampled.squeeze(),
        )
        ratio_error = torch.abs(ratio - 1.0)  #

        loss_d_ratio += torch.sum(
            (ratio_error * uu + self.ALPHA_MOTION * torch.log(1.0 / uu)) * flow_masks_step_
        ) / (torch.sum(flow_masks_step_) + 1e-8)

        flow_error = torch.abs(pts_2D_tgt - pixel_locations)
        loss_flow += torch.sum(
            (
                    flow_error * uu[..., None]
                    + self.ALPHA_MOTION * torch.log(1.0 / uu[..., None])
            )
            * flow_masks_step_[..., None]
        ) / (torch.sum(flow_masks_step_) * 2.0 + 1e-8)

        # prior mono-depth reg loss
        loss_prior = self.si_loss(init_disp, disp_data)
        KK = torch.inverse(K_inv)

        # multi gradient consistency
        disp_data_ds = disp_data[:, None, ...]
        init_disp_ds = init_disp[:, None, ...]
        K_rescale = KK.clone()
        K_inv_rescale = torch.inverse(K_rescale)
        pred_normal = compute_normals[0](
            1.0 / torch.clamp(disp_data_ds, 1e-3, 1e3), K_inv_rescale[None]
        )
        init_normal = compute_normals[0](
            1.0 / torch.clamp(init_disp_ds, 1e-3, 1e3), K_inv_rescale[None]
        )

        loss_normal = torch.mean(
            fg_alpha * (1.0 - torch.sum(pred_normal * init_normal, dim=1))
        )  # / (1e-8 + torch.sum(fg_alpha))

        loss_grad = 0.0
        for scale in range(4):
            interval = 2 ** scale
            disp_data_ds = torch.nn.functional.interpolate(
                disp_data[:, None, ...],
                scale_factor=(1.0 / interval, 1.0 / interval),
                mode="nearest-exact",
            )
            init_disp_ds = torch.nn.functional.interpolate(
                init_disp[:, None, ...],
                scale_factor=(1.0 / interval, 1.0 / interval),
                mode="nearest-exact",
            )
            uncertainty_rs = torch.nn.functional.interpolate(
                uncertainty,
                scale_factor=(1.0 / interval, 1.0 / interval),
                mode="nearest-exact",
            )
            loss_grad += self.gradient_loss(
                torch.log(disp_data_ds), torch.log(init_disp_ds), uncertainty_rs
            )

        return (
                w_ratio * loss_d_ratio
                + w_si * loss_prior
                + w_flow * loss_flow
                + w_normal * loss_normal
                + loss_grad * w_grad
        )

    def video_cvd_estimation(self, video_id, image_list, args):

        video_reconstructions_dir = os.path.join(self._camera_tracking_root, "reconstructions", video_id)
        video_flows_dir = os.path.join(self._flow_root, video_id)

        img_data = np.load(os.path.join(video_reconstructions_dir, "images.npy"))[:, ::-1, ...]
        disp_data = (np.load(os.path.join(video_reconstructions_dir, "disps.npy")) + 1e-6)
        intrinsics = np.load(os.path.join(video_reconstructions_dir, "intrinsics.npy"))
        poses = np.load(os.path.join(video_reconstructions_dir, "poses.npy"))
        mot_prob = np.load(os.path.join(video_reconstructions_dir, "motion_prob.npy"))

        flows = np.load(os.path.join(video_flows_dir, "flows.npy"), allow_pickle=True)
        flow_masks = np.load(os.path.join(video_flows_dir, "flow_masks.npy"), allow_pickle=True)
        iijj = np.load(os.path.join(video_flows_dir, "ii-jj.npy"), allow_pickle=True)

        intrinsics = intrinsics[0]
        poses_th = torch.as_tensor(poses, device="cpu").float().cuda()

        K = np.eye(3)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]

        img_data_pt = (
                torch.from_numpy(np.ascontiguousarray(img_data)).float().cuda() / 255.0
        )
        flows = torch.from_numpy(np.ascontiguousarray(flows)).float().cuda()
        flow_masks = (
            torch.from_numpy(np.ascontiguousarray(flow_masks)).float().cuda()
        )  # .unsqueeze(1)
        iijj = torch.from_numpy(np.ascontiguousarray(iijj)).float().cuda()
        ii = iijj[0, ...].long()
        jj = iijj[1, ...].long()
        K = torch.from_numpy(K).float().cuda()

        init_disp = torch.from_numpy(disp_data).float().cuda()
        disp_data = torch.from_numpy(disp_data).float().cuda()

        assert init_disp.shape == disp_data.shape

        init_disp = torch.nn.functional.interpolate(
            init_disp.unsqueeze(1),
            scale_factor=(self.RESIZE_FACTOR, self.RESIZE_FACTOR),
            mode="bilinear",
        ).squeeze(1)
        disp_data = torch.nn.functional.interpolate(
            disp_data.unsqueeze(1),
            scale_factor=(self.RESIZE_FACTOR, self.RESIZE_FACTOR),
            mode="bilinear",
        ).squeeze(1)

        fg_alpha = self.sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
        fg_alpha = fg_alpha.squeeze(1).float() + 0.2

        cvd_prob = torch.nn.functional.interpolate(
            torch.from_numpy(mot_prob).unsqueeze(1).cuda(),
            scale_factor=(4, 4),
            mode="bilinear",
        )
        cvd_prob[cvd_prob > 0.5] = 0.5
        cvd_prob = torch.clamp(cvd_prob, 1e-3, 1.0)

        # rescale intrinsic matrix to small resolution
        K_o = K.clone()
        K[0:2, ...] *= self.RESIZE_FACTOR
        K_inv = torch.linalg.inv(K)

        disp_data.requires_grad = False
        poses_th.requires_grad = False

        uncertainty = cvd_prob

        # First optimize scale and shift to align them
        log_scale_ = torch.log(torch.ones(init_disp.shape[0]).to(disp_data.device))
        shift_ = torch.zeros(init_disp.shape[0]).to(disp_data.device)
        log_scale_.requires_grad = True
        shift_.requires_grad = True
        uncertainty.requires_grad = True

        optim = torch.optim.Adam([
            {"params": log_scale_, "lr": 1e-2},
            {"params": shift_, "lr": 1e-2},
            {"params": uncertainty, "lr": 1e-2},
        ])

        compute_normals = []
        compute_normals.append(
            NormalGenerator(disp_data.shape[-2], disp_data.shape[-1])
        )
        init_disp = torch.clamp(init_disp, 1e-3, 1e3)

        for i in range(100):
            optim.zero_grad()
            cam_c2w = SE3(poses_th).inv().matrix()
            scale_ = torch.exp(log_scale_)

            loss = self.consistency_loss(
                cam_c2w,
                K,
                K_inv,
                torch.clamp(
                    disp_data * scale_[..., None, None] + shift_[..., None, None],
                    1e-3,
                    1e3,
                ),
                init_disp,
                torch.clamp(uncertainty, 1e-4, 1e3),
                flows,
                flow_masks,
                ii,
                jj,
                compute_normals,
                fg_alpha,
            )

            loss.backward()
            uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
            log_scale_.grad = torch.nan_to_num(log_scale_.grad, nan=0.0)
            shift_.grad = torch.nan_to_num(shift_.grad, nan=0.0)

            optim.step()
            print("step ", i, loss.item())

        # Then optimize depth and uncertainty
        disp_data = (
                disp_data * torch.exp(log_scale_)[..., None, None].detach()
                + shift_[..., None, None].detach()
        )
        init_disp = (
                init_disp * torch.exp(log_scale_)[..., None, None].detach()
                + shift_[..., None, None].detach()
        )
        init_disp = torch.clamp(init_disp, 1e-3, 1e3)

        disp_data.requires_grad = True
        uncertainty.requires_grad = True
        poses_th.requires_grad = False  # True

        optim = torch.optim.Adam([
            {"params": disp_data, "lr": 5e-3},
            {"params": uncertainty, "lr": 5e-3},
        ])

        losses = []
        for i in range(400):
            optim.zero_grad()
            cam_c2w = SE3(poses_th).inv().matrix()
            loss = self.consistency_loss(
                cam_c2w,
                K,
                K_inv,
                torch.clamp(disp_data, 1e-3, 1e3),
                init_disp,
                torch.clamp(uncertainty, 1e-4, 1e3),
                flows,
                flow_masks,
                ii,
                jj,
                compute_normals,
                fg_alpha,
                w_ratio=1.0,
                w_flow=0.2,
                w_si=1,
                w_grad=args.w_grad,
                w_normal=args.w_normal,
            )

            loss.backward()
            disp_data.grad = torch.nan_to_num(disp_data.grad, nan=0.0)
            uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)

            optim.step()
            print("step ", i, loss.item())
            losses.append(loss)

        disp_data_opt = (
            torch.nn.functional.interpolate(
                disp_data.unsqueeze(1), scale_factor=(2, 2), mode="bilinear"
            )
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
        )

        # poses_ = poses_th.detach().cpu().numpy()
        cvd_video_dir = os.path.join(self._cvd_opt_root, video_id)
        os.makedirs(cvd_video_dir, exist_ok=True)

        np.savez(
            os.path.join(cvd_video_dir, f"{video_id}_sgd_cvd_hr.npz"),
            images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0),
            depths=np.clip(np.float16(1.0 / disp_data_opt), 1e-3, 1e2),
            intrinsic=K_o.detach().cpu().numpy(),
            cam_c2w=cam_c2w.detach().cpu().numpy(),
        )

    def run_cvd_opt(self, args):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = self.video_id_frame_id_list[video_id]
            frame_id_list = sorted(np.unique(frame_id_list))
            for frame_id in frame_id_list:
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            self.video_cvd_estimation(video_id, img_paths, args)

        print("Flow estimation completed for all videos.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/data/rohith/ag/")

    parser.add_argument("--mode", type=str,
                        choices=["depth_anything", "uni_depth", "camera_tracking", "preprocess_flow", "cvd_opt"],
                        required=True)

    # # ------------------------------ DEPTH ANYTHING ----------------------------
    # parser.add_argument('--encoder', type=str, default='vitl')
    # parser.add_argument('--load_from', type=str,
    #                     default='/home/rxp190007/CODE/mega-sam/Depth_Anything/checkpoints/depth_anything_vitl14.pth')
    # parser.add_argument('--localhub', dest='localhub', action='store_true', default=False)
    #
    # # ------------------------------ CAMERA TRACKING ----------------------------
    # parser.add_argument("--weights", default="/home/rxp190007/CODE/mega-sam/checkpoints/megasam_final.pth")
    # parser.add_argument("--buffer", type=int, default=1024)
    # parser.add_argument("--image_size", default=[240, 320])
    # parser.add_argument("--disable_vis", action="store_true")
    # parser.add_argument("--beta", type=float, default=0.3)
    #
    # # motion threshold for keyframe
    # parser.add_argument("--filter_thresh", type=float, default=2.0)
    # parser.add_argument("--warmup", type=int, default=8)
    # parser.add_argument("--keyframe_thresh", type=float, default=2.0)
    # parser.add_argument("--frontend_thresh", type=float, default=12.0)
    # parser.add_argument("--frontend_window", type=int, default=25)
    # parser.add_argument("--frontend_radius", type=int, default=2)
    # parser.add_argument("--frontend_nms", type=int, default=1)
    #
    # parser.add_argument("--stereo", action="store_true")
    # parser.add_argument("--depth", action="store_true")
    # parser.add_argument("--upsample", action="store_true")
    # parser.add_argument("--scene_name", help="scene_name")
    #
    # parser.add_argument("--backend_thresh", type=float, default=16.0)
    # parser.add_argument("--backend_radius", type=int, default=2)
    # parser.add_argument("--backend_nms", type=int, default=3)
    #
    # parser.add_argument("--mono_depth_path", default="/data/rohith/ag/ag4D/mega_sam/depth_anything")
    # parser.add_argument("--metric_depth_path", default="/data/rohith/ag/ag4D/mega_sam/unidepth")

    # ------------------------------ FLOW ESTIMATION ----------------------------
    # parser.add_argument(
    #     '--model', default='cvd_opt/raft-things.pth', help='restore checkpoint'
    # )
    # parser.add_argument('--small', action='store_true', help='use small model')
    #
    # parser.add_argument('--path', help='dataset for evaluation')
    # parser.add_argument(
    #     '--num_heads',
    #     default=1,
    #     type=int,
    #     help='number of heads in attention and aggregation',
    # )
    # parser.add_argument(
    #     '--position_only',
    #     default=False,
    #     action='store_true',
    #     help='only use position-wise attention',
    # )
    # parser.add_argument(
    #     '--position_and_content',
    #     default=False,
    #     action='store_true',
    #     help='use position and content-wise attention',
    # )
    # parser.add_argument(
    #     '--mixed_precision', action='store_true', help='use mixed precision'
    # )

    # --------------------------------- CVD OPT ---------------------------------

    parser.add_argument("--w_grad", type=float, default=2.0, help="w_grad")
    parser.add_argument("--w_normal", type=float, default=5.0, help="w_normal")

    args = parser.parse_args()
    ag_mega_sam = AgMegaSam(datapath=args.datapath)

    # ----- Run depth anything depth estimation -----
    if args.mode == "depth_anything":
        print("Running depth estimation using Depth Anything...")
        ag_mega_sam._load_depth_anything_model(args)
        ag_mega_sam.run_ag_depth_anything_estimation()
    # ----- Run unidepth depth estimation -----
    elif args.mode == "uni_depth":
        print("Running depth estimation using UniDepth...")
        ag_mega_sam._load_unidepth_model(args)
        ag_mega_sam.run_ag_unidepth_estimation()
    # ----- Run camera tracking -----
    elif args.mode == "camera_tracking":
        print("Running camera tracking...")
        ag_mega_sam.run_camera_tracking(args)
    elif args.mode == "preprocess_flow":
        print("Running flow estimation...")
        ag_mega_sam._load_flow_estimation_model(args)
        ag_mega_sam.run_flow_estimation(args)
    # ----- Run consistent video depth estimation -----
    elif args.mode == "cvd_opt":
        print("Running consistent video depth estimation...")
        ag_mega_sam.run_cvd_opt(args)
    else:
        raise ValueError("Invalid mode selected. Choose from ['depth_anything', 'uni_depth']")


if __name__ == '__main__':
    main()
