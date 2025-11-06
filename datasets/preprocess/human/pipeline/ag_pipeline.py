import glob
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from smplcodec import SMPLCodec
from smplx import SMPLX

from .camera import run_metric_slam, calibrate_intrinsics, run_slam
from .detector import segment
from .detector.vitpose_estimator import load_vit_model, estimate_kp2ds_from_bbox_vitpose
from .kp_utils import convert_kps
from .mcs_export_cam import export_scene_with_camera
from .phmr_vid import PromptHMR_Video
from .postprocessing import post_optimization
from .spec.cam_calib import run_pi3_spec_calib
from .tools import detect_track, detect_segment_track_sam, est_camera, est_calib
from .world import world_hps_estimation
# import sys
# sys.path.append('../')
from ..data_config import CONFIG_PATH, SMPLX_NEUTRAL_MODEL_PATH, SMPLX_NEUTRAL_F32_PATH


class AgPipeline:

    def __init__(
            self,
            static_cam=False,
            ag_root_directory="/data/rohith/ag",
            sampled_frames_path="/data/rohith/ag/sampled_frames_jpg",
            dynamic_scene_dir_path="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
            results_output_dir_path="/data2/rohith/ag/ag4D/human",
    ):
        self.images = None
        self.cfg = OmegaConf.load(CONFIG_PATH)
        self.cfg.static_cam = static_cam

        self.ag_root_directory = Path(ag_root_directory)

        self.sampled_frames_path = Path(sampled_frames_path)
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)
        self.results_output_dir_path = Path(results_output_dir_path)
        os.makedirs(self.results_output_dir_path, exist_ok=True)

        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir_path = self.ag_root_directory / "sampled_frames_idx"

        checkpoint_dir = os.path.join(os.path.dirname(__file__), '../data/pretrain')
        self.data_dict = {
            'droid': os.path.join(checkpoint_dir, 'droid.pth'),
            'sam': os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth"),
            'sam2': os.path.join(checkpoint_dir, "sam2_ckpts"),
            'yolo': os.path.join(checkpoint_dir, 'yolo11x.pt'),
            'vitpose': os.path.join(checkpoint_dir, 'vitpose-h-coco_25.pth'),
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.smplx = SMPLX(
            SMPLX_NEUTRAL_MODEL_PATH,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10
        )
        self.smplx.to(self.device)

        self.points = None
        self.camera_poses = None

    def run_detect_track(self, ):
        if self.cfg.tracker == 'bytetrack':
            tracks = detect_track(self.images, bbox_interp=self.cfg.bbox_interp)
            masks = segment.segment_subjects(self.images)
        elif self.cfg.tracker == 'sam2':
            tracks, masks = detect_segment_track_sam(
                self.images,
                self.seq_folder,
                self.data_dict,
                debug_masks=False,
                sam2_type='tiny',
                detector_type='detectron2',
                num_max_people=self.cfg.num_max_people,
                det_thresh=self.cfg.det_thresh,
                score_thresh=self.cfg.det_score_thresh,
                height_thresh=self.cfg.det_height_thresh,
                bbox_interp=self.cfg.bbox_interp
            )

        self.results['masks'] = masks
        self.results['people'] = tracks
        self.results['has_tracks'] = True

    def estimate_2d_keypoints(self, ):
        model_path = os.path.join(os.path.dirname(__file__), '../data/pretrain/vitpose-h-coco_25.pth')
        model = load_vit_model(model_path=model_path)
        for k, v in self.results['people'].items():
            kpts_2d = estimate_kp2ds_from_bbox_vitpose(model, self.images, v['bboxes'], k, v['frames'])
            kpts_2d = convert_kps(kpts_2d, 'vitpose25', 'openpose')
            self.results['people'][k]['keypoints_2d'] = kpts_2d
            coco_kp2d = convert_kps(kpts_2d, 'ophandface', 'cocoophf')
            self.results['people'][k]['vitpose'] = coco_kp2d
        self.results['has_2d_kpts'] = True
        del model
        return self.results

    def hps_estimation(self, ):
        if self.cfg.tracker == 'sam2':
            mask_prompt = True
        else:
            mask_prompt = False

        phmr = PromptHMR_Video()
        self.results = phmr.run(self.images, self.results, mask_prompt)
        self.results['contact_joint_ids'] = [7, 10, 8, 11, 20, 21]
        self.results['has_hps_cam'] = True
        return

    def camera_motion_estimation(self, static_cam=False, camera_poses=None):
        masks = self.results['masks']
        masks = torch.from_numpy(masks)

        assert masks.shape[0] == len(self.images), (
            f"Masks and images should be same length {masks.shape[0]} != {len(self.images)}"
        )

        # ----------------------------------------------------
        # 1) Intrinsics logic (Pi3 doesn't give intrinsics)
        # ----------------------------------------------------
        opt_intr = False if self.cfg.use_depth else True
        keyframes = None

        if self.cfg.static_cam or static_cam:
            print("Using static camera assumption")
            static_cam = True
            if self.cfg.calib is None:
                cam_int = est_calib(self.images)
            else:
                cam_int = np.loadtxt(self.cfg.calib)
                opt_intr = False
        else:
            if self.cfg.calib is None:
                if self.cfg.focal is None and opt_intr is False:
                    try:
                        if self.cfg.calib_method == 'ba':
                            _, _, cam_int, keyframes = run_slam(
                                self.images,
                                masks=masks,
                                opt_intr=True,
                                stride=self.cfg.calib_stride,
                            )
                        elif self.cfg.calib_method == 'iterative':
                            cam_int = calibrate_intrinsics(self.cfg.img_folder, masks)
                    except ValueError as e:
                        static_cam = True
                        print(e)
                        print("Warning: probably there is not much camera motion in the video!!")
                        cam_int = est_calib(self.images)

                elif self.cfg.focal is not None:
                    cam_int = est_calib(self.images)
                    cam_int[0] = self.cfg.focal
                    cam_int[1] = self.cfg.focal
                    opt_intr = False
                else:
                    cam_int = est_calib(self.images)
            else:
                cam_int = np.loadtxt(self.cfg.calib)
                opt_intr = False

        # ----------------------------------------------------
        # 2) Use Pi3 camera_poses if provided (Pi3 is c2w)
        # ----------------------------------------------------
        if camera_poses is not None:
            print("Using Pi3 provided camera poses for camera motion estimation...")
            # Pi3: camera_poses.shape = (N, 4, 4), each is camera -> world
            camera_poses = np.asarray(camera_poses)
            assert camera_poses.shape[0] == len(self.images), (
                f"Pi3 camera_poses ({camera_poses.shape[0]}) must match num images ({len(self.images)})"
            )

            # make first frame the world: T_rel_i = T0^{-1} @ Ti
            T0 = camera_poses[0]  # c0 -> W
            T0_inv = np.linalg.inv(T0)  # W -> c0

            R_list = []
            t_list = []
            for Ti in camera_poses:  # Ti: ci -> W
                T_rel = T0_inv @ Ti  # ci -> c0  (now world = c0)
                R_rel = T_rel[:3, :3]
                t_rel = T_rel[:3, 3]
                R_list.append(R_rel)
                t_list.append(t_rel)

            cam_R = torch.from_numpy(np.stack(R_list, axis=0)).float()
            cam_T = torch.from_numpy(np.stack(t_list, axis=0)).float()

            static_cam = False  # we do have motion now

        else:
            # ------------------------------------------------
            # 3) Fallback: original SLAM / static path
            # ------------------------------------------------
            print("Estimating camera motion via Droid SLAM...bypassed Pi3 camera poses")
            if static_cam:
                cam_R = torch.eye(3)[None].repeat_interleave(len(masks), 0)
                cam_T = torch.zeros((len(masks), 3))
                print("Warning: probably there is not much camera motion in the video!!")
                print("Setting camera motion to zero")
            else:
                try:
                    cam_R, cam_T, cam_int = run_metric_slam(
                        self.images,
                        masks=masks,
                        calib=cam_int,
                        monodepth_method=self.cfg.depth_method,
                        use_depth_inp=self.cfg.use_depth,
                        stride=self.cfg.stride,
                        opt_intr=opt_intr,
                        save_depth=self.cfg.save_depth,
                        keyframes=keyframes,
                    )
                except ValueError as e:
                    if str(e).startswith("not enough values to unpack"):
                        cam_R = torch.eye(3)[None].repeat_interleave(len(masks), 0)
                        cam_T = torch.zeros((len(masks), 3))
                        print("Warning: probably there is not much camera motion in the video!!")
                        print("Setting camera motion to zero")
                    else:
                        raise e

        # ----------------------------------------------------
        # 4) Pack results
        # ----------------------------------------------------
        print("Camera intrinsics:", cam_int)
        camera = {
            'pred_cam_R': cam_R.numpy(),
            'pred_cam_T': cam_T.numpy(),
            'img_focal': cam_int[0],
            'img_center': cam_int[2:],
        }
        print("cam focal length: ", cam_int[0])
        self.results['camera'] = camera
        self.results['has_slam'] = True
        return

    def world_hps_estimation(self, ):
        self.results = world_hps_estimation(self.cfg, self.results, self.smplx, self.device)
        self.results['has_hps_world'] = True
        return

    def post_optimization(self):
        self.results = post_optimization(
            self.cfg, self.results, self.images,
            self.smplx, opt_contact=True,
        )
        self.results['has_post_opt'] = True

    def get_K(self, ):
        camera = self.results['camera']
        K = np.eye(3)
        K[0, 0] = camera['img_focal']
        K[1, 1] = camera['img_focal']
        K[:2, -1] = camera['img_center']
        K = torch.tensor(K, dtype=torch.float)
        return K

    def create_world4d(self, results=None, total=None, step=1):
        if results is None:
            results = self.results
        if total is None:
            total = len(results['camera']['pred_cam_R'])
        else:
            total = min(total, len(results['camera']['pred_cam_R']))

        world4d = {}
        for i in range(0, total, step):
            pose = []
            shape = []
            transl = []
            track_id = []

            # People
            for pid in results['people']:
                people = results['people'][pid]
                frames = people['frames']
                in_frame = np.where(frames == i)[0]

                if len(in_frame) == 1:
                    smplx_w = people['smplx_world']
                    pose.append(smplx_w['pose'][in_frame])
                    shape.append(smplx_w['shape'][in_frame])
                    transl.append(smplx_w['trans'][in_frame])
                    track_id.append(people['track_id'])

            # Camera
            camera_w = results['camera_world']
            Rwc = camera_w['Rwc'][i]
            Twc = camera_w['Twc'][i]
            camera = np.eye(4)
            camera[:3, :3] = Rwc
            camera[:3, 3] = Twc

            if len(track_id) > 0:
                world4d[i] = {
                    'pose': torch.tensor(np.concatenate(pose)).float().reshape(len(track_id), -1, 3),
                    'shape': torch.tensor(np.concatenate(shape)).float(),
                    'trans': torch.tensor(np.concatenate(transl)).float(),
                    'track_id': torch.tensor(np.array(track_id)) - 1,
                    'camera': camera
                }
            else:
                world4d[i] = {'track_id': np.array([]), 'camera': camera}

        return world4d

    def load_frames(self, video_id, H, W):
        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)

        annotated_frame_id_list = os.listdir(video_frames_annotated_dir_path)
        annotated_frame_id_list = [f for f in annotated_frame_id_list if f.endswith('.png')]
        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        # Get the mapping for sampled_frame_id and the actual frame id
        # Now start from the sampled frame which corresponds to the first annotated frame and keep the rest of the sampled frames
        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir_path, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()  # Numbers only

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_first_frame_id)
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_last_frame_id)
        sample_idx = list(range(an_first_id_in_vid_sam_frame_id_list, an_last_id_in_vid_sam_frame_id_list + 1))

        final_sampled_images = [video_sampled_frame_id_list[i] for i in sample_idx]


        video_frames_dir_path = self.sampled_frames_path / video_id
        for sampled_frame_id in final_sampled_images:
            assert os.path.exists(
                os.path.join(self.sampled_frames_path, video_id,
                             f"{video_sampled_frame_id_list[sampled_frame_id]:06d}.jpg")), \
                f"Frame {video_sampled_frame_id_list[sampled_frame_id]:06d}.jpg does not exist in {os.path.join(self.sampled_frames_path, video_id)}"

            # Load each image from the sampled frames path - Convert to 000006 format
            frame_path = video_frames_dir_path / f"{sampled_frame_id:06d}.png"
            img = cv2.imread(str(frame_path))[:, :, ::-1]
            img = cv2.resize(img, (W, H))
            self.images.append(img)

    def load_dynamic_predictions(self, video_id):
        video_dynamic_3d_scene_path = self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        video_dynamic_predictions = np.load(video_dynamic_3d_scene_path, allow_pickle=True)
        return video_dynamic_predictions

    # ------------------------------------------------------------
    # NEW: expose per-frame dynamic point cloud
    # ------------------------------------------------------------
    def get_frame_pointcloud(self, frame_idx: int) -> np.ndarray:
        """
        Returns (H, W, 3) point cloud for frame_idx from the dynamic scene.
        """
        return self.points[frame_idx]

    # ------------------------------------------------------------
    # NEW: expose per-frame camera pose (Pi3 gives c2w)
    # ------------------------------------------------------------
    def get_frame_camera_pose(self, frame_idx: int) -> np.ndarray:
        """
        Returns (4,4) camera->world for this frame.
        """
        return self.camera_poses[frame_idx]

    def __call__(
            self,
            video_id,
            static_cam=False,
            save_only_essential=False,
            max_frame=None
    ):
        def cvt_to_numpy(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    cvt_to_numpy(v)
                elif isinstance(v, torch.Tensor):
                    d[k] = v.detach().cpu().numpy()

        self.fps = 10
        video_results_output_path = self.results_output_dir_path / video_id
        video_results_output_path.mkdir(parents=True, exist_ok=True)

        self.cfg.output_folder = str(video_results_output_path)
        self.cfg.sequence_folder = video_results_output_path
        self.seq_folder = video_results_output_path

        video_dynamic_predictions = self.load_dynamic_predictions(video_id)

        imgs_f32 = video_dynamic_predictions['images']
        self.images = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)  # (S, H, W, 3)
        self.camera_poses = video_dynamic_predictions['camera_poses']
        self.points = video_dynamic_predictions['points']

        if os.path.isfile(f'{video_results_output_path}/results.pkl'):
            print('Loading available results...')
            self.results = joblib.load(f'{video_results_output_path}/results.pkl')
            return self.results

        self.results = {'camera': est_camera(self.images[0]), 'people': {}, 'timings': {}, 'masks': None,
                        'has_tracks': False, 'has_hps_cam': False, 'has_hps_world': False, 'has_slam': False,
                        'has_hands': False, 'has_2d_kpts': False, 'has_post_opt': False}

        ### Spec Camera
        if not self.results['has_slam']:
            print(f"[{video_id}] Running spectral camera calibration...")
            spec_calib_output_dir = video_results_output_path / "spec_calib"
            os.makedirs(spec_calib_output_dir, exist_ok=True)

            spec_calib = run_pi3_spec_calib(
                images=self.images,
                out_folder=spec_calib_output_dir,
                loss_type='softargmax_l2',
                stride=1,
                first_frame_idx=0,
                camera_poses=self.camera_poses)

            self.results['spec_calib'] = spec_calib
            print("---------------------------------------------------------------------------")

        ### detect_segment_track
        if not self.results['has_tracks']:
            print(f"[{video_id}] Running detect, segment, and track pipeline...")
            self.run_detect_track()
            print("---------------------------------------------------------------------------")

        ### slam
        if not self.results['has_slam']:
            print("Running camera motion estimation...")
            self.camera_motion_estimation(static_cam, camera_poses=self.camera_poses)
            print("---------------------------------------------------------------------------")

        ### keypoints detection
        if not self.results['has_2d_kpts']:
            print("Estimating 2D keypoints...")
            self.estimate_2d_keypoints()
            print("---------------------------------------------------------------------------")

        ### hps
        if not self.results['has_hps_cam']:
            print("Running human mesh estimation...")
            self.hps_estimation()
            print("---------------------------------------------------------------------------")

        ### convert hps to world coordinate
        if not self.results['has_hps_world']:
            print("Running world coordinates estimation...")
            self.world_hps_estimation()
            print("---------------------------------------------------------------------------")

        cvt_to_numpy(self.results)

        # ### post optimization
        if self.cfg.run_post_opt and not self.results['has_post_opt']:
            print("Running post optimization...")
            self.post_optimization()
            print("---------------------------------------------------------------------------")

        if save_only_essential:
            _ = self.results.pop('masks', None)
            for tid, track in self.results['people'].items():
                _ = track.pop('masks', None)
                _ = track.pop('keypoints_2d', None)
                _ = track.pop('vitpose', None)
                _ = track.pop('prhmr_img_feats', None)

        joblib.dump(self.results, f'{video_results_output_path}/results.pkl')

        NUM_FRAMES = len(self.images)
        MCS_OUTPUT_PATH = f'{video_results_output_path}/world4d.mcs'
        smpl_paths = []
        per_body_frame_presence = []
        for k, v in self.results['people'].items():
            out_smpl_f = f'{os.path.abspath(self.seq_folder)}/subject-{k}.smpl'
            SMPLCodec(
                shape_parameters=v['smplx_world']['shape'].mean(0),
                body_pose=v['smplx_world']['pose'][:, :22 * 3].reshape(-1, 22, 3),
                body_translation=v['smplx_world']['trans'],
                frame_count=v['frames'].shape[0], frame_rate=float(self.fps)
            ).write(out_smpl_f)
            smpl_paths.append(out_smpl_f)
            per_body_frame_presence.append([int(v['frames'][0]), int(v['frames'][-1]) + 1])

        export_scene_with_camera(
            smpl_buffers=[open(path, 'rb').read() for path in smpl_paths],
            frame_presences=per_body_frame_presence,
            num_frames=NUM_FRAMES,
            output_path=MCS_OUTPUT_PATH,
            rotation_matrices=self.results['camera_world']['Rcw'],
            translations=self.results['camera_world']['Tcw'],
            focal_length=self.results['camera_world']['img_focal'],
            principal_point=self.results['camera_world']['img_center'],
            frame_rate=float(self.cfg.fps),
            smplx_path=SMPLX_NEUTRAL_F32_PATH,
        )

        return self.results