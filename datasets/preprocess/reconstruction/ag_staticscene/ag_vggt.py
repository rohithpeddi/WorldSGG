import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
import cv2

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)


class AgStaticVideoDataset:

    def __init__(
            self,
            static_frames_dir,
    ):
        self.static_frames_dir = Path(static_frames_dir)
        self.static_frames = list(self.static_frames_dir.glob("*"))

    def __len__(self):
        return len(self.static_frames)

    def __getitem__(self, idx):
        # (1) Loads a videos from the static_videos list
        video_name = self.static_frames[idx]
        video_path = self.static_frames_dir / video_name

        # (2) Load the video using cv2 and extract frames as images and return the images
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = np.stack(frames, axis=0)  # (N,H,W,3)
        return frames, video_path.name


class AgVGGT:

    def __init__(
            self,
            args
    ):
        self.video_scene_dir = None
        self.args = args
        if args.static_frames_dir is not None:
            self.static_videos_dir = Path(args.static_frames_dir)

        self.static_video_dataset = AgStaticVideoDataset(static_frames_dir=self.static_videos_dir)

        self.max_reproj_error = args.max_reproj_error
        self.shared_camera = args.shared_camera
        self.camera_type = args.camera_type
        self.vis_thresh = args.vis_thresh
        self.query_frame_num = args.query_frame_num
        self.max_query_pts = args.max_query_pts
        self.fine_tracking = args.fine_tracking
        self.conf_threshold_value = args.conf_thres_value

        self.output_dir = Path(args.video_scene_dir)

        self.static_scenes_dir = args.static_scenes_dir
        os.makedirs(self.static_scenes_dir, exist_ok=True)

        # Seeds
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Setting seed as: {seed}")

        # Device & dtype
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
            0] >= 8 else torch.float16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")

        # Load VGGT weights
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval().to(self.device)
        print("Model loaded")

        self.vggt_fixed_resolution = 518
        self.img_load_resolution = 1024

    def run_video_vggt(self, images, resolution=518):
        # images: [B, 3, H, W]
        assert len(images.shape) == 4 and images.shape[1] == 3
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images_in = images[None]  # add batch dimension -> (1,B,3,H,W)
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_in)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_in.shape[-2:])
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images_in, ps_idx)

        extrinsic = extrinsic.squeeze(0).cpu().numpy()  # (B,4,4)
        intrinsic = intrinsic.squeeze(0).cpu().numpy()  # (B,3,3)
        depth_map = depth_map.squeeze(0).cpu().numpy()  # (B,H,W)
        depth_conf = depth_conf.squeeze(0).cpu().numpy()  # (B,H,W)
        return extrinsic, intrinsic, depth_map, depth_conf

    def preprocess_static_video(self, video_id, use_ba=True):
        video_static_frames_dir = self.static_videos_dir / video_id
        assert video_static_frames_dir.exists(), f"Video frames directory {video_static_frames_dir} does not exist."

        frames_path_list = sorted([str(p) for p in video_static_frames_dir.glob("*.png")])

        self.video_scene_dir = os.path.join(self.static_scenes_dir, video_id)
        os.makedirs(self.video_scene_dir, exist_ok=True)

        # Load images & original coords (pad+resize to square for VGGT input)
        images, original_coords = load_and_preprocess_images_square(frames_path_list, self.img_load_resolution)
        images = images.to(self.device)
        original_coords = original_coords.to(self.device)

        # Run VGGT (cameras+depth)
        extrinsic, intrinsic, depth_map, depth_conf = self.run_video_vggt(images, self.vggt_fixed_resolution)
        points_3d_dense = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (S,H,W,3)
        shared_camera = self.shared_camera

        if use_ba:
            image_size = np.array(images.shape[-2:])  # (H,W) at 1024 load-res
            scale = self.img_load_resolution / self.vggt_fixed_resolution

            with torch.cuda.amp.autocast(dtype=self.dtype):
                # Predict 2D tracks + sparse 3D seeds (VGGSfM tracker)
                pred_tracks, pred_vis_scores, pred_confs, points_3d_sparse, points_rgb_sparse = predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d_dense,
                    masks=None,
                    max_query_pts=self.max_query_pts,
                    query_frame_num=self.query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=self.fine_tracking,
                )
                torch.cuda.empty_cache()

            # Rescale intrinsics from 518->1024
            intrinsic[:, :2, :] *= scale
            track_mask = pred_vis_scores > self.vis_thresh  # (S,P)

            # Build reconstruction from tracks & run COLMAP BA
            reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                points_3d_sparse,
                extrinsic,
                intrinsic,
                pred_tracks,
                image_size,
                masks=track_mask,
                max_reproj_error=self.max_reproj_error,
                shared_camera=shared_camera,
                camera_type=self.camera_type,
                points_rgb=points_rgb_sparse,
            )

            if reconstruction is None:
                raise ValueError("No reconstruction can be built with BA")

            ba_options = pycolmap.BundleAdjustmentOptions()
            pycolmap.bundle_adjustment(reconstruction, ba_options)
            reconstruction_resolution = self.img_load_resolution

        # ==============================
        # No-BA feed-forward export path
        # ==============================
        conf_thres_value = self.conf_threshold_value
        max_points_for_colmap = 100000
        shared_camera = False  # feed-forward path: no shared camera
        camera_type = "PINHOLE"

        image_size = np.array([self.vggt_fixed_resolution, self.vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d_dense.shape

        points_rgb = F.interpolate(
            images, size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        pts3d = points_3d_dense[conf_mask]
        ptsxyf = points_xyf[conf_mask]
        ptsrgb = points_rgb[conf_mask]

        print("Converting to COLMAP format (feed-forward)")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            pts3d,
            ptsxyf,
            ptsrgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction = self.rename_colmap_recons_and_rescale_camera(
            reconstruction,
            frames_path_list,
            original_coords.cpu().numpy(),
            img_size=self.vggt_fixed_resolution,
            shift_point2d_to_original_res=True,
            shared_camera=shared_camera,
        )

        print(f"Saving reconstruction to {self.video_scene_dir}/sparse")
        sparse_reconstruction_dir = os.path.join(self.video_scene_dir, "sparse")
        os.makedirs(sparse_reconstruction_dir, exist_ok=True)
        reconstruction.write(sparse_reconstruction_dir)

        # Save point cloud for quick visualization
        try:
            trimesh.PointCloud(pts3d, colors=ptsrgb).export(os.path.join(self.video_scene_dir, "sparse/points.ply"))
        except Exception as e:
            print(f"[Warn] Failed to export points.ply: {e}")

        return True

    def rename_colmap_recons_and_rescale_camera(
            self,
            reconstruction,
            image_paths,
            original_coords,
            img_size,
            shift_point2d_to_original_res=False,
            shared_camera=False
    ):
        rescale_camera = True

        for pyimageid in reconstruction.images:
            # Reshape padded&resized image back to the original size + rename
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_paths[pyimageid - 1]

            if rescale_camera:
                pred_params = copy.deepcopy(pycamera.params)
                real_image_size = original_coords[pyimageid - 1, -2:]
                resize_ratio = max(real_image_size) / img_size
                pred_params = pred_params * resize_ratio
                real_pp = real_image_size / 2
                pred_params[-2:] = real_pp  # set principal point to image center
                pycamera.params = pred_params
                pycamera.width = real_image_size[0]
                pycamera.height = real_image_size[1]

            if shift_point2d_to_original_res:
                top_left = original_coords[pyimageid - 1, :2]
                for point2D in pyimage.points2D:
                    point2D.xy = (point2D.xy - top_left) * resize_ratio

            if shared_camera:
                rescale_camera = False

        return reconstruction


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--static_frames_dir", default="/data/rohith/ag/ag4D/static_frames",
                        help="Folder containing static scene videos.")
    parser.add_argument("--static_scenes_dir", default="/data/rohith/ag/ag4D/static_scenes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # BA selection
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use bundle adjustment")

    # BA (shared) params
    parser.add_argument("--max_reproj_error", type=float, default=8.0, help="Max reprojection error for reconstruction")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument("--fine_tracking", action="store_true", default=True,
                        help="Use fine tracking (slower but more accurate)")

    # No-BA feed-forward export
    parser.add_argument("--conf_thres_value", type=float, default=5.0,
                        help="Confidence threshold for depth filtering (wo BA)")
    return parser.parse_args()
