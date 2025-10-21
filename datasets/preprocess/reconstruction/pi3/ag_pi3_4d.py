import argparse
import gc
import os
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import rerun as rr
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as SciRot
from pi3.utils.basic import load_images_as_tensor
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


# ---------------------------
# Split logic (yours)
# ---------------------------

def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    """
    Get the split that the video belongs to based on its ID.
    Accepts either a bare ID (e.g., '0DJ6R') or a filename (e.g., '0DJ6R.mp4').
    """
    stem = Path(video_id).stem
    if not stem:
        return None
    first_letter = stem[0]
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
    return None


def _ensure_nhwc(images: np.ndarray) -> np.ndarray:
    """Ensure images are NHWC in [0,1]."""
    if images.ndim != 4:
        raise ValueError("images must be 4D")
    if images.shape[1] == 3:  # NCHW -> NHWC
        images = np.transpose(images, (0, 2, 3, 1))
    return np.clip(images, 0.0, 1.0)


def _flatten_points_colors_frames(points_wh: np.ndarray,
                                  colors_wh: np.ndarray,
                                  conf_wh: np.ndarray,
                                  conf_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten (S,H,W,3) points/colors and (S,H,W[,1]) conf to arrays:
      P: (N,3) points, C: (N,3) uint8 colors, F: (N,) frame idx
    Filter by conf >= conf_min and conf > 1e-5; drop NaNs/Infs.
    """
    S = points_wh.shape[0]
    if conf_wh.ndim == 4 and conf_wh.shape[-1] == 1:
        conf_wh = conf_wh[..., 0]
    mask = (conf_wh >= conf_min) & (conf_wh > 1e-5)

    P = points_wh[mask]  # (N,3)
    C = (colors_wh[mask] * 255.0).astype(np.uint8)  # (N,3)
    F = np.repeat(np.arange(S), repeats=points_wh.shape[1] * points_wh.shape[2])[mask.ravel()]

    good = np.isfinite(P).all(axis=1)
    return P[good], C[good], F[good]


def _voxel_ids(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Integer voxel coordinates for each point."""
    return np.floor(points / voxel_size).astype(np.int64)


def predictions_to_glb(
        predictions,
        conf_thres=50.0,
        filter_by_frames="all",
        show_cam=True,
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        show_cam (bool): Include camera visualization (default: True)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_poses = predictions["camera_poses"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_poses = camera_poses[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        # conf_threshold = np.percentile(conf, conf_thres)
        conf_threshold = conf_thres / 100

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_poses)

    # Rotate scene for better visualize
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()  # plane rotate
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()  # roll
    scene_3d.apply_transform(align_rotation)

    print("GLB Scene built")
    return scene_3d


def build_static_background(
        predictions: dict,
        conf_min: float = 0.5,
        voxel_size: float = 0.03,
        min_frames: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replacement that *uses predictions_to_glb* to generate a background point cloud,
    then extracts (xyz, rgb) from the returned trimesh.Scene.

    Notes / differences vs the old implementation:
      - Ignores true 'static across frames' logic (no per-voxel frame counting).
      - Approximates a static background by aggregating all points that survive
        predictions_to_glb's confidence filtering, then (optionally) voxel-downsampling.
      - Automatically *undoes* the visualization rotation applied inside predictions_to_glb,
        so outputs are back in the original world frame.
      - 'min_frames' is kept for signature compatibility but not used.

    Returns:
        static_points: (N,3) float32
        static_colors: (N,3) uint8
    """
    # 1) Call your existing predictions_to_glb with show_cam=False to avoid camera meshes.
    #    Map conf_min (0..1) -> conf_thres percentage (0..100).
    conf_thres_pct = float(conf_min) * 100.0
    scene = predictions_to_glb(
        predictions=predictions,
        conf_thres=conf_thres_pct,
        filter_by_frames="all",
        show_cam=False
    )

    # 2) Extract point cloud geometry from the scene (PointCloud only).
    pts_list, cols_list = [], []
    for geom in scene.geometry.values():
        # We only want point clouds, not camera meshes or other geometry
        if isinstance(geom, trimesh.points.PointCloud):
            v = np.asarray(geom.vertices, dtype=np.float32)
            c = np.asarray(geom.colors)
            # colors can be (N,3) or (N,4); drop alpha if present
            if c.ndim == 2 and c.shape[1] >= 3:
                c = c[:, :3]
            else:
                # If no colors, default to white
                c = np.full((v.shape[0], 3), 255, dtype=np.uint8)
            # Ensure types
            v = v.astype(np.float32, copy=False)
            c = c.astype(np.uint8, copy=False)
            if v.size:
                pts_list.append(v)
                cols_list.append(c)

    if not pts_list:
        # Fallback: empty scene protection
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(pts_list, axis=0)
    colors = np.concatenate(cols_list, axis=0)

    # 3) Undo predictions_to_glb's visualization alignment (Y=100°, then X=155°).
    #    In predictions_to_glb:
    #        R = R_y(100) @ R_x(155)
    #    We apply R^T to revert.
    R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    R = R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    points = points @ R.T  # undo rotation

    # 4) Optional voxel downsample to keep the cloud compact (averaging xyz & rgb per voxel).
    if voxel_size and voxel_size > 0:
        points, colors = _voxel_downsample_mean(points, colors, voxel_size)

    return points.astype(np.float32), colors.astype(np.uint8)


def _voxel_downsample_mean(points: np.ndarray, colors: np.ndarray, voxel: float):
    """
    Simple voxel grid downsampling that averages positions and colors per voxel.
    """
    if points.size == 0:
        return points, colors

    # Compute voxel indices
    idx = np.floor(points / voxel).astype(np.int64)
    # Pack 3D indices to a single key for grouping
    keys = np.ascontiguousarray(idx).view([('', idx.dtype)] * 3).ravel()

    uniq, inv = np.unique(keys, return_inverse=True)
    G = uniq.shape[0]

    # Aggregate means per voxel
    counts = np.bincount(inv, minlength=G).astype(np.float32)
    sum_x = np.bincount(inv, weights=points[:, 0], minlength=G)
    sum_y = np.bincount(inv, weights=points[:, 1], minlength=G)
    sum_z = np.bincount(inv, weights=points[:, 2], minlength=G)

    sum_r = np.bincount(inv, weights=colors[:, 0].astype(np.float32), minlength=G)
    sum_g = np.bincount(inv, weights=colors[:, 1].astype(np.float32), minlength=G)
    sum_b = np.bincount(inv, weights=colors[:, 2].astype(np.float32), minlength=G)

    counts[counts == 0] = 1.0
    out_xyz = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
    out_rgb = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    return out_xyz.astype(np.float32), out_rgb.astype(np.uint8)


def merge_static_with_frame(predictions: dict,
                            static_points: np.ndarray,
                            static_colors: np.ndarray,
                            frame_idx: int,
                            conf_min: float = 0.1,
                            dedup_voxel: Optional[float] = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge per-frame points with static background. Optionally de-duplicate with a small voxel size.
    """
    images = _ensure_nhwc(predictions["images"])  # (S,H,W,3)
    points = predictions["points"]  # (S,H,W,3)
    conf = predictions["conf"]  # (S,H,W[,1])

    # slice this frame
    pts_f = points[frame_idx]  # (H,W,3)
    img_f = images[frame_idx]  # (H,W,3)
    conf_f = conf[frame_idx]  # (H,W[,1])

    P, C, _ = _flatten_points_colors_frames(pts_f[None], img_f[None], conf_f[None], conf_min)

    if P.size == 0:
        merged_P = static_points
        merged_C = static_colors
    else:
        merged_P = np.concatenate([static_points, P], axis=0)
        merged_C = np.concatenate([static_colors, C], axis=0)

    if dedup_voxel is None or merged_P.size == 0:
        return merged_P.astype(np.float32), merged_C.astype(np.uint8)

    # light voxel de-dup to reduce overlap
    vox = _voxel_ids(merged_P, dedup_voxel)
    vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
    uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
    G = uniq_keys.shape[0]

    counts = np.bincount(inv, minlength=G).astype(np.float32)
    sum_x = np.bincount(inv, weights=merged_P[:, 0], minlength=G)
    sum_y = np.bincount(inv, weights=merged_P[:, 1], minlength=G)
    sum_z = np.bincount(inv, weights=merged_P[:, 2], minlength=G)

    sum_r = np.bincount(inv, weights=merged_C[:, 0].astype(np.float32), minlength=G)
    sum_g = np.bincount(inv, weights=merged_C[:, 1].astype(np.float32), minlength=G)
    sum_b = np.bincount(inv, weights=merged_C[:, 2].astype(np.float32), minlength=G)

    counts[counts == 0] = 1.0
    mean_xyz = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1).astype(np.float32)
    mean_rgb = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
    return mean_xyz, mean_rgb


def _camera_R_t_from_4x4(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotation (3x3) and translation (3,) from a 4x4 transform."""
    if T.shape == (3, 4):
        R, t = T[:, :3], T[:, 3]
    elif T.shape == (4, 4):
        R, t = T[:3, :3], T[:3, 3]
    else:
        raise ValueError("camera pose must be (3,4) or (4,4)")
    return R.astype(np.float32), t.astype(np.float32)


def visualize_with_rerun(predictions: dict,
                         static_points: np.ndarray,
                         static_colors: np.ndarray,
                         per_frame_points: list[np.ndarray],
                         per_frame_colors: list[np.ndarray],
                         log_cameras: bool = True,
                         spawn: bool = True):
    rr.init("AG-Pi3 Scenes", spawn=spawn)
    rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

    # Log static background timelessly
    if static_points.size > 0:
        rr.log("world/static",
               rr.Points3D(positions=static_points.astype(np.float32),
                           colors=static_colors.astype(np.uint8)),
               timeless=True)

    # Cameras (optional): log transforms for each frame
    if log_cameras and "camera_poses" in predictions:
        cam_poses = predictions["camera_poses"]  # (S, 4, 4) or (S, 3, 4)
        for i in range(len(cam_poses)):
            R, t = _camera_R_t_from_4x4(cam_poses[i])
            q_xyzw = SciRot.from_matrix(R).as_quat()  # [x, y, z, w]
            # Rerun expects a transform; we log per-frame transform timeless (pose-only)
            rr.log(
                f"world/cameras/cam_{i}",
                rr.Transform3D(
                    translation=t,
                    rotation=rr.Quaternion(xyzw=q_xyzw),
                ),
            )

    # Animate per-frame merged points
    for i, (P, C) in enumerate(zip(per_frame_points, per_frame_colors)):
        rr.set_time_sequence("frame", i)
        if P.size == 0:
            continue
        rr.log("world/frame/merged_points",
               rr.Points3D(
                   positions=P.astype(np.float32),
                   colors=C.astype(np.uint8),
                   radii=0.01)
               )


class AgPi3:

    def __init__(
            self,
            root_dir_path,  # data/rohith/ag/ag4D/static_frames
            output_dir_path=None,
            frame_annotated_dir_path=None,
    ):
        self.model = None
        self.root_dir_path = root_dir_path
        self.output_dir_path = output_dir_path if output_dir_path is not None else root_dir_path
        os.makedirs(self.output_dir_path, exist_ok=True)

        self.frame_annotated_dir_path = frame_annotated_dir_path
        self.sampled_frames_idx_root_dir = "/data/rohith/ag/sampled_frames_idx"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        pass

    def preprocess_image_list(self, data_path, is_video=False, video_id=None, sample_idx=None):
        interval = 10 if is_video else 1
        print(f'Sampling interval: {interval}')
        imgs = load_images_as_tensor(
            data_path,
            interval=interval,
            video_id=video_id,
            sample_idx=sample_idx
        ).to(self.device)  # (N, 3, H, W)
        return imgs

    def infer_video_points_3d(self, video_id, conf_min: float = 0.01, spawn: bool = True):
        """
        Load saved predictions for a video and visualize *all* points per frame
        as a time sequence in Rerun.

        Args:
            video_id: e.g., "BHP7U" or "BHP7U.mp4"
            conf_min: confidence threshold in [0,1] to keep a point
            spawn: whether to spawn the Rerun viewer window
        """
        # Use a readable tag for the folder name based on confidence threshold
        conf_tag = int(conf_min * 100)
        video_save_dir = os.path.join(self.output_dir_path, f"{video_id}_10")
        prediction_save_path = os.path.join(video_save_dir, "predictions.npz")
        if os.path.exists(prediction_save_path):
            predictions = np.load(prediction_save_path, allow_pickle=True)
            predictions = {k: predictions[k] for k in predictions.files}
            print(f"Loaded existing predictions for video {video_id} from {prediction_save_path}")
        else:
            raise NotImplementedError("Prediction generation not implemented in this snippet.")

        # Extract tensors
        points_wh = predictions["points"]  # (S, H, W, 3)
        conf_wh = predictions.get("conf", np.ones(points_wh.shape[:-1], dtype=np.float32))  # (S, H, W) or (S,H,W,1)
        images = _ensure_nhwc(predictions["images"])  # to (S, H, W, 3) in [0,1]

        S = points_wh.shape[0]
        print(f"[viz] Visualizing {S} frames, conf_min={conf_min}")

        # Initialize Rerun
        rr.init(f"AG-Pi3 Per-Frame Points: {video_id}", spawn=spawn)
        rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

        # Optionally log cameras (animated by frame if available)
        if "camera_poses" in predictions:
            cam_poses = predictions["camera_poses"]  # (S, 4, 4) or (S, 3, 4)
            for i in range(len(cam_poses)):
                R, t = _camera_R_t_from_4x4(cam_poses[i])
                q_xyzw = SciRot.from_matrix(R).as_quat()  # [x, y, z, w]
                rr.set_time_sequence("frame", i)
                rr.log(
                    f"world/cameras/cam",
                    rr.Transform3D(
                        translation=t.astype(np.float32),
                        rotation=rr.Quaternion(xyzw=q_xyzw.astype(np.float32)),
                    ),
                )

        # Animate per-frame raw points
        for i in range(S):
            rr.set_time_sequence("frame", i)

            # Slice single frame to shapes with batch dim for the helper
            pts_i = points_wh[i][None]  # (1, H, W, 3)
            img_i = images[i][None]  # (1, H, W, 3)
            conf_i = conf_wh[i][None]  # (1, H, W) or (1, H, W, 1)

            P, C, _ = _flatten_points_colors_frames(
                points_wh=pts_i,
                colors_wh=img_i,
                conf_wh=conf_i,
                conf_min=float(conf_min),
            )

            if P.size == 0:
                # Nothing to draw for this frame
                continue

            rr.log(
                "world/frame/points_raw",
                rr.Points3D(
                    positions=P.astype(np.float32),
                    colors=C.astype(np.uint8),
                    radii=0.01,  # tweak if your scale is very small/large
                ),
            )

        print("[viz] Done streaming per-frame points to Rerun.")

    def infer_video(self, video_id, conf_thres=10.0):
        data_path = f'{self.root_dir_path}/{video_id}'
        video_save_dir = os.path.join(self.output_dir_path, f"{video_id}_{int(conf_thres)}")

        # video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)
        # annotated_frame_id_list = os.listdir(video_frames_annotated_dir_path)
        # annotated_frame_id_list = [f for f in annotated_frame_id_list if f.endswith('.png')]
        # annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])  # Number only

        # Get the mapping for sampled_frame_id and the actual frame id
        # Now start from the sampled frame which corresponds to the first annotated frame and keep the rest of the sampled frames
        # video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        # video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()  # Numbers only

        prediction_save_path = os.path.join(video_save_dir, "predictions.npz")
        if os.path.exists(prediction_save_path):
            predictions = np.load(prediction_save_path, allow_pickle=True)
            predictions = {k: predictions[k] for k in predictions.files}
            print(f"Loaded existing predictions for video {video_id} from {prediction_save_path}")
        else:
            raise NotImplementedError("Prediction generation not implemented in this snippet.")

        # -------------------------
        # NEW: background + per-frame scenes + rerun visualization
        # -------------------------
        # 1) Static background from 'sparse' points (multi-frame consistency via voxels)
        static_P, static_C = build_static_background(
            predictions,
            conf_min=0.1,  # high-confidence for background
            voxel_size=0.01,  # 3cm voxels
            min_frames=3  # seen in >= 3 frames -> static
        )

        # 2) Per-frame merges (background + frame points)
        S = predictions["points"].shape[0]
        per_frame_P, per_frame_C = [], []
        for fi in range(S):
            Pi, Ci = merge_static_with_frame(
                predictions,
                static_P, static_C,
                frame_idx=fi,
                conf_min=0.01,  # slightly looser for live frame
                dedup_voxel=0.02  # 2cm dedup to keep clouds tidy
            )
            per_frame_P.append(Pi)
            per_frame_C.append(Ci)

        # 3) Visualize with rerun (animated over frames)
        visualize_with_rerun(
            predictions,
            static_P, static_C,
            per_frame_P, per_frame_C,
            log_cameras=True,
            spawn=True
        )

        # Cleanup
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

    def infer_all_videos(self, split):
        video_id_list = os.listdir(self.root_dir_path)
        video_id_list = ["BHP7U"]
        for video_id in tqdm(video_id_list):
            if get_video_belongs_to_split(video_id) != split:
                print(f"Skipping video {video_id} not in split {split}")
                continue
            # self.infer_video(video_id)
            self.infer_video_points_3d(video_id)
            # try:
            #     self.infer_video(video_id)
            # except Exception as e:
            #     print(f"[ERROR] Error processing video {video_id}: {e}")


def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample frames from videos based on homography-overlap filtering."
    )
    parser.add_argument(
        "--root_dir_path", type=str, default=r"E:\DATA\COLLECTED\AG4D\Pi3",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )
    # parser.add_argument(
    #     "--root_dir_path", type=str, default="/data/rohith/ag/ag4D/static_frames",
    #     help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    # )
    # parser.add_argument(
    #     "--root_dir_path", type=str, default="/data/rohith/ag/segmentation/masks/rectangular_overlayed_frames",
    #     help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    # )
    parser.add_argument(
        "--frames_annotated_dir_path", type=str, default="/data/rohith/ag/frames_annotated",
        help="Path to directory containing annotated frames (with masks)."
    )
    parser.add_argument(
        "--output_dir_path", type=str, default=r"E:\DATA\COLLECTED\AG4D\Pi3",
        help="Path to output directory where results will be saved."
    )
    # parser.add_argument(
    #     "--output_dir_path", type=str, default="/data3/rohith/ag/ag4D/static_scenes/pi3_inpaint",
    #     help="Path to output directory where results will be saved."
    # )
    parser.add_argument(
        "--split", type=_parse_split, default="AD",
        help="Optional shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}. "
             "If omitted, processes all videos."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ag_pi3 = AgPi3(
        root_dir_path=args.root_dir_path,
        output_dir_path=args.output_dir_path,
        frame_annotated_dir_path=args.frames_annotated_dir_path,
    )
    ag_pi3.infer_all_videos(args.split)


if __name__ == "__main__":
    main()
