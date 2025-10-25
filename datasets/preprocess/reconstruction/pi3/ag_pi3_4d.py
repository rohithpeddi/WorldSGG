import argparse
import gc
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rerun as rr
import torch
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as SciRot
from tqdm import tqdm


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



def predictions_to_glb_with_static(
        predictions: dict,
        *,
        conf_min: float = 0.5,           # 0..1 threshold on predictions["conf"]
        filter_by_frames: str = "all",   # e.g. "12:..." to use only frame index 12
) -> tuple[trimesh.Scene, np.ndarray, np.ndarray]:
    """
    Build a GLB-ready trimesh.Scene from VGGT-style predictions AND return a background
    point cloud (xyz,rgb) in the original world frame.

    Inputs (expected prediction keys):
      - points:  (S, H, W, 3) float32   world-space points
      - conf:    (S, H, W)    float32   confidence per point in [0,1] (fallback: ones)
      - images:  (S, H, W, 3) or (S, 3, H, W) float32 in [0,1] for colors
      - camera_poses: (S, 3, 4) or (S,4,4); kept for signature parity (not visualized here)

    Returns:
      scene_3d      : trimesh.Scene with a point cloud (rotated for nicer viewing)
      static_points : (N,3) float32 background points in ORIGINAL world frame
      static_colors : (N,3) uint8   RGB colors aligned with static_points
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # ------------ Frame selection ------------
    selected_frame_idx = None
    if filter_by_frames not in ("all", "All"):
        try:
            selected_frame_idx = int(str(filter_by_frames).split(":")[0])
        except (ValueError, IndexError):
            selected_frame_idx = None

    pts = predictions["points"]
    conf = predictions.get("conf", np.ones_like(pts[..., 0], dtype=np.float32))
    imgs = predictions["images"]
    cam_poses = predictions.get("camera_poses", None)

    if selected_frame_idx is not None:
        pts = pts[selected_frame_idx][None]
        conf = conf[selected_frame_idx][None]
        imgs = imgs[selected_frame_idx][None]
        cam_poses = cam_poses[selected_frame_idx][None] if cam_poses is not None else None

    # ------------ Color layout handling ------------
    if imgs.ndim == 4 and imgs.shape[1] == 3:  # NCHW -> NHWC
        imgs_nhwc = np.transpose(imgs, (0, 2, 3, 1))
    else:
        imgs_nhwc = imgs
    colors_rgb = (imgs_nhwc.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

    # ------------ Confidence filtering ------------
    verts = pts.reshape(-1, 3)
    conf_flat = conf.reshape(-1).astype(np.float32)

    thr = float(conf_min) if conf_min is not None else 0.1
    mask = (conf_flat >= thr) & (conf_flat > 1e-5)

    verts = verts[mask]
    colors_rgb = colors_rgb[mask]

    if verts.size == 0:
        # robust fallback
        verts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        colors_rgb = np.array([[255, 255, 255]], dtype=np.uint8)

    # Keep an ORIGINAL copy for the background return (world frame)
    static_points = verts.astype(np.float32, copy=True)
    static_colors = colors_rgb.astype(np.uint8, copy=True)

    # ------------ Build Scene (with visualization alignment rotation) ------------
    scene_3d = trimesh.Scene()
    point_cloud = trimesh.points.PointCloud(vertices=verts, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud)

    # Nice-view rotation: R = R_y(100°) @ R_x(155°)
    align_R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    align_R = align_R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = align_R
    scene_3d.apply_transform(T)

    # (Optional) If you later want to visualize camera frustums/axes, add them here
    # using cam_poses if available. Currently `show_cam` is a no-op for signature parity.
    R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    R = R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    static_points = static_points @ R.T  # undo rotation

    return scene_3d, static_points, static_colors


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

def _frustum_path(i: int) -> str:
    return f"world/frames/t{i}/frustum"

def _pinhole_from_fov(W: int, H: int, fov_y_rad: float) -> Tuple[float, float, float, float]:
    """Compute fx, fy, cx, cy from vertical FOV (in radians) and resolution."""
    fy = (H * 0.5) / np.tan(0.5 * fov_y_rad)
    fx = fy * (W / H)
    cx = W * 0.5
    cy = H * 0.5
    return fx, fy, cx, cy

class AgPi3:

    def __init__(
            self,
            root_dir_path,  # data/rohith/ag/ag4D/static_frames
            output_dir_path=None
    ):
        self.model = None
        self.root_dir_path = root_dir_path
        self.output_dir_path = output_dir_path if output_dir_path is not None else root_dir_path
        os.makedirs(self.output_dir_path, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer_video_points_3d(self, video_id, conf_min: float = 0.01, spawn: bool = True):
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

        S, H, W = points_wh.shape[:3]

        S = points_wh.shape[0]
        print(f"[viz] Visualizing {S} frames, conf_min={conf_min}")

        # Initialize Rerun
        rr.init(f"AG-Pi3 Per-Frame Points: {video_id}", spawn=spawn)
        rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

        cam_poses = predictions["camera_poses"]  # (S, 4, 4) or (S, 3, 4)

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

            R, t = _camera_R_t_from_4x4(cam_poses[i])
            R_wc = R
            t_wc = t

            fov_y = 0.96  # mirrors your viser default
            fx, fy, cx, cy = _pinhole_from_fov(W, H, fov_y)

            q_xyzw = SciRot.from_matrix(R).as_quat()  # [x, y, z, w]
            rr.set_time_sequence("frame", i)

            frus_path = _frustum_path(i)
            rr.log(
                frus_path,
                rr.Transform3D(
                    translation=t_wc.astype(np.float32),
                    rotation=rr.Quaternion(xyzw=q_xyzw),
                )
            )
            rr.log(
                f"{frus_path}/camera",
                rr.Pinhole(focal_length=(fx, fy), principal_point=(cx, cy), resolution=(W, H)),
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
        video_save_dir = os.path.join(self.output_dir_path, f"{video_id}_{int(conf_thres)}")

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
        scene_3d, static_P, static_C = predictions_to_glb_with_static(predictions, conf_min=0.1)

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
        video_id_list = ["0DJ6R"]
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
        "--root_dir_path", type=str, default="/data/rohith/ag/frames",
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
        "--output_dir_path", type=str, default="/data3/rohith/ag/ag4D/static_scenes/pi3_full",
        help="Path to output directory where results will be saved."
    )
    # parser.add_argument(
    #     "--output_dir_path", type=str, default="/data3/rohith/ag/ag4D/static_scenes/pi3_inpaint",
    #     help="Path to output directory where results will be saved."
    # )
    parser.add_argument(
        "--split", type=_parse_split, default="04",
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
