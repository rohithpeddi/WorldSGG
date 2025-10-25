import argparse
import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rerun as rr
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rot
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


# ---------------------------
# Utils
# ---------------------------

def _ensure_nhwc(images: np.ndarray) -> np.ndarray:
    """Ensure images are NHWC in [0,1]."""
    if images.ndim != 4:
        raise ValueError("images must be 4D")
    if images.shape[1] == 3:  # NCHW -> NHWC
        images = np.transpose(images, (0, 2, 3, 1))
    return np.clip(images, 0.0, 1.0)


def _flatten_points_colors_frames(
    points_wh: np.ndarray,
    colors_wh: np.ndarray,
    conf_wh: np.ndarray,
    conf_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _camera_R_t_from_4x4(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rotation (3x3) and translation (3,) from a 4x4 or 3x4 transform."""
    if T.shape == (3, 4):
        Rm, t = T[:, :3], T[:, 3]
    elif T.shape == (4, 4):
        Rm, t = T[:3, :3], T[:3, 3]
    else:
        raise ValueError("camera pose must be (3,4) or (4,4)")
    return Rm.astype(np.float32), t.astype(np.float32)


def _pinhole_from_fov(W: int, H: int, fov_y_rad: float) -> Tuple[float, float, float, float]:
    """Compute fx, fy, cx, cy from vertical FOV (in radians) and resolution."""
    fy = (H * 0.5) / np.tan(0.5 * fov_y_rad)
    fx = fy * (W / H)
    cx = W * 0.5
    cy = H * 0.5
    return fx, fy, cx, cy


def _frustum_path(i: int) -> str:
    return f"world/frames/t{i}/frustum"


# ---------------------------
# Static scene builder
# ---------------------------

def predictions_to_glb_with_static(
    predictions: Dict,
    *,
    conf_min: float = 0.5,  # 0..1 threshold on predictions["conf"]
    filter_by_frames: str = "all",  # e.g. "12:..." to use only frame index 12
) -> Tuple[trimesh.Scene, np.ndarray, np.ndarray]:
    """
    Build a GLB-ready trimesh.Scene from VGGT-style predictions AND return a background
    point cloud (xyz,rgb) in the original world frame.

    Inputs (expected prediction keys):
      - points:        (S, H, W, 3) float32   world-space points
      - conf:          (S, H, W)    float32   confidence per point in [0,1] (fallback: ones)
      - images:        (S, H, W, 3) or (S, 3, H, W) float32 in [0,1] for colors
      - camera_poses:  (S, 3, 4) or (S,4,4); kept for signature parity (not visualized here)

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
    align_R = Rot.from_euler("y", 100, degrees=True).as_matrix()
    align_R = align_R @ Rot.from_euler("x", 155, degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = align_R
    scene_3d.apply_transform(T)

    # (Optional) If you later want to visualize camera frustums/axes, add them here
    # using cam_poses if available. Currently `show_cam` is a no-op for signature parity.
    # Undo the visualization rotation for the returned static points (stay in world frame)
    static_points = static_points @ align_R.T
    return scene_3d, static_points, static_colors


def merge_static_with_frame(
    predictions: Dict,
    static_points: np.ndarray,
    static_colors: np.ndarray,
    frame_idx: int,
    conf_min: float = 0.1,
    dedup_voxel: Optional[float] = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge per-frame points with static background. Optionally de-duplicate with a small voxel size."""
    images = _ensure_nhwc(predictions["images"])  # (S,H,W,3)
    points = predictions["points"]  # (S,H,W,3)
    conf = predictions.get("conf", np.ones(points.shape[:-1], dtype=np.float32))  # (S,H,W[,1])

    # slice this frame
    pts_f = points[frame_idx]  # (H,W,3)
    img_f = images[frame_idx]  # (H,W,3)
    conf_f = conf[frame_idx]   # (H,W[,1])

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


# ---------------------------
# Rerun visualization
# ---------------------------

def _log_cameras(predictions: Dict, fov_y: float, W: int, H: int):
    if "camera_poses" not in predictions:
        return
    cam_poses = predictions["camera_poses"]  # (S, 4, 4) or (S, 3, 4)
    fx, fy, cx, cy = _pinhole_from_fov(W, H, fov_y)

    for i, Tcw in enumerate(cam_poses):
        Rcw, tcw = _camera_R_t_from_4x4(Tcw)
        q_xyzw = R.from_matrix(Rcw).as_quat().astype(np.float32)  # [x, y, z, w]

        frus_path = _frustum_path(i)
        rr.log(
            frus_path,
            rr.Transform3D(
                translation=tcw.astype(np.float32),
                rotation=rr.Quaternion(xyzw=q_xyzw),
            ),
        )
        rr.log(
            f"{frus_path}/camera",
            rr.Pinhole(
                focal_length=(float(fx), float(fy)),
                principal_point=(float(cx), float(cy)),
                resolution=(int(W), int(H)),
            ),
        )


# ---------------------------
# Main class
# ---------------------------

class AgPi3:
    def __init__(
        self,
        root_dir_path: str,
        output_dir_path: Optional[str] = None,
        static_scene_dir_path: Optional[str] = None,  # accepted for parity; not used here
        frame_annotated_dir_path: Optional[str] = None,  # accepted for parity; not used here
    ):
        self.model = None
        self.root_dir_path = root_dir_path
        self.static_scene_dir_path = static_scene_dir_path
        self.output_dir_path = output_dir_path if output_dir_path is not None else root_dir_path
        self.frame_annotated_dir_path = frame_annotated_dir_path
        os.makedirs(self.output_dir_path, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- Unified pipeline ---------
    def infer_video(
        self,
        video_id: str,
        *,
        mode: str = "both",            # one of {"raw", "merged", "both"}
        conf_static: float = 0.10,      # confidence for static background build
        conf_frame: float = 0.01,       # confidence for per-frame points
        dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
        fov_y: float = 0.96,            # radians; matches your earlier default
        spawn: bool = True,
        log_cameras: bool = True,
    ) -> None:
        """
        Unified visualization that replaces the old `infer_video` and `infer_video_points_3d`.

        - Builds a static background once (using `conf_static`).
        - Optionally logs per-frame RAW points (like the old `infer_video_points_3d`).
        - Optionally logs per-frame MERGED (static + frame) points (like the old `infer_video`).
        """
        assert mode in {"raw", "merged", "both"}, "mode must be one of {'raw','merged','both'}"

        # ---- Load predictions once ----
        pred_path = os.path.join(self.static_scene_dir_path, f"{video_id}_{10}", "predictions.npz")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(
                f"predictions.npz not found for {video_id} under {self.output_dir_path} (looked for {video_id}_*/predictions.npz)"
            )

        arr = np.load(pred_path, allow_pickle=True, mmap_mode=None)
        predictions = {k: arr[k] for k in arr.files}
        points_wh = predictions["points"]  # (S,H,W,3)
        images = _ensure_nhwc(predictions["images"])  # (S,H,W,3) in [0,1]
        conf_wh = predictions.get("conf")  # (S,H,W) or (S,H,W,1)

        S, H, W = points_wh.shape[:3]
        print(f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame} | mode={mode}")

        # ---- Build static background (once) ----
        _scene_3d, static_P, static_C = predictions_to_glb_with_static(
            predictions, conf_min=float(conf_static)
        )

        # ---- Precompute per-frame RAW points (avoid recomputing for both branches) ----
        raw_P: List[np.ndarray] = []
        raw_C: List[np.ndarray] = []
        for i in range(S):
            P_i, C_i, _ = _flatten_points_colors_frames(
                points_wh=points_wh[i][None],
                colors_wh=images[i][None],
                conf_wh=conf_wh[i][None],
                conf_min=float(conf_frame),
            )
            raw_P.append(P_i)
            raw_C.append(C_i)

        # ---- Precompute per-frame MERGED with static ----
        merged_P: List[np.ndarray] = []
        merged_C: List[np.ndarray] = []
        if mode in {"merged", "both"}:
            for i in range(S):
                Pi, Ci = merge_static_with_frame(
                    predictions,
                    static_P, static_C,
                    frame_idx=i,
                    conf_min=float(conf_frame),
                    dedup_voxel=dedup_voxel,
                )
                merged_P.append(Pi)
                merged_C.append(Ci)

        # ---- Rerun setup ----
        rr.init(f"AG-Pi3: {video_id}", spawn=spawn)
        rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

        # Log static background timelessly
        if static_P.size > 0:
            rr.log(
                "world/static",
                rr.Points3D(
                    positions=static_P.astype(np.float32),
                    colors=static_C.astype(np.uint8),
                ),
                timeless=True,
            )

        # Cameras & frustums (timeless transforms, separate camera nodes per frame)
        if log_cameras:
            _fx, _fy, _cx, _cy = _pinhole_from_fov(W, H, fov_y)
            _log_cameras(predictions, fov_y=fov_y, W=W, H=H)

        # ---- Animate per-frame content ----
        for i in range(S):
            rr.set_time_sequence("frame", i)

            if mode in {"raw", "both"} and raw_P[i].size:
                rr.log(
                    "world/frame/points_raw",
                    rr.Points3D(
                        positions=raw_P[i].astype(np.float32),
                        colors=raw_C[i].astype(np.uint8),
                        radii=0.01,
                    ),
                )

            if mode in {"merged", "both"} and merged_P:
                if merged_P[i].size:
                    rr.log(
                        "world/frame/points_merged",
                        rr.Points3D(
                            positions=merged_P[i].astype(np.float32),
                            colors=merged_C[i].astype(np.uint8),
                            radii=0.01,
                        ),
                    )

        print("[viz] Done streaming frames to Rerun.")

        # Cleanup
        del predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------- Batch over a split ---------
    def infer_all_videos(
        self,
        split: str,
        *,
        mode: str = "both",
        allowlist: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Process all videos in a split (optionally restricted by an allowlist)."""
        video_id_list = sorted(os.listdir(self.root_dir_path))

        # Filter by naming convention and split
        filtered: List[str] = []
        for vid in video_id_list:
            if allowlist is not None and vid not in allowlist:
                continue
            if get_video_belongs_to_split(vid) == split:
                filtered.append(vid)

        if not filtered:
            print(f"[warn] No videos matching split={split} under {self.root_dir_path}")
            return

        for video_id in tqdm(filtered, desc=f"Split {split}"):
            try:
                self.infer_video(video_id, mode=mode, **kwargs)
            except Exception as e:
                print(f"[ERROR] {video_id}: {e}")


# ---------------------------
# CLI
# ---------------------------

def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize static + per-frame 3D points with Rerun (AG-Pi3 unified)."
    )

    # Paths
    parser.add_argument(
        "--root_dir_path",
        type=str,
        default="/data/rohith/ag/frames",
        help="Path whose entries include the video IDs to process.",
    )
    parser.add_argument(
        "--frames_annotated_dir_path",
        type=str,
        default="/data/rohith/ag/frames_annotated",
        help="Optional: directory containing annotated frames (unused here).",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/static_scenes/pi3_full",
        help="Path to output directory where predictions folders live (e.g., <video>_10/).",
    )
    parser.add_argument(
        "--static_scene_dir_path",
        type=str,
        default="/data2/rohith/ag/ag4D/static_scenes/pi3",
        help="Path to output directory where predictions folders live (e.g., <video>_10/).",
    )

    # Selection
    parser.add_argument(
        "--split",
        type=_parse_split,
        default="04",
        help="Shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}.",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default=None,
        help="If set, run only this video ID (ignores --split filter).",
    )

    # Viz options
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["raw", "merged", "both"],
        help="What to log per frame: raw points, merged (static+frame), or both.",
    )
    parser.add_argument(
        "--conf_static",
        type=float,
        default=0.10,
        help="Confidence threshold for building static background (0..1).",
    )
    parser.add_argument(
        "--conf_frame",
        type=float,
        default=0.01,
        help="Confidence threshold for per-frame points (0..1).",
    )
    parser.add_argument(
        "--dedup_voxel",
        type=float,
        default=0.02,
        help="Voxel size (m) for de-duplication in merged clouds; set <=0 to disable.",
    )
    parser.add_argument(
        "--fov_y",
        type=float,
        default=0.96,
        help="Vertical field-of-view (radians) for Pinhole used in camera logging.",
    )
    parser.add_argument(
        "--no_spawn",
        action="store_true",
        help="Do not spawn the external Rerun viewer (use in-process).",
    )
    parser.add_argument(
        "--no_cam",
        action="store_true",
        help="Disable camera/frustum logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dedup = None if (args.dedup_voxel is not None and args.dedup_voxel <= 0) else args.dedup_voxel

    ag_pi3 = AgPi3(
        root_dir_path=args.root_dir_path,
        output_dir_path=args.output_dir_path,
        frame_annotated_dir_path=args.frames_annotated_dir_path,
    )

    if args.video_id:
        ag_pi3.infer_video(
            args.video_id,
            mode=args.mode,
            conf_static=args.conf_static,
            conf_frame=args.conf_frame,
            dedup_voxel=dedup,
            fov_y=args.fov_y,
            spawn=not args.no_spawn,
            log_cameras=not args.no_cam,
        )
    else:
        ag_pi3.infer_all_videos(
            split=args.split,
            mode=args.mode,
            conf_static=args.conf_static,
            conf_frame=args.conf_frame,
            dedup_voxel=dedup,
            fov_y=args.fov_y,
            spawn=not args.no_spawn,
            log_cameras=not args.no_cam,
        )


if __name__ == "__main__":
    main()
