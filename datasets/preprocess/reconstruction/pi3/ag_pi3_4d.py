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
from scipy.spatial import cKDTree


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
    # static_points = static_points @ align_R.T
    return scene_3d, static_points, static_colors


def merge_static_with_frame(
        predictions: Dict,
        static_points: np.ndarray,
        static_colors: np.ndarray,
        frame_idx: int,
        conf_min: float = 0.1,
        dedup_voxel: Optional[float] = 0.02,
        *,
        # ICP knobs (safe defaults)
        icp_max_iters: int = 30,
        icp_tol: float = 1e-5,
        trim_frac: float = 0.8,  # keep this fraction of closest pairs each iter
        max_corr_dist: Optional[float] = None,  # meters; None disables
        src_sample_max: int = 20000,  # subsample source for speed
        tgt_sample_max: int = 100000,  # subsample target for speed
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align per-frame points to the static background via ICP and return the merged cloud.

    Returns:
        mean_xyz (N,3) float32, mean_rgb (N,3) uint8
    """
    # ---------- Helpers ----------
    def _weighted_rigid_fit(A: np.ndarray, B: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """Find R,t minimizing sum_i w_i || R A_i + t - B_i ||^2 using Kabsch with optional weights."""
        if w is None:
            w = np.ones((A.shape[0],), dtype=np.float64)
        w = w.astype(np.float64)
        w_sum = np.sum(w) + 1e-12
        mu_A = (A * w[:, None]).sum(axis=0) / w_sum
        mu_B = (B * w[:, None]).sum(axis=0) / w_sum
        AA = A - mu_A
        BB = B - mu_B
        H = (AA * w[:, None]).T @ BB  # 3x3
        U, S, Vt = np.linalg.svd(H, full_matrices=True)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_B - R @ mu_A
        return R.astype(np.float64), t.astype(np.float64)

    def _apply(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        return (R @ X.T).T + t[None, :]

    # ---------- Prepare frame data ----------
    images = _ensure_nhwc(predictions["images"])  # (S,H,W,3)
    points = predictions["points"]  # (S,H,W,3)
    conf = predictions.get("conf", np.ones(points.shape[:-1], dtype=np.float32))

    pts_f = points[frame_idx]  # (H,W,3)
    img_f = images[frame_idx]  # (H,W,3)
    conf_f = conf[frame_idx]  # (H,W) or (H,W,1)

    pts_flat = pts_f.reshape(-1, 3).astype(np.float64)
    col_flat = (img_f.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_flat = conf_f[..., 0] if (conf_f.ndim == 3 and conf_f.shape[-1] == 1) else conf_f.reshape(-1)
    conf_flat = conf_flat.astype(np.float64)

    good = np.isfinite(pts_flat).all(axis=1) & (conf_flat >= conf_min) & (conf_flat > 1e-5)
    src_full = pts_flat[good]  # dynamic frame points (source)
    col_full = col_flat[good]
    w_full = conf_flat[good]

    if src_full.shape[0] == 0 or static_points.shape[0] == 0:
        # Nothing to register; return static only or simple concat
        if src_full.shape[0] == 0:
            merged_P = static_points
            merged_C = static_colors
        else:
            merged_P = np.concatenate([static_points, src_full.astype(np.float32)], axis=0)
            merged_C = np.concatenate([static_colors, col_full], axis=0)
        if dedup_voxel is None or merged_P.size == 0:
            return merged_P.astype(np.float32), merged_C.astype(np.uint8)
        # voxel average (same as below)
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

    # ---------- Subsample for speed (registration only; we still transform full set) ----------
    rng = np.random.default_rng(1337 + frame_idx)
    if src_full.shape[0] > src_sample_max:
        idx = rng.choice(src_full.shape[0], size=src_sample_max, replace=False)
        src = src_full[idx]
        w_src = w_full[idx]
    else:
        src = src_full
        w_src = w_full

    if static_points.shape[0] > tgt_sample_max:
        jdx = rng.choice(static_points.shape[0], size=tgt_sample_max, replace=False)
        tgt = static_points[jdx].astype(np.float64)
    else:
        tgt = static_points.astype(np.float64)

    # ---------- ICP loop ----------
    tree = cKDTree(tgt)
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    prev_err = np.inf

    src_iter = src.copy()

    for it in range(max(1, icp_max_iters)):
        # nearest neighbors in target
        dists, nn = tree.query(src_iter, k=1, workers=-1)
        valid = np.isfinite(dists)

        if max_corr_dist is not None:
            valid &= (dists <= max_corr_dist)

        if not np.any(valid):
            break

        # trim the worst matches
        if trim_frac < 1.0:
            cutoff = np.percentile(dists[valid], trim_frac * 100.0)
            valid &= (dists <= cutoff)

        A = src_iter[valid]  # current transformed source (subset)
        B = tgt[nn[valid]]  # matched target
        w = w_src[valid]
        if A.shape[0] < 10:
            break

        R_inc, t_inc = _weighted_rigid_fit(A, B, w)

        # compose transforms: new_total = R_inc * old_total, t_inc + R_inc * t_total
        R_total = R_inc @ R_total
        t_total = R_inc @ t_total + t_inc

        src_iter = _apply(R_inc, t_inc, src_iter)

        err = float(np.mean((A - B) ** 2))
        if abs(prev_err - err) < icp_tol:
            break
        prev_err = err

    # ---------- Apply final transform to full frame points ----------
    src_full_aligned = _apply(R_total, t_total, src_full)

    # ---------- Merge with static and optional voxel de-dup ----------
    merged_P = np.concatenate([static_points.astype(np.float64), src_full_aligned], axis=0)
    merged_C = np.concatenate([static_colors, col_full], axis=0)

    if dedup_voxel is None or merged_P.size == 0:
        return merged_P.astype(np.float32), merged_C.astype(np.uint8)

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

# Helper to build a wireframe frustum in the *camera's local frame* (RUB; camera looks along -Z).
def _make_frustum_lines(W, H, fx, fy, cx, cy, near=0.12, far=0.45):
    def rect_at(depth):
        z = -float(depth)  # forward along -Z in R-U-B
        x0 = (0 - cx) * (z / fx)
        x1 = (W - 1 - cx) * (z / fx)
        y0 = (0 - cy) * (z / fy)
        y1 = (H - 1 - cy) * (z / fy)
        return np.array([[x0, y0, z],
                         [x1, y0, z],
                         [x1, y1, z],
                         [x0, y1, z]], dtype=np.float32)

    n = rect_at(near)
    f = rect_at(far)
    strips = [
        np.vstack([n, n[0]]),  # near loop
        np.vstack([f, f[0]]),  # far loop
        np.vstack([n[0], f[0]]),  # connect near/far
        np.vstack([n[1], f[1]]),
        np.vstack([n[2], f[2]]),
        np.vstack([n[3], f[3]]),
        np.vstack([np.zeros(3, np.float32), n[0]]),  # rays from center
        np.vstack([np.zeros(3, np.float32), n[1]]),
        np.vstack([np.zeros(3, np.float32), n[2]]),
        np.vstack([np.zeros(3, np.float32), n[3]]),
    ]
    return strips


def _log_cameras(
        predictions: Dict,
        fov_y: float,
        W: int,
        H: int,
        type: str,
        color
) -> None:
    if "camera_poses" not in predictions:
        return
    cam_poses = predictions["camera_poses"]  # (S, 4, 4) or (S, 3, 4)
    fx, fy, cx, cy = _pinhole_from_fov(W, H, fov_y)

    for i, Tcw in enumerate(cam_poses):
        Rcw, tcw = _camera_R_t_from_4x4(Tcw)
        q_xyzw = Rot.from_matrix(Rcw).as_quat().astype(np.float32)  # [x, y, z, w]

        frus_path = _frustum_path(i)
        rr.log(
            frus_path,
            rr.Transform3D(
                translation=tcw.astype(np.float32),
                rotation=rr.Quaternion(xyzw=q_xyzw),
            ),
        )

        frustum_strips = _make_frustum_lines(W, H, fx, fy, cx, cy, near=0.12, far=0.45)

        pred_path = f"world/{type}/frames/t{i}/predicted/frustum"
        # Use same intrinsics for visualization; swap to predicted intrinsics if you have them.
        rr.log(
            f"{pred_path}/camera",
            rr.Pinhole(focal_length=(fx, fy), principal_point=(cx, cy), resolution=(W, H)),
        )
        # predicted frustum wire + center dot (ORANGE)
        rr.log(
            f"{pred_path}/frustum_wire",
            rr.LineStrips3D(
                frustum_strips,  # <-- positional instead of line_strips=
                colors=[color] * len(frustum_strips),
                radii=0.003,
            ),
        )
        rr.log(
            f"{pred_path}/center",
            rr.Points3D(
                positions=np.zeros((1, 3), dtype=np.float32),
                colors=color[None, :],
                radii=0.01,
            ),
        )


# ---------------------------
# Main class
# ---------------------------

class AgPi3:
    def __init__(
            self,
            root_dir_path: str,
            dynamic_scene_dir_path: Optional[str] = None,
            static_scene_dir_path: Optional[str] = None,  # accepted for parity; not used here
            frame_annotated_dir_path: Optional[str] = None,  # accepted for parity; not used here
    ):
        self.model = None
        self.root_dir_path = root_dir_path
        self.static_scene_dir_path = static_scene_dir_path
        self.dynamic_scene_dir_path = dynamic_scene_dir_path if dynamic_scene_dir_path is not None else root_dir_path
        self.frame_annotated_dir_path = frame_annotated_dir_path
        os.makedirs(self.dynamic_scene_dir_path, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------- Infer basic video visualization -----------------------------
    def infer_basic_video(
            self,
            video_id: str,
            *,
            mode: str = "both",  # one of {"raw", "merged", "both"}
            conf_static: float = 0.10,  # confidence for static background build
            conf_frame: float = 0.01,  # confidence for per-frame points
            dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
            fov_y: float = 0.96,  # radians; matches your earlier default
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
        static_scene_pred_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(self.dynamic_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        if not os.path.exists(static_scene_pred_path) or not os.path.exists(dynamic_scene_pred_path):
            raise FileNotFoundError(f"predictions.npz not found for {video_id}")

        static_scene_arr = np.load(static_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_arr = np.load(dynamic_scene_pred_path, allow_pickle=True, mmap_mode=None)

        static_scene_predictions = {k: static_scene_arr[k] for k in static_scene_arr.files}
        dynamic_scene_predictions = {k: dynamic_scene_arr[k] for k in dynamic_scene_arr.files}

        static_scene_points_wh = static_scene_predictions["points"]  # (S,H,W,3)
        dynamic_scene_points_wh = dynamic_scene_predictions["points"]  # (S,H,W,3)

        static_scene_images = _ensure_nhwc(static_scene_predictions["images"])  # (S,H,W,3) in [0,1]
        dynamic_scene_images = _ensure_nhwc(dynamic_scene_predictions["images"])  # (S,H,W,3) in [0,1]

        static_scene_conf_wh = static_scene_predictions.get("conf")  # (S,H,W) or (S,H,W,1)
        dynamic_scene_conf_wh = dynamic_scene_predictions.get("conf")  # (S,H,W) or (S,H,W,1)

        S, H, W = static_scene_points_wh.shape[:3]
        print(
            f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame} | mode={mode}")

        # ---- Build static background (once) ----
        scene_3d, static_P, static_C = predictions_to_glb_with_static(
            static_scene_predictions, conf_min=float(conf_static)
        )

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
                )
            )

        # Cameras & frustums (timeless transforms, separate camera nodes per frame)
        if log_cameras:
            _fx, _fy, _cx, _cy = _pinhole_from_fov(W, H, fov_y)
            _log_cameras(static_scene_predictions, fov_y=fov_y, W=W, H=H, type="static",
                         color=[255, 0, 0])  # RED for static
            _log_cameras(dynamic_scene_predictions, fov_y=fov_y, W=W, H=H, type="dynamic",
                         color=[0, 255, 0])  # GREEN for dynamic

        # ---- Precompute per-frame MERGED with static ----
        merged_P: List[np.ndarray] = []
        merged_C: List[np.ndarray] = []
        for i in tqdm(range(S), desc=f"[viz] Merging frames for {video_id}"):
            Pi, Ci = merge_static_with_frame(
                dynamic_scene_predictions,
                static_P, static_C,
                frame_idx=i,
                conf_min=float(conf_frame),
                dedup_voxel=dedup_voxel,
            )
            merged_P.append(Pi)
            merged_C.append(Ci)

            rr.set_time_sequence("frame", i)

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
        del static_scene_predictions
        del dynamic_scene_predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------- Unified pipeline ---------
    def infer_video(
            self,
            video_id: str,
            *,
            mode: str = "both",  # one of {"raw", "merged", "both"}
            conf_static: float = 0.10,  # confidence for static background build
            conf_frame: float = 0.01,  # confidence for per-frame points
            dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
            fov_y: float = 0.96,  # radians; matches your earlier default
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
        static_scene_pred_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(self.dynamic_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        if not os.path.exists(static_scene_pred_path) or not os.path.exists(dynamic_scene_pred_path):
            raise FileNotFoundError(f"predictions.npz not found for {video_id}")

        static_scene_arr = np.load(static_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_arr = np.load(dynamic_scene_pred_path, allow_pickle=True, mmap_mode=None)

        static_scene_predictions = {k: static_scene_arr[k] for k in static_scene_arr.files}
        dynamic_scene_predictions = {k: dynamic_scene_arr[k] for k in dynamic_scene_arr.files}

        static_scene_points_wh = static_scene_predictions["points"]  # (S,H,W,3)
        dynamic_scene_points_wh = dynamic_scene_predictions["points"]  # (S,H,W,3)

        static_scene_images = _ensure_nhwc(static_scene_predictions["images"])  # (S,H,W,3) in [0,1]
        dynamic_scene_images = _ensure_nhwc(dynamic_scene_predictions["images"])  # (S,H,W,3) in [0,1]

        static_scene_conf_wh = static_scene_predictions.get("conf")  # (S,H,W) or (S,H,W,1)
        dynamic_scene_conf_wh = dynamic_scene_predictions.get("conf")  # (S,H,W) or (S,H,W,1)

        S, H, W = static_scene_points_wh.shape[:3]
        print(
            f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame} | mode={mode}")

        # ---- Build static background (once) ----
        scene_3d, static_P, static_C = predictions_to_glb_with_static(
            static_scene_predictions, conf_min=float(conf_static)
        )

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
                )
            )

        # Cameras & frustums (timeless transforms, separate camera nodes per frame)
        if log_cameras:
            _fx, _fy, _cx, _cy = _pinhole_from_fov(W, H, fov_y)
            _log_cameras(static_scene_predictions, fov_y=fov_y, W=W, H=H)

        # ---- Precompute per-frame MERGED with static ----
        merged_P: List[np.ndarray] = []
        merged_C: List[np.ndarray] = []
        for i in tqdm(range(S), desc=f"[viz] Merging frames for {video_id}"):
            Pi, Ci = merge_static_with_frame(
                dynamic_scene_predictions,
                static_P, static_C,
                frame_idx=i,
                conf_min=float(conf_frame),
                dedup_voxel=dedup_voxel,
            )
            merged_P.append(Pi)
            merged_C.append(Ci)

            rr.set_time_sequence("frame", i)

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
        del static_scene_predictions
        del dynamic_scene_predictions
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

        video_id_list = ["0DJ6R.mp4"]

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
            self.infer_video(video_id, mode=mode, **kwargs)


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
        default="/data/rohith/ag/ag4D/static_scenes/pi3",
        help="Path to output directory where predictions folders live (e.g., <video>_10/).",
    )
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data/rohith/ag/ag4D/static_scenes/pi3_full"
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
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        static_scene_dir_path=args.static_scene_dir_path,
        frame_annotated_dir_path=args.frames_annotated_dir_path,
    )

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
