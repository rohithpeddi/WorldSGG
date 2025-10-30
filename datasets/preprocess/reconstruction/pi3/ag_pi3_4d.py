import argparse
import gc
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from typing import List

import numpy as np
import rerun as rr
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
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

def predictions_to_colors(
        predictions: Dict,
        conf_min: float = 0.5,                  # 0..1 threshold on predictions["conf"]
        filter_by_frames: str = "all",          # e.g. "12:..." to use only frame index 12
        filter_low_conf_black: bool = False,    # enable extra filtering
        *,
        black_rgb_max: int = 8,                 # 0..255; <= this per-channel is considered "black"
        black_conf_max: float = 1.0,           # 0..1; if conf < this AND pixel is black -> drop
):
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # ------------ Frame selection ------------
    selected_frame_idx = None
    if filter_by_frames not in ("all", "All"):
        try:
            selected_frame_idx = int(str(filter_by_frames).split(":")[0])
        except (ValueError, IndexError):
            selected_frame_idx = None

    print("Selected frame index for extraction:", selected_frame_idx)

    pts = predictions["points"]                             # (S,H,W,3)
    conf = predictions.get("conf", np.ones_like(pts[..., 0], dtype=np.float32))  # (S,H,W) or (S,H,W,1)
    imgs = predictions["images"]                            # (S,?,H,W) or (S,H,W,3/4/1)
    cam_poses = predictions.get("camera_poses", None)

    if selected_frame_idx is not None:
        pts = pts[selected_frame_idx][None]
        conf = conf[selected_frame_idx][None]
        imgs = imgs[selected_frame_idx][None]
        cam_poses = cam_poses[selected_frame_idx][None] if cam_poses is not None else None

    # ------------ Color layout handling ------------
    # Normalize to NHWC; keep only the first 3 channels (RGB); expand grayscale to RGB.
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3, 4):            # NCHW
        imgs_nhwc = np.transpose(imgs, (0, 2, 3, 1))
    elif imgs.ndim == 4 and imgs.shape[-1] in (1, 3, 4):         # already NHWC
        imgs_nhwc = imgs
    else:
        raise ValueError(f"`images` must be 4D with channels in {{1,3,4}}, got shape {imgs.shape}")

    C = imgs_nhwc.shape[-1]
    if C == 1:
        imgs_nhwc = np.repeat(imgs_nhwc, 3, axis=-1)
    elif C >= 3:
        imgs_nhwc = imgs_nhwc[..., :3]  # drop alpha if present

    # Convert to uint8 RGB for color output
    colors_rgb = (imgs_nhwc.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

    # ------------ Confidence filtering ------------
    verts = pts.reshape(-1, 3)
    conf_flat = conf.reshape(-1).astype(np.float32)

    thr = float(conf_min) if conf_min is not None else 0.1
    base_mask = (conf_flat >= thr) & (conf_flat > 1e-5)

    # Optional: drop pixels that are BOTH near-black AND have low confidence (below black_conf_max)
    if filter_low_conf_black:
        # Per-channel black check (inclusive): 0..black_rgb_max
        is_black = (
            (colors_rgb[:, 0] <= black_rgb_max) &
            (colors_rgb[:, 1] <= black_rgb_max) &
            (colors_rgb[:, 2] <= black_rgb_max)
        )
        low_conf_black = is_black & (conf_flat < float(black_conf_max))
        mask = base_mask & (~low_conf_black)
    else:
        mask = base_mask

    # Apply mask
    verts = verts[mask]
    colors_rgb = colors_rgb[mask]

    if verts.size == 0:
        # robust fallback
        verts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        colors_rgb = np.array([[255, 255, 255]], dtype=np.uint8)

    # Keep an ORIGINAL copy for the background return (world frame)
    static_points = verts.astype(np.float32, copy=True)
    static_colors = colors_rgb.astype(np.uint8, copy=True)

    return static_points, static_colors, verts, colors_rgb


def glb_to_points(glb_scene_path:  str) -> Tuple[np.ndarray, np.ndarray]:
    scene = trimesh.load(glb_scene_path, force='scene')
    all_points = []
    all_colors = []

    for geom_name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.points.PointCloud):
            all_points.append(geom.vertices.astype(np.float32))
            if geom.colors is not None:
                all_colors.append(geom.colors.astype(np.uint8))
            else:
                # Default to white if no colors
                all_colors.append(np.ones((geom.vertices.shape[0], 3), dtype=np.uint8) * 255)

    if len(all_points) == 0:
        raise ValueError(f"No point clouds found in GLB file: {glb_scene_path}")

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)

    # Ensure colors are uint8 and in the correct range and points in float32
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    points = points.astype(np.float32)

    align_R = Rot.from_euler("y", 100, degrees=True).as_matrix()
    align_R = align_R @ Rot.from_euler("x", 155, degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = align_R

    points = (points @ align_R.T).astype(np.float32)

    return points, colors


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

    print("Estimating static scene with a confidence mask threshold of:", conf_min)

    static_points, static_colors, verts, colors_rgb = predictions_to_colors(
        predictions, conf_min=conf_min, filter_by_frames=filter_by_frames, filter_low_conf_black=True
    )

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

def _conf_to_hw(conf_f: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Normalize a per-frame confidence array to shape (H, W), handling common variants:
      - (H, W)
      - (H, W, 1)
      - (H,)  -> repeat across W
      - (W,)  -> repeat across H
      - (H*W,) -> reshape to (H, W)
    Falls back to ones(H,W) if shape is unexpected.
    """
    c = np.asarray(conf_f)
    if c.ndim == 2 and c.shape == (H, W):
        return c
    if c.ndim == 3 and c.shape[2] == 1 and c.shape[:2] == (H, W):
        return c[..., 0]
    if c.ndim == 1:
        if c.shape[0] == H * W:
            return c.reshape(H, W)
        if c.shape[0] == H:
            return np.repeat(c[:, None], W, axis=1)
        if c.shape[0] == W:
            return np.repeat(c[None, :], H, axis=0)
    # Fallback: uniform confidence
    return np.ones((H, W), dtype=np.float64)


def ground_dynamic_scene_to_static_scene(
        predictions: Dict,
        static_points: np.ndarray,
        static_colors: np.ndarray,  # kept for API parity; unused here
        frame_idx: int,
        conf_min: float = 0.1,
        dedup_voxel: Optional[float] = 0.02,  # used in fallback paths only
        *,
        # ICP knobs (safe defaults)
        icp_max_iters: int = 100,
        icp_tol: float = 1e-5,
        trim_frac: float = 0.8,  # keep this fraction of closest pairs each iter
        max_corr_dist: Optional[float] = None,  # meters; None disables
        src_sample_max: int = 50000,  # NOTE: ignored for ICP (we use full dynamic), kept for API compat
        tgt_sample_max: int = 100000,  # we may still subsample static for speed
        dynamic_voxel: Optional[float] = 0.01,
        merge_voxel: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register the dynamic points of a single frame to the static background via ICP,
    then return ONLY the grounded dynamic cloud (discard the static).

    Returns:
        dyn_P (N,3) float32, dyn_C (N,3) uint8
    """
    # ---------- Helpers ----------
    def _weighted_rigid_fit(A: np.ndarray, B: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted Kabsch: find R,t minimizing sum_i w_i || R A_i + t - B_i ||^2."""
        if w is None:
            w = np.ones((A.shape[0],), dtype=np.float64)
        w = w.astype(np.float64)
        w_sum = np.sum(w) + 1e-12
        mu_A = (A * w[:, None]).sum(axis=0) / w_sum
        mu_B = (B * w[:, None]).sum(axis=0) / w_sum
        AA = A - mu_A
        BB = B - mu_B
        H = (AA * w[:, None]).T @ BB
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
    images = _ensure_nhwc(predictions["images"])   # (S,H,W,3)
    points = predictions["points"]                 # (S,H,W,3)
    conf    = predictions.get("conf", None)

    pts_f = points[frame_idx]  # (H,W,3)
    img_f = images[frame_idx]  # (H,W,3)
    H, W = pts_f.shape[:2]

    # Normalize confidence to (H,W)
    if conf is None:
        conf_hw = np.ones((H, W), dtype=np.float64)
        orig_conf_shape = None
    else:
        conf_f = conf[frame_idx]
        conf_hw = _conf_to_hw(conf_f, H, W).astype(np.float64)
        orig_conf_shape = getattr(conf_f, "shape", None)

    # Clean confidences
    conf_hw = np.nan_to_num(conf_hw, nan=0.0, posinf=0.0, neginf=0.0)
    conf_hw = np.maximum(conf_hw, 0.0)

    # Flatten
    pts_flat = pts_f.reshape(-1, 3).astype(np.float64)                    # (H*W,3)
    col_flat = (img_f.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_vec = conf_hw.reshape(-1)                                        # (H*W,)

    # Sanity check
    if conf_vec.shape[0] != pts_flat.shape[0]:
        raise ValueError(
            f"conf has {conf_vec.shape[0]} elems but points have {pts_flat.shape[0]} rows. "
            f"Original conf frame shape: {orig_conf_shape} normalized to {(H, W)}"
        )

    # Base validity (finite + confidence)
    good = np.isfinite(pts_flat).all(axis=1) & (conf_vec >= conf_min) & (conf_vec > 1e-5)

    # Dynamic sets (full frame)
    src_full = pts_flat[good]
    col_full = col_flat[good]
    w_full   = conf_vec[good].astype(np.float64)

    # Early outs
    if src_full.shape[0] == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    # If no static target, we cannot register; just (optionally) voxel-reduce and return the raw dynamic
    if static_points is None or static_points.shape[0] == 0:
        dyn_P = src_full.copy()
        dyn_C = col_full.copy()
        # Prefer dynamic_voxel; if None but dedup_voxel is set, use it as fallback
        vox_sz = dynamic_voxel if dynamic_voxel is not None else dedup_voxel
        if vox_sz is not None and dyn_P.size:
            vox = _voxel_ids(dyn_P, vox_sz)
            vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
            uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
            G = uniq_keys.shape[0]
            counts = np.bincount(inv, minlength=G).astype(np.float32)
            sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
            sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
            sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
            sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
            sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
            sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
            counts[counts == 0] = 1.0
            dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
            dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
        if merge_voxel is not None and dyn_P.size:
            vox = _voxel_ids(dyn_P, merge_voxel)
            vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
            uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
            G = uniq_keys.shape[0]
            counts = np.bincount(inv, minlength=G).astype(np.float32)
            sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
            sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
            sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
            sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
            sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
            sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
            counts[counts == 0] = 1.0
            dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
            dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
        return dyn_P.astype(np.float32), dyn_C.astype(np.uint8)

    # ---------- Build static target (optionally subsampled for speed) ----------
    if static_points.shape[0] > tgt_sample_max:
        rng = np.random.default_rng(1337 + frame_idx)
        jdx = rng.choice(static_points.shape[0], size=tgt_sample_max, replace=False)
        tgt = static_points[jdx].astype(np.float64)
    else:
        tgt = static_points.astype(np.float64)

    # ---------- ICP using the FULL dynamic cloud ----------
    tree = cKDTree(tgt)
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    prev_err = np.inf
    src_iter = src_full.copy()

    for it in range(max(1, icp_max_iters)):
        dists, nn = tree.query(src_iter, k=1, workers=-1)
        valid = np.isfinite(dists)
        if max_corr_dist is not None:
            valid &= (dists <= max_corr_dist)
        if not np.any(valid):
            break

        if trim_frac < 1.0:
            cutoff = np.percentile(dists[valid], trim_frac * 100.0)
            valid &= (dists <= cutoff)

        A = src_iter[valid]
        B = tgt[nn[valid]]
        w = w_full[valid]
        if A.shape[0] < 10:
            break

        R_inc, t_inc = _weighted_rigid_fit(A, B, w)

        # compose transforms
        R_total = R_inc @ R_total
        t_total = R_inc @ t_total + t_inc
        src_iter = _apply(R_inc, t_inc, src_iter)

        err = float(np.mean((A - B) ** 2))
        if abs(prev_err - err) < icp_tol:
            break
        prev_err = err

    # ---------- Apply final transform to ALL dynamic points of this frame ----------
    dyn_P = _apply(R_total, t_total, src_full).astype(np.float64)
    dyn_C = col_full

    # ---------- Optional voxel reductions on the grounded dynamic cloud ----------
    if dynamic_voxel is not None and dyn_P.size:
        vox = _voxel_ids(dyn_P, dynamic_voxel)
        vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
        uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
        G = uniq_keys.shape[0]
        counts = np.bincount(inv, minlength=G).astype(np.float32)
        sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
        sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
        sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
        sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
        sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
        sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
        counts[counts == 0] = 1.0
        dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
        dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    if merge_voxel is not None and dyn_P.size:
        vox = _voxel_ids(dyn_P, merge_voxel)
        vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
        uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
        G = uniq_keys.shape[0]
        counts = np.bincount(inv, minlength=G).astype(np.float32)
        sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
        sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
        sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
        sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
        sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
        sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
        counts[counts == 0] = 1.0
        dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
        dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    return dyn_P.astype(np.float32), dyn_C.astype(np.uint8)



def merge_static_with_frame(
        predictions: Dict,
        static_points: np.ndarray,
        static_colors: np.ndarray,
        interaction_masks: np.ndarray,
        frame_idx: int,
        conf_min: float = 0.1,
        dedup_voxel: Optional[float] = 0.02,
        *,
        # ICP knobs (safe defaults)
        icp_max_iters: int = 100,
        icp_tol: float = 1e-5,
        trim_frac: float = 0.8,  # keep this fraction of closest pairs each iter
        max_corr_dist: Optional[float] = None,  # meters; None disables
        src_sample_max: int = 50000,  # NOTE: ignored for ICP (we use full dynamic), kept for API compat
        tgt_sample_max: int = 100000,  # we may still subsample static for speed
        dynamic_voxel: Optional[float] = 0.01,
        merge_voxel: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align per-frame points to the static background via ICP and return the merged cloud.

    Changes vs previous:
      (1) ICP uses the full dynamic frame (no source subsampling).
      (2) Only dynamic points within the frame interaction mask are merged with static.
    Returns:
        mean_xyz (N,3) float32, mean_rgb (N,3) uint8
    """
    # ---------- Helpers ----------
    def _weighted_rigid_fit(A: np.ndarray, B: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find R,t minimizing sum_i w_i || R A_i + t - B_i ||^2 via weighted Kabsch."""
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
    points = predictions["points"]                # (S,H,W,3)
    conf    = predictions.get("conf", None)

    pts_f = points[frame_idx]  # (H,W,3)
    img_f = images[frame_idx]  # (H,W,3)
    mask_f = interaction_masks[frame_idx]  # (H,W) boolean/0-1

    H, W = pts_f.shape[:2]

    # Normalize confidence to (H,W)
    if conf is None:
        conf_hw = np.ones((H, W), dtype=np.float64)
    else:
        conf_f = conf[frame_idx]
        conf_hw = _conf_to_hw(conf_f, H, W).astype(np.float64)

    # Clean confidences
    conf_hw = np.nan_to_num(conf_hw, nan=0.0, posinf=0.0, neginf=0.0)
    conf_hw = np.maximum(conf_hw, 0.0)

    # Flatten
    pts_flat  = pts_f.reshape(-1, 3).astype(np.float64)                # (H*W,3)
    col_flat  = (img_f.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_vec  = conf_hw.reshape(-1)                                    # (H*W,)
    mask_vec  = (mask_f.reshape(-1) > 0)                               # (H*W,) bool

    # Sanity check
    if conf_vec.shape[0] != pts_flat.shape[0]:
        raise ValueError(
            f"conf has {conf_vec.shape[0]} elems but points have {pts_flat.shape[0]} rows. "
            f"Original conf frame shape: {getattr(conf_f, 'shape', None)} normalized to {(H,W)}"
        )
    if mask_vec.shape[0] != pts_flat.shape[0]:
        raise ValueError(f"interaction mask shape {mask_f.shape} incompatible with points frame {(H,W)}")

    # Base validity (finite + confidence)
    good_all = np.isfinite(pts_flat).all(axis=1) & (conf_vec >= conf_min) & (conf_vec > 1e-5)

    # Split into: full dynamic for ICP vs masked dynamic for merging
    sel_all   = good_all
    sel_mask  = good_all & mask_vec   # merge only these later

    # Build dynamic sets
    src_full        = pts_flat[sel_all]        # used for ICP (full)
    col_full        = col_flat[sel_all]
    w_full          = conf_vec[sel_all].astype(np.float64)

    src_masked_raw  = pts_flat[sel_mask]       # used for MERGE (subset)
    col_masked_raw  = col_flat[sel_mask]

    # Early exits if empty
    if static_points.shape[0] == 0 and src_masked_raw.shape[0] == 0:
        return np.empty((0,3), np.float32), np.empty((0,3), np.uint8)
    if src_full.shape[0] == 0 or static_points.shape[0] == 0:
        # Nothing to register; just merge masked dynamic (if any) with static
        if src_masked_raw.shape[0] == 0 or static_points.shape[0] == 0:
            merged_P = static_points
            merged_C = static_colors
        else:
            dyn_P, dyn_C = src_masked_raw.astype(np.float64), col_masked_raw
            if dynamic_voxel is not None and dyn_P.size:
                vox = _voxel_ids(dyn_P, dynamic_voxel)
                vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
                uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
                G = uniq_keys.shape[0]
                counts = np.bincount(inv, minlength=G).astype(np.float32)
                sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
                sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
                sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
                sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
                sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
                sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
                counts[counts == 0] = 1.0
                dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
                dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
            merged_P = np.concatenate([static_points.astype(np.float64), dyn_P], axis=0)
            merged_C = np.concatenate([static_colors, dyn_C], axis=0)

        # Optional final global voxel
        if dedup_voxel is not None and merged_P.size:
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
            merged_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
            merged_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
        if merge_voxel is not None and merged_P.size:
            vox = _voxel_ids(merged_P, merge_voxel)
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
            merged_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
            merged_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
        return merged_P.astype(np.float32), merged_C.astype(np.uint8)

    # ---------- Build static target (optionally subsampled for speed) ----------
    if static_points.shape[0] > tgt_sample_max:
        rng = np.random.default_rng(1337 + frame_idx)
        jdx = rng.choice(static_points.shape[0], size=tgt_sample_max, replace=False)
        tgt = static_points[jdx].astype(np.float64)
    else:
        tgt = static_points.astype(np.float64)

    # ---------- ICP using the FULL dynamic cloud ----------
    tree = cKDTree(tgt)
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    prev_err = np.inf
    src_iter = src_full.copy()

    for it in range(max(1, icp_max_iters)):
        dists, nn = tree.query(src_iter, k=1, workers=-1)
        valid = np.isfinite(dists)
        if max_corr_dist is not None:
            valid &= (dists <= max_corr_dist)
        if not np.any(valid):
            break

        if trim_frac < 1.0:
            cutoff = np.percentile(dists[valid], trim_frac * 100.0)
            valid &= (dists <= cutoff)

        A = src_iter[valid]     # transformed source samples
        B = tgt[nn[valid]]      # matched static
        w = w_full[valid]
        if A.shape[0] < 10:
            break

        R_inc, t_inc = _weighted_rigid_fit(A, B, w)

        # compose transforms
        R_total = R_inc @ R_total
        t_total = R_inc @ t_total + t_inc
        src_iter = _apply(R_inc, t_inc, src_iter)

        err = float(np.mean((A - B) ** 2))
        if abs(prev_err - err) < icp_tol:
            break
        prev_err = err

    # ---------- Apply final transform to MASKED dynamic only (for merging) ----------
    dyn_P = _apply(R_total, t_total, src_masked_raw).astype(np.float64)
    dyn_C = col_masked_raw

    # Voxel-reduce the dynamic subset if requested
    if dynamic_voxel is not None and dyn_P.size:
        vox = _voxel_ids(dyn_P, dynamic_voxel)
        vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
        uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
        G = uniq_keys.shape[0]
        counts = np.bincount(inv, minlength=G).astype(np.float32)
        sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
        sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
        sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
        sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
        sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
        sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
        counts[counts == 0] = 1.0
        dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
        dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    # ---------- Merge with static (do NOT voxelize static by default) ----------
    if dyn_P.size:
        merged_P = np.concatenate([static_points.astype(np.float64), dyn_P], axis=0)
        merged_C = np.concatenate([static_colors,              dyn_C], axis=0)
    else:
        merged_P = static_points.astype(np.float64)
        merged_C = static_colors

    # Optional final global voxel (usually keep None to preserve static density)
    if merge_voxel is not None and merged_P.size:
        vox = _voxel_ids(merged_P, merge_voxel)
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
        merged_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
        merged_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    return merged_P.astype(np.float32), merged_C.astype(np.uint8)


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
        col = np.asarray(color, dtype=np.uint8).reshape(1, 3)  # (1,3), 0–255
        rr.log(
            f"{pred_path}/center",
            rr.Points3D(
                positions=np.zeros((1, 3), dtype=np.float32),
                colors=col,
                radii=0.01,  # or np.array([0.01], dtype=np.float32)
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
            masks_dir_path: Optional[str] = None,  # accepted for parity; not used here
    ):
        self.model = None
        self.root_dir_path = root_dir_path
        self.static_scene_dir_path = static_scene_dir_path
        self.dynamic_scene_dir_path = dynamic_scene_dir_path if dynamic_scene_dir_path is not None else root_dir_path
        self.frame_annotated_dir_path = frame_annotated_dir_path
        self.masks_dir_path = masks_dir_path

        os.makedirs(self.dynamic_scene_dir_path, exist_ok=True)

        self.sampled_frames_idx_root_dir = "/data/rohith/ag/sampled_frames_idx"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------- Infer basic video visualization -----------------------------

    def voxel_merge(self, static_P, static_C, dyn_P, dyn_C, voxel_size=None):
        """Concatenate static & dynamic. If voxel_size is set, dedup by quantizing."""
        if dyn_P is None or dyn_P.size == 0:
            return static_P, static_C

        P = np.concatenate([static_P, dyn_P], axis=0)
        C = np.concatenate([static_C, dyn_C], axis=0)

        if voxel_size is None:
            return P, C

        # Quantize to voxel grid and keep first occurrence per voxel.
        q = np.floor(P / float(voxel_size)).astype(np.int64)
        # Pack rows for fast unique:
        q_view = q.view([('', q.dtype)] * q.shape[1]).reshape(q.shape[0])
        uniq, idx = np.unique(q_view, return_index=True)
        idx = np.sort(idx)
        return P[idx], C[idx]

    def infer_basic_video(
            self,
            video_id: str,
            *,
            conf_static: float = 0.10,
            conf_frame: float = 0.01,
            dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
            fov_y: float = 0.96,
            spawn: bool = True,
            log_cameras: bool = True,
            apply_transform: bool = True
    ) -> None:
        # ---- Load predictions ----
        stem = video_id[:-4] if video_id.endswith(".mp4") else video_id
        static_scene_pred_path = os.path.join(self.static_scene_dir_path, f"{stem}_{10}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(self.dynamic_scene_dir_path, f"{stem}_{10}", "predictions.npz")
        if not (os.path.exists(static_scene_pred_path) and os.path.exists(dynamic_scene_pred_path)):
            raise FileNotFoundError(f"predictions.npz not found for {video_id}")

        static_npz = np.load(static_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_npz = np.load(dynamic_scene_pred_path, allow_pickle=True, mmap_mode=None)
        static_pred = {k: static_npz[k] for k in static_npz.files}
        dynamic_pred = {k: dynamic_npz[k] for k in dynamic_npz.files}

        static_points_wh = static_pred["points"]  # (S,H,W,3)
        S_static, H, W = static_points_wh.shape[:3]
        # Prefer the dynamic sequence length for the loop if present:
        S_dynamic = dynamic_pred.get("points", static_points_wh).shape[0]
        S = min(S_static, S_dynamic)

        print(f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame}")

        if apply_transform:
            align_R = Rot.from_euler("y", 100, degrees=True).as_matrix()
            align_R = align_R @ Rot.from_euler("x", 155, degrees=True).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = align_R

        # ---- Build static background (once) ----
        # Returns (scene_3d, static_P, static_C); we only need P,C here.
        _, static_P, static_C = predictions_to_glb_with_static(static_pred, conf_min=float(conf_static))
        static_P = static_P.astype(np.float32, copy=False)
        static_C = static_C.astype(np.uint8, copy=False)

        # ---- Rerun setup ----
        rr.init(f"AG-Pi3: {video_id}", spawn=spawn)
        rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

        # Log static background timelessly (persists across all frames)
        if static_P.size > 0:
            if apply_transform:
                static_P = static_P @ align_R.T
            rr.log(
                "world/static",
                rr.Points3D(
                    positions=static_P,
                    colors=static_C,
                ),
            )

        # Cameras (do this once; use np.array for color to avoid list-indexing errors)
        if log_cameras:
            _fx, _fy, _cx, _cy = _pinhole_from_fov(W, H, fov_y)
            _log_cameras(static_pred, fov_y=fov_y, W=W, H=H, type="static",
                         color=np.array([255, 0, 0], dtype=np.uint8))  # RED
            _log_cameras(dynamic_pred, fov_y=fov_y, W=W, H=H, type="dynamic",
                         color=np.array([0, 255, 0], dtype=np.uint8))  # GREEN

        # ---- Stream frames ----
        for i in tqdm(range(S), desc=f"[viz] Streaming frames for {video_id}"):
            # Dynamic points/colors for frame i
            dyn_P, dyn_C, _, _ = predictions_to_colors(
                dynamic_pred, conf_min=float(conf_frame), filter_by_frames=f"{i}:"
            )
            dyn_P = np.asarray(dyn_P, dtype=np.float32)
            dyn_C = np.asarray(dyn_C, dtype=np.uint8)

            if apply_transform:
                dyn_P = dyn_P @ align_R.T

            rr.set_time_sequence("frame", i)

            # Merged = static ⊕ dynamic (with optional voxel dedup)
            merged_P, merged_C = self.voxel_merge(static_P, static_C, dyn_P, dyn_C, voxel_size=dedup_voxel)
            rr.log(
                "world/frame/points_merged",
                rr.Points3D(
                    positions=merged_P.astype(np.float32, copy=False),
                    colors=merged_C.astype(np.uint8, copy=False),
                    radii=0.01,
                ),
            )

        print("[viz] Done streaming frames to Rerun.")

        # Cleanup
        del static_pred, dynamic_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------- Infer grounded dynamic video visualization -----------------------------

    def infer_grounded_dynamic_video(
            self,
            video_id: str,
            *,
            conf_static: float = 0.1,  # confidence for static background build
            conf_frame: float = 0.01,  # confidence for per-frame points
            dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
            fov_y: float = 0.96,  # radians; matches your earlier default
            spawn: bool = True,
            log_cameras: bool = True,
            load_from_glb: bool = False,
    ) -> None:
        static_scene_pred_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(self.dynamic_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        if not os.path.exists(static_scene_pred_path) or not os.path.exists(dynamic_scene_pred_path):
            raise FileNotFoundError(f"predictions.npz not found for {video_id}")

        static_scene_arr = np.load(static_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_arr = np.load(dynamic_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_predictions = {k: dynamic_scene_arr[k] for k in dynamic_scene_arr.files}
        S, H, W = dynamic_scene_predictions["points"].shape[:3]

        if load_from_glb:
            print(f"[viz] Loading static scene from GLB for {video_id}...")
            glb_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", f"{video_id[:-4]}.glb")
            static_P, static_C = glb_to_points(glb_path)
        else:
            print("Loading static scene from predictions...")
            static_scene_predictions = {k: static_scene_arr[k] for k in static_scene_arr.files}
            static_scene_points_wh = static_scene_predictions["points"]  # (S,H,W,3)
            S, H, W = static_scene_points_wh.shape[:3]
            print(f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame}")

            # ---- Build static background (once) ----
            scene_3d, static_P, static_C = predictions_to_glb_with_static(
                static_scene_predictions, conf_min=float(conf_static),
            )

        # ---- Rerun setup ----
        rr.init(f"AG-Pi3: {video_id}", spawn=spawn)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
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
            if not load_from_glb:
                _log_cameras(static_scene_predictions, fov_y=fov_y, W=W, H=H, type="static",
                             color=np.array([255, 0, 0], dtype=np.uint8))  # RED
            _log_cameras(dynamic_scene_predictions, fov_y=fov_y, W=W, H=H, type="dynamic",
                         color=np.array([0, 255, 0], dtype=np.uint8))  # GREEN

        # ---- Precompute per-frame MERGED with static ----
        grounded_P: List[np.ndarray] = []
        grounded_C: List[np.ndarray] = []
        for i in tqdm(range(S), desc=f"[viz] Merging frames for {video_id}"):
            Pi, Ci = ground_dynamic_scene_to_static_scene(
                dynamic_scene_predictions,
                static_P, static_C,
                frame_idx=i,
                conf_min=float(conf_frame),
                dedup_voxel=dedup_voxel,
            )
            grounded_P.append(Pi)
            grounded_C.append(Ci)

            rr.set_time_sequence("frame", i)

            if grounded_P[i].size:
                rr.log(
                    "world/frame/points_merged",
                    rr.Points3D(
                        positions=grounded_P[i].astype(np.float32),
                        colors=grounded_C[i].astype(np.uint8),
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

    # ----------------------------- Infer 4D reconstruction video -----------------------------

    def load_masks_for_video(
            self,
            video_id: str,
            target_hw: Optional[Tuple[int, int]] = None,  # (H_out, W_out); default: (H, W)
    ) -> np.ndarray:
        # --- Determine output spatial size ---
        H_out, W_out = int(target_hw[0]), int(target_hw[1])

        # --- Identify annotated frames and the sampled-frame mapping ---
        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)
        annotated_frame_id_list = [f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith('.png')]
        if not annotated_frame_id_list:
            raise FileNotFoundError(f"No annotated frames found in: {video_frames_annotated_dir_path}")
        # Ensure deterministic first frame
        annotated_first_frame_id = int(sorted(annotated_frame_id_list)[0][:-4])

        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list: List[int] = np.load(video_sampled_frames_npy_path).tolist()
        start_idx = video_sampled_frame_id_list.index(annotated_first_frame_id)
        video_sampled_frame_id_list = video_sampled_frame_id_list[start_idx:]

        # --- Load & per-frame resize (2D only) ---
        video_masks_dir_path = os.path.join(self.masks_dir_path, video_id) if self.masks_dir_path is not None else None
        assert video_masks_dir_path is not None, "self.masks_dir_path must be set."

        masks_out = []
        for frame_id in video_sampled_frame_id_list:
            mask_path = os.path.join(video_masks_dir_path, f"{frame_id:06d}.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            # Keep integer IDs crisp
            img = Image.open(mask_path).convert("L")
            img = img.resize((W_out, H_out), resample=Image.NEAREST)
            mask_np = np.array(img, dtype=np.uint8)
            masks_out.append(mask_np)

        masks = np.stack(masks_out, axis=0)  # (S, H_out, W_out)
        return masks

    def infer_4d_reconstruction_video(
            self,
            video_id: str,
            *,
            conf_static: float = 0.1,  # confidence for static background build
            conf_frame: float = 0.01,  # confidence for per-frame points
            dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
            fov_y: float = 0.96,  # radians; matches your earlier default
            spawn: bool = True,
            log_cameras: bool = True,
            load_from_glb: bool = False,
    ) -> None:
        """
        Unified visualization that replaces the old `infer_video` and `infer_video_points_3d`.

        - Builds a static background once (using `conf_static`).
        - Optionally logs per-frame RAW points (like the old `infer_video_points_3d`).
        - Optionally logs per-frame MERGED (static + frame) points (like the old `infer_video`).
        """
        # ---- Load predictions once ----
        static_scene_pred_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(self.dynamic_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        if not os.path.exists(static_scene_pred_path) or not os.path.exists(dynamic_scene_pred_path):
            raise FileNotFoundError(f"predictions.npz not found for {video_id}")

        static_scene_arr = np.load(static_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_arr = np.load(dynamic_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_predictions = {k: dynamic_scene_arr[k] for k in dynamic_scene_arr.files}
        S, H, W = dynamic_scene_predictions["points"].shape[:3]

        interaction_masks = self.load_masks_for_video(video_id, target_hw=(H, W))  # (S,H,W), uint8

        if load_from_glb:
            print(f"[viz] Loading static scene from GLB for {video_id}...")
            glb_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", f"{video_id[:-4]}.glb")
            static_P, static_C = glb_to_points(glb_path)
        else:
            print("Loading static scene from predictions...")
            static_scene_predictions = {k: static_scene_arr[k] for k in static_scene_arr.files}
            static_scene_points_wh = static_scene_predictions["points"]  # (S,H,W,3)
            S, H, W = static_scene_points_wh.shape[:3]
            print(f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame}")

            # ---- Build static background (once) ----
            scene_3d, static_P, static_C = predictions_to_glb_with_static(
                static_scene_predictions, conf_min=float(conf_static),
            )

        # ---- Rerun setup ----
        rr.init(f"AG-Pi3: {video_id}", spawn=spawn)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
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
            if not load_from_glb:
                _log_cameras(static_scene_predictions, fov_y=fov_y, W=W, H=H, type="static",
                             color=np.array([255, 0, 0], dtype=np.uint8))  # RED
            _log_cameras(dynamic_scene_predictions, fov_y=fov_y, W=W, H=H, type="dynamic",
                         color=np.array([0, 255, 0], dtype=np.uint8))  # GREEN

        # ---- Precompute per-frame MERGED with static ----
        merged_P: List[np.ndarray] = []
        merged_C: List[np.ndarray] = []
        for i in tqdm(range(S), desc=f"[viz] Merging frames for {video_id}"):
            Pi, Ci = merge_static_with_frame(
                dynamic_scene_predictions,
                static_P, static_C,
                interaction_masks,
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
            self.infer_grounded_dynamic_video(video_id, **kwargs)


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
        "--mask_dir_path",
        type=str,
        default="/data/rohith/ag/segmentation/masks/rectangular_overlayed_masks",
        help="Path to directory containing trained model checkpoints.",
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
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/static_scenes/pi3_full"
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
        masks_dir_path=args.mask_dir_path,
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
