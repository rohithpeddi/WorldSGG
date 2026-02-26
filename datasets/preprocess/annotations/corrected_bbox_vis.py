#!/usr/bin/env python3
"""
corrected_bbox_vis.py
=====================
Shared rerun visualization helpers for corrected bbox generators.

Provides ``rerun_visualize_corrected_bboxes`` that loads a saved corrected
bbox PKL + dynamic predictions and launches a rerun viewer showing:
  - Point cloud (confidence-filtered)
  - OBB wireframes (floor-parallel or AABB fallback) per object
  - Camera frustum with image
  - Floor mesh (optional)

Works for:
  - corrected_world_bbox_generator (world-frame, ``frames`` key)
  - corrected_frame_bbox_generator (final-frame, ``frames_final.bbox_frames`` key)
  - corrected_4d_bbox_generator (final-coords, ``frames`` key with ``*_final`` corners)
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + "/..")
sys.path.insert(0, os.path.dirname(__file__))

from annotation_utils import (
    _faces_u32,
    _as_np,
    _npz_open,
    _pinhole_from_fov,
)

try:
    import rerun as rr
except ImportError:
    rr = None

try:
    from scipy.spatial.transform import Rotation as SciRot
except ImportError:
    SciRot = None


# -----------------------------------------------------------------
# Cuboid edge indices (8-corner box → 12 edges)
# -----------------------------------------------------------------
CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
]

# Default label → colour mapping
_DEFAULT_COLORS = {
    "person": [0, 255, 0],
}
_DEFAULT_COLOR = [255, 180, 0]
_GDINO_COLOR = [0, 180, 255]  # cyan-ish for GDino-sourced objects


# -----------------------------------------------------------------
# Corner extraction helper
# -----------------------------------------------------------------

def _extract_corners(obj: Dict[str, Any], prefer_obb: bool = True) -> Optional[np.ndarray]:
    """Extract 8×3 corners from an object dict.

    Tries in order:
      1. OBB floor-parallel corners (world or final)
      2. AABB corners (world or final)
      3. Top-level ``corners_world`` / ``corners_final``
    """
    # OBB floor-parallel
    for key in ("obb_floor_parallel", "obb_floor_parallel_final"):
        obb = obj.get(key)
        if isinstance(obb, dict):
            for ck in ("corners_world", "corners_final"):
                cw = obb.get(ck)
                if cw is not None:
                    c = np.asarray(cw, dtype=np.float32)
                    if c.shape == (8, 3):
                        return c

    # AABB
    for key in ("aabb_floor_aligned", "aabb_final"):
        aabb = obj.get(key)
        if isinstance(aabb, dict):
            for ck in ("corners_world", "corners_final"):
                cw = aabb.get(ck)
                if cw is not None:
                    c = np.asarray(cw, dtype=np.float32)
                    if c.shape == (8, 3):
                        return c

    # Top-level fallback
    for ck in ("corners_world", "corners_final"):
        cw = obj.get(ck)
        if cw is not None:
            c = np.asarray(cw, dtype=np.float32)
            if c.shape == (8, 3):
                return c

    return None


# -----------------------------------------------------------------
# Main visualizer
# -----------------------------------------------------------------

def rerun_visualize_corrected_bboxes(
    *,
    video_id: str,
    pkl_path: str,
    dynamic_scene_dir_path: str,
    idx_to_frame_idx_path_fn=None,
    app_id: str = "Corrected-BBox",
    img_maxsize: int = 480,
    vis_floor: bool = True,
    min_conf_default: float = 1e-6,
    frames_key: str = "frames",
    floor_vertices_key: str = "gv",
    floor_faces_key: str = "gf",
    floor_colors_key: str = "gc",
) -> None:
    """Load a corrected bbox PKL and launch rerun visualization.

    Parameters
    ----------
    video_id : str
        E.g. ``"001YG.mp4"``.
    pkl_path : str
        Absolute path to the saved corrected bbox PKL.
    dynamic_scene_dir_path : str
        Root of dynamic_scenes/pi3_dynamic.
    idx_to_frame_idx_path_fn : callable, optional
        ``generator.idx_to_frame_idx_path(video_id)`` — returns
        ``(frame_idx_frame_path_map, sample_idx, ..., annotated_frame_idx_in_sample_idx)``.
        If not supplied, we fall back to sequential indexing.
    app_id : str
        Rerun application ID.
    img_maxsize : int
        Max image dimension for rerun display.
    vis_floor : bool
        Whether to render the floor mesh.
    min_conf_default : float
        Minimum confidence threshold for point cloud.
    frames_key : str
        Key in the PKL dict containing ``{frame_name: {objects: [...]}}`` data.
        Default ``"frames"`` for world and 4D generators.
        Set to ``"frames_final.bbox_frames"`` for frame generator.
    floor_vertices_key, floor_faces_key, floor_colors_key : str
        Keys for floor mesh data in the PKL.
    """
    if rr is None:
        raise ImportError("rerun is required for visualization. Install with: pip install rerun-sdk")
    if SciRot is None:
        raise ImportError("scipy is required for visualization.")

    # ---- 1) Load saved PKL ----
    with open(pkl_path, "rb") as f:
        saved: Dict[str, Any] = pickle.load(f)

    # Resolve frames_map (support dotted paths like "frames_final.bbox_frames")
    frames_map = saved
    for part in frames_key.split("."):
        if isinstance(frames_map, dict):
            frames_map = frames_map.get(part, {})
        else:
            frames_map = {}
    if not isinstance(frames_map, dict):
        frames_map = {}

    # Floor mesh
    gv = saved.get(floor_vertices_key)
    gf = saved.get(floor_faces_key)
    gc = saved.get(floor_colors_key)

    # Corrected floor transform for floor mesh positioning
    cft = saved.get("corrected_floor_transform", {})
    original_gfs = cft.get("original_global_floor_sim") if isinstance(cft, dict) else None

    # ---- 2) Load dynamic predictions (images + points + cameras) ----
    pred_path = Path(dynamic_scene_dir_path) / f"{video_id[:-4]}_10" / "predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"[vis] Missing dynamic predictions file: {pred_path}")

    with _npz_open(pred_path) as npz:
        imgs_f32 = npz["images"]
        images_u8 = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)
        images = [images_u8[i] for i in range(images_u8.shape[0])]

        points = npz["points"].astype(np.float32) if "points" in npz else None
        conf = npz["conf"].astype(np.float32) if "conf" in npz else None

        camera_poses = npz["camera_poses"] if "camera_poses" in npz else None
        S_full = int(images_u8.shape[0])

    # ---- 3) Reconstruct indexing ----
    frame_idx_frame_path_map = None
    annotated_frame_idx_in_sample_idx = None
    if idx_to_frame_idx_path_fn is not None:
        try:
            result = idx_to_frame_idx_path_fn(video_id)
            frame_idx_frame_path_map = result[0]
            sample_idx = result[1]
            annotated_frame_idx_in_sample_idx = result[4]
            if len(sample_idx) != S_full:
                S_full = min(S_full, len(sample_idx))
                images = images[:S_full]
        except Exception as e:
            print(f"[vis][warn] idx_to_frame_idx_path failed ({e}); using sequential indexing.")

    if annotated_frame_idx_in_sample_idx is None:
        annotated_frame_idx_in_sample_idx = list(range(S_full))

    sampled_indices = list(range(S_full))

    # ---- 4) Init rerun ----
    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # ---- 5) Floor mesh ----
    floor_vertices_tf = None
    floor_faces_rr = None
    floor_kwargs = None
    if vis_floor and gv is not None and gf is not None:
        gv0 = np.asarray(gv, dtype=np.float32)
        gf0 = _faces_u32(np.asarray(gf))

        # Transform floor vertices through original global_floor_sim if available
        if original_gfs is not None and isinstance(original_gfs, dict):
            s_g = float(original_gfs.get("s", 1.0))
            R_g = _as_np(original_gfs.get("R", np.eye(3)), np.float32)
            t_g = _as_np(original_gfs.get("t", np.zeros(3)), np.float32)
            floor_vertices_tf = s_g * (gv0 @ R_g.T) + t_g
        else:
            floor_vertices_tf = gv0

        floor_faces_rr = gf0
        floor_kwargs = {}
        if gc is not None:
            floor_kwargs["vertex_colors"] = np.asarray(gc, dtype=np.uint8)
        else:
            floor_kwargs["albedo_factor"] = [160, 160, 160]

    # ---- 6) Build frame_name → frame_idx lookup from frames_map ----
    frame_name_to_sidx = {}
    if frame_idx_frame_path_map is not None:
        for fidx, fpath in frame_idx_frame_path_map.items():
            fname = Path(fpath).name
            for sidx_i, si in enumerate(annotated_frame_idx_in_sample_idx):
                if sampled_indices[si] == fidx:
                    frame_name_to_sidx[fname] = sidx_i
                    break

    # ---- 7) Iterate frames ----
    def _resize_image(img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    for vis_t, sample_idx_val in enumerate(annotated_frame_idx_in_sample_idx):
        if sample_idx_val < 0 or sample_idx_val >= len(sampled_indices):
            continue

        frame_idx = sampled_indices[sample_idx_val]

        # Resolve frame_name for this step
        frame_name = None
        if frame_idx_frame_path_map is not None and frame_idx in frame_idx_frame_path_map:
            frame_name = Path(frame_idx_frame_path_map[frame_idx]).name

        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        # Floor
        if floor_vertices_tf is not None and floor_faces_rr is not None:
            rr.log(
                f"{BASE}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_tf.astype(np.float32),
                    triangle_indices=floor_faces_rr,
                    **(floor_kwargs or {}),
                ),
            )

        # Points (confidence-filtered)
        if points is not None and sample_idx_val < points.shape[0]:
            pts = points[sample_idx_val].reshape(-1, 3)
            colors_flat = images_u8[sample_idx_val].reshape(-1, 3) if sample_idx_val < images_u8.shape[0] else None
            cfs = conf[sample_idx_val].reshape(-1) if conf is not None else None

            # Adaptive threshold
            if cfs is not None:
                good = np.isfinite(cfs)
                cfs_valid = cfs[good]
                if cfs_valid.size > 0:
                    p5 = np.percentile(cfs_valid, 5)
                    thr = max(min_conf_default, p5)
                else:
                    thr = min_conf_default
                keep = (cfs >= thr) & np.isfinite(pts).all(axis=1)
            else:
                keep = np.isfinite(pts).all(axis=1)

            pts_keep = pts[keep]
            if pts_keep.shape[0] > 0:
                cols_keep = colors_flat[keep] if colors_flat is not None else None
                kwargs = {}
                if cols_keep is not None:
                    kwargs["colors"] = cols_keep
                rr.log(f"{BASE}/points", rr.Points3D(pts_keep, **kwargs))

        # BBox wireframes
        if frame_name and frame_name in frames_map:
            frame_rec = frames_map[frame_name]
            objs = frame_rec.get("objects", [])
            for obj_idx, obj in enumerate(objs):
                corners = _extract_corners(obj)
                if corners is None:
                    continue

                label = obj.get("label", f"obj_{obj_idx}")
                source = obj.get("source", "gt")
                col = _GDINO_COLOR if source == "gdino" else _DEFAULT_COLORS.get(label, _DEFAULT_COLOR)

                strips = [corners[[e0, e1], :] for (e0, e1) in CUBOID_EDGES]
                rr.log(
                    f"{BASE}/bboxes/{label}_{obj_idx}",
                    rr.LineStrips3D(strips=strips, colors=[col] * len(strips)),
                )

        # Camera frustum
        if camera_poses is not None and sample_idx_val < camera_poses.shape[0]:
            cam_i = np.asarray(camera_poses[sample_idx_val], dtype=np.float32)
            if cam_i.shape == (4, 4):
                R_wc = cam_i[:3, :3]
                t_wc = cam_i[:3, 3]
            elif cam_i.shape == (3, 4):
                R_wc = cam_i[:3, :3]
                t_wc = cam_i[:3, 3]
            else:
                continue

            image = images[frame_idx] if frame_idx < len(images) else None
            if image is not None:
                image = _resize_image(image)
                H_img, W_img = image.shape[:2]
            else:
                H_img, W_img = 480, 640

            fov_y = 0.96
            fx, fy, cx, cy = _pinhole_from_fov(W_img, H_img, fov_y)
            quat_xyzw = SciRot.from_matrix(R_wc).as_quat().astype(np.float32)

            frus_path = f"{BASE}/frustum"
            rr.log(
                frus_path,
                rr.Transform3D(
                    translation=t_wc.astype(np.float32),
                    rotation=rr.Quaternion(xyzw=quat_xyzw),
                ),
            )
            rr.log(
                f"{frus_path}/camera",
                rr.Pinhole(
                    focal_length=(fx, fy),
                    principal_point=(cx, cy),
                    resolution=(W_img, H_img),
                ),
            )
            if image is not None:
                rr.log(f"{frus_path}/image", rr.Image(image))

    print(f"[vis] Rerun visualization running for {video_id}. Scrub the 'frame' timeline.")
