#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from torch.utils.data import DataLoader

from dataloader.standard.action_genome.ag_dataset import StandardAG
from annotation_utils import (
    get_video_belongs_to_split,
    _load_pkl_if_exists,
    _npz_open,
    _faces_u32,
    _resize_mask_to,
    _mask_from_bbox,
    _resize_bbox_to,
    _xywh_to_xyxy,
    _finite_and_nonzero,
    _pinhole_from_fov,
    _is_empty_array,
)
from datasets.preprocess.annotations.frame_to_world_base import FrameToWorldBase


# --------------------------------------------------------------------------------------
# Final world transform helper
# --------------------------------------------------------------------------------------

def compute_final_world_transform(
    floor: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    global_floor_sim: Optional[Tuple[float, np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Compute the world -> final transform used in the visualization code.

    Pipeline:
        1) Apply global floor similarity (s_g, R_g, t_g) to the floor mesh.
        2) Build a floor-aligned frame where:
               - X, Z axes lie in the floor plane
               - Y is the floor normal
           This gives a rotation R_align that maps WORLD -> FLOOR frame,
           with origin at floor_origin_world = t_g.
        3) Optionally apply a mirror across the ZY plane in the FLOOR frame
           (i.e., x -> -x).

    The final transform is:
        x_final = R_final @ x_world + t_final

    Returns a dict with:
        - R_world_to_floor   : (3,3) WORLD -> FLOOR
        - t_world_to_floor   : (3,)  WORLD -> FLOOR
        - T_world_to_floor   : (4,4)

        - R_world_to_final   : (3,3) WORLD -> FINAL (floor (+ mirror))
        - t_world_to_final   : (3,)
        - T_world_to_final   : (4,4)

        - floor_origin_world : (3,)

    Coordinate conventions:
        - All R, t are in column-vector form:
              x_out = R @ x_in + t

    If floor or global_floor_sim is None, this function returns an IDENTITY transform.
    """

    R_identity = np.eye(3, dtype=np.float32)
    t_zero = np.zeros(3, dtype=np.float32)

    result = {
        "R_world_to_floor": R_identity.copy(),
        "t_world_to_floor": t_zero.copy(),
        "T_world_to_floor": np.eye(4, dtype=np.float32),

        "R_world_to_final": R_identity.copy(),
        "t_world_to_final": t_zero.copy(),
        "T_world_to_final": np.eye(4, dtype=np.float32),

        "floor_origin_world": t_zero.copy(),
    }

    if floor is None or global_floor_sim is None:
        return result

    # Unpack
    floor_verts0, floor_faces0, floor_colors0 = floor
    s_g, R_g, t_g = global_floor_sim

    R_g = np.asarray(R_g, dtype=np.float32)      # (3,3)
    t_g = np.asarray(t_g, dtype=np.float32)      # (3,)
    s_g = float(s_g)

    floor_origin_world = t_g.astype(np.float32)

    # Floor-aligned basis
    t1 = R_g[:, 0]  # in-plane
    t2 = R_g[:, 2]  # in-plane
    n  = R_g[:, 1]  # normal

    F = np.stack([t1, t2, n], axis=1)           # (3,3)
    R_align = F.T.astype(np.float32)           # WORLD -> FLOOR

    # Translation WORLD -> FLOOR:
    #   x_floor = R_align @ x_world + t_align
    t_align = -R_align @ floor_origin_world

    T_world_to_floor = np.eye(4, dtype=np.float32)
    T_world_to_floor[:3, :3] = R_align
    T_world_to_floor[:3, 3] = t_align

    # Optional mirror across ZY plane (x -> -x) in FLOOR frame
    M_mirror = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

    R_final = M_mirror @ R_align
    t_final = M_mirror @ t_align

    T_world_to_final = np.eye(4, dtype=np.float32)
    T_world_to_final[:3, :3] = R_final
    T_world_to_final[:3, 3] = t_final

    result.update(
        {
            "R_world_to_floor": R_align,
            "t_world_to_floor": t_align,
            "T_world_to_floor": T_world_to_floor,

            "R_world_to_final": R_final,
            "t_world_to_final": t_final,
            "T_world_to_final": T_world_to_final,

            "floor_origin_world": floor_origin_world,
        }
    )

    return result


# --------------------------------------------------------------------------------------
# Simple (no-transform) Rerun visualization
#   - All geometry must already be in the desired coordinates.
#   - This function only logs what it's given.
# --------------------------------------------------------------------------------------

def rerun_frame_vis_results(
    video_id: str,
    stems_S: List[str],
    frame_annotated_dir_path: Path,
    *,
    # BEFORE (world/original) data
    points_before: Optional[np.ndarray] = None,      # (S,H,W,3)
    conf_before: Optional[np.ndarray] = None,        # (S,H,W)
    colors_before: Optional[np.ndarray] = None,      # (S,H,W,3)
    cameras_before: Optional[np.ndarray] = None,     # (S,4,4) or (S,3,4)
    floor_vertices_before: Optional[np.ndarray] = None,
    floor_axes_before: Optional[np.ndarray] = None,  # (3,3) row-wise vectors
    floor_origin_before: Optional[np.ndarray] = None,  # (3,)
    frame_3dbb_before: Optional[Dict[Any, Dict[str, Any]]] = None,
    # AFTER (final, transformed) data
    points_after: Optional[np.ndarray] = None,       # (S,H,W,3)
    colors_after: Optional[np.ndarray] = None,       # (S,H,W,3)
    cameras_after: Optional[np.ndarray] = None,      # (S,4,4)
    floor_vertices_after: Optional[np.ndarray] = None,
    floor_axes_after: Optional[np.ndarray] = None,   # (3,3)
    floor_origin_after: Optional[np.ndarray] = None, # (3,) usually [0,0,0]
    frame_3dbb_after: Optional[Dict[Any, Dict[str, Any]]] = None,
    # Floor mesh topology & style (shared)
    floor_faces: Optional[np.ndarray] = None,
    floor_kwargs: Optional[Dict[str, Any]] = None,
    img_maxsize: int = 320,
    app_id: str = "World4D-Original",
    min_conf_default: float = 1e-6,
    vis_mode: str = "both",  # "before", "after", or "both"
) -> None:
    """
    Pure visualization function.

    This function assumes:
      - All "before" arrays are in the original/world frame.
      - All "after" arrays are already transformed into the FINAL frame
        (e.g., floor-aligned + mirrored) by external code.

    It does *no* geometric transformations, only logging.
    """

    # -------------------------------------------------------------------------
    # Normalize vis_mode
    # -------------------------------------------------------------------------
    vis_mode = (vis_mode or "both").lower()
    if vis_mode not in {"before", "after", "both"}:
        print(f"[orig-pts][{video_id}] Unknown vis_mode={vis_mode!r}, defaulting to 'both'.")
        vis_mode = "both"

    show_before = vis_mode in {"before", "both"}
    show_after  = vis_mode in {"after", "both"}

    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    BASE = "world"
    BASE_BEFORE = f"{BASE}/before"
    BASE_AFTER = f"{BASE}/after"

    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # Cuboid edge list
    cuboid_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # Try to get grid size for pinhole aspect ratio
    H_grid = W_grid = 1
    if points_before is not None and points_before.ndim == 4:
        _, H_grid, W_grid, _ = points_before.shape
    elif points_after is not None and points_after.ndim == 4:
        _, H_grid, W_grid, _ = points_after.shape

    # -------------------------------------------------------------------------
    # Static world & XYZ axes
    # -------------------------------------------------------------------------
    axis_len_world = 0.5
    world_axes = rr.Arrows3D(
        origins=[[0.0, 0.0, 0.0]] * 3,
        vectors=[
            [axis_len_world, 0.0, 0.0],
            [0.0, axis_len_world, 0.0],
            [0.0, 0.0, axis_len_world],
        ],
        colors=[
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        labels=["World +X", "World +Y", "World +Z"],
    )

    axis_len_xyz = 0.4
    xyz_axes = rr.Arrows3D(
        origins=[[0.0, 0.0, 0.0]] * 3,
        vectors=[
            [axis_len_xyz, 0.0, 0.0],
            [0.0, axis_len_xyz, 0.0],
            [0.0, 0.0, axis_len_xyz],
        ],
        colors=[
            [255, 128, 128],
            [128, 255, 128],
            [128, 128, 255],
        ],
        labels=["X", "Y", "Z"],
    )

    def _log_static_info() -> None:
        rr.log(f"{BASE}/world_axes", world_axes)
        rr.log(f"{BASE}/xyz_axes", xyz_axes)
        rr.log(
            f"{BASE}/info",
            rr.TextLog(
                f"[{video_id}] BEFORE = original/world frame. "
                f"AFTER = final transformed frame (e.g., floor-aligned + mirror). "
                f"vis_mode='{vis_mode}' (before/after/both)."
            ),
        )

    def _get_image_for_stem(stem: str) -> Optional[np.ndarray]:
        img_path = frame_annotated_dir_path / video_id / f"{stem}.png"
        if not img_path.exists():
            return None
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        return img

    # Number of frames
    S = len(stems_S)

    for vis_t in range(S):
        stem = stems_S[vis_t]

        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        _log_static_info()

        # ---------------------------------------------------------------------
        # FLOOR MESH: BEFORE / AFTER
        # ---------------------------------------------------------------------
        if floor_faces is not None and floor_kwargs is None:
            floor_kwargs_local: Dict[str, Any] = {"albedo_factor": [160, 160, 160]}
        else:
            floor_kwargs_local = floor_kwargs or {}

        if show_before and floor_vertices_before is not None and floor_faces is not None:
            rr.log(
                f"{BASE_BEFORE}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_before.astype(np.float32),
                    triangle_indices=floor_faces,
                    **floor_kwargs_local,
                ),
            )

        if show_after and floor_vertices_after is not None and floor_faces is not None:
            rr.log(
                f"{BASE_AFTER}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_after.astype(np.float32),
                    triangle_indices=floor_faces,
                    **floor_kwargs_local,
                ),
            )

        # ---------------------------------------------------------------------
        # FLOOR AXES: BEFORE / AFTER
        # ---------------------------------------------------------------------
        if show_before and floor_origin_before is not None and floor_axes_before is not None:
            origins_floor_before = np.repeat(
                floor_origin_before[None, :], 3, axis=0
            )
            rr.log(
                f"{BASE_BEFORE}/floor_frame",
                rr.Arrows3D(
                    origins=origins_floor_before,
                    vectors=floor_axes_before,
                    colors=[
                        [200, 200, 0],
                        [0, 200, 200],
                        [200, 0, 200],
                    ],
                    labels=["Floor(Before) +X", "Floor(Before) +Y", "Floor(Before) +Z"],
                ),
            )

        if show_after and floor_axes_after is not None:
            if floor_origin_after is None:
                floor_origin_after = np.zeros(3, dtype=np.float32)
            origins_floor_after = np.repeat(
                floor_origin_after[None, :], 3, axis=0
            )
            rr.log(
                f"{BASE_AFTER}/floor_frame",
                rr.Arrows3D(
                    origins=origins_floor_after,
                    vectors=floor_axes_after,
                    colors=[
                        [255, 200, 0],
                        [0, 255, 255],
                        [255, 0, 255],
                    ],
                    labels=["Floor(After) +X", "Floor(After) +Y", "Floor(After) +Z"],
                ),
            )

        # ---------------------------------------------------------------------
        # CAMERAS: BEFORE / AFTER
        # ---------------------------------------------------------------------
        def _log_camera(path_prefix: str, T: np.ndarray, prefix_label: str) -> None:
            if T.shape == (3, 4):
                T_full = np.eye(4, dtype=np.float32)
                T_full[:3, :4] = T
            else:
                T_full = T.astype(np.float32)

            cam_origin = T_full[:3, 3]
            R_cam = T_full[:3, :3]

            axis_len_cam = 0.4
            cam_axes_vec = np.stack(
                [
                    R_cam[:, 0] * axis_len_cam,
                    R_cam[:, 1] * axis_len_cam,
                    R_cam[:, 2] * axis_len_cam,
                ],
                axis=0,
            )
            origins_cam = np.repeat(cam_origin[None, :], 3, axis=0)

            rr.log(
                f"{path_prefix}/camera_axes",
                rr.Arrows3D(
                    origins=origins_cam,
                    vectors=cam_axes_vec,
                    colors=[
                        [180, 0, 0],
                        [0, 180, 0],
                        [0, 0, 180],
                    ],
                    labels=[
                        f"Cam({prefix_label}) +X",
                        f"Cam({prefix_label}) +Y",
                        f"Cam({prefix_label}) +Z",
                    ],
                ),
            )

            rr.log(
                f"{path_prefix}/camera/frustum",
                rr.Pinhole(
                    fov_y=0.7853982,
                    aspect_ratio=float(W_grid) / float(H_grid),
                    camera_xyz=rr.ViewCoordinates.RUB,
                    image_plane_distance=0.1,
                ),
                rr.Transform3D(
                    translation=cam_origin.tolist(),
                    mat3x3=R_cam,
                ),
            )

        if show_before and cameras_before is not None and vis_t < cameras_before.shape[0]:
            _log_camera(BASE_BEFORE, cameras_before[vis_t], "Before")

        if show_after and cameras_after is not None and vis_t < cameras_after.shape[0]:
            _log_camera(BASE_AFTER, cameras_after[vis_t], "After")

        # ---------------------------------------------------------------------
        # POINTS: BEFORE (dark) / AFTER (colorful)
        # ---------------------------------------------------------------------
        pts_b = None
        pts_a = None
        cols_a = None
        conf_flat = None

        if points_before is not None:
            pts_b = points_before[vis_t].reshape(-1, 3)
        if points_after is not None:
            pts_a = points_after[vis_t].reshape(-1, 3)
        if colors_after is not None:
            cols_a = colors_after[vis_t].reshape(-1, 3)

        if conf_before is not None:
            conf_flat = conf_before[vis_t].reshape(-1)

        if conf_flat is not None:
            good = np.isfinite(conf_flat)
            cfs_valid = conf_flat[good]
            if cfs_valid.size > 0:
                med = np.median(cfs_valid)
                p5 = np.percentile(cfs_valid, 5)
                thr = max(min_conf_default, p5)
                print(
                    f"[orig-pts][{video_id}] frame {stem}: "
                    f"conf thr = {thr:.4f} (med={med:.4f}, n_valid={cfs_valid.size})"
                )
            else:
                thr = min_conf_default
            keep_mask = (conf_flat >= thr)
        else:
            keep_mask = None

        if pts_b is not None:
            if keep_mask is None:
                keep_b = np.isfinite(pts_b).all(axis=1)
            else:
                keep_b = keep_mask & np.isfinite(pts_b).all(axis=1)

            if show_before:
                pts_b_keep = pts_b[keep_b]
                if pts_b_keep.shape[0] > 0:
                    rr.log(
                        f"{BASE_BEFORE}/points",
                        rr.Points3D(
                            pts_b_keep.astype(np.float32),
                            colors=np.array([[60, 60, 60]], dtype=np.uint8),
                        ),
                    )

        if pts_a is not None and show_after:
            if keep_mask is None:
                keep_a = np.isfinite(pts_a).all(axis=1)
            else:
                keep_a = keep_mask & np.isfinite(pts_a).all(axis=1)

            pts_a_keep = pts_a[keep_a]
            if cols_a is not None:
                cols_a_keep = cols_a[keep_a].astype(np.uint8)
            else:
                cols_a_keep = None

            if pts_a_keep.shape[0] > 0:
                kwargs_pts: Dict[str, Any] = {}
                if cols_a_keep is not None:
                    kwargs_pts["colors"] = cols_a_keep
                rr.log(
                    f"{BASE_AFTER}/points",
                    rr.Points3D(
                        pts_a_keep.astype(np.float32),
                        **kwargs_pts,
                    ),
                )

        # -----------------------------------------------------------------------------
        # 3D BOUNDING BOXES: BEFORE / AFTER + LABELS
        # -----------------------------------------------------------------------------
        def _log_bbox_with_label(
            base_path: str,
            frame_idx: int,
            bbox_index: int,
            corners: np.ndarray,
            color: List[int],
            label: Optional[str],
        ) -> None:
            """
            Log a wireframe 3D box (LineStrips3D) and, if available, a point label
            slightly above the box center.
            """
            # Wireframe edges
            strips = []
            for e0, e1 in cuboid_edges:
                strips.append(corners[[e0, e1], :])

            rr.log(
                f"{base_path}/bboxes/frame_{frame_idx}/bbox_{bbox_index}",
                rr.LineStrips3D(
                    strips=strips,
                    colors=[color] * len(strips),
                ),
            )

            # Optional label above the box
            if label is not None:
                # Center of the box in 3D
                center = corners.mean(axis=0)

                # Use ~5% of the diagonal length as vertical offset; fall back to small constant
                diag_len = np.linalg.norm(
                    corners.max(axis=0) - corners.min(axis=0)
                )
                offset = 0.05 * diag_len if diag_len > 0 else 0.05

                # Assume +Y is "up" in both BEFORE (world) and AFTER (final) frames
                label_pos = center + np.array([0.0, offset, 0.0], dtype=np.float32)

                rr.log(
                    f"{base_path}/bboxes/frame_{frame_idx}/bbox_{bbox_index}_label",
                    rr.Points3D(
                        positions=label_pos[None, :].astype(np.float32),
                        labels=[str(label)],
                        colors=[color],
                    ),
                )

        if show_before and frame_3dbb_before is not None:
            frame_name = f"{stem}.png"
            if frame_name in frame_3dbb_before:
                frame_objects = frame_3dbb_before[frame_name]["objects"]
                for bi, obj in enumerate(frame_objects):
                    bbox_3d = obj["aabb_floor_aligned"]
                    corners_world = np.asarray(
                        bbox_3d["corners_world"], dtype=np.float32
                    )  # (8,3)

                    col = obj.get("color", [255, 180, 0])
                    label = obj.get("label", None)

                    _log_bbox_with_label(
                        base_path=BASE_BEFORE,
                        frame_idx=vis_t,
                        bbox_index=bi,
                        corners=corners_world,
                        color=col,
                        label=label,
                    )

        if show_after and frame_3dbb_after is not None:
            frame_name = f"{stem}.png"
            if frame_name in frame_3dbb_after:
                frame_objects = frame_3dbb_after[frame_name]["objects"]
                for bi, obj in enumerate(frame_objects):
                    bbox_final = obj["aabb_final"]
                    corners_final = np.asarray(
                        bbox_final["corners_final"], dtype=np.float32
                    )  # (8,3)

                    col = obj.get("color_after", [255, 230, 80])
                    label = obj.get("label", None)

                    _log_bbox_with_label(
                        base_path=BASE_AFTER,
                        frame_idx=vis_t,
                        bbox_index=bi,
                        corners=corners_final,
                        color=col,
                        label=label,
                    )

        # ---------------------------------------------------------------------
        # ORIGINAL RGB FRAME (single copy, not transformed)
        # ---------------------------------------------------------------------
        img = _get_image_for_stem(stem)
        if img is not None:
            rr.log(f"{BASE}/image", rr.Image(img))

    print(
        "[orig-pts] visualization running for "
        f"{video_id}. BEFORE = original frame, AFTER = final transformed frame. "
        f"vis_mode='{vis_mode}'. Scrub the 'frame' timeline in Rerun and "
        "toggle entities in the UI."
    )


# --------------------------------------------------------------------------------------
# FrameToWorldAnnotations
#   - loads 3D bbox annotations (.pkl produced by BBox3DGenerator)
#   - can visualize ORIGINAL Pi3 points + floor mesh + 3D boxes for annotated frames
# --------------------------------------------------------------------------------------


class FrameToWorldAnnotations(FrameToWorldBase):

    def __init__(self, ag_root_directory: str, dynamic_scene_dir_path: str):
        super().__init__(ag_root_directory, dynamic_scene_dir_path)

    # ----------------------------------------------------------------------------------
    # World 4D bbox annotations (skeleton) + ORIGINAL-results visualization
    # ----------------------------------------------------------------------------------

    def generate_video_world_bb_annotations(
        self,
        video_id: str,
        video_id_gt_annotations,
        video_id_gdino_annotations,
        video_id_3d_bbox_predictions,
        visualize: bool = True,
    ) -> None:
        """
        Generate world 4D bbox annotations for a single video.

        Steps:
          1) Load 3D bbox predictions (.pkl) if not already provided.
          2) Inspect per-frame 3D objects and collect stats.
          3) Enforce object permanence: every label in `all_labels` is present
             in every annotated frame. Missing instances are filled using:
                 (a) last known bbox for that object if it exists;
                 (b) otherwise the first known bbox for that object;
                 (c) if neither exists, raise an error.
          4) Compute WORLD -> FINAL transform (floor-aligned + optional mirror).
          5) For each filled object, attach:
                 - original `aabb_floor_aligned` (corners_world)
                 - `aabb_final` (corners in FINAL coords)
          6) For static labels, union bboxes **in FINAL coords** and overwrite
             their `aabb_final` in all frames (world boxes unchanged).
          7) Save the resulting 4D annotations to bbox_4d_root_dir.
          8) Optionally visualize final 4D bboxes (wireframe only, no points).
        """
        print(f"[world4d][{video_id}] Generating world SGG annotations (4D bboxes)")

        # ----------------------------------------------------------------------
        # Load / validate 3D bbox annotations
        # ----------------------------------------------------------------------
        if video_id_3d_bbox_predictions is None:
            video_3dgt = self.get_video_3d_annotations(video_id)
        else:
            video_3dgt = video_id_3d_bbox_predictions

        if video_3dgt is None:
            print(f"[world4d][{video_id}] No 3D bbox annotations found. Skipping.")
            return

        # Optional floor mesh & global floor similarity
        floor = None
        global_floor_sim: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        frame_3dbb_map_world: Optional[Dict[str, Dict[str, Any]]] = None
        floor_vertices_before = None
        floor_vertices_after = None
        floor_axes_before = None
        floor_axes_after = None
        floor_origin_world = None
        floor_faces = None
        floor_kwargs = None

        gv = video_3dgt.get("gv", None)
        gf = video_3dgt.get("gf", None)
        gc = video_3dgt.get("gc", None)
        if gv is not None and gf is not None:
            floor = (gv, gf, gc)

        gfs = video_3dgt.get("global_floor_sim", None)
        if gfs is not None:
            s_g = float(gfs["s"])
            R_g = np.asarray(gfs["R"], dtype=np.float32)
            t_g = np.asarray(gfs["t"], dtype=np.float32)
            global_floor_sim = (s_g, R_g, t_g)

        # 3D bboxes per frame (WORLD coords)
        frame_3dbb_map_world = video_3dgt.get("frames", None)
        if frame_3dbb_map_world is None or not frame_3dbb_map_world:
            print(f"[world4d][{video_id}] 3D bbox frames map is empty. Skipping.")
            return

        # Precompute floor mesh in WORLD coords (BEFORE)
        if floor is not None and global_floor_sim is not None:
            floor_verts0 = np.asarray(gv, dtype=np.float32)
            floor_faces0 = _faces_u32(np.asarray(gf))
            floor_faces = floor_faces0

            s_g, R_g0, t_g0 = global_floor_sim
            R_g0 = np.asarray(R_g0, dtype=np.float32)
            t_g0 = np.asarray(t_g0, dtype=np.float32)

            floor_vertices_before = s_g * (floor_verts0 @ R_g0.T) + t_g0

            # Axes BEFORE (in world)
            t1 = R_g0[:, 0]  # in-plane
            t2 = R_g0[:, 2]  # in-plane
            n = R_g0[:, 1]   # normal

            floor_origin_world = t_g0.astype(np.float32)
            axis_len_floor = float(s_g) * 0.5 if s_g is not None else 0.5
            floor_axes_before = np.stack(
                [
                    t1 * axis_len_floor,
                    t2 * axis_len_floor,
                    n * axis_len_floor,
                ],
                axis=0,
            ).astype(np.float32)

            # Floor colors
            if gc is not None:
                gc_arr = np.asarray(gc, dtype=np.uint8)
                floor_kwargs = {"vertex_colors": gc_arr}
            else:
                floor_kwargs = {"albedo_factor": [160, 160, 160]}

        # ----------------------------------------------------------------------
        # Compute WORLD -> FINAL transform using helper
        # ----------------------------------------------------------------------
        tf = compute_final_world_transform(floor=floor, global_floor_sim=global_floor_sim)
        R_final = tf["R_world_to_final"]        # (3,3)
        t_final = tf["t_world_to_final"]        # (3,)
        floor_origin_world_tf = tf["floor_origin_world"]  # (3,)

        # ----------------------------------------------------------------------
        # Transform floor mesh & axes to FINAL coords
        # ----------------------------------------------------------------------
        floor_origin_final = None
        if floor_vertices_before is not None:
            v_flat = floor_vertices_before.reshape(-1, 3)
            v_final_flat = (R_final @ v_flat.T).T + t_final[None, :]
            floor_vertices_after = v_final_flat.reshape(floor_vertices_before.shape)

            # floor origin in world maps to origin in FINAL (by construction),
            # but we can be explicit:
            floor_origin_final = (R_final @ floor_origin_world_tf) + t_final
            # For visualization, we expect this to be ~[0,0,0].

            axis_len_floor = (
                np.linalg.norm(floor_axes_before[0])
                if floor_axes_before is not None
                else 0.5
            )
            floor_axes_after = np.array(
                [
                    [axis_len_floor, 0.0, 0.0],
                    [0.0, axis_len_floor, 0.0],
                    [0.0, 0.0, axis_len_floor],
                ],
                dtype=np.float32,
            )

        # ----------------------------------------------------------------------
        # Basic stats before 4D filling
        # ----------------------------------------------------------------------
        all_labels = set()
        num_frames_with_objects = 0
        num_total_objects = 0

        for frame_name, frame_rec in frame_3dbb_map_world.items():
            objects = frame_rec.get("objects", [])
            if not objects:
                continue
            num_frames_with_objects += 1
            num_total_objects += len(objects)
            for obj in objects:
                lbl = obj.get("label", None)
                if lbl:
                    all_labels.add(lbl)

        print(
            f"[world4d][{video_id}] frames_with_objects={num_frames_with_objects}, "
            f"total_objects={num_total_objects}, "
            f"unique_labels={sorted(all_labels)}"
        )

        if not all_labels:
            print(f"[world4d][{video_id}] No object labels found. Skipping 4D generation.")
            return

        # ----------------------------------------------------------------------
        # Create world-4D annotations with object permanence
        # ----------------------------------------------------------------------
        # Sort frames chronologically by frame index (000123.png -> 123)
        frame_names_sorted = sorted(
            frame_3dbb_map_world.keys(),
            key=lambda fn: int(Path(fn).stem)
            if Path(fn).stem.isdigit()
            else Path(fn).stem,
        )

        # First known bbox per label: label -> (source_frame_name, obj_dict)
        label_first_source: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for fname in frame_names_sorted:
            objects = frame_3dbb_map_world[fname].get("objects", [])
            for obj in objects:
                lbl = obj.get("label", None)
                if not lbl:
                    continue
                if lbl not in label_first_source:
                    label_first_source[lbl] = (fname, obj)

        # Safety check: every label must appear at least once
        for lbl in all_labels:
            if lbl not in label_first_source:
                raise ValueError(
                    f"[world4d][{video_id}] Label '{lbl}' in all_labels "
                    "has no actual bbox in any frame."
                )

        # Helper to clone an object and attach 4D metadata
        def _clone_with_meta(
            base_obj: Dict[str, Any],
            *,
            filled: bool,
            source_frame: str,
            target_frame: str,
        ) -> Dict[str, Any]:
            new_obj = dict(base_obj)  # shallow copy is enough; we don't mutate nested dicts
            new_obj["world4d_filled"] = bool(filled)
            new_obj["world4d_source_frame"] = source_frame
            new_obj["world4d_frame"] = target_frame
            return new_obj

        # last known bbox per label on/before current frame
        last_seen: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        frames_filled_world: Dict[str, Dict[str, Any]] = {}

        for fname in frame_names_sorted:
            frame_rec = frame_3dbb_map_world.get(fname, {})
            objects = frame_rec.get("objects", [])

            # Map label -> object for this frame (if present)
            label_to_obj_current: Dict[str, Dict[str, Any]] = {}
            for obj in objects:
                lbl = obj.get("label", None)
                if not lbl:
                    continue
                # If multiple instances of the same label exist, we keep the first.
                if lbl not in label_to_obj_current:
                    label_to_obj_current[lbl] = obj

            filled_objects: List[Dict[str, Any]] = []

            for lbl in sorted(all_labels):
                if lbl in label_to_obj_current:
                    base_obj = label_to_obj_current[lbl]
                    last_seen[lbl] = (fname, base_obj)
                    filled_obj = _clone_with_meta(
                        base_obj,
                        filled=False,
                        source_frame=fname,
                        target_frame=fname,
                    )
                else:
                    # Use last known bbox if available; otherwise use first known bbox
                    if lbl in last_seen:
                        src_frame, base_obj = last_seen[lbl]
                    else:
                        src_frame, base_obj = label_first_source[lbl]

                    filled_obj = _clone_with_meta(
                        base_obj,
                        filled=True,
                        source_frame=src_frame,
                        target_frame=fname,
                    )

                # Ensure label is consistent
                filled_obj["label"] = lbl

                # Attach FINAL-coords bbox (aabb_final) from WORLD corners
                if "aabb_floor_aligned" in filled_obj:
                    bbox_3d = filled_obj["aabb_floor_aligned"]
                    corners_world = np.asarray(
                        bbox_3d.get("corners_world", []), dtype=np.float32
                    )
                    if corners_world.size == 0:
                        corners_final = corners_world
                    else:
                        corners_final = (R_final @ corners_world.T).T + t_final[None, :]

                    filled_obj["aabb_final"] = {
                        "corners_final": corners_final,
                    }

                # Choose a default color for AFTER visualization if not present
                if "color_after" not in filled_obj:
                    filled_obj["color_after"] = filled_obj.get("color", [255, 230, 80])

                filled_objects.append(filled_obj)

            frames_filled_world[fname] = {"objects": filled_objects}

        # ----------------------------------------------------------------------
        # Static-object union logic (UNION IN FINAL COORDS, WORLD UNCHANGED)
        # ----------------------------------------------------------------------
        # Loads object labels corresponding to active objects in the dataset
        video_active_object_labels = self.video_id_active_objects_annotations_map[video_id]
        video_reasoned_active_object_labels = self.video_id_active_objects_b_reasoned_map[video_id]

        # Heuristic list of non-moving labels
        non_moving_objects = ["floor", "sofa", "couch", "bed", "doorway", "table", "chair"]

        video_dynamic_object_labels = [
            obj for obj in video_reasoned_active_object_labels
            if obj not in non_moving_objects
        ]

        # Objects active in the video but not classified as dynamic => treated as static
        video_static_object_labels = [
            obj for obj in video_active_object_labels
            if obj not in video_dynamic_object_labels
        ]

        # Only consider static labels that actually have 3D boxes
        static_labels_in_3d = [
            lbl for lbl in video_static_object_labels
            if lbl in all_labels
        ]

        print(
            f"[world4d][{video_id}] static_labels_in_3d={sorted(static_labels_in_3d)} "
            f"(from active={sorted(video_active_object_labels)})"
        )

        def _make_aabb_corners_from_minmax(min_xyz: np.ndarray,
                                           max_xyz: np.ndarray) -> np.ndarray:
            """
            Build 8 axis-aligned cuboid corners from min/max in FINAL coords.

            Corner indexing is consistent with a standard cuboid and works
            with the existing `cuboid_edges` list in rerun_frame_vis_results.
            """
            x0, y0, z0 = min_xyz
            x1, y1, z1 = max_xyz
            return np.array(
                [
                    [x0, y0, z0],
                    [x1, y0, z0],
                    [x0, y0, z1],
                    [x1, y0, z1],
                    [x0, y1, z0],
                    [x1, y1, z0],
                    [x0, y1, z1],
                    [x1, y1, z1],
                ],
                dtype=np.float32,
            )

        static_union_map_final: Dict[str, np.ndarray] = {}

        # 1) Compute union bbox (FINAL coords) per static label
        for lbl in static_labels_in_3d:
            all_corners_list_final: List[np.ndarray] = []

            for fname in frame_names_sorted:
                frame_rec = frames_filled_world[fname]
                for obj in frame_rec["objects"]:
                    if obj.get("label") != lbl:
                        continue

                    aabb_final = obj.get("aabb_final", None)
                    if not aabb_final:
                        continue

                    corners_final = np.asarray(
                        aabb_final.get("corners_final", []), dtype=np.float32
                    )
                    if corners_final.size == 0:
                        continue

                    all_corners_list_final.append(corners_final)

            if not all_corners_list_final:
                print(
                    f"[world4d][{video_id}] WARNING: static label '{lbl}' has no "
                    "aabb_final corners available; skipping union."
                )
            else:
                all_pts_final = np.concatenate(all_corners_list_final, axis=0)  # (N*8, 3)
                min_xyz_f = all_pts_final.min(axis=0)
                max_xyz_f = all_pts_final.max(axis=0)
                union_corners_final = _make_aabb_corners_from_minmax(min_xyz_f, max_xyz_f)
                static_union_map_final[lbl] = union_corners_final

        # 2) Apply union bbox (FINAL coords) to all frames for those static labels
        if static_union_map_final:
            print(
                f"[world4d][{video_id}] Applying static union bboxes in FINAL coords "
                f"for {len(static_union_map_final)} labels."
            )

            for fname in frame_names_sorted:
                frame_rec = frames_filled_world[fname]
                for obj in frame_rec["objects"]:
                    lbl = obj.get("label", None)
                    if lbl not in static_union_map_final:
                        continue

                    union_corners_final = static_union_map_final[lbl]  # (8,3)

                    # Ensure aabb_final exists, then overwrite only FINAL box
                    if "aabb_final" not in obj:
                        obj["aabb_final"] = {}

                    obj["aabb_final"]["corners_final"] = union_corners_final

                    # NOTE: we intentionally do NOT touch
                    # obj["aabb_floor_aligned"]["corners_world"]
                    # so WORLD-space boxes remain as originally constructed.

        # ----------------------------------------------------------------------
        # Optional: visualize world-4D bboxes + POINTS (BEFORE/AFTER) over time
        # ----------------------------------------------------------------------
        if visualize:
            # --- 1) Load original Pi3 outputs for annotated frames (same slicing logic) ---
            P = self._load_original_points_for_video(video_id)

            # P["frame_stems"] are like ["000123", ...]
            stem_to_idx = {f"{s}.png": i for i, s in enumerate(P["frame_stems"])}

            # Make sure ordering matches your bbox frame_names_sorted
            keep_frame_names = [fn for fn in frame_names_sorted if fn in stem_to_idx]
            if len(keep_frame_names) == 0:
                raise RuntimeError(
                    f"[world4d][{video_id}] No overlap between bbox frames and loaded points stems."
                )

            idxs = [stem_to_idx[fn] for fn in keep_frame_names]
            stems_S = [Path(fn).stem for fn in keep_frame_names]

            points_world = np.asarray(P["points"], dtype=np.float32)[idxs]          # (S,H,W,3)
            conf_world = (np.asarray(P["conf"], dtype=np.float32)[idxs]
                          if P["conf"] is not None else None)                       # (S,H,W) or None
            colors_world = (np.asarray(P["colors"], dtype=np.uint8)[idxs]
                            if P["colors"] is not None else None)                   # (S,H,W,3) or None
            cameras_world = (np.asarray(P["camera_poses"], dtype=np.float32)[idxs]
                             if P["camera_poses"] is not None else None)            # (S,4,4)/(S,3,4) or None

            # --- 2) Transform points into FINAL coords ---
            pts_flat = points_world.reshape(-1, 3)
            pts_final_flat = (R_final @ pts_flat.T).T + t_final[None, :]
            points_final = pts_final_flat.reshape(points_world.shape).astype(np.float32)

            # Reuse same RGB for FINAL
            colors_final = colors_world

            # --- 3) Transform cameras into FINAL coords ---
            cameras_final = None
            if cameras_world is not None:
                cam_list = []
                for cam_pose in cameras_world:
                    if cam_pose.shape == (3, 4):
                        T_wc = np.eye(4, dtype=np.float32)
                        T_wc[:3, :4] = cam_pose
                    elif cam_pose.shape == (4, 4):
                        T_wc = cam_pose.astype(np.float32)
                    else:
                        T_wc = np.eye(4, dtype=np.float32)

                    R_wc = T_wc[:3, :3]
                    t_wc = T_wc[:3, 3]

                    # x_final = R_final @ x_world + t_final
                    # so: R_fc = R_final @ R_wc, t_fc = R_final @ t_wc + t_final
                    R_fc = R_final @ R_wc
                    t_fc = (R_final @ t_wc) + t_final

                    T_fc = np.eye(4, dtype=np.float32)
                    T_fc[:3, :3] = R_fc
                    T_fc[:3, 3] = t_fc
                    cam_list.append(T_fc)

                cameras_final = np.stack(cam_list, axis=0).astype(np.float32)

            # (Optional) trim bbox dict to the same set of frames for cleanliness
            frames_filled_world_vis = {fn: frames_filled_world[fn] for fn in keep_frame_names}

            print(
                f"[world4d][{video_id}] visualize=True -> launching Rerun "
                "(4D bboxes + points over time, FINAL coords)."
            )

            rerun_frame_vis_results(
                video_id=video_id,
                stems_S=stems_S,
                frame_annotated_dir_path=self.frame_annotated_dir_path,

                # Points/cameras BEFORE
                points_before=None,
                conf_before=conf_world,
                colors_before=None,
                cameras_before=None,

                # Points/cameras AFTER
                points_after=points_final,
                colors_after=colors_final,
                cameras_after=cameras_final,

                # Floor mesh / axes BEFORE + AFTER (if available)
                floor_vertices_before=floor_vertices_before,
                floor_axes_before=floor_axes_before,
                floor_origin_before=floor_origin_world,
                floor_vertices_after=floor_vertices_after,
                floor_axes_after=floor_axes_after,
                floor_origin_after=floor_origin_final,

                # 4D bboxes in FINAL coords
                frame_3dbb_before=None,
                frame_3dbb_after=frames_filled_world_vis,

                floor_faces=floor_faces,
                floor_kwargs=floor_kwargs,

                img_maxsize=480,
                app_id="World4D-BBoxes+Points",
                vis_mode="after",   # change to "both" if you want BEFORE/AFTER side-by-side
            )

        # ----------------------------------------------------------------------
        # Save world-4D annotations
        # ----------------------------------------------------------------------
        out_4d_path = self.bbox_4d_root_dir / f"{video_id[:-4]}.pkl"
        world4d_annotations = {
            "video_id": video_id,
            "frames": frames_filled_world,          # frame_name -> {objects: [...]}
            "frame_names": frame_names_sorted,      # ordered list of frame_name strings
            "all_labels": sorted(all_labels),
            "meta": {
                "num_frames": len(frame_names_sorted),
                "labels": sorted(all_labels),
            },
        }

        with open(out_4d_path, "wb") as f:
            pickle.dump(world4d_annotations, f)

        print(f"[world4d][{video_id}] Saved world-4D bbox annotations to {out_4d_path}")


    # ----------------------------------------------------------------------------------
    # NEW: centralized transform + visualization entry point
    # ----------------------------------------------------------------------------------

    def visualize_original_results(self, video_id: str, vis_mode: str = "after") -> None:
        """
        1) Loads original Pi3 points using the same slicing logic as bbox_3D construction.
        2) Loads floor mesh + global_floor_sim from the 3D bbox .pkl (if present).
        3) Uses `compute_final_world_transform` to compute WORLD -> FINAL transform.
        4) Applies that transform to:
             - points
             - cameras
             - floor mesh
             - 3D bbox corners
        5) Calls `rerun_frame_vis_results` with both BEFORE (world) and AFTER (final) data.
        """
        P = self._load_original_points_for_video(video_id)

        points_world = P["points"]          # (S,H,W,3)
        conf_world = P["conf"]              # (S,H,W) or None
        stems = P["frame_stems"]            # ["000123", ...]
        colors_world = P["colors"]          # (S,H,W,3)
        camera_poses_world = P["camera_poses"]  # (S,4,4) or (S,3,4) or None

        # Optional floor mesh + similarity transform from bbox_3d pkl
        floor = None
        global_floor_sim = None
        frame_3dbb_map_world = None
        floor_vertices_before = None
        floor_vertices_after = None
        floor_axes_before = None
        floor_axes_after = None
        floor_origin_world = None
        floor_faces = None
        floor_kwargs = None

        video_3dgt = self.get_video_3d_annotations(video_id)
        if video_3dgt is not None:
            gv = video_3dgt.get("gv", None)
            gf = video_3dgt.get("gf", None)
            gc = video_3dgt.get("gc", None)
            if gv is not None and gf is not None:
                floor = (gv, gf, gc)

            gfs = video_3dgt.get("global_floor_sim", None)
            if gfs is not None:
                s_g = float(gfs["s"])
                R_g = np.asarray(gfs["R"], dtype=np.float32)
                t_g = np.asarray(gfs["t"], dtype=np.float32)
                global_floor_sim = (s_g, R_g, t_g)

            # 3D bboxes per frame
            frame_3dbb_map_world = video_3dgt.get("frames", None)

            # Precompute floor mesh in WORLD coords
            if floor is not None and global_floor_sim is not None:
                floor_verts0 = np.asarray(gv, dtype=np.float32)
                floor_faces0 = _faces_u32(np.asarray(gf))
                floor_faces = floor_faces0

                s_g, R_g, t_g = global_floor_sim
                R_g = np.asarray(R_g, dtype=np.float32)
                t_g = np.asarray(t_g, dtype=np.float32)

                floor_vertices_before = s_g * (floor_verts0 @ R_g.T) + t_g

                # Axes BEFORE (in world)
                t1 = R_g[:, 0]  # in-plane
                t2 = R_g[:, 2]  # in-plane
                n  = R_g[:, 1]  # normal

                floor_origin_world = t_g.astype(np.float32)
                axis_len_floor = float(s_g) * 0.5 if s_g is not None else 0.5
                floor_axes_before = np.stack(
                    [
                        t1 * axis_len_floor,
                        t2 * axis_len_floor,
                        n  * axis_len_floor,
                    ],
                    axis=0,
                )  # (3,3)

                # Floor colors
                if gc is not None:
                    gc = np.asarray(gc, dtype=np.uint8)
                    floor_kwargs = {"vertex_colors": gc}
                else:
                    floor_kwargs = {"albedo_factor": [160, 160, 160]}

        # ----------------------------------------------------------------------
        # Compute WORLD -> FINAL transform using helper
        # ----------------------------------------------------------------------
        tf = compute_final_world_transform(floor=floor, global_floor_sim=global_floor_sim)
        R_final = tf["R_world_to_final"]        # (3,3)
        t_final = tf["t_world_to_final"]        # (3,)
        floor_origin_world_tf = tf["floor_origin_world"]  # (3,)

        # ----------------------------------------------------------------------
        # Transform points
        # ----------------------------------------------------------------------
        points_final = None
        if points_world is not None:
            pts_flat = points_world.reshape(-1, 3)
            pts_final_flat = (R_final @ pts_flat.T).T + t_final[None, :]
            points_final = pts_final_flat.reshape(points_world.shape)

        # Colors: we reuse the same colors as BEFORE for AFTER
        colors_final = colors_world

        # ----------------------------------------------------------------------
        # Transform cameras
        # ----------------------------------------------------------------------
        cameras_final = None
        if camera_poses_world is not None:
            cam_list = []
            for cam_pose in camera_poses_world:
                if cam_pose.shape == (3, 4):
                    T_wc = np.eye(4, dtype=np.float32)
                    T_wc[:3, :4] = cam_pose
                elif cam_pose.shape == (4, 4):
                    T_wc = cam_pose.astype(np.float32)
                else:
                    T_wc = np.eye(4, dtype=np.float32)

                R_wc = T_wc[:3, :3]
                t_wc = T_wc[:3, 3]

                # FINAL-from-CAMERA
                R_fc = R_final @ R_wc
                t_fc = R_final @ t_wc + t_final

                T_fc = np.eye(4, dtype=np.float32)
                T_fc[:3, :3] = R_fc
                T_fc[:3, 3] = t_fc
                cam_list.append(T_fc)

            cameras_final = np.stack(cam_list, axis=0)

        # ----------------------------------------------------------------------
        # Transform floor mesh & axes to FINAL coords
        # ----------------------------------------------------------------------
        floor_origin_final = None
        if floor_vertices_before is not None:
            v_flat = floor_vertices_before.reshape(-1, 3)
            v_final_flat = (R_final @ v_flat.T).T + t_final[None, :]
            floor_vertices_after = v_final_flat.reshape(floor_vertices_before.shape)

            # floor origin in world maps to origin in FINAL (by construction),
            # but we can be explicit:
            floor_origin_final = (R_final @ floor_origin_world_tf) + t_final
            # For visualization, we expect this to be ~[0,0,0].
            # Axes AFTER: canonical basis in FINAL, scaled similarly
            axis_len_floor = np.linalg.norm(floor_axes_before[0]) if floor_axes_before is not None else 0.5
            floor_axes_after = np.array(
                [
                    [axis_len_floor, 0.0, 0.0],
                    [0.0, axis_len_floor, 0.0],
                    [0.0, 0.0, axis_len_floor],
                ],
                dtype=np.float32,
            )

        # ----------------------------------------------------------------------
        # Transform 3D bounding boxes
        #   - Add `aabb_final` with `corners_final` into a separate map
        # ----------------------------------------------------------------------
        frame_3dbb_map_final: Optional[Dict[str, Dict[str, Any]]] = None
        if frame_3dbb_map_world is not None:
            frame_3dbb_map_final = {}
            for frame_name, frame_rec in frame_3dbb_map_world.items():
                objects_world = frame_rec.get("objects", [])
                objects_final = []
                for obj in objects_world:
                    bbox_3d = obj["aabb_floor_aligned"]
                    corners_world = np.asarray(
                        bbox_3d["corners_world"], dtype=np.float32
                    )  # (8,3)

                    corners_final = (R_final @ corners_world.T).T + t_final[None, :]

                    obj_final = dict(obj)
                    obj_final["aabb_final"] = {
                        "corners_final": corners_final,
                    }
                    # Optional: separate color for AFTER
                    obj_final["color_after"] = obj.get("color", [255, 230, 80])

                    objects_final.append(obj_final)

                frame_3dbb_map_final[frame_name] = {
                    "objects": objects_final
                }

        # ----------------------------------------------------------------------
        # Call visualization (no transforms inside)
        # ----------------------------------------------------------------------
        rerun_frame_vis_results(
            video_id=video_id,
            stems_S=stems,
            frame_annotated_dir_path=self.frame_annotated_dir_path,
            points_before=points_world,
            conf_before=conf_world,
            colors_before=colors_world,
            cameras_before=camera_poses_world,
            floor_vertices_before=floor_vertices_before,
            floor_axes_before=floor_axes_before,
            floor_origin_before=floor_origin_world,
            frame_3dbb_before=frame_3dbb_map_world,
            points_after=points_final,
            colors_after=colors_final,
            cameras_after=cameras_final,
            floor_vertices_after=floor_vertices_after,
            floor_axes_after=floor_axes_after,
            floor_origin_after=floor_origin_final,
            frame_3dbb_after=frame_3dbb_map_final,
            floor_faces=floor_faces,
            floor_kwargs=floor_kwargs,
            img_maxsize=480,
            app_id="World4D-Original",
            vis_mode=vis_mode,
        )

    def generate_sample_gt_world_4D_annotations(self, video_id: str) -> None:
        """
        Example/debug entry point for a single video.

        Right now:
          - loads GT / GDINO / 3D bboxes;
          - calls generate_video_bb_annotations (skeleton stats);
          - visualizes original Pi3 point clouds + floor mesh + frames + camera + 3D boxes.
        """
        video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
        video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
        video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)

        self.generate_video_world_bb_annotations(
            video_id,
            video_id_gt_annotations,
            video_id_gdino_annotations,
            video_id_3d_bbox_predictions,
            visualize=True,
        )

        # Now show the original Pi3-space outputs (points + floor + frames + camera + 3D boxes)
        self.visualize_original_results(video_id=video_id)

    def generate_gt_world_bb_annotations(
        self, dataloader: DataLoader, split: str
    ) -> None:
        """
        Iterate over an AG dataloader and call generate_video_bb_annotations
        for videos in the given AG split.

        This is useful if/when you implement full world-4D GT generation
        for all videos.
        """
        for data in dataloader:
            video_id = data["video_id"]
            if get_video_belongs_to_split(video_id) != split:
                continue

            try:
                print(f"[world4d] processing video {video_id}...")
                video_id_gt_bboxes, video_id_gt_annotations = (
                    self.get_video_gt_annotations(video_id)
                )
                video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
                video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)

                self.generate_video_world_bb_annotations(
                    video_id,
                    video_id_gt_annotations,
                    video_id_gdino_annotations,
                    video_id_3d_bbox_predictions,
                    visualize=False,
                )
            except Exception as e:
                print(f"[world4d] failed to process video {video_id}: {e}")


# --------------------------------------------------------------------------------------
# Dataset + CLI (unchanged)
# --------------------------------------------------------------------------------------

def load_dataset(ag_root_directory: str):
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    dataloader_train = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0,
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False,
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "World4D GT helper: "
            "(a) inspect 3D bbox annotations, "
            "(b) visualize original Pi3 outputs (points + floor + frames + camera + 3D boxes) "
            "for annotated frames."
        )
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument("--split", type=str, default="04")
    return parser.parse_args()


def main():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    _, _, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)

    frame_to_world_generator.generate_gt_world_bb_annotations(
        dataloader=dataloader_train, split=args.split
    )
    frame_to_world_generator.generate_gt_world_bb_annotations(
        dataloader=dataloader_test, split=args.split
    )


def main_sample():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    video_id = "00T1E.mp4"
    # frame_to_world_generator.visualize_original_results(
    #     video_id=video_id,
    #     vis_mode="after",  # "before", "after", or "both"
    # )

    frame_to_world_generator.fetch_stored_active_objects_in_video(video_id)
    frame_to_world_generator.generate_video_world_bb_annotations(
        video_id=video_id,
        video_id_gt_annotations=frame_to_world_generator.get_video_gt_annotations(video_id)[1],
        video_id_gdino_annotations=frame_to_world_generator.get_video_gdino_annotations(video_id),
        video_id_3d_bbox_predictions=frame_to_world_generator.get_video_3d_annotations(video_id),
        visualize=True,
    )


if __name__ == "__main__":
    # main()
    main_sample()
