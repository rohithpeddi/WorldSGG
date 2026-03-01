#!/usr/bin/env python3
"""
frame_to_world4D_annotations.py
================================
Generates world 4D bounding box annotations from bridge PKL files.

Input:  bbox_annotations_3d_obb_bridge/{video}.pkl   (from bb3D_bridge_generator_obb.py)
Output: bbox_annotations_4d/{video}.pkl

Pipeline:
    1) Load bridge PKL (GT + GDino merged, corners_final pre-computed).
    2) Normalize all object labels to GT short-form space.
    3) Enforce object permanence across all annotated frames:
       - Detected objects (GT or GDino) are used as-is.
       - Static objects (sofa, bed, table, ...) are copied from nearest detection.
       - Dynamic objects are interpolated (center lerp + rotation slerp + extent lerp).
       - If only one side detection exists, it is held.
    4) Compute static OBB union per label in FINAL coords (cv2.minAreaRect on XZ).
    5) Save 4D annotations with all bridge metadata.

OUTPUT PKL STRUCTURE
--------------------
{
    "video_id":        str,                   # e.g. "00T1E.mp4"
    "frame_names":     List[str],             # sorted: ["000010.png", "000015.png", ...]
    "all_labels":      List[str],             # sorted unique labels: ["bed", "person", ...]
    "meta": {
        "num_frames":  int,
        "labels":      List[str],
    },

    # --- Per-frame 4D-filled objects ---
    "frames": {
        "<frame>.png": {
            "objects": [
                {
                    "label":                str,           # normalized GT short-form
                    "source":               "gt"|"gdino",  # original detection source
                    "obb_floor_parallel": {                 # WORLD-space OBB (pass-through)
                        "corners_world":    ndarray (8,3),
                        "center_world":     ndarray (3,),
                        ...
                    },
                    "corners_final":        ndarray (8,3),  # FINAL-space corners (from bridge)
                    "obb_final": {                          # FINAL-space OBB (pipeline output)
                        "corners_final":    ndarray (8,3),  # may be union for static labels
                    },
                    "color":                [R, G, B],
                    "color_after":          [R, G, B],      # visualization color

                    # --- 4D filling metadata ---
                    "world4d_filled":       bool,           # True if not directly detected
                    "world4d_fill_method":  str,            # "detected" | "interpolation"
                                                           # | "static_copy" | "hold_prev"
                                                           # | "hold_next"
                    "world4d_source_frame": str,            # frame the data was copied from
                    "world4d_frame":        str,            # this frame name
                },
                ...
            ]
        },
        ...
    },

    # --- Bridge pass-through data ---
    "camera_poses":    ndarray (S, 4, 4),     # camera-to-FINAL transforms
    "frame_stems":     List[str],             # ["000010", "000015", ...]
    "floor_mesh": {                           # raw floor mesh (WORLD space)
        "gv":          ndarray (V, 3),        # vertices
        "gf":          ndarray (F, 3),        # faces
        "gc":          ndarray (V, 3)|None,   # vertex colors
    } | None,
    "global_floor_sim": {                     # similarity transform params
        "s":           float,                 # scale
        "R":           ndarray (3, 3),        # rotation
        "t":           ndarray (3,),          # translation
    } | None,
    "world_to_final": {                       # pre-computed WORLD→FINAL transform
        "origin_world":       ndarray (3,),   # floor origin in WORLD
        "A_world_to_final":   ndarray (3, 3), # rotation+mirror matrix
    } | None,
    "floor_final":     dict | None,           # floor mesh already in FINAL coords
}

COORDINATE SYSTEMS
------------------
- WORLD:  Raw Pi3 reconstruction space (arbitrary orientation).
- FINAL:  Floor-aligned space (Y=up, XZ=floor plane, origin at floor center).

TRANSFORMATIONS
---------------
To transform a point from WORLD to FINAL:

    Using world_to_final (row-vector, from bridge):
        pts_final = (pts_world - origin_world) @ A_world_to_final.T

    Using compute_final_world_transform (column-vector, re-derived):
        pts_final = R_final @ pts_world + t_final

    Both are equivalent:
        R_final = A_world_to_final
        t_final = -A_world_to_final @ origin_world

To visualize:
    - obb_final.corners_final are ALREADY in FINAL coords (ready to render).
    - camera_poses are ALREADY in FINAL coords.
    - To transform raw point clouds: load predictions.npz, apply WORLD→FINAL above.
    - floor_final contains floor mesh in FINAL coords (ready to render).
    - floor_mesh.gv needs WORLD→FINAL if rendering raw WORLD floor.
"""
import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from datasets.preprocess.annotations.annotation_utils import (
    _faces_u32,
)
from datasets.preprocess.annotations.raw.frame_to_world4D_base import FrameToWorldBase, rerun_frame_vis_results

# --------------------------------------------------------------------------------------
# Logger setup — writes to both console and code root log file
# --------------------------------------------------------------------------------------
_CODE_ROOT = Path(__file__).resolve().parents[4]  # WorldSGG/
_LOG_DIR = _CODE_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("world4d")
logger.setLevel(logging.DEBUG)

# Console handler (INFO level)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_ch)

# File handler (DEBUG level — captures everything)
_fh = logging.FileHandler(_LOG_DIR / "frame_to_world4D_annotations.log", mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_fh)


# --------------------------------------------------------------------------------------
# Label normalization (shared with corrected_world_bbox_generator.py)
# --------------------------------------------------------------------------------------

# GT compound names → short form
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}

# Reverse mapping: expanded GDino labels → GT short forms
GDINO_LABEL_TO_GT_LABEL = {
    "cabinet": "closet",
    "glass": "cup",
    "bottle": "cup",
    "notebook": "paper",
    "couch": "sofa",
    "camera": "phone",
}


def _normalize_label(label: str) -> str:
    """Normalize an object label to GT short-form space."""
    label = label.lower().strip()
    label = LABEL_NORMALIZE_MAP.get(label, label)
    label = GDINO_LABEL_TO_GT_LABEL.get(label, label)
    return label


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

    R_g = np.asarray(R_g, dtype=np.float32)  # (3,3)
    t_g = np.asarray(t_g, dtype=np.float32)  # (3,)
    s_g = float(s_g)

    floor_origin_world = t_g.astype(np.float32)

    # Floor-aligned basis
    t1 = R_g[:, 0]  # in-plane
    t2 = R_g[:, 2]  # in-plane
    n = R_g[:, 1]  # normal

    F = np.stack([t1, t2, n], axis=1)  # (3,3)
    R_align = F.T.astype(np.float32)  # WORLD -> FLOOR

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
# FrameToWorldAnnotations
#   - loads bridge PKL (GT + GDino merged 3D OBB annotations)
#   - generates 4D annotations with object permanence filling
#   - can visualize Pi3 points + floor mesh + 3D boxes via Rerun
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
            video_id_3d_bbox_predictions,
            visualize: bool = True,
    ) -> None:
        """
        Generate world 4D bbox annotations for a single video.

        Steps:
          1) Load bridge PKL (GT + GDino merged, FINAL-coords transforms already applied).
          2) Collect per-frame object labels (already filtered by bridge script).
          3) Enforce object permanence: every label in `all_labels` is present
             in every annotated frame (via interpolation, hold, or static copy).
          4) For static labels, union bboxes in FINAL coords across all frames.
          5) Save the resulting 4D annotations.
          6) Optionally visualize final 4D bboxes + points via Rerun.
        """
        logger.info(f"[world4d][{video_id}] Generating world SGG annotations (4D bboxes)")

        # ----------------------------------------------------------------------
        # Load / validate 3D bbox annotations
        # ----------------------------------------------------------------------
        if video_id_3d_bbox_predictions is None:
            video_3dgt = self.get_video_3d_bridge_annotations(video_id)
        else:
            video_3dgt = video_id_3d_bbox_predictions

        if video_3dgt is None:
            logger.warning(f"[world4d][{video_id}] No 3D bbox annotations found. Skipping.")
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
        frame_3dbb_map_world = video_3dgt.get("frames_final", None)
        if frame_3dbb_map_world is None or not frame_3dbb_map_world:
            logger.warning(f"[world4d][{video_id}] 3D bbox frames map is empty. Skipping.")
            return

        obb_bbox_frames = frame_3dbb_map_world.get("obb_bbox_frames", None)
        if obb_bbox_frames is None or not obb_bbox_frames:
            logger.warning(f"[world4d][{video_id}] 'obb_bbox_frames' missing or empty in 3D bbox data.")
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
            n = R_g0[:, 1]  # normal

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
        R_final = tf["R_world_to_final"]  # (3,3)
        t_final = tf["t_world_to_final"]  # (3,)
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
        # Collect labels from bridge PKL (already filtered by bridge script)
        # ----------------------------------------------------------------------
        all_labels = set()
        num_frames_with_objects = 0
        num_total_objects = 0

        for frame_name, frame_rec in obb_bbox_frames.items():
            objects = frame_rec.get("objects", [])
            if not objects:
                continue
            # Normalize labels in-place
            for obj in objects:
                lbl = obj.get("label", None)
                if lbl:
                    obj["label"] = _normalize_label(lbl)
            valid = [o for o in objects if o.get("label")]
            frame_rec["objects"] = valid
            if valid:
                num_frames_with_objects += 1
                num_total_objects += len(valid)
                for obj in valid:
                    all_labels.add(obj["label"])

        logger.info(
            f"[world4d][{video_id}] frames_with_objects={num_frames_with_objects}, "
            f"total_objects={num_total_objects}, "
            f"unique_labels={sorted(all_labels)}"
        )

        if not all_labels:
            logger.warning(f"[world4d][{video_id}] No object labels found. Skipping 4D generation.")
            return

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

        logger.info(
            f"[world4d][{video_id}] static_labels_in_3d={sorted(static_labels_in_3d)} "
            f"(from active={sorted(video_active_object_labels)})"
        )

        # ----------------------------------------------------------------------
        # Create world-4D annotations with object permanence
        # ----------------------------------------------------------------------
        # Sort frames chronologically by frame index (000123.png -> 123)
        frame_names_sorted = sorted(
            obb_bbox_frames.keys(),
            key=lambda fn: int(Path(fn).stem)
            if Path(fn).stem.isdigit()
            else Path(fn).stem,
        )

        # ==================================================================
        # PASS 1: Collect all detected objects per label per frame
        # ==================================================================
        # label -> list of (frame_idx, frame_name, obj_dict) for detected objects
        label_detections: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {
            lbl: [] for lbl in all_labels
        }
        frame_idx_map: Dict[str, int] = {}  # frame_name -> idx

        for fi, fname in enumerate(frame_names_sorted):
            frame_idx_map[fname] = fi
            frame_rec = obb_bbox_frames.get(fname, {})
            objects = frame_rec.get("objects", [])

            for obj in objects:
                lbl = obj.get("label", None)
                if not lbl or lbl not in all_labels:
                    continue
                # Keep first instance per label per frame
                if not any(d[1] == fname for d in label_detections[lbl]):
                    label_detections[lbl].append((fi, fname, obj))

        # ------------------------------------------------------------------
        # Per-video detailed debug log
        # ------------------------------------------------------------------
        logger.debug(f"=== World4D Pipeline Log: {video_id} ===")
        logger.debug(f"Total unique labels in video: {len(all_labels)}")
        logger.debug(f"Labels: {sorted(all_labels)}")
        logger.debug(f"Static labels: {sorted(static_labels_in_3d)}")
        dynamic_labels_in_3d = sorted(set(all_labels) - set(static_labels_in_3d))
        logger.debug(f"Dynamic labels: {dynamic_labels_in_3d}")
        logger.debug(f"Total frames: {len(frame_names_sorted)}")

        # Per-label detection summary
        for lbl in sorted(all_labels):
            det_frames = [d[1] for d in label_detections[lbl]]
            sources = []
            for d in label_detections[lbl]:
                src = d[2].get("source", "gt")
                sources.append(src)
            gt_count = sum(1 for s in sources if s == "gt")
            gdino_count = sum(1 for s in sources if s == "gdino")
            logger.debug(
                f"  {lbl}: detected in {len(det_frames)} frames "
                f"(GT={gt_count}, GDino={gdino_count})"
            )

        # Safety check
        for lbl in all_labels:
            if not label_detections[lbl]:
                raise ValueError(
                    f"[world4d][{video_id}] Label '{lbl}' in all_labels "
                    "has no actual bbox in any frame."
                )

        # ==================================================================
        # OBB decomposition + interpolation helpers
        # ==================================================================
        from scipy.spatial.transform import Rotation, Slerp

        def _obb_corners_to_params(corners: np.ndarray):
            """
            Decompose 8 OBB corners into (center, R, extent).

            corners: (8, 3) — first 4 are bottom face, next 4 are top face.
            Returns: center (3,), rotation Rotation obj, extent (3,)
            """
            center = corners.mean(axis=0)

            # Bottom face center and top face center for Y extent
            bottom_center = corners[:4].mean(axis=0)
            top_center = corners[4:].mean(axis=0)
            half_height = np.linalg.norm(top_center - bottom_center) / 2.0

            # XZ extent: use bottom face edges
            # Bottom corners: 0,1,2,3 in order from minAreaRect
            edge1 = corners[1] - corners[0]
            edge2 = corners[3] - corners[0]

            len1 = np.linalg.norm(edge1)
            len2 = np.linalg.norm(edge2)

            if len1 < 1e-8 or len2 < 1e-8:
                return center, Rotation.identity(), np.array([0.01, half_height, 0.01])

            half_w = len1 / 2.0
            half_d = len2 / 2.0

            # Build rotation from edges (X = edge1 dir, Y = up, Z = edge2 dir)
            x_axis = edge1 / len1
            z_axis = edge2 / len2
            y_axis = np.cross(z_axis, x_axis)
            y_norm = np.linalg.norm(y_axis)
            if y_norm < 1e-8:
                y_axis = (top_center - bottom_center)
                y_norm = np.linalg.norm(y_axis)
                if y_norm < 1e-8:
                    y_axis = np.array([0.0, 1.0, 0.0])
                else:
                    y_axis = y_axis / y_norm
            else:
                y_axis = y_axis / y_norm

            rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3, 3)
            # Ensure valid rotation (fix determinant)
            if np.linalg.det(rot_mat) < 0:
                rot_mat[:, 2] *= -1

            try:
                rot = Rotation.from_matrix(rot_mat)
            except Exception:
                rot = Rotation.identity()

            extent = np.array([half_w, half_height, half_d])
            return center, rot, extent

        def _params_to_obb_corners(center, rot, extent):
            """
            Reconstruct 8 OBB corners from (center, rotation, extent).

            Returns: (8, 3) corners — bottom 4 then top 4.
            """
            half_w, half_h, half_d = extent

            # Local corners (centered at origin)
            local = np.array([
                [-half_w, -half_h, -half_d],
                [+half_w, -half_h, -half_d],
                [+half_w, -half_h, +half_d],
                [-half_w, -half_h, +half_d],
                [-half_w, +half_h, -half_d],
                [+half_w, +half_h, -half_d],
                [+half_w, +half_h, +half_d],
                [-half_w, +half_h, +half_d],
            ], dtype=np.float32)

            R_mat = rot.as_matrix()
            world = (R_mat @ local.T).T + center[None, :]
            return world.astype(np.float32)

        def _interpolate_obb(
                corners_prev: np.ndarray,
                corners_next: np.ndarray,
                t: float,
        ) -> np.ndarray:
            """
            Interpolate between two OBB corner sets using center+R+extent decomposition.
            t in [0, 1]: 0 = prev, 1 = next.
            """
            c_prev, r_prev, e_prev = _obb_corners_to_params(corners_prev)
            c_next, r_next, e_next = _obb_corners_to_params(corners_next)

            # Lerp center and extent
            center_interp = (1.0 - t) * c_prev + t * c_next
            extent_interp = (1.0 - t) * e_prev + t * e_next

            # Slerp rotation
            try:
                key_rots = Rotation.concatenate([r_prev, r_next])
                slerp_fn = Slerp([0.0, 1.0], key_rots)
                rot_interp = slerp_fn(t)
            except Exception:
                # Fallback: just use prev rotation
                rot_interp = r_prev

            return _params_to_obb_corners(center_interp, rot_interp, extent_interp)

        # Helper to extract FINAL-coord corners from an object
        def _get_corners_final(obj: Dict[str, Any]) -> Optional[np.ndarray]:
            # Bridge PKL objects have corners_final at top level
            cf = obj.get("corners_final", None)
            if cf is not None:
                c = np.asarray(cf, dtype=np.float32)
                if c.shape == (8, 3):
                    return c
            # Also check nested obb_final (from prior fill passes)
            obb_final = obj.get("obb_final", None)
            if obb_final and obb_final.get("corners_final") is not None:
                c = np.asarray(obb_final["corners_final"], dtype=np.float32)
                if c.shape == (8, 3):
                    return c
            return None

        # Helper to clone an object and attach 4D metadata
        def _clone_with_meta(
                base_obj: Dict[str, Any],
                *,
                filled: bool,
                fill_method: str,
                source_frame: str,
                target_frame: str,
        ) -> Dict[str, Any]:
            new_obj = dict(base_obj)
            new_obj["world4d_filled"] = bool(filled)
            new_obj["world4d_fill_method"] = fill_method
            new_obj["world4d_source_frame"] = source_frame
            new_obj["world4d_frame"] = target_frame
            return new_obj

        # ==================================================================
        # PASS 2: Fill all labels in all frames
        # ==================================================================
        frames_filled_world: Dict[str, Dict[str, Any]] = {}
        n_detected = 0
        n_static_copy = 0
        n_interpolated = 0
        n_hold = 0

        for fi, fname in enumerate(frame_names_sorted):
            frame_rec = frame_3dbb_map_world.get(fname, {})
            objects = frame_rec.get("objects", [])

            # Map label -> object for this frame
            label_to_obj_current: Dict[str, Dict[str, Any]] = {}
            for obj in objects:
                lbl = obj.get("label", None)
                if not lbl:
                    continue
                if lbl not in label_to_obj_current:
                    label_to_obj_current[lbl] = obj

            filled_objects: List[Dict[str, Any]] = []

            for lbl in sorted(all_labels):
                is_static = lbl in static_labels_in_3d

                if lbl in label_to_obj_current:
                    # ---- DETECTED (GT or GDino) ----
                    base_obj = label_to_obj_current[lbl]
                    filled_obj = _clone_with_meta(
                        base_obj,
                        filled=False,
                        fill_method="detected",
                        source_frame=fname,
                        target_frame=fname,
                    )
                    n_detected += 1

                elif is_static:
                    # ---- STATIC COPY (last/first known) ----
                    detections = label_detections[lbl]
                    # Find nearest previous detection
                    prev_det = None
                    for d_fi, d_fn, d_obj in reversed(detections):
                        if d_fi <= fi:
                            prev_det = (d_fi, d_fn, d_obj)
                            break
                    if prev_det is None:
                        # Use first known
                        prev_det = detections[0]

                    _, src_frame, base_obj = prev_det
                    filled_obj = _clone_with_meta(
                        base_obj,
                        filled=True,
                        fill_method="static_copy",
                        source_frame=src_frame,
                        target_frame=fname,
                    )
                    n_static_copy += 1

                else:
                    # ---- DYNAMIC: INTERPOLATE between nearest detections ----
                    detections = label_detections[lbl]

                    # Find prev and next detections
                    prev_det = None
                    next_det = None
                    for d_fi, d_fn, d_obj in detections:
                        if d_fi <= fi:
                            prev_det = (d_fi, d_fn, d_obj)
                        elif next_det is None and d_fi > fi:
                            next_det = (d_fi, d_fn, d_obj)
                            break

                    if prev_det is not None and next_det is not None:
                        # Interpolate between prev and next in FINAL coords
                        prev_corners = _get_corners_final(prev_det[2])
                        next_corners = _get_corners_final(next_det[2])

                        if prev_corners is not None and next_corners is not None:
                            span = next_det[0] - prev_det[0]
                            t = (fi - prev_det[0]) / span if span > 0 else 0.5
                            interp_corners = _interpolate_obb(prev_corners, next_corners, t)

                            filled_obj = _clone_with_meta(
                                prev_det[2],
                                filled=True,
                                fill_method="interpolation",
                                source_frame=f"{prev_det[1]}..{next_det[1]}",
                                target_frame=fname,
                            )
                            # Store interpolated corners in FINAL coords
                            filled_obj["obb_final"] = {
                                "corners_final": interp_corners,
                            }
                            n_interpolated += 1
                        else:
                            # One side has no corners; hold the one that does
                            hold_det = prev_det if prev_corners is not None else next_det
                            if hold_det is None:
                                continue
                            filled_obj = _clone_with_meta(
                                hold_det[2],
                                filled=True,
                                fill_method="hold_prev" if hold_det == prev_det else "hold_next",
                                source_frame=hold_det[1],
                                target_frame=fname,
                            )
                            n_hold += 1

                    elif prev_det is not None:
                        # Only prev exists — hold
                        filled_obj = _clone_with_meta(
                            prev_det[2],
                            filled=True,
                            fill_method="hold_prev",
                            source_frame=prev_det[1],
                            target_frame=fname,
                        )
                        n_hold += 1

                    elif next_det is not None:
                        # Only next exists — hold
                        filled_obj = _clone_with_meta(
                            next_det[2],
                            filled=True,
                            fill_method="hold_next",
                            source_frame=next_det[1],
                            target_frame=fname,
                        )
                        n_hold += 1

                    else:
                        # No detections at all for this label (shouldn't happen)
                        continue

                # Ensure label is consistent
                filled_obj["label"] = lbl

                # Ensure obb_final exists with corners_final
                # For detected/held/copied objects, corners_final comes from bridge PKL
                if "obb_final" not in filled_obj:
                    cf = _get_corners_final(filled_obj)
                    if cf is not None:
                        filled_obj["obb_final"] = {"corners_final": cf}

                # Default color for AFTER visualization
                if "color_after" not in filled_obj:
                    filled_obj["color_after"] = filled_obj.get("color", [255, 230, 80])

                filled_objects.append(filled_obj)

            frames_filled_world[fname] = {"objects": filled_objects}

        # Logging: filling summary
        logger.info(
            f"[world4d][{video_id}] Filling: detected={n_detected}, "
            f"static_copy={n_static_copy}, interpolated={n_interpolated}, "
            f"hold={n_hold}, "
            f"total={n_detected + n_static_copy + n_interpolated + n_hold}"
        )
        logger.debug(f"--- Filling Summary [{video_id}] ---")
        logger.debug(f"  Detected (GT+GDino): {n_detected}")
        logger.debug(f"  Static copy:         {n_static_copy}")
        logger.debug(f"  Interpolated:        {n_interpolated}")
        logger.debug(f"  Hold (prev/next):    {n_hold}")

        # ----------------------------------------------------------------------
        # Static-object OBB union (UNION IN FINAL COORDS, WORLD UNCHANGED)
        # Uses cv2.minAreaRect on XZ projection of pooled corners
        # ----------------------------------------------------------------------

        def _compute_obb_union_final(all_corners: np.ndarray) -> np.ndarray:
            """
            Compute tightest OBB union from pooled corners in FINAL coords.

            all_corners: (N, 3) points in FINAL (floor-aligned) coords.
            Returns: (8, 3) OBB corners — bottom 4 then top 4.
            """
            # Project onto XZ plane (Y is floor normal in FINAL coords)
            pts_xz = all_corners[:, [0, 2]].astype(np.float32)
            rect = cv2.minAreaRect(pts_xz[:, None, :])
            box_2d = cv2.boxPoints(rect)  # (4, 2) in XZ

            y_min = float(all_corners[:, 1].min())
            y_max = float(all_corners[:, 1].max())

            # Build 8 corners: bottom face (y_min) then top face (y_max)
            corners = np.zeros((8, 3), dtype=np.float32)
            for i in range(4):
                corners[i] = [box_2d[i][0], y_min, box_2d[i][1]]
                corners[i + 4] = [box_2d[i][0], y_max, box_2d[i][1]]

            return corners

        static_union_map_final: Dict[str, np.ndarray] = {}

        # 1) Compute OBB union (FINAL coords) per static label
        for lbl in static_labels_in_3d:
            all_corners_list_final: List[np.ndarray] = []

            for fname in frame_names_sorted:
                frame_rec = frames_filled_world[fname]
                for obj in frame_rec["objects"]:
                    if obj.get("label") != lbl:
                        continue

                    obb_final = obj.get("obb_final", None)
                    if not obb_final:
                        continue

                    corners_final = np.asarray(
                        obb_final.get("corners_final", []), dtype=np.float32
                    )
                    if corners_final.size == 0:
                        continue

                    all_corners_list_final.append(corners_final)

            if not all_corners_list_final:
                logger.warning(
                    f"[world4d][{video_id}] static label '{lbl}' has no "
                    "obb_final corners; skipping union."
                )
            else:
                all_pts_final = np.concatenate(all_corners_list_final, axis=0)
                union_corners_final = _compute_obb_union_final(all_pts_final)
                static_union_map_final[lbl] = union_corners_final

        if static_union_map_final:
            logger.info(
                f"[world4d][{video_id}] Applying static OBB union in FINAL coords "
                f"for {len(static_union_map_final)} labels."
            )

            for fname in frame_names_sorted:
                frame_rec = frames_filled_world[fname]
                for obj in frame_rec["objects"]:
                    lbl = obj.get("label", None)
                    if lbl not in static_union_map_final:
                        continue

                    union_corners_final = static_union_map_final[lbl]

                    if "obb_final" not in obj:
                        obj["obb_final"] = {}
                    obj["obb_final"]["corners_final"] = union_corners_final

                    # NOTE: WORLD-space boxes remain as originally constructed.


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

            points_world = np.asarray(P["points"], dtype=np.float32)[idxs]  # (S,H,W,3)
            conf_world = (np.asarray(P["conf"], dtype=np.float32)[idxs]
                          if P["conf"] is not None else None)  # (S,H,W) or None
            colors_world = (np.asarray(P["colors"], dtype=np.uint8)[idxs]
                            if P["colors"] is not None else None)  # (S,H,W,3) or None
            cameras_world = (np.asarray(P["camera_poses"], dtype=np.float32)[idxs]
                             if P["camera_poses"] is not None else None)  # (S,4,4)/(S,3,4) or None

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
                vis_mode="after",  # change to "both" if you want BEFORE/AFTER side-by-side
            )

        # ----------------------------------------------------------------------
        # Save world-4D annotations
        # ----------------------------------------------------------------------
        out_4d_path = self.bbox_4d_root_dir / f"{video_id[:-4]}.pkl"
        world4d_annotations = {
            "video_id": video_id,
            "frames": frames_filled_world,  # frame_name -> {objects: [...]}
            "frame_names": frame_names_sorted,  # ordered list of frame_name strings
            "all_labels": sorted(all_labels),
            "meta": {
                "num_frames": len(frame_names_sorted),
                "labels": sorted(all_labels),
            },
            # --- Pass-through from bridge PKL ---
            "camera_poses": frame_3dbb_map_world.get("camera_poses", None),  # (S,4,4) FINAL coords
            "frame_stems": frame_3dbb_map_world.get("frame_stems", None),    # list of stem strings
            "floor_mesh": {
                "gv": gv,
                "gf": gf,
                "gc": gc,
            } if floor is not None else None,
            "global_floor_sim": gfs,   # {"s", "R", "t"}
            "world_to_final": video_3dgt.get("world_to_final", None),  # {"origin_world", "A_world_to_final"}
            "floor_final": frame_3dbb_map_world.get("floor", None),    # floor mesh already in FINAL coords
        }

        with open(out_4d_path, "wb") as f:
            pickle.dump(world4d_annotations, f)

        logger.info(f"[world4d][{video_id}] Saved world-4D bbox annotations to {out_4d_path}")

    # ----------------------------------------------------------------------------------
    # Generate World4D from Firebase annotations
    # ----------------------------------------------------------------------------------

    def generate_world4d_from_firebase(
            self,
            video_id: str,
            firebase_service,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate world4d annotations from Firebase data.

        Priority:
          1) worldframe_obb/final_alignments/{video_id}/latest  (pre-computed)
          2) worldframe_obb/edited/{video_id}/{frame}/latest    (edited per-frame)
          3) worldframe_obb/world/{video_id}/{frame}            (original per-frame)
          4) local PKL fallback (and if only WORLD corners exist, compute FINAL using global_floor_sim)

        Semantics match:
          - object permanence ONLY for static labels
          - dynamic objects appear only in frames they exist
          - static union computed in FINAL coords; WORLD data left untouched
        """
        logger.info(f"[world4d][{video_id}] Generating from Firebase annotations")

        video_id_clean = video_id.replace(".mp4", "").replace(".", "_")

        # Ensure we have static/dynamic classification
        self.fetch_stored_active_objects_in_video(video_id)

        video_active_object_labels = self.video_id_active_objects_annotations_map.get(video_id, [])
        video_reasoned_active_object_labels = self.video_id_active_objects_b_reasoned_map.get(video_id, [])

        non_moving_objects = ["floor", "sofa", "couch", "bed", "doorway", "table", "chair"]
        video_dynamic_object_labels = [
            obj for obj in video_reasoned_active_object_labels
            if obj not in non_moving_objects
        ]
        video_static_object_labels = [
            obj for obj in video_active_object_labels
            if obj not in video_dynamic_object_labels
        ]

        # --- 1) Try final_alignments first ---
        final_alignment_path = f"worldframe_obb/final_alignments/{video_id_clean}/latest"
        final_alignment_data = firebase_service.get_data(final_alignment_path)
        if final_alignment_data and final_alignment_data.get("frames"):
            print(f"[world4d][{video_id}] Using final_alignments data")
            return self._generate_world4d_from_final_alignment(
                video_id=video_id,
                final_alignment_data=final_alignment_data,
                video_static_object_labels=video_static_object_labels,
            )

        # --- 2) Need local PKL to know which frames to query ---
        video_3dgt = self.get_video_3d_annotations(video_id)
        if video_3dgt is None:
            print(f"[world4d][{video_id}] No local 3D annotations found")
            return None

        frame_3dbb_map_world = video_3dgt.get("frames", {})
        if not frame_3dbb_map_world:
            print(f"[world4d][{video_id}] No frame data in 3D annotations")
            return None

        # Sort frames like your main pipeline
        frame_names_sorted = sorted(
            frame_3dbb_map_world.keys(),
            key=lambda fn: int(Path(fn).stem) if Path(fn).stem.isdigit() else Path(fn).stem,
        )

        # Optional: compute WORLD->FINAL for local fallback (only if needed)
        floor = None
        global_floor_sim = None
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

        tf = compute_final_world_transform(floor=floor, global_floor_sim=global_floor_sim)
        R_final = tf["R_world_to_final"]
        t_final = tf["t_world_to_final"]

        def _to_list_corners(corners_any: Any) -> List[List[float]]:
            if corners_any is None:
                return []
            if isinstance(corners_any, np.ndarray):
                return corners_any.astype(np.float32).tolist()
            # assume list-like
            return np.asarray(corners_any, dtype=np.float32).tolist()

        def _obj_from_corners(label: str, corners_final: Any, color: Any, object_id: str = None) -> Dict[str, Any]:
            corners_list = _to_list_corners(corners_final)
            if len(corners_list) > 0:
                c = np.asarray(corners_list, dtype=np.float32)
                center = c.mean(axis=0).tolist()
            else:
                center = [0.0, 0.0, 0.0]
            obj = {
                "label": label,
                "obb_final": {"corners_final": corners_list},
                "corners_final": corners_list,  # keep both for compatibility
                "center": center,
                "color": color if color is not None else [255, 180, 0],
            }
            if object_id:
                obj["object_id"] = object_id
            return obj

        print(f"[world4d][{video_id}] Checking Firebase per-frame data for {len(frame_names_sorted)} frames")

        frames_from_firebase: Dict[str, Dict[str, Any]] = {}
        firebase_frames_found = 0

        for frame_name in frame_names_sorted:
            frame_stem = Path(frame_name).stem

            # 2a) Edited annotations first
            edited_path = f"worldframe_obb/edited/{video_id_clean}/{frame_stem}/latest"
            edited_data = firebase_service.get_data(edited_path)

            objects: List[Dict[str, Any]] = []
            if edited_data and edited_data.get("annotations"):
                for ann in edited_data["annotations"]:
                    objects.append(
                        _obj_from_corners(
                            label=ann.get("object_label", "Object"),
                            corners_final=ann.get("bbox_corners", []),
                            color=ann.get("color", [255, 180, 0]),
                            object_id=ann.get("object_id"),
                        )
                    )
                frames_from_firebase[frame_name] = {"objects": objects, "relationships": []}
                firebase_frames_found += 1
                continue

            # 2b) Original worldframe_obb/world
            original_path = f"worldframe_obb/world/{video_id_clean}/{frame_stem}"
            original_data = firebase_service.get_data(original_path)

            if original_data and original_data.get("objects"):
                objects_raw = original_data["objects"]
                if isinstance(objects_raw, dict):
                    # When stored as dict, keys are object_ids
                    for obj_id, obj_raw in objects_raw.items():
                        objects.append(
                            _obj_from_corners(
                                label=obj_raw.get("object_label", "Object"),
                                corners_final=obj_raw.get("bbox_corners", []),
                                color=obj_raw.get("color", [255, 180, 0]),
                                object_id=obj_raw.get("object_id", obj_id),
                            )
                        )
                else:
                    for obj_raw in objects_raw:
                        objects.append(
                            _obj_from_corners(
                                label=obj_raw.get("object_label", "Object"),
                                corners_final=obj_raw.get("bbox_corners", []),
                                color=obj_raw.get("color", [255, 180, 0]),
                                object_id=obj_raw.get("object_id"),
                            )
                        )

                # Fetch relationships for this frame
                relationships = original_data.get("relationships", [])
                if not isinstance(relationships, list):
                    relationships = []

                frames_from_firebase[frame_name] = {"objects": objects, "relationships": relationships}
                firebase_frames_found += 1
                continue

            # 2c) Local PKL fallback
            local_frame = frame_3dbb_map_world.get(frame_name, {})
            local_objects = local_frame.get("objects", [])

            for obj_raw in local_objects:
                lbl = obj_raw.get("label", "Object")
                col = obj_raw.get("color", [255, 180, 0])

                corners_final = None

                # Prefer existing obb_final if present
                obb_final = obj_raw.get("obb_final", None)
                if isinstance(obb_final, dict) and obb_final.get("corners_final") is not None:
                    corners_final = obb_final.get("corners_final")

                # Compute FINAL from WORLD OBB corners
                if corners_final is None:
                    obb_world = obj_raw.get("obb_floor_parallel", None)
                    if isinstance(obb_world, dict) and obb_world.get("corners_world") is not None:
                        corners_world = np.asarray(obb_world["corners_world"], dtype=np.float32)
                        if corners_world.size > 0:
                            corners_final = (R_final @ corners_world.T).T + t_final[None, :]

                objects.append(_obj_from_corners(lbl, corners_final if corners_final is not None else [], col))

            frames_from_firebase[frame_name] = {"objects": objects, "relationships": []}

        print(f"[world4d][{video_id}] Loaded {firebase_frames_found} frames from Firebase (rest from local fallback)")

        return self._apply_world4d_filling(
            video_id=video_id,
            frames_data=frames_from_firebase,
            frame_names_sorted=frame_names_sorted,
            video_static_object_labels=video_static_object_labels,
            include_relationships=True,
        )

    def _generate_world4d_from_final_alignment(
            self,
            video_id: str,
            final_alignment_data: Dict[str, Any],
            video_static_object_labels: List[str],
    ) -> Dict[str, Any]:
        """
        Normalize final_alignment frames to:
          frames_data[<frame>.png] = {"objects": [ {label, obb_final:{corners_final}, ...}, ... ]}

        Then apply the SAME filling + static union logic via _apply_world4d_filling.
        """
        frames_raw = final_alignment_data.get("frames", {})
        if not isinstance(frames_raw, dict) or not frames_raw:
            return {
                "video_id": video_id,
                "frames": {},
                "frame_names": [],
                "all_labels": [],
                "static_labels": [],
                "meta": {"num_frames": 0},
            }

        def _canon_frame_name(k: str) -> str:
            st = Path(str(k)).stem
            return f"{st}.png"

        def _sort_key(fn: str):
            st = Path(fn).stem
            return int(st) if st.isdigit() else st

        def _to_list_corners(corners_any: Any) -> List[List[float]]:
            if corners_any is None:
                return []
            if isinstance(corners_any, np.ndarray):
                return corners_any.astype(np.float32).tolist()
            return np.asarray(corners_any, dtype=np.float32).tolist()

        def _obj_norm(obj: Dict[str, Any]) -> Dict[str, Any]:
            lbl = obj.get("label", "Object")

            # accept either "corners" or "corners_final" or nested obb_final
            corners = None
            if isinstance(obj.get("obb_final"), dict) and obj["obb_final"].get("corners_final") is not None:
                corners = obj["obb_final"]["corners_final"]
            elif obj.get("corners_final") is not None:
                corners = obj.get("corners_final")
            elif obj.get("corners") is not None:
                corners = obj.get("corners")

            corners_list = _to_list_corners(corners)

            if len(corners_list) > 0:
                c = np.asarray(corners_list, dtype=np.float32)
                center = c.mean(axis=0).tolist()
            else:
                center = obj.get("center", [0.0, 0.0, 0.0])

            color = obj.get("color", [255, 180, 0])
            object_id = obj.get("object_id") or obj.get("id")

            out = dict(obj)
            out["label"] = lbl
            out["obb_final"] = {"corners_final": corners_list}
            out["corners_final"] = corners_list
            out["center"] = center
            out["color"] = color
            if object_id:
                out["object_id"] = object_id
            return out

        normalized_frames: Dict[str, Dict[str, Any]] = {}

        # preserve order of keys but sort numerically like main pipeline
        canon_names = [_canon_frame_name(k) for k in frames_raw.keys()]
        frame_names_sorted = sorted(set(canon_names), key=_sort_key)

        for k, v in frames_raw.items():
            fname = _canon_frame_name(k)

            objects: List[Dict[str, Any]] = []

            if isinstance(v, list):
                # list of objects
                for obj in v:
                    if isinstance(obj, dict):
                        objects.append(_obj_norm(obj))
            elif isinstance(v, dict):
                # could be {"objects":[...]} or dict-of-objects
                if isinstance(v.get("objects"), list):
                    for obj in v["objects"]:
                        if isinstance(obj, dict):
                            objects.append(_obj_norm(obj))
                else:
                    # dict-of-objects
                    for obj in v.values():
                        if isinstance(obj, dict):
                            objects.append(_obj_norm(obj))

            normalized_frames[fname] = {"objects": objects, "relationships": []}

        return self._apply_world4d_filling(
            video_id=video_id,
            frames_data=normalized_frames,
            frame_names_sorted=frame_names_sorted,
            video_static_object_labels=video_static_object_labels,
            include_relationships=False,  # final_alignments don't have relationships yet
        )

    def _apply_world4d_filling(
            self,
            video_id: str,
            frames_data: Dict[str, Dict[str, Any]],
            frame_names_sorted: List[str],
            video_static_object_labels: List[str],
            include_relationships: bool = False,
    ) -> Dict[str, Any]:
        """
        EXACT semantic match to your main pipeline:

          - all_labels collected from observed objects
          - static_labels_in_3d = static_labels ) all_labels
          - for each frame:
              include objects that exist
              fill ONLY static labels using last_seen else first_seen
              skip missing dynamic labels
          - compute static union AABB in FINAL coords and overwrite ONLY FINAL corners
            (do not modify any world-space data if present)
        """

        def _sort_key(fn: str):
            st = Path(fn).stem
            return int(st) if st.isdigit() else st

        # Ensure deterministic frame ordering
        frame_names_sorted = sorted(list(frame_names_sorted), key=_sort_key)

        def _get_label(obj: Dict[str, Any]) -> Optional[str]:
            return obj.get("label", None)

        def _get_corners_final(obj: Dict[str, Any]) -> List[List[float]]:
            if isinstance(obj.get("obb_final"), dict) and obj["obb_final"].get("corners_final") is not None:
                return obj["obb_final"]["corners_final"]
            if obj.get("corners_final") is not None:
                return obj["corners_final"]
            return []

        def _set_corners_final(obj: Dict[str, Any], corners_list: List[List[float]]) -> None:
            if not isinstance(obj.get("obb_final"), dict):
                obj["obb_final"] = {}
            obj["obb_final"]["corners_final"] = corners_list
            obj["corners_final"] = corners_list  # keep both for compatibility

            if len(corners_list) > 0:
                c = np.asarray(corners_list, dtype=np.float32)
                obj["center"] = c.mean(axis=0).tolist()
            else:
                obj["center"] = [0.0, 0.0, 0.0]

        def _deepish_clone_obj(base_obj: Dict[str, Any]) -> Dict[str, Any]:
            # Avoid shared nested references that later union-overwrites would mutate across frames
            out = dict(base_obj)

            # Copy obb_final dict + corners
            if isinstance(base_obj.get("obb_final"), dict):
                out["obb_final"] = dict(base_obj["obb_final"])
                if out["obb_final"].get("corners_final") is not None:
                    out["obb_final"]["corners_final"] = np.asarray(
                        out["obb_final"]["corners_final"], dtype=np.float32
                    ).tolist()

            # Copy top-level corners_final too
            if base_obj.get("corners_final") is not None:
                out["corners_final"] = np.asarray(base_obj["corners_final"], dtype=np.float32).tolist()

            return out

        # ----------------------------------------------------------------------
        # Collect all labels from observed objects
        # ----------------------------------------------------------------------
        # Determine which labels are static (strictly based on input)
        # ----------------------------------------------------------------------
        all_labels = set()
        for fname in frame_names_sorted:
            for obj in frames_data.get(fname, {}).get("objects", []):
                lbl = _get_label(obj)
                if lbl:
                    all_labels.add(lbl)

        static_labels_in_3d = [lbl for lbl in video_static_object_labels if lbl in all_labels]

        print(f"[world4d][{video_id}] All labels: {sorted(all_labels)}")
        print(f"[world4d][{video_id}] Static labels in 3D (filled): {sorted(static_labels_in_3d)}")

        if not all_labels:
            return {
                "video_id": video_id,
                "frames": {},
                "frame_names": frame_names_sorted,
                "all_labels": [],
                "static_labels": [],
                "meta": {"num_frames": len(frame_names_sorted)},
            }

        # ----------------------------------------------------------------------
        # First-seen source per label (must exist for every label in all_labels)
        # ----------------------------------------------------------------------
        label_first_source: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for fname in frame_names_sorted:
            objects = frames_data.get(fname, {}).get("objects", [])
            for obj in objects:
                lbl = _get_label(obj)
                if lbl and lbl not in label_first_source:
                    label_first_source[lbl] = (fname, obj)

        for lbl in all_labels:
            if lbl not in label_first_source:
                raise ValueError(
                    f"[world4d][{video_id}] Label '{lbl}' appears in all_labels but has no source frame."
                )

        # ----------------------------------------------------------------------
        # Object permanence fill (STATIC ONLY), dynamic missing => skip
        # Smart object_id assignment:
        #   - Visible objects: keep their original object_id from worldframe_obb
        #   - Filled objects: assign new object_id by incrementing from max in frame
        # ----------------------------------------------------------------------
        last_seen: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        frames_filled: Dict[str, Dict[str, Any]] = {}

        def _extract_obj_id_number(obj_id: str) -> int:
            """Extract the numeric part from object_id like 'obj_5' -> 5"""
            if not obj_id or not isinstance(obj_id, str):
                return -1
            if obj_id.startswith("obj_"):
                try:
                    return int(obj_id[4:])
                except ValueError:
                    return -1
            return -1

        for fname in frame_names_sorted:
            objects = frames_data.get(fname, {}).get("objects", [])

            # Map first instance per label, and track max object_id in this frame
            label_to_obj_current: Dict[str, Dict[str, Any]] = {}
            max_obj_id_num = -1

            for obj in objects:
                lbl = _get_label(obj)
                if lbl and lbl not in label_to_obj_current:
                    label_to_obj_current[lbl] = obj

                # Track max object_id number in visible objects
                obj_id = obj.get("object_id", "")
                obj_num = _extract_obj_id_number(obj_id)
                if obj_num > max_obj_id_num:
                    max_obj_id_num = obj_num

            filled_objects: List[Dict[str, Any]] = []
            next_obj_id_num = max_obj_id_num + 1  # For filled objects

            for lbl in sorted(all_labels):
                is_static = lbl in static_labels_in_3d

                if lbl in label_to_obj_current:
                    # Object is visible in this frame - keep original object_id
                    base_obj = label_to_obj_current[lbl]
                    last_seen[lbl] = (fname, base_obj)

                    new_obj = _deepish_clone_obj(base_obj)
                    new_obj["world4d_filled"] = False
                    new_obj["world4d_source_frame"] = fname
                    new_obj["world4d_frame"] = fname
                    new_obj["label"] = lbl
                    # object_id is preserved from base_obj via _deepish_clone_obj

                elif is_static:
                    # Static object NOT visible but needs to be filled
                    if lbl in last_seen:
                        src_frame, base_obj = last_seen[lbl]
                    else:
                        src_frame, base_obj = label_first_source[lbl]

                    new_obj = _deepish_clone_obj(base_obj)
                    new_obj["world4d_filled"] = True
                    new_obj["world4d_source_frame"] = src_frame
                    new_obj["world4d_frame"] = fname
                    new_obj["label"] = lbl

                    # Assign new object_id for filled object
                    new_obj["object_id"] = f"obj_{next_obj_id_num}"
                    next_obj_id_num += 1

                else:
                    # dynamic and missing => do NOT fill
                    continue

                # Ensure corners_final is consistently stored and center is correct
                corners = _get_corners_final(new_obj)
                corners_list = np.asarray(corners, dtype=np.float32).tolist() if corners else []
                _set_corners_final(new_obj, corners_list)

                filled_objects.append(new_obj)

            # Include relationships if available
            frame_relationships = []
            if include_relationships:
                frame_relationships = frames_data.get(fname, {}).get("relationships", [])

            frames_filled[fname] = {"objects": filled_objects, "relationships": frame_relationships}

        # ----------------------------------------------------------------------
        # Static OBB union: UNION IN FINAL COORDS, WORLD UNCHANGED
        # Uses cv2.minAreaRect on XZ projection of pooled corners
        # ----------------------------------------------------------------------
        def _compute_obb_union_final_filling(all_corners: np.ndarray) -> List[List[float]]:
            pts_xz = all_corners[:, [0, 2]].astype(np.float32)
            rect = cv2.minAreaRect(pts_xz[:, None, :])
            box_2d = cv2.boxPoints(rect)  # (4, 2) in XZ
            y_min = float(all_corners[:, 1].min())
            y_max = float(all_corners[:, 1].max())
            corners = []
            for i in range(4):
                corners.append([float(box_2d[i][0]), y_min, float(box_2d[i][1])])
            for i in range(4):
                corners.append([float(box_2d[i][0]), y_max, float(box_2d[i][1])])
            return corners

        static_union_map_final: Dict[str, List[List[float]]] = {}

        for lbl in static_labels_in_3d:
            all_corners_list: List[np.ndarray] = []
            for fname in frame_names_sorted:
                for obj in frames_filled.get(fname, {}).get("objects", []):
                    if obj.get("label") != lbl:
                        continue
                    corners = _get_corners_final(obj)
                    if not corners:
                        continue
                    c = np.asarray(corners, dtype=np.float32)
                    if c.shape != (8, 3):
                        continue
                    all_corners_list.append(c)

            if not all_corners_list:
                print(
                    f"[world4d][{video_id}] WARNING: static label '{lbl}' has no valid corners_final; skipping union.")
                continue

            all_pts = np.concatenate(all_corners_list, axis=0)  # (N*8,3)
            static_union_map_final[lbl] = _compute_obb_union_final_filling(all_pts)

        if static_union_map_final:
            print(
                f"[world4d][{video_id}] Applying static OBB union in FINAL coords for {len(static_union_map_final)} labels.")
            for fname in frame_names_sorted:
                for obj in frames_filled.get(fname, {}).get("objects", []):
                    lbl = obj.get("label")
                    if lbl not in static_union_map_final:
                        continue

                    # Overwrite ONLY FINAL coords
                    _set_corners_final(obj, static_union_map_final[lbl])

                    # NOTE: world-space fields are intentionally untouched.

        result = {
            "video_id": video_id,
            "frames": frames_filled,
            "frame_names": frame_names_sorted,
            "all_labels": sorted(all_labels),
            "static_labels": static_labels_in_3d,
            "meta": {
                "num_frames": len(frame_names_sorted),
                "total_objects_per_frame": {
                    fname: len(frames_filled.get(fname, {}).get("objects", []))
                    for fname in frame_names_sorted
                },
            },
        }

        logger.info(f"[world4d][{video_id}] Generated 4D annotations for {len(frame_names_sorted)} frames")
        return result

    # ----------------------------------------------------------------------------------
    # Visualization: original Pi3 points + bridge 3D bboxes
    # ----------------------------------------------------------------------------------

    def visualize_original_results(self, video_id: str, vis_mode: str = "after") -> None:
        """
        Visualize original Pi3 points + 3D bounding boxes for a video.

        1) Loads raw Pi3 points (WORLD coords) from predictions.npz.
        2) Loads bridge PKL for floor mesh, global_floor_sim, and
           pre-computed corners_final.
        3) Computes WORLD -> FINAL transform for points/cameras/floor.
        4) Uses pre-computed corners_final from bridge PKL for bbox display.
        5) Calls `rerun_frame_vis_results` with BEFORE (world) and AFTER (final).
        """
        P = self._load_original_points_for_video(video_id)

        points_world = P["points"]  # (S,H,W,3)
        conf_world = P["conf"]  # (S,H,W) or None
        stems = P["frame_stems"]  # ["000123", ...]
        colors_world = P["colors"]  # (S,H,W,3)
        camera_poses_world = P["camera_poses"]  # (S,4,4) or (S,3,4) or None

        # Optional floor mesh + similarity transform from bbox_3d pkl
        floor = None
        global_floor_sim = None
        frame_3dbb_map_world = None
        obb_bbox_frames = None
        floor_vertices_before = None
        floor_vertices_after = None
        floor_axes_before = None
        floor_axes_after = None
        floor_origin_world = None
        floor_faces = None
        floor_kwargs = None

        video_3dgt = self.get_video_3d_bridge_annotations(video_id)
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

            # 3D bboxes per frame (from bridge PKL: already have corners_final)
            frames_final_data = video_3dgt.get("frames_final", None)
            frame_3dbb_map_world = None
            obb_bbox_frames = None
            if frames_final_data:
                obb_bbox_frames = frames_final_data.get("obb_bbox_frames", None)
                # Also try original world-space frames for BEFORE visualization
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
                n = R_g[:, 1]  # normal

                floor_origin_world = t_g.astype(np.float32)
                axis_len_floor = float(s_g) * 0.5 if s_g is not None else 0.5
                floor_axes_before = np.stack(
                    [
                        t1 * axis_len_floor,
                        t2 * axis_len_floor,
                        n * axis_len_floor,
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
        R_final = tf["R_world_to_final"]  # (3,3)
        t_final = tf["t_world_to_final"]  # (3,)
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
        # 3D bounding boxes in FINAL coords (from bridge PKL)
        # ----------------------------------------------------------------------
        frame_3dbb_map_final: Optional[Dict[str, Dict[str, Any]]] = None
        if obb_bbox_frames is not None:
            frame_3dbb_map_final = {}
            for frame_name, frame_rec in obb_bbox_frames.items():
                objects_final = []
                for obj in frame_rec.get("objects", []):
                    corners_final = obj.get("corners_final", None)
                    if corners_final is None:
                        continue
                    obj_final = dict(obj)
                    obj_final["obb_final"] = {
                        "corners_final": np.asarray(corners_final, dtype=np.float32),
                    }
                    obj_final["color_after"] = obj.get("color", [255, 230, 80])
                    objects_final.append(obj_final)
                frame_3dbb_map_final[frame_name] = {"objects": objects_final}

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

        Loads bridge PKL (GT + GDino merged) and generates 4D annotations.
        """
        video_id_3d_bbox_predictions = self.get_video_3d_bridge_annotations(video_id)

        self.generate_video_world_bb_annotations(
            video_id,
            video_id_3d_bbox_predictions,
            visualize=True,
        )

        # Now show the original Pi3-space outputs (points + floor + frames + camera + 3D boxes)
        self.visualize_original_results(video_id=video_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate world-4D bbox annotations for Action Genome videos. "
            "Loads pre-computed 3D bbox predictions, applies floor alignment, "
            "enforces object permanence, and computes static unions."
        )
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Process only videos belonging to this split. "
            "Valid splits: 04, 59, AD, EH, IL, MP, QT, UZ. "
            "If not set, all videos are processed."
        ),
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Process a single video (e.g., '00T1E.mp4').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing 4D annotation files.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch rerun visualization for each video (slow).",
    )
    return parser.parse_args()


def main():
    import random
    from tqdm import tqdm
    from datasets.preprocess.annotations.annotation_utils import get_video_belongs_to_split

    args = parse_args()

    generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )

    # ------------------------------------------------------------------
    # Discover available videos from 3D bbox PKL directory
    # ------------------------------------------------------------------
    if args.video:
        # Single video mode
        video_ids = [args.video if args.video.endswith(".mp4") else f"{args.video}.mp4"]
    else:
        bbox_3d_dir = generator.bbox_3d_obb_bridge_root_dir
        if not bbox_3d_dir.exists():
            logger.error(f"ERROR: Bridge PKL directory not found: {bbox_3d_dir}")
            sys.exit(1)

        video_ids = sorted([
            f"{p.stem}.mp4"
            for p in bbox_3d_dir.glob("*.pkl")
        ])

    if not video_ids:
        print("No videos found to process.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Filter by split
    # ------------------------------------------------------------------
    if args.split:
        video_ids = [
            vid for vid in video_ids
            if get_video_belongs_to_split(vid) == args.split
        ]
        print(f"Split '{args.split}': {len(video_ids)} videos")

    if not video_ids:
        print(f"No videos found for split '{args.split}'.")
        sys.exit(0)

    # Shuffle for multi-GPU parallelism
    random.shuffle(video_ids)

    logger.info(f"Processing {len(video_ids)} videos (overwrite={args.overwrite}, visualize={args.visualize})")

    # ------------------------------------------------------------------
    # Process each video
    # ------------------------------------------------------------------
    success_count = 0
    skip_count = 0
    error_count = 0

    for video_id in tqdm(video_ids, desc="World4D generation"):
        # Check if output already exists
        out_4d_path = generator.bbox_4d_root_dir / f"{video_id[:-4]}.pkl"
        if out_4d_path.exists() and not args.overwrite:
            skip_count += 1
            continue

        try:
            # Load active objects classification (needed for static/dynamic split)
            generator.fetch_stored_active_objects_in_video(video_id)

            # Load bridge 3D bbox predictions
            video_3d = generator.get_video_3d_bridge_annotations(video_id)
            if video_3d is None:
                logger.warning(f"  [{video_id}] No bridge 3D bbox annotations, skipping.")
                error_count += 1
                continue

            # Generate world-4D annotations
            generator.generate_video_world_bb_annotations(
                video_id=video_id,
                video_id_3d_bbox_predictions=video_3d,
                visualize=args.visualize,
            )
            success_count += 1

        except Exception as e:
            logger.error(f"  [{video_id}] Error: {e}", exc_info=True)
            error_count += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(video_ids)
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Done: {success_count}/{total} succeeded, "
        f"{skip_count} skipped (already exist), "
        f"{error_count} errors."
    )
    logger.info(f"Output directory: {generator.bbox_4d_root_dir}")
    logger.info(f"Log file: {_LOG_DIR / 'frame_to_world4D_annotations.log'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
