#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from datasets.preprocess.annotations.annotation_utils import (
    _faces_u32,
)
from datasets.preprocess.annotations.raw.frame_to_world4D_base import FrameToWorldBase, rerun_frame_vis_results


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
        # Basic stats before 4D filling
        # ----------------------------------------------------------------------

        # Step 1: Estimate video-level GT label set from GT annotations.
        # GDino-sourced objects in the 3D PKL will be filtered to this set.
        try:
            _, raw_gt_annotations = self.get_video_gt_annotations(video_id)
            video_gt_labels: set = set()
            for frame_items in raw_gt_annotations:
                for item in frame_items:
                    if "person_bbox" in item:
                        video_gt_labels.add("person")
                    else:
                        cid = item.get("class", None)
                        cat_name = self.catid_to_name_map.get(cid, None)
                        if cat_name:
                            video_gt_labels.add(_normalize_label(cat_name))
            print(
                f"[world4d][{video_id}] video GT labels "
                f"({len(video_gt_labels)}): {sorted(video_gt_labels)}"
            )
        except FileNotFoundError:
            # If GT annotations not available, don't filter
            video_gt_labels = None
            print(
                f"[world4d][{video_id}] WARNING: GT annotations not found, "
                "skipping video-level label filtering for GDino objects."
            )

        # Step 2: Collect labels from 3D bbox PKL and filter by video GT set
        all_labels = set()
        num_frames_with_objects = 0
        num_total_objects = 0
        num_gdino_filtered = 0

        for frame_name, frame_rec in frame_3dbb_map_world.items():
            objects = frame_rec.get("objects", [])
            if not objects:
                continue

            # Filter objects: normalize labels, reject GDino objects not in video GT
            filtered_objects = []
            for obj in objects:
                lbl = obj.get("label", None)
                if not lbl:
                    continue
                lbl_norm = _normalize_label(lbl)
                obj["label"] = lbl_norm  # normalize in-place

                source = obj.get("source", "gt")
                if source == "gdino" and video_gt_labels is not None:
                    if lbl_norm not in video_gt_labels:
                        num_gdino_filtered += 1
                        continue  # skip this GDino object
                filtered_objects.append(obj)

            frame_rec["objects"] = filtered_objects

            if filtered_objects:
                num_frames_with_objects += 1
                num_total_objects += len(filtered_objects)
                for obj in filtered_objects:
                    all_labels.add(obj["label"])

        print(
            f"[world4d][{video_id}] frames_with_objects={num_frames_with_objects}, "
            f"total_objects={num_total_objects}, "
            f"unique_labels={sorted(all_labels)}"
        )
        if num_gdino_filtered > 0:
            print(
                f"[world4d][{video_id}] filtered {num_gdino_filtered} GDino objects "
                f"(labels not in video GT set)"
            )

        if not all_labels:
            print(f"[world4d][{video_id}] No object labels found. Skipping 4D generation.")
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

        print(
            f"[world4d][{video_id}] static_labels_in_3d={sorted(static_labels_in_3d)} "
            f"(from active={sorted(video_active_object_labels)})"
        )

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

            # Only fill/extrapolate for static labels; dynamic objects only appear where detected
            for lbl in sorted(all_labels):
                is_static = lbl in static_labels_in_3d

                if lbl in label_to_obj_current:
                    # Object exists in this frame - always include it
                    base_obj = label_to_obj_current[lbl]
                    last_seen[lbl] = (fname, base_obj)
                    filled_obj = _clone_with_meta(
                        base_obj,
                        filled=False,
                        source_frame=fname,
                        target_frame=fname,
                    )
                elif is_static:
                    # Static object missing in this frame - fill from last/first known
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
                else:
                    # Dynamic object missing - do NOT fill/extrapolate
                    # Skip this label for this frame
                    continue

                # Ensure label is consistent
                filled_obj["label"] = lbl

                # Attach FINAL-coords bbox (obb_final) from WORLD OBB corners
                # Prefer obb_floor_parallel; fall back to aabb_floor_aligned
                corners_world = None
                obb_data = filled_obj.get("obb_floor_parallel", None)
                if obb_data and obb_data.get("corners_world"):
                    corners_world = np.asarray(
                        obb_data["corners_world"], dtype=np.float32
                    )
                elif "aabb_floor_aligned" in filled_obj:
                    bbox_3d = filled_obj["aabb_floor_aligned"]
                    cw = bbox_3d.get("corners_world", [])
                    if len(cw) > 0:
                        corners_world = np.asarray(cw, dtype=np.float32)

                if corners_world is not None and corners_world.size > 0:
                    corners_final = (R_final @ corners_world.T).T + t_final[None, :]
                    filled_obj["obb_final"] = {
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

        # 1) Compute union bbox (FINAL coords) per static label using OBB corners
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
                print(
                    f"[world4d][{video_id}] WARNING: static label '{lbl}' has no "
                    "obb_final corners available; skipping union."
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

                    # Ensure obb_final exists, then overwrite only FINAL box
                    if "obb_final" not in obj:
                        obj["obb_final"] = {}

                    obj["obb_final"]["corners_final"] = union_corners_final

                    # NOTE: we intentionally do NOT touch
                    # obj["obb_floor_parallel"]["corners_world"]
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
        }

        with open(out_4d_path, "wb") as f:
            pickle.dump(world4d_annotations, f)

        print(f"[world4d][{video_id}] Saved world-4D bbox annotations to {out_4d_path}")

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
        print(f"[world4d][{video_id}] Generating from Firebase annotations")

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

                # Otherwise compute FINAL from WORLD OBB corners if available
                if corners_final is None:
                    obb_world = obj_raw.get("obb_floor_parallel", None)
                    if isinstance(obb_world, dict) and obb_world.get("corners_world") is not None:
                        corners_world = np.asarray(obb_world["corners_world"], dtype=np.float32)
                        if corners_world.size > 0:
                            corners_final = (R_final @ corners_world.T).T + t_final[None, :]

                # Fall back to aabb_floor_aligned if OBB not available
                if corners_final is None:
                    aabb_world = obj_raw.get("aabb_floor_aligned", None)
                    if isinstance(aabb_world, dict) and aabb_world.get("corners_world") is not None:
                        corners_world = np.asarray(aabb_world["corners_world"], dtype=np.float32)
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
        # Static union logic: UNION IN FINAL COORDS, WORLD UNCHANGED
        # ----------------------------------------------------------------------
        def _make_aabb_corners_from_minmax(min_xyz: np.ndarray, max_xyz: np.ndarray) -> List[List[float]]:
            x0, y0, z0 = min_xyz.tolist()
            x1, y1, z1 = max_xyz.tolist()
            return [
                [x0, y0, z0],
                [x1, y0, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x0, y1, z0],
                [x1, y1, z0],
                [x0, y1, z1],
                [x1, y1, z1],
            ]

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
            min_xyz = all_pts.min(axis=0)
            max_xyz = all_pts.max(axis=0)
            static_union_map_final[lbl] = _make_aabb_corners_from_minmax(min_xyz, max_xyz)

        if static_union_map_final:
            print(
                f"[world4d][{video_id}] Applying static union bboxes in FINAL coords for {len(static_union_map_final)} labels.")
            for fname in frame_names_sorted:
                for obj in frames_filled.get(fname, {}).get("objects", []):
                    lbl = obj.get("label")
                    if lbl not in static_union_map_final:
                        continue

                    # Overwrite ONLY FINAL coords
                    _set_corners_final(obj, static_union_map_final[lbl])

                    # NOTE: if obj carries any world-space fields, we intentionally do not touch them.

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

        print(f"[world4d][{video_id}] Generated 4D annotations for {len(frame_names_sorted)} frames")
        return result

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

        points_world = P["points"]  # (S,H,W,3)
        conf_world = P["conf"]  # (S,H,W) or None
        stems = P["frame_stems"]  # ["000123", ...]
        colors_world = P["colors"]  # (S,H,W,3)
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
        # Transform 3D bounding boxes
        #   - Add `obb_final` with `corners_final` into a separate map
        #   - Prefer obb_floor_parallel; fall back to aabb_floor_aligned
        # ----------------------------------------------------------------------
        frame_3dbb_map_final: Optional[Dict[str, Dict[str, Any]]] = None
        if frame_3dbb_map_world is not None:
            frame_3dbb_map_final = {}
            for frame_name, frame_rec in frame_3dbb_map_world.items():
                objects_world = frame_rec.get("objects", [])
                objects_final = []
                for obj in objects_world:
                    # Prefer OBB corners, fall back to AABB
                    corners_world = None
                    obb_data = obj.get("obb_floor_parallel", None)
                    if obb_data and obb_data.get("corners_world"):
                        corners_world = np.asarray(
                            obb_data["corners_world"], dtype=np.float32
                        )
                    elif "aabb_floor_aligned" in obj:
                        bbox_3d = obj["aabb_floor_aligned"]
                        cw = bbox_3d.get("corners_world", [])
                        if len(cw) > 0:
                            corners_world = np.asarray(cw, dtype=np.float32)

                    if corners_world is None or corners_world.size == 0:
                        continue

                    corners_final = (R_final @ corners_world.T).T + t_final[None, :]

                    obj_final = dict(obj)
                    obj_final["obb_final"] = {
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
        bbox_3d_dir = generator.bbox_3d_root_dir
        if not bbox_3d_dir.exists():
            print(f"ERROR: 3D bbox directory not found: {bbox_3d_dir}")
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

    print(f"Processing {len(video_ids)} videos (overwrite={args.overwrite}, visualize={args.visualize})")

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

            # Load 3D bbox predictions
            video_3d = generator.get_video_3d_annotations(video_id)
            if video_3d is None:
                print(f"  ⚠️  [{video_id}] No 3D bbox annotations, skipping.")
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
            print(f"  ⚠️  [{video_id}] Error: {e}")
            error_count += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(video_ids)
    print(f"\n{'='*60}")
    print(
        f"Done: {success_count}/{total} succeeded, "
        f"{skip_count} skipped (already exist), "
        f"{error_count} errors."
    )
    print(f"Output directory: {generator.bbox_4d_root_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
