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
from tqdm import tqdm

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
from datasets.preprocess.annotations.frame_to_world_base import (
    FrameToWorldBase,
    compute_final_world_transform,
    rerun_frame_vis_results,
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
            print(f"[world4d][{video_id}] No labels found in any frame. Skipping.")
            return

        # ----------------------------------------------------------------------
        # Enforce object permanence & fill missing
        # ----------------------------------------------------------------------
        # We assume `frame_3dbb_map_world` keys are "000123.png", etc.
        # Sort them numerically.
        def _get_frame_idx(fn: str) -> int:
            return int(fn[:-4]) if fn.endswith(".png") else -1

        frame_names_sorted = sorted(frame_3dbb_map_world.keys(), key=_get_frame_idx)
        frames_filled_world: Dict[str, Dict[str, Any]] = {}

        # Track "last seen" bbox for each label
        last_seen_bbox_world: Dict[str, Dict[str, Any]] = {}

        # First pass: identify first occurrence of each label (for fallback)
        first_seen_bbox_world: Dict[str, Dict[str, Any]] = {}
        for fn in frame_names_sorted:
            objs = frame_3dbb_map_world[fn].get("objects", [])
            for obj in objs:
                lbl = obj.get("label", None)
                if lbl and lbl not in first_seen_bbox_world:
                    first_seen_bbox_world[lbl] = obj

        # Second pass: fill
        for fn in frame_names_sorted:
            current_objs = frame_3dbb_map_world[fn].get("objects", [])
            # Map label -> obj for current frame
            current_label_map = {}
            for obj in current_objs:
                lbl = obj.get("label", None)
                if lbl:
                    current_label_map[lbl] = obj

            new_objs_list = []
            for lbl in all_labels:
                if lbl in current_label_map:
                    # Present
                    obj = current_label_map[lbl]
                    last_seen_bbox_world[lbl] = obj
                    # We'll append a copy to be safe
                    new_objs_list.append(dict(obj))
                else:
                    # Missing -> fill
                    if lbl in last_seen_bbox_world:
                        # Use last seen
                        filled_obj = dict(last_seen_bbox_world[lbl])
                        filled_obj["is_filled"] = True  # mark as filled
                        new_objs_list.append(filled_obj)
                    elif lbl in first_seen_bbox_world:
                        # Use first seen (backward fill)
                        filled_obj = dict(first_seen_bbox_world[lbl])
                        filled_obj["is_filled"] = True
                        new_objs_list.append(filled_obj)
                    else:
                        # Should not happen given logic above
                        print(f"[world4d][{video_id}] Label '{lbl}' missing everywhere??")

            frames_filled_world[fn] = {
                "objects": new_objs_list,
                # Copy other frame-level metadata if needed
                "camera_pose": frame_3dbb_map_world[fn].get("camera_pose", None),
            }

        # ----------------------------------------------------------------------
        # Compute FINAL coords for every object
        # ----------------------------------------------------------------------
        # x_final = R_final @ x_world + t_final
        # We'll store `aabb_final` inside each object dict.
        # Structure of `aabb_floor_aligned`:
        #   { "center_world": [x,y,z], "size_world": [w,h,d], "corners_world": (8,3), ... }

        for fn in frame_names_sorted:
            objs = frames_filled_world[fn]["objects"]
            for obj in objs:
                bbox_world = obj["aabb_floor_aligned"]
                corners_w = np.asarray(bbox_world["corners_world"], dtype=np.float32) # (8,3)

                # Transform corners
                corners_f = (R_final @ corners_w.T).T + t_final[None, :]

                # Recompute center/size in final frame from transformed corners
                min_f = corners_f.min(axis=0)
                max_f = corners_f.max(axis=0)
                center_f = (min_f + max_f) / 2.0
                size_f = (max_f - min_f)

                obj["aabb_final"] = {
                    "corners_final": corners_f,
                    "center_final": center_f,
                    "size_final": size_f,
                }

        # ----------------------------------------------------------------------
        # Static Object Union (in FINAL coords)
        # ----------------------------------------------------------------------
        # Identify static labels
        if video_id not in self.video_id_active_objects_annotations_map:
            self.fetch_stored_active_objects_in_video(video_id)

        annotated_active = set(self.video_id_active_objects_annotations_map.get(video_id, []))
        reasoned_active = set(self.video_id_active_objects_b_reasoned_map.get(video_id, []))

        # "Active" usually means "interacted with" or "moving".
        # We'll treat anything NOT in reasoned_active as potentially static?
        # Or use your heuristic:
        non_moving_candidates = {"floor", "sofa", "couch", "bed", "doorway", "table", "chair", "shelf", "closet", "cabinet"}
        # Actually, let's define "dynamic" as reasoned_active minus non_moving_candidates
        # and "static" as everything else in `all_labels`.

        # Refined logic from your stats script:
        # dynamic_labels = [lbl for lbl in reasoned_active if lbl not in non_moving_candidates]
        # static_labels = [lbl for lbl in all_labels if lbl not in dynamic_labels]

        # Let's just use a simpler check: if it's in `reasoned_active` AND NOT in `non_moving_candidates`, it's dynamic.
        # Otherwise static.
        def _is_static(lbl: str) -> bool:
            # normalize
            lbl_norm = lbl.lower().strip()
            if lbl_norm in non_moving_candidates:
                return True
            if lbl_norm in reasoned_active:
                return False  # It is active and not a non-moving type => dynamic
            # If not active, treat as static
            return True

        static_labels = [lbl for lbl in all_labels if _is_static(lbl)]
        # dynamic_labels = [lbl for lbl in all_labels if not _is_static(lbl)]

        # Compute union box for each static label
        static_union_bboxes: Dict[str, Dict[str, Any]] = {}  # label -> {min_f, max_f}

        for fn in frame_names_sorted:
            objs = frames_filled_world[fn]["objects"]
            for obj in objs:
                lbl = obj.get("label", None)
                if lbl in static_labels:
                    aabb_f = obj["aabb_final"]
                    c_f = aabb_f["corners_final"]
                    curr_min = c_f.min(axis=0)
                    curr_max = c_f.max(axis=0)

                    if lbl not in static_union_bboxes:
                        static_union_bboxes[lbl] = {
                            "min": curr_min,
                            "max": curr_max,
                        }
                    else:
                        static_union_bboxes[lbl]["min"] = np.minimum(static_union_bboxes[lbl]["min"], curr_min)
                        static_union_bboxes[lbl]["max"] = np.maximum(static_union_bboxes[lbl]["max"], curr_max)

        # Overwrite static objects with union box
        for fn in frame_names_sorted:
            objs = frames_filled_world[fn]["objects"]
            for obj in objs:
                lbl = obj.get("label", None)
                if lbl in static_union_bboxes:
                    # It's static, overwrite aabb_final
                    u = static_union_bboxes[lbl]
                    min_u = u["min"]
                    max_u = u["max"]
                    center_u = (min_u + max_u) / 2.0
                    size_u = (max_u - min_u)

                    # Reconstruct 8 corners from min/max
                    # (min_x, min_y, min_z), ...
                    # A simple way is itertools product, or manual
                    x0, y0, z0 = min_u
                    x1, y1, z1 = max_u
                    corners_u = np.array([
                        [x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
                        [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]
                    ], dtype=np.float32)
                    # Note: order doesn't strictly matter for bounding volume,
                    # but for wireframe edges it might.
                    # Let's use a standard order consistent with your cuboid_edges if possible.
                    # Your cuboid_edges expects:
                    # 0:000, 1:100, 2:010, 3:110, 4:001, 5:101, 6:011, 7:111 (example)
                    # Let's just use the helper from annotation_utils if available, or manual:
                    # _corners_from_mins_maxs
                    corners_u = np.array([
                        [x0, y0, z0], # 0
                        [x1, y0, z0], # 1
                        [x0, y1, z0], # 2
                        [x1, y1, z0], # 3
                        [x0, y0, z1], # 4
                        [x1, y0, z1], # 5
                        [x0, y1, z1], # 6
                        [x1, y1, z1], # 7
                    ], dtype=np.float32)

                    obj["aabb_final"] = {
                        "corners_final": corners_u,
                        "center_final": center_u,
                        "size_final": size_u,
                    }
                    obj["is_static_union"] = True

        # ----------------------------------------------------------------------
        # Save 4D annotations
        # ----------------------------------------------------------------------
        video_4d_out = {
            "video_id": video_id,
            "frames": frames_filled_world,  # keyed by "000123.png"
            "global_floor_sim": gfs,
            "floor_mesh": {
                "gv": gv,
                "gf": gf,
                "gc": gc
            } if floor else None,
            "world_to_final": {
                "R_final": R_final,
                "t_final": t_final,
            },
            "all_labels": sorted(list(all_labels)),
            "static_labels": sorted(list(static_labels)),
        }

        out_path = self.bbox_4d_root_dir / f"{video_id[:-4]}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(video_4d_out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[world4d][{video_id}] Saved 4D annotations to {out_path}")

        # ----------------------------------------------------------------------
        # Visualization (Optional)
        # ----------------------------------------------------------------------
        if visualize:
            # Optional: visualize world-4D bboxes + POINTS (BEFORE/AFTER) over time
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

            rerun_frame_vis_results(
                video_id=video_id,
                stems_S=stems_S,
                frame_annotated_dir_path=self.frame_annotated_dir_path,

                # Points/cameras BEFORE
                points_before=points_world,
                conf_before=conf_world,
                colors_before=colors_world,
                cameras_before=cameras_world,

                # Points/cameras AFTER
                points_after=points_final,
                colors_after=colors_final,
                cameras_after=cameras_final,

                # Floor mesh (BEFORE/AFTER)
                floor_vertices_before=floor_vertices_before,
                floor_axes_before=floor_axes_before,
                floor_origin_before=floor_origin_world,

                floor_vertices_after=floor_vertices_after,
                floor_axes_after=floor_axes_after,
                floor_origin_after=floor_origin_final,

                floor_faces=floor_faces,
                floor_kwargs=floor_kwargs,

                # 4D bboxes in FINAL coords
                frame_3dbb_after=frames_filled_world_vis,

                app_id="World4D-BBoxes+Points",
                vis_mode="after",   # change to "both" if you want BEFORE/AFTER side-by-side
            )


# --------------------------------------------------------------------------------------
# Dataset + CLI
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
        description="Generate World 4D BBox Annotations (fill missing + static union)."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument("--split", type=str, default="04")
    parser.add_argument("--visualize", action="store_true", help="Visualize results in Rerun")
    return parser.parse_args()


def main():
    args = parse_args()

    generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    _, _, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)

    # Process train
    for data in tqdm(dataloader_train, desc="Processing Train"):
        video_id = data["video_id"]
        if get_video_belongs_to_split(video_id) != args.split:
            continue
        try:
            generator.generate_video_world_bb_annotations(
                video_id=video_id,
                video_id_gt_annotations=None,
                video_id_gdino_annotations=None,
                video_id_3d_bbox_predictions=None,
                visualize=args.visualize,
            )
        except Exception as e:
            print(f"Error processing {video_id}: {e}")

    # Process test
    for data in tqdm(dataloader_test, desc="Processing Test"):
        video_id = data["video_id"]
        if get_video_belongs_to_split(video_id) != args.split:
            continue
        try:
            generator.generate_video_world_bb_annotations(
                video_id=video_id,
                video_id_gt_annotations=None,
                video_id_gdino_annotations=None,
                video_id_3d_bbox_predictions=None,
                visualize=args.visualize,
            )
        except Exception as e:
            print(f"Error processing {video_id}: {e}")


def main_sample():
    args = parse_args()
    generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    video_id = "00T1E.mp4"
    generator.generate_video_world_bb_annotations(
        video_id=video_id,
        video_id_gt_annotations=None,
        video_id_gdino_annotations=None,
        video_id_3d_bbox_predictions=None,
        visualize=True,
    )


if __name__ == "__main__":
    # main()
    main_sample()
