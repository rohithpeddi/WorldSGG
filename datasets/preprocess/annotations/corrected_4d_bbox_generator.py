#!/usr/bin/env python3
"""
Corrected 4D BBox Generator
============================

Generates 4D bounding box annotations (object-permanence-filled, with static
union logic) from corrected world bboxes that were built using manually
corrected floor transforms.

Flow:
    1.  Load corrected world bboxes from ``bbox_annotations_3d_obb_corrected/<video>.pkl``
        (produced by ``corrected_world_bbox_generator.py``).
    2.  Extract the corrected floor transform (``corrected_floor_transform.combined_transform_4x4``)
        and use it as the WORLD → FINAL mapping (replacing the old ``compute_final_world_transform``).
    3.  For each frame, convert WORLD corners → FINAL corners for AABB, OBB-floor-parallel,
        and OBB-arbitrary variants.
    4.  Apply object-permanence filling (static labels only) + static-union in FINAL coords —
        semantics identical to the original ``frame_to_world4D_annotations.py``.
    5.  Save to ``bbox_annotations_4d_corrected/<video>.pkl``.

Reads from:
    - bbox_annotations_3d_obb_corrected/<video>.pkl

Writes to:
    - bbox_annotations_4d_corrected/<video>.pkl

Usage:
    python corrected_4d_bbox_generator.py --corrections-only
    python corrected_4d_bbox_generator.py --video 001YG.mp4
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + "/..")
sys.path.insert(0, os.path.dirname(__file__))

from annotation_utils import (
    get_video_belongs_to_split,
    _faces_u32,
    _load_pkl_if_exists,
)

from datasets.preprocess.annotations.raw.frame_to_world4D_base import (
    FrameToWorldBase,
)


# =====================================================================
# STATIC HELPERS
# =====================================================================

def _transform_corners(corners_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform (N,3) corners from world to final frame: p_final = R @ p_world + t."""
    return (R @ corners_world.T).T + t[None, :]


def _make_aabb_corners_from_minmax(
    min_xyz: np.ndarray, max_xyz: np.ndarray,
) -> List[List[float]]:
    """Build 8 axis-aligned cuboid corners from min/max bounds (same indexing
    as the original ``frame_to_world4D_annotations._make_aabb_corners_from_minmax``)."""
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


# =====================================================================
# CORRECTED 4D BBOX GENERATOR
# =====================================================================

class Corrected4DBBoxGenerator(FrameToWorldBase):
    """
    Generates 4D bounding-box annotations from corrected world bboxes.

    Inherits ``FrameToWorldBase`` for shared paths, loaders, and the
    active-object classification used by the filling/union logic.
    """

    def __init__(self, ag_root_directory: str, dynamic_scene_dir_path: str):
        super().__init__(ag_root_directory, dynamic_scene_dir_path)

        # Corrected 3D bbox source
        self.bbox_3d_obb_corrected_root_dir = (
            self.world_annotations_root_dir / "bbox_annotations_3d_obb_corrected"
        )

        # Output directory for corrected 4D bboxes
        self.bbox_4d_corrected_root_dir = (
            self.world_annotations_root_dir / "bbox_annotations_4d_corrected"
        )
        os.makedirs(self.bbox_4d_corrected_root_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def get_corrected_video_3d_annotations(
        self, video_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load the corrected world bbox PKL."""
        pkl_path = self.bbox_3d_obb_corrected_root_dir / f"{video_id[:-4]}.pkl"
        return _load_pkl_if_exists(pkl_path)

    # ------------------------------------------------------------------
    # Core: generate corrected 4D bboxes for one video
    # ------------------------------------------------------------------

    def generate_corrected_video_4d_annotations(
        self,
        video_id: str,
        *,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Generate corrected 4D bounding-box annotations for a single video.

        Steps:
          1) Load corrected world bboxes.
          2) Extract corrected floor transform (4×4) for WORLD → FINAL.
          3) Build per-frame objects with FINAL-coords corners (AABB + OBB).
          4) Classify labels as static/dynamic using active-object lists.
          5) Apply object-permanence filling (static only, forward/backward).
          6) Compute static union in FINAL coords.
          7) Save to ``bbox_annotations_4d_corrected/<video>.pkl``.
        """
        out_path = self.bbox_4d_corrected_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[corrected-4d][{video_id}] exists, skipping.")
            return out_path

        # ---- 1) Load corrected world bboxes ----
        video_3dgt = self.get_corrected_video_3d_annotations(video_id)
        if video_3dgt is None:
            print(f"[corrected-4d][{video_id}] no corrected world bbox PKL. Skipping.")
            return None

        cft = video_3dgt.get("corrected_floor_transform", None)
        if cft is None:
            print(f"[corrected-4d][{video_id}] no corrected_floor_transform. Skipping.")
            return None

        frame_3dbb_map_world = video_3dgt.get("frames", None)
        if not frame_3dbb_map_world:
            print(f"[corrected-4d][{video_id}] no frames in corrected bboxes. Skipping.")
            return None

        # ---- 2) Extract corrected floor transform ----
        T_4x4 = np.asarray(
            cft["combined_transform_4x4"], dtype=np.float32,
        )
        R_final = T_4x4[:3, :3]
        t_final = T_4x4[:3, 3]

        # ---- 3) Build per-frame objects with FINAL-coords corners ----
        frame_names_sorted = sorted(
            frame_3dbb_map_world.keys(),
            key=lambda fn: int(Path(fn).stem) if Path(fn).stem.isdigit() else Path(fn).stem,
        )

        # Collect all labels and build normalized frame data
        all_labels = set()
        frames_data: Dict[str, Dict[str, Any]] = {}

        for fname in frame_names_sorted:
            frame_rec = frame_3dbb_map_world[fname]
            objects = frame_rec.get("objects", [])
            if not objects:
                frames_data[fname] = {"objects": []}
                continue

            out_objs: List[Dict[str, Any]] = []
            for obj in objects:
                lbl = obj.get("label", None)
                if lbl:
                    all_labels.add(lbl)

                new_obj = dict(obj)

                # ---- Transform AABB corners to FINAL ----
                aabb = obj.get("aabb_floor_aligned", None)
                if aabb is not None and "corners_world" in aabb:
                    cw = np.asarray(aabb["corners_world"], dtype=np.float32)
                    cf = _transform_corners(cw, R_final, t_final)
                    new_obj["aabb_final"] = {
                        "corners_final": cf.astype(np.float32).tolist(),
                    }
                    new_obj["corners_final"] = cf.astype(np.float32).tolist()
                    new_obj["center"] = cf.mean(axis=0).tolist()

                # ---- Transform OBB floor-parallel corners to FINAL ----
                obb_fp = obj.get("obb_floor_parallel", None)
                if obb_fp is not None and "corners_world" in obb_fp:
                    cw = np.asarray(obb_fp["corners_world"], dtype=np.float32)
                    cf = _transform_corners(cw, R_final, t_final)
                    new_obj["obb_floor_parallel_final"] = {
                        "corners_final": cf.astype(np.float32).tolist(),
                    }

                # ---- Transform OBB arbitrary corners to FINAL ----
                obb_arb = obj.get("obb_arbitrary", None)
                if obb_arb is not None and "corners_world" in obb_arb:
                    cw = np.asarray(obb_arb["corners_world"], dtype=np.float32)
                    cf = _transform_corners(cw, R_final, t_final)
                    new_obj["obb_arbitrary_final"] = {
                        "corners_final": cf.astype(np.float32).tolist(),
                    }

                new_obj["color_after"] = obj.get("color", [255, 230, 80])
                out_objs.append(new_obj)

            frames_data[fname] = {"objects": out_objs}

        print(
            f"[corrected-4d][{video_id}] "
            f"frames={len(frame_names_sorted)}, "
            f"unique_labels={sorted(all_labels)}"
        )

        if not all_labels:
            print(f"[corrected-4d][{video_id}] No object labels found. Skipping.")
            return None

        # ---- 4) Classify labels as static / dynamic ----
        self.fetch_stored_active_objects_in_video(video_id)
        video_active_object_labels = self.video_id_active_objects_annotations_map.get(
            video_id, [],
        )
        video_reasoned_active_object_labels = (
            self.video_id_active_objects_b_reasoned_map.get(video_id, [])
        )

        non_moving_objects = [
            "floor", "sofa", "couch", "bed", "doorway", "table", "chair",
        ]
        video_dynamic_object_labels = [
            obj for obj in video_reasoned_active_object_labels
            if obj not in non_moving_objects
        ]
        video_static_object_labels = [
            obj for obj in video_active_object_labels
            if obj not in video_dynamic_object_labels
        ]

        static_labels_in_3d = [
            lbl for lbl in video_static_object_labels if lbl in all_labels
        ]

        print(
            f"[corrected-4d][{video_id}] "
            f"static_labels_in_3d={sorted(static_labels_in_3d)}"
        )

        # ---- 5) Object-permanence filling (same semantics as original) ----
        label_first_source: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for fname in frame_names_sorted:
            for obj in frames_data.get(fname, {}).get("objects", []):
                lbl = obj.get("label", None)
                if lbl and lbl not in label_first_source:
                    label_first_source[lbl] = (fname, obj)

        for lbl in all_labels:
            if lbl not in label_first_source:
                raise ValueError(
                    f"[corrected-4d][{video_id}] Label '{lbl}' has no source frame."
                )

        last_seen: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        frames_filled: Dict[str, Dict[str, Any]] = {}

        def _deepish_clone_obj(base_obj: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(base_obj)
            if isinstance(base_obj.get("aabb_final"), dict):
                out["aabb_final"] = dict(base_obj["aabb_final"])
                if out["aabb_final"].get("corners_final") is not None:
                    out["aabb_final"]["corners_final"] = list(
                        out["aabb_final"]["corners_final"]
                    )
            if base_obj.get("corners_final") is not None:
                out["corners_final"] = list(base_obj["corners_final"])
            # Clone OBB final dicts too
            for obb_key in ("obb_floor_parallel_final", "obb_arbitrary_final"):
                if isinstance(base_obj.get(obb_key), dict):
                    out[obb_key] = dict(base_obj[obb_key])
                    if out[obb_key].get("corners_final") is not None:
                        out[obb_key]["corners_final"] = list(
                            out[obb_key]["corners_final"]
                        )
            return out

        for fname in frame_names_sorted:
            objects = frames_data.get(fname, {}).get("objects", [])
            label_to_obj_current: Dict[str, Dict[str, Any]] = {}
            for obj in objects:
                lbl = obj.get("label", None)
                if lbl and lbl not in label_to_obj_current:
                    label_to_obj_current[lbl] = obj

            filled_objects: List[Dict[str, Any]] = []

            for lbl in sorted(all_labels):
                is_static = lbl in static_labels_in_3d

                if lbl in label_to_obj_current:
                    base_obj = label_to_obj_current[lbl]
                    last_seen[lbl] = (fname, base_obj)
                    new_obj = _deepish_clone_obj(base_obj)
                    new_obj["world4d_filled"] = False
                    new_obj["world4d_source_frame"] = fname
                    new_obj["world4d_frame"] = fname
                    new_obj["label"] = lbl

                elif is_static:
                    if lbl in last_seen:
                        src_frame, base_obj = last_seen[lbl]
                    else:
                        src_frame, base_obj = label_first_source[lbl]

                    new_obj = _deepish_clone_obj(base_obj)
                    new_obj["world4d_filled"] = True
                    new_obj["world4d_source_frame"] = src_frame
                    new_obj["world4d_frame"] = fname
                    new_obj["label"] = lbl
                else:
                    # dynamic + missing → skip
                    continue

                filled_objects.append(new_obj)

            frames_filled[fname] = {"objects": filled_objects}

        # ---- 6) Static union logic (UNION IN FINAL COORDS) ----
        def _get_corners_final(obj: Dict[str, Any]) -> List:
            af = obj.get("aabb_final")
            if isinstance(af, dict) and af.get("corners_final") is not None:
                return af["corners_final"]
            if obj.get("corners_final") is not None:
                return obj["corners_final"]
            return []

        def _set_corners_final(obj: Dict[str, Any], corners_list: List) -> None:
            if not isinstance(obj.get("aabb_final"), dict):
                obj["aabb_final"] = {}
            obj["aabb_final"]["corners_final"] = corners_list
            obj["corners_final"] = corners_list
            if len(corners_list) > 0:
                c = np.asarray(corners_list, dtype=np.float32)
                obj["center"] = c.mean(axis=0).tolist()
            else:
                obj["center"] = [0.0, 0.0, 0.0]

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
                    f"[corrected-4d][{video_id}] WARNING: static label '{lbl}' "
                    "has no valid corners_final; skipping union."
                )
                continue

            all_pts = np.concatenate(all_corners_list, axis=0)  # (N*8, 3)
            min_xyz = all_pts.min(axis=0)
            max_xyz = all_pts.max(axis=0)
            static_union_map_final[lbl] = _make_aabb_corners_from_minmax(
                min_xyz, max_xyz,
            )

        if static_union_map_final:
            print(
                f"[corrected-4d][{video_id}] Applying static union for "
                f"{len(static_union_map_final)} labels"
            )
            for fname in frame_names_sorted:
                for obj in frames_filled.get(fname, {}).get("objects", []):
                    lbl = obj.get("label")
                    if lbl not in static_union_map_final:
                        continue
                    _set_corners_final(obj, static_union_map_final[lbl])

        # ---- 7) Assemble + save ----
        world4d_annotations = {
            "video_id": video_id,
            "frames": frames_filled,
            "frame_names": frame_names_sorted,
            "all_labels": sorted(all_labels),
            "static_labels": static_labels_in_3d,
            "corrected_floor_transform": {
                "combined_transform_4x4": T_4x4.tolist(),
                "source": cft.get("source", "unknown"),
            },
            "meta": {
                "num_frames": len(frame_names_sorted),
                "total_objects_per_frame": {
                    fname: len(frames_filled.get(fname, {}).get("objects", []))
                    for fname in frame_names_sorted
                },
                "generator": "corrected_4d_bbox_generator",
            },
        }

        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(world4d_annotations, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[corrected-4d][{video_id}] Saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def generate_from_corrections_only(
        self,
        *,
        overwrite: bool = False,
    ) -> Dict[str, bool]:
        """
        Process only videos that have corrected world bboxes.
        Discovers videos from ``bbox_annotations_3d_obb_corrected/``.
        """
        results = {}
        if not self.bbox_3d_obb_corrected_root_dir.exists():
            print(
                f"[corrected-4d] corrected bbox dir not found: "
                f"{self.bbox_3d_obb_corrected_root_dir}"
            )
            return results

        correction_files = sorted(
            self.bbox_3d_obb_corrected_root_dir.glob("*.pkl"),
        )
        print(
            f"[corrected-4d] found {len(correction_files)} corrected bbox files"
        )

        for pkl_path in tqdm(correction_files, desc="Corrected 4D"):
            video_id = pkl_path.stem + ".mp4"
            try:
                out = self.generate_corrected_video_4d_annotations(
                    video_id, overwrite=overwrite,
                )
                results[video_id] = out is not None
            except Exception as e:
                print(f"[corrected-4d][{video_id}] error: {e}")
                import traceback
                traceback.print_exc()
                results[video_id] = False

        return results

    def generate_all(
        self,
        dataloader,
        split: str,
        *,
        overwrite: bool = False,
    ) -> Dict[str, bool]:
        """Generate corrected 4D bboxes for all videos in a split."""
        results = {}
        for data in tqdm(dataloader, desc=f"Corrected 4D [split={split}]"):
            video_id = data["video_id"]
            if get_video_belongs_to_split(video_id) != split:
                continue
            try:
                out = self.generate_corrected_video_4d_annotations(
                    video_id, overwrite=overwrite,
                )
                results[video_id] = out is not None
            except Exception as e:
                print(f"[corrected-4d][{video_id}] error: {e}")
                import traceback
                traceback.print_exc()
                results[video_id] = False
        return results

    # ------------------------------------------------------------------
    # Visualization (from saved PKLs, no recomputation)
    # ------------------------------------------------------------------

    def visualize_from_saved(
        self,
        video_id: str,
        *,
        app_id: str = "Corrected-4DBBox",
        img_maxsize: int = 480,
        vis_floor: bool = True,
    ) -> None:
        """Launch rerun visualization from a saved corrected 4D bbox PKL."""
        sys.path.insert(0, os.path.dirname(__file__))
        from corrected_bbox_vis import rerun_visualize_corrected_bboxes

        pkl_path = self.bbox_4d_corrected_root_dir / f"{video_id[:-4]}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"[vis] Missing corrected 4D bbox PKL: {pkl_path}\n"
                f"Run generation first."
            )

        rerun_visualize_corrected_bboxes(
            video_id=video_id,
            pkl_path=str(pkl_path),
            dynamic_scene_dir_path=str(self.dynamic_scene_dir_path),
            idx_to_frame_idx_path_fn=self.idx_to_frame_idx_path,
            app_id=app_id,
            img_maxsize=img_maxsize,
            vis_floor=vis_floor,
            frames_key="frames",
        )


# =====================================================================
# CLI
# =====================================================================

def load_dataset(ag_root_directory: str):
    from dataloader.ag_dataset import StandardAG
    from torch.utils.data import DataLoader

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
        train_dataset, shuffle=True, collate_fn=lambda b: b[0],
        pin_memory=False, num_workers=0,
    )
    dataloader_test = DataLoader(
        test_dataset, shuffle=False, collate_fn=lambda b: b[0], pin_memory=False,
    )
    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate corrected 4D bbox annotations from corrected world bboxes.",
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path", type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument(
        "--video", type=str, default=None, help="Process a single video",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--corrections-only", action="store_true",
        help="Only process videos that have corrected world bboxes",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize saved corrected 4D bboxes with rerun (requires --video)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    generator = Corrected4DBBoxGenerator(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )

    if args.visualize:
        if not args.video:
            print("ERROR: --visualize requires --video")
            return
        generator.visualize_from_saved(args.video)
        return

    if args.video:
        generator.generate_corrected_video_4d_annotations(
            args.video, overwrite=args.overwrite,
        )
        return

    # Default: process only videos that have corrected world bboxes
    # (skip-if-exists is handled inside generate_corrected_video_4d_annotations)
    results = generator.generate_from_corrections_only(
        overwrite=args.overwrite,
    )
    success = sum(1 for v in results.values() if v)
    print(f"\n[Summary] {success}/{len(results)} videos processed successfully")


def main_sample():
    """Process a single sample video, then launch rerun visualization."""
    args = parse_args()
    video_id = args.video or "001YG.mp4"

    generator = Corrected4DBBoxGenerator(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )

    # Generate (skips if already exists unless --overwrite)
    out = generator.generate_corrected_video_4d_annotations(
        video_id, overwrite=args.overwrite,
    )
    print(f"[main_sample] Generation result for {video_id}: {'SUCCESS' if out is not None else 'FAILED'}")

    if out is not None:
        generator.visualize_from_saved(video_id)


if __name__ == "__main__":
    # main()
    main_sample()

