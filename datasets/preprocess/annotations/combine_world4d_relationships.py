#!/usr/bin/env python3
"""
combine_world4d_relationships.py
=================================
Merge augmented relationship annotations (GT + RAG) with world4D 3D bounding
box annotations into a unified per-video PKL file.

Inputs:
  - ``<ag_root>/world_rel_annotations/<phase>/<video>.pkl``
    (from ``augment_relationships.py``)
  - ``<ag_root>/world_annotations/bbox_annotations_4d/<video>.pkl``
    (from ``frame_to_world4D_annotations.py``)

Output:
  - ``<ag_root>/world4d_rel_annotations/<phase>/<video>.pkl``

Pipeline:
  1) Load augmented_rel PKL (GT + RAG relationships, 2D bboxes, filtered frames).
  2) Load world4D PKL (3D OBBs, camera poses, floor mesh, object permanence).
  3) Per-frame: verify object label sets match, extract person 3D data,
     merge each object's relationship labels with its 3D bbox data.
  4) Filter camera_poses to only frames present in augmented_rel.
  5) Save combined PKL; log label mismatches to ``verify.txt``.

OUTPUT PKL STRUCTURE
--------------------
{
    "video_id":        str,
    "num_frames":      int,
    "all_labels":      List[str],              # sorted unique short-form labels

    "frames": {
        "video_id/000042.png": {
            "person_info": {
                # 2D (from augmented_rel)
                "bbox_2d":             np.ndarray (4,),    # xyxy person bbox
                "bbox_size":           (w, h),             # frame dimensions
                # 3D (from world4D person object)
                "corners_world":       np.ndarray (8,3),   # WORLD-space OBB
                "corners_final":       np.ndarray (8,3),   # FINAL-space OBB
                "obb_final":           dict,               # {"corners_final": ...}
                "world4d_filled":      bool,
                "world4d_fill_method": str,
            },
            "object_info_list": [
                {
                    # From augmented_rel
                    "class":                    str,        # full AG name
                    "label":                    str,        # short-form
                    "bbox_2d":                  np.ndarray | None,  # xyxy
                    "visible":                  bool,
                    "attention_relationship":   list[str],
                    "contacting_relationship":  list[str],
                    "spatial_relationship":     list[str],
                    "source":                   "gt"|"rag",
                    "attention_scores":         dict | None,
                    "contacting_scores":        dict | None,
                    "spatial_scores":           dict | None,
                    # From world4D (matched by short-form label)
                    "corners_world":            np.ndarray (8,3),
                    "corners_final":            np.ndarray (8,3),
                    "obb_final":                dict,
                    "world4d_filled":           bool,
                    "world4d_fill_method":      str,
                    "world4d_source":           "gt"|"gdino",
                },
                ...
            ],
        },
    },

    # Filtered camera poses (only valid frames, ordered)
    "camera_poses":      np.ndarray (N, 4, 4),   # FINAL coords
    "camera_frame_keys": List[str],              # corresponding frame keys

    # Pass-through from world4D
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

    # Pass-through from augmented_rel
    "rag_model":          str,
    "rag_mode":           str,
    "filter_stats":       dict,
    "augmentation_stats": dict,
}

COORDINATE SYSTEMS
------------------
- corners_world:  Raw Pi3 reconstruction space.
- corners_final:  Floor-aligned space (Y=up, XZ=floor plane).
- camera_poses:   Already in FINAL coords.
- floor_final:    Already in FINAL coords.

Usage:
    python datasets/preprocess/annotations/combine_world4d_relationships.py \\
        --ag_root_directory /data/rohith/ag \\
        --phase train --rag_mode predcls --rag_model qwen3vl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Label normalization (short-form ↔ full AG name)
# ---------------------------------------------------------------------------

LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}
LABEL_DENORMALIZE_MAP = {v: k for k, v in LABEL_NORMALIZE_MAP.items()}


def _to_short(label: str) -> str:
    """Normalize a full AG label to short form.

    ``"closet/cabinet"`` → ``"closet"``, ``"bed"`` → ``"bed"``.
    """
    return LABEL_NORMALIZE_MAP.get(label, label)


def _to_full(label: str) -> str:
    """Expand a short-form label back to the full AG name.

    ``"closet"`` → ``"closet/cabinet"``, ``"bed"`` → ``"bed"``.
    """
    return LABEL_DENORMALIZE_MAP.get(label, label)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_CODE_ROOT = Path(__file__).resolve().parents[3]  # WorldSGG/
_LOG_DIR = _CODE_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("combine_w4d_rel")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_ch)

_fh = logging.FileHandler(
    _LOG_DIR / "combine_world4d_relationships.log", mode="a",
)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# 3D field extraction helpers
# ---------------------------------------------------------------------------

def _extract_3d_fields(w4d_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the relevant 3D fields from a world4D object dict."""
    obb_fp = w4d_obj.get("obb_floor_parallel", {})
    return {
        "corners_world": np.asarray(obb_fp.get("corners_world", []), dtype=np.float32)
                         if obb_fp.get("corners_world") is not None else None,
        "corners_final": np.asarray(w4d_obj["corners_final"], dtype=np.float32)
                         if w4d_obj.get("corners_final") is not None else None,
        "obb_final": w4d_obj.get("obb_final", None),
        "world4d_filled": w4d_obj.get("world4d_filled", False),
        "world4d_fill_method": w4d_obj.get("world4d_fill_method", "detected"),
        "world4d_source": w4d_obj.get("source", "gt"),
    }


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(
    video_id: str,
    rel_pkl_path: Path,
    w4d_pkl_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Merge augmented_rel + world4D for a single video.

    Returns
    -------
    success : bool
    mismatch_lines : list[str]
        Verification lines for label mismatches (empty if clean).
    """
    if output_path.exists() and not overwrite:
        logger.debug(f"[{video_id}] Skipping: output exists at {output_path}")
        return True, [], None

    # ------------------------------------------------------------------
    # 1) Load PKLs
    # ------------------------------------------------------------------
    if not rel_pkl_path.exists():
        logger.warning(f"[{video_id}] Augmented rel PKL not found: {rel_pkl_path}")
        return False, [], None

    if not w4d_pkl_path.exists():
        logger.warning(f"[{video_id}] World4D PKL not found: {w4d_pkl_path}")
        return False, [], None

    with open(rel_pkl_path, "rb") as f:
        rel_data = pickle.load(f)

    with open(w4d_pkl_path, "rb") as f:
        w4d_data = pickle.load(f)

    # ------------------------------------------------------------------
    # 2) Build stem→camera index from world4D
    # ------------------------------------------------------------------
    w4d_frame_stems = w4d_data.get("frame_stems", [])
    w4d_camera_poses = w4d_data.get("camera_poses", None)
    stem_to_cam_idx: Dict[str, int] = {
        s: i for i, s in enumerate(w4d_frame_stems)
    }

    w4d_frames = w4d_data.get("frames", {})

    # ------------------------------------------------------------------
    # 3) Per-frame merge
    # ------------------------------------------------------------------
    rel_object_bbox = rel_data.get("object_bbox", {})
    rel_person_bbox = rel_data.get("person_bbox", {})

    combined_frames: Dict[str, Dict[str, Any]] = {}
    all_labels: Set[str] = set()
    mismatch_lines: List[str] = []

    valid_cam_indices: List[int] = []
    camera_frame_keys: List[str] = []

    for frame_key in sorted(rel_object_bbox.keys()):
        frame_name = frame_key.split("/")[-1]  # "000042.png"
        frame_stem = frame_name.replace(".png", "")

        # ---- World4D lookup ----
        w4d_frame_rec = w4d_frames.get(frame_name, {})
        w4d_objects = w4d_frame_rec.get("objects", [])

        # Build label→obj map from world4D (short-form keys)
        w4d_label_map: Dict[str, Dict[str, Any]] = {}
        w4d_person_obj: Optional[Dict[str, Any]] = None

        for obj in w4d_objects:
            lbl = obj.get("label", "")
            if lbl == "person":
                w4d_person_obj = obj
            else:
                # world4D already uses short-form labels
                if lbl not in w4d_label_map:
                    w4d_label_map[lbl] = obj

        # ---- Verify object label sets ----
        rel_objs = rel_object_bbox.get(frame_key, [])
        rel_labels = {_to_short(o["class"]) for o in rel_objs}
        w4d_labels = set(w4d_label_map.keys())

        if rel_labels != w4d_labels:
            rel_only = rel_labels - w4d_labels
            w4d_only = w4d_labels - rel_labels
            line = (
                f"[{video_id}] Frame {frame_key}: label mismatch\n"
                f"  rel_only:  {sorted(rel_only) if rel_only else '{}'}\n"
                f"  w4d_only:  {sorted(w4d_only) if w4d_only else '{}'}"
            )
            mismatch_lines.append(line)
            logger.debug(line)

        # ---- Person info ----
        person_rel = rel_person_bbox.get(frame_key, {})
        person_info: Dict[str, Any] = {
            # 2D from augmented_rel
            "bbox_2d": person_rel.get("bbox", None),
            "bbox_size": person_rel.get("bbox_size", (0, 0)),
        }
        if w4d_person_obj is not None:
            person_3d = _extract_3d_fields(w4d_person_obj)
            person_info.update(person_3d)
        else:
            # No person in world4D — fill with None
            person_info.update({
                "corners_world": None,
                "corners_final": None,
                "obb_final": None,
                "world4d_filled": None,
                "world4d_fill_method": None,
                "world4d_source": None,
            })

        # ---- Object info list ----
        object_info_list: List[Dict[str, Any]] = []

        for rel_obj in rel_objs:
            cls_full = rel_obj["class"]          # full AG name
            cls_short = _to_short(cls_full)      # short-form for matching
            all_labels.add(cls_short)

            # Base from augmented_rel
            merged: Dict[str, Any] = {
                "class": cls_full,
                "label": cls_short,
                "bbox_2d": rel_obj.get("bbox", None),
                "visible": rel_obj.get("visible", True),
                "attention_relationship": rel_obj.get("attention_relationship", []),
                "contacting_relationship": rel_obj.get("contacting_relationship", []),
                "spatial_relationship": rel_obj.get("spatial_relationship", []),
                "source": rel_obj.get("source", "gt"),
                "attention_scores": rel_obj.get("attention_scores", None),
                "contacting_scores": rel_obj.get("contacting_scores", None),
                "spatial_scores": rel_obj.get("spatial_scores", None),
            }

            # Merge 3D from world4D
            w4d_match = w4d_label_map.get(cls_short)
            if w4d_match is not None:
                merged.update(_extract_3d_fields(w4d_match))
            else:
                # No world4D match (shouldn't happen normally, logged in verify)
                merged.update({
                    "corners_world": None,
                    "corners_final": None,
                    "obb_final": None,
                    "world4d_filled": None,
                    "world4d_fill_method": None,
                    "world4d_source": None,
                })

            object_info_list.append(merged)

        combined_frames[frame_key] = {
            "person_info": person_info,
            "object_info_list": object_info_list,
        }

        # ---- Camera pose index ----
        if frame_stem in stem_to_cam_idx:
            valid_cam_indices.append(stem_to_cam_idx[frame_stem])
            camera_frame_keys.append(frame_key)

    # ------------------------------------------------------------------
    # 4) Filter camera poses
    # ------------------------------------------------------------------
    camera_poses_filtered = None
    if w4d_camera_poses is not None and valid_cam_indices:
        camera_poses_filtered = np.asarray(w4d_camera_poses, dtype=np.float32)[
            valid_cam_indices
        ]

    # ------------------------------------------------------------------
    # 5) Build and save output
    # ------------------------------------------------------------------
    output_record = {
        "video_id": video_id,
        "num_frames": len(combined_frames),
        "all_labels": sorted(all_labels),

        "frames": combined_frames,

        # Filtered camera poses
        "camera_poses": camera_poses_filtered,
        "camera_frame_keys": camera_frame_keys,

        # Pass-through from world4D
        "floor_mesh": w4d_data.get("floor_mesh", None),
        "global_floor_sim": w4d_data.get("global_floor_sim", None),
        "world_to_final": w4d_data.get("world_to_final", None),
        "floor_final": w4d_data.get("floor_final", None),

        # Pass-through from augmented_rel
        "rag_model": rel_data.get("rag_model", ""),
        "rag_mode": rel_data.get("rag_mode", ""),
        "filter_stats": rel_data.get("filter_stats", {}),
        "augmentation_stats": rel_data.get("augmentation_stats", {}),
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_record, f)

    n_objs = sum(
        len(fr["object_info_list"]) for fr in combined_frames.values()
    )
    cam_shape = (
        camera_poses_filtered.shape if camera_poses_filtered is not None
        else None
    )
    logger.info(
        f"[{video_id}] Saved {len(combined_frames)} frames, "
        f"{n_objs} objects, camera={cam_shape} → {output_path.name}"
    )

    return True, mismatch_lines, output_record


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine augmented relationship annotations with world4D "
            "3D bounding box annotations into a unified per-video PKL."
        ),
    )
    parser.add_argument(
        "--ag_root_directory", type=str, default="/data/rohith/ag",
    )
    parser.add_argument(
        "--phase", type=str, default="train", choices=["train", "test"],
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--rag_mode", type=str, default="predcls",
        help="RAG mode (pass-through metadata)",
    )
    parser.add_argument(
        "--rag_model", type=str, default="qwen3vl",
        help="RAG model name (pass-through metadata)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Process a single video (e.g. '001YG.mp4')",
    )
    return parser.parse_args()


def main():
    import random

    args = parse_args()
    ag_root = Path(args.ag_root_directory)

    rel_dir = ag_root / "world_rel_annotations" / args.phase
    w4d_dir = ag_root / "world_annotations" / "bbox_annotations_4d"
    output_dir = ag_root / "world4d_rel_annotations" / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Augmented rel dir:  {rel_dir}")
    logger.info(f"World4D dir:        {w4d_dir}")
    logger.info(f"Output dir:         {output_dir}")
    logger.info(f"Phase: {args.phase}, Mode: {args.rag_mode}, Model: {args.rag_model}")

    # ------------------------------------------------------------------
    # Discover videos
    # ------------------------------------------------------------------
    if args.video:
        vid = args.video
        if not vid.endswith(".mp4"):
            vid = f"{vid}.mp4"
        video_ids = [vid]
    else:
        if not rel_dir.exists():
            logger.error(f"Augmented rel directory not found: {rel_dir}")
            sys.exit(1)
        video_ids = sorted([
            p.stem for p in rel_dir.glob("*.pkl")
        ])
        # video_ids are like "001YG.mp4" (stem of "001YG.mp4.pkl")
        # Actually check — the pkl is named <video_id>.pkl where video_id = "001YG.mp4"
        video_ids = sorted([
            p.name.replace(".pkl", "") for p in rel_dir.glob("*.pkl")
        ])

    if not video_ids:
        logger.error("No videos found to process.")
        sys.exit(1)

    random.shuffle(video_ids)
    logger.info(f"Processing {len(video_ids)} videos (overwrite={args.overwrite})")

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------
    success_count = 0
    error_count = 0
    all_mismatches: List[str] = []

    for video_id in tqdm(video_ids, desc=f"Combining ({args.phase})"):
        # Derive video stem for world4D PKL (world4D uses stem without .mp4)
        video_stem = video_id.replace(".mp4", "")

        rel_pkl = rel_dir / f"{video_id}.pkl"
        w4d_pkl = w4d_dir / f"{video_stem}.pkl"
        out_pkl = output_dir / f"{video_id}.pkl"

        try:
            ok, mismatches, output_record = process_video(
                video_id=video_id,
                rel_pkl_path=rel_pkl,
                w4d_pkl_path=w4d_pkl,
                output_path=out_pkl,
                overwrite=args.overwrite,
            )
            if ok:
                success_count += 1
            all_mismatches.extend(mismatches)

        except Exception as e:
            logger.error(f"[{video_id}] Error: {e}", exc_info=True)
            error_count += 1

    # ------------------------------------------------------------------
    # Write verify.txt
    # ------------------------------------------------------------------
    verify_path = output_dir / "verify.txt"
    with open(verify_path, "w", encoding="utf-8") as vf:
        if all_mismatches:
            vf.write(f"Label mismatches: {len(all_mismatches)} entries\n")
            vf.write("=" * 60 + "\n\n")
            for line in all_mismatches:
                vf.write(line + "\n\n")
        else:
            vf.write("No label mismatches found.\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Done: {success_count}/{len(video_ids)} succeeded, "
        f"{error_count} errors."
    )
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Verify:     {verify_path} ({len(all_mismatches)} mismatches)")
    logger.info(f"Log:        {_LOG_DIR / 'combine_world4d_relationships.log'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
