#!/usr/bin/env python3
"""
World Scene Graph Generator
============================

Merges two data sources per video into unified world scene graph PKLs:

1. **Augmented relationships** (``augmented_relationships/<video>.pkl``):
   Per-frame observed (GT) + missing (RAG-predicted) object relationships.

2. **Corrected 4D bboxes** (``bbox_annotations_4d_corrected/<video>.pkl``):
   Per-frame 3D bounding boxes (FINAL coords) with object permanence filling.

Output: ``world_scene_graph/<video>.pkl``

Usage:
    python world_scene_graph_generator.py --corrections-only
    python world_scene_graph_generator.py --video 001YG.mp4
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Object class vocabulary (Action Genome)
# ---------------------------------------------------------------------------

OBJECT_CLASSES = [
    "__background__", "person", "bag", "bed", "blanket", "book", "box",
    "broom", "chair", "closet/cabinet", "clothes", "cup/glass/bottle",
    "dish", "door", "doorknob", "doorway", "floor", "food", "groceries",
    "laptop", "light", "medicine", "mirror", "paper/notebook",
    "phone/camera", "picture", "pillow", "refrigerator", "sandwich",
    "shelf", "shoe", "sofa/couch", "table", "television", "towel",
    "vacuum", "window",
]
NAME_TO_IDX = {name: idx for idx, name in enumerate(OBJECT_CLASSES) if idx > 0}

# Normalized label map (same as used in bbox generators)
_LABEL_NORMALIZE = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}

# Reverse map for looking up class index from normalized label
_LABEL_DENORMALIZE = {v: k for k, v in _LABEL_NORMALIZE.items()}


def _normalize_label(label: str) -> str:
    return _LABEL_NORMALIZE.get(label, label)


def _class_idx_for_label(label: str) -> int:
    """Get class index, trying both normalized and original forms."""
    if label in NAME_TO_IDX:
        return NAME_TO_IDX[label]
    original = _LABEL_DENORMALIZE.get(label, label)
    return NAME_TO_IDX.get(original, -1)


def _frame_key_to_stem(frame_key: str) -> str:
    """``'00T1E/000042.png'`` → ``'000042'``"""
    return Path(frame_key).stem


def _frame_name_to_key(frame_name: str, video_id: str) -> str:
    """``'000042.png'`` → ``'00T1E/000042.png'``"""
    return f"{video_id}/{frame_name}"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_pkl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_video(
    video_id: str,
    rel_data: Dict[str, Any],
    bbox4d_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge augmented relationships + corrected 4D bboxes for a single video.

    For each frame present in *either* source, produces a unified object list:
      - Objects in both → merge (relationships + 3D geometry)
      - Objects in relationships only (missing/RAG) → include with corners_final=None
      - Objects in 4D bboxes only → include with default/empty relationships
    """
    # --- Extract frame maps from both sources ---
    rel_frames = rel_data.get("frames", {})
    bbox_frames = bbox4d_data.get("frames", {})
    bbox_frame_names_sorted = bbox4d_data.get("frame_names", [])
    static_labels = bbox4d_data.get("static_labels", [])
    all_bbox_labels = set(bbox4d_data.get("all_labels", []))
    corrected_floor_transform = bbox4d_data.get("corrected_floor_transform", None)

    # Build a lookup from frame stem → 4D bbox frame name
    # (bbox frames use names like "000042.png", rel frames use "video_id/000042.png")
    stem_to_bbox_frame: Dict[str, str] = {}
    for fname in bbox_frame_names_sorted:
        stem = Path(fname).stem
        stem_to_bbox_frame[stem] = fname

    # Collect all frame stems from both sources
    all_frame_stems: Set[str] = set()
    frame_key_map: Dict[str, str] = {}  # stem → rel frame_key

    for frame_key in rel_frames:
        stem = _frame_key_to_stem(frame_key)
        all_frame_stems.add(stem)
        frame_key_map[stem] = frame_key

    for fname in bbox_frame_names_sorted:
        stem = Path(fname).stem
        all_frame_stems.add(stem)

    # Sort frame stems numerically
    sorted_stems = sorted(
        all_frame_stems,
        key=lambda s: int(s) if s.isdigit() else s,
    )

    # --- Merge per frame ---
    merged_frames: Dict[str, Dict[str, Any]] = {}
    all_video_labels: Set[str] = set()

    for stem in sorted_stems:
        # Get relationship data for this frame
        rel_frame_key = frame_key_map.get(stem)
        rel_frame = rel_frames.get(rel_frame_key, {}) if rel_frame_key else {}

        # Get 4D bbox data for this frame
        bbox_frame_name = stem_to_bbox_frame.get(stem)
        bbox_frame = bbox_frames.get(bbox_frame_name, {}) if bbox_frame_name else {}

        # Person bbox from relationships
        person_bbox = rel_frame.get("person_bbox", None)

        # Collect relationship objects by normalized label
        rel_objects_by_label: Dict[str, Dict[str, Any]] = {}
        for obj in rel_frame.get("observed", []):
            lbl = _normalize_label(obj.get("label", ""))
            if lbl and lbl not in rel_objects_by_label:
                rel_objects_by_label[lbl] = obj
        for obj in rel_frame.get("missing", []):
            lbl = _normalize_label(obj.get("label", ""))
            if lbl and lbl not in rel_objects_by_label:
                rel_objects_by_label[lbl] = obj

        # Collect 4D bbox objects by label
        bbox_objects_by_label: Dict[str, Dict[str, Any]] = {}
        for obj in bbox_frame.get("objects", []):
            lbl = obj.get("label", "")
            if lbl and lbl not in bbox_objects_by_label:
                bbox_objects_by_label[lbl] = obj

        # Merge: union of all labels in this frame
        frame_labels = set(rel_objects_by_label.keys()) | set(bbox_objects_by_label.keys())
        all_video_labels.update(frame_labels)

        merged_objects: List[Dict[str, Any]] = []
        for lbl in sorted(frame_labels):
            rel_obj = rel_objects_by_label.get(lbl)
            bbox_obj = bbox_objects_by_label.get(lbl)

            merged = _merge_single_object(lbl, rel_obj, bbox_obj)
            merged_objects.append(merged)

        # Canonical frame key: video_id/stem.png
        canonical_key = f"{video_id}/{stem}.png"
        merged_frames[canonical_key] = {
            "person_bbox": person_bbox,
            "objects": merged_objects,
        }

    return {
        "video_id": video_id,
        "num_frames": len(merged_frames),
        "all_labels": sorted(all_video_labels),
        "static_labels": static_labels,
        "frames": merged_frames,
        "corrected_floor_transform": corrected_floor_transform,
    }


def _merge_single_object(
    label: str,
    rel_obj: Optional[Dict[str, Any]],
    bbox_obj: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge a single object's relationship data with its 3D geometry.

    Cases:
      - Both present → full merge
      - Relationship only → include with corners_final=None
      - Bbox only → include with empty relationships
    """
    merged: Dict[str, Any] = {"label": label}

    # Class index
    merged["class"] = _class_idx_for_label(label)

    # --- Relationship data ---
    if rel_obj is not None:
        merged["bbox_2d"] = rel_obj.get("bbox", None)
        merged["visible"] = rel_obj.get("visible", True)
        merged["attention_relationship"] = rel_obj.get("attention_relationship", [])
        merged["contacting_relationship"] = rel_obj.get("contacting_relationship", [])
        merged["spatial_relationship"] = rel_obj.get("spatial_relationship", [])
        merged["rel_source"] = rel_obj.get("source", "gt")
    else:
        merged["bbox_2d"] = None
        merged["visible"] = True  # has 3D bbox
        merged["attention_relationship"] = []
        merged["contacting_relationship"] = []
        merged["spatial_relationship"] = []
        merged["rel_source"] = "none"

    # --- 3D geometry (from 4D bboxes) ---
    if bbox_obj is not None:
        # AABB final corners
        aabb_final = bbox_obj.get("aabb_final")
        if isinstance(aabb_final, dict) and aabb_final.get("corners_final") is not None:
            merged["corners_final"] = aabb_final["corners_final"]
        elif bbox_obj.get("corners_final") is not None:
            merged["corners_final"] = bbox_obj["corners_final"]
        else:
            merged["corners_final"] = None

        merged["center_3d"] = bbox_obj.get("center", None)

        # OBB variants
        obb_fp = bbox_obj.get("obb_floor_parallel_final")
        if isinstance(obb_fp, dict) and obb_fp.get("corners_final") is not None:
            merged["obb_floor_parallel_corners"] = obb_fp["corners_final"]
        else:
            merged["obb_floor_parallel_corners"] = None

        obb_arb = bbox_obj.get("obb_arbitrary_final")
        if isinstance(obb_arb, dict) and obb_arb.get("corners_final") is not None:
            merged["obb_arbitrary_corners"] = obb_arb["corners_final"]
        else:
            merged["obb_arbitrary_corners"] = None

        merged["world4d_filled"] = bbox_obj.get("world4d_filled", False)
        merged["world4d_source_frame"] = bbox_obj.get("world4d_source_frame", None)
        merged["bbox_source"] = bbox_obj.get("source", "unknown")
    else:
        merged["corners_final"] = None
        merged["center_3d"] = None
        merged["obb_floor_parallel_corners"] = None
        merged["obb_arbitrary_corners"] = None
        merged["world4d_filled"] = False
        merged["world4d_source_frame"] = None
        merged["bbox_source"] = "none"

    # Combined source tag
    if rel_obj is not None and bbox_obj is not None:
        merged["source"] = "merged"
    elif rel_obj is not None:
        merged["source"] = merged["rel_source"]
    else:
        merged["source"] = merged["bbox_source"]

    return merged


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class WorldSceneGraphGenerator:
    """
    Generates unified world scene graph PKLs by merging augmented relationships
    with corrected 4D bounding boxes.
    """

    def __init__(
        self,
        ag_root_directory: str,
        augmented_rel_dir: Optional[str] = None,
        bbox_4d_corrected_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.ag_root = Path(ag_root_directory)
        self.world_annotations_dir = self.ag_root / "world_annotations"

        # Input directories
        if augmented_rel_dir:
            self.augmented_rel_dir = Path(augmented_rel_dir)
        else:
            self.augmented_rel_dir = (
                self.world_annotations_dir / "augmented_relationships"
            )

        if bbox_4d_corrected_dir:
            self.bbox_4d_corrected_dir = Path(bbox_4d_corrected_dir)
        else:
            self.bbox_4d_corrected_dir = (
                self.world_annotations_dir / "bbox_annotations_4d_corrected"
            )

        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.world_annotations_dir / "world_scene_graph"
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Single video
    # ------------------------------------------------------------------

    def generate_for_video(
        self,
        video_id: str,
        *,
        overwrite: bool = False,
    ) -> bool:
        """Generate world scene graph PKL for a single video."""
        out_path = self.output_dir / f"{video_id}.pkl"
        if out_path.exists() and not overwrite:
            logger.info(f"[world-sg][{video_id}] exists, skipping.")
            return True

        # Load augmented relationships
        rel_path = self.augmented_rel_dir / f"{video_id}.pkl"
        rel_data = _load_pkl(rel_path)

        # Load corrected 4D bboxes
        video_key = video_id.replace(".mp4", "")
        bbox_path = self.bbox_4d_corrected_dir / f"{video_key}.pkl"
        bbox4d_data = _load_pkl(bbox_path)

        if rel_data is None and bbox4d_data is None:
            logger.warning(
                f"[world-sg][{video_id}] no augmented rels AND no 4D bboxes. Skipping."
            )
            return False

        # Handle partial data
        if rel_data is None:
            logger.info(f"[world-sg][{video_id}] no augmented rels — bbox-only mode")
            rel_data = {"frames": {}}

        if bbox4d_data is None:
            logger.info(f"[world-sg][{video_id}] no 4D bboxes — rel-only mode")
            bbox4d_data = {"frames": {}, "frame_names": [], "static_labels": []}

        # Merge
        merged = merge_video(video_id, rel_data, bbox4d_data)

        # Save
        with open(out_path, "wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

        n_frames = merged["num_frames"]
        n_labels = len(merged["all_labels"])
        total_objs = sum(
            len(fd.get("objects", [])) for fd in merged["frames"].values()
        )
        logger.info(
            f"[world-sg][{video_id}] saved: {n_frames} frames, "
            f"{n_labels} labels, {total_objs} total object entries → {out_path}"
        )
        return True

    # ------------------------------------------------------------------
    # Batch: discover from available data
    # ------------------------------------------------------------------

    def generate_from_available(
        self,
        *,
        overwrite: bool = False,
    ) -> Dict[str, bool]:
        """
        Process all videos that have either augmented relationships
        or corrected 4D bboxes (or both).
        """
        # Discover video IDs from both sources
        video_ids: Set[str] = set()

        if self.augmented_rel_dir.exists():
            for pkl in self.augmented_rel_dir.glob("*.pkl"):
                video_ids.add(pkl.stem)

        if self.bbox_4d_corrected_dir.exists():
            for pkl in self.bbox_4d_corrected_dir.glob("*.pkl"):
                # 4D bbox PKLs are named without .mp4 extension
                video_ids.add(pkl.stem)

        # Normalize: we use video_id without .mp4 as the key
        video_ids_sorted = sorted(video_ids)
        logger.info(f"[world-sg] found {len(video_ids_sorted)} unique videos")

        results = {}
        for vid in tqdm(video_ids_sorted, desc="World SG"):
            try:
                ok = self.generate_for_video(vid, overwrite=overwrite)
                results[vid] = ok
            except Exception as e:
                logger.error(f"[world-sg][{vid}] error: {e}")
                import traceback
                traceback.print_exc()
                results[vid] = False

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge augmented relationships + corrected 4D bboxes "
            "into unified world scene graph PKLs."
        ),
    )
    parser.add_argument(
        "--ag_root_directory", type=str, default="/data/rohith/ag",
    )
    parser.add_argument(
        "--augmented_rel_dir", type=str, default=None,
        help="Dir with augmented relationship PKLs (default: <ag_root>/world_annotations/augmented_relationships)",
    )
    parser.add_argument(
        "--bbox_4d_corrected_dir", type=str, default=None,
        help="Dir with corrected 4D bbox PKLs (default: <ag_root>/world_annotations/bbox_annotations_4d_corrected)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output dir (default: <ag_root>/world_annotations/world_scene_graph)",
    )
    parser.add_argument(
        "--video", type=str, default=None, help="Process a single video",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    generator = WorldSceneGraphGenerator(
        ag_root_directory=args.ag_root_directory,
        augmented_rel_dir=args.augmented_rel_dir,
        bbox_4d_corrected_dir=args.bbox_4d_corrected_dir,
        output_dir=args.output_dir,
    )

    if args.video:
        ok = generator.generate_for_video(args.video, overwrite=args.overwrite)
        print(f"Result: {'SUCCESS' if ok else 'FAILED'}")
        return

    results = generator.generate_from_available(overwrite=args.overwrite)
    success = sum(1 for v in results.values() if v)
    print(f"\n[Summary] {success}/{len(results)} videos processed successfully")


if __name__ == "__main__":
    main()
