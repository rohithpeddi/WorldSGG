#!/usr/bin/env python3
"""
augment_relationships.py
========================
Merge **observed** relationship annotations (from ``object_bbox_and_relationship.pkl``)
with **RAG-predicted** missing-object relationships (per-video pkl files from
``process_ag_rag.py``) into combined per-video pkl files.

Output directory: ``<ag_root>/world_annotations/augmented_relationships/``

For each video and each annotated frame:
  - ``observed``: objects visible in the frame with GT relationship labels
  - ``missing``:  objects NOT visible in the frame but predicted by the RAG pipeline

This is an intermediate format; the next step augments these with 3D bbox annotations.
"""

import argparse
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Relationship label vocabulary (Action Genome)
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

ATTENTION_RELATIONSHIPS = ["looking_at", "not_looking_at", "unsure"]
CONTACTING_RELATIONSHIPS = [
    "carrying", "covered_by", "drinking_from", "eating",
    "have_it_on_the_back", "holding", "leaning_on", "lying_on",
    "not_contacting", "other_relationship", "sitting_on", "standing_on",
    "touching", "twisting", "wearing", "wiping", "writing_on",
]
SPATIAL_RELATIONSHIPS = [
    "above", "beneath", "in_front_of", "behind", "on_the_side_of", "in",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GT annotation loader
# ---------------------------------------------------------------------------

class GTAnnotationLoader:
    """Load GT annotations from ``object_bbox_and_relationship.pkl``
    and ``person_bbox.pkl``."""

    def __init__(self, ag_root_directory: str):
        self.ag_root = Path(ag_root_directory)
        self.annotations_path = self.ag_root / "annotations"

        person_path = self.annotations_path / "person_bbox.pkl"
        object_path = self.annotations_path / "object_bbox_and_relationship.pkl"

        logger.info(f"Loading person_bbox.pkl ...")
        with open(person_path, "rb") as f:
            self.person_bbox: Dict[str, Any] = pickle.load(f)

        logger.info(f"Loading object_bbox_and_relationship.pkl ...")
        with open(object_path, "rb") as f:
            self.object_bbox: Dict[str, Any] = pickle.load(f)

        logger.info(
            f"Loaded {len(self.person_bbox)} person entries, "
            f"{len(self.object_bbox)} object entries."
        )

    # ---- Video / frame enumeration ------------------------------------

    def get_all_video_ids(self, phase: str = "testing") -> List[str]:
        """Return sorted list of unique video IDs for the given phase."""
        videos: Set[str] = set()
        for key in self.object_bbox:
            metadata = self.object_bbox[key][0].get("metadata", {})
            if metadata.get("set") == phase:
                video_id = key.split("/")[0]
                videos.add(video_id)
        return sorted(videos)

    def get_video_frame_keys(
        self, video_id: str, phase: str = "testing"
    ) -> List[str]:
        """Return sorted list of frame keys (``video_id/frame_name``) for
        a video in the given phase."""
        frame_keys = []
        prefix = video_id + "/"
        for key in self.object_bbox:
            if key.startswith(prefix):
                metadata = self.object_bbox[key][0].get("metadata", {})
                if metadata.get("set") == phase:
                    frame_keys.append(key)
        return sorted(frame_keys)

    # ---- Per-frame GT extraction --------------------------------------

    def get_observed_for_frame(
        self, frame_key: str,
    ) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Extract person bbox and observed object annotations for a single
        frame.

        Returns
        -------
        person_bbox : np.ndarray | None
            Person bounding box for the frame.
        observed : list[dict]
            Each dict has: class (int), label (str), bbox (list[float] xyxy | None),
            visible (bool), attention_relationship, contacting_relationship,
            spatial_relationship (each list[str]), source="gt".
        """
        # Person bbox
        person_entry = self.person_bbox.get(frame_key)
        person_bb = None
        if person_entry is not None:
            person_bb = person_entry.get("bbox", None)

        # Object annotations
        obj_entries = self.object_bbox.get(frame_key, [])
        observed: List[Dict[str, Any]] = []

        for obj_ann in obj_entries:
            if not obj_ann.get("visible", False):
                continue

            cls_raw = obj_ann.get("class")
            if cls_raw is None:
                continue

            # Resolve class name → int index
            if isinstance(cls_raw, int):
                if cls_raw <= 0 or cls_raw >= len(OBJECT_CLASSES):
                    continue
                class_idx = cls_raw
                label = OBJECT_CLASSES[cls_raw]
            elif isinstance(cls_raw, str):
                label = cls_raw
                class_idx = NAME_TO_IDX.get(label, -1)
                if class_idx <= 0:
                    continue
            else:
                continue

            if label in ("person", "__background__"):
                continue

            # Relationship labels (already strings in the raw pkl)
            att_rel = obj_ann.get("attention_relationship", [])
            cont_rel = obj_ann.get("contacting_relationship", [])
            spa_rel = obj_ann.get("spatial_relationship", [])

            # Bbox (raw format is [x, y, w, h] in the pkl)
            raw_bbox = obj_ann.get("bbox", None)
            bbox = None
            if raw_bbox is not None:
                if isinstance(raw_bbox, np.ndarray):
                    raw_bbox = raw_bbox.tolist()
                if len(raw_bbox) == 4:
                    bbox = raw_bbox  # keep raw format; consumer can convert

            observed.append({
                "class": class_idx,
                "label": label,
                "bbox": bbox,
                "visible": True,
                "attention_relationship": list(att_rel) if att_rel else ["unsure"],
                "contacting_relationship": list(cont_rel) if cont_rel else ["not_contacting"],
                "spatial_relationship": list(spa_rel) if spa_rel else ["in_front_of"],
                "source": "gt",
            })

        return person_bb, observed


# ---------------------------------------------------------------------------
# RAG prediction loader
# ---------------------------------------------------------------------------

def load_rag_predictions(
    rag_results_dir: str, mode: str, model_name: str, video_id: str,
) -> Optional[Dict[str, Any]]:
    """Load the RAG prediction pkl for a single video."""
    pkl_path = Path(rag_results_dir) / mode / model_name / f"{video_id}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def extract_missing_for_frame(
    rag_data: Dict[str, Any], frame_stem: str,
) -> List[Dict[str, Any]]:
    """
    Extract RAG-predicted missing-object relationships for a frame.

    ``frame_stem`` is just the numeric part (e.g. ``"000042"``), which is
    the key used in the RAG output's ``frames`` dict.

    Returns a list of dicts in the same schema as observed objects but with
    ``bbox=None``, ``visible=False``, ``source="rag"``.
    """
    frames = rag_data.get("frames", {})
    frame_data = frames.get(frame_stem)
    if frame_data is None:
        return []

    missing: List[Dict[str, Any]] = []
    for pred in frame_data.get("predictions", []):
        obj_name = pred.get("missing_object", "")
        if not obj_name:
            continue

        class_idx = NAME_TO_IDX.get(obj_name, -1)
        if class_idx <= 0:
            continue

        # Extract relationship labels, filter out "unknown"
        att = pred.get("attention", "unknown")
        if isinstance(att, list):
            att = att[0] if att else "unknown"
        att_list = [att] if att != "unknown" and att in ATTENTION_RELATIONSHIPS else ["unsure"]

        cont = pred.get("contacting", ["unknown"])
        if isinstance(cont, str):
            cont = [cont]
        cont_list = [c for c in cont if c != "unknown" and c in CONTACTING_RELATIONSHIPS]
        if not cont_list:
            cont_list = ["not_contacting"]

        spa = pred.get("spatial", ["unknown"])
        if isinstance(spa, str):
            spa = [spa]
        spa_list = [s for s in spa if s != "unknown" and s in SPATIAL_RELATIONSHIPS]
        if not spa_list:
            spa_list = ["in_front_of"]

        missing.append({
            "class": class_idx,
            "label": obj_name,
            "bbox": None,
            "visible": False,
            "attention_relationship": att_list,
            "contacting_relationship": cont_list,
            "spatial_relationship": spa_list,
            "source": "rag",
        })

    return missing


# ---------------------------------------------------------------------------
# Frame-stem extraction helper
# ---------------------------------------------------------------------------

def frame_key_to_stem(frame_key: str) -> str:
    """``'00T1E/000042.png'`` → ``'000042'``"""
    fname = frame_key.split("/")[-1]
    return os.path.splitext(fname)[0]


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_video(
    video_id: str,
    gt_loader: GTAnnotationLoader,
    rag_results_dir: str,
    mode: str,
    model_name: str,
    phase: str,
    output_dir: Path,
) -> bool:
    """Process a single video: merge GT + RAG and save combined pkl."""
    save_path = output_dir / f"{video_id}.pkl"
    if save_path.exists():
        logger.info(f"Skipping {video_id}: output already exists at {save_path}")
        return True

    # 1. Get all frame keys for this video
    frame_keys = gt_loader.get_video_frame_keys(video_id, phase=phase)
    if not frame_keys:
        logger.warning(f"Skipping {video_id}: no frames found for phase={phase}")
        return False

    # 2. Load RAG predictions (may be None)
    rag_data = load_rag_predictions(rag_results_dir, mode, model_name, video_id)
    if rag_data is None:
        logger.info(f"[{video_id}] No RAG predictions found — saving observed-only")

    # 3. Process each frame
    all_object_labels: Set[str] = set()
    frames_combined: Dict[str, Any] = {}

    for frame_key in frame_keys:
        person_bb, observed = gt_loader.get_observed_for_frame(frame_key)

        # Collect observed object labels
        for obj in observed:
            all_object_labels.add(obj["label"])

        # Extract missing-object predictions
        stem = frame_key_to_stem(frame_key)
        missing: List[Dict[str, Any]] = []
        if rag_data is not None:
            missing = extract_missing_for_frame(rag_data, stem)
            for obj in missing:
                all_object_labels.add(obj["label"])

        frames_combined[frame_key] = {
            "person_bbox": person_bb,
            "observed": observed,
            "missing": missing,
        }

    # 4. Save combined result
    output_record = {
        "video_id": video_id,
        "video_objects": sorted(all_object_labels),
        "num_frames": len(frames_combined),
        "frames": frames_combined,
    }
    with open(save_path, "wb") as f:
        pickle.dump(output_record, f)

    n_obs = sum(len(fd["observed"]) for fd in frames_combined.values())
    n_mis = sum(len(fd["missing"]) for fd in frames_combined.values())
    logger.info(
        f"[{video_id}] Saved {len(frames_combined)} frames "
        f"({n_obs} observed, {n_mis} missing) → {save_path}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge observed GT relationship annotations with RAG-predicted "
            "missing-object relationships into per-video pkl files."
        ),
    )
    parser.add_argument(
        "--ag_root_directory", type=str,
        default="/data/rohith/ag",
        help="Root directory of the Action Genome dataset",
    )
    parser.add_argument(
        "--rag_results_dir", type=str,
        default="/data/rohith/ag/rag_results",
        help="Directory with RAG output pkl files (<dir>/<mode>/<model_name>/<video>.pkl)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Output directory for combined pkl files. "
            "Defaults to <ag_root>/world_annotations/augmented_relationships/"
        ),
    )
    parser.add_argument(
        "--mode", type=str, default="predcls",
        choices=["predcls", "sgdet"],
        help="Evaluation mode (selects RAG subdirectory)",
    )
    parser.add_argument(
        "--model_name", type=str, default="kimikvl",
        help="VLM model name (selects RAG subdirectory)",
    )
    parser.add_argument(
        "--phase", type=str, default="testing",
        choices=["train", "testing"],
        help="Dataset phase to process",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many videos (for dev/debug)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve output dir
    if args.output_dir is None:
        output_dir = (
            Path(args.ag_root_directory)
            / "world_annotations"
            / "augmented_relationships"
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load GT annotations
    gt_loader = GTAnnotationLoader(args.ag_root_directory)

    # Get all video IDs for the phase
    video_ids = gt_loader.get_all_video_ids(phase=args.phase)
    logger.info(f"Found {len(video_ids)} videos for phase={args.phase}")

    if args.limit is not None:
        video_ids = video_ids[:args.limit]
        logger.info(f"Limited to {len(video_ids)} videos")

    # Process each video
    success_count = 0
    for i, video_id in enumerate(video_ids):
        logger.info(f"Processing [{i + 1}/{len(video_ids)}] {video_id}")
        ok = process_video(
            video_id=video_id,
            gt_loader=gt_loader,
            rag_results_dir=args.rag_results_dir,
            mode=args.mode,
            model_name=args.model_name,
            phase=args.phase,
            output_dir=output_dir,
        )
        if ok:
            success_count += 1

    logger.info(
        f"Done. {success_count}/{len(video_ids)} videos processed. "
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
