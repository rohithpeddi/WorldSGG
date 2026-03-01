#!/usr/bin/env python3
"""
augment_relationships.py
========================
Merge **observed** relationship annotations (from ``object_bbox_and_relationship.pkl``)
with **RAG-predicted** missing-object relationships (per-video pkl files from
``process_ag_rag.py``) into combined per-video pkl files.

Output directory: ``<ag_root>/world_rel_annotations/<phase>/``

For each video and each annotated frame:
  - ``observed``: objects visible in the frame with GT relationship labels
  - ``missing``:  objects NOT visible in the frame but predicted by the RAG pipeline

OUTPUT PKL STRUCTURE
--------------------
{
    "video_id":       str,
    "video_objects":  sorted list of all unique object label strings,
    "num_frames":     int,
    "frames": {
        "<video_id>/<frame>.png": {
            "person_bbox":  np.ndarray | None,
            "observed": [
                {
                    "class": int,             # AG class index (1-36)
                    "label": str,             # e.g. "closet/cabinet"
                    "bbox":  list[float]|None, # [x, y, w, h] or None
                    "visible": True,
                    "attention_relationship":  list[str],
                    "contacting_relationship": list[str],
                    "spatial_relationship":    list[str],
                    "source": "gt",
                },
                ...
            ],
            "missing": [
                {
                    "class": int,
                    "label": str,
                    "bbox":  None,
                    "visible": False,
                    "attention_relationship":  list[str],
                    "contacting_relationship": list[str],
                    "spatial_relationship":    list[str],
                    "attention_scores":        dict[str, float],
                    "contacting_scores":       dict[str, float],
                    "spatial_scores":          dict[str, float],
                    "source": "rag",
                },
                ...
            ],
        },
        ...
    },
}

Usage:
    python datasets/preprocess/annotations/augment_relationships.py \
        --ag_root_directory /data/rohith/ag \
        --rag_results_dir /data/rohith/ag/rag_results \
        --mode predcls --model_name qwen3vl --phase train
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
from tqdm import tqdm

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

# Normalized short-form → AG class index
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet", "cup/glass/bottle": "cup",
    "paper/notebook": "paper", "sofa/couch": "sofa",
    "phone/camera": "phone",
}
LABEL_DENORMALIZE_MAP = {v: k for k, v in LABEL_NORMALIZE_MAP.items()}

NAME_TO_IDX = {name: idx for idx, name in enumerate(OBJECT_CLASSES) if idx > 0}
# Also map short forms → idx
for _short, _full in LABEL_DENORMALIZE_MAP.items():
    NAME_TO_IDX[_short] = NAME_TO_IDX[_full]

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

# Map space-separated labels (from LLM) → underscore format
_LABEL_SPACE_TO_UNDERSCORE = {
    "looking at": "looking_at", "not looking at": "not_looking_at",
    "covered by": "covered_by", "drinking from": "drinking_from",
    "have it on the back": "have_it_on_the_back",
    "leaning on": "leaning_on", "lying on": "lying_on",
    "not contacting": "not_contacting", "other relationship": "other_relationship",
    "sitting on": "sitting_on", "standing on": "standing_on",
    "writing on": "writing_on",
    "in front of": "in_front_of", "on the side of": "on_the_side_of",
}

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
_CODE_ROOT = Path(__file__).resolve().parents[3]  # WorldSGG/
_LOG_DIR = _CODE_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("augment_rel")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
logger.addHandler(_ch)

_fh = logging.FileHandler(_LOG_DIR / "augment_relationships.log", mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_fh)


def _normalize_rel_label(label: str) -> str:
    """Normalize a relationship label: strip, lowercase, space→underscore."""
    label = label.strip().lower()
    return _LABEL_SPACE_TO_UNDERSCORE.get(label, label.replace(" ", "_"))


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

    def get_all_video_ids(self, phase: str = "train") -> List[str]:
        """Return sorted list of unique video IDs for the given phase.

        AG uses ``"train"`` for training and ``"testing"`` for test.
        """
        videos: Set[str] = set()
        for key in self.object_bbox:
            metadata = self.object_bbox[key][0].get("metadata", {})
            if metadata.get("set") == phase:
                video_id = key.split("/")[0]
                videos.add(video_id)
        return sorted(videos)

    def get_video_frame_keys(
        self, video_id: str, phase: str = "train"
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
        Extract person bbox and observed object annotations for a single frame.

        Returns
        -------
        person_bbox : np.ndarray | None
        observed : list[dict]
            Each dict: class (int), label (str), bbox, visible (True),
            attention_relationship, contacting_relationship,
            spatial_relationship (lists of underscore-formatted strings),
            source="gt".
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

            # Normalize relationship labels to underscore format
            att_rel = [_normalize_rel_label(r) for r in att_rel] if att_rel else ["unsure"]
            cont_rel = [_normalize_rel_label(r) for r in cont_rel] if cont_rel else ["not_contacting"]
            spa_rel = [_normalize_rel_label(r) for r in spa_rel] if spa_rel else ["in_front_of"]

            # Bbox (raw format varies in the pkl)
            raw_bbox = obj_ann.get("bbox", None)
            bbox = None
            if raw_bbox is not None:
                if isinstance(raw_bbox, np.ndarray):
                    raw_bbox = raw_bbox.tolist()
                if len(raw_bbox) == 4:
                    bbox = raw_bbox

            observed.append({
                "class": class_idx,
                "label": label,
                "bbox": bbox,
                "visible": True,
                "attention_relationship": att_rel,
                "contacting_relationship": cont_rel,
                "spatial_relationship": spa_rel,
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


def _extract_rel_label_and_score(
    entry: Any, valid_set: List[str], default: str,
) -> Tuple[str, float]:
    """Extract (label, score) from a RAG prediction entry.

    RAG output format can be:
      - dict: {"label": "...", "score": 0.8}
      - str:  "label_name"
    """
    if isinstance(entry, dict):
        label = _normalize_rel_label(entry.get("label", default))
        score = float(entry.get("score", 0.5))
    elif isinstance(entry, str):
        label = _normalize_rel_label(entry)
        score = 0.5  # no score available
    else:
        label = default
        score = 0.0

    if label not in valid_set:
        label = default
        score = 0.0
    return label, score


def extract_missing_for_frame(
    rag_data: Dict[str, Any], frame_stem: str,
) -> List[Dict[str, Any]]:
    """
    Extract RAG-predicted missing-object relationships for a frame.

    ``frame_stem`` is the numeric part (e.g. ``"000042"``).

    Returns a list of dicts with ``bbox=None``, ``visible=False``,
    ``source="rag"``, plus ``*_scores`` dicts for downstream weighting.
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

        # Resolve back to full AG label for consistency
        full_label = LABEL_DENORMALIZE_MAP.get(obj_name, obj_name)
        if full_label not in NAME_TO_IDX and obj_name in NAME_TO_IDX:
            full_label = obj_name

        # --- Attention (single label) ---
        att_raw = pred.get("attention", "unsure")
        if isinstance(att_raw, dict):
            att_label, att_score = _extract_rel_label_and_score(
                att_raw, ATTENTION_RELATIONSHIPS, "unsure"
            )
        elif isinstance(att_raw, list) and att_raw:
            att_label, att_score = _extract_rel_label_and_score(
                att_raw[0], ATTENTION_RELATIONSHIPS, "unsure"
            )
        elif isinstance(att_raw, str):
            att_label = _normalize_rel_label(att_raw)
            att_score = 0.5
            if att_label not in ATTENTION_RELATIONSHIPS:
                att_label, att_score = "unsure", 0.0
        else:
            att_label, att_score = "unsure", 0.0
        att_list = [att_label]
        att_scores = {att_label: att_score}

        # --- Contacting (multi-label) ---
        cont_raw = pred.get("contacting", ["not_contacting"])
        if isinstance(cont_raw, str):
            cont_raw = [cont_raw]
        elif isinstance(cont_raw, dict):
            cont_raw = [cont_raw]

        cont_list = []
        cont_scores = {}
        for c in cont_raw:
            lbl, sc = _extract_rel_label_and_score(
                c, CONTACTING_RELATIONSHIPS, "not_contacting"
            )
            if lbl not in cont_scores:  # deduplicate
                cont_list.append(lbl)
                cont_scores[lbl] = sc
        if not cont_list:
            cont_list = ["not_contacting"]
            cont_scores = {"not_contacting": 0.0}

        # --- Spatial (multi-label) ---
        spa_raw = pred.get("spatial", ["in_front_of"])
        if isinstance(spa_raw, str):
            spa_raw = [spa_raw]
        elif isinstance(spa_raw, dict):
            spa_raw = [spa_raw]

        spa_list = []
        spa_scores = {}
        for s in spa_raw:
            lbl, sc = _extract_rel_label_and_score(
                s, SPATIAL_RELATIONSHIPS, "in_front_of"
            )
            if lbl not in spa_scores:
                spa_list.append(lbl)
                spa_scores[lbl] = sc
        if not spa_list:
            spa_list = ["in_front_of"]
            spa_scores = {"in_front_of": 0.0}

        missing.append({
            "class": class_idx,
            "label": full_label,
            "bbox": None,
            "visible": False,
            "attention_relationship": att_list,
            "contacting_relationship": cont_list,
            "spatial_relationship": spa_list,
            "attention_scores": att_scores,
            "contacting_scores": cont_scores,
            "spatial_scores": spa_scores,
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
    overwrite: bool = False,
) -> bool:
    """Process a single video: merge GT + RAG and save combined pkl."""
    save_path = output_dir / f"{video_id}.pkl"
    if save_path.exists() and not overwrite:
        logger.debug(f"Skipping {video_id}: output already exists at {save_path}")
        return True

    # 1. Get all frame keys for this video
    frame_keys = gt_loader.get_video_frame_keys(video_id, phase=phase)
    if not frame_keys:
        logger.warning(f"Skipping {video_id}: no frames found for phase={phase}")
        return False

    # 2. Load RAG predictions (may be None)
    rag_data = load_rag_predictions(rag_results_dir, mode, model_name, video_id)
    if rag_data is None:
        logger.debug(f"[{video_id}] No RAG predictions found — saving observed-only")

    # 3. Process each frame
    all_object_labels: Set[str] = set()
    frames_combined: Dict[str, Any] = {}
    n_rag_total = 0

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
            n_rag_total += len(missing)

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
        "rag_model": model_name,
        "rag_mode": mode,
        "frames": frames_combined,
    }
    with open(save_path, "wb") as f:
        pickle.dump(output_record, f)

    n_obs = sum(len(fd["observed"]) for fd in frames_combined.values())
    n_mis = sum(len(fd["missing"]) for fd in frames_combined.values())
    logger.info(
        f"[{video_id}] {len(frames_combined)} frames "
        f"({n_obs} observed, {n_mis} missing) → {save_path.name}"
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
        default=None,
        help=(
            "Directory with RAG output pkl files (<dir>/<mode>/<model_name>/<video>.pkl). "
            "Defaults to <ag_root>/rag_results/"
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Output directory for combined pkl files. "
            "Defaults to <ag_root>/world_rel_annotations/<phase>/"
        ),
    )
    parser.add_argument(
        "--mode", type=str, default="predcls",
        choices=["predcls", "sgdet"],
        help="Evaluation mode (selects RAG subdirectory)",
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen3vl",
        help="VLM model name (selects RAG subdirectory)",
    )
    parser.add_argument(
        "--phase", type=str, default="train",
        choices=["train", "testing"],
        help="Dataset phase to process (default: train)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many videos (for dev/debug)",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Process only this video ID (e.g. '001YG')",
    )

    args = parser.parse_args()

    # Resolve RAG results dir
    if args.rag_results_dir is None:
        args.rag_results_dir = str(Path(args.ag_root_directory) / "rag_results")

    # Resolve output dir
    if args.output_dir is None:
        output_dir = Path(args.ag_root_directory) / "world_rel_annotations" / args.phase
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"RAG results: {args.rag_results_dir}/{args.mode}/{args.model_name}/")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Phase: {args.phase}, Mode: {args.mode}, Model: {args.model_name}")

    # Load GT annotations
    gt_loader = GTAnnotationLoader(args.ag_root_directory)

    # Get video IDs
    if args.video:
        video_ids = [args.video]
    else:
        video_ids = gt_loader.get_all_video_ids(phase=args.phase)
    logger.info(f"Found {len(video_ids)} videos for phase={args.phase}")

    if args.limit is not None:
        video_ids = video_ids[:args.limit]
        logger.info(f"Limited to {len(video_ids)} videos")

    # Process each video
    success_count = 0
    skip_count = 0
    error_count = 0

    for video_id in tqdm(video_ids, desc=f"Augmenting ({args.phase})"):
        try:
            ok = process_video(
                video_id=video_id,
                gt_loader=gt_loader,
                rag_results_dir=args.rag_results_dir,
                mode=args.mode,
                model_name=args.model_name,
                phase=args.phase,
                output_dir=output_dir,
                overwrite=args.overwrite,
            )
            if ok:
                success_count += 1
        except Exception as e:
            logger.error(f"[{video_id}] Error: {e}", exc_info=True)
            error_count += 1

    logger.info(
        f"Done. {success_count}/{len(video_ids)} videos processed, "
        f"{error_count} errors. Output: {output_dir}"
    )
    logger.info(f"Log: {_LOG_DIR / 'augment_relationships.log'}")


if __name__ == "__main__":
    main()
