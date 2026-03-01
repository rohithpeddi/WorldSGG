#!/usr/bin/env python3
"""
augment_relationships.py
========================
Merge **observed** relationship annotations (from ``object_bbox_and_relationship.pkl``)
with **RAG-predicted** missing-object relationships (per-video pkl files from
``process_ag_rag.py``) into combined per-video pkl files.

Output directory: ``<ag_root>/world_rel_annotations/<phase>/``

OUTPUT PKL FORMAT (compatible with ``base_ag_dataset.py``)
----------------------------------------------------------
The per-video pkl stores **two dicts** that mirror the global
``person_bbox.pkl`` and ``object_bbox_and_relationship.pkl`` structure:

{
    "video_id":      str,
    "video_objects":  sorted list of unique object label strings,
    "num_frames":    int,
    "rag_model":     str,
    "rag_mode":      str,

    # Same schema as person_bbox.pkl  (keyed by "video_id/frame.png")
    "person_bbox": {
        "video_id/000042.png": {
            "bbox":      np.ndarray,       # person bounding box
            "bbox_size": (w, h),           # frame dimensions (from GT)
        },
        ...
    },

    # Same schema as object_bbox_and_relationship.pkl
    # Each value is a LIST of object dicts (GT observed + RAG missing)
    "object_bbox": {
        "video_id/000042.png": [
            {   # ----- GT observed object -----
                "class":                    str,   # e.g. "bed" (full AG name)
                "bbox":                     np.ndarray([x1,y1,x2,y2]) | None,  # xyxy
                "visible":                  True,
                "attention_relationship":   list[str],   # underscore format
                "contacting_relationship":  list[str],
                "spatial_relationship":     list[str],
                "metadata":                 {"set": "train"},
                "source":                   "gt",
            },
            {   # ----- RAG missing object -----
                "class":                    str,   # e.g. "laptop"
                "bbox":                     None,
                "visible":                  False,
                "attention_relationship":   list[str],   # [] if unknown
                "contacting_relationship":  list[str],
                "spatial_relationship":     list[str],
                "metadata":                 {"set": "train"},
                "source":                   "rag",
                "attention_scores":         dict[str, float],
                "contacting_scores":        dict[str, float],
                "spatial_scores":           dict[str, float],
            },
            ...
        ],
        ...
    },
}

Usage:
    python datasets/preprocess/annotations/augment_relationships.py \\
        --ag_root_directory /data/rohith/ag \\
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

# Short-form ↔ full AG name mappings
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet", "cup/glass/bottle": "cup",
    "paper/notebook": "paper", "sofa/couch": "sofa",
    "phone/camera": "phone",
}
LABEL_DENORMALIZE_MAP = {v: k for k, v in LABEL_NORMALIZE_MAP.items()}

NAME_TO_IDX = {name: idx for idx, name in enumerate(OBJECT_CLASSES) if idx > 0}
# Also register short forms for lookup
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

# Space-separated → underscore (from LLM output)
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


def _resolve_object_label(raw_name: str) -> str:
    """Resolve a raw object name to the canonical full AG class name (string).

    For example: "closet" → "closet/cabinet", "bed" → "bed".
    Raises ValueError if the name is not in the AG vocabulary.
    """
    name = raw_name.strip().lower()
    # Direct match in full names
    if name in OBJECT_CLASSES and name not in ("__background__", "person"):
        return name
    # Short form → full name
    full = LABEL_DENORMALIZE_MAP.get(name)
    if full and full in OBJECT_CLASSES:
        return full
    # Check NAME_TO_IDX (includes short forms)
    idx = NAME_TO_IDX.get(name, -1)
    if idx > 0:
        return OBJECT_CLASSES[idx]
    raise ValueError(
        f"Unknown object label: '{raw_name}'. "
        f"Not found in AG vocabulary."
    )


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
        """Return sorted list of unique video IDs for the given phase."""
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

    def get_frame_data(
        self, frame_key: str, phase: str = "train",
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extract person bbox entry and object annotations for a frame,
        formatted to match the raw pkl schema expected by ``base_ag_dataset.py``.

        Returns
        -------
        person_entry : dict
            ``{"bbox": np.ndarray, "bbox_size": (w, h)}``
        observed_objects : list[dict]
            Each dict has: class (str), bbox (np.ndarray xywh), visible (True),
            attention_relationship, contacting_relationship,
            spatial_relationship (lists of underscore-formatted strings),
            metadata ({"set": phase}), source ("gt").
        """
        # ---- Person bbox ----
        raw_person = self.person_bbox.get(frame_key, {})
        person_entry = {
            "bbox": raw_person.get("bbox", np.zeros((0, 4))),
            "bbox_size": raw_person.get("bbox_size", (0, 0)),
        }

        # ---- Object annotations ----
        obj_entries = self.object_bbox.get(frame_key, [])
        observed: List[Dict[str, Any]] = []

        for obj_ann in obj_entries:
            if not obj_ann.get("visible", False):
                continue

            cls_raw = obj_ann.get("class")
            if cls_raw is None:
                continue

            # Resolve to string label (the dataloader expects string)
            if isinstance(cls_raw, int):
                if cls_raw <= 0 or cls_raw >= len(OBJECT_CLASSES):
                    continue
                label = OBJECT_CLASSES[cls_raw]
            elif isinstance(cls_raw, str):
                label = cls_raw
                if NAME_TO_IDX.get(label, -1) <= 0:
                    continue
            else:
                continue

            if label in ("person", "__background__"):
                continue

            # Relationship labels (normalize to underscore format)
            att_rel = obj_ann.get("attention_relationship", [])
            cont_rel = obj_ann.get("contacting_relationship", [])
            spa_rel = obj_ann.get("spatial_relationship", [])

            att_rel = [_normalize_rel_label(r) for r in att_rel] if att_rel else ["unsure"]
            cont_rel = [_normalize_rel_label(r) for r in cont_rel] if cont_rel else ["not_contacting"]
            spa_rel = [_normalize_rel_label(r) for r in spa_rel] if spa_rel else ["in_front_of"]

            # Bbox — convert from raw xywh to xyxy
            raw_bbox = obj_ann.get("bbox", None)
            if raw_bbox is not None:
                if isinstance(raw_bbox, np.ndarray):
                    raw_bbox = raw_bbox.tolist()
                if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                    x, y, w, h = float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])
                    bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
                else:
                    bbox = None
            else:
                bbox = None

            observed.append({
                "class": label,                          # STRING (dataloader converts)
                "bbox": bbox,                            # np.ndarray xyxy
                "visible": True,
                "attention_relationship": att_rel,
                "contacting_relationship": cont_rel,
                "spatial_relationship": spa_rel,
                "metadata": {"set": phase},
                "source": "gt",
            })

        return person_entry, observed


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
) -> Tuple[Optional[str], float]:
    """Extract (label, score) from a RAG prediction entry.

    RAG output format can be:
      - dict: {"label": "...", "yes_prob": 0.8}   (from verification)
      - dict: {"label": "...", "score": 0.8}       (from skip-verification)
      - str:  "label_name"

    Returns (None, 0.0) if the label is 'unknown' or not in the valid set.
    """
    if isinstance(entry, dict):
        label = _normalize_rel_label(entry.get("label", default))
        # RAG verification uses 'yes_prob'; skip-verification uses 'score'
        score = float(entry.get("yes_prob", entry.get("score", 0.5)))
    elif isinstance(entry, str):
        label = _normalize_rel_label(entry)
        score = 0.5  # no score available
    else:
        raise ValueError(
            f"Unexpected relationship entry type {type(entry)}: {entry}"
        )

    # 'unknown' or invalid → return None so caller can drop it
    if label == "unknown" or label not in valid_set:
        return None, 0.0
    return label, score


def extract_missing_for_frame(
    rag_data: Dict[str, Any], frame_stem: str, phase: str = "train",
) -> List[Dict[str, Any]]:
    """
    Extract RAG-predicted missing-object relationships for a frame.

    ``frame_stem`` is the numeric part (e.g. ``"000042"``).

    Returns a list of dicts in the same schema as GT objects but with
    ``bbox=None``, ``visible=False``, ``source="rag"``, ``class=str``.
    """
    frames = rag_data.get("frames", {})
    frame_data = frames.get(frame_stem)
    if frame_data is None:
        return []

    missing: List[Dict[str, Any]] = []
    for pred in frame_data.get("predictions", []):
        raw_obj_name = pred.get("missing_object", "").strip().lower()
        if not raw_obj_name:
            raise ValueError(
                f"Empty missing_object in RAG prediction for frame '{frame_stem}': {pred}"
            )

        # Resolve to canonical full AG class NAME (string)
        label = _resolve_object_label(raw_obj_name)

        # --- Attention (single label) ---
        att_raw = pred.get("attention", "unknown")
        att_label, att_score = None, 0.0
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
            if att_label == "unknown" or att_label not in ATTENTION_RELATIONSHIPS:
                att_label = None

        att_list = [att_label] if att_label is not None else []
        att_scores = {att_label: att_score} if att_label is not None else {}

        # --- Contacting (multi-label) ---
        cont_raw = pred.get("contacting", ["unknown"])
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
            if lbl is not None and lbl not in cont_scores:
                cont_list.append(lbl)
                cont_scores[lbl] = sc

        # --- Spatial (multi-label) ---
        spa_raw = pred.get("spatial", ["unknown"])
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
            if lbl is not None and lbl not in spa_scores:
                spa_list.append(lbl)
                spa_scores[lbl] = sc

        missing.append({
            "class": label,                          # STRING (matches GT format)
            "bbox": None,                            # no bbox for missing objects
            "visible": False,
            "attention_relationship": att_list,       # [] if unknown
            "contacting_relationship": cont_list,
            "spatial_relationship": spa_list,
            "metadata": {"set": phase},
            "source": "rag",
            "attention_scores": att_scores,
            "contacting_scores": cont_scores,
            "spatial_scores": spa_scores,
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
    """Process a single video: merge GT + RAG and save combined pkl.

    Output format mirrors person_bbox.pkl + object_bbox_and_relationship.pkl
    so the existing ``base_ag_dataset.py`` dataloader can consume it.
    """
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
        logger.warning(f"[{video_id}] No RAG file found — saving with missing=[] for all frames")

    # 3. Process each frame → build person_bbox and object_bbox dicts
    all_object_labels: Set[str] = set()
    person_bbox_dict: Dict[str, Any] = {}
    object_bbox_dict: Dict[str, List[Dict[str, Any]]] = {}
    n_observed = 0
    n_missing = 0

    for frame_key in frame_keys:
        person_entry, observed = gt_loader.get_frame_data(frame_key, phase=phase)
        person_bbox_dict[frame_key] = person_entry

        for obj in observed:
            all_object_labels.add(obj["class"])

        # Extract missing-object predictions
        stem = frame_key_to_stem(frame_key)
        missing: List[Dict[str, Any]] = []
        if rag_data is not None:
            missing = extract_missing_for_frame(rag_data, stem, phase=phase)
            for obj in missing:
                all_object_labels.add(obj["class"])

        # Combine into flat list (GT observed first, then RAG missing)
        object_bbox_dict[frame_key] = observed + missing
        n_observed += len(observed)
        n_missing += len(missing)

    # 4. Save combined result
    output_record = {
        "video_id": video_id,
        "video_objects": sorted(all_object_labels),
        "num_frames": len(frame_keys),
        "rag_model": model_name,
        "rag_mode": mode,
        "person_bbox": person_bbox_dict,
        "object_bbox": object_bbox_dict,
    }
    with open(save_path, "wb") as f:
        pickle.dump(output_record, f)

    logger.info(
        f"[{video_id}] {len(frame_keys)} frames "
        f"({n_observed} observed, {n_missing} missing) → {save_path.name}"
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
