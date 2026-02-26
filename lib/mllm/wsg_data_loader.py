#!/usr/bin/env python3
"""
wsg_data_loader.py
==================
Unified data loaders for World Scene Graph evaluation.

Provides:
  - ``WorldSGGroundTruthLoader``: loads world scene graph PKLs and returns
    per-frame GT relationships filtered by scope (standard / world / missing).
  - ``PredictionLoader``: loads RAG prediction PKLs (from
    ``process_ag_rag_all.py`` or ``process_ag_rag.py``).
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# ---------------------------------------------------------------------------
# Label normalization (shared with world_scene_graph_generator.py)
# ---------------------------------------------------------------------------

_LABEL_NORMALIZE: Dict[str, str] = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}

_LABEL_DENORMALIZE: Dict[str, str] = {v: k for k, v in _LABEL_NORMALIZE.items()}

# Build a bidirectional alias map for fuzzy matching:
# "closet" <-> "closet/cabinet", "cup" <-> "cup/glass/bottle", etc.
_LABEL_ALIASES: Dict[str, str] = {}
_LABEL_ALIASES.update(_LABEL_NORMALIZE)
_LABEL_ALIASES.update(_LABEL_DENORMALIZE)


def normalize_label(label: str) -> str:
    """Normalize an AG object label to its short form."""
    return _LABEL_NORMALIZE.get(label, label)


# Valid evaluation scopes
VALID_SCOPES = ("standard", "world", "missing")


# ---------------------------------------------------------------------------
# Relationship label sets (Action Genome vocabulary)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# World Scene Graph Ground Truth Loader
# ---------------------------------------------------------------------------

class WorldSGGroundTruthLoader:
    """Load world scene graph PKLs and provide per-frame GT relationships.

    Each PKL (produced by ``world_scene_graph_generator.py``) contains per-frame
    objects with ``visible``, ``source``, ``rel_source``, and ``bbox_source``
    fields.  The ``scope`` parameter controls which objects are included:

    Visibility rule
    ~~~~~~~~~~~~~~~
    An object is considered **visible in the frame** if either:
      - ``visible == True`` (observed in GT annotations), OR
      - ``bbox_source == "gdino"`` (detected by Grounding-DINO)

    An object is **missing** only when it is absent from both GT annotations
    and GDino detections.

    Scope definitions
    ~~~~~~~~~~~~~~~~~
    - ``"standard"``: visible objects (GT-observed OR GDino-detected)
    - ``"world"``:    all objects (visible + missing)
    - ``"missing"``:  only truly missing objects (not in GT AND not in GDino)
    """

    def __init__(self, world_sg_dir: str):
        self.world_sg_dir = Path(world_sg_dir)
        if not self.world_sg_dir.exists():
            raise FileNotFoundError(
                f"World SG directory not found: {self.world_sg_dir}"
            )

        # Index: video_id → loaded dict (lazy-loaded on first access)
        self._video_index: Dict[str, Path] = {}
        self._video_cache: Dict[str, Dict[str, Any]] = {}
        self._build_index()

    def _build_index(self):
        """Discover all PKL files and map video_id → path."""
        pkl_files = sorted(self.world_sg_dir.glob("*.pkl"))
        for pkl_path in pkl_files:
            video_id = pkl_path.stem
            self._video_index[video_id] = pkl_path
        logger.info(
            f"[WorldSGGT] Indexed {len(self._video_index)} video PKLs "
            f"from {self.world_sg_dir}"
        )

    def _load_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load and cache a video's world SG data."""
        if video_id in self._video_cache:
            return self._video_cache[video_id]

        pkl_path = self._video_index.get(video_id)
        if pkl_path is None:
            return None

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._video_cache[video_id] = data
        return data

    def get_video_ids(self) -> List[str]:
        """Return all available video IDs."""
        return sorted(self._video_index.keys())

    def get_frame_keys(self, video_id: str) -> List[str]:
        """Return sorted frame keys for a video."""
        data = self._load_video(video_id)
        if data is None:
            return []
        frames = data.get("frames", {})
        return sorted(
            frames.keys(),
            key=lambda k: int(Path(k).stem) if Path(k).stem.isdigit() else k,
        )

    def get_gt_for_frame(
        self,
        video_id: str,
        frame_key: str,
        scope: str = "world",
    ) -> Dict[str, Dict[str, Any]]:
        """Get GT relationship annotations for a frame, filtered by scope.

        Parameters
        ----------
        video_id : str
            Video identifier (stem, without .mp4).
        frame_key : str
            Frame key (e.g. ``"video_id/000042.png"``).
        scope : str
            One of ``"standard"``, ``"world"``, ``"missing"``.

        Returns
        -------
        dict
            ``{object_label: {"attention": [str], "contacting": [str], "spatial": [str]}}``
        """
        if scope not in VALID_SCOPES:
            raise ValueError(
                f"Invalid scope '{scope}'. Must be one of {VALID_SCOPES}"
            )

        data = self._load_video(video_id)
        if data is None:
            return {}

        frames = data.get("frames", {})
        frame_data = frames.get(frame_key)
        if frame_data is None:
            return {}

        gt: Dict[str, Dict[str, Any]] = {}

        for obj in frame_data.get("objects", []):
            # Use precomputed is_visible flag from the PKL (set by
            # world_scene_graph_generator.py).  Falls back to computing
            # it if the field is missing (backward compat with older PKLs).
            if "is_visible" in obj:
                is_visible = obj["is_visible"]
            else:
                is_visible = obj.get("visible", True) or obj.get("bbox_source", "none") == "gdino"

            # Scope filtering
            if scope == "standard" and not is_visible:
                continue
            if scope == "missing" and is_visible:
                continue
            # scope == "world" → include all

            label = obj.get("label", "")
            if not label or label in ("person", "__background__"):
                continue

            att_rel = obj.get("attention_relationship", [])
            cont_rel = obj.get("contacting_relationship", [])
            spa_rel = obj.get("spatial_relationship", [])

            # Ensure non-empty defaults
            if not att_rel:
                att_rel = ["unsure"]
            if not cont_rel:
                cont_rel = ["not_contacting"]
            if not spa_rel:
                spa_rel = ["in_front_of"]

            gt[label] = {
                "attention": list(att_rel),
                "contacting": list(cont_rel),
                "spatial": list(spa_rel),
            }

        return gt

    def get_gt_counts_for_video(
        self, video_id: str, scope: str = "world",
    ) -> Dict[str, int]:
        """Return counts of GT objects per frame for a video, useful for
        debugging / sanity checks.

        Returns ``{frame_key: num_objects}``
        """
        frame_keys = self.get_frame_keys(video_id)
        counts: Dict[str, int] = {}
        for fk in frame_keys:
            gt = self.get_gt_for_frame(video_id, fk, scope=scope)
            counts[fk] = len(gt)
        return counts


# ---------------------------------------------------------------------------
# Prediction Loader
# ---------------------------------------------------------------------------

class PredictionLoader:
    """Load RAG prediction PKLs for evaluation.

    Handles both output formats:
    - ``process_ag_rag.py``:     predictions use ``"missing_object"`` key
    - ``process_ag_rag_all.py``: predictions use ``"object"`` key

    Each prediction entry has:

    .. code-block:: python

        {
            "object": "cup",
            "raw_response": "...",
            "attention": {"label": "looking_at", "yes_prob": 0.92},
            "contacting": [{"label": "holding", "yes_prob": 0.85}, ...],
            "spatial": [{"label": "in_front_of", "yes_prob": 0.78}, ...],
        }
    """

    def __init__(self, predictions_dir: str):
        self.predictions_dir = Path(predictions_dir)
        if not self.predictions_dir.exists():
            raise FileNotFoundError(
                f"Predictions directory not found: {self.predictions_dir}"
            )

        # Index: video_id → path
        self._video_index: Dict[str, Path] = {}
        self._video_cache: Dict[str, Dict[str, Any]] = {}
        self._build_index()

    def _build_index(self):
        """Discover all prediction PKL files."""
        pkl_files = sorted(self.predictions_dir.glob("*.pkl"))
        for pkl_path in pkl_files:
            video_id = pkl_path.stem
            self._video_index[video_id] = pkl_path
        logger.info(
            f"[PredLoader] Indexed {len(self._video_index)} prediction PKLs "
            f"from {self.predictions_dir}"
        )

    def get_video_ids(self) -> List[str]:
        """Return all video IDs with predictions."""
        return sorted(self._video_index.keys())

    def load_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load and cache the full prediction dict for a video."""
        if video_id in self._video_cache:
            return self._video_cache[video_id]

        pkl_path = self._video_index.get(video_id)
        if pkl_path is None:
            return None

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._video_cache[video_id] = data
        return data

    def get_predictions_for_frame(
        self,
        video_id: str,
        frame_stem: str,
    ) -> List[Dict[str, Any]]:
        """Get normalized predictions for a single frame.

        Parameters
        ----------
        video_id : str
            Video identifier.
        frame_stem : str
            Frame stem (e.g. ``"000042"``), used as key in the prediction
            PKL's ``frames`` dict.

        Returns
        -------
        list[dict]
            Each dict has:
            ``{"object": str, "attention": str, "contacting": [str], "spatial": [str]}``
            where labels are extracted from the scored dicts.
        """
        data = self.load_video(video_id)
        if data is None:
            return []

        frames = data.get("frames", {})
        frame_data = frames.get(frame_stem)
        if frame_data is None:
            return []

        raw_preds = frame_data.get("predictions", [])
        normalized: List[Dict[str, Any]] = []

        for pred in raw_preds:
            # Object name: try both keys
            obj_name = pred.get("object", pred.get("missing_object", ""))
            if not obj_name:
                continue

            # --- Extract attention (single label) ---
            att_raw = pred.get("attention", "unknown")
            if isinstance(att_raw, dict):
                att_label = att_raw.get("label", "unknown")
            elif isinstance(att_raw, str):
                att_label = att_raw
            else:
                att_label = "unknown"

            # --- Extract contacting (multi-label) ---
            cont_raw = pred.get("contacting", ["unknown"])
            if isinstance(cont_raw, list):
                cont_labels = []
                for item in cont_raw:
                    if isinstance(item, dict):
                        cont_labels.append(item.get("label", "unknown"))
                    elif isinstance(item, str):
                        cont_labels.append(item)
                if not cont_labels:
                    cont_labels = ["unknown"]
            elif isinstance(cont_raw, str):
                cont_labels = [cont_raw]
            elif isinstance(cont_raw, dict):
                cont_labels = [cont_raw.get("label", "unknown")]
            else:
                cont_labels = ["unknown"]

            # --- Extract spatial (multi-label) ---
            spa_raw = pred.get("spatial", ["unknown"])
            if isinstance(spa_raw, list):
                spa_labels = []
                for item in spa_raw:
                    if isinstance(item, dict):
                        spa_labels.append(item.get("label", "unknown"))
                    elif isinstance(item, str):
                        spa_labels.append(item)
                if not spa_labels:
                    spa_labels = ["unknown"]
            elif isinstance(spa_raw, str):
                spa_labels = [spa_raw]
            elif isinstance(spa_raw, dict):
                spa_labels = [spa_raw.get("label", "unknown")]
            else:
                spa_labels = ["unknown"]

            normalized.append({
                "object": obj_name,
                "attention": att_label,
                "contacting": cont_labels,
                "spatial": spa_labels,
            })

        return normalized

    def get_frame_stems(self, video_id: str) -> List[str]:
        """Return sorted frame stems available in the prediction PKL."""
        data = self.load_video(video_id)
        if data is None:
            return []
        frames = data.get("frames", {})
        return sorted(frames.keys())
