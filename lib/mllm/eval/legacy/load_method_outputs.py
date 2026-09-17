#!/usr/bin/env python3
"""
load_method_outputs.py
======================
Load and normalise relationship-prediction PKLs produced by the RAG
(``process_ag_rag_all.py``) and Caption (``process_ag_caption_all.py``)
pipelines into a uniform format suitable for evaluation.

The prediction PKLs use a **scored-dict** format for relationships::

    attention:  {"label": "looking_at", "yes_prob": 0.92}
    contacting: [{"label": "holding", "yes_prob": 0.85}, ...]
    spatial:    [{"label": "in_front_of", "yes_prob": 0.91}, ...]

This loader extracts both labels and scores into a uniform format.

Usage
-----
::

    python -m lib.mllm.eval.legacy.load_method_outputs \\
        --method rag \\
        --mode predcls \\
        --model_name qwen3vl \\
        --predictions_dir /data/rohith/ag/mllms/rag_all_objects_results/

    # Or caption pipeline
    python -m lib.mllm.eval.legacy.load_method_outputs \\
        --method caption \\
        --mode predcls \\
        --model_name qwen3vl \\
        --predictions_dir /data/rohith/ag/mllms/caption_all_objects_results/
"""

import os
import glob
import pickle
import argparse
from typing import Dict, Any, Optional, List

from lib.mllm.eval.legacy.label_constants import (
    normalise_object_label,
    normalise_relationship_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Labels that should be treated as "no prediction"
_EMPTY_LABELS = {"", "unknown", "none", "n/a"}


def _normalise_label(label: str) -> str:
    """Normalise an object label to **normalized** short form.

    Maps compound AG names: ``"phone/camera"`` → ``"phone"``, etc.
    See :data:`label_constants.LABEL_NORMALIZE_MAP` for full mapping.
    """
    return normalise_object_label(label)


def _normalise_rel_label(label: str) -> str:
    """Normalise a relationship label to **normalized** underscore form.

    Maps space-separated LLM output: ``"looking at"`` → ``"looking_at"``, etc.
    See :data:`label_constants._LABEL_SPACE_TO_UNDERSCORE` for full mapping.
    """
    return normalise_relationship_label(label)


def _extract_scored_field(value, multi: bool = False):
    """Extract labels and scores from a scored-dict field.

    Parameters
    ----------
    value
        One of: ``str``, ``list[str]``, ``{label: score}``,
        ``{"label": str, "yes_prob": float}``, or
        ``[{"label": str, "yes_prob": float}, ...]``.
    multi : bool
        If True, expect multiple labels (contacting, spatial).
        If False, expect a single label (attention).

    Returns
    -------
    tuple
        ``(labels, scores)`` where *labels* is ``str`` or ``list[str]``
        and *scores* is ``dict`` mapping label→score (empty if no scores
        available).
    """
    scores: Dict[str, float] = {}

    if value is None:
        return ("" if not multi else [], scores)

    # ---- Single-label field (attention) ----------------------------------
    if not multi:
        if isinstance(value, str):
            label = _normalise_rel_label(value)
            if label in _EMPTY_LABELS:
                return ("", scores)
            return (label, scores)
        if isinstance(value, dict):
            # {"label": "looking_at", "yes_prob": 0.92}
            if "label" in value:
                label = _normalise_rel_label(value["label"])
                if label and label != "unknown":
                    scores[label] = value.get("yes_prob", 1.0)
                    return (label, scores)
                return ("", scores)
            # {label_string: score} format
            if len(value) == 1:
                k, v = next(iter(value.items()))
                label = _normalise_rel_label(k)
                if label in _EMPTY_LABELS:
                    return ("", scores)
                scores[label] = float(v) if isinstance(v, (int, float)) else 1.0
                return (label, scores)
            # Multiple entries — pick highest score
            if value:
                # Filter out empty/unknown keys first
                valid = {k: v for k, v in value.items() if _normalise_rel_label(k) not in _EMPTY_LABELS}
                if not valid:
                    return ("", scores)
                best_k = max(valid, key=lambda k: float(valid[k]) if isinstance(valid[k], (int, float)) else 0)
                label = _normalise_rel_label(best_k)
                for k, v in valid.items():
                    scores[_normalise_rel_label(k)] = float(v) if isinstance(v, (int, float)) else 1.0
                return (label, scores)
        if isinstance(value, list):
            # Shouldn't happen for attention, but handle gracefully
            if value:
                lbl, sc = _extract_scored_field(value[0], multi=False)
                return (lbl, sc)
        return ("", scores)

    # ---- Multi-label field (contacting, spatial) -------------------------
    if isinstance(value, str):
        label = _normalise_rel_label(value)
        return ([label] if label and label not in _EMPTY_LABELS else [], scores)

    if isinstance(value, dict):
        # {label: score, label: score, ...}
        if value and "label" in value:
            # Single {"label": ..., "yes_prob": ...}
            label = _normalise_rel_label(value["label"])
            if label and label != "unknown":
                scores[label] = value.get("yes_prob", 1.0)
                return ([label], scores)
            return ([], scores)
        labels = []
        for k, v in value.items():
            lbl = _normalise_rel_label(k)
            if lbl and lbl not in _EMPTY_LABELS:
                labels.append(lbl)
                scores[lbl] = float(v) if isinstance(v, (int, float)) else 1.0
        return (labels, scores)

    if isinstance(value, list):
        labels = []
        for item in value:
            if isinstance(item, str):
                lbl = _normalise_rel_label(item)
                if lbl and lbl not in _EMPTY_LABELS:
                    labels.append(lbl)
            elif isinstance(item, dict):
                lbl = _normalise_rel_label(item.get("label", ""))
                if lbl and lbl != "unknown":
                    labels.append(lbl)
                    scores[lbl] = item.get("yes_prob", 1.0)
        return (labels, scores)

    return ([], scores)


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------

class MethodOutputLoader:
    """Load relationship predictions from a method's output directory.

    Parameters
    ----------
    predictions_dir : str
        Root output directory (e.g. ``rag_all_objects_results/``).
    mode : str
        ``"predcls"`` or ``"sgdet"``.
    model_name : str
        ``"qwen3vl"``, ``"internvl"``, etc.
    """

    def __init__(
        self,
        predictions_dir: str,
        mode: str = "predcls",
        model_name: str = "qwen3vl",
    ):
        self.predictions_dir = predictions_dir
        self.mode = mode
        self.model_name = model_name
        self.pkl_dir = os.path.join(predictions_dir, mode, model_name)

    # ------------------------------------------------------------------
    # Single video
    # ------------------------------------------------------------------

    def load_video(self, video_id: str) -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
        """Load predictions for one video.

        All labels in the returned dict are in **normalized** form:

        - **Object names** (dict keys): short form via ``normalise_object_label``
          (e.g. ``"phone"`` not ``"phone/camera"``).
        - **Relationship labels** (attention/contacting/spatial values):
          underscore form via ``normalise_relationship_label``
          (e.g. ``"looking_at"`` not ``"looking at"``).
        - **Frame stems** (dict keys): extension-stripped
          (e.g. ``"000042"`` not ``"000042.png"``).

        Returns
        -------
        dict or None
            ``{frame_stem: {object_name: {attention, contacting, spatial,
            attention_scores, contacting_scores, spatial_scores}}}``

            All object names and relationship labels are **NORMALIZED**.
        """
        pkl_path = os.path.join(self.pkl_dir, f"{video_id}.pkl")
        if not os.path.exists(pkl_path):
            return None

        with open(pkl_path, "rb") as f:
            record = pickle.load(f)

        return self._normalise_record(record)

    def load_video_raw(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load the raw PKL record for one video (for estimation_meta, etc.)."""
        pkl_path = os.path.join(self.pkl_dir, f"{video_id}.pkl")
        if not os.path.exists(pkl_path):
            return None
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # All videos
    # ------------------------------------------------------------------

    def load_all(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
        """Load predictions for every video in the PKL directory.

        Returns
        -------
        dict
            ``{video_id: {frame_stem: {object_name: {attention, contacting,
            spatial, ..._scores}}}}``
        """
        if not os.path.isdir(self.pkl_dir):
            print(f"[MethodOutputLoader] Directory not found: {self.pkl_dir}")
            return {}

        results: Dict[str, Dict] = {}
        pkl_files = sorted(glob.glob(os.path.join(self.pkl_dir, "*.pkl")))
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                record = pickle.load(f)
            video_id = record.get("video_id", os.path.basename(pkl_path).replace(".pkl", ""))
            results[video_id] = self._normalise_record(record)
        return results

    def get_available_video_ids(self) -> List[str]:
        """Return sorted list of video IDs with available predictions."""
        if not os.path.isdir(self.pkl_dir):
            return []
        return sorted(
            os.path.basename(p).replace(".pkl", "")
            for p in glob.glob(os.path.join(self.pkl_dir, "*.pkl"))
        )

    # ------------------------------------------------------------------
    # Internal normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_record(
        record: Dict[str, Any],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Convert a raw PKL record into the uniform evaluation format.

        Handles both the **scored-dict** format (from verified pipeline)::

            attention:  {"label": "looking_at", "yes_prob": 0.92}
            contacting: [{"label": "holding", "yes_prob": 0.85}]

        and the **plain string** format::

            attention:  "looking_at"
            contacting: ["holding", "touching"]

        Output::

            {
                frame_stem: {
                    object_name: {
                        "attention":          str,
                        "contacting":         [str],
                        "spatial":            [str],
                        "attention_scores":   {label: score},
                        "contacting_scores":  {label: score},
                        "spatial_scores":     {label: score},
                    }
                }
            }
        """
        frames = record.get("frames", {})
        output: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for raw_frame_stem, frame_data in frames.items():
            # Strip extensions so keys match the evaluator's format
            frame_stem = raw_frame_stem.replace(".png", "").replace(".jpg", "")
            preds = frame_data.get("predictions", [])
            obj_map: Dict[str, Dict[str, Any]] = {}

            for pred in preds:
                obj_name = _normalise_label(pred.get("object", ""))
                if not obj_name:
                    continue

                att, att_scores = _extract_scored_field(
                    pred.get("attention"), multi=False,
                )
                cont, cont_scores = _extract_scored_field(
                    pred.get("contacting"), multi=True,
                )
                spat, spat_scores = _extract_scored_field(
                    pred.get("spatial"), multi=True,
                )

                obj_map[obj_name] = {
                    "attention": att,
                    "contacting": cont,
                    "spatial": spat,
                    "attention_scores": att_scores,
                    "contacting_scores": cont_scores,
                    "spatial_scores": spat_scores,
                }

            output[frame_stem] = obj_map

        return output


# ---------------------------------------------------------------------------
# CLI (for inspection / debugging)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect method relationship-prediction outputs",
    )
    parser.add_argument(
        "--predictions_dir", type=str, required=True,
        help="Root output directory for the method",
    )
    parser.add_argument(
        "--method", type=str, choices=["rag", "caption"], default="rag",
        help="Which method produced the predictions",
    )
    parser.add_argument(
        "--mode", type=str, default="predcls",
        choices=["predcls", "sgdet"],
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen3vl",
        help="Model name (qwen3vl | internvl)",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Inspect a single video",
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Max videos to print (when --video is not set)",
    )
    args = parser.parse_args()

    loader = MethodOutputLoader(
        predictions_dir=args.predictions_dir,
        mode=args.mode,
        model_name=args.model_name,
    )

    if args.video:
        data = loader.load_video(args.video)
        if data is None:
            print(f"No predictions found for {args.video}")
        else:
            for frame_stem, obj_map in sorted(data.items()):
                print(f"\n  Frame: {frame_stem}")
                for obj, rels in sorted(obj_map.items()):
                    scores_str = ""
                    if rels.get("attention_scores"):
                        scores_str += f" att_scores={rels['attention_scores']}"
                    print(f"    {obj}: att={rels['attention']}, "
                          f"cont={rels['contacting']}, spat={rels['spatial']}{scores_str}")
    else:
        video_ids = loader.get_available_video_ids()
        print(f"Found {len(video_ids)} videos in {loader.pkl_dir}")
        for vid in video_ids[:args.limit]:
            data = loader.load_video(vid)
            n_frames = len(data) if data else 0
            n_preds = sum(len(m) for m in data.values()) if data else 0
            print(f"  {vid}: {n_frames} frames, {n_preds} predictions")


if __name__ == "__main__":
    main()
