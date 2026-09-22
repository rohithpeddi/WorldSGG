#!/usr/bin/env python3
"""
evaluate_relationships.py
=========================
Match predicted relationships against augmented ground truth and compute
Precision, Recall, and F1 score.

Supports two evaluation modes:

- **predcls** — Object labels are provided upfront. Matching is by exact
  normalised object class name. Relationship metrics only.
- **sgdet** — Objects are estimated by the MLLM. Produces:
  (a) Object detection P/R/F1
  (b) Relationship metrics over all GT objects (unmatched → all FN)
  (c) Relationship metrics over matched objects only

Three comparison slices are always computed for relationship metrics:

1. **GT + Corrections vs Predictions** — full augmented ground truth
2. **Corrections-only vs Predictions** — only missing/occluded objects
3. **Matched-objects-only** (sgdet only) — relationship metrics restricted
   to GT objects that were also predicted

Usage
-----
::

    python -m lib.mllm.eval.legacy.evaluate_relationships \\
        --augmented_dir ./augmented \\
        --predictions_dir /data/rohith/ag/mllms/rag_all_objects_results/ \\
        --mode predcls --model_name qwen3vl \\
        --output_file ./eval_results.json

    python -m lib.mllm.eval.legacy.evaluate_relationships \\
        --augmented_dir ./augmented \\
        --predictions_dir /data/rohith/ag/mllms/rag_all_objects_results/ \\
        --mode sgdet --model_name internvl \\
        --output_file ./eval_results_sgdet.json
"""

import os
import sys
import json
import glob
import pickle
import logging
import argparse
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict

from tqdm import tqdm


from lib.mllm.eval.legacy.load_method_outputs import MethodOutputLoader
from lib.mllm.eval.legacy.label_constants import (
    normalise_object_label,
    normalise_relationship_label,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("evaluate_relationships")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    log_path = os.environ.get(
        "MLLM_LEGACY_EVAL_LOG",
        os.path.join(tempfile.gettempdir(), "evaluate_relationships.log"),
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[eval] %(message)s"))
    logger.addHandler(ch)

    return logger


log = _setup_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_label(label: str) -> str:
    """Normalise an object label to **normalized** short form.

    Maps compound AG names: ``"phone/camera"`` → ``"phone"``, etc.
    """
    return normalise_object_label(label)


def _normalise_rel_label(label: str) -> str:
    """Normalise a relationship label to **normalized** underscore form.

    Maps space-separated LLM output: ``"looking at"`` → ``"looking_at"``, etc.
    """
    return normalise_relationship_label(label)


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

class RelationshipAccumulator:
    """Accumulates TP / FP / FN counts across relationship types.

    Tracks both aggregate counts and **per-label** counts so that
    macro-averaged metrics (mean of per-class P/R/F1) can be computed.

    All labels are expected in **normalized** form (see ``_normalise_rel_label``).
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.counts: Dict[str, Dict[str, int]] = {
            "attention": {"tp": 0, "fp": 0, "fn": 0},
            "contacting": {"tp": 0, "fp": 0, "fn": 0},
            "spatial": {"tp": 0, "fp": 0, "fn": 0},
        }
        # Per-label counts: {rel_type: {label: {"tp", "fp", "fn"}}}
        self.per_label: Dict[str, Dict[str, Dict[str, int]]] = {
            "attention": {},
            "contacting": {},
            "spatial": {},
        }
        self.n_pairs = 0

    def _ensure_label(self, rel_type: str, label: str):
        if label not in self.per_label[rel_type]:
            self.per_label[rel_type][label] = {"tp": 0, "fp": 0, "fn": 0}

    def add(self, rel_type: str, pred_labels: Set[str], gt_labels: Set[str]):
        tp_set = pred_labels & gt_labels
        fp_set = pred_labels - gt_labels
        fn_set = gt_labels - pred_labels

        self.counts[rel_type]["tp"] += len(tp_set)
        self.counts[rel_type]["fp"] += len(fp_set)
        self.counts[rel_type]["fn"] += len(fn_set)

        for lbl in tp_set:
            self._ensure_label(rel_type, lbl)
            self.per_label[rel_type][lbl]["tp"] += 1
        for lbl in fp_set:
            self._ensure_label(rel_type, lbl)
            self.per_label[rel_type][lbl]["fp"] += 1
        for lbl in fn_set:
            self._ensure_label(rel_type, lbl)
            self.per_label[rel_type][lbl]["fn"] += 1

    def add_pair(self, pred: Dict[str, Any], gt_obj: Dict[str, Any]):
        """Compare a prediction dict against a GT object dict."""
        self.n_pairs += 1

        # Attention
        pred_att = pred.get("attention", "")
        if isinstance(pred_att, list):
            pred_att_set = {_normalise_rel_label(a) for a in pred_att if a}
        else:
            pred_att_set = {_normalise_rel_label(pred_att)} if pred_att else set()
        gt_att_set = {_normalise_rel_label(a) for a in gt_obj.get("attention", []) if a}
        self.add("attention", pred_att_set, gt_att_set)

        # Contacting
        pred_cont = pred.get("contacting", [])
        if isinstance(pred_cont, str):
            pred_cont = [pred_cont]
        pred_cont_set = {_normalise_rel_label(c) for c in pred_cont if c}
        gt_cont_set = {_normalise_rel_label(c) for c in gt_obj.get("contacting", []) if c}
        self.add("contacting", pred_cont_set, gt_cont_set)

        # Spatial
        pred_spat = pred.get("spatial", [])
        if isinstance(pred_spat, str):
            pred_spat = [pred_spat]
        pred_spat_set = {_normalise_rel_label(s) for s in pred_spat if s}
        gt_spat_set = {_normalise_rel_label(s) for s in gt_obj.get("spatial", []) if s}
        self.add("spatial", pred_spat_set, gt_spat_set)

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    def compute(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        total_tp, total_fp, total_fn = 0, 0, 0

        for rel_type in ("attention", "contacting", "spatial"):
            c = self.counts[rel_type]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn

            prec, rec, f1 = self._prf(tp, fp, fn)
            results[rel_type] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
            }

            # Per-class metrics + macro average for this rel_type
            per_class: Dict[str, Dict[str, Any]] = {}
            class_p, class_r, class_f = [], [], []
            for lbl in sorted(self.per_label[rel_type].keys()):
                lc = self.per_label[rel_type][lbl]
                lp, lr, lf = self._prf(lc["tp"], lc["fp"], lc["fn"])
                per_class[lbl] = {
                    "precision": round(lp, 4), "recall": round(lr, 4),
                    "f1": round(lf, 4),
                    "tp": lc["tp"], "fp": lc["fp"], "fn": lc["fn"],
                }
                class_p.append(lp)
                class_r.append(lr)
                class_f.append(lf)

            n_cls = len(per_class)
            results[rel_type]["per_class"] = per_class
            results[rel_type]["macro_avg"] = {
                "precision": round(sum(class_p) / n_cls, 4) if n_cls else 0.0,
                "recall": round(sum(class_r) / n_cls, 4) if n_cls else 0.0,
                "f1": round(sum(class_f) / n_cls, 4) if n_cls else 0.0,
                "n_classes": n_cls,
            }

        # Micro average (across all rel types)
        prec, rec, f1 = self._prf(total_tp, total_fp, total_fn)
        results["micro_avg"] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        }

        # Global macro average across all labels from all rel types
        all_f1s = []
        for rt in ("attention", "contacting", "spatial"):
            for li in results[rt]["per_class"].values():
                all_f1s.append(li["f1"])
        results["macro_avg"] = {
            "f1": round(sum(all_f1s) / len(all_f1s), 4) if all_f1s else 0.0,
            "n_classes": len(all_f1s),
        }

        results["n_pairs"] = self.n_pairs
        return results


class ObjectDetectionAccumulator:
    """Accumulates object-level TP / FP / FN for sgdet evaluation.

    All object names are expected in **normalized** short form
    (e.g. ``"phone"`` not ``"phone/camera"``).
    """

    def __init__(self):
        self.tp = 0   # Predicted and in GT
        self.fp = 0   # Predicted but not in GT
        self.fn = 0   # In GT but not predicted
        self.n_frames = 0

    def add_frame(self, pred_objects: Set[str], gt_objects: Set[str]):
        """Compare predicted vs GT object sets for one frame."""
        self.n_frames += 1
        matched = pred_objects & gt_objects
        self.tp += len(matched)
        self.fp += len(pred_objects - gt_objects)
        self.fn += len(gt_objects - pred_objects)

    def compute(self) -> Dict[str, Any]:
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_frames": self.n_frames,
        }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

_EMPTY_PRED = {"attention": "", "contacting": [], "spatial": []}


def evaluate(
    augmented_dir: str,
    predictions_dir: str,
    mode: str,
    model_name: str,
) -> Dict[str, Any]:
    """Run evaluation across all videos.

    Both augmented GT and predictions are expected to use **normalized** labels:

    - **Object classes**: short form (``"phone"``, ``"cup"``, ``"sofa"``, etc.).
    - **Relationship labels**: underscore form (``"looking_at"``, etc.).
    - **Frame stems**: extension-stripped (``"000042"``).

    Returns a dict with metric groups depending on mode.
    """
    loader = MethodOutputLoader(
        predictions_dir=predictions_dir,
        mode=mode,
        model_name=model_name,
    )

    log.info("=" * 70)
    log.info("EVALUATE_START | timestamp=%s", datetime.now().isoformat())
    log.info(
        "CONFIG | augmented_dir=%s | predictions_dir=%s | mode=%s | model_name=%s",
        augmented_dir, predictions_dir, mode, model_name,
    )
    log.info("PKL_DIR | %s", loader.pkl_dir)

    # Accumulators
    acc_all = RelationshipAccumulator("gt_plus_corrections")
    acc_corr = RelationshipAccumulator("corrections_only")
    acc_matched = RelationshipAccumulator("matched_objects_only")  # sgdet
    obj_det = ObjectDetectionAccumulator()  # sgdet

    is_sgdet = (mode == "sgdet")

    # Load augmented PKLs
    aug_files = sorted(glob.glob(os.path.join(augmented_dir, "*.pkl")))
    if not aug_files:
        log.warning("NO_AUGMENTED_FILES | dir=%s", augmented_dir)
        return {}

    log.info("AUGMENTED_FILES | count=%d", len(aug_files))

    n_videos_evaluated = 0
    n_videos_skipped = 0
    total_pairs = 0

    pbar = tqdm(aug_files, desc=f"Evaluating [{mode}/{model_name}]", unit="video")
    for aug_path in pbar:
        with open(aug_path, "rb") as f:
            aug_record = pickle.load(f)
        video_id = aug_record.get("video_id", os.path.basename(aug_path).replace(".pkl", ""))

        # Load method predictions for this video
        pred_video = loader.load_video(video_id)
        if pred_video is None:
            n_videos_skipped += 1
            log.debug("SKIP_VIDEO | video=%s | reason=no_predictions", video_id)
            pbar.set_postfix(eval=n_videos_evaluated, skip=n_videos_skipped, pairs=total_pairs)
            continue

        n_videos_evaluated += 1
        video_gt_count = 0
        video_corr_count = 0
        video_matched_count = 0

        # Diagnostic: on first video, log sample keys so we can spot mismatches
        if n_videos_evaluated == 1:
            gt_frame_keys = list(aug_record.get("frames", {}).keys())[:3]
            gt_stems = [k.split("/")[-1].replace(".png", "").replace(".jpg", "") for k in gt_frame_keys]
            pred_stems = list(pred_video.keys())[:3]

            # Get sample object names
            sample_gt_objs = set()
            for fk in gt_frame_keys[:1]:
                for o in aug_record["frames"][fk].get("objects", []):
                    sample_gt_objs.add(o.get("class", ""))
            sample_pred_objs = set()
            for fs in pred_stems[:1]:
                sample_pred_objs.update(pred_video.get(fs, {}).keys())

            log.info(
                "FIRST_VIDEO_DIAGNOSTIC | video=%s | "
                "sample_gt_frame_keys=%s → stems=%s | sample_pred_stems=%s | "
                "sample_gt_objects=%s | sample_pred_objects=%s",
                video_id, gt_frame_keys, gt_stems, pred_stems,
                sorted(sample_gt_objs), sorted(sample_pred_objs),
            )

        log.debug("VIDEO_START | video=%s | gt_frames=%d", video_id, len(aug_record.get("frames", {})))

        for frame_key, frame_data in aug_record.get("frames", {}).items():
            frame_stem = frame_key.split("/")[-1].replace(".png", "").replace(".jpg", "")
            pred_frame = pred_video.get(frame_stem, {})
            gt_objects = frame_data.get("objects", [])

            # Build sets for object detection (sgdet)
            gt_obj_names = {
                _normalise_label(o.get("class", ""))
                for o in gt_objects if o.get("class")
            }
            pred_obj_names = set(pred_frame.keys())

            if is_sgdet:
                obj_det.add_frame(pred_obj_names, gt_obj_names)
                matched_names = pred_obj_names & gt_obj_names
                log.debug(
                    "  FRAME_OBJECTS | frame=%s | "
                    "Current: gt=%s, pred=%s | "
                    "Final: matched=%s, missed=%s, extra=%s",
                    frame_stem,
                    sorted(gt_obj_names), sorted(pred_obj_names),
                    sorted(matched_names),
                    sorted(gt_obj_names - pred_obj_names),
                    sorted(pred_obj_names - gt_obj_names),
                )

            for gt_obj in gt_objects:
                obj_class = _normalise_label(gt_obj.get("class", ""))
                if not obj_class:
                    continue

                pred_obj = pred_frame.get(obj_class)
                is_matched = pred_obj is not None

                if pred_obj is None:
                    pred_obj = _EMPTY_PRED

                log.debug(
                    "    PAIR | frame=%s | object=%s | source=%s | matched=%s | "
                    "Current: gt_att=%s, gt_cont=%s, gt_spat=%s | "
                    "pred_att=%s, pred_cont=%s, pred_spat=%s",
                    frame_stem, obj_class, gt_obj.get("source", "?"), is_matched,
                    gt_obj.get("attention", []),
                    gt_obj.get("contacting", []),
                    gt_obj.get("spatial", []),
                    pred_obj.get("attention", ""),
                    pred_obj.get("contacting", []),
                    pred_obj.get("spatial", []),
                )

                # --- GT + corrections accumulator (all GT objects) ---
                acc_all.add_pair(pred_obj, gt_obj)
                video_gt_count += 1

                # --- Corrections-only ---
                if gt_obj.get("source") == "correction":
                    acc_corr.add_pair(pred_obj, gt_obj)
                    video_corr_count += 1

                # --- Matched objects only (sgdet) ---
                if is_sgdet and is_matched:
                    acc_matched.add_pair(pred_obj, gt_obj)
                    video_matched_count += 1

        log.debug(
            "VIDEO_DONE | video=%s | gt_pairs=%d, corr_pairs=%d, matched_pairs=%d",
            video_id, video_gt_count, video_corr_count, video_matched_count,
        )
        total_pairs += video_gt_count
        pbar.set_postfix(eval=n_videos_evaluated, skip=n_videos_skipped, pairs=total_pairs)

    pbar.close()

    log.info(
        "EVALUATED | videos=%d, skipped=%d",
        n_videos_evaluated, n_videos_skipped,
    )

    # ---- Build results dict ----------------------------------------------
    results: Dict[str, Any] = {
        "mode": mode,
        "model_name": model_name,
        "n_videos_evaluated": n_videos_evaluated,
        "n_videos_skipped": n_videos_skipped,
        "gt_plus_corrections": acc_all.compute(),
        "corrections_only": acc_corr.compute(),
    }

    if is_sgdet:
        results["object_detection"] = obj_det.compute()
        results["matched_objects_only"] = acc_matched.compute()

    # Log final summary
    log.info("=" * 70)
    log.info("RESULTS_SUMMARY | mode=%s | model=%s", mode, model_name)
    if is_sgdet:
        od = results["object_detection"]
        log.info(
            "OBJECT_DETECTION | P=%.4f R=%.4f F1=%.4f | TP=%d FP=%d FN=%d | frames=%d",
            od["precision"], od["recall"], od["f1"],
            od["tp"], od["fp"], od["fn"], od["n_frames"],
        )

    for slice_name in ("gt_plus_corrections", "corrections_only", "matched_objects_only"):
        if slice_name not in results:
            continue
        m = results[slice_name]
        micro = m.get("micro_avg", {})
        log.info(
            "REL_%s | P=%.4f R=%.4f F1=%.4f | pairs=%d",
            slice_name.upper(),
            micro.get("precision", 0), micro.get("recall", 0), micro.get("f1", 0),
            m.get("n_pairs", 0),
        )
    log.info("=" * 70)

    return results


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _print_rel_table(metrics: Dict[str, Any], label: str, show_per_class: bool = True):
    """Print a relationship metrics table with optional per-class breakdown."""
    n_pairs = metrics.get("n_pairs", 0)

    print(f"\n  ▸ {label}  ({n_pairs} object-frame pairs)")
    hdr = f"  {'Rel Type':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}   {'TP':>6} {'FP':>6} {'FN':>6}"
    print(hdr)
    print(f"  {'-' * 75}")

    for rel in ("attention", "contacting", "spatial"):
        m = metrics.get(rel, {})
        if not m:
            continue
        name = rel.replace("_", " ").title()
        # Micro row for this rel type
        print(
            f"  {name:<25} {m.get('precision', 0):>10.4f} "
            f"{m.get('recall', 0):>10.4f} {m.get('f1', 0):>10.4f}   "
            f"{m.get('tp', 0):>6} {m.get('fp', 0):>6} {m.get('fn', 0):>6}"
        )
        # Macro average for this rel type
        macro = m.get("macro_avg", {})
        if macro and macro.get("n_classes", 0) > 0:
            print(
                f"    {'(macro avg)':<23} {macro.get('precision', 0):>10.4f} "
                f"{macro.get('recall', 0):>10.4f} {macro.get('f1', 0):>10.4f}   "
                f"{'':>6} {'':>6} {macro.get('n_classes', 0):>6} cls"
            )
        # Per-class breakdown
        if show_per_class:
            for lbl, lm in sorted(m.get("per_class", {}).items()):
                print(
                    f"      {lbl:<23} {lm.get('precision', 0):>10.4f} "
                    f"{lm.get('recall', 0):>10.4f} {lm.get('f1', 0):>10.4f}   "
                    f"{lm.get('tp', 0):>6} {lm.get('fp', 0):>6} {lm.get('fn', 0):>6}"
                )

    # Overall micro avg
    print(f"  {'-' * 75}")
    micro = metrics.get("micro_avg", {})
    if micro:
        print(
            f"  {'Micro Avg':<25} {micro.get('precision', 0):>10.4f} "
            f"{micro.get('recall', 0):>10.4f} {micro.get('f1', 0):>10.4f}   "
            f"{micro.get('tp', 0):>6} {micro.get('fp', 0):>6} {micro.get('fn', 0):>6}"
        )
    macro_all = metrics.get("macro_avg", {})
    if macro_all:
        print(
            f"  {'Macro Avg':<25} {'':>10} {'':>10} {macro_all.get('f1', 0):>10.4f}   "
            f"{'':>6} {'':>6} {macro_all.get('n_classes', 0):>6} cls"
        )


def print_results(results: Dict[str, Any]):
    """Print evaluation results as formatted tables."""
    if not results:
        print("No results to display.")
        return

    mode = results.get("mode", "?")
    model = results.get("model_name", "?")
    is_sgdet = (mode == "sgdet")

    print(f"\n{'=' * 70}")
    print(f"  Evaluation Results — mode={mode}, model={model}")
    print(f"  Videos evaluated: {results.get('n_videos_evaluated', 0)}, "
          f"skipped: {results.get('n_videos_skipped', 0)}")
    print(f"{'=' * 70}")

    # Object detection table (sgdet only)
    if is_sgdet and "object_detection" in results:
        od = results["object_detection"]
        print(f"\n  ▸ Object Detection  ({od.get('n_frames', 0)} frames)")
        print(f"  {'Metric':<15} {'Value':>10}")
        print(f"  {'-' * 30}")
        for k in ("precision", "recall", "f1"):
            print(f"  {k.title():<15} {od.get(k, 0):>10.4f}")
        print(f"  {'TP':<15} {od.get('tp', 0):>10}")
        print(f"  {'FP':<15} {od.get('fp', 0):>10}")
        print(f"  {'FN':<15} {od.get('fn', 0):>10}")

    # Relationship tables
    _print_rel_table(
        results.get("gt_plus_corrections", {}),
        "GT + Corrections vs Predictions",
    )
    _print_rel_table(
        results.get("corrections_only", {}),
        "Corrections-only vs Predictions",
    )

    if is_sgdet and "matched_objects_only" in results:
        _print_rel_table(
            results["matched_objects_only"],
            "Matched Objects Only (SGDet)",
        )

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate relationship predictions against augmented GT. "
            "Computes P/R/F1 for predcls and sgdet modes."
        ),
    )
    parser.add_argument(
        "--augmented_dir", type=str, default="/data/rohith/ag/wsg_2d_augmentations/",
        help="Directory containing augmented annotation PKL files",
    )
    parser.add_argument(
        "--predictions_dir", type=str, required=True,
        help="Root output directory for the method predictions",
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
        "--method", type=str, default="rag",
        choices=["rag", "caption"],
        help="Method pipeline (rag | caption)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="JSON file to save results (default: /data/rohith/ag/mllms/predictions/<mode>/<method>_<model_name>.json)",
    )
    args = parser.parse_args()

    results = evaluate(
        augmented_dir=args.augmented_dir,
        predictions_dir=args.predictions_dir,
        mode=args.mode,
        model_name=args.model_name,
    )
    results["method"] = args.method

    print_results(results)

    # Determine output path
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(
            "/data/rohith/ag/mllms/predictions",
            args.mode,
            f"{args.method}_{args.model_name}.json",
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
