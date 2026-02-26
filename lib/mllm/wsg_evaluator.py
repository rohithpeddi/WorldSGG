#!/usr/bin/env python3
"""
wsg_evaluator.py
================
Evaluate scene graph predictions against World Scene Graph ground truth.

Three evaluation scopes:
  1. **Standard SGG** — only GT-observed (visible) objects per frame
  2. **World SGG**    — all objects (observed + missing/RAG-predicted)
  3. **Missing SGG**  — only missing (invisible, RAG-predicted) objects

Metrics computed per scope:
  • Per-class Recall for each relationship label
  • Mean Recall across relationship categories
  • Overall Accuracy per relationship type

Usage::

    python -m lib.mllm.wsg_evaluator \\
        --world_sg_dir /data/rohith/ag/world_annotations/world_scene_graph \\
        --predictions_dir /data/rohith/ag/rag_all_objects_results/predcls/kimikvl \\
        --output_file /data/rohith/ag/eval_results.json
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wsg_data_loader import (
    ATTENTION_RELATIONSHIPS,
    CONTACTING_RELATIONSHIPS,
    SPATIAL_RELATIONSHIPS,
    VALID_SCOPES,
    PredictionLoader,
    WorldSGGroundTruthLoader,
    normalize_label,
    _LABEL_ALIASES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Evaluator (scope-agnostic)
# ---------------------------------------------------------------------------

class RelationshipEvaluator:
    """Evaluate relationship predictions against GT for a single scope.

    Tracks per-class true positives and totals for attention, contacting,
    and spatial relationships, then computes per-class recall, mean recall,
    and overall accuracy.
    """

    def __init__(self, scope_name: str):
        self.scope_name = scope_name

        # Per-class TP / total tracking
        self.attention_tp: Dict[str, int] = defaultdict(int)
        self.attention_total: Dict[str, int] = defaultdict(int)

        self.contacting_tp: Dict[str, int] = defaultdict(int)
        self.contacting_total: Dict[str, int] = defaultdict(int)

        self.spatial_tp: Dict[str, int] = defaultdict(int)
        self.spatial_total: Dict[str, int] = defaultdict(int)

        # Overall counts
        self.total_gt_objects = 0
        self.matched_objects = 0

        # Per-video tracking
        self.video_results: Dict[str, Dict[str, Any]] = {}

    def evaluate_frame(
        self,
        gt_per_object: Dict[str, Dict[str, Any]],
        pred_per_object: Dict[str, Dict[str, Any]],
        video_id: str = "",
    ) -> Dict[str, int]:
        """Evaluate predictions for a single frame.

        Parameters
        ----------
        gt_per_object : dict
            ``{object_label: {"attention": [str], "contacting": [str], "spatial": [str]}}``
        pred_per_object : dict
            Same schema, but with single values (not lists) for attention.

        Returns
        -------
        dict with keys ``att_correct``, ``cont_correct``, ``spa_correct``, ``total``.
        """
        att_correct = 0
        cont_correct = 0
        spa_correct = 0
        total = 0

        for gt_label, gt_rels in gt_per_object.items():
            self.total_gt_objects += 1

            # Find matching prediction — try exact match first, then alias
            pred = pred_per_object.get(gt_label)
            if pred is None:
                alias = _LABEL_ALIASES.get(gt_label)
                if alias:
                    pred = pred_per_object.get(alias)

            if pred is None:
                # No prediction for this GT object — count as miss for all
                for label in gt_rels["attention"]:
                    self.attention_total[label] += 1
                for label in gt_rels["contacting"]:
                    self.contacting_total[label] += 1
                for label in gt_rels["spatial"]:
                    self.spatial_total[label] += 1
                continue

            self.matched_objects += 1
            total += 1

            # --- Attention (single-label prediction vs multi-label GT) ---
            gt_att_labels = gt_rels["attention"]
            pred_att = pred.get("attention", "unknown")
            for label in gt_att_labels:
                self.attention_total[label] += 1
            if pred_att in gt_att_labels:
                self.attention_tp[pred_att] += 1
                att_correct += 1

            # --- Contacting (multi-label) ---
            gt_cont_labels = set(gt_rels["contacting"])
            pred_cont = set(pred.get("contacting", ["unknown"]))
            for label in gt_cont_labels:
                self.contacting_total[label] += 1
                if label in pred_cont:
                    self.contacting_tp[label] += 1
                    cont_correct += 1

            # --- Spatial (multi-label) ---
            gt_spa_labels = set(gt_rels["spatial"])
            pred_spa = set(pred.get("spatial", ["unknown"]))
            for label in gt_spa_labels:
                self.spatial_total[label] += 1
                if label in pred_spa:
                    self.spatial_tp[label] += 1
                    spa_correct += 1

        return {
            "att_correct": att_correct,
            "cont_correct": cont_correct,
            "spa_correct": spa_correct,
            "total": total,
        }

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_per_class_recall(
        tp_dict: Dict[str, int],
        total_dict: Dict[str, int],
        all_labels: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """Compute per-class recall and mean recall."""
        per_class: Dict[str, float] = {}
        recalls = []
        for label in all_labels:
            total = total_dict.get(label, 0)
            recall = tp_dict.get(label, 0) / total if total > 0 else 0.0
            per_class[label] = recall
            recalls.append(recall)
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        return per_class, mean_recall

    @staticmethod
    def compute_overall_accuracy(
        tp_dict: Dict[str, int],
        total_dict: Dict[str, int],
    ) -> float:
        """Compute overall accuracy across all classes."""
        total_tp = sum(tp_dict.values())
        total_count = sum(total_dict.values())
        return total_tp / total_count if total_count > 0 else 0.0

    def get_results(self) -> Dict[str, Any]:
        """Compute and return all evaluation metrics."""
        att_per_class, att_mr = self.compute_per_class_recall(
            self.attention_tp, self.attention_total, ATTENTION_RELATIONSHIPS,
        )
        att_acc = self.compute_overall_accuracy(
            self.attention_tp, self.attention_total,
        )

        cont_per_class, cont_mr = self.compute_per_class_recall(
            self.contacting_tp, self.contacting_total, CONTACTING_RELATIONSHIPS,
        )
        cont_acc = self.compute_overall_accuracy(
            self.contacting_tp, self.contacting_total,
        )

        spa_per_class, spa_mr = self.compute_per_class_recall(
            self.spatial_tp, self.spatial_total, SPATIAL_RELATIONSHIPS,
        )
        spa_acc = self.compute_overall_accuracy(
            self.spatial_tp, self.spatial_total,
        )

        overall_mr = float(np.mean([att_mr, cont_mr, spa_mr]))

        return {
            "scope": self.scope_name,
            "total_gt_objects": self.total_gt_objects,
            "matched_objects": self.matched_objects,
            "overall_mean_recall": overall_mr,
            "attention": {
                "accuracy": att_acc,
                "mean_recall": att_mr,
                "per_class_recall": att_per_class,
                "class_counts": dict(self.attention_total),
            },
            "contacting": {
                "accuracy": cont_acc,
                "mean_recall": cont_mr,
                "per_class_recall": cont_per_class,
                "class_counts": dict(self.contacting_total),
            },
            "spatial": {
                "accuracy": spa_acc,
                "mean_recall": spa_mr,
                "per_class_recall": spa_per_class,
                "class_counts": dict(self.spatial_total),
            },
        }

    def print_results(self):
        """Print formatted evaluation results for this scope."""
        results = self.get_results()

        header = f"  {self.scope_name.upper()} SGG Evaluation"
        print(f"\n{'=' * 70}")
        print(header)
        print(f"{'=' * 70}")
        print(f"  GT objects:             {results['total_gt_objects']}")
        print(f"  Matched predictions:    {results['matched_objects']}")
        print(f"  Overall Mean Recall:    {results['overall_mean_recall']:.4f}")
        print()

        for rel_type in ["attention", "contacting", "spatial"]:
            rel_data = results[rel_type]
            print(f"  --- {rel_type.upper()} ---")
            print(f"  Accuracy:     {rel_data['accuracy']:.4f}")
            print(f"  Mean Recall:  {rel_data['mean_recall']:.4f}")
            print(f"  Per-class recall:")
            for label, recall in sorted(
                rel_data["per_class_recall"].items(),
                key=lambda x: -x[1],
            ):
                count = rel_data["class_counts"].get(label, 0)
                bar = "█" * int(recall * 20) + "░" * (20 - int(recall * 20))
                print(f"    {label:30s}  {bar}  {recall:.4f}  (n={count})")
            print()

        print("=" * 70)


# ---------------------------------------------------------------------------
# Evaluation Runner (orchestrates all three scopes)
# ---------------------------------------------------------------------------

class WorldSGGEvaluationRunner:
    """Orchestrates evaluation across Standard, World, and Missing scopes.

    Loads GT from world scene graph PKLs and predictions from RAG output PKLs,
    then runs all three evaluators in a single pass.
    """

    def __init__(
        self,
        world_sg_dir: str,
        predictions_dir: str,
        scopes: Optional[List[str]] = None,
    ):
        self.gt_loader = WorldSGGroundTruthLoader(world_sg_dir)
        self.pred_loader = PredictionLoader(predictions_dir)

        # Which scopes to evaluate
        self.active_scopes = scopes or list(VALID_SCOPES)
        for s in self.active_scopes:
            if s not in VALID_SCOPES:
                raise ValueError(f"Invalid scope '{s}'. Choose from {VALID_SCOPES}")

        # Create one evaluator per active scope
        self.evaluators: Dict[str, RelationshipEvaluator] = {
            scope: RelationshipEvaluator(scope)
            for scope in self.active_scopes
        }

        logger.info(
            f"[EvalRunner] Active scopes: {self.active_scopes}, "
            f"GT videos: {len(self.gt_loader.get_video_ids())}, "
            f"Pred videos: {len(self.pred_loader.get_video_ids())}"
        )

    @staticmethod
    def _frame_key_to_stem(frame_key: str) -> str:
        """``'video_id/000042.png'`` → ``'000042'``"""
        return Path(frame_key).stem

    def _build_pred_lookup(
        self, preds: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Build a lookup dict from prediction list: ``{object_name: pred}``.

        If there are duplicate object names (shouldn't happen normally),
        keeps the first occurrence.
        """
        lookup: Dict[str, Dict[str, Any]] = {}
        for pred in preds:
            obj_name = pred.get("object", "")
            if obj_name and obj_name not in lookup:
                lookup[obj_name] = pred
        return lookup

    def evaluate_video(self, video_id: str):
        """Evaluate predictions for a single video across all active scopes."""
        frame_keys = self.gt_loader.get_frame_keys(video_id)
        if not frame_keys:
            logger.debug(f"No GT frames for {video_id}")
            return

        per_scope_stats: Dict[str, Dict[str, int]] = {
            scope: {"att": 0, "cont": 0, "spa": 0, "total": 0}
            for scope in self.active_scopes
        }

        for frame_key in frame_keys:
            frame_stem = self._frame_key_to_stem(frame_key)

            # Load predictions for this frame
            preds = self.pred_loader.get_predictions_for_frame(
                video_id, frame_stem,
            )
            pred_lookup = self._build_pred_lookup(preds)

            # Evaluate each scope
            for scope in self.active_scopes:
                gt = self.gt_loader.get_gt_for_frame(
                    video_id, frame_key, scope=scope,
                )
                if not gt:
                    continue

                stats = self.evaluators[scope].evaluate_frame(
                    gt, pred_lookup, video_id=video_id,
                )
                per_scope_stats[scope]["att"] += stats["att_correct"]
                per_scope_stats[scope]["cont"] += stats["cont_correct"]
                per_scope_stats[scope]["spa"] += stats["spa_correct"]
                per_scope_stats[scope]["total"] += stats["total"]

        # Store per-video summary in each evaluator
        for scope in self.active_scopes:
            self.evaluators[scope].video_results[video_id] = per_scope_stats[scope]

    def evaluate_all(
        self,
        video_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ):
        """Evaluate all videos (or a subset).

        If *video_ids* is None, uses the intersection of GT and prediction
        video IDs.
        """
        if video_ids is None:
            gt_ids = set(self.gt_loader.get_video_ids())
            pred_ids = set(self.pred_loader.get_video_ids())
            common_ids = sorted(gt_ids & pred_ids)
            logger.info(
                f"[EvalRunner] GT: {len(gt_ids)}, Pred: {len(pred_ids)}, "
                f"Common: {len(common_ids)}"
            )
            video_ids = common_ids

        if limit is not None:
            video_ids = video_ids[:limit]

        logger.info(f"[EvalRunner] Evaluating {len(video_ids)} videos …")

        for video_id in tqdm(video_ids, desc="Evaluating"):
            try:
                self.evaluate_video(video_id)
            except Exception as e:
                logger.error(f"[{video_id}] Evaluation error: {e}")

    def get_combined_results(self) -> Dict[str, Any]:
        """Return results for all active scopes."""
        combined: Dict[str, Any] = {}
        for scope in self.active_scopes:
            combined[scope] = self.evaluators[scope].get_results()

        # Sanity check: ensure world = standard + missing (object counts)
        if "standard" in combined and "missing" in combined and "world" in combined:
            std_gt = combined["standard"]["total_gt_objects"]
            mis_gt = combined["missing"]["total_gt_objects"]
            wld_gt = combined["world"]["total_gt_objects"]
            combined["_sanity_check"] = {
                "standard_gt": std_gt,
                "missing_gt": mis_gt,
                "world_gt": wld_gt,
                "sum_matches_world": (std_gt + mis_gt) == wld_gt,
            }

        return combined

    def print_all_results(self):
        """Print results for all active scopes."""
        combined = self.get_combined_results()

        for scope in self.active_scopes:
            self.evaluators[scope].print_results()

        # Print sanity check
        if "_sanity_check" in combined:
            sc = combined["_sanity_check"]
            print(f"\n{'─' * 70}")
            print(f"  Sanity Check: Standard({sc['standard_gt']}) + "
                  f"Missing({sc['missing_gt']}) = {sc['standard_gt'] + sc['missing_gt']}  "
                  f"vs  World({sc['world_gt']})  "
                  f"{'✓' if sc['sum_matches_world'] else '✗ MISMATCH'}")
            print(f"{'─' * 70}")

        # Summary table
        print(f"\n{'=' * 70}")
        print("  SUMMARY: Mean Recall across Scopes")
        print(f"{'=' * 70}")
        print(f"  {'Scope':<15s}  {'Att MR':>8s}  {'Cont MR':>8s}  {'Spa MR':>8s}  {'Overall MR':>10s}")
        print(f"  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
        for scope in self.active_scopes:
            r = combined[scope]
            print(
                f"  {scope:<15s}  "
                f"{r['attention']['mean_recall']:>8.4f}  "
                f"{r['contacting']['mean_recall']:>8.4f}  "
                f"{r['spatial']['mean_recall']:>8.4f}  "
                f"{r['overall_mean_recall']:>10.4f}"
            )
        print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate scene graph predictions against World Scene Graph "
            "ground truth across Standard, World, and Missing scopes."
        ),
    )
    parser.add_argument(
        "--world_sg_dir", type=str, required=True,
        help=(
            "Directory with world scene graph PKLs (GT source). "
            "E.g. /data/rohith/ag/world_annotations/world_scene_graph"
        ),
    )
    parser.add_argument(
        "--predictions_dir", type=str, required=True,
        help=(
            "Directory with prediction PKLs from process_ag_rag_all.py. "
            "E.g. /data/rohith/ag/rag_all_objects_results/predcls/kimikvl"
        ),
    )
    parser.add_argument(
        "--scope", type=str, default=None,
        choices=["standard", "world", "missing"],
        help="Evaluate only this scope. If omitted, all three are evaluated.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate at most this many videos (for dev/debug).",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Optional path to save results as JSON.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    scopes = [args.scope] if args.scope else None
    runner = WorldSGGEvaluationRunner(
        world_sg_dir=args.world_sg_dir,
        predictions_dir=args.predictions_dir,
        scopes=scopes,
    )

    runner.evaluate_all(limit=args.limit)
    runner.print_all_results()

    results = runner.get_combined_results()

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                results, f, indent=2,
                default=lambda x: float(x) if isinstance(x, np.floating) else x,
            )
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
