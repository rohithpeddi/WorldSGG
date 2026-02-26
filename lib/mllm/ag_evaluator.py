#!/usr/bin/env python3
"""
ag_evaluator.py
===============
Evaluate scene graph predictions against Action Genome ground truth.

Metrics computed:
  • Per-relationship-type Recall@K (K=1,3,5)
  • Mean Recall across relationship categories
  • Per-class accuracy for individual labels (e.g., "looking_at", "holding")
  • Overall accuracy per relationship type

Usage:
    python -m evaluation.ag_evaluator \
        --predictions_dir /data/rohith/ag/sg_results/kimikvl/ \
        --ag_root_directory /data/rohith/ag \
        --split test
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relationship label sets (Action Genome vocabulary)
# ---------------------------------------------------------------------------

ATTENTION_RELATIONSHIPS = [
    "looking_at", "not_looking_at", "unsure",
]

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
# GT loader (from ag_wsg_data / AG annotations)
# ---------------------------------------------------------------------------

class AGGroundTruthLoader:
    """Load ground truth scene graph annotations from Action Genome."""

    def __init__(self, ag_root_directory: str):
        self.ag_root = Path(ag_root_directory)
        self.annotations_path = self.ag_root / "annotations"

        # Object class names
        self.object_classes = [
            '__background__', 'person', 'bag', 'bed', 'blanket', 'book', 'box',
            'broom', 'chair', 'closet/cabinet', 'clothes', 'cup/glass/bottle',
            'dish', 'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries',
            'laptop', 'light', 'medicine', 'mirror', 'paper/notebook',
            'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich',
            'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel',
            'vacuum', 'window',
        ]

        # Load annotation files
        self.person_bbox = {}
        self.object_bbox = {}
        self._load_annotations()

    def _load_annotations(self):
        person_path = self.annotations_path / "person_bbox.pkl"
        object_path = self.annotations_path / "object_bbox_and_relationship.pkl"

        if person_path.exists():
            with open(person_path, "rb") as f:
                self.person_bbox = pickle.load(f)
        else:
            logger.warning(f"Person bbox file not found: {person_path}")

        if object_path.exists():
            with open(object_path, "rb") as f:
                self.object_bbox = pickle.load(f)
        else:
            logger.warning(f"Object bbox file not found: {object_path}")

    def get_gt_for_frame(
        self, frame_key: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get GT relationship annotations for a frame.

        Returns:
            {object_name: {"attention": [str], "contacting": [str], "spatial": [str]}}
        """
        if frame_key not in self.object_bbox:
            return {}

        gt: Dict[str, Dict[str, Any]] = {}
        for obj_ann in self.object_bbox[frame_key]:
            if not obj_ann.get("visible", False):
                continue

            cls_idx = obj_ann.get("class")
            if cls_idx is None or cls_idx == 0:
                continue

            if isinstance(cls_idx, int) and cls_idx < len(self.object_classes):
                object_name = self.object_classes[cls_idx]
            elif isinstance(cls_idx, str):
                object_name = cls_idx
            else:
                continue

            if object_name in ("person", "__background__"):
                continue

            att_rel = obj_ann.get("attention_relationship", [])
            cont_rel = obj_ann.get("contacting_relationship", [])
            spa_rel = obj_ann.get("spatial_relationship", [])

            gt[object_name] = {
                "attention": att_rel if att_rel else ["unsure"],
                "contacting": cont_rel if cont_rel else ["not_contacting"],
                "spatial": spa_rel if spa_rel else ["in_front_of"],
            }

        return gt

    def get_video_frame_keys(self, video_id: str, phase: str = "testing") -> List[str]:
        """Get all frame keys for a video that belong to the given phase."""
        frame_keys = []
        prefix = video_id.replace(".mp4", "") + "/"
        for key in self.object_bbox:
            if key.startswith(prefix):
                metadata = self.object_bbox[key][0].get("metadata", {})
                if metadata.get("set") == phase:
                    frame_keys.append(key)
        return sorted(frame_keys)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class SceneGraphEvaluator:
    """Evaluate scene graph predictions against ground truth."""

    def __init__(self, gt_loader: AGGroundTruthLoader):
        self.gt_loader = gt_loader

        # Tracking per-class metrics
        self.attention_tp: Dict[str, int] = defaultdict(int)
        self.attention_total: Dict[str, int] = defaultdict(int)

        self.contacting_tp: Dict[str, int] = defaultdict(int)
        self.contacting_total: Dict[str, int] = defaultdict(int)

        self.spatial_tp: Dict[str, int] = defaultdict(int)
        self.spatial_total: Dict[str, int] = defaultdict(int)

        # Overall counts
        self.total_objects = 0
        self.matched_objects = 0

        # Per-video results
        self.video_results: Dict[str, Dict[str, Any]] = {}

    def evaluate_video(
        self,
        video_id: str,
        predictions: Dict[str, Any],
        phase: str = "testing",
    ):
        """
        Evaluate predictions for a single video against GT.

        Args:
            video_id: Video identifier
            predictions: Dict loaded from the prediction pickle file
            phase: 'testing' or 'train'
        """
        frames_pred = predictions.get("frames", {})
        if not frames_pred:
            return

        video_att_correct = 0
        video_cont_correct = 0
        video_spa_correct = 0
        video_total = 0

        for frame_stem, frame_pred in frames_pred.items():
            pred_list = frame_pred.get("predictions", [])
            if not pred_list:
                continue

            # Try to match frame_stem to a GT frame key
            # frame_stem could be like "000001" or "video_id/000001.png"
            gt_frame_keys = self._resolve_gt_frame_key(video_id, frame_stem)
            if not gt_frame_keys:
                continue

            for gt_key in gt_frame_keys:
                gt_per_object = self.gt_loader.get_gt_for_frame(gt_key)
                if not gt_per_object:
                    continue

                for pred in pred_list:
                    obj_name = pred.get("object_name", pred.get("missing_object", ""))
                    if obj_name not in gt_per_object:
                        continue

                    self.total_objects += 1
                    self.matched_objects += 1
                    video_total += 1

                    gt_obj = gt_per_object[obj_name]

                    # --- Attention (single-label) ---
                    pred_att = pred.get("attention", "unknown")
                    gt_att_labels = gt_obj["attention"]
                    if pred_att in gt_att_labels:
                        self.attention_tp[pred_att] += 1
                        video_att_correct += 1
                    for label in gt_att_labels:
                        self.attention_total[label] += 1

                    # --- Contacting (multi-label) ---
                    pred_cont = pred.get("contacting", ["unknown"])
                    gt_cont_labels = set(gt_obj["contacting"])
                    for label in gt_cont_labels:
                        self.contacting_total[label] += 1
                        if label in pred_cont:
                            self.contacting_tp[label] += 1
                            video_cont_correct += 1

                    # --- Spatial (multi-label) ---
                    pred_spa = pred.get("spatial", ["unknown"])
                    gt_spa_labels = set(gt_obj["spatial"])
                    for label in gt_spa_labels:
                        self.spatial_total[label] += 1
                        if label in pred_spa:
                            self.spatial_tp[label] += 1
                            video_spa_correct += 1

        self.video_results[video_id] = {
            "total_objects": video_total,
            "attention_correct": video_att_correct,
            "contacting_correct": video_cont_correct,
            "spatial_correct": video_spa_correct,
        }

    def _resolve_gt_frame_key(
        self, video_id: str, frame_stem: str
    ) -> List[str]:
        """Try to resolve a frame stem to GT frame keys."""
        clean_vid = video_id.replace(".mp4", "")

        # Try direct match: "video_id/frame_stem"
        candidates = [
            f"{clean_vid}/{frame_stem}",
            f"{clean_vid}/{frame_stem}.png",
            frame_stem,
        ]

        matched = []
        for key in candidates:
            if key in self.gt_loader.object_bbox:
                matched.append(key)

        if not matched:
            # Try prefix matching
            for key in self.gt_loader.object_bbox:
                if key.startswith(f"{clean_vid}/") and frame_stem in key:
                    matched.append(key)
                    break

        return matched

    def compute_per_class_recall(
        self,
        tp_dict: Dict[str, int],
        total_dict: Dict[str, int],
        all_labels: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """
        Compute per-class recall and mean recall.

        Returns:
            (per_class_recall_dict, mean_recall)
        """
        per_class: Dict[str, float] = {}
        recalls = []
        for label in all_labels:
            total = total_dict.get(label, 0)
            if total > 0:
                recall = tp_dict.get(label, 0) / total
            else:
                recall = 0.0
            per_class[label] = recall
            recalls.append(recall)

        mean_recall = np.mean(recalls) if recalls else 0.0
        return per_class, float(mean_recall)

    def compute_overall_accuracy(
        self,
        tp_dict: Dict[str, int],
        total_dict: Dict[str, int],
    ) -> float:
        """Compute overall accuracy across all classes."""
        total_tp = sum(tp_dict.values())
        total_count = sum(total_dict.values())
        return total_tp / total_count if total_count > 0 else 0.0

    def get_results(self) -> Dict[str, Any]:
        """Compute and return all evaluation metrics."""
        # Attention
        att_per_class, att_mean_recall = self.compute_per_class_recall(
            self.attention_tp, self.attention_total, ATTENTION_RELATIONSHIPS
        )
        att_accuracy = self.compute_overall_accuracy(
            self.attention_tp, self.attention_total
        )

        # Contacting
        cont_per_class, cont_mean_recall = self.compute_per_class_recall(
            self.contacting_tp, self.contacting_total, CONTACTING_RELATIONSHIPS
        )
        cont_accuracy = self.compute_overall_accuracy(
            self.contacting_tp, self.contacting_total
        )

        # Spatial
        spa_per_class, spa_mean_recall = self.compute_per_class_recall(
            self.spatial_tp, self.spatial_total, SPATIAL_RELATIONSHIPS
        )
        spa_accuracy = self.compute_overall_accuracy(
            self.spatial_tp, self.spatial_total
        )

        # Overall mean recall across all three types
        overall_mean_recall = np.mean([att_mean_recall, cont_mean_recall, spa_mean_recall])

        return {
            "total_objects_evaluated": self.total_objects,
            "matched_objects": self.matched_objects,
            "num_videos": len(self.video_results),
            "overall_mean_recall": float(overall_mean_recall),
            "attention": {
                "accuracy": att_accuracy,
                "mean_recall": att_mean_recall,
                "per_class_recall": att_per_class,
                "class_counts": dict(self.attention_total),
            },
            "contacting": {
                "accuracy": cont_accuracy,
                "mean_recall": cont_mean_recall,
                "per_class_recall": cont_per_class,
                "class_counts": dict(self.contacting_total),
            },
            "spatial": {
                "accuracy": spa_accuracy,
                "mean_recall": spa_mean_recall,
                "per_class_recall": spa_per_class,
                "class_counts": dict(self.spatial_total),
            },
            "per_video": self.video_results,
        }

    def print_results(self):
        """Print formatted evaluation results."""
        results = self.get_results()

        print("\n" + "=" * 70)
        print("  Action Genome Scene Graph Evaluation Results")
        print("=" * 70)
        print(f"  Videos evaluated:       {results['num_videos']}")
        print(f"  Total objects matched:  {results['matched_objects']}")
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
                key=lambda x: -x[1]
            ):
                count = rel_data["class_counts"].get(label, 0)
                bar = "█" * int(recall * 20) + "░" * (20 - int(recall * 20))
                print(f"    {label:30s}  {bar}  {recall:.4f}  (n={count})")
            print()

        print("=" * 70)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions_dir: str,
    ag_root_directory: str,
    split: str = "test",
    output_file: Optional[str] = None,
):
    """
    Load all prediction pickle files from a directory and evaluate
    against Action Genome ground truth.
    """
    pred_dir = Path(predictions_dir)
    if not pred_dir.exists():
        logger.error(f"Predictions directory not found: {pred_dir}")
        return None

    # Find all prediction pickle files
    pkl_files = sorted(pred_dir.glob("*.pkl"))
    if not pkl_files:
        logger.error(f"No .pkl files found in {pred_dir}")
        return None

    logger.info(f"Found {len(pkl_files)} prediction files in {pred_dir}")

    # Load GT
    gt_loader = AGGroundTruthLoader(ag_root_directory)
    evaluator = SceneGraphEvaluator(gt_loader)

    # Evaluate each video
    for pkl_path in tqdm(pkl_files, desc="Evaluating"):
        try:
            with open(pkl_path, "rb") as f:
                predictions = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {pkl_path}: {e}")
            continue

        video_id = predictions.get("video_id", pkl_path.stem)
        evaluator.evaluate_video(video_id, predictions, phase="testing")

    # Print and save results
    evaluator.print_results()
    results = evaluator.get_results()

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            # Convert numpy types for JSON serialization
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        logger.info(f"Results saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate scene graph predictions against Action Genome ground truth.",
    )
    parser.add_argument(
        "--predictions_dir", type=str, required=True,
        help=(
            "Directory containing prediction pickle files. "
            "Can be the output of zero_shot_ag.py (e.g., /data/rohith/ag/sg_results/kimikvl/)"
        ),
    )
    parser.add_argument(
        "--ag_root_directory", type=str,
        default="/data/rohith/ag",
        help="Root directory of the Action Genome dataset",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to evaluate (test or train)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Optional path to save results as JSON",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    evaluate_predictions(
        predictions_dir=args.predictions_dir,
        ag_root_directory=args.ag_root_directory,
        split=args.split,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    from tqdm import tqdm
    main()
