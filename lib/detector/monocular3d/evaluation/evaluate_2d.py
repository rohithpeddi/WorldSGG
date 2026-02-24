#!/usr/bin/env python3
"""
Standalone 2D evaluation script for DINOv2 Monocular 3D detector.

Reports COCO-style 2D metrics:
  - mAP averaged over IoU 0.5:0.95:0.05
  - mAP@50, mAP@75
  - Per-class AP breakdown
  - Simple precision/recall at a fixed IoU threshold (diagnostic)

Usage:
    # Full 2D evaluation
    python -m lib.detector.monocular3d.evaluation.evaluate_2d \
        --checkpoint /path/to/checkpoint_XX \
        --data_path /path/to/Datasets/action_genome

    # Limit test size and save JSON
    python -m lib.detector.monocular3d.evaluation.evaluate_2d \
        --checkpoint /path/to/checkpoint_XX \
        --data_path /path/to/Datasets/action_genome \
        --max_test_samples 2000 \
        --output results/metrics_2d.json

    # Quick diagnostic precision/recall only (skip full mAP)
    python -m lib.detector.monocular3d.evaluation.evaluate_2d \
        --checkpoint /path/to/checkpoint_XX \
        --no_coco_map
"""

import argparse
import gc
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# Project root: .../Scene4Cast/lib/detector/monocular3d/evaluation/evaluate_2d.py → Scene4Cast
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MONO3D_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_MONO3D_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ..datasets.ag_dataset_3d import ActionGenomeDataset3D, collate_fn
from ..models.dino_mono_3d import DinoV3Monocular3D


# ────────────────────────────────────────────────────────────────────
#  Utility
# ────────────────────────────────────────────────────────────────────

def clear_cuda_cache_for_current_process(sync: bool = True) -> None:
    """Clear CUDA cache for all visible devices in this process."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    if sync:
        torch.cuda.synchronize()
    for dev in range(torch.cuda.device_count()):
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────────────
#  IoU & matching helpers
# ────────────────────────────────────────────────────────────────────

def compute_iou_2d(box1: np.ndarray, box2: np.ndarray) -> float:
    """2D IoU for [x1, y1, x2, y2]."""
    if isinstance(box1, torch.Tensor):
        box1 = box1.detach().cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.detach().cpu().numpy()
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


# ────────────────────────────────────────────────────────────────────
#  Per-class AP formatting
# ────────────────────────────────────────────────────────────────────

def format_per_class_ap(ap_array: np.ndarray, class_names: list) -> str:
    """Pretty-print per-class AP as a table string."""
    lines = []
    lines.append(f"  {'Class':<25s} {'AP':>8s}")
    lines.append("  " + "-" * 35)
    for idx, ap_val in enumerate(ap_array):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        lines.append(f"  {name:<25s} {float(ap_val):8.4f}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
#  Simple precision / recall at a fixed IoU threshold
# ────────────────────────────────────────────────────────────────────

def evaluate_precision_recall(
    model, dataloader, device, iou_threshold: float = 0.5
) -> dict:
    """
    Simple precision and recall at a fixed IoU threshold over all test data.
    Also reports per-class precision and recall.
    """
    model.eval()

    # Per-class accumulators: class_id -> {'TP': int, 'FP': int, 'FN': int}
    class_stats = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="2D Precision/Recall", ascii=True):
            images = torch.stack([img for img in images]).to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images, targets_dev)

            for i in range(len(images)):
                pred_boxes = outputs[i]["boxes"].detach().cpu().numpy()
                pred_labels = outputs[i]["labels"].detach().cpu().numpy()
                pred_scores = outputs[i]["scores"].detach().cpu().numpy()

                gt_boxes = targets[i]["boxes"].detach().cpu().numpy()
                gt_labels = targets[i]["labels"].detach().cpu().numpy()

                # Sort predictions by score (descending) for greedy matching
                order = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[order]
                pred_labels = pred_labels[order]
                pred_scores = pred_scores[order]

                used_gt = set()

                for p_idx in range(len(pred_boxes)):
                    p_label = int(pred_labels[p_idx])
                    best_iou = 0.0
                    best_gt = -1

                    for g_idx in range(len(gt_boxes)):
                        if g_idx in used_gt or int(gt_labels[g_idx]) != p_label:
                            continue
                        iou = compute_iou_2d(pred_boxes[p_idx], gt_boxes[g_idx])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = g_idx

                    if best_gt >= 0 and best_iou >= iou_threshold:
                        used_gt.add(best_gt)
                        total_tp += 1
                        class_stats.setdefault(p_label, {"TP": 0, "FP": 0, "FN": 0})
                        class_stats[p_label]["TP"] += 1
                    else:
                        total_fp += 1
                        class_stats.setdefault(p_label, {"TP": 0, "FP": 0, "FN": 0})
                        class_stats[p_label]["FP"] += 1

                # Unmatched GT are false negatives
                for g_idx in range(len(gt_boxes)):
                    if g_idx not in used_gt:
                        g_label = int(gt_labels[g_idx])
                        total_fn += 1
                        class_stats.setdefault(g_label, {"TP": 0, "FP": 0, "FN": 0})
                        class_stats[g_label]["FN"] += 1

            clear_cuda_cache_for_current_process(sync=False)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-class precision/recall
    per_class = {}
    for cls_id, stats in sorted(class_stats.items()):
        tp, fp, fn = stats["TP"], stats["FP"], stats["FN"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[cls_id] = {"precision": p, "recall": r, "TP": tp, "FP": fp, "FN": fn}

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "iou_threshold": iou_threshold,
        "per_class": per_class,
    }


# ────────────────────────────────────────────────────────────────────
#  COCO-style 2D mAP via torchmetrics (self-contained)
# ────────────────────────────────────────────────────────────────────

def evaluate_2d_coco_map(
    model,
    dataloader,
    device,
    accelerator=None,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Full COCO-style 2D mAP evaluation using torchmetrics.MeanAveragePrecision.

    Args:
        model: Detection model in eval mode.
        dataloader: DataLoader yielding (images, targets).
        device: torch.device for inference.
        accelerator: Optional accelerator (multi-GPU). If set and not main, returns {}.
        iou_thresholds: IoU thresholds for mAP. None = COCO default [0.5:0.95:0.05].

    Returns:
        {
            "map": float,          # mAP averaged over IoU 0.5:0.95:0.05
            "map_50": float,       # mAP at IoU 0.5
            "map_75": float,       # mAP at IoU 0.75
            "map_per_class": np.ndarray or None,
            "raw": dict,           # full torchmetrics output
        }
    """
    model.eval()
    map_metric = MeanAveragePrecision(
        iou_type="bbox",
        sync_on_compute=False,
        iou_thresholds=iou_thresholds,  # None => COCO default [0.5:0.95:0.05]
    )

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="2D COCO mAP", ascii=True):
            # images is already a list of tensors from the DataLoader batch
            batch_images = torch.stack(images).to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(batch_images, targets_dev)

            preds, gts = [], []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                })

            for t in targets:
                gts.append({
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu(),
                })

            min_len = min(len(preds), len(gts))
            if min_len > 0:
                map_metric.update(preds[:min_len], gts[:min_len])

            del batch_images, outputs, preds, gts

    if accelerator is not None and not accelerator.is_main_process:
        return {}

    raw = map_metric.compute()

    out: Dict[str, Any] = {"raw": raw}
    out["map"] = float(raw["map"].item())
    out["map_50"] = float(raw["map_50"].item())
    out["map_75"] = float(raw["map_75"].item())
    out["map_per_class"] = (
        raw.get("map_per_class", None).cpu().numpy()
        if raw.get("map_per_class", None) is not None
        else None
    )
    return out


# ────────────────────────────────────────────────────────────────────
#  Fused 2D + 3D evaluation (single forward pass)
# ────────────────────────────────────────────────────────────────────

def evaluate_2d_and_3d_fused(
    model,
    dataloader,
    device,
    accelerator=None,
    iou_thresholds: Optional[List[float]] = None,
    iou_threshold_2d_match: float = 0.5,
) -> Dict[str, Any]:
    """
    Fused evaluation: computes COCO-style 2D mAP and 3D metrics in ONE forward pass.

    Returns:
        {
            "metrics_2d": { "map", "map_50", "map_75", "map_per_class", "raw" },
            "metrics_3d": { "n_matched", "n_gt_3d", "chamfer_mean", "corner_l2_mean",
                            "iou3d_hit_50", "iou3d_hit_75", "mean_iou_3d",
                            "center_l2_mean", "dims_l1_mean", "rotation_deg_mean" },
        }
    """
    from .evaluate_3d import (
        match_predictions_to_gt_2d,
        chamfer_per_box,
        corner_l2_per_box,
        compute_iou_3d_obb,
        compute_3d_attribute_errors,
    )

    model.eval()

    # ── 2D accumulator ──
    map_metric = MeanAveragePrecision(
        iou_type="bbox",
        sync_on_compute=False,
        iou_thresholds=iou_thresholds,
    )

    # ── 3D accumulators ──
    chamfer_list, corner_l2_list, iou_3d_list = [], [], []
    center_l2_list, dims_l1_list, rotation_deg_list = [], [], []
    n_matched, n_gt_total = 0, 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Eval 2D+3D (fused)", ascii=True):
            batch_images = torch.stack(images).to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(batch_images, targets_dev)

            # ── Feed 2D accumulator ──
            preds_2d, gts_2d = [], []
            for output in outputs:
                preds_2d.append({
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                })
            for t in targets:
                gts_2d.append({
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu(),
                })
            min_len = min(len(preds_2d), len(gts_2d))
            if min_len > 0:
                map_metric.update(preds_2d[:min_len], gts_2d[:min_len])

            # ── Feed 3D accumulator ──
            for i in range(len(images)):
                pred_boxes = outputs[i]["boxes"].detach().cpu().numpy()
                pred_labels = outputs[i]["labels"].detach().cpu().numpy()
                pred_scores = outputs[i]["scores"].detach().cpu().numpy()
                pred_3d = outputs[i]["boxes_3d"].detach().cpu().numpy()

                gt_boxes = targets[i]["boxes"].detach().cpu().numpy()
                gt_labels = targets[i]["labels"].detach().cpu().numpy()
                gt_3d = targets[i]["boxes_3d"].detach().cpu().numpy()

                # Filter out GT with near-zero 3D annotations
                valid_gt = (np.abs(gt_3d.reshape(gt_3d.shape[0], -1)).sum(axis=1) > 1e-6)
                gt_boxes = gt_boxes[valid_gt]
                gt_labels = gt_labels[valid_gt]
                gt_3d = gt_3d[valid_gt]

                n_gt_total += len(gt_boxes)  # count only GT with valid 3D annotations

                if len(gt_boxes) == 0 or pred_3d.shape[0] == 0:
                    continue

                pairs = match_predictions_to_gt_2d(
                    pred_boxes, pred_labels, pred_scores, pred_3d,
                    gt_boxes, gt_labels, gt_3d,
                    iou_threshold=iou_threshold_2d_match,
                )
                for pred_c, gt_c in pairs:
                    gt_flat = gt_c.reshape(-1)
                    if np.abs(gt_flat).sum() < 1e-6:
                        continue
                    pred_c = np.asarray(pred_c, dtype=np.float64)
                    gt_c = np.asarray(gt_c, dtype=np.float64)
                    chamfer_list.append(chamfer_per_box(pred_c, gt_c))
                    corner_l2_list.append(corner_l2_per_box(pred_c, gt_c))
                    iou_3d_list.append(compute_iou_3d_obb(pred_c.reshape(8, 3), gt_c.reshape(8, 3)))
                    attrs = compute_3d_attribute_errors(pred_c, gt_c)
                    center_l2_list.append(attrs["center_l2"])
                    dims_l1_list.append(attrs["dims_l1"])
                    rotation_deg_list.append(attrs["rotation_deg"])
                    n_matched += 1

            del batch_images, outputs

    # ── Compute 2D results ──
    metrics_2d = {}
    if accelerator is None or accelerator.is_main_process:
        raw = map_metric.compute()
        metrics_2d = {
            "map": float(raw["map"].item()),
            "map_50": float(raw["map_50"].item()),
            "map_75": float(raw["map_75"].item()),
            "map_per_class": (
                raw.get("map_per_class", None).cpu().numpy()
                if raw.get("map_per_class", None) is not None
                else None
            ),
            "raw": raw,
        }

    # ── Compute 3D results ──
    iou_3d = np.array(iou_3d_list) if iou_3d_list else np.array([0.0])
    metrics_3d = {
        "n_matched": n_matched,
        "n_gt_3d": n_gt_total,
        "chamfer_mean": float(np.mean(chamfer_list)) if chamfer_list else 0.0,
        "corner_l2_mean": float(np.mean(corner_l2_list)) if corner_l2_list else 0.0,
        "iou3d_hit_50": float((iou_3d >= 0.5).mean()),  # hit rate, not mAP
        "iou3d_hit_75": float((iou_3d >= 0.75).mean()),  # hit rate, not mAP
        "mean_iou_3d": float(iou_3d.mean()),
        "center_l2_mean": float(np.mean(center_l2_list)) if center_l2_list else 0.0,
        "dims_l1_mean": float(np.mean(dims_l1_list)) if dims_l1_list else 0.0,
        "rotation_deg_mean": float(np.mean(rotation_deg_list)) if rotation_deg_list else 0.0,
    }

    return {"metrics_2d": metrics_2d, "metrics_3d": metrics_3d}


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2D Evaluation for DINOv2 Monocular 3D detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint dir (containing checkpoint_state.pth) or to .pth file")
    parser.add_argument("--data_path", type=str, default="/data/rohith/ag/",
                        help="Action Genome root (frames + annotations)")
    parser.add_argument("--world_3d_annotations_path", type=str, default=None,
                        help="Folder of 3D pkls; default data_path/world_annotations/bbox_annotations_3d_final")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Cap number of test samples (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save metrics JSON to this path")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for simple precision/recall")
    parser.add_argument("--no_coco_map", action="store_true",
                        help="Skip COCO-style mAP (only compute simple P/R)")
    parser.add_argument("--no_precision_recall", action="store_true",
                        help="Skip simple precision/recall diagnostic")
    parser.add_argument("--model", type=str, default="v3l",
                        help="Model variant: v2, v2l, v3l")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Dataset ──
    kwargs = {"phase": "test", "target_size": args.target_size}
    if args.world_3d_annotations_path:
        kwargs["world_3d_annotations_path"] = args.world_3d_annotations_path
    test_dataset = ActionGenomeDataset3D(args.data_path, **kwargs)

    if args.max_test_samples is not None:
        test_dataset = torch.utils.data.Subset(
            test_dataset, list(range(min(args.max_test_samples, len(test_dataset))))
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── Model ──
    ds = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset
    num_classes = len(ds.object_classes) if hasattr(ds, "object_classes") else 37
    class_names = list(ds.object_classes) if hasattr(ds, "object_classes") else [f"class_{i}" for i in range(num_classes)]

    model = DinoV3Monocular3D(num_classes=num_classes, pretrained=False, model=args.model)
    model.to(device)
    model.eval()

    # ── Load checkpoint ──
    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "checkpoint_state.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    print(f"✓ Loaded checkpoint: {ckpt_path}")
    print(f"  Model: {args.model}  |  Classes: {num_classes}  |  Device: {device}")
    print(f"  Test samples: {len(test_dataset)}")
    print()

    metrics = {}

    # ── Simple Precision / Recall ──
    if not args.no_precision_recall:
        print("=" * 60)
        print(f"  2D Precision / Recall  (IoU ≥ {args.iou_threshold})")
        print("=" * 60)
        pr_metrics = evaluate_precision_recall(model, test_loader, device, iou_threshold=args.iou_threshold)
        metrics["precision_recall"] = {
            "precision": pr_metrics["precision"],
            "recall": pr_metrics["recall"],
            "f1": pr_metrics["f1"],
            "TP": pr_metrics["TP"],
            "FP": pr_metrics["FP"],
            "FN": pr_metrics["FN"],
            "iou_threshold": pr_metrics["iou_threshold"],
        }
        print(f"  Precision: {pr_metrics['precision']:.4f}")
        print(f"  Recall:    {pr_metrics['recall']:.4f}")
        print(f"  F1:        {pr_metrics['f1']:.4f}")
        print(f"  TP: {pr_metrics['TP']}  FP: {pr_metrics['FP']}  FN: {pr_metrics['FN']}")

        # Per-class table
        print()
        print(f"  {'Class':<25s} {'Prec':>8s} {'Recall':>8s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
        print("  " + "-" * 60)
        for cls_id, stats in sorted(pr_metrics["per_class"].items()):
            name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            print(f"  {name:<25s} {stats['precision']:8.4f} {stats['recall']:8.4f} "
                  f"{stats['TP']:6d} {stats['FP']:6d} {stats['FN']:6d}")
        print()

    # ── COCO-style mAP ──
    if not args.no_coco_map:
        print("=" * 60)
        print("  COCO-style 2D mAP  (IoU 0.50:0.95:0.05)")
        print("=" * 60)
        coco_metrics = evaluate_2d_coco_map(model, test_loader, device, accelerator=None)

        if coco_metrics:
            metrics["coco_map"] = {
                "map": coco_metrics.get("map", 0.0),
                "map_50": coco_metrics.get("map_50", 0.0),
                "map_75": coco_metrics.get("map_75", 0.0),
            }
            print(f"  mAP:        {coco_metrics['map']:.4f}")
            print(f"  mAP@50:     {coco_metrics['map_50']:.4f}")
            print(f"  mAP@75:     {coco_metrics['map_75']:.4f}")

            if coco_metrics.get("map_per_class") is not None:
                print()
                print(format_per_class_ap(coco_metrics["map_per_class"], class_names))
        else:
            print("  ⚠️  COCO mAP computation returned empty results.")
        print()

    # ── Save ──
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"✓ Metrics saved to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
