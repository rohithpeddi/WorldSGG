#!/usr/bin/env python3
"""
Evaluation script for DINOv2 Monocular 3D detector.

Reports:
  - 2D: COCO-style mAP (map, map_50, map_75, map_per_class).
  - 3D: Chamfer distance, corner L2 error, axis-aligned 3D IoU (mAP_3d_50, mAP_3d_75),
        and per-attribute errors (center, dimensions, rotation) on matched pred-GT pairs.

Matching: For each image, predictions are matched to GT by 2D box IoU and same label.
Only matched pairs contribute to 3D metrics. Run from Scene4Cast root or with PYTHONPATH set.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Project root: from .../Scene4Cast/lib/detector/monocular3d/evaluation/evaluate_3d.py -> Scene4Cast
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MONO3D_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_MONO3D_DIR)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from .evaluate_2d import clear_cuda_cache_for_current_process, evaluate_2d_coco_map

from ..datasets.ag_dataset_3d import ActionGenomeDataset3D, collate_fn
from ..models.dino_mono_3d import DinoV3Monocular3D


def corners_to_aabb(corners: np.ndarray) -> np.ndarray:
    """Convert 8x3 corners to axis-aligned box [x1, y1, z1, x2, y2, z2]."""
    if isinstance(corners, torch.Tensor):
        corners = corners.detach().cpu().numpy()
    corners = np.asarray(corners, dtype=np.float64)
    if corners.ndim == 2:
        corners = corners.reshape(1, 8, 3)
    mins = corners.min(axis=1)  # (N, 3)
    maxs = corners.max(axis=1)
    return np.concatenate([mins, maxs], axis=1)  # (N, 6)


def compute_iou_3d_aabb(box1: np.ndarray, box2: np.ndarray) -> float:
    """3D IoU for axis-aligned [x1, y1, z1, x2, y2, z2]."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    z1 = max(float(box1[2]), float(box2[2]))
    x2 = min(float(box1[3]), float(box2[3]))
    y2 = min(float(box1[4]), float(box2[4]))
    z2 = min(float(box1[5]), float(box2[5]))
    inter_vol = max(0.0, x2 - x1) * max(0.0, y2 - y1) * max(0.0, z2 - z1)
    v1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    v2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union_vol = v1 + v2 - inter_vol
    return inter_vol / union_vol if union_vol > 0.0 else 0.0


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


def chamfer_per_box(pred: np.ndarray, gt: np.ndarray) -> float:
    """Chamfer distance for one box: pred (8,3), gt (8,3)."""
    diff = pred[:, None, :] - gt[None, :, :]  # (8, 8, 3)
    dist_sq = (diff ** 2).sum(axis=2)
    min_p2g = dist_sq.min(axis=1).mean()
    min_g2p = dist_sq.min(axis=0).mean()
    return float(min_p2g + min_g2p)


def corner_l2_per_box(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean L2 distance over 8 corners (one-to-one order). pred/gt (8,3)."""
    return float(np.sqrt(((pred - gt) ** 2).sum(axis=1)).mean())


def match_predictions_to_gt_2d(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    pred_boxes_3d: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_boxes_3d: np.ndarray,
    iou_threshold: float = 0.5,
) -> list:
    """
    Match each GT to best overlapping prediction (2D IoU + same label).
    Returns list of (pred_3d, gt_3d, pred_box_2d, gt_box_2d) for matched pairs.
    pred_boxes_3d: (N, 8, 3), gt_boxes_3d: (M, 8, 3).
    """
    out = []
    if gt_boxes.size == 0:
        return out
    used = set()
    for g in range(len(gt_boxes)):
        gl = int(gt_labels[g])
        best_i = -1
        best_iou = 0.0
        for i in range(len(pred_boxes)):
            if i in used or int(pred_labels[i]) != gl:
                continue
            iou = compute_iou_2d(pred_boxes[i], gt_boxes[g])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_i = i
        if best_i >= 0:
            used.add(best_i)
            gt_3d = gt_boxes_3d[g]
            if gt_3d.size == 24:
                gt_3d = gt_3d.reshape(8, 3)
            pred_3d = pred_boxes_3d[best_i]
            if pred_3d.size == 24:
                pred_3d = pred_3d.reshape(8, 3)
            out.append((pred_3d, gt_3d))
    return out


def compute_3d_attribute_errors(pred_corners: np.ndarray, gt_corners: np.ndarray) -> dict:
    """Center L2, dimension L1 (axis-aligned extent), rotation error (degrees, wrapped)."""
    pred_c = pred_corners.mean(axis=0)
    gt_c = gt_corners.mean(axis=0)
    center_l2 = float(np.linalg.norm(pred_c - gt_c))

    pred_mins = pred_corners.min(axis=0)
    pred_maxs = pred_corners.max(axis=0)
    gt_mins = gt_corners.min(axis=0)
    gt_maxs = gt_corners.max(axis=0)
    pred_dims = pred_maxs - pred_mins
    gt_dims = gt_maxs - gt_mins
    dims_l1 = float(np.abs(pred_dims - gt_dims).mean())

    # Rotation: yaw from first edge in xy (fragile but consistent with training)
    def yaw_from_corners(c):
        e = c[1, :2] - c[0, :2]
        return np.arctan2(e[1], e[0])
    r_pred = yaw_from_corners(pred_corners)
    r_gt = yaw_from_corners(gt_corners)
    r_diff = np.arctan2(np.sin(r_pred - r_gt), np.cos(r_pred - r_gt))
    rotation_deg = float(np.rad2deg(np.abs(r_diff)))

    return {"center_l2": center_l2, "dims_l1": dims_l1, "rotation_deg": rotation_deg}


def evaluate_2d_coco(model, dataloader, device, accelerator=None):
    """COCO-style 2D mAP via torchmetrics (self-contained)."""
    return evaluate_2d_coco_map(model, dataloader, device, accelerator=accelerator)


def evaluate_3d_metrics(model, dataloader, device, iou_threshold_2d=0.5):
    """3D metrics on matched pred-GT pairs: Chamfer, corner L2, AABB mAP, attribute errors."""
    model.eval()
    chamfer_list = []
    corner_l2_list = []
    iou_3d_list = []
    center_l2_list = []
    dims_l1_list = []
    rotation_deg_list = []
    n_matched = 0
    n_gt_total = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="3D eval", ascii=True):
            images = torch.stack([img for img in images]).to(device)
            outputs = model(images)

            for i in range(len(images)):
                pred_boxes = outputs[i]["boxes"].detach().cpu().numpy()
                pred_labels = outputs[i]["labels"].detach().cpu().numpy()
                pred_scores = outputs[i]["scores"].detach().cpu().numpy()
                pred_3d = outputs[i]["boxes_3d"].detach().cpu().numpy()  # (K, 8, 3)

                gt_boxes = targets[i]["boxes"].detach().cpu().numpy()
                gt_labels = targets[i]["labels"].detach().cpu().numpy()
                gt_3d = targets[i]["boxes_3d"].detach().cpu().numpy()  # (M, 8, 3)

                n_gt_total += len(gt_boxes)
                valid_gt = (np.abs(gt_3d.reshape(gt_3d.shape[0], -1)).sum(axis=1) > 1e-6)
                gt_boxes = gt_boxes[valid_gt]
                gt_labels = gt_labels[valid_gt]
                gt_3d = gt_3d[valid_gt]
                if len(gt_boxes) == 0:
                    continue

                if pred_3d.shape[0] == 0:
                    continue
                pairs = match_predictions_to_gt_2d(
                    pred_boxes, pred_labels, pred_scores, pred_3d,
                    gt_boxes, gt_labels, gt_3d,
                    iou_threshold=iou_threshold_2d,
                )
                for pred_c, gt_c in pairs:
                    gt_flat = gt_c.reshape(-1)
                    if np.abs(gt_flat).sum() < 1e-6:
                        continue
                    pred_c = np.asarray(pred_c, dtype=np.float64)
                    gt_c = np.asarray(gt_c, dtype=np.float64)
                    chamfer_list.append(chamfer_per_box(pred_c, gt_c))
                    corner_l2_list.append(corner_l2_per_box(pred_c, gt_c))
                    aabb_pred = corners_to_aabb(pred_c.reshape(1, 8, 3))[0]
                    aabb_gt = corners_to_aabb(gt_c.reshape(1, 8, 3))[0]
                    iou_3d_list.append(compute_iou_3d_aabb(aabb_pred, aabb_gt))
                    attrs = compute_3d_attribute_errors(pred_c, gt_c)
                    center_l2_list.append(attrs["center_l2"])
                    dims_l1_list.append(attrs["dims_l1"])
                    rotation_deg_list.append(attrs["rotation_deg"])
                    n_matched += 1
            clear_cuda_cache_for_current_process(sync=False)

    # mAP at 0.5 and 0.75 (over matched pairs: count as TP if IoU >= thresh)
    iou_3d = np.array(iou_3d_list) if iou_3d_list else np.array([0.0])
    ap_50 = (iou_3d >= 0.5).mean()
    ap_75 = (iou_3d >= 0.75).mean()
    mean_iou_3d = float(iou_3d.mean())

    return {
        "n_matched": n_matched,
        "n_gt_3d": n_gt_total,
        "chamfer_mean": float(np.mean(chamfer_list)) if chamfer_list else 0.0,
        "corner_l2_mean": float(np.mean(corner_l2_list)) if corner_l2_list else 0.0,
        "mAP_3d_50": ap_50,
        "mAP_3d_75": ap_75,
        "mean_iou_3d": mean_iou_3d,
        "center_l2_mean": float(np.mean(center_l2_list)) if center_l2_list else 0.0,
        "dims_l1_mean": float(np.mean(dims_l1_list)) if dims_l1_list else 0.0,
        "rotation_deg_mean": float(np.mean(rotation_deg_list)) if rotation_deg_list else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 Monocular 3D detector")
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
    parser.add_argument("--no_2d", action="store_true", help="Skip 2D mAP evaluation")
    parser.add_argument("--no_3d", action="store_true", help="Skip 3D metrics")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Dataset
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

    # Model
    ds = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset
    num_classes = len(ds.object_classes) if hasattr(ds, "object_classes") else 37
    model = DinoV3Monocular3D(num_classes=num_classes, pretrained=False, model="v3l")
    model.to(device)
    model.eval()

    # Load checkpoint
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
    print(f"Loaded checkpoint: {ckpt_path}")

    metrics = {}

    if not args.no_2d:
        print("Evaluating 2D COCO mAP...")
        coco_2d = evaluate_2d_coco(model, test_loader, device, accelerator=None)
        metrics["2d"] = {
            "map": coco_2d.get("map", 0.0),
            "map_50": coco_2d.get("map_50", 0.0),
            "map_75": coco_2d.get("map_75", 0.0),
            "map_per_class": coco_2d.get("map_per_class") is not None,
        }
        print(f"  2D mAP: {metrics['2d']['map']:.4f}  mAP@50: {metrics['2d']['map_50']:.4f}  mAP@75: {metrics['2d']['map_75']:.4f}")

    if not args.no_3d:
        print("Evaluating 3D metrics (matched by 2D IoU)...")
        metrics_3d = evaluate_3d_metrics(model, test_loader, device, iou_threshold_2d=0.5)
        metrics["3d"] = metrics_3d
        print(f"  Matched pairs: {metrics_3d['n_matched']}  GT 3D boxes: {metrics_3d['n_gt_3d']}")
        print(f"  Chamfer (mean): {metrics_3d['chamfer_mean']:.4f}  Corner L2 (mean): {metrics_3d['corner_l2_mean']:.4f}")
        print(f"  mAP_3d@50: {metrics_3d['mAP_3d_50']:.4f}  mAP_3d@75: {metrics_3d['mAP_3d_75']:.4f}  Mean IoU 3D: {metrics_3d['mean_iou_3d']:.4f}")
        print(f"  Center L2: {metrics_3d['center_l2_mean']:.4f}  Dims L1: {metrics_3d['dims_l1_mean']:.4f}  Rotation (deg): {metrics_3d['rotation_deg_mean']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")

    return metrics


if __name__ == "__main__":
    main()


""" 
# Full evaluation (2D + 3D)
python -m lib.detector.monocular3d.evaluation.evaluate_3d \
  --checkpoint /path/to/checkpoint_XX \
  --data_path /path/to/Datasets/action_genome

# Optional: 3D pkl folder, limit test size, save JSON
python -m lib.detector.monocular3d.evaluation.evaluate_3d \
  --checkpoint /path/to/checkpoint_XX \
  --data_path /path/to/Datasets/action_genome \
  --world_3d_annotations_path /path/to/bbox_annotations_3d_obb_camera \
  --max_test_samples 2000 \
  --output results/metrics.json

# Only 2D or only 3D
python -m lib.detector.monocular3d.evaluation.evaluate_3d --checkpoint ... --no_3d
python -m lib.detector.monocular3d.evaluation.evaluate_3d --checkpoint ... --no_2d


--checkpoint can be:
a directory containing checkpoint_state.pth (e.g. path_to_experiment/checkpoint_5), or
the path to a .pth file.
Notes
2D mAP is computed over all test samples with torchmetrics (COCO-style).
3D metrics are computed only on matched pred–GT pairs (2D IoU >= 0.5, same class). If there is no 3D GT or no matches, 3D metrics are zero or N/A.
3D IoU uses axis-aligned boxes (AABB) derived from the 8 corners, so it is not oriented IoU but is stable and easy to interpret.
"""
