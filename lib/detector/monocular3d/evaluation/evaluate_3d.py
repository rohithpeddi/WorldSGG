#!/usr/bin/env python3
"""
Evaluation script for DINOv2 Monocular 3D detector.

Reports:
  - 2D: COCO-style mAP (map, map_50, map_75, map_per_class).
  - 3D: Chamfer distance, corner L2 error, oriented 3D IoU (iou3d_hit_50, iou3d_hit_75),
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

from .evaluate_2d import compute_iou_2d, evaluate_2d_coco_map
from ..utils.cuda_utils import clear_cuda_cache_for_current_process

from ..datasets.ag_dataset_3d import ActionGenomeDataset3D, collate_fn
from ..models.dino_mono_3d import DinoV3Monocular3D
from ..models.resnet_mono_3d import ResNetMonocular3D


def _polygon_clip(subject: list, clip: list) -> list:
    """Sutherland-Hodgman polygon clipping. subject and clip are lists of (x, y) tuples."""
    def _inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def _intersect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = a;  x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = list(subject)
    for i in range(len(clip)):
        if len(output) == 0:
            return []
        a, b = clip[i - 1], clip[i]
        inp = list(output)
        output = []
        for j in range(len(inp)):
            p_cur = inp[j]
            p_prev = inp[j - 1]
            if _inside(p_cur, a, b):
                if not _inside(p_prev, a, b):
                    output.append(_intersect(p_prev, p_cur, a, b))
                output.append(p_cur)
            elif _inside(p_prev, a, b):
                output.append(_intersect(p_prev, p_cur, a, b))
    return output


def _polygon_area(poly: list) -> float:
    """Shoelace formula for area of a polygon given as list of (x,y)."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def _convex_hull_2d(points: list) -> list:
    """Andrew's monotone chain convex hull for 2D points."""
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _corners_to_bottom_face(corners: np.ndarray) -> list:
    """Extract the bottom 4 corners (lowest z) as convex hull for OBB IoU in XY plane."""
    corners = corners.reshape(8, 3)
    z_vals = corners[:, 2]
    z_mid = (z_vals.max() + z_vals.min()) / 2.0
    bottom = corners[z_vals <= z_mid]
    pts = [(float(p[0]), float(p[1])) for p in bottom]
    if len(pts) < 3:
        pts = [(float(p[0]), float(p[1])) for p in corners[:4]]
    return _convex_hull_2d(pts)


def compute_iou_3d_obb(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """Oriented 3D IoU between two sets of 8 corners (8, 3).
    Projects to XY plane for polygon intersection, then multiplies by Z overlap."""
    corners1 = np.asarray(corners1, dtype=np.float64).reshape(8, 3)
    corners2 = np.asarray(corners2, dtype=np.float64).reshape(8, 3)

    # Z overlap
    z_min1, z_max1 = corners1[:, 2].min(), corners1[:, 2].max()
    z_min2, z_max2 = corners2[:, 2].min(), corners2[:, 2].max()
    z_overlap = max(0.0, min(z_max1, z_max2) - max(z_min1, z_min2))
    if z_overlap <= 0:
        return 0.0

    # XY polygon intersection
    poly1 = _corners_to_bottom_face(corners1)
    poly2 = _corners_to_bottom_face(corners2)
    if len(poly1) < 3 or len(poly2) < 3:
        return 0.0

    inter_poly = _polygon_clip(poly1, poly2)
    inter_area = _polygon_area(inter_poly)

    area1 = _polygon_area(poly1)
    area2 = _polygon_area(poly2)

    inter_vol = inter_area * z_overlap
    h1 = z_max1 - z_min1
    h2 = z_max2 - z_min2
    vol1 = area1 * h1
    vol2 = area2 * h2
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


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


def evaluate_3d_metrics(model, dataloader, device, iou_threshold_2d=0.5):
    """3D metrics on matched pred-GT pairs: Chamfer, corner L2, OBB IoU hit rates, attribute errors."""
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

                valid_gt = (np.abs(gt_3d.reshape(gt_3d.shape[0], -1)).sum(axis=1) > 1e-6)
                gt_boxes = gt_boxes[valid_gt]
                gt_labels = gt_labels[valid_gt]
                gt_3d = gt_3d[valid_gt]
                n_gt_total += len(gt_boxes)  # count only GT with valid 3D annotations
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
                    iou_3d_list.append(compute_iou_3d_obb(pred_c.reshape(8, 3), gt_c.reshape(8, 3)))
                    attrs = compute_3d_attribute_errors(pred_c, gt_c)
                    center_l2_list.append(attrs["center_l2"])
                    dims_l1_list.append(attrs["dims_l1"])
                    rotation_deg_list.append(attrs["rotation_deg"])
                    n_matched += 1
            clear_cuda_cache_for_current_process(sync=False)

    # Hit rate at IoU thresholds (fraction of matched pairs with IoU >= thresh)
    iou_3d = np.array(iou_3d_list) if iou_3d_list else np.array([0.0])
    hit_50 = (iou_3d >= 0.5).mean()
    hit_75 = (iou_3d >= 0.75).mean()
    mean_iou_3d = float(iou_3d.mean())

    return {
        "n_matched": n_matched,
        "n_gt_3d": n_gt_total,
        "chamfer_mean": float(np.mean(chamfer_list)) if chamfer_list else 0.0,
        "corner_l2_mean": float(np.mean(corner_l2_list)) if corner_l2_list else 0.0,
        "iou3d_hit_50": hit_50,   # hit rate, not mAP
        "iou3d_hit_75": hit_75,   # hit rate, not mAP
        "mean_iou_3d": mean_iou_3d,
        "center_l2_mean": float(np.mean(center_l2_list)) if center_l2_list else 0.0,
        "dims_l1_mean": float(np.mean(dims_l1_list)) if dims_l1_list else 0.0,
        "rotation_deg_mean": float(np.mean(rotation_deg_list)) if rotation_deg_list else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Monocular 3D detector (DINOv2/v3 or ResNet50)")
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
    parser.add_argument("--model", type=str, default="v3l",
                        help="Model variant: v2, v2l, v3l (used for DINOv2/v3 backbones)")
    parser.add_argument("--backbone", type=str, default="dino_v3",
                        choices=["dino_v2", "dino_v3", "resnet50"],
                        help="Backbone architecture")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Dataset
    use_patch = (args.backbone != "resnet50")
    kwargs = {"phase": "test", "target_size": args.target_size, "use_patch_alignment": use_patch}
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
    if args.backbone == "resnet50":
        model = ResNetMonocular3D(num_classes=num_classes, pretrained=False)
    else:
        model = DinoV3Monocular3D(num_classes=num_classes, pretrained=False, model=args.model)
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
    print(f"  Backbone: {args.backbone}  |  Model: {args.model}  |  Classes: {num_classes}")

    metrics = {}

    if not args.no_2d:
        print("Evaluating 2D COCO mAP...")
        coco_2d = evaluate_2d_coco_map(model, test_loader, device, accelerator=None)
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
        print(f"  IoU3D Hit@50: {metrics_3d['iou3d_hit_50']:.4f}  IoU3D Hit@75: {metrics_3d['iou3d_hit_75']:.4f}  Mean IoU 3D: {metrics_3d['mean_iou_3d']:.4f}")
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
3D IoU uses oriented bounding box (OBB) intersection via polygon clipping in the XY plane with Z-axis overlap.
"""
