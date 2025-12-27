import gc
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


def clear_cuda_cache_for_current_process(sync: bool = True) -> None:
    """
    Clear CUDA cache for all visible devices in this process.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    if sync:
        torch.cuda.synchronize()
    for dev in range(torch.cuda.device_count()):
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()


class DetectionEvaluator:
    """
    Unified evaluator for 2D and 3D bounding box detectors.

    COCO-style metrics:

    2D:
      - mAP over IoU thresholds 0.5:0.95:0.05 (torchmetrics)
      - map, map_50, map_75, map_per_class, etc.

    3D (axis-aligned [x1, y1, z1, x2, y2, z2]):
      - mAP over IoU thresholds 0.5:0.95:0.05, COCO-style:
        * For each IoU threshold T:
            - AP per class (VOC interpolation)
            - mAP(T) = mean over classes
        * Overall mAP = mean over IoU thresholds of mAP(T)
        * Also expose map_3d_50, map_3d_75, per-IoU and per-class APs.

    Additionally:
      - 2D and 3D simple Precision / Recall at fixed IoU thresholds
        (convenient diagnostic but not "COCO official" metrics).
    """

    def __init__(
        self,
        device: torch.device,
        accelerator: Optional[Any] = None,
        iou_threshold_2d: float = 0.5,
        iou_threshold_3d: float = 0.5,
        frame_batch_size: int = 10,
        # If None, torchmetrics uses COCO-style [0.5:0.95:0.05] for 2D
        iou_thresholds_2d: Optional[List[float]] = None,
        # For 3D, we mimic COCO-style IoU thresholds 0.5:0.95:0.05
        iou_thresholds_3d: Optional[List[float]] = None,
    ) -> None:
        self.device = device
        self.accelerator = accelerator
        self.iou_threshold_2d = iou_threshold_2d
        self.iou_threshold_3d = iou_threshold_3d
        self.frame_batch_size = frame_batch_size

        self.iou_thresholds_2d = iou_thresholds_2d  # None => COCO default in torchmetrics
        if iou_thresholds_3d is None:
            self.iou_thresholds_3d = [float(x) for x in np.arange(0.5, 0.96, 0.05)]
        else:
            self.iou_thresholds_3d = iou_thresholds_3d

        # 2D COCO-style mAP metric
        self.map_2d = MeanAveragePrecision(
            iou_type="bbox",
            sync_on_compute=False,
            iou_thresholds=self.iou_thresholds_2d,
        )

    # ------------------------------------------------------------------
    # IoU utilities
    # ------------------------------------------------------------------
    @staticmethod
    def compute_iou_2d(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        2D IoU for [x1, y1, x2, y2].
        """
        if not isinstance(box1, np.ndarray):
            box1 = box1.detach().cpu().numpy()
        if not isinstance(box2, np.ndarray):
            box2 = box2.detach().cpu().numpy()

        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        b1_area = max(0.0, float(box1[2]) - float(box1[0])) * max(
            0.0, float(box1[3]) - float(box1[1])
        )
        b2_area = max(0.0, float(box2[2]) - float(box2[0])) * max(
            0.0, float(box2[3]) - float(box2[1])
        )

        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area > 0.0 else 0.0

    @staticmethod
    def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        3D IoU for axis-aligned boxes [x1, y1, z1, x2, y2, z2].
        """
        if not isinstance(box1, np.ndarray):
            box1 = box1.detach().cpu().numpy()
        if not isinstance(box2, np.ndarray):
            box2 = box2.detach().cpu().numpy()

        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        z1 = max(float(box1[2]), float(box2[2]))

        x2 = min(float(box1[3]), float(box2[3]))
        y2 = min(float(box1[4]), float(box2[4]))
        z2 = min(float(box1[5]), float(box2[5]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_d = max(0.0, z2 - z1)
        inter_vol = inter_w * inter_h * inter_d

        b1_vol = (
            max(0.0, float(box1[3]) - float(box1[0]))
            * max(0.0, float(box1[4]) - float(box1[1]))
            * max(0.0, float(box1[5]) - float(box1[2]))
        )
        b2_vol = (
            max(0.0, float(box2[3]) - float(box2[0]))
            * max(0.0, float(box2[4]) - float(box2[1]))
            * max(0.0, float(box2[5]) - float(box2[2]))
        )

        union_vol = b1_vol + b2_vol - inter_vol
        return inter_vol / union_vol if union_vol > 0.0 else 0.0

    # ------------------------------------------------------------------
    # Internal helper for matching (used by both 2D & 3D PR)
    # ------------------------------------------------------------------
    @staticmethod
    def _match_predictions_to_gt(
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        iou_func,
        iou_threshold: float,
    ) -> Tuple[int, int, int]:
        """
        Returns (TP, FP, FN) for a single image.
        """

        pred_boxes = pred_boxes.detach().cpu()
        pred_labels = pred_labels.detach().cpu()
        pred_scores = pred_scores.detach().cpu()

        gt_boxes = gt_boxes.detach().cpu()
        gt_labels = gt_labels.detach().cpu()

        if pred_scores.numel() > 0:
            sorted_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_idx]
            pred_labels = pred_labels[sorted_idx]
            pred_scores = pred_scores[sorted_idx]

        matched_gt = set()
        TP = 0
        FP = 0

        for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
            if gt_boxes.numel() == 0:
                FP += 1
                continue

            ious = [iou_func(pb, gb) for gb in gt_boxes]
            max_iou_idx = int(np.argmax(ious)) if len(ious) > 0 else -1
            max_iou = ious[max_iou_idx] if max_iou_idx >= 0 else 0.0

            if (
                max_iou_idx >= 0
                and max_iou >= iou_threshold
                and pl.item() == gt_labels[max_iou_idx].item()
            ):
                if max_iou_idx not in matched_gt:
                    TP += 1
                    matched_gt.add(max_iou_idx)
                else:
                    FP += 1
            else:
                FP += 1

        FN = len(gt_boxes) - len(matched_gt)
        return TP, FP, FN

    # ------------------------------------------------------------------
    # Simple 2D Precision / Recall at fixed IoU (non-COCO diagnostic)
    # ------------------------------------------------------------------
    def evaluate_2d_precision_recall(
        self,
        model: torch.nn.Module,
        dataloader,
        num_batches: int,
    ) -> Tuple[float, float]:
        """
        Simple average precision and recall over batches for 2D boxes at fixed IoU.
        """
        model.eval()
        total_precision, total_recall = 0.0, 0.0
        batch_count = 0

        with torch.no_grad():
            for images, gt in tqdm(dataloader, ascii=True):
                images = torch.stack([img for img in images]).to(self.device)
                pred, _ = model(images, targets=None, mode="test")

                TP_total, FP_total, FN_total = 0, 0, 0

                for i in range(len(pred["boxes"])):
                    pred_boxes = pred["boxes"][i]
                    pred_labels = pred["labels"][i]
                    pred_scores = pred["scores"][i]

                    gt_boxes = gt[i]["boxes"]
                    gt_labels = gt[i]["labels"]

                    TP, FP, FN = self._match_predictions_to_gt(
                        pred_boxes,
                        pred_labels,
                        pred_scores,
                        gt_boxes,
                        gt_labels,
                        self.compute_iou_2d,
                        self.iou_threshold_2d,
                    )
                    TP_total += TP
                    FP_total += FP
                    FN_total += FN

                precision = TP_total / (TP_total + FP_total + 1e-7)
                recall = TP_total / (TP_total + FN_total + 1e-7)

                total_precision += precision
                total_recall += recall
                batch_count += 1

        denom = batch_count if batch_count > 0 else num_batches
        return total_precision / denom, total_recall / denom

    # ------------------------------------------------------------------
    # COCO-style 2D mAP via torchmetrics
    # ------------------------------------------------------------------
    def evaluate_2d_map_coco(
        self,
        model: torch.nn.Module,
        dataloader,
    ) -> Dict[str, Any]:
        """
        COCO-style 2D mAP using torchmetrics.MeanAveragePrecision.

        Returns:
          {
            "raw": torchmetrics output dict,
            "map": float,        # averaged over IoU 0.5:0.95
            "map_50": float,
            "map_75": float,
            "map_per_class": np.ndarray or None,
          }
        """
        model.eval()
        self.map_2d = MeanAveragePrecision(
            iou_type="bbox",
            sync_on_compute=False,
            iou_thresholds=self.iou_thresholds_2d,  # None => COCO 0.5:0.95:0.05
        )

        frame_batch_size = self.frame_batch_size

        with torch.no_grad():
            for images, targets in tqdm(dataloader, ascii=True):
                total_frames = len(images)

                for start_idx in range(0, total_frames, frame_batch_size):
                    end_idx = min(start_idx + frame_batch_size, total_frames)
                    batch_frames = images[start_idx:end_idx]

                    batch_images = torch.stack(batch_frames).to(self.device)
                    outputs = model(batch_images)

                    preds, gts = [], []

                    for output in outputs:
                        preds.append(
                            {
                                "boxes": output["boxes"].detach().cpu(),
                                "scores": output["scores"].detach().cpu(),
                                "labels": output["labels"].detach().cpu(),
                            }
                        )

                    for i in range(start_idx, end_idx):
                        gts.append(
                            {
                                "boxes": targets[i]["boxes"].detach().cpu(),
                                "labels": targets[i]["labels"].detach().cpu(),
                            }
                        )

                    min_len = min(len(preds), len(gts))
                    if min_len > 0:
                        self.map_2d.update(preds[:min_len], gts[:min_len])

                    clear_cuda_cache_for_current_process(sync=False)

        if self.accelerator is not None and not self.accelerator.is_main_process:
            return {}

        raw = self.map_2d.compute()

        out: Dict[str, Any] = {"raw": raw}
        out["map"] = float(raw["map"].item())
        out["map_50"] = float(raw["map_50"].item())
        out["map_75"] = float(raw["map_75"].item())
        # map_per_class may be None if class-wise aggregation is disabled
        out["map_per_class"] = (
            raw.get("map_per_class", None).cpu().numpy()
            if raw.get("map_per_class", None) is not None
            else None
        )
        return out

    # ------------------------------------------------------------------
    # Simple 3D Precision / Recall at fixed IoU (non-COCO diagnostic)
    # ------------------------------------------------------------------
    def evaluate_3d_precision_recall(
        self,
        model: torch.nn.Module,
        dataloader,
        num_batches: int,
    ) -> Tuple[float, float]:
        """
        Simple average precision and recall over batches for 3D boxes at fixed IoU.
        """
        model.eval()
        total_precision, total_recall = 0.0, 0.0
        batch_count = 0

        with torch.no_grad():
            for images, gt in tqdm(dataloader, ascii=True):
                images = torch.stack([img for img in images]).to(self.device)
                pred, _ = model(images, targets=None, mode="test")

                TP_total, FP_total, FN_total = 0, 0, 0

                for i in range(len(pred["boxes_3d"])):
                    pred_boxes_3d = pred["boxes_3d"][i]
                    pred_labels = pred["labels"][i]
                    pred_scores = pred["scores"][i]

                    gt_boxes_3d = gt[i]["boxes_3d"]
                    gt_labels = gt[i]["labels"]

                    TP, FP, FN = self._match_predictions_to_gt(
                        pred_boxes_3d,
                        pred_labels,
                        pred_scores,
                        gt_boxes_3d,
                        gt_labels,
                        self.compute_iou_3d,
                        self.iou_threshold_3d,
                    )
                    TP_total += TP
                    FP_total += FP
                    FN_total += FN

                precision = TP_total / (TP_total + FP_total + 1e-7)
                recall = TP_total / (TP_total + FN_total + 1e-7)

                total_precision += precision
                total_recall += recall
                batch_count += 1

        denom = batch_count if batch_count > 0 else num_batches
        return total_precision / denom, total_recall / denom

    # ------------------------------------------------------------------
    # Helper: AP computation (VOC-style) for 3D mAP
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Compute AP as area under precision-recall curve (VOC 2010-style).
        """
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)

    # ------------------------------------------------------------------
    # COCO-style 3D mAP over IoU thresholds 0.5:0.95
    # ------------------------------------------------------------------
    def evaluate_3d_map_coco(
        self,
        model: torch.nn.Module,
        dataloader,
    ) -> Dict[str, Any]:
        """
        COCO-style 3D mAP using axis-aligned 3D IoU.

        Returns:
          {
            "map_3d": float,              # mean over IoUs and classes
            "map_3d_50": float,
            "map_3d_75": float,
            "map_3d_per_iou": {iou: float},
            "ap_3d_per_class_per_iou": {iou: {class_id: ap}},
          }
        """
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return {}

        model.eval()

        all_preds: List[Dict[str, Any]] = []
        all_gts: List[Dict[str, Any]] = []
        image_id_counter = 0

        with torch.no_grad():
            for images, targets in tqdm(dataloader, ascii=True):
                images_tensor = torch.stack([img for img in images]).to(self.device)
                pred, _ = model(images_tensor, targets=None, mode="test")

                batch_size = len(images)
                for i in range(batch_size):
                    all_preds.append(
                        {
                            "image_id": image_id_counter,
                            "boxes_3d": pred["boxes_3d"][i].detach().cpu(),
                            "labels": pred["labels"][i].detach().cpu(),
                            "scores": pred["scores"][i].detach().cpu(),
                        }
                    )
                    all_gts.append(
                        {
                            "image_id": image_id_counter,
                            "boxes_3d": targets[i]["boxes_3d"].detach().cpu(),
                            "labels": targets[i]["labels"].detach().cpu(),
                        }
                    )
                    image_id_counter += 1

                clear_cuda_cache_for_current_process(sync=False)

        # collect all classes from GT
        all_labels = set()
        for gt in all_gts:
            if gt["labels"].numel() > 0:
                all_labels.update(gt["labels"].tolist())

        map_3d_per_iou: Dict[float, float] = {}
        ap_3d_per_class_per_iou: Dict[float, Dict[int, float]] = {}

        for thr in self.iou_thresholds_3d:
            ap_per_class: Dict[int, float] = {}
            iou_thr = thr

            for cls in sorted(all_labels):
                # GT structures
                gt_boxes_per_img: Dict[int, Dict[str, Any]] = {}
                npos = 0

                for gt in all_gts:
                    img_id = gt["image_id"]
                    labels = gt["labels"]
                    boxes = gt["boxes_3d"]

                    cls_mask = labels == cls
                    if cls_mask.sum() == 0:
                        continue

                    cls_boxes = boxes[cls_mask].numpy()
                    gt_boxes_per_img[img_id] = {
                        "boxes": cls_boxes,
                        "detected": [False] * cls_boxes.shape[0],
                    }
                    npos += cls_boxes.shape[0]

                if npos == 0:
                    # no GT for this class at all
                    continue

                # Predictions flattened
                pred_list: List[Tuple[int, float, np.ndarray]] = []
                for pred in all_preds:
                    img_id = pred["image_id"]
                    labels = pred["labels"]
                    boxes = pred["boxes_3d"]
                    scores = pred["scores"]

                    cls_mask = labels == cls
                    if cls_mask.sum() == 0:
                        continue

                    for box, score in zip(boxes[cls_mask], scores[cls_mask]):
                        pred_list.append(
                            (img_id, float(score.item()), box.detach().cpu().numpy())
                        )

                if len(pred_list) == 0:
                    # no predictions for this class
                    ap_per_class[int(cls)] = 0.0
                    continue

                pred_list.sort(key=lambda x: x[1], reverse=True)

                tp = np.zeros(len(pred_list), dtype=np.float64)
                fp = np.zeros(len(pred_list), dtype=np.float64)

                for k, (img_id, score, box) in enumerate(pred_list):
                    if img_id not in gt_boxes_per_img:
                        fp[k] = 1.0
                        continue

                    gt_info = gt_boxes_per_img[img_id]
                    gt_boxes_img = gt_info["boxes"]
                    detected = gt_info["detected"]

                    if len(gt_boxes_img) == 0:
                        fp[k] = 1.0
                        continue

                    ious = [self.compute_iou_3d(box, gt_box) for gt_box in gt_boxes_img]
                    max_iou_idx = int(np.argmax(ious))
                    max_iou = ious[max_iou_idx]

                    if max_iou >= iou_thr and not detected[max_iou_idx]:
                        tp[k] = 1.0
                        detected[max_iou_idx] = True
                    else:
                        fp[k] = 1.0

                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / float(npos)
                precision = tp_cum / np.maximum(
                    tp_cum + fp_cum, np.finfo(np.float64).eps
                )

                ap_per_class[int(cls)] = self._compute_ap(recall, precision)

            if len(ap_per_class) == 0:
                map_3d_per_iou[thr] = 0.0
            else:
                map_3d_per_iou[thr] = float(np.mean(list(ap_per_class.values())))

            ap_3d_per_class_per_iou[thr] = ap_per_class

        if len(map_3d_per_iou) == 0:
            map_3d = 0.0
        else:
            map_3d = float(np.mean(list(map_3d_per_iou.values())))

        map_3d_50 = map_3d_per_iou.get(0.5, 0.0)
        map_3d_75 = map_3d_per_iou.get(0.75, 0.0)

        return {
            "map_3d": map_3d,
            "map_3d_50": map_3d_50,
            "map_3d_75": map_3d_75,
            "map_3d_per_iou": map_3d_per_iou,
            "ap_3d_per_class_per_iou": ap_3d_per_class_per_iou,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers: everything COCO-style
    # ------------------------------------------------------------------
    def evaluate_all_2d(
        self,
        model: torch.nn.Module,
        dataloader_for_pr,
        dataloader_for_map,
        num_batches_for_pr: int,
    ) -> Dict[str, Any]:
        """
        2D summary including:
          - simple precision/recall at fixed IoU (diagnostic)
          - COCO-style mAP (map, map_50, map_75, map_per_class, raw)
        """
        precision_2d, recall_2d = self.evaluate_2d_precision_recall(
            model, dataloader_for_pr, num_batches_for_pr
        )
        coco_2d = self.evaluate_2d_map_coco(model, dataloader_for_map)

        return {
            "precision_2d_fixed": precision_2d,
            "recall_2d_fixed": recall_2d,
            "coco_2d": coco_2d,
        }

    def evaluate_all_3d(
        self,
        model: torch.nn.Module,
        dataloader_for_pr_3d,
        num_batches_for_pr_3d: int,
        dataloader_for_map_3d=None,
    ) -> Dict[str, Any]:
        """
        3D summary including:
          - simple precision/recall at fixed IoU (diagnostic)
          - COCO-style mAP over IoU 0.5:0.95
        """
        precision_3d, recall_3d = self.evaluate_3d_precision_recall(
            model, dataloader_for_pr_3d, num_batches_for_pr_3d
        )

        if dataloader_for_map_3d is None:
            dataloader_for_map_3d = dataloader_for_pr_3d

        coco_3d = self.evaluate_3d_map_coco(model, dataloader_for_map_3d)

        return {
            "precision_3d_fixed": precision_3d,
            "recall_3d_fixed": recall_3d,
            "coco_3d": coco_3d,
        }
