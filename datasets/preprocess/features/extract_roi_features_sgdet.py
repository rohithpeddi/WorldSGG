#!/usr/bin/env python3
"""
ROI Feature Extraction for SGDet
====================================

Extracts pre-computed ROI features using the DinoV3 detector's own detections
(not GT bboxes). Mirrors the original Faster R-CNN Detector's sgdet pipeline:

  1. Run DinoV3 detector → per-frame detections (boxes, labels, scores)
  2. Apply score threshold + keep single highest-scoring person per frame
  3. Match detections to GT via IoU (train: 0.5, test: 0.3)
  4. [Train only] Augment with unfound GT boxes (SUPPLY_RELATIONS)
  5. Extract ROI features + union features for finalized bounding boxes
  6. Save per-video PKL files

Usage:
    python datasets/preprocess/features/extract_roi_features_sgdet.py \
        --config configs/features/sgdet/ex_roi_feat_v1_dinov2b_saurabh.yaml

    # Override via CLI:
    python datasets/preprocess/features/extract_roi_features_sgdet.py \
        --config configs/features/sgdet/ex_roi_feat_v1_dinov2b_saurabh.yaml \
        --split train --video 001YG.mp4 --overwrite true
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.preprocess.features.extract_roi_features_base import (
    BaseROIFeatureExtractor,
    ExtractConfig,
    CATID_TO_NAME,
    DATASET_CLASSNAMES,
    LABEL_TO_CLASSIDX,
    MODEL_TO_DIR,
    _normalize_label,
    build_parser,
    logger,
    parse_config,
    setup_logging,
)

from lib.supervised.funcs import assign_relations


# ---------------------------------------------------------------------------
# SGDet ROI Feature Extractor
# ---------------------------------------------------------------------------

class SGDetROIFeatureExtractor(BaseROIFeatureExtractor):
    """
    SGDet: runs DinoV3 detector → NMS → GT matching → GT augmentation (train).

    For training: detections are matched to GT at IoU >= 0.5.
                  Unfound GT objects are injected with score=1.0 and true labels.
    For testing:  detections are matched to GT at IoU >= 0.3. No augmentation.
    """

    MODE_NAME = "sgdet"

    def _get_output_dir(self, split_name: str) -> Path:
        """SGDet uses separate train/test output directories."""
        model_dir = MODEL_TO_DIR.get(self.cfg.model, self.cfg.model)
        if self.cfg.output_dir:
            return Path(self.cfg.output_dir) / split_name
        return Path(self.cfg.data_path) / "features" / "roi_features" / "sgdet" / model_dir / split_name

    # ------------------------------------------------------------------
    # Detection + filtering
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _detect_frame(
        self,
        img_chw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the DinoV3 detector on a preprocessed frame.

        The model's built-in postprocess_detections applies:
          - Per-class NMS (default IoU 0.5)
          - Score-based top-k (default 100 detections/image)

        Returns:
            boxes: (N, 4) xyxy in resized image coordinates
            labels: (N,) class indices (1-indexed, 0=background)
            scores: (N,) confidence scores
            boxes_3d: (N, 8, 3) predicted 3D corners
        """
        detections = self.model([img_chw.to(self.device)])
        det = detections[0]
        return det["boxes"], det["labels"], det["scores"], det["boxes_3d"]

    def _filter_detections(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes_3d: torch.Tensor,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[List[List[float]], List[str], List[int], List[float], np.ndarray]:
        """
        Filter detections:
          1. Score threshold (default 0.1)
          2. Person class: keep only the single highest-scoring detection
          3. Convert boxes back to original image coordinates

        Returns:
            bboxes_xyxy_orig: bboxes in original image coords
            det_labels: normalized label strings
            det_label_ids: AG class indices
            det_scores: confidence scores
            det_boxes_3d: (N, 8, 3) filtered 3D predictions as numpy
        """
        if len(boxes) == 0:
            return [], [], [], [], np.empty((0, 8, 3), dtype=np.float32)

        # Score threshold
        keep_mask = scores >= self.cfg.score_threshold
        boxes = boxes[keep_mask]
        labels = labels[keep_mask]
        scores = scores[keep_mask]
        boxes_3d = boxes_3d[keep_mask]

        if len(boxes) == 0:
            return [], [], [], [], np.empty((0, 8, 3), dtype=np.float32)

        # Person class (index 1): keep only top-1
        person_mask = labels == 1
        non_person_mask = ~person_mask

        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []
        filtered_boxes_3d = []

        if person_mask.any():
            person_scores = scores[person_mask]
            best_person_idx = person_scores.argmax()
            person_boxes_all = boxes[person_mask]
            filtered_boxes.append(person_boxes_all[best_person_idx].unsqueeze(0))
            filtered_labels.append(labels[person_mask][best_person_idx].unsqueeze(0))
            filtered_scores.append(person_scores[best_person_idx].unsqueeze(0))
            filtered_boxes_3d.append(boxes_3d[person_mask][best_person_idx].unsqueeze(0))

        if non_person_mask.any():
            filtered_boxes.append(boxes[non_person_mask])
            filtered_labels.append(labels[non_person_mask])
            filtered_scores.append(scores[non_person_mask])
            filtered_boxes_3d.append(boxes_3d[non_person_mask])

        if not filtered_boxes:
            return [], [], [], [], np.empty((0, 8, 3), dtype=np.float32)

        boxes = torch.cat(filtered_boxes, dim=0)
        labels = torch.cat(filtered_labels, dim=0)
        scores = torch.cat(filtered_scores, dim=0)
        boxes_3d = torch.cat(filtered_boxes_3d, dim=0)

        # Convert to original image coordinates
        boxes_np = boxes.cpu().numpy()
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        boxes_3d_np = boxes_3d.cpu().numpy()  # (N, 8, 3)

        bboxes_xyxy_orig = []
        det_labels = []
        det_label_ids = []
        det_scores = []

        for i in range(len(boxes_np)):
            x1, y1, x2, y2 = boxes_np[i]
            # Convert from resized coords to original coords
            bboxes_xyxy_orig.append([
                float(x1 / scale_x),
                float(y1 / scale_y),
                float(x2 / scale_x),
                float(y2 / scale_y),
            ])
            cls_idx = int(labels_np[i])
            cls_name = CATID_TO_NAME.get(cls_idx, None)
            if cls_name is not None:
                det_labels.append(_normalize_label(cls_name))
            else:
                det_labels.append(f"class_{cls_idx}")
            det_label_ids.append(cls_idx)
            det_scores.append(float(scores_np[i]))

        return bboxes_xyxy_orig, det_labels, det_label_ids, det_scores, boxes_3d_np

    # ------------------------------------------------------------------
    # GT augmentation (training only)
    # ------------------------------------------------------------------

    def _augment_with_unfound_gt(
        self,
        supply_relations: List[Dict[str, Any]],
        features: Dict[str, torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        scale_x: float,
        scale_y: float,
        target_w: int,
        target_h: int,
    ) -> Tuple[List[List[float]], List[str], List[int], List[float], torch.Tensor, np.ndarray]:
        """
        Extract features and 3D predictions for unfound GT objects (SUPPLY_RELATIONS).

        Args:
            supply_relations: list of GT annotation dicts for unfound objects
            features: FPN feature maps from backbone (already computed)
            image_sizes: from transform
            scale_x, scale_y: bbox scale factors
            target_w, target_h: resized image dims

        Returns:
            supply_bboxes_orig: bboxes in original coords
            supply_labels: normalized label strings
            supply_label_ids: AG class indices
            supply_scores: all 1.0
            supply_features: (M, 1024) tensor on device
            supply_boxes_3d: (M, 8, 3) 3D predictions as numpy
        """
        supply_bboxes_orig = []
        supply_labels = []
        supply_label_ids = []
        supply_scores = []
        supply_scaled_bboxes = []

        for gt_item in supply_relations:
            if "person_bbox" in gt_item:
                # Person bbox
                pb = gt_item["person_bbox"]
                if isinstance(pb, np.ndarray):
                    pb = pb.reshape(-1).tolist()
                elif isinstance(pb, (list, tuple)):
                    if isinstance(pb[0], (list, tuple, np.ndarray)):
                        pb = list(pb[0])
                bbox_orig = [float(v) for v in pb[:4]]
                label_str = "person"
                label_id = 1
            elif "bbox" in gt_item and "class" in gt_item:
                bbox = gt_item["bbox"]
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                bbox_orig = [float(v) for v in bbox[:4]]
                cls_idx = int(gt_item["class"])
                cls_name = CATID_TO_NAME.get(cls_idx, None)
                label_str = _normalize_label(cls_name) if cls_name else f"class_{cls_idx}"
                label_id = cls_idx
            else:
                continue

            # Scale to resized coords
            x1 = max(0.0, min(bbox_orig[0] * scale_x, target_w - 1))
            y1 = max(0.0, min(bbox_orig[1] * scale_y, target_h - 1))
            x2 = max(0.0, min(bbox_orig[2] * scale_x, target_w - 1))
            y2 = max(0.0, min(bbox_orig[3] * scale_y, target_h - 1))

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            supply_bboxes_orig.append(bbox_orig)
            supply_labels.append(label_str)
            supply_label_ids.append(label_id)
            supply_scores.append(1.0)
            supply_scaled_bboxes.append([x1, y1, x2, y2])

        if not supply_scaled_bboxes:
            empty_3d = np.empty((0, 8, 3), dtype=np.float32)
            return [], [], [], [], torch.tensor([]).to(self.device), empty_3d

        # Extract ROI features for supply bboxes using pre-computed FPN features
        supply_features = self._extract_roi_features_for_bboxes(
            features, image_sizes, supply_scaled_bboxes
        )

        # Compute 3D predictions for supply bboxes
        supply_boxes_3d = self._predict_3d_for_bboxes(
            supply_features, supply_scaled_bboxes, image_sizes
        ).cpu().numpy()

        return supply_bboxes_orig, supply_labels, supply_label_ids, supply_scores, supply_features, supply_boxes_3d

    # ------------------------------------------------------------------
    # Build prediction dict for assign_relations
    # ------------------------------------------------------------------

    def _build_prediction_for_assign(
        self,
        all_frame_bboxes: List[List[List[float]]],
        all_frame_labels: List[List[int]],
    ) -> Dict[str, Any]:
        """
        Build the prediction dict expected by assign_relations().

        assign_relations expects:
            prediction["FINAL_BBOXES"]: (N, 5) tensor with [frame_idx, x1, y1, x2, y2]
            prediction["FINAL_LABELS"]: (N,) tensor with class indices

        Args:
            all_frame_bboxes: list of per-frame bbox lists (original coords)
            all_frame_labels: list of per-frame label_id lists

        Returns:
            prediction dict with FINAL_BBOXES and FINAL_LABELS tensors
        """
        all_bboxes = []
        all_labels = []

        for frame_idx, (bboxes, label_ids) in enumerate(zip(all_frame_bboxes, all_frame_labels)):
            for bbox, label_id in zip(bboxes, label_ids):
                all_bboxes.append([frame_idx] + bbox)
                all_labels.append(label_id)

        if not all_bboxes:
            return {
                "FINAL_BBOXES": torch.zeros((0, 5)),
                "FINAL_LABELS": torch.zeros(0, dtype=torch.int64),
            }

        return {
            "FINAL_BBOXES": torch.tensor(all_bboxes, dtype=torch.float32),
            "FINAL_LABELS": torch.tensor(all_labels, dtype=torch.int64),
        }

    # ------------------------------------------------------------------
    # Main extraction logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_features_for_video(
        self,
        video_id: str,
        gt_annotations: List[List[Dict[str, Any]]],
        frame_names: List[str],
        split_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        SGDet extraction pipeline:
          1. Detect objects in each frame
          2. Filter (score threshold + person top-1)
          3. Match to GT via assign_relations
          4. [Train] Augment with unfound GT boxes
          5. Extract ROI + union features for finalized bboxes
        """
        is_train = (split_name == "train")
        assign_iou = self.cfg.assign_iou_train if is_train else self.cfg.assign_iou_test

        # ====================================================================
        # Phase 1: Run detector on all frames, collect detections
        # ====================================================================
        per_frame_data = []  # [(frame_file, img_chw, orig_w, orig_h, target_w, target_h, bboxes_orig, labels, label_ids, scores, scaled_bboxes)]

        all_frame_bboxes_for_assign = []
        all_frame_label_ids_for_assign = []

        for frame_items in gt_annotations:
            if not frame_items:
                all_frame_bboxes_for_assign.append([])
                all_frame_label_ids_for_assign.append([])
                per_frame_data.append(None)
                continue

            # Get frame name
            first_item = frame_items[0]
            frame_relpath = first_item.get("frame", "")
            frame_file = frame_relpath.split("/")[-1] if "/" in frame_relpath else frame_relpath

            if not frame_file:
                all_frame_bboxes_for_assign.append([])
                all_frame_label_ids_for_assign.append([])
                per_frame_data.append(None)
                continue

            # Load and preprocess image
            result = self._load_and_preprocess_frame(video_id, frame_file)
            if result is None:
                all_frame_bboxes_for_assign.append([])
                all_frame_label_ids_for_assign.append([])
                per_frame_data.append(None)
                continue

            img_chw, orig_w, orig_h, target_w, target_h = result
            scale_x = target_w / float(orig_w)
            scale_y = target_h / float(orig_h)

            # Run detector
            det_boxes, det_labels, det_scores, det_boxes_3d = self._detect_frame(img_chw)

            # Filter detections
            bboxes_orig, labels, label_ids, scores, boxes_3d_np = self._filter_detections(
                det_boxes, det_labels, det_scores, det_boxes_3d, scale_x, scale_y
            )

            # Scale filtered bboxes to resized coords for feature extraction
            if bboxes_orig:
                scaled_bboxes, valid_indices = self._scale_and_validate_bboxes(
                    bboxes_orig, scale_x, scale_y, target_w, target_h
                )
                # Filter to only valid bboxes
                bboxes_orig = [bboxes_orig[i] for i in valid_indices]
                labels = [labels[i] for i in valid_indices]
                label_ids = [label_ids[i] for i in valid_indices]
                scores = [scores[i] for i in valid_indices]
                boxes_3d_np = boxes_3d_np[valid_indices]
            else:
                scaled_bboxes = []
                boxes_3d_np = np.empty((0, 8, 3), dtype=np.float32)

            all_frame_bboxes_for_assign.append(bboxes_orig)
            all_frame_label_ids_for_assign.append(label_ids)

            per_frame_data.append({
                "frame_file": frame_file,
                "img_chw": img_chw,
                "orig_w": orig_w,
                "orig_h": orig_h,
                "target_w": target_w,
                "target_h": target_h,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "bboxes_orig": bboxes_orig,
                "labels": labels,
                "label_ids": label_ids,
                "scores": scores,
                "scaled_bboxes": scaled_bboxes,
                "boxes_3d": boxes_3d_np,
            })

        # ====================================================================
        # Phase 2: Match detections to GT via assign_relations
        # ====================================================================
        prediction = self._build_prediction_for_assign(
            all_frame_bboxes_for_assign, all_frame_label_ids_for_assign
        )

        if prediction["FINAL_BBOXES"].shape[0] == 0 and not is_train:
            logger.info(f"[{video_id}] No detections produced")
            return None

        DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(
            prediction, gt_annotations, assign_IOU_threshold=assign_iou
        )

        logger.info(
            f"[{video_id}] assign_relations (IoU={assign_iou}): "
            f"found={sum(len(d) for d in DETECTOR_FOUND_IDX)}, "
            f"supply={sum(len(s) for s in SUPPLY_RELATIONS)}"
        )

        # ====================================================================
        # Phase 3: Augment with unfound GT (train only) + extract features
        # ====================================================================
        frames_data = {}

        for frame_idx, frame_data in enumerate(per_frame_data):
            if frame_data is None:
                continue

            frame_file = frame_data["frame_file"]
            img_chw = frame_data["img_chw"]
            bboxes_orig = list(frame_data["bboxes_orig"])
            labels = list(frame_data["labels"])
            label_ids = list(frame_data["label_ids"])
            scores = list(frame_data["scores"])
            scaled_bboxes = list(frame_data["scaled_bboxes"])
            frame_boxes_3d = frame_data["boxes_3d"]  # (N_det, 8, 3)
            sources = ["detector"] * len(bboxes_orig)

            supply_features_tensor = None

            # GT augmentation for training
            if is_train and self.cfg.use_supply and frame_idx < len(SUPPLY_RELATIONS):
                supply_rels = SUPPLY_RELATIONS[frame_idx]
                if len(supply_rels) > 0:
                    # We need FPN features for this frame to extract supply box features
                    images, _ = self.transform([img_chw], None)
                    features = self.backbone(images.tensors.to(self.device))
                    if isinstance(features, torch.Tensor):
                        features = {"0": features}

                    (
                        supply_bboxes_orig, supply_labels, supply_label_ids,
                        supply_scores, supply_features_tensor, supply_boxes_3d
                    ) = self._augment_with_unfound_gt(
                        supply_rels, features, [images.image_sizes[0]],
                        frame_data["scale_x"], frame_data["scale_y"],
                        frame_data["target_w"], frame_data["target_h"],
                    )

                    if supply_bboxes_orig:
                        # Scale supply bboxes to resized coords
                        supply_scaled, supply_valid = self._scale_and_validate_bboxes(
                            supply_bboxes_orig,
                            frame_data["scale_x"], frame_data["scale_y"],
                            frame_data["target_w"], frame_data["target_h"],
                        )

                        bboxes_orig.extend(supply_bboxes_orig)
                        labels.extend(supply_labels)
                        label_ids.extend(supply_label_ids)
                        scores.extend(supply_scores)
                        sources.extend(["gt_supply"] * len(supply_bboxes_orig))
                        scaled_bboxes.extend(supply_scaled)
                        # Concatenate 3D predictions: detector + supply
                        frame_boxes_3d = np.concatenate(
                            [frame_boxes_3d, supply_boxes_3d], axis=0
                        )

                        logger.debug(
                            f"[{video_id}][{frame_file}] Supply: added {len(supply_bboxes_orig)} GT boxes"
                        )

            if not bboxes_orig:
                continue

            # Extract ROI + union features for all finalized bboxes
            roi_features, union_features_np, pair_indices = self._extract_roi_and_union_features(
                img_chw, scaled_bboxes, labels, label_ids,
            )

            # If we have supply features, we need to replace their entries in roi_features
            # since we already extracted them from the supply augmentation step.
            # However, _extract_roi_and_union_features re-extracts features for ALL bboxes
            # (including the supply ones), which is correct — it uses the same backbone
            # and produces consistent features.
            # The supply_features_tensor was used to ensure we CAN extract features for
            # these boxes; the final features come from the unified extraction call.

            # Get assigned labels for this frame's detections
            frame_assigned_labels = []
            if assigned_labels is not None:
                # assigned_labels is flat across all frames; find this frame's entries
                final_bboxes = prediction["FINAL_BBOXES"]
                frame_mask = final_bboxes[:, 0] == frame_idx
                frame_assigned = assigned_labels[frame_mask.numpy() if isinstance(frame_mask, torch.Tensor) else frame_mask]
                frame_assigned_labels = frame_assigned.tolist()
                # Pad with 0s for supply boxes
                frame_assigned_labels.extend([0] * (len(bboxes_orig) - len(frame_assigned_labels)))

            # Detector-found indices for this frame
            frame_detector_found = []
            if frame_idx < len(DETECTOR_FOUND_IDX):
                frame_detector_found = DETECTOR_FOUND_IDX[frame_idx]

            logger.debug(
                f"[{video_id}][{frame_file}] Final: {len(bboxes_orig)} boxes "
                f"(detector={sources.count('detector')}, supply={sources.count('gt_supply')}), "
                f"{len(pair_indices)} pairs"
            )

            # Store frame entry
            frame_entry = {
                "roi_features": roi_features.cpu().numpy().astype(self.store_dtype),
                "bboxes_xyxy": np.array(bboxes_orig, dtype=np.float32),
                "boxes_3d": frame_boxes_3d.astype(self.store_dtype),
                "labels": labels,
                "label_ids": label_ids,
                "sources": sources,
                "scores": scores,
                "assigned_labels": frame_assigned_labels,
                "detector_found_idx": frame_detector_found,
                "pair_indices": pair_indices,
            }
            if union_features_np is not None:
                frame_entry["union_features"] = union_features_np

            frames_data[frame_file] = frame_entry

        if not frames_data:
            logger.info(f"[{video_id}] No frames produced features")
            return None

        # Log per-video summary
        total_detector = sum(
            fd["sources"].count("detector") for fd in frames_data.values()
        )
        total_supply = sum(
            fd["sources"].count("gt_supply") for fd in frames_data.values()
        )
        logger.info(
            f"[{video_id}] SUMMARY: frames={len(frames_data)}, "
            f"detector_boxes={total_detector}, supply_boxes={total_supply}, "
            f"split={split_name}, iou_threshold={assign_iou}"
        )

        return {
            "video_id": video_id,
            "model": self.cfg.model,
            "mode": "sgdet",
            "split": split_name,
            "feature_dim": 1024,
            "assign_iou_threshold": assign_iou,
            "use_supply": self.cfg.use_supply and is_train,
            "score_threshold": self.cfg.score_threshold,
            "checkpoint": os.path.join(
                self.cfg.working_dir or "", self.cfg.experiment_name,
                self.cfg.ckpt or "pretrained_only",
            ),
            "frames": frames_data,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser(description="Extract ROI features (SGDet mode — detector + GT augmentation)")
    args = parser.parse_args()

    cfg, config_path = parse_config(args)

    # Setup logging early
    model_dir = MODEL_TO_DIR.get(cfg.model, cfg.model)
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(cfg.data_path) / "features" / "roi_features" / "sgdet" / model_dir
    log_dir = Path("logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = setup_logging(log_dir, config_path)
    print(f"Log file: {log_file}")

    logger.info(f"Config file: {config_path}")
    logger.info(f"Resolved config: {cfg}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"SGDet params: score_threshold={cfg.score_threshold}, "
                f"assign_iou_train={cfg.assign_iou_train}, assign_iou_test={cfg.assign_iou_test}, "
                f"use_supply={cfg.use_supply}")

    extractor = SGDetROIFeatureExtractor(cfg)
    extractor.extract_all()


if __name__ == "__main__":
    main()
