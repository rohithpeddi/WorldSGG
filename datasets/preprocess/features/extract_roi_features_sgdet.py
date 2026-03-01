#!/usr/bin/env python3
"""
ROI Feature Extraction for SGDet
====================================

Extracts pre-computed ROI features using the DinoV3 detector's own detections
(not GT bboxes). Mirrors the original Faster R-CNN Detector's sgdet pipeline:

  1. Run DinoV3 detector → per-frame detections (boxes, labels, scores) in Pi-3 space
  2. Apply score threshold + keep single highest-scoring person per frame
  3. Match detections to GT via IoU in Pi-3 space (train: 0.5, test: 0.3)
  4. [Train only] Augment with unfound GT boxes (SUPPLY_RELATIONS)
  5. Extract ROI features + union features for all finalized bounding boxes
  6. Compute 3D predictions for ALL finalized bboxes using unified ROI features
  7. Save per-video PKL files (train/ and test/ subdirectories)

All bboxes throughout the pipeline are in Pi-3 scaled space.

Output PKL format
-----------------
One ``.pkl`` file per video, saved under ``<output_dir>/<split>/``:

::

    {
        "video_id":              str,            # e.g. "001YG.mp4"
        "model":                 str,            # backbone id, e.g. "v2b"
        "mode":                  "sgdet",
        "coord_space":           "pi3",          # all bboxes in Pi-3 scaled space
        "split":                 str,            # "train" or "test"
        "feature_dim":           int,            # 1024
        "assign_iou_threshold":  float,          # IoU used for GT matching
        "use_supply":            bool,           # whether GT augmentation was enabled
        "score_threshold":       float,          # detector score cutoff
        "checkpoint":            str,            # model checkpoint path
        "frames": {
            "<frame_filename>": {                # e.g. "000001.png"
                "roi_features":      np.ndarray, # (N, 1024) float16/32 ROI-pooled features
                "bboxes_xyxy":       np.ndarray, # (N, 4)    float32 [x1, y1, x2, y2] in Pi-3
                "boxes_3d":          np.ndarray, # (N, D)    float16/32 predicted 3D params
                "target_size":       (int, int), # (width, height) of the Pi-3 image
                "labels":            list[str],  # N normalized label strings
                "label_ids":         list[int],  # N AG class indices (1-indexed)
                "sources":           list[str],  # N, each "detector" or "gt_supply"
                "scores":            list[float],# N detection scores (supply=1.0)
                "assigned_labels":   list[int],  # N GT class ids from assign_relations
                "detector_found_idx":list[int],  # indices of GT objects matched by detector
                "pair_indices":      list[tuple],# person-object pairs [(person_idx, obj_idx)]
                "union_features":    np.ndarray, # (P, 1024) float16/32 union features (optional)
            },
            ...
        }
    }

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
    assign_relations,
    scale_gt_annotations_to_pi3,
    build_parser,
    logger,
    parse_config,
    setup_logging,
)


# ---------------------------------------------------------------------------
# SGDet ROI Feature Extractor
# ---------------------------------------------------------------------------

class SGDetROIFeatureExtractor(BaseROIFeatureExtractor):
    """
    SGDet: runs DinoV3 detector → NMS → GT matching → GT augmentation (train).

    All operations are in Pi-3 scaled space (resized image coordinates).

    For training: detections are matched to GT at IoU >= 0.5.
                  Unfound GT objects are injected with score=1.0 and true labels.
    For testing:  detections are matched to GT at IoU >= 0.3. No augmentation.
    """

    MODE_NAME = "sgdet"

    def _get_output_dir(self, split_name: str) -> Path:
        """Return the output directory for the given split.

        Resolves to ``<output_dir>/<split>`` when ``output_dir`` is set,
        otherwise ``<data_path>/features/roi_features/sgdet/<model>/<split>``.

        Args:
            split_name: ``"train"`` or ``"test"``.

        Returns:
            Path to the split-specific output directory.
        """
        model_dir = MODEL_TO_DIR.get(self.cfg.model, self.cfg.model)
        if self.cfg.output_dir:
            return Path(self.cfg.output_dir) / split_name
        return Path(self.cfg.data_path) / "features" / "roi_features" / "sgdet" / model_dir / split_name

    # ------------------------------------------------------------------
    # Detection + filtering (all in Pi-3 space)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _detect_frame(
        self,
        img_chw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the DinoV3 detector on a single preprocessed frame.

        The model's built-in ``postprocess_detections`` applies per-class NMS
        (default IoU 0.5) and score-based top-k filtering (default 100
        detections per image).

        Args:
            img_chw: (3, H, W) preprocessed image tensor.

        Returns:
            boxes:  (N, 4) xyxy bounding boxes in Pi-3 coordinates.
            labels: (N,) class indices (1-indexed; 0 = background).
            scores: (N,) confidence scores.
        """
        detections = self.model([img_chw.to(self.device)])
        det = detections[0]
        return det["boxes"], det["labels"], det["scores"]

    def _filter_detections(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[List[List[float]], List[str], List[int], List[float]]:
        """Filter raw detector outputs in Pi-3 space.

        Two-stage filtering:
          1. Drop detections below ``cfg.score_threshold`` (default 0.1).
          2. For the *person* class (index 1), keep only the single
             highest-scoring detection.

        Args:
            boxes:  (N, 4) xyxy tensor from the detector.
            labels: (N,) class-index tensor (1-indexed).
            scores: (N,) confidence-score tensor.

        Returns:
            bboxes_pi3:    list of [x1, y1, x2, y2] float lists (Pi-3 coords).
            det_labels:    list of normalized label strings.
            det_label_ids: list of AG class indices.
            det_scores:    list of confidence scores.
        """
        if len(boxes) == 0:
            return [], [], [], []

        # Score threshold
        keep_mask = scores >= self.cfg.score_threshold
        boxes = boxes[keep_mask]
        labels = labels[keep_mask]
        scores = scores[keep_mask]

        if len(boxes) == 0:
            return [], [], [], []

        # Person class (index 1): keep only top-1
        person_mask = labels == 1
        non_person_mask = ~person_mask

        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []

        if person_mask.any():
            person_scores = scores[person_mask]
            best_person_idx = person_scores.argmax()
            person_boxes_all = boxes[person_mask]
            filtered_boxes.append(person_boxes_all[best_person_idx].unsqueeze(0))
            filtered_labels.append(labels[person_mask][best_person_idx].unsqueeze(0))
            filtered_scores.append(person_scores[best_person_idx].unsqueeze(0))

        if non_person_mask.any():
            filtered_boxes.append(boxes[non_person_mask])
            filtered_labels.append(labels[non_person_mask])
            filtered_scores.append(scores[non_person_mask])

        if not filtered_boxes:
            return [], [], [], []

        boxes = torch.cat(filtered_boxes, dim=0)
        labels = torch.cat(filtered_labels, dim=0)
        scores = torch.cat(filtered_scores, dim=0)

        # Convert to lists (already in Pi-3 coords — no scale conversion)
        boxes_np = boxes.cpu().numpy()
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()

        bboxes_pi3 = []
        det_labels = []
        det_label_ids = []
        det_scores = []

        for i in range(len(boxes_np)):
            x1, y1, x2, y2 = boxes_np[i]
            bboxes_pi3.append([float(x1), float(y1), float(x2), float(y2)])
            cls_idx = int(labels_np[i])
            cls_name = CATID_TO_NAME.get(cls_idx, None)
            if cls_name is not None:
                det_labels.append(_normalize_label(cls_name))
            else:
                det_labels.append(f"class_{cls_idx}")
            det_label_ids.append(cls_idx)
            det_scores.append(float(scores_np[i]))

        return bboxes_pi3, det_labels, det_label_ids, det_scores

    # ------------------------------------------------------------------
    # GT matching with Pi-3 scaled annotations
    # ------------------------------------------------------------------

    def _assign_relations_pi3(
        self,
        prediction: Dict[str, Any],
        gt_annotations: List[List[Dict[str, Any]]],
        scale_x: float,
        scale_y: float,
        target_w: int,
        target_h: int,
        assign_iou: float,
    ):
        """Match detections to GT annotations with bboxes in Pi-3 space.

        Wrapper around :func:`assign_relations` that first scales raw GT
        annotation bboxes (in original image coordinates) to Pi-3 space so
        that IoU matching is performed in a consistent coordinate frame.

        Args:
            prediction:     dict with ``FINAL_BBOXES`` (N, 5) and
                            ``FINAL_LABELS`` (N,) tensors (see
                            :meth:`_build_prediction_for_assign`).
            gt_annotations: per-frame list of GT annotation dicts in
                            original image coordinates.
            scale_x:        horizontal scale factor (Pi-3 / original).
            scale_y:        vertical scale factor (Pi-3 / original).
            target_w:       Pi-3 image width.
            target_h:       Pi-3 image height.
            assign_iou:     IoU threshold for matching (train: 0.5,
                            test: 0.3).

        Returns:
            Tuple of ``(DETECTOR_FOUND_IDX, GT_RELATIONS,
            SUPPLY_RELATIONS, assigned_labels)`` — all with bboxes in
            Pi-3 space.
        """
        scaled_gt = scale_gt_annotations_to_pi3(
            gt_annotations, scale_x, scale_y, target_w, target_h
        )
        return assign_relations(prediction, scaled_gt, assign_IOU_threshold=assign_iou)

    # ------------------------------------------------------------------
    # GT augmentation (training only) — bboxes already in Pi-3
    # ------------------------------------------------------------------

    def _augment_with_unfound_gt(
        self,
        supply_relations: List[Dict[str, Any]],
        features: Dict[str, torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> Tuple[List[List[float]], List[str], List[int], List[float], torch.Tensor]:
        """Extract ROI features for GT objects not found by the detector.

        During training, ``assign_relations`` returns ``SUPPLY_RELATIONS``
        containing GT objects that the detector missed (IoU below threshold).
        This method extracts ROI features for those objects so they can be
        injected into the final feature set with a confidence score of 1.0.

        Supply bboxes are **already in Pi-3 space** (produced by
        :meth:`_assign_relations_pi3`), so no coordinate conversion occurs.

        Degenerate boxes (width or height < 1 px) are silently skipped.

        Args:
            supply_relations: list of GT annotation dicts with Pi-3 bboxes.
                Each dict contains either ``person_bbox`` or ``bbox`` +
                ``class`` keys.
            features:    FPN feature-map dict from the backbone (already
                         computed for the current frame).
            image_sizes: list of ``(H, W)`` tuples from the model transform.

        Returns:
            supply_bboxes_pi3:  list of [x1, y1, x2, y2] float lists.
            supply_labels:      list of normalized label strings.
            supply_label_ids:   list of AG class indices (1-indexed).
            supply_scores:      list of floats, all 1.0.
            supply_features:    (M, 1024) tensor on ``self.device``; empty
                                tensor when no valid supply boxes exist.
        """
        supply_bboxes_pi3 = []
        supply_labels = []
        supply_label_ids = []
        supply_scores = []

        for gt_item in supply_relations:
            if "person_bbox" in gt_item:
                # Person bbox (already Pi-3 scaled)
                pb = gt_item["person_bbox"]
                if isinstance(pb, np.ndarray):
                    pb = pb.reshape(-1).tolist()
                elif isinstance(pb, (list, tuple)):
                    if isinstance(pb[0], (list, tuple, np.ndarray)):
                        pb = list(pb[0])
                bbox_pi3 = [float(v) for v in pb[:4]]
                label_str = "person"
                label_id = 1
            elif "bbox" in gt_item and "class" in gt_item:
                bbox = gt_item["bbox"]
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                bbox_pi3 = [float(v) for v in bbox[:4]]
                cls_idx = int(gt_item["class"])
                cls_name = CATID_TO_NAME.get(cls_idx, None)
                label_str = _normalize_label(cls_name) if cls_name else f"class_{cls_idx}"
                label_id = cls_idx
            else:
                continue

            # Validate (already Pi-3 — just check non-degenerate)
            x1, y1, x2, y2 = bbox_pi3
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            supply_bboxes_pi3.append(bbox_pi3)
            supply_labels.append(label_str)
            supply_label_ids.append(label_id)
            supply_scores.append(1.0)

        if not supply_bboxes_pi3:
            return [], [], [], [], torch.tensor([]).to(self.device)

        # Extract ROI features for supply bboxes using pre-computed FPN features
        supply_features = self._extract_roi_features_for_bboxes(
            features, image_sizes, supply_bboxes_pi3
        )

        return supply_bboxes_pi3, supply_labels, supply_label_ids, supply_scores, supply_features

    # ------------------------------------------------------------------
    # Build prediction dict for assign_relations
    # ------------------------------------------------------------------

    def _build_prediction_for_assign(
        self,
        all_frame_bboxes: List[List[List[float]]],
        all_frame_labels: List[List[int]],
    ) -> Dict[str, Any]:
        """Assemble per-frame detections into the dict expected by ``assign_relations``.

        ``assign_relations`` expects:

        * ``prediction["FINAL_BBOXES"]``: ``(N, 5)`` float tensor with
          ``[frame_idx, x1, y1, x2, y2]`` rows.
        * ``prediction["FINAL_LABELS"]``: ``(N,)`` int64 tensor of AG
          class indices.

        All bboxes must be in Pi-3 space.

        Args:
            all_frame_bboxes: per-frame list of bbox coordinate lists,
                each ``[x1, y1, x2, y2]`` in Pi-3 space.
            all_frame_labels: per-frame list of AG class-index ints,
                aligned with *all_frame_bboxes*.

        Returns:
            dict with ``FINAL_BBOXES`` and ``FINAL_LABELS`` tensors.
            Returns zero-length tensors when there are no detections.
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
        """Run the full SGDet feature-extraction pipeline for one video.

        All coordinates are in Pi-3 (resized-image) space throughout.

        Pipeline stages:
          1. Run the DinoV3 detector on every frame.
          2. Filter detections (score threshold + person top-1).
          3. Match detections to GT via ``assign_relations`` (GT scaled
             to Pi-3; IoU threshold differs for train vs. test).
          4. **[Train only]** Augment with unfound GT boxes
             (``SUPPLY_RELATIONS``).
          5. Extract ROI-pooled features and person–object union features
             for all finalized bounding boxes.
          6. Compute 3D predictions from unified ROI features.

        Args:
            video_id:       identifier of the video (e.g. ``"001YG.mp4"``).
            gt_annotations: per-frame list of GT annotation dicts.
            frame_names:    ordered frame filenames for the video.
            split_name:     ``"train"`` or ``"test"``.

        Returns:
            A dict matching the output PKL schema documented in the module
            docstring, or ``None`` when no frames yield valid features.
        """
        is_train = (split_name == "train")
        assign_iou = self.cfg.assign_iou_train if is_train else self.cfg.assign_iou_test

        # ====================================================================
        # Phase 1: Run detector on all frames, collect detections (Pi-3 space)
        # ====================================================================
        per_frame_data = []

        all_frame_bboxes_for_assign = []
        all_frame_label_ids_for_assign = []

        # We need consistent scale factors — compute from first valid frame
        video_scale_x = None
        video_scale_y = None
        video_target_w = None
        video_target_h = None

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

            # Store scale for assign_relations (same across all frames in a video)
            if video_scale_x is None:
                video_scale_x = scale_x
                video_scale_y = scale_y
                video_target_w = target_w
                video_target_h = target_h

            # Run detector (output is already Pi-3)
            det_boxes, det_labels, det_scores = self._detect_frame(img_chw)

            # Filter detections (stays in Pi-3)
            bboxes_pi3, labels, label_ids, scores = self._filter_detections(
                det_boxes, det_labels, det_scores
            )

            all_frame_bboxes_for_assign.append(bboxes_pi3)
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
                "bboxes_pi3": bboxes_pi3,
                "labels": labels,
                "label_ids": label_ids,
                "scores": scores,
            })

        # ====================================================================
        # Phase 2: Match detections to GT via assign_relations (Pi-3 space)
        # ====================================================================
        prediction = self._build_prediction_for_assign(
            all_frame_bboxes_for_assign, all_frame_label_ids_for_assign
        )

        if prediction["FINAL_BBOXES"].shape[0] == 0 and not is_train:
            logger.info(f"[{video_id}] No detections produced")
            return None

        # Use scale from first valid frame (all frames in a video have same resolution)
        if video_scale_x is None:
            logger.info(f"[{video_id}] No valid frames found")
            return None

        DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = self._assign_relations_pi3(
            prediction, gt_annotations,
            video_scale_x, video_scale_y, video_target_w, video_target_h,
            assign_iou,
        )

        logger.info(
            f"[{video_id}] assign_relations (IoU={assign_iou}, Pi-3 space): "
            f"found={sum(len(d) for d in DETECTOR_FOUND_IDX)}, "
            f"supply={sum(len(s) for s in SUPPLY_RELATIONS)}"
        )

        # ====================================================================
        # Phase 3: Augment + extract features + 3D predictions (all Pi-3)
        # ====================================================================
        frames_data = {}

        for frame_idx, frame_data in enumerate(per_frame_data):
            if frame_data is None:
                continue

            frame_file = frame_data["frame_file"]
            img_chw = frame_data["img_chw"]
            bboxes_pi3 = list(frame_data["bboxes_pi3"])
            labels = list(frame_data["labels"])
            label_ids = list(frame_data["label_ids"])
            scores = list(frame_data["scores"])
            sources = ["detector"] * len(bboxes_pi3)

            # GT augmentation for training
            if is_train and self.cfg.use_supply and frame_idx < len(SUPPLY_RELATIONS):
                supply_rels = SUPPLY_RELATIONS[frame_idx]
                if len(supply_rels) > 0:
                    # We need FPN features for supply box feature extraction
                    images, _ = self.transform([img_chw], None)
                    features = self.backbone(images.tensors.to(self.device))
                    if isinstance(features, torch.Tensor):
                        features = {"0": features}

                    (
                        supply_bboxes_pi3, supply_labels, supply_label_ids,
                        supply_scores, supply_features_tensor
                    ) = self._augment_with_unfound_gt(
                        supply_rels, features, [images.image_sizes[0]],
                    )

                    if supply_bboxes_pi3:
                        bboxes_pi3.extend(supply_bboxes_pi3)
                        labels.extend(supply_labels)
                        label_ids.extend(supply_label_ids)
                        scores.extend(supply_scores)
                        sources.extend(["gt_supply"] * len(supply_bboxes_pi3))

                        logger.debug(
                            f"[{video_id}][{frame_file}] Supply: added {len(supply_bboxes_pi3)} GT boxes"
                        )

            if not bboxes_pi3:
                continue

            # Extract ROI + union features for ALL finalized bboxes (Pi-3)
            roi_features, union_features_np, pair_indices = self._extract_roi_and_union_features(
                img_chw, bboxes_pi3, labels, label_ids,
            )

            # Compute 3D predictions for ALL finalized bboxes using unified ROI features
            images_for_3d, _ = self.transform([img_chw], None)
            boxes_3d = self._predict_3d_for_bboxes(
                roi_features, bboxes_pi3, [images_for_3d.image_sizes[0]]
            )

            # Get assigned labels for this frame's detections
            frame_assigned_labels = []
            if assigned_labels is not None:
                final_bboxes = prediction["FINAL_BBOXES"]
                frame_mask = final_bboxes[:, 0] == frame_idx
                frame_assigned = assigned_labels[frame_mask.numpy() if isinstance(frame_mask, torch.Tensor) else frame_mask]
                frame_assigned_labels = frame_assigned.tolist()
                # Pad with 0s for supply boxes
                frame_assigned_labels.extend([0] * (len(bboxes_pi3) - len(frame_assigned_labels)))

            # Detector-found indices for this frame
            frame_detector_found = []
            if frame_idx < len(DETECTOR_FOUND_IDX):
                frame_detector_found = DETECTOR_FOUND_IDX[frame_idx]

            logger.debug(
                f"[{video_id}][{frame_file}] Final: {len(bboxes_pi3)} boxes "
                f"(detector={sources.count('detector')}, supply={sources.count('gt_supply')}), "
                f"{len(pair_indices)} pairs"
            )

            # Store frame entry (all in Pi-3 space)
            frame_entry = {
                "roi_features": roi_features.cpu().numpy().astype(self.store_dtype),
                "bboxes_xyxy": np.array(bboxes_pi3, dtype=np.float32),
                "boxes_3d": boxes_3d.cpu().numpy().astype(self.store_dtype),
                "target_size": (frame_data["target_w"], frame_data["target_h"]),
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
            "coord_space": "pi3",
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
    """CLI entry-point: parse args, configure logging, and run SGDet extraction."""
    parser = build_parser(description="Extract ROI features (SGDet mode — detector + GT augmentation)")
    args = parser.parse_args()

    cfg, config_path = parse_config(args)

    # Setup logging early
    model_dir = MODEL_TO_DIR.get(cfg.model, cfg.model)
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(cfg.data_path) / "features" / "roi_features" / "sgdet" / model_dir
    log_dir = Path(_PROJECT_ROOT) / "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = setup_logging(log_dir, config_path, mode_name="sgdet")
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
