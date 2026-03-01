#!/usr/bin/env python3
"""
ROI Feature Extraction for PredCls
====================================

Extracts pre-computed ROI features from trained DINOv2/v3 detectors for all
annotated objects (GT + GDino-filled missing objects) per frame.

Uses GT bounding boxes directly — no detection, no NMS.

Output PKL format
-----------------
One ``.pkl`` file per video, saved under ``<output_dir>/<split>/``:

::

    {
        "video_id":     str,            # e.g. "001YG.mp4"
        "model":        str,            # backbone id, e.g. "v2b"
        "mode":         "predcls",
        "coord_space":  "pi3",          # all bboxes in Pi-3 scaled space
        "feature_dim":  int,            # 1024
        "checkpoint":   str,            # model checkpoint path
        "frames": {
            "<frame_filename>": {        # e.g. "000001.png"
                "roi_features":  np.ndarray, # (N, 1024) float16/32 ROI-pooled features
                "bboxes_xyxy":   np.ndarray, # (N, 4)    float32 [x1,y1,x2,y2] in Pi-3
                "target_size":   (int, int), # (width, height) of the Pi-3 image
                "labels":        list[str],  # N normalized label strings
                "label_ids":     list[int],  # N AG class indices (1-indexed)
                "sources":       list[str],  # N, each "gt" or "gdino"
                "gdino_scores":  list[float],# N, 0.0 for GT objects
                "pair_indices":  list[tuple],# person-object pairs [(person_idx, obj_idx)]
                "union_features":np.ndarray, # (P, 1024) float16/32 union features (optional)
            },
            ...
        }
    }

Usage:
    python datasets/preprocess/features/extract_roi_features_predcls.py \
        --config configs/features/predcls/ex_roi_feat_v1_dinov2b_saurabh.yaml

    # Override via CLI:
    python datasets/preprocess/features/extract_roi_features_predcls.py \
        --config configs/features/predcls/ex_roi_feat_v1_dinov2b_saurabh.yaml \
        --model v2l --ckpt checkpoint_50 --video 001YG.mp4
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    LABEL_TO_CLASSIDX,
    MODEL_TO_DIR,
    _load_gdino_predictions,
    _normalize_label,
    build_parser,
    logger,
    parse_config,
    setup_logging,
)


class PredClsROIFeatureExtractor(BaseROIFeatureExtractor):
    """
    PredCls: uses GT bboxes + GDino fill for missing labels.
    No detection, no NMS. Features are extracted for all known objects.
    """

    MODE_NAME = "predcls"

    def _get_output_dir(self, split_name: str) -> Path:
        """Return the output directory for the given split.

        Resolves to ``<output_dir>/<split>`` when ``output_dir`` is set,
        otherwise ``<data_path>/features/roi_features/predcls/<model>/<split>``.

        Args:
            split_name: ``"train"`` or ``"test"``.

        Returns:
            Path to the split-specific output directory.
        """
        from pathlib import Path as _Path
        model_dir = MODEL_TO_DIR.get(self.cfg.model, self.cfg.model)
        if self.cfg.output_dir:
            return _Path(self.cfg.output_dir) / split_name
        return _Path(self.cfg.data_path) / "features" / "roi_features" / "predcls" / model_dir / split_name

    @torch.no_grad()
    def _extract_features_for_video(
        self,
        video_id: str,
        gt_annotations: List[List[Dict[str, Any]]],
        frame_names: List[str],
        split_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Run the PredCls feature-extraction pipeline for one video.

        For each annotated frame the method:
          1. Loads and preprocesses the image.
          2. Collects GT bounding boxes (scaled to Pi-3 space) and
             optionally fills in missing object labels with GDino
             detections that pass score / label filtering.
          3. Extracts ROI-pooled features and person–object union
             features for every collected bounding box.

        No detector is run — bounding boxes come directly from
        annotations and (optionally) GDino predictions.

        Args:
            video_id:       identifier of the video (e.g. ``"001YG.mp4"``).
            gt_annotations: per-frame list of GT annotation dicts.
            frame_names:    ordered frame filenames for the video.
            split_name:     ``"train"`` or ``"test"``.

        Returns:
            A dict matching the output PKL schema documented in the
            module docstring, or ``None`` when no frames yield valid
            features.
        """

        # Load GDino detections
        gdino_preds = _load_gdino_predictions(self.cfg.data_path, video_id)

        # Pre-compute video GT label set
        video_gt_labels = self._compute_video_gt_labels(gt_annotations)

        logger.info(f"[{video_id}] Video GT labels ({len(video_gt_labels)}): {sorted(video_gt_labels)}")
        logger.info(f"[{video_id}] GDino detections available: {gdino_preds is not None}")
        if gdino_preds is not None:
            logger.info(f"[{video_id}] GDino frames available: {len(gdino_preds)}")

        # Per-video stats
        video_total_gt = 0
        video_total_gdino_added = 0
        video_total_gdino_raw = 0
        video_total_gdino_rejected_score = 0
        video_total_gdino_rejected_in_gt = 0
        video_total_gdino_rejected_not_in_video = 0

        frames_data = {}

        for frame_items in gt_annotations:
            if not frame_items:
                continue

            # Get frame name
            first_item = frame_items[0]
            frame_relpath = first_item.get("frame", "")
            if "/" in frame_relpath:
                frame_file = frame_relpath.split("/")[-1]
            else:
                frame_file = frame_relpath

            if not frame_file:
                continue

            # Load and preprocess image
            result = self._load_and_preprocess_frame(video_id, frame_file)
            if result is None:
                continue
            img_chw, orig_w, orig_h, target_w, target_h = result

            # Compute bbox scale factors
            scale_x = target_w / float(orig_w)
            scale_y = target_h / float(orig_h)

            # Collect bboxes (scaled to Pi-3 space)
            gdino_frame = gdino_preds.get(frame_file, None) if gdino_preds else None
            (
                bboxes_xyxy, labels, sources, gdino_scores,
                raw_gdino_dets, accepted_gdino, rej_score, rej_in_gt, rej_not_in_video
            ) = self._collect_bboxes_for_frame(
                frame_items, gdino_frame, self.cfg.gdino_score_threshold,
                scale_x, scale_y, target_w, target_h,
                video_gt_labels=video_gt_labels,
            )

            # Log per-frame details
            gt_labels_this_frame = [l for l, s in zip(labels, sources) if s == "gt"]
            gdino_labels_this_frame = [l for l, s in zip(labels, sources) if s == "gdino"]
            logger.debug(
                f"[{video_id}][{frame_file}] GT objects ({len(gt_labels_this_frame)}): {gt_labels_this_frame}"
            )
            if raw_gdino_dets:
                logger.debug(
                    f"[{video_id}][{frame_file}] Raw GDino detections ({len(raw_gdino_dets)}): "
                    f"{[(lbl, f'{sc:.3f}') for lbl, sc in raw_gdino_dets]}"
                )
            if rej_score:
                logger.debug(
                    f"[{video_id}][{frame_file}] GDino rejected (low score): "
                    f"{[(lbl, f'{sc:.3f}') for lbl, sc in rej_score]}"
                )
            if rej_in_gt:
                logger.debug(
                    f"[{video_id}][{frame_file}] GDino rejected (already in GT): "
                    f"{[(raw, norm, f'{sc:.3f}') for raw, norm, sc in rej_in_gt]}"
                )
            if rej_not_in_video:
                logger.debug(
                    f"[{video_id}][{frame_file}] GDino rejected (not in video GT labels): "
                    f"{[(raw, norm, f'{sc:.3f}') for raw, norm, sc in rej_not_in_video]}"
                )
            if accepted_gdino:
                logger.debug(
                    f"[{video_id}][{frame_file}] GDino ACCEPTED ({len(accepted_gdino)}): "
                    f"{[(lbl, f'{sc:.3f}') for lbl, sc in accepted_gdino]}"
                )

            # Accumulate video stats
            video_total_gt += len(gt_labels_this_frame)
            video_total_gdino_added += len(accepted_gdino)
            video_total_gdino_raw += len(raw_gdino_dets)
            video_total_gdino_rejected_score += len(rej_score)
            video_total_gdino_rejected_in_gt += len(rej_in_gt)
            video_total_gdino_rejected_not_in_video += len(rej_not_in_video)

            if not bboxes_xyxy:
                continue

            # Bboxes are already in Pi-3 space — no filtering needed for PredCls
            # Label IDs (AG class indices)
            label_ids = [LABEL_TO_CLASSIDX.get(l, 0) for l in labels]

            # Extract ROI + union features (bboxes already Pi-3)
            roi_features, union_features_np, pair_indices = self._extract_roi_and_union_features(
                img_chw, bboxes_xyxy, labels, label_ids,
            )

            if pair_indices:
                logger.debug(
                    f"[{video_id}][{frame_file}] Union features: {len(pair_indices)} pairs"
                )

            # Store frame entry (all bboxes in Pi-3 space)
            frame_entry = {
                "roi_features": roi_features.cpu().numpy().astype(self.store_dtype),
                "bboxes_xyxy": np.array(bboxes_xyxy, dtype=np.float32),
                "target_size": (target_w, target_h),
                "labels": labels,
                "label_ids": label_ids,
                "sources": sources,
                "gdino_scores": gdino_scores,
                "pair_indices": pair_indices,
            }
            if union_features_np is not None:
                frame_entry["union_features"] = union_features_np

            frames_data[frame_file] = frame_entry

        if not frames_data:
            logger.info(f"[{video_id}] No frames produced features")
            return None

        # Log per-video summary
        logger.info(
            f"[{video_id}] SUMMARY: frames={len(frames_data)}, "
            f"gt_objects={video_total_gt}, gdino_added={video_total_gdino_added}, "
            f"gdino_raw={video_total_gdino_raw}, "
            f"gdino_rej_score={video_total_gdino_rejected_score}, "
            f"gdino_rej_in_gt={video_total_gdino_rejected_in_gt}, "
            f"gdino_rej_not_in_video={video_total_gdino_rejected_not_in_video}"
        )

        return {
            "video_id": video_id,
            "model": self.cfg.model,
            "mode": "predcls",
            "coord_space": "pi3",
            "feature_dim": 1024,
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
    """CLI entry-point: parse args, configure logging, and run PredCls extraction."""
    parser = build_parser(description="Extract ROI features (PredCls mode — GT bboxes)")
    args = parser.parse_args()

    cfg, config_path = parse_config(args)

    # Setup logging early
    from pathlib import Path as _Path
    from datasets.preprocess.features.extract_roi_features_base import MODEL_TO_DIR
    model_dir = MODEL_TO_DIR.get(cfg.model, cfg.model)
    if cfg.output_dir:
        output_dir = _Path(cfg.output_dir)
    else:
        output_dir = _Path(cfg.data_path) / "features" / "roi_features" / "predcls" / model_dir
    log_dir = _Path(_PROJECT_ROOT) / "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = setup_logging(log_dir, config_path, mode_name="predcls")
    print(f"Log file: {log_file}")

    logger.info(f"Config file: {config_path}")
    logger.info(f"Resolved config: {cfg}")
    logger.info(f"Log file: {log_file}")

    extractor = PredClsROIFeatureExtractor(cfg)
    extractor.extract_all()


if __name__ == "__main__":
    main()
