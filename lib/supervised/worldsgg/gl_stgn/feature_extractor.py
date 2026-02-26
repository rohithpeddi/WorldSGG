"""
DINO Feature Extractor
======================

Wraps the pre-trained frozen DinoV3Monocular3D detector to extract
per-object visual features (1024-dim ROI features) from video frames.

No gradients flow through the detector — it is used purely for feature
extraction during GL-STGN training and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class DINOFeatureExtractor(nn.Module):
    """
    Frozen DINO detector wrapper for visual feature extraction.

    Loads a pre-trained DinoV3Monocular3D model, freezes all parameters,
    and provides a method to extract 1024-dim ROI box features for
    detected or GT-specified objects.

    Args:
        detector_ckpt: Path to pre-trained detector checkpoint.
        detector_model: DINO model variant string (e.g., "v3l", "v2l").
        num_classes: Number of object classes in the detector.
        device: Device to load the model on.
    """

    def __init__(
        self,
        detector_ckpt: str = "",
        detector_model: str = "v3l",
        num_classes: int = 37,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self._detector = None
        self._detector_ckpt = detector_ckpt
        self._detector_model = detector_model
        self._num_classes = num_classes

    def load_detector(self):
        """Lazy-load the detector to avoid import issues at module level."""
        if self._detector is not None:
            return

        from lib.detector.monocular3d.models.dino_mono_3d import DinoV3Monocular3D

        self._detector = DinoV3Monocular3D(
            num_classes=self._num_classes,
            pretrained=True,
            model=self._detector_model,
            head_3d_mode="unified",
        )

        # Load checkpoint if provided
        if self._detector_ckpt:
            state = torch.load(self._detector_ckpt, map_location="cpu")
            if "model_state_dict" in state:
                self._detector.load_state_dict(state["model_state_dict"], strict=False)
            elif "state_dict" in state:
                self._detector.load_state_dict(state["state_dict"], strict=False)
            else:
                self._detector.load_state_dict(state, strict=False)
            print(f"[DINOFeatureExtractor] Loaded detector from: {self._detector_ckpt}")

        # Freeze everything
        self._detector.to(self.device)
        self._detector.eval()
        for param in self._detector.parameters():
            param.requires_grad = False

        print(f"[DINOFeatureExtractor] Detector frozen ({self._detector_model})")

    @torch.no_grad()
    def extract_features(
        self,
        images: List[torch.Tensor],
        gt_boxes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract ROI features from the frozen detector.

        Args:
            images: List of (3, H, W) tensors — video frames.
            gt_boxes: Optional list of (K_i, 4) tensors — GT 2D boxes to pool
                      features from (for predcls/sgcls modes).

        Returns:
            dict:
                roi_features: (N_total, 1024) — ROI features for all detections.
                boxes_2d: (N_total, 4) — 2D bounding boxes.
                labels: (N_total,) — predicted class labels.
                scores: (N_total,) — detection confidence scores.
                boxes_3d: (N_total, 8, 3) — predicted 3D corners.
                splits: List[int] — number of detections per image.
        """
        self.load_detector()
        detector = self._detector

        # Ensure eval mode
        detector.eval()

        # Run backbone + FPN
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            img_list = list(images.unbind(0))
        else:
            img_list = [img.to(self.device) for img in images]

        original_image_sizes = [img.shape[-2:] for img in img_list]
        images_transformed, targets = detector.transform(img_list, None)
        features = detector.backbone(images_transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        # Run RPN
        proposals, _ = detector.rpn(images_transformed, features, None)

        if gt_boxes is not None:
            # Use GT boxes instead of proposals for feature extraction
            proposals = [b.to(self.device) for b in gt_boxes]

        # Pool features through ROI heads
        roi_heads = detector.roi_heads
        if hasattr(roi_heads, 'base'):
            # Unified mode (Mono3DRoIHeads wraps base)
            base_roi = roi_heads.base
        else:
            base_roi = roi_heads

        box_features = base_roi.box_roi_pool(features, proposals, images_transformed.image_sizes)
        box_features = base_roi.box_head(box_features)  # (N_total, 1024)

        # Get class predictions
        class_logits, box_regression = base_roi.box_predictor(box_features)

        # Post-process for detection results
        boxes_list, scores_list, labels_list = base_roi.postprocess_detections(
            class_logits, box_regression, proposals, images_transformed.image_sizes,
        )

        # Re-pool features for final detections (boxes may have changed)
        det_boxes = boxes_list
        splits = [len(b) for b in det_boxes]

        if sum(splits) > 0:
            det_features = base_roi.box_roi_pool(
                features, det_boxes, images_transformed.image_sizes
            )
            det_features = base_roi.box_head(det_features)  # (N_total, 1024)
        else:
            det_features = torch.zeros(0, 1024, device=self.device)

        # Concatenate
        all_boxes = torch.cat(det_boxes, dim=0) if splits else torch.zeros(0, 4, device=self.device)
        all_scores = torch.cat(scores_list, dim=0) if splits else torch.zeros(0, device=self.device)
        all_labels = torch.cat(labels_list, dim=0) if splits else torch.zeros(0, dtype=torch.long, device=self.device)

        return {
            "roi_features": det_features,
            "boxes_2d": all_boxes,
            "labels": all_labels,
            "scores": all_scores,
            "splits": splits,
        }

    @torch.no_grad()
    def extract_features_for_boxes(
        self,
        images: List[torch.Tensor],
        boxes: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract ROI features for specific 2D boxes (e.g. GT boxes in predcls mode).

        Args:
            images: List of (3, H, W) tensors.
            boxes: List of (K_i, 4) tensors — 2D boxes to extract features for.

        Returns:
            roi_features: (N_total, 1024) concatenated features.
        """
        self.load_detector()
        detector = self._detector
        detector.eval()

        img_list = [img.to(self.device) for img in images]
        images_transformed, _ = detector.transform(img_list, None)
        features = detector.backbone(images_transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        roi_heads = detector.roi_heads
        if hasattr(roi_heads, 'base'):
            base_roi = roi_heads.base
        else:
            base_roi = roi_heads

        box_list = [b.to(self.device) for b in boxes]
        total = sum(len(b) for b in box_list)

        if total == 0:
            return torch.zeros(0, 1024, device=self.device)

        box_features = base_roi.box_roi_pool(
            features, box_list, images_transformed.image_sizes
        )
        box_features = base_roi.box_head(box_features)  # (N_total, 1024)

        return box_features

    def forward(self, *args, **kwargs):
        """Not used directly — use extract_features or extract_features_for_boxes."""
        raise NotImplementedError("Use extract_features() or extract_features_for_boxes()")
