"""
ResNet50-FPN Monocular 3D Object Detector.

Uses torchvision's fasterrcnn_resnet50_fpn_v2 with COCO-pretrained weights,
combined with the same 3D head architecture used by the DINOv2/v3 detector.

Architecture:
  images → _NoOpRCNNTransform (batch-only) → ResNet50+FPN → RPN → ROI Heads (+3D) → detections

The ResNet50 body + FPN is fully trainable (no frozen layers).
The 3D head (Mono3DRoIHeads / SeparateMono3DHead) is identical to the DINOv2 path.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

# Reuse shared components from the DINOv2 model module
from .dino_mono_3d import (
    _NoOpRCNNTransform,
    Mono3DRoIHeads,
    SeparateMono3DHead,
)


class ResNetMonocular3D(nn.Module):
    """
    Full Monocular 3D Object Detector combining:
      - ResNet50 + FPN backbone (COCO-pretrained, fully trainable)
      - Faster R-CNN (RPN + ROI heads) for 2D detection
      - Configurable 3D head (unified or separate)

    head_3d_mode:
      "unified"  — 3D branch inside ROI heads, shares FC layers directly
      "separate" — standalone 3D head hooks into shared FC features

    Forward pipeline:
      images → _NoOpRCNNTransform (batch-only) → ResNet50+FPN → RPN → ROI Heads (+3D) → detections
    """

    def __init__(
        self,
        num_classes=37,
        pretrained=True,
        head_3d_mode="unified",
        max_3d_proposals=64,
        head_3d_version="v1",
        input_reference_size=1000.0,
    ):
        super().__init__()
        self.head_3d_mode = head_3d_mode
        self.head_3d_version = head_3d_version

        # ---- Step 1: Load COCO-pretrained FasterRCNN-ResNet50-FPN-V2 ----
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        base_detector = fasterrcnn_resnet50_fpn_v2(weights=weights)
        print(f"  Backbone: ResNet50-FPN-V2  pretrained={'COCO' if pretrained else 'None'}")

        # ---- Step 2: Replace box_predictor for AG class set ----
        in_features = base_detector.roi_heads.box_predictor.cls_score.in_features
        base_detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print(f"  Box predictor: {in_features} → {num_classes} classes")

        # ---- Step 3: All parameters trainable (full fine-tuning) ----
        # No freezing — the user chose to fine-tune the full ResNet50 body.
        trainable = sum(p.numel() for p in base_detector.parameters() if p.requires_grad)
        total = sum(p.numel() for p in base_detector.parameters())
        print(f"  Base detector: {trainable:,} trainable / {total:,} total")

        # ---- Step 4: Extract Faster R-CNN components for explicit forward control ----
        self.backbone = base_detector.backbone
        self.rpn = base_detector.rpn

        # ---- Step 5: Transform — batch-only (dataset already normalizes and resizes) ----
        # _NoOpRCNNTransform skips resize and normalize — only handles batching + padding.
        # size_divisible=32 is the standard for ResNet (stride of 32 at deepest feature level).
        self.transform = _NoOpRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            size_divisible=32,
        )

        # ---- Step 6: Configure 3D head mode ----
        base_roi_heads = base_detector.roi_heads
        if head_3d_mode == "unified":
            self.roi_heads = Mono3DRoIHeads(
                base_roi_heads,
                max_3d_proposals=max_3d_proposals,
                head_3d_version=head_3d_version,
                input_reference_size=input_reference_size,
            )
            self.head_3d_separate = None
            print(f"  ✓ 3D head mode: unified (shared FC layers), version: {head_3d_version}")
        elif head_3d_mode == "separate":
            self.roi_heads = base_roi_heads
            self.head_3d_separate = SeparateMono3DHead(
                base_roi_heads,
                max_3d_proposals=max_3d_proposals,
                head_3d_version=head_3d_version,
                input_reference_size=input_reference_size,
            )
            print(f"  ✓ 3D head mode: separate (hooked features), version: {head_3d_version}")
        else:
            raise ValueError(f"Unknown head_3d_mode: {head_3d_mode!r}. Use 'unified' or 'separate'.")

    def forward(self, images, targets=None):
        """
        Full forward pass: 2D detection + 3D box regression.

        Training: returns dict of losses (cls, box, rpn, objectness, 3d)
        Inference: returns list of detection dicts with boxes, labels, scores, boxes_3d
        """
        # ---- Stage 1: Transform (batching only) ----
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = list(images.unbind(0))

        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        # ---- Stage 2: Backbone (ResNet50 + FPN) ----
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        # ---- Stage 3: RPN ----
        proposals, proposal_losses = self.rpn(images, features, targets)

        # ---- Stage 4+5: ROI Heads + 3D (mode-dependent) ----
        if self.head_3d_mode == "unified":
            # Unified: roi_heads handles 2D + 3D in one forward call
            detections, detector_losses = self.roi_heads(
                features, proposals, images.image_sizes, targets
            )
        else:  # "separate"
            # Separate: standard ROI heads run first (hook captures features + matched_idxs)
            detections, detector_losses = self.roi_heads(
                features, proposals, images.image_sizes, targets
            )

            if self.training:
                device = features[list(features.keys())[0]].device
                loss_3d, loss_3d_raw = self.head_3d_separate.compute_training_loss(
                    targets, device
                )
                detector_losses["loss_3d"] = loss_3d
                detector_losses["loss_3d_raw"] = loss_3d_raw
            else:
                self.head_3d_separate.predict_for_detections(
                    detections, features, images.image_sizes, self.roi_heads, targets,
                )

        # ---- Combine all losses ----
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        if self.training:
            return losses
        else:
            # Post-process: clip boxes to real image boundaries (undo padding)
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections
