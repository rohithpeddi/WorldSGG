
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from huggingface_hub import login
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.roi_heads import fastrcnn_loss
from transformers import AutoModel

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(_hf_token)

# ---------------------------------------------------------------------------
# Model registry: config key → HuggingFace model ID
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    "v2":    "facebook/dinov2-base",       # ViT-B/14  86M   hidden_size=768
    "v2l":   "facebook/dinov2-large",      # ViT-L/14  304M  hidden_size=1024
    "v3l":   "facebook/dinov3-vitl16-pretrain-lvd1689m",   # hidden_size=1024
}

# Import loss
try:
    from ..losses.ovmono3d_loss import ovmono3d_loss
except ImportError:
    # For standalone testing
    from ovmono3d_loss import ovmono3d_loss


# ---------------------------------------------------------------------------
# Shared 3D corner computation (used by both head modes)
# ---------------------------------------------------------------------------
def _compute_3d_corners(dims, rot_sin_cos, depth, center_offset, bbox_2d, focal_lengths, principal_point):
    """Reconstruct 8 corners from factorized parameters via pinhole back-projection."""
    cx_2d = (bbox_2d[:, 0] + bbox_2d[:, 2]) / 2.0
    cy_2d = (bbox_2d[:, 1] + bbox_2d[:, 3]) / 2.0
    u_final = cx_2d + center_offset[:, 0]
    v_final = cy_2d + center_offset[:, 1]

    l, w, h = dims[:, 0], dims[:, 1], dims[:, 2]
    x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1)
    y_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1)
    z_corners = torch.stack([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=1)

    sin, cos = rot_sin_cos[:, 0:1], rot_sin_cos[:, 1:2]
    x_rot = x_corners * cos - y_corners * sin
    y_rot = x_corners * sin + y_corners * cos

    z_c = depth
    px, py = principal_point[:, 0:1], principal_point[:, 1:2]
    fx, fy = focal_lengths[:, 0:1], focal_lengths[:, 1:2]
    x_c = (u_final.unsqueeze(1) - px) * z_c / fx
    y_c = (v_final.unsqueeze(1) - py) * z_c / fy

    return torch.stack([x_rot + x_c, y_rot + y_c, z_corners + z_c], dim=-1)  # (N, 8, 3)


# ---------------------------------------------------------------------------
# Lightweight 3D prediction layers (shared between both modes)
# ---------------------------------------------------------------------------
class _Mono3DPredictionLayers(nn.Module):
    """Lightweight 3D prediction branch: shared_features(1024)+bbox(4)+intrinsics(4) → 3D corners."""

    def __init__(self, representation_size=1024):
        super().__init__()
        self.context_fc = nn.Linear(representation_size + 8, 512)
        self.dim_pred = nn.Linear(512, 3)
        self.rot_pred = nn.Linear(512, 2)
        self.depth_pred = nn.Linear(512, 1)
        self.center_offset_pred = nn.Linear(512, 2)
        self.mu_pred = nn.Linear(512, 1)

    def forward(self, shared_features, bbox_2d, camera_intrinsics):
        """
        Args:
            shared_features: (N, 1024) from shared FC layers
            bbox_2d: (N, 4)
            camera_intrinsics: (N, 4) — [fx, fy, cx, cy]
        Returns:
            corners: (N, 8, 3), mu: (N,)
        """
        x = F.relu(self.context_fc(torch.cat([shared_features, bbox_2d, camera_intrinsics], dim=1)))
        dims = F.softplus(self.dim_pred(x))
        rot_sin_cos = F.normalize(self.rot_pred(x), p=2, dim=1)
        depth = self.depth_pred(x)
        center_offset = self.center_offset_pred(x)
        mu = self.mu_pred(x).squeeze(-1)
        corners = _compute_3d_corners(
            dims, rot_sin_cos, depth, center_offset,
            bbox_2d, camera_intrinsics[:, :2], camera_intrinsics[:, 2:],
        )
        return corners, mu


# ---------------------------------------------------------------------------
# Helper: build per-proposal intrinsics from targets or defaults
# ---------------------------------------------------------------------------
def _gather_intrinsics(targets, proposals_or_boxes, device, image_shapes=None):
    """Build (N, 4) intrinsics tensor aligned with proposals/boxes."""
    intrinsics_list = []
    for i in range(len(proposals_or_boxes)):
        n = len(proposals_or_boxes[i])
        if n == 0:
            continue
        if targets is not None and i < len(targets):
            fl = targets[i].get('focal_lengths', torch.tensor([500.0, 500.0], device=device))
            pp = targets[i].get('principal_point', torch.tensor([0.0, 0.0], device=device))
        elif image_shapes is not None:
            h, w = image_shapes[i]
            f = float(max(h, w))
            fl = torch.tensor([f, f], device=device)
            pp = torch.tensor([w / 2.0, h / 2.0], device=device)
        else:
            fl = torch.tensor([500.0, 500.0], device=device)
            pp = torch.tensor([0.0, 0.0], device=device)
        intr = torch.cat([fl, pp], dim=0)
        intrinsics_list.append(intr.unsqueeze(0).expand(n, -1))
    return torch.cat(intrinsics_list, dim=0) if intrinsics_list else torch.empty((0, 4), device=device)


# =====================================================================
# Option 3: Unified ROI Heads (3D branch inside, sharing FC layers)
# =====================================================================
class Mono3DRoIHeads(nn.Module):
    """
    Wraps standard RoIHeads via composition and adds a 3D prediction branch
    that shares the same ROI pooling and FC layers as the 2D detector.

    Training: uses Faster R-CNN's built-in proposal sampling (512/img)
              and matched_idxs to get GT 3D for positive proposals.
    Inference: re-pools detected boxes through shared layers, predicts 3D.
    """

    def __init__(self, base_roi_heads, representation_size=1024, max_3d_proposals=64):
        super().__init__()
        self.base = base_roi_heads
        self.max_3d_proposals = max_3d_proposals
        self.pred_3d = _Mono3DPredictionLayers(representation_size)

    def forward(self, features, proposals, image_shapes, targets=None):
        # ----- Training: match, pool, shared FC, 2D loss + 3D loss -----
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.base.select_training_samples(proposals, targets)

        box_features = self.base.box_roi_pool(features, proposals, image_shapes)
        box_features = self.base.box_head(box_features)          # (Total, 1024)
        class_logits, box_regression = self.base.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses: Dict[str, torch.Tensor] = {}

        if self.training:
            loss_cls, loss_box = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses["loss_classifier"] = loss_cls
            losses["loss_box_reg"] = loss_box
            losses["loss_3d"] = self._compute_3d_loss(
                box_features, proposals, matched_idxs, labels, targets,
            )
        else:
            boxes, scores, labels_out = self.base.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes,
            )
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": labels_out[i], "scores": scores[i]})
            self._predict_3d_for_detections(result, features, image_shapes)

        return result, losses

    # ----- Training helper: 3D loss on positive proposals -----
    _debug_counter = 0

    def _compute_3d_loss(self, box_features, proposals, matched_idxs, labels, targets):
        # Force float32 — Chamfer squared-distances on world coordinates overflow float16
        with torch.amp.autocast(device_type='cuda', enabled=False):
            return self._compute_3d_loss_fp32(
                box_features.float(), proposals, matched_idxs, labels, targets,
            )

    def _compute_3d_loss_fp32(self, box_features, proposals, matched_idxs, labels, targets):
        device = box_features.device
        Mono3DRoIHeads._debug_counter += 1
        _log = (Mono3DRoIHeads._debug_counter <= 5)  # Log first 5 batches

        cat_labels = torch.cat(labels, dim=0)
        pos_mask = cat_labels > 0
        n_pos = pos_mask.sum().item()
        if _log:
            print(f"  [3D debug] total_proposals={len(cat_labels)}, positives={n_pos}")
        if not pos_mask.any():
            if _log:
                print(f"  [3D debug] → ZERO LOSS: no positive proposals")
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_features = box_features[pos_mask]
        cat_proposals = torch.cat(proposals, dim=0)
        pos_proposals = cat_proposals[pos_mask]

        # Gather GT 3D and intrinsics for positive proposals
        pos_gt_3d, pos_intr = [], []
        for i in range(len(proposals)):
            img_pos = labels[i] > 0
            if not img_pos.any():
                continue
            gt_3d = targets[i]['boxes_3d']
            if _log:
                n_gt = len(gt_3d)
                n_nonzero = (gt_3d.reshape(n_gt, -1).abs().sum(dim=1) > 1e-6).sum().item()
                print(f"  [3D debug] img {i}: gt_3d.shape={gt_3d.shape}, non-zero={n_nonzero}/{n_gt}, pos_matches={img_pos.sum().item()}")
            pos_gt_3d.append(gt_3d[matched_idxs[i][img_pos]])
            fl = targets[i].get('focal_lengths', torch.tensor([500.0, 500.0], device=device))
            pp = targets[i].get('principal_point', torch.tensor([0.0, 0.0], device=device))
            intr = torch.cat([fl, pp], dim=0)
            pos_intr.append(intr.unsqueeze(0).expand(img_pos.sum(), -1))

        cat_gt_3d = torch.cat(pos_gt_3d, dim=0)
        cat_intr = torch.cat(pos_intr, dim=0)

        # Filter zero-padded GT and extreme values (world coords can overflow float16 in Chamfer)
        gt_mag = cat_gt_3d.reshape(cat_gt_3d.shape[0], -1).abs().sum(dim=1)
        valid = (gt_mag > 1e-6) & (gt_mag < 1e6)
        n_valid = valid.sum().item()
        if _log:
            print(f"  [3D debug] matched_gt_3d={len(cat_gt_3d)}, valid(non-zero & bounded)={n_valid}")
        if not valid.any():
            if _log:
                print(f"  [3D debug] → ZERO LOSS: all matched GT 3D boxes are zero-padded")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Subsample to cap memory: ovmono3d_loss creates 5× the batch for disentangled supervision
        valid_features = pos_features[valid]
        valid_proposals = pos_proposals[valid]
        valid_gt_3d = cat_gt_3d[valid]
        valid_intr = cat_intr[valid]

        if len(valid_features) > self.max_3d_proposals:
            perm = torch.randperm(len(valid_features), device=device)[:self.max_3d_proposals]
            valid_features = valid_features[perm]
            valid_proposals = valid_proposals[perm]
            valid_gt_3d = valid_gt_3d[perm]
            valid_intr = valid_intr[perm]

        pred_corners, pred_mu = self.pred_3d(valid_features, valid_proposals, valid_intr)
        loss_3d, _, _ = ovmono3d_loss(pred_corners.view(-1, 24), valid_gt_3d, pred_mu, use_smooth_l1=True)
        # Guard against NaN/Inf from numerical instabilities (large world coords, degenerate boxes)
        if not torch.isfinite(loss_3d):
            if _log:
                print(f"  [3D debug] → NaN/Inf detected, returning 0")
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Clamp loss to prevent gradient explosion from outlier GT boxes
        loss_3d_raw = loss_3d.item()
        loss_3d = torch.clamp(loss_3d, max=10.0)
        if _log:
            print(f"  [3D debug] → loss_3d={loss_3d.item():.6f} (from {len(valid_features)} samples)")
        return loss_3d

    # ----- Inference helper: 3D predictions for detected boxes -----
    def _predict_3d_for_detections(self, result, features, image_shapes):
        det_boxes = [r["boxes"] for r in result]
        if all(len(b) == 0 for b in det_boxes):
            for r in result:
                r["boxes_3d"] = torch.empty((0, 8, 3), device=r["boxes"].device)
            return

        non_empty = [b for b in det_boxes if len(b) > 0]
        device = non_empty[0].device

        det_features = self.base.box_roi_pool(features, det_boxes, image_shapes)
        det_features = self.base.box_head(det_features)
        cat_intr = _gather_intrinsics(None, det_boxes, device, image_shapes)
        cat_boxes = torch.cat([b for b in det_boxes if len(b) > 0], dim=0)

        with torch.no_grad():
            pred_3d, _ = self.pred_3d(det_features, cat_boxes, cat_intr)

        sizes = [len(b) for b in det_boxes]
        pred_split = pred_3d.split(sizes)
        for i, r in enumerate(result):
            r["boxes_3d"] = pred_split[i].view(-1, 8, 3) if sizes[i] > 0 \
                else torch.empty((0, 8, 3), device=device)


# =====================================================================
# Option 2: Separate 3D Head (hooks into shared features)
# =====================================================================
class SeparateMono3DHead(nn.Module):
    """
    Standalone 3D head that reuses the shared FC features from ROI heads
    via a forward hook. Same prediction layers, but called outside ROI heads.
    Captures matched_idxs/labels by wrapping select_training_samples.
    """

    def __init__(self, base_roi_heads, representation_size=1024, max_3d_proposals=64):
        super().__init__()
        self.base_roi_heads = base_roi_heads
        self.max_3d_proposals = max_3d_proposals
        self.pred_3d = _Mono3DPredictionLayers(representation_size)
        self._hooked_features = None
        self._matched_idxs = None
        self._labels = None
        self._sampled_proposals = None

        # Hook on box_head to capture shared 1024-dim features
        self.base_roi_heads.box_head.register_forward_hook(self._capture_hook)

        # Wrap select_training_samples to capture matched_idxs/labels
        _original_fn = self.base_roi_heads.select_training_samples

        def _wrapped_select(proposals, targets):
            result = _original_fn(proposals, targets)
            # result = (proposals, matched_idxs, labels, regression_targets)
            self._sampled_proposals = result[0]
            self._matched_idxs = result[1]
            self._labels = result[2]
            return result

        self.base_roi_heads.select_training_samples = _wrapped_select

    def _capture_hook(self, module, input, output):
        self._hooked_features = output

    _debug_counter = 0

    def compute_training_loss(self, targets, device):
        """Compute 3D loss using captured features + matched_idxs from the most recent forward."""
        # Force float32 — Chamfer squared-distances on world coordinates overflow float16
        with torch.amp.autocast(device_type='cuda', enabled=False):
            return self._compute_training_loss_fp32(targets, device)

    def _compute_training_loss_fp32(self, targets, device):
        SeparateMono3DHead._debug_counter += 1
        _log = (SeparateMono3DHead._debug_counter <= 5)

        if self._hooked_features is None or self._labels is None:
            if _log:
                print(f"  [3D-sep debug] → ZERO LOSS: no hooked features or labels")
            return torch.tensor(0.0, device=device, requires_grad=True)

        proposals = self._sampled_proposals
        labels = self._labels
        matched_idxs = self._matched_idxs

        cat_labels = torch.cat(labels, dim=0)
        pos_mask = cat_labels > 0
        n_pos = pos_mask.sum().item()
        if _log:
            print(f"  [3D-sep debug] total_proposals={len(cat_labels)}, positives={n_pos}")
        if not pos_mask.any():
            if _log:
                print(f"  [3D-sep debug] → ZERO LOSS: no positive proposals")
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_features = self._hooked_features[pos_mask]
        cat_proposals = torch.cat(proposals, dim=0)
        pos_proposals = cat_proposals[pos_mask]

        pos_gt_3d, pos_intr = [], []
        for i in range(len(proposals)):
            img_pos = labels[i] > 0
            if not img_pos.any():
                continue
            gt_3d = targets[i]['boxes_3d']
            if _log:
                n_gt = len(gt_3d)
                n_nonzero = (gt_3d.reshape(n_gt, -1).abs().sum(dim=1) > 1e-6).sum().item()
                print(f"  [3D-sep debug] img {i}: gt_3d.shape={gt_3d.shape}, non-zero={n_nonzero}/{n_gt}, pos_matches={img_pos.sum().item()}")
            pos_gt_3d.append(gt_3d[matched_idxs[i][img_pos]])
            fl = targets[i].get('focal_lengths', torch.tensor([500.0, 500.0], device=device))
            pp = targets[i].get('principal_point', torch.tensor([0.0, 0.0], device=device))
            intr = torch.cat([fl, pp], dim=0)
            pos_intr.append(intr.unsqueeze(0).expand(img_pos.sum(), -1))

        cat_gt_3d = torch.cat(pos_gt_3d, dim=0)
        cat_intr = torch.cat(pos_intr, dim=0)

        gt_mag = cat_gt_3d.reshape(cat_gt_3d.shape[0], -1).abs().sum(dim=1)
        valid = (gt_mag > 1e-6) & (gt_mag < 1e6)
        n_valid = valid.sum().item()
        if _log:
            print(f"  [3D-sep debug] matched_gt_3d={len(cat_gt_3d)}, valid(non-zero & bounded)={n_valid}")
        if not valid.any():
            if _log:
                print(f"  [3D-sep debug] → ZERO LOSS: all matched GT 3D boxes are zero-padded")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Subsample to cap memory: ovmono3d_loss creates 5× the batch for disentangled supervision
        valid_features = pos_features[valid]
        valid_proposals = pos_proposals[valid]
        valid_gt_3d = cat_gt_3d[valid]
        valid_intr = cat_intr[valid]

        if len(valid_features) > self.max_3d_proposals:
            perm = torch.randperm(len(valid_features), device=device)[:self.max_3d_proposals]
            valid_features = valid_features[perm]
            valid_proposals = valid_proposals[perm]
            valid_gt_3d = valid_gt_3d[perm]
            valid_intr = valid_intr[perm]

        pred_corners, pred_mu = self.pred_3d(valid_features.float(), valid_proposals, valid_intr)
        loss_3d, _, _ = ovmono3d_loss(pred_corners.view(-1, 24), valid_gt_3d, pred_mu, use_smooth_l1=True)
        if not torch.isfinite(loss_3d):
            if _log:
                print(f"  [3D-sep debug] → NaN/Inf detected, returning 0")
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss_3d_raw = loss_3d.item()
        loss_3d = torch.clamp(loss_3d, max=10.0)
        if loss_3d_raw > 10.0:
            print(f"  [3D-sep clamp] raw={loss_3d_raw:.4f} → clamped=10.0")
        if _log:
            print(f"  [3D-sep debug] → loss_3d={loss_3d.item():.6f} (from {len(valid_features)} samples)")
        return loss_3d

    def predict_for_detections(self, detections, features, image_shapes, base_roi_heads):
        """Run 3D prediction on detected boxes using shared ROI pool + FC."""
        det_boxes = [d["boxes"] for d in detections]
        if all(len(b) == 0 for b in det_boxes):
            for d in detections:
                d["boxes_3d"] = torch.empty((0, 8, 3), device=d["boxes"].device)
            return

        non_empty = [b for b in det_boxes if len(b) > 0]
        device = non_empty[0].device

        det_feats = base_roi_heads.box_roi_pool(features, det_boxes, image_shapes)
        det_feats = base_roi_heads.box_head(det_feats)
        cat_intr = _gather_intrinsics(None, det_boxes, device, image_shapes)
        cat_boxes = torch.cat([b for b in det_boxes if len(b) > 0], dim=0)

        with torch.no_grad():
            pred_3d, _ = self.pred_3d(det_feats, cat_boxes, cat_intr)

        sizes = [len(b) for b in det_boxes]
        pred_split = pred_3d.split(sizes)
        for i, d in enumerate(detections):
            d["boxes_3d"] = pred_split[i].view(-1, 8, 3) if sizes[i] > 0 \
                else torch.empty((0, 8, 3), device=device)


class LastLevelMaxPool(nn.Module):
    """Pooling to create p6 feature map (for larger object detection)."""

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [nn.functional.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class SimpleFeaturePyramid(nn.Module):
    """
    Simple Feature Pyramid Network (SimpleFPN) adapted from ViTDet.
    Creates multiscale pyramid features from single-scale backbone output.
    """

    def __init__(
            self,
            in_channels=768,
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            top_block=None,
            norm="BN",
    ):
        """
        Args:
            in_channels: Input feature channels from backbone (auto-detected)
            out_channels: Output feature channels (typically 256)
            scale_factors: List of scaling factors for pyramid levels
            top_block: Optional top block to add p6 feature
            norm: Normalization type ('BN' or 'LN')
        """
        super().__init__()

        # in_channels is now passed directly from base_backbone.out_channels
        # (no more hardcoded overrides per model variant)
        self.scale_factors = scale_factors
        self.top_block = top_block

        # Calculate strides (assuming patch_size=16, base stride=16)
        base_stride = 16
        strides = [int(base_stride / scale) for scale in scale_factors]

        # Create pyramid stages
        self.stages = nn.ModuleList()
        self._out_feature_strides = {}
        self._out_features = []

        for idx, scale in enumerate(scale_factors):
            out_dim = in_channels
            layers = []

            # Upsample/downsample layers
            if scale == 4.0:
                # 4x upsampling: 2x transpose convs
                layers.extend([
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(in_channels // 2) if norm == "BN" else nn.GroupNorm(1, in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2, bias=False),
                ])
                out_dim = in_channels // 4
            elif scale == 2.0:
                # 2x upsampling
                layers.extend(
                    [nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False), ])
                out_dim = in_channels // 2
            elif scale == 1.0:
                # No scaling
                pass
            elif scale == 0.5:
                # 2x downsampling
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported")

            # Channel reduction and refinement
            use_bias = norm == ""
            layers.extend([
                nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=use_bias),
                nn.BatchNorm2d(out_channels) if norm == "BN" else nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
                nn.BatchNorm2d(out_channels) if norm == "BN" else nn.GroupNorm(1, out_channels),
            ])

            stage = nn.Sequential(*layers)
            self.stages.append(stage)

            # Feature map naming: p2, p3, p4, p5 (stride = 2^stage)
            stage_num = int(torch.log2(torch.tensor(strides[idx])).item())
            feat_name = f"p{stage_num}"
            self._out_feature_strides[feat_name] = strides[idx]
            self._out_features.append(feat_name)

        # Add top block features (p6)
        if self.top_block is not None:
            last_stage = int(torch.log2(torch.tensor(strides[-1])).item())
            for s in range(last_stage, last_stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)
                self._out_features.append(f"p{s + 1}")

        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self.out_channels = out_channels

    def forward(self, features) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Dict of backbone features {'0': tensor, '1': tensor, ...}

        Returns:
            Dict of pyramid features {'p2': tensor, 'p3': tensor, ...}
        """
        # Use the finest feature map (typically '0' or '1') as base
        # For DINOv2, we'll use the first feature map
        base_feature = features  # [B, C, H, W]

        results = []
        for stage in self.stages:
            results.append(stage(base_feature))

        # Add top block features (p6)
        if self.top_block is not None:
            top_block_in = results[-1]  # Use last pyramid feature
            top_results = self.top_block(top_block_in)
            results.extend(top_results)

        # Create output dict
        output_features = {f: res for f, res in zip(self._out_features, results)}
        return output_features


class Dinov3ModelBackbone(nn.Module):

    def __init__(self, model: str = "v2"):
        """
        Args:
            model: Key into MODEL_REGISTRY (e.g. 'v2', 'v2s', 'v2l', 'v3l').
                   Defaults to 'v2' (DINOv2-Base, 86M params — fastest option).
        """
        super().__init__()
        if model in MODEL_REGISTRY:
            model_name = MODEL_REGISTRY[model]
        else:
            # Allow direct HuggingFace model IDs for flexibility
            model_name = model
        self.model_name = model_name
        self.bck_model = AutoModel.from_pretrained(self.model_name)
        self.bck_model.eval()
        # Read actual hidden_size and patch_size from model config
        self.out_channels = self.bck_model.config.hidden_size
        self.patch_size = getattr(self.bck_model.config, 'patch_size', 14)
        print(f"  Backbone: {model_name}  hidden_size={self.out_channels}  patch_size={self.patch_size}")

    def forward(self, x):
        # x shape is (B, 3, Img_H, Img_W)
        B, _, Img_H, Img_W = x.shape
        H = Img_H // self.patch_size
        W = Img_W // self.patch_size
        n_patches = H * W

        # Use autocast for aggressive AMP — backbone is frozen, so float16/bfloat16 is safe
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = self.bck_model(x)
        features = outputs.last_hidden_state  # (B, N_total, C)
        C = features.shape[-1]

        # Strip CLS + any register tokens: keep only the last n_patches spatial tokens.
        features = features[:, -n_patches:, :]  # (B, H*W, C)

        features = features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return features

class _NoOpRCNNTransform(GeneralizedRCNNTransform):
    """GeneralizedRCNNTransform that ONLY batches — no resize, no normalization.
    The dataset already produces normalized images at the correct Pi3-compatible
    resolution (multiples of patch_size), so any further transforms are pure overhead.
    """

    def resize(self, image, target):
        return image, target

    def normalize(self, image):
        # Skip normalization — already done in dataset __getitem__
        return image


class DinoV3Monocular3D(nn.Module):
    """
    Full Monocular 3D Object Detector combining:
      - DINOv2 frozen backbone (ViT) for feature extraction
      - SimpleFPN for multi-scale feature pyramid
      - Faster R-CNN (RPN + ROI heads) for 2D detection
      - Configurable 3D head (unified or separate)

    head_3d_mode:
      "unified"  — 3D branch inside ROI heads, shares FC layers directly (Option 3)
      "separate" — standalone 3D head hooks into shared FC features (Option 2)

    Forward pipeline:
      images → Transform → Backbone → FPN → RPN → ROI Heads (+3D) → detections
    """

    def __init__(self, num_classes=37, pretrained=True, model="v3l", head_3d_mode="unified", max_3d_proposals=64):
        super().__init__()
        self.head_3d_mode = head_3d_mode

        # Create the base Faster R-CNN detector with DINOv2 backbone + FPN
        self.base_detector = create_model(num_classes=num_classes, pretrained=pretrained, use_fpn=True, model=model)

        # Extract Faster R-CNN components for explicit forward control
        self.backbone = self.base_detector.backbone
        self.rpn = self.base_detector.rpn

        # Transform: batching ONLY — no resize, no normalization.
        _ps = self.backbone.base_backbone.patch_size if hasattr(self.backbone, 'base_backbone') else 14
        self.transform = _NoOpRCNNTransform(
            min_size=800, max_size=1333,
            image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0],
            size_divisible=_ps,
        )

        # Configure 3D head mode
        base_roi_heads = self.base_detector.roi_heads
        if head_3d_mode == "unified":
            # Option 3: 3D branch lives INSIDE ROI heads (shares pool + FC)
            self.roi_heads = Mono3DRoIHeads(base_roi_heads, max_3d_proposals=max_3d_proposals)
            self.head_3d_separate = None
            print(f"  ✓ 3D head mode: unified (shared FC layers)")
        elif head_3d_mode == "separate":
            # Option 2: standard ROI heads + separate 3D head with hook
            self.roi_heads = base_roi_heads
            self.head_3d_separate = SeparateMono3DHead(base_roi_heads, max_3d_proposals=max_3d_proposals)
            print(f"  ✓ 3D head mode: separate (hooked shared features)")
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

        # ---- Stage 2: Backbone (frozen DINOv2 ViT → FPN) ----
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        # ---- Stage 3: RPN ----
        proposals, proposal_losses = self.rpn(images, features, targets)

        # ---- Stage 4+5: ROI Heads + 3D (mode-dependent) ----
        if self.head_3d_mode == "unified":
            # Unified: roi_heads handles 2D + 3D in one forward call
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        else:  # "separate"
            # Separate: standard ROI heads run first (hook captures features + matched_idxs)
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

            if self.training:
                device = features[list(features.keys())[0]].device
                detector_losses["loss_3d"] = self.head_3d_separate.compute_training_loss(targets, device)
            else:
                self.head_3d_separate.predict_for_detections(
                    detections, features, images.image_sizes, self.roi_heads,
                )

        # ---- Combine all losses ----
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        if self.training:
            return losses
        else:
            return detections


def create_model(num_classes=37, pretrained=True, coco_model=False, use_fpn=True, model="v2"):
    """
    Factory function to build the full Faster R-CNN detector.

    Args:
        num_classes: Number of object classes (including background)
        pretrained: Load pretrained DINOv2 weights from HuggingFace
        use_fpn: Wrap backbone with SimpleFPN (recommended)
        model: Key from MODEL_REGISTRY ('v2', 'v2s', 'v2l', 'v3l', etc.)

    Returns:
        FasterRCNN model with DINOv2 backbone
    """
    # Create base backbone — model key selects from MODEL_REGISTRY
    print(f"  Creating backbone: model={model}")
    base_backbone = Dinov3ModelBackbone(model=model)

    # Freeze backbone (keep adapter trainable)
    for name, params in base_backbone.named_parameters():
        if 'adapter' not in name:
            params.requires_grad_(False)

    # Wrap with SimpleFeaturePyramid
    if use_fpn:
        backbone = SimpleFeaturePyramid(
            in_channels=base_backbone.out_channels,  # auto-detected from backbone config
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),  # Creates p2, p3, p4, p5
            top_block=LastLevelMaxPool(),  # Adds p6
            norm="BN",
        )

        class BackboneWithFPN(nn.Module):
            def __init__(self, base_backbone, fpn):
                super().__init__()
                self.base_backbone = base_backbone
                self.fpn = fpn
                self.out_channels = fpn.out_channels

            def forward(self, x):
                with torch.no_grad():
                    features = self.base_backbone(x)
                return self.fpn(features)

        backbone = BackboneWithFPN(base_backbone, backbone)
        print("✅ Using SimpleFeaturePyramid (FPN)")
    else:
        backbone = base_backbone
        print("Using backbone without FPN")

    # Anchor generator
    if use_fpn:
        featmap_names = backbone.fpn._out_features
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128, 256, 512, 1024),) * len(featmap_names),
            aspect_ratios=((0.5, 1.0, 2.0),) * len(featmap_names)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=7,
            sampling_ratio=2
        )
    else:
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),) * 4,
            aspect_ratios=((0.5, 1.0, 2.0),) * 4
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model
