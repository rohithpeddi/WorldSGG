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
    "v2": "facebook/dinov2-base",  # ViT-B/14  86M   hidden_size=768
    "v2l": "facebook/dinov2-large",  # ViT-L/14  304M  hidden_size=1024
    "v3l": "facebook/dinov3-vitl16-pretrain-lvd1689m",  # hidden_size=1024
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
    x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)
    y_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)
    z_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1)

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
        self._init_weights()

    def _init_weights(self):
        """Targeted initialization for consistent initial 3D predictions.

        Without this, default Kaiming-uniform init causes 4-order-of-magnitude
        variance in the initial 3D loss across runs because:
          - depth_pred: softplus(random) can produce huge depths → huge corners
          - dim_pred:   softplus(random) can produce huge box dimensions
          - mu_pred:    negative mu → exp(-mu) exponentially amplifies L3D
        """
        # Shared FC: Xavier for ReLU-activated layer
        nn.init.xavier_uniform_(self.context_fc.weight)
        nn.init.zeros_(self.context_fc.bias)

        # depth: softplus(1.0) ≈ 1.31m — moderate initial depth
        nn.init.normal_(self.depth_pred.weight, std=0.001)
        nn.init.constant_(self.depth_pred.bias, 1.0)

        # dims: softplus(0) ≈ 0.69 — moderate initial dimensions (~0.7m)
        nn.init.normal_(self.dim_pred.weight, std=0.001)
        nn.init.zeros_(self.dim_pred.bias)

        # rotation: bias → (sin=0, cos=1) = identity rotation
        nn.init.normal_(self.rot_pred.weight, std=0.001)
        with torch.no_grad():
            self.rot_pred.bias.copy_(torch.tensor([0.0, 1.0]))

        # center offset: zero initially (no 2D offset)
        nn.init.normal_(self.center_offset_pred.weight, std=0.001)
        nn.init.zeros_(self.center_offset_pred.bias)

        # mu (uncertainty): exp(-0) = 1, no loss amplification
        nn.init.normal_(self.mu_pred.weight, std=0.001)
        nn.init.zeros_(self.mu_pred.bias)

    def forward(self, shared_features, bbox_2d, camera_intrinsics, depth_stats=None):
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
        depth = F.softplus(self.depth_pred(x)) + 1e-4  # ensure strictly positive depth
        center_offset = self.center_offset_pred(x)
        mu = self.mu_pred(x).squeeze(-1)
        corners = _compute_3d_corners(
            dims, rot_sin_cos, depth, center_offset,
            bbox_2d, camera_intrinsics[:, :2], camera_intrinsics[:, 2:],
        )
        return corners, mu


# ---------------------------------------------------------------------------
# V2 prediction layers: adds per-ROI depth stats from pre-computed depth maps
# ---------------------------------------------------------------------------
def compute_depth_stats_per_roi(depth_maps, boxes, targets, device):
    """Compute per-ROI depth statistics from pre-computed depth maps.

    Args:
        depth_maps: list of (H, W) tensors, one per image (from targets['depth_map'])
        boxes: (N, 4) concatenated proposal/detection boxes across all images
        targets: list of target dicts (to count proposals per image)
        device: torch device

    Returns:
        (N, 5) tensor: [median, mean, std, min, max] depth per ROI
    """
    if depth_maps is None or len(depth_maps) == 0:
        return torch.zeros((len(boxes), 5), device=device)

    stats_list = []
    box_idx = 0
    for i, dm in enumerate(depth_maps):
        if dm is None:
            # Count how many boxes belong to this image
            # We'll fill with zeros below
            continue
        dm = dm.to(device)
        h, w = dm.shape

    # Simpler: operate on cat boxes with image index tracking
    # Rebuild per-image splits from targets
    return _compute_depth_stats_flat(depth_maps, boxes, device)


def _compute_depth_stats_flat(depth_maps, boxes, device):
    """Compute (median, mean, std, min, max) for each box from its depth map region."""
    N = len(boxes)
    stats = torch.zeros((N, 5), device=device)
    if N == 0:
        return stats

    for i in range(N):
        if i >= len(boxes):
            break
        x1, y1, x2, y2 = boxes[i].detach().long()
        # Find which image this box belongs to — determined by caller
        # For now, we assume depth_maps[0] since this is called per-image
        dm = depth_maps if not isinstance(depth_maps, list) else depth_maps[0]
        dm = dm.to(device)
        h, w = dm.shape[-2:]

        # Clamp box coordinates to image bounds
        x1 = max(0, min(x1.item(), w - 1))
        x2 = max(x1 + 1, min(x2.item(), w))
        y1 = max(0, min(y1.item(), h - 1))
        y2 = max(y1 + 1, min(y2.item(), h))

        crop = dm[y1:y2, x1:x2].flatten()
        if len(crop) == 0:
            continue
        stats[i, 0] = crop.median()
        stats[i, 1] = crop.mean()
        stats[i, 2] = crop.std() if len(crop) > 1 else 0.0
        stats[i, 3] = crop.min()
        stats[i, 4] = crop.max()

    return stats


class _Mono3DPredictionLayersV2(nn.Module):
    """V2 3D prediction branch: adds per-ROI depth stats (5 dims) from pre-computed depth maps.

    Input: shared_features(1024) + bbox(4) + intrinsics(4) + depth_stats(5) = 1037 → 3D corners.
    Depth stats: [median, mean, std, min, max] from DepthAnything within each proposal bbox.
    """

    def __init__(self, representation_size=1024):
        super().__init__()
        # 5 extra dims for depth stats: median, mean, std, min, max
        self.context_fc = nn.Linear(representation_size + 8 + 5, 512)
        self.ln = nn.LayerNorm(512)  # V2 uses LayerNorm for better stability
        self.dim_pred = nn.Linear(512, 3)
        self.rot_pred = nn.Linear(512, 2)
        self.depth_pred = nn.Linear(512, 1)
        self.center_offset_pred = nn.Linear(512, 2)
        self.mu_pred = nn.Linear(512, 1)
        self._init_weights()

    def _init_weights(self):
        """Same targeted init as V1, extended for the larger input."""
        nn.init.xavier_uniform_(self.context_fc.weight)
        nn.init.zeros_(self.context_fc.bias)

        # depth: softplus(1.5) ≈ 1.8m — slightly deeper than V1 for indoor scenes
        nn.init.normal_(self.depth_pred.weight, std=0.001)
        nn.init.constant_(self.depth_pred.bias, 1.5)

        nn.init.normal_(self.dim_pred.weight, std=0.001)
        nn.init.zeros_(self.dim_pred.bias)

        nn.init.normal_(self.rot_pred.weight, std=0.001)
        with torch.no_grad():
            self.rot_pred.bias.copy_(torch.tensor([0.0, 1.0]))

        nn.init.normal_(self.center_offset_pred.weight, std=0.001)
        nn.init.zeros_(self.center_offset_pred.bias)

        nn.init.normal_(self.mu_pred.weight, std=0.001)
        nn.init.zeros_(self.mu_pred.bias)

    def forward(self, shared_features, bbox_2d, camera_intrinsics, depth_stats=None):
        """
        Args:
            shared_features: (N, 1024) from shared FC layers
            bbox_2d: (N, 4)
            camera_intrinsics: (N, 4) — [fx, fy, cx, cy]
            depth_stats: (N, 5) — [median, mean, std, min, max] per ROI, or None
        Returns:
            corners: (N, 8, 3), mu: (N,)
        """
        if depth_stats is None:
            depth_stats = torch.zeros((shared_features.shape[0], 5), device=shared_features.device)
        x = F.relu(self.ln(self.context_fc(
            torch.cat([shared_features, bbox_2d, camera_intrinsics, depth_stats], dim=1)
        )))
        dims = F.softplus(self.dim_pred(x))
        rot_sin_cos = F.normalize(self.rot_pred(x), p=2, dim=1)
        depth = F.softplus(self.depth_pred(x)) + 1e-4
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

    def __init__(self, base_roi_heads, representation_size=1024, max_3d_proposals=64, head_3d_version="v1"):
        super().__init__()
        self.base = base_roi_heads
        self.max_3d_proposals = max_3d_proposals
        self.head_3d_version = head_3d_version
        if head_3d_version == "v2":
            self.pred_3d = _Mono3DPredictionLayersV2(representation_size)
        else:
            self.pred_3d = _Mono3DPredictionLayers(representation_size)

    def forward(self, features, proposals, image_shapes, targets=None):
        # ----- Training: match, pool, shared FC, 2D loss + 3D loss -----
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.base.select_training_samples(proposals, targets)

        box_features = self.base.box_roi_pool(features, proposals, image_shapes)
        box_features = self.base.box_head(box_features)  # (Total, 1024)
        class_logits, box_regression = self.base.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses: Dict[str, torch.Tensor] = {}

        if self.training:
            loss_cls, loss_box = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses["loss_classifier"] = loss_cls
            losses["loss_box_reg"] = loss_box
            loss_3d, loss_3d_raw = self._compute_3d_loss(
                box_features, proposals, matched_idxs, labels, targets,
            )
            losses["loss_3d"] = loss_3d
            losses["loss_3d_raw"] = loss_3d_raw
        else:
            boxes, scores, labels_out = self.base.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes,
            )
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": labels_out[i], "scores": scores[i]})
            self._predict_3d_for_detections(result, features, image_shapes, targets)

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
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)

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
                print(
                    f"  [3D debug] img {i}: gt_3d.shape={gt_3d.shape}, non-zero={n_nonzero}/{n_gt}, pos_matches={img_pos.sum().item()}")
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
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)

        # Subsample to cap memory: ovmono3d_loss creates 5× the batch for disentangled supervision
        valid_features = pos_features[valid]
        valid_proposals = pos_proposals[valid]
        valid_gt_3d = cat_gt_3d[valid]
        valid_intr = cat_intr[valid]

        # Compute depth stats for V2 head (before subsampling)
        depth_stats = None
        if self.head_3d_version == "v2":
            depth_maps = [t.get('depth_map', None) for t in targets]
            depth_maps = [dm for dm in depth_maps if dm is not None]
            if depth_maps:
                depth_stats = _compute_depth_stats_flat(depth_maps, valid_proposals, device)
            else:
                depth_stats = torch.zeros((len(valid_features), 5), device=device)

        if len(valid_features) > self.max_3d_proposals:
            perm = torch.randperm(len(valid_features), device=device)[:self.max_3d_proposals]
            valid_features = valid_features[perm]
            valid_proposals = valid_proposals[perm]
            valid_gt_3d = valid_gt_3d[perm]
            valid_intr = valid_intr[perm]
            if depth_stats is not None:
                depth_stats = depth_stats[perm]

        pred_corners, pred_mu = self.pred_3d(valid_features, valid_proposals, valid_intr, depth_stats)
        loss_3d, _, _ = ovmono3d_loss(pred_corners.view(-1, 24), valid_gt_3d, pred_mu, use_smooth_l1=True)
        # Guard against NaN/Inf from numerical instabilities (large world coords, degenerate boxes)
        if not torch.isfinite(loss_3d):
            if _log:
                print(f"  [3D debug] → NaN/Inf detected, returning 0")
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)
        loss_3d_raw = loss_3d.detach()
        # No clamp — gradient clipping (max_grad_norm) handles stability
        if _log:
            print(f"  [3D debug] → loss_3d={loss_3d.item():.6f} (from {len(valid_features)} samples)")
        return loss_3d, loss_3d_raw

    # ----- Inference helper: 3D predictions for detected boxes -----
    def _predict_3d_for_detections(self, result, features, image_shapes, targets=None):
        det_boxes = [r["boxes"] for r in result]
        if all(len(b) == 0 for b in det_boxes):
            for r in result:
                r["boxes_3d"] = torch.empty((0, 8, 3), device=r["boxes"].device)
            return

        non_empty = [b for b in det_boxes if len(b) > 0]
        device = non_empty[0].device

        det_features = self.base.box_roi_pool(features, det_boxes, image_shapes)
        det_features = self.base.box_head(det_features)
        cat_intr = _gather_intrinsics(targets, det_boxes, device, image_shapes)
        cat_boxes = torch.cat([b for b in det_boxes if len(b) > 0], dim=0)

        # Compute depth stats for V2 head during inference
        depth_stats = None
        if self.head_3d_version == "v2" and targets is not None:
            depth_maps = [t.get('depth_map', None) for t in targets]
            depth_maps = [dm for dm in depth_maps if dm is not None]
            if depth_maps:
                depth_stats = _compute_depth_stats_flat(depth_maps, cat_boxes, device)

        with torch.no_grad():
            pred_3d, _ = self.pred_3d(det_features, cat_boxes, cat_intr, depth_stats)

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

    def __init__(self, base_roi_heads, representation_size=1024, max_3d_proposals=64, head_3d_version="v1"):
        super().__init__()
        self.base_roi_heads = base_roi_heads
        self.max_3d_proposals = max_3d_proposals
        self.head_3d_version = head_3d_version
        if head_3d_version == "v2":
            self.pred_3d = _Mono3DPredictionLayersV2(representation_size)
        else:
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
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)

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
                print(
                    f"  [3D-sep debug] img {i}: gt_3d.shape={gt_3d.shape}, non-zero={n_nonzero}/{n_gt}, pos_matches={img_pos.sum().item()}")
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
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)

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
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)
        loss_3d_raw = loss_3d.detach()
        # No clamp — gradient clipping (max_grad_norm) handles stability
        if _log:
            print(f"  [3D-sep debug] → loss_3d={loss_3d.item():.6f} (from {len(valid_features)} samples)")
        return loss_3d, loss_3d_raw

    def predict_for_detections(self, detections, features, image_shapes, base_roi_heads, targets=None):
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
        cat_intr = _gather_intrinsics(targets, det_boxes, device, image_shapes)
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
            patch_size=14,
    ):
        """
        Args:
            in_channels: Input feature channels from backbone (auto-detected)
            out_channels: Output feature channels (typically 256)
            scale_factors: List of scaling factors for pyramid levels
            top_block: Optional top block to add p6 feature
            norm: Normalization type ('BN' or 'LN')
            patch_size: Backbone patch size (used as true base stride)
        """
        super().__init__()

        # in_channels is now passed directly from base_backbone.out_channels
        # (no more hardcoded overrides per model variant)
        self.scale_factors = scale_factors
        self.top_block = top_block

        # Calculate strides from actual backbone patch_size (not hardcoded 16)
        base_stride = patch_size
        strides = [base_stride / scale for scale in scale_factors]

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

            # Sequential level naming: p2, p3, p4, p5 (stride may not be a power of 2)
            feat_name = f"p{idx + 2}"
            self._out_feature_strides[feat_name] = strides[idx]
            self._out_features.append(feat_name)

        # Add top block features (p6)
        if self.top_block is not None:
            last_level = len(scale_factors) + 1  # p5 → next is p6
            for s in range(self.top_block.num_levels):
                feat_name = f"p{last_level + s + 1}"
                self._out_feature_strides[feat_name] = strides[-1] * (2 ** (s + 1))
                self._out_features.append(feat_name)

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
      - SimpleFPN for multiscale feature pyramid
      - Faster R-CNN (RPN + ROI heads) for 2D detection
      - Configurable 3D head (unified or separate)

    head_3d_mode:
      "unified"  — 3D branch inside ROI heads, shares FC layers directly (Option 3)
      "separate" — standalone 3D head hooks into shared FC features (Option 2)

    Forward pipeline:
      images → Transform → Backbone → FPN → RPN → ROI Heads (+3D) → detections
    """

    def __init__(self, num_classes=37, pretrained=True, model="v3l", head_3d_mode="unified",
                 max_3d_proposals=64, head_3d_version="v1"):
        super().__init__()
        self.head_3d_mode = head_3d_mode
        self.head_3d_version = head_3d_version

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
            self.roi_heads = Mono3DRoIHeads(base_roi_heads, max_3d_proposals=max_3d_proposals,
                                            head_3d_version=head_3d_version)
            self.head_3d_separate = None
            print(f"  ✓ 3D head mode: unified (shared FC layers), version: {head_3d_version}")
        elif head_3d_mode == "separate":
            self.roi_heads = base_roi_heads
            self.head_3d_separate = SeparateMono3DHead(base_roi_heads, max_3d_proposals=max_3d_proposals,
                                                       head_3d_version=head_3d_version)
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
                loss_3d, loss_3d_raw = self.head_3d_separate.compute_training_loss(targets, device)
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
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
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
            patch_size=base_backbone.patch_size,  # Use actual backbone patch_size as stride
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
        # Standard FPN: one anchor size per level (matched to level's receptive field)
        # p2→32, p3→64, p4→128, p5→256, p6→512  ×  3 aspect ratios = 3 anchors/loc/level
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
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
