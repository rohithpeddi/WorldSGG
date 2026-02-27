"""
WSGG Shared Architectural Components
======================================

All reusable modules shared across WSGG methods live here.
Method-specific modules (memory banks, tokenizers, retrievers) stay
in their respective directories.

Components:
  1. GlobalStructuralEncoder — PointNet 3D bbox → tokens
  2. SpatialPositionalEncoding — 3D geometry-aware PE
  3. SpatialGNN — Transformer encoder with spatial PE
  4. NodePredictor — Object class MLP
  5. EdgePredictor — Relationship MLP with spatial features
  6. DINOFeatureExtractor — Frozen detector wrapper
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Global Structural Encoder
# ============================================================================

class GlobalStructuralEncoder(nn.Module):
    """
    Encodes world-frame 3D bounding boxes into per-object structural tokens
    and a global summary token using a PointNet-style architecture.

    Input:  corners (B, N, 8, 3), valid_mask (B, N)
    Output: object_tokens (B, N, d_struct), global_token (B, d_struct)
    """

    def __init__(self, d_struct: int = 256, d_hidden: int = 128):
        super().__init__()
        self.d_struct = d_struct
        self.corner_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_hidden),
            nn.ReLU(inplace=True),
        )
        self.object_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden * 2),
            nn.Linear(d_hidden * 2, d_struct),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(d_struct, d_struct),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_struct),
            nn.Linear(d_struct, d_struct),
        )

    def forward(self, corners: torch.Tensor, valid_mask: torch.Tensor) -> tuple:
        B, N, C, D = corners.shape
        assert C == 8 and D == 3

        x = corners.reshape(B * N * C, D)
        x = self.corner_mlp(x)
        x = x.view(B, N, C, -1)

        mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1)
        x = x.masked_fill(~mask_expanded, float("-inf"))
        x, _ = x.max(dim=2)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        object_tokens = self.object_mlp(x)
        object_tokens = object_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        global_pool = object_tokens.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
        global_pool, _ = global_pool.max(dim=1)
        all_invalid = ~valid_mask.any(dim=1, keepdim=True)
        global_pool = global_pool.masked_fill(all_invalid, 0.0)
        global_token = self.global_mlp(global_pool)

        return object_tokens, global_token


# ============================================================================
# 2. Spatial Positional Encoding
# ============================================================================

class SpatialPositionalEncoding(nn.Module):
    """
    3D geometry-aware positional encodings from object bounding boxes.
    Computes pairwise spatial features (distance, direction, volume ratio)
    and aggregates into per-object encodings.

    Input:  corners (N, 8, 3), valid_mask (N,)
    Output: spatial_pe (N, d_model)
    """

    def __init__(self, d_model: int = 256, d_hidden: int = 64):
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(5, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Linear(d_hidden, d_model)

    def forward(self, corners: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        N = corners.shape[0]

        centers = corners.mean(dim=1)
        mins, _ = corners.min(dim=1)
        maxs, _ = corners.max(dim=1)
        extents = (maxs - mins).clamp(min=1e-6)
        volumes = extents.prod(dim=-1)
        log_volumes = torch.log(volumes + 1e-6)

        diff = centers.unsqueeze(1) - centers.unsqueeze(0)
        dist = diff.norm(dim=-1, keepdim=True)
        direction = diff / (dist + 1e-6)
        log_vol_ratio = (log_volumes.unsqueeze(1) - log_volumes.unsqueeze(0)).unsqueeze(-1)

        pair_feats = torch.cat([dist, direction, log_vol_ratio], dim=-1)
        pair_encoded = self.pair_mlp(pair_feats)

        pair_valid = valid_mask.unsqueeze(0) & valid_mask.unsqueeze(1)
        pair_encoded = pair_encoded * pair_valid.unsqueeze(-1).float()

        n_valid = pair_valid.float().sum(dim=1, keepdim=True).clamp(min=1)
        agg = pair_encoded.sum(dim=1) / n_valid

        return self.out_proj(agg)


# ============================================================================
# 3. Spatial GNN
# ============================================================================

class SpatialGNN(nn.Module):
    """
    Transformer encoder with 3D spatial positional encoding.
    Stateless context propagation based on geometric proximity.

    Input:  tokens (N, d_model), corners (N, 8, 3), valid_mask (N,)
    Output: enriched (N, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_pe = SpatialPositionalEncoding(d_model=d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        spatial_enc = self.spatial_pe(corners, valid_mask)
        x = tokens + spatial_enc
        x = self.pre_norm(x)
        x = x.unsqueeze(0)
        padding_mask = ~valid_mask.unsqueeze(0)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x.squeeze(0)
        x = self.post_norm(x)
        x = x * valid_mask.unsqueeze(-1).float()
        return x


# ============================================================================
# 4. Node Predictor
# ============================================================================

class NodePredictor(nn.Module):
    """
    Object class prediction from enriched representations.

    Input:  (N, d_memory)
    Output: (N, num_classes) logits
    """

    def __init__(self, d_memory: int, num_classes: int, d_hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_memory, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(memory_states)


# ============================================================================
# 5. Edge Predictor
# ============================================================================

class EdgePredictor(nn.Module):
    """
    Relationship prediction for person-object pairs.
    Concatenates representations + relative 3D spatial features.

    Output: attention (K, 3), spatial (K, 6), contacting (K, 17)
    """

    def __init__(
        self,
        d_memory: int,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num

        self.spatial_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_hidden),
        )

        pair_input_dim = d_memory * 2 + d_hidden
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, d_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
        )

        self.a_rel_compress = nn.Linear(d_hidden, attention_class_num)
        self.s_rel_compress = nn.Linear(d_hidden, spatial_class_num)
        self.c_rel_compress = nn.Linear(d_hidden, contact_class_num)

    def compute_pair_spatial(
        self, person_corners: torch.Tensor, object_corners: torch.Tensor,
    ) -> torch.Tensor:
        p_center = person_corners.mean(dim=1)
        o_center = object_corners.mean(dim=1)
        rel_center = o_center - p_center
        dist = rel_center.norm(dim=-1, keepdim=True)

        p_mins, _ = person_corners.min(dim=1)
        p_maxs, _ = person_corners.max(dim=1)
        p_vol = (p_maxs - p_mins).clamp(min=1e-6).prod(dim=-1)

        o_mins, _ = object_corners.min(dim=1)
        o_maxs, _ = object_corners.max(dim=1)
        o_vol = (o_maxs - o_mins).clamp(min=1e-6).prod(dim=-1)

        log_vol_ratio = (torch.log(o_vol + 1e-6) - torch.log(p_vol + 1e-6)).unsqueeze(-1)
        return torch.cat([rel_center, dist, log_vol_ratio], dim=-1)

    def forward(
        self,
        enriched_states: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
        corners: torch.Tensor,
    ) -> dict:
        K = person_idx.shape[0]
        if K == 0:
            device = enriched_states.device
            return {
                "attention_distribution": torch.zeros(0, self.attention_class_num, device=device),
                "spatial_distribution": torch.zeros(0, self.spatial_class_num, device=device),
                "contacting_distribution": torch.zeros(0, self.contact_class_num, device=device),
            }

        person_repr = enriched_states[person_idx]
        object_repr = enriched_states[object_idx]

        spatial_feats = self.compute_pair_spatial(corners[person_idx], corners[object_idx])
        spatial_encoded = self.spatial_encoder(spatial_feats)

        pair_input = torch.cat([person_repr, object_repr, spatial_encoded], dim=-1)
        pair_features = self.pair_mlp(pair_input)

        return {
            "attention_distribution": self.a_rel_compress(pair_features),
            "spatial_distribution": torch.sigmoid(self.s_rel_compress(pair_features)),
            "contacting_distribution": torch.sigmoid(self.c_rel_compress(pair_features)),
        }


# ============================================================================
# 6. DINO Feature Extractor
# ============================================================================

class DINOFeatureExtractor(nn.Module):
    """
    Frozen DINO detector wrapper for visual feature extraction.
    Lazy-loads the detector to avoid import issues.

    Input: images + boxes
    Output: (N, 1024) ROI features
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
        """Lazy-load and freeze the detector."""
        if self._detector is not None:
            return

        from lib.detector.monocular3d.models.dino_mono_3d import DinoV3Monocular3D

        self._detector = DinoV3Monocular3D(
            num_classes=self._num_classes,
            pretrained=True,
            model=self._detector_model,
            head_3d_mode="unified",
        )

        if self._detector_ckpt:
            state = torch.load(self._detector_ckpt, map_location="cpu")
            if "model_state_dict" in state:
                self._detector.load_state_dict(state["model_state_dict"], strict=False)
            elif "state_dict" in state:
                self._detector.load_state_dict(state["state_dict"], strict=False)
            else:
                self._detector.load_state_dict(state, strict=False)
            print(f"[DINOFeatureExtractor] Loaded detector from: {self._detector_ckpt}")

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
        """Extract ROI features from the frozen detector."""
        self.load_detector()
        detector = self._detector
        detector.eval()

        if isinstance(images, torch.Tensor) and images.dim() == 4:
            img_list = list(images.unbind(0))
        else:
            img_list = [img.to(self.device) for img in images]

        original_image_sizes = [img.shape[-2:] for img in img_list]
        images_transformed, targets = detector.transform(img_list, None)
        features = detector.backbone(images_transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, _ = detector.rpn(images_transformed, features, None)
        if gt_boxes is not None:
            proposals = [b.to(self.device) for b in gt_boxes]

        roi_heads = detector.roi_heads
        if hasattr(roi_heads, 'base'):
            base_roi = roi_heads.base
        else:
            base_roi = roi_heads

        box_features = base_roi.box_roi_pool(features, proposals, images_transformed.image_sizes)
        box_features = base_roi.box_head(box_features)

        class_logits, box_regression = base_roi.box_predictor(box_features)
        boxes_list, scores_list, labels_list = base_roi.postprocess_detections(
            class_logits, box_regression, proposals, images_transformed.image_sizes,
        )

        det_boxes = boxes_list
        splits = [len(b) for b in det_boxes]

        if sum(splits) > 0:
            det_features = base_roi.box_roi_pool(features, det_boxes, images_transformed.image_sizes)
            det_features = base_roi.box_head(det_features)
        else:
            det_features = torch.zeros(0, 1024, device=self.device)

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
        """Extract ROI features for specific 2D boxes."""
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

        box_features = base_roi.box_roi_pool(features, box_list, images_transformed.image_sizes)
        box_features = base_roi.box_head(box_features)
        return box_features

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use extract_features() or extract_features_for_boxes()")


# ============================================================================
# 7. Physics Veto — Deterministic 3D Geometric Filter
# ============================================================================

class PhysicsVeto(nn.Module):
    """
    Non-differentiable geometric filter that vetoes VLM pseudo-labels
    violating basic spatial physics.

    Rules:
      1. If VLM predicts a contact predicate (touching/holding/sitting_on/etc.)
         but the 3D centroids of subject and object are > dist_thresh apart → veto
      2. If VLM predicts [inside] but no bounding box containment → veto

    Args:
        dist_thresh: Meters — max distance for contact predicates.
        contact_predicate_indices: Set of predicate indices that require proximity
            (touching, holding, sitting_on, lying_on, carrying, etc.).
        inside_predicate_idx: Index of the "inside" predicate, or None.
    """

    def __init__(
        self,
        dist_thresh: float = 2.0,
        contact_predicate_indices: set = None,
        inside_predicate_idx: int = None,
    ):
        super().__init__()
        self.dist_thresh = dist_thresh
        # Default contact predicates (common Action Genome indices)
        # These should be overridden with actual dataset indices
        self.contact_predicate_indices = contact_predicate_indices or set()
        self.inside_predicate_idx = inside_predicate_idx

    @torch.no_grad()
    def compute_veto_mask(
        self,
        corners: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
        pred_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-edge veto mask.

        Args:
            corners: (N, 8, 3) — world-frame 3D bbox corners.
            person_idx: (K,) long — person indices.
            object_idx: (K,) long — object indices.
            pred_labels: (K,) long or (K, C) float — predicted labels/logits.

        Returns:
            keep_mask: (K,) bool — True = keep, False = veto.
        """
        K = person_idx.shape[0]
        if K == 0:
            return torch.ones(0, dtype=torch.bool, device=corners.device)

        # Compute centroids from corners: mean of 8 corner points
        centroids = corners.mean(dim=1)  # (N, 3)
        p_centroids = centroids[person_idx]  # (K, 3)
        o_centroids = centroids[object_idx]  # (K, 3)
        dists = torch.norm(p_centroids - o_centroids, dim=-1)  # (K,)

        keep_mask = torch.ones(K, dtype=torch.bool, device=corners.device)

        # Get predicted class indices
        if pred_labels.dim() == 2:
            # pred_labels is logits → get argmax
            pred_cls = pred_labels.argmax(dim=-1)  # (K,)
        else:
            pred_cls = pred_labels  # (K,) already indices

        # Rule 1: veto contact predicates if distance > threshold
        if self.contact_predicate_indices:
            for idx in self.contact_predicate_indices:
                is_contact = (pred_cls == idx)
                too_far = (dists > self.dist_thresh)
                keep_mask = keep_mask & ~(is_contact & too_far)

        # Rule 2: veto "inside" if no containment
        if self.inside_predicate_idx is not None:
            is_inside = (pred_cls == self.inside_predicate_idx)
            # Simple containment check: object centroid within person bbox bounds
            p_mins = corners[person_idx].min(dim=1).values  # (K, 3)
            p_maxs = corners[person_idx].max(dim=1).values  # (K, 3)
            o_cent = o_centroids  # (K, 3)
            inside_bounds = (
                (o_cent >= p_mins).all(dim=-1) &
                (o_cent <= p_maxs).all(dim=-1)
            )
            keep_mask = keep_mask & ~(is_inside & ~inside_bounds)

        return keep_mask


# ============================================================================
# 8. Label Smoother — Soft Targets for VLM Pseudo-Labels
# ============================================================================

class LabelSmoother:
    """
    Apply label smoothing to VLM pseudo-labels.

    Tells the model: "The VLM strongly suspects this is [holding], but it might be wrong."

    For CE targets (single-label):
        1-hot [0, 0, 1, 0] → [ε/3, ε/3, 1-ε, ε/3]

    For BCE targets (multi-label):
        multi-hot [0, 1, 0, 1] → [ε/(C-k), 1-ε, ε/(C-k), 1-ε]
        where k = number of active labels

    Args:
        epsilon: Smoothing factor (0.0 = no smoothing, 1.0 = uniform).
    """

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon

    def smooth_ce_target(self, target_idx: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Convert hard class index to smoothed distribution.

        Args:
            target_idx: (K,) long — hard class targets.
            num_classes: int — total classes.

        Returns:
            (K, num_classes) float — smoothed probability distribution.
        """
        K = target_idx.shape[0]
        device = target_idx.device
        smooth = torch.full((K, num_classes), self.epsilon / num_classes, device=device)
        smooth.scatter_(1, target_idx.unsqueeze(1), 1.0 - self.epsilon + self.epsilon / num_classes)
        return smooth

    def smooth_bce_target(self, target_multihot: torch.Tensor) -> torch.Tensor:
        """
        Smooth multi-hot BCE targets.

        Active labels: 1.0 → (1.0 - epsilon)
        Inactive labels: 0.0 → epsilon / (C - k)
        where k = number of active labels per row.

        Args:
            target_multihot: (K, C) float — multi-hot target.

        Returns:
            (K, C) float — smoothed target.
        """
        K, C = target_multihot.shape
        k = target_multihot.sum(dim=1, keepdim=True).clamp(min=1)  # (K, 1)

        smoothed = target_multihot.clone()
        # Active labels: reduce confidence
        smoothed[target_multihot == 1.0] = 1.0 - self.epsilon
        # Inactive labels: small positive
        inactive_val = self.epsilon / (C - k)  # (K, 1) broadcast
        inactive_mask = (target_multihot == 0.0)
        smoothed[inactive_mask] = inactive_val.expand_as(target_multihot)[inactive_mask]

        return smoothed

