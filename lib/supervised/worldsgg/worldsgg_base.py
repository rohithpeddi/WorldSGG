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
  5. EdgePredictor — Relationship MLP with spatial + union features
  6. CameraPoseEncoder — Camera extrinsic → viewpoint tokens
  7. CameraTemporalEncoder — Ego-motion between consecutive frames
  8. PhysicsVeto — Deterministic 3D geometric filter
  9. LabelSmoother — Soft targets for VLM pseudo-labels
  10. ObservabilityClassifier — Per-object observability state from pose
  12. MotionFeatureEncoder — 3D velocity/acceleration → d_motion tokens
  13. FeatureAging — Staleness + pose-delta aware feature blending
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

    Concatenates:
      - Person representation (d_memory)
      - Object representation (d_memory)
      - 3D spatial features: rel_center(3) + dist(1) + log_vol_ratio(1) = 5
      - 2D spatial features: IoU(1) + rel_center_2d(2) + log_area_ratio(1) = 4
      - Union features: projected union ROI features (d_hidden), or [NO_UNION]

    Output: attention (K, 3), spatial (K, 6), contacting (K, 17)

    Args:
        d_memory: Enriched representation dimension.
        attention_class_num: Number of attention relationship classes.
        spatial_class_num: Number of spatial relationship classes.
        contact_class_num: Number of contacting relationship classes.
        d_hidden: Hidden dimension in prediction MLPs.
        d_union_roi: Raw union ROI feature dimension (1024 from DINO).
    """

    def __init__(
        self,
        d_memory: int,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
        d_hidden: int = 256,
        d_union_roi: int = 1024,
    ):
        super().__init__()
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.d_hidden = d_hidden

        # 3D spatial: 5 dims (rel_center + dist + log_vol_ratio)
        # 2D spatial: 4 dims (IoU + rel_center_2d + log_area_ratio)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_hidden),
        )

        # Union feature projector (1024 → d_hidden)
        self.union_projector = nn.Sequential(
            nn.Linear(d_union_roi, d_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden),
        )

        # Learnable fallback for pairs without union features (unseen objects)
        self.no_union_embedding = nn.Parameter(torch.randn(d_hidden) * 0.02)

        # pair_input = person(d_memory) + object(d_memory) + spatial(d_hidden) + union(d_hidden)
        pair_input_dim = d_memory * 2 + d_hidden * 2
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

    def _compute_2d_iou(
        self, boxes_a: torch.Tensor, boxes_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IoU between paired 2D boxes.

        Args:
            boxes_a: (K, 4) xyxy format.
            boxes_b: (K, 4) xyxy format.

        Returns:
            iou: (K, 1)
        """
        x1 = torch.max(boxes_a[:, 0], boxes_b[:, 0])
        y1 = torch.max(boxes_a[:, 1], boxes_b[:, 1])
        x2 = torch.min(boxes_a[:, 2], boxes_b[:, 2])
        y2 = torch.min(boxes_a[:, 3], boxes_b[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=1e-6) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=1e-6)
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=1e-6) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=1e-6)
        union = area_a + area_b - inter
        iou = inter / union.clamp(min=1e-6)
        return iou.unsqueeze(-1)  # (K, 1)

    def compute_pair_spatial(
        self,
        person_corners: torch.Tensor,
        object_corners: torch.Tensor,
        person_bboxes_2d: Optional[torch.Tensor] = None,
        object_bboxes_2d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined 3D + 2D spatial features for person-object pairs.

        Args:
            person_corners: (K, 8, 3) person 3D bbox corners.
            object_corners: (K, 8, 3) object 3D bbox corners.
            person_bboxes_2d: (K, 4) xyxy or None.
            object_bboxes_2d: (K, 4) xyxy or None.

        Returns:
            spatial_feats: (K, 9) — [3D(5) + 2D(4)]
        """
        K = person_corners.shape[0]
        device = person_corners.device

        # --- 3D spatial features (existing) ---
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
        feats_3d = torch.cat([rel_center, dist, log_vol_ratio], dim=-1)  # (K, 5)

        # --- 2D spatial features ---
        if person_bboxes_2d is not None and object_bboxes_2d is not None:
            # IoU
            iou_2d = self._compute_2d_iou(person_bboxes_2d, object_bboxes_2d)  # (K, 1)

            # Relative center in 2D (normalized by person bbox size)
            p_cx = (person_bboxes_2d[:, 0] + person_bboxes_2d[:, 2]) / 2
            p_cy = (person_bboxes_2d[:, 1] + person_bboxes_2d[:, 3]) / 2
            o_cx = (object_bboxes_2d[:, 0] + object_bboxes_2d[:, 2]) / 2
            o_cy = (object_bboxes_2d[:, 1] + object_bboxes_2d[:, 3]) / 2
            p_w = (person_bboxes_2d[:, 2] - person_bboxes_2d[:, 0]).clamp(min=1e-6)
            p_h = (person_bboxes_2d[:, 3] - person_bboxes_2d[:, 1]).clamp(min=1e-6)
            rel_cx = ((o_cx - p_cx) / p_w).unsqueeze(-1)
            rel_cy = ((o_cy - p_cy) / p_h).unsqueeze(-1)

            # Log area ratio
            p_area = p_w * p_h
            o_w = (object_bboxes_2d[:, 2] - object_bboxes_2d[:, 0]).clamp(min=1e-6)
            o_h = (object_bboxes_2d[:, 3] - object_bboxes_2d[:, 1]).clamp(min=1e-6)
            o_area = o_w * o_h
            log_area_ratio = (torch.log(o_area + 1e-6) - torch.log(p_area + 1e-6)).unsqueeze(-1)

            feats_2d = torch.cat([iou_2d, rel_cx, rel_cy, log_area_ratio], dim=-1)  # (K, 4)
        else:
            feats_2d = torch.zeros(K, 4, device=device)

        return torch.cat([feats_3d, feats_2d], dim=-1)  # (K, 9)

    def forward(
        self,
        enriched_states: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
        corners: torch.Tensor,
        union_features: Optional[torch.Tensor] = None,
        bboxes_2d: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            enriched_states: (N, d_memory) context-enriched representations.
            person_idx: (K,) long — person indices in each pair.
            object_idx: (K,) long — object indices in each pair.
            corners: (N, 8, 3) 3D bbox corners.
            union_features: (K, d_union_roi) or None — union ROI features per pair.
            bboxes_2d: (N, 4) or None — 2D bounding boxes in xyxy format.

        Returns:
            dict with attention/spatial/contacting distributions.
        """
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

        # Spatial features (3D + 2D)
        person_bboxes_2d = bboxes_2d[person_idx] if bboxes_2d is not None else None
        object_bboxes_2d = bboxes_2d[object_idx] if bboxes_2d is not None else None
        spatial_feats = self.compute_pair_spatial(
            corners[person_idx], corners[object_idx],
            person_bboxes_2d, object_bboxes_2d,
        )
        spatial_encoded = self.spatial_encoder(spatial_feats)

        # Union features
        if union_features is not None:
            union_proj = self.union_projector(union_features)  # (K, d_hidden)
        else:
            union_proj = self.no_union_embedding.unsqueeze(0).expand(K, -1)  # (K, d_hidden)

        pair_input = torch.cat([person_repr, object_repr, spatial_encoded, union_proj], dim=-1)
        pair_features = self.pair_mlp(pair_input)

        return {
            "attention_distribution": self.a_rel_compress(pair_features),
            "spatial_distribution": torch.sigmoid(self.s_rel_compress(pair_features)),
            "contacting_distribution": torch.sigmoid(self.c_rel_compress(pair_features)),
        }

# ============================================================================
# 6. Camera Pose Encoder
# ============================================================================

def _rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D representation (Zhou et al.).

    Takes the first two columns of R, which provides a continuous and
    unique representation suitable for neural network learning.

    Args:
        R: (..., 3, 3) rotation matrix.

    Returns:
        (..., 6) 6D rotation representation.
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


class CameraPoseEncoder(nn.Module):
    """
    Encodes camera extrinsic (4×4 pose matrix) into:
      1. A global camera token (d_camera) — summarizes current viewpoint
      2. Per-object camera-relative features (N, d_camera) — how each
         object relates to the current camera position and viewing direction

    The camera pose [R|t] is decomposed as:
      - 6D rotation representation (continuous, from first 2 cols of R)
      - 3D translation (camera position in world)
      → 9-dim input → MLP → d_camera global token

    Per-object features encode:
      - Distance from camera to object center
      - Dot product of viewing direction with camera→object direction
      - Azimuth angle of object relative to camera optical axis
      → 3-dim per-object → MLP → d_camera per-object features

    Args:
        d_camera: Output dimension for camera tokens.
        d_hidden: Hidden dimension in the encoder MLPs.
    """

    def __init__(self, d_camera: int = 128, d_hidden: int = 64):
        super().__init__()
        self.d_camera = d_camera

        # Global camera token: 6D rot + 3D translation = 9
        self.global_mlp = nn.Sequential(
            nn.Linear(9, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_camera),
        )

        # Per-object camera-relative features:
        #   distance(1) + view_alignment(1) + azimuth_sin(1) + azimuth_cos(1) = 4
        self.per_object_mlp = nn.Sequential(
            nn.Linear(4, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_camera),
        )

    def forward(
        self,
        camera_pose: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple:
        """
        Args:
            camera_pose: (4, 4) — camera-to-world extrinsic matrix.
            corners: (N, 8, 3) — 3D bbox corners for all objects.
            valid_mask: (N,) bool — True for real objects.

        Returns:
            camera_token: (d_camera,) — global viewpoint summary.
            per_object_cam_feats: (N, d_camera) — camera-relative per-object features.
        """
        N = corners.shape[0]
        device = corners.device

        # --- Extract rotation and translation from pose ---
        R = camera_pose[:3, :3]  # (3, 3)
        t = camera_pose[:3, 3]   # (3,) — camera position in world

        # --- Global camera token ---
        rot_6d = _rotation_matrix_to_6d(R)  # (6,)
        global_input = torch.cat([rot_6d, t], dim=-1)  # (9,)
        camera_token = self.global_mlp(global_input)  # (d_camera,)

        # --- Per-object camera-relative features ---
        # Object centers
        centers = corners.mean(dim=1)  # (N, 3)

        # Camera viewing direction (negative Z axis of camera in world frame)
        # R columns are world-frame axes of the camera
        view_dir = -R[:, 2]  # (3,) — optical axis direction in world
        view_dir = view_dir / (view_dir.norm() + 1e-8)

        # Vector from camera to each object
        cam_to_obj = centers - t.unsqueeze(0)  # (N, 3)
        dist = cam_to_obj.norm(dim=-1, keepdim=True)  # (N, 1)
        cam_to_obj_norm = cam_to_obj / (dist + 1e-8)  # (N, 3)

        # View alignment: dot product with viewing direction
        # +1 = directly in front, -1 = directly behind
        view_alignment = (cam_to_obj_norm * view_dir.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (N, 1)

        # Azimuth angle (sin, cos) for rotational equivariance
        # Project cam_to_obj onto the camera's XZ plane
        right_dir = R[:, 0]  # camera right axis in world
        right_component = (cam_to_obj_norm * right_dir.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (N, 1)
        azimuth_sin = right_component
        azimuth_cos = view_alignment

        # Normalize distance with log for better gradient behavior
        log_dist = torch.log(dist + 1e-6)  # (N, 1)

        per_obj_input = torch.cat([
            log_dist, view_alignment, azimuth_sin, azimuth_cos,
        ], dim=-1)  # (N, 4)

        per_object_cam_feats = self.per_object_mlp(per_obj_input)  # (N, d_camera)

        # Zero out padding
        per_object_cam_feats = per_object_cam_feats * valid_mask.unsqueeze(-1).float()

        return camera_token, per_object_cam_feats


# ============================================================================
# 8. Camera Temporal Encoder (Ego-Motion)
# ============================================================================

class CameraTemporalEncoder(nn.Module):
    """
    Encodes the CHANGE between consecutive camera poses (ego-motion).

    Captures: camera panned left, tilted up, moved forward, etc.
    Critical for temporal methods — tells the memory update how much
    the viewpoint changed since the last frame, informing which objects
    transitioned visible→unseen or unseen→visible.

    Input:  prev_pose (4,4), curr_pose (4,4)
    Output: ego_motion_token (d_camera)

    Args:
        d_camera: Output dimension for ego-motion token.
        d_hidden: Hidden dimension in the encoder MLP.
    """

    def __init__(self, d_camera: int = 128, d_hidden: int = 64):
        super().__init__()
        self.d_camera = d_camera

        # Relative pose: 6D rotation + 3D translation = 9
        self.ego_mlp = nn.Sequential(
            nn.Linear(9, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_camera),
        )

        # Learnable "no previous frame" embedding (for t=0)
        self.no_prev_embedding = nn.Parameter(torch.randn(d_camera) * 0.02)

    def forward(
        self,
        prev_pose: torch.Tensor,
        curr_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_pose: (4, 4) or None — previous frame's camera pose.
                       If None, returns the learnable no_prev_embedding.
            curr_pose: (4, 4) — current frame's camera pose.

        Returns:
            ego_motion_token: (d_camera,) — encodes how the camera moved.
        """
        if prev_pose is None:
            return self.no_prev_embedding

        # Compute relative pose: T_rel = T_curr @ T_prev^{-1}
        # This gives the transformation FROM prev camera TO current camera
        T_rel = curr_pose @ torch.inverse(prev_pose)  # (4, 4)

        R_rel = T_rel[:3, :3]  # (3, 3)
        t_rel = T_rel[:3, 3]   # (3,)

        rot_6d = _rotation_matrix_to_6d(R_rel)  # (6,)
        ego_input = torch.cat([rot_6d, t_rel], dim=-1)  # (9,)

        return self.ego_mlp(ego_input)  # (d_camera,)


# ============================================================================
# 9. Physics Veto — Deterministic 3D Geometric Filter
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


# ============================================================================
# 11. Observability Classifier
# ============================================================================

class ObservabilityClassifier(nn.Module):
    """
    Non-differentiable classifier that categorizes each object's observability
    state based on camera pose, visibility mask, and history.

    States:
        0 = NEVER_SEEN:      Object has never had a GT/GDino bbox
        1 = OUT_OF_FRUSTUM:  Object's center is behind/beside camera
        2 = OCCLUDED:        Object is in camera frustum but not visible
                             (blocked by another object)
        3 = VISIBLE:         Object has a detection this frame

    This classification is critical because:
        - OUT_OF_FRUSTUM: no visual evidence at all → rely on geometry+dynamics
        - OCCLUDED: contextual cues exist (nearby objects) → intermediate reliability
        - NEVER_SEEN: the model has zero prior appearance info → geometry only

    Args:
        frustum_thresh: View-alignment threshold below which an object
                        is considered out-of-frustum. Default -0.1 means
                        objects slightly behind the camera still count as
                        "in frustum" (generous margin).
    """

    # Class constants for readability
    NEVER_SEEN = 0
    OUT_OF_FRUSTUM = 1
    OCCLUDED = 2
    VISIBLE = 3

    def __init__(self, frustum_thresh: float = -0.1):
        super().__init__()
        self.frustum_thresh = frustum_thresh

    @torch.no_grad()
    def forward(
        self,
        camera_pose: torch.Tensor,
        corners: torch.Tensor,
        visibility_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        ever_seen_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classify each object's observability state.

        Args:
            camera_pose: (4, 4) — camera extrinsic matrix.
            corners: (N, 8, 3) — 3D bbox corners.
            visibility_mask: (N,) bool — True if object is detected this frame.
            valid_mask: (N,) bool — True for real objects.
            ever_seen_mask: (N,) bool or None — True if object was ever visible.
                           If None, all valid objects are treated as "ever seen."

        Returns:
            obs_type: (N,) long — per-object observability state (0-3).
        """
        N = corners.shape[0]
        device = corners.device

        # Extract camera position and viewing direction
        R = camera_pose[:3, :3]  # (3, 3)
        cam_pos = camera_pose[:3, 3]  # (3,)
        view_dir = -R[:, 2]  # optical axis in world frame
        view_dir = view_dir / (view_dir.norm() + 1e-8)

        # Object centers
        centers = corners.mean(dim=1)  # (N, 3)

        # View alignment: dot(cam→object, view_dir)
        cam_to_obj = centers - cam_pos.unsqueeze(0)  # (N, 3)
        cam_to_obj_norm = cam_to_obj / (cam_to_obj.norm(dim=-1, keepdim=True) + 1e-8)
        view_alignment = (cam_to_obj_norm * view_dir.unsqueeze(0)).sum(dim=-1)  # (N,)

        # In-frustum: view_alignment > threshold
        in_frustum = view_alignment > self.frustum_thresh  # (N,)

        # Ever-seen mask
        if ever_seen_mask is None:
            ever_seen_mask = valid_mask  # assume all valid objects were seen before

        # Classify
        obs_type = torch.zeros(N, dtype=torch.long, device=device)
        # Default: NEVER_SEEN (0) — for objects that were never detected

        # Objects that have been seen before but are currently not visible
        seen_but_invisible = ever_seen_mask & (~visibility_mask) & valid_mask
        obs_type[seen_but_invisible & (~in_frustum)] = ObservabilityClassifier.OUT_OF_FRUSTUM
        obs_type[seen_but_invisible & in_frustum] = ObservabilityClassifier.OCCLUDED

        # Currently visible
        obs_type[visibility_mask & valid_mask] = ObservabilityClassifier.VISIBLE

        # Zero out invalid objects
        obs_type[~valid_mask] = 0

        return obs_type

    @staticmethod
    def to_onehot(obs_type: torch.Tensor, num_states: int = 4) -> torch.Tensor:
        """
        Convert obs_type indices to one-hot encoding.

        Args:
            obs_type: (N,) long — observability states.
            num_states: Number of states (4).

        Returns:
            (N, num_states) float — one-hot encoding.
        """
        return F.one_hot(obs_type, num_classes=num_states).float()


# ============================================================================
# 12. Motion Feature Encoder
# ============================================================================

class MotionFeatureEncoder(nn.Module):
    """
    Encodes 3D motion features (velocity, acceleration) into per-object tokens.

    Uses finite-difference on 3D bbox centers from consecutive frames.
    Optionally computes camera-relative velocity for parallax-aware motion.

    Input:  velocity (N, 3), acceleration (N, 3), camera_R (3, 3) [optional]
    Output: motion_features (N, d_motion)

    Args:
        d_motion: Output motion feature dimension.
        d_hidden: Hidden dimension in the encoder MLP.
        use_cam_relative: Whether to include camera-relative velocity.
    """

    def __init__(
        self,
        d_motion: int = 64,
        d_hidden: int = 32,
        use_cam_relative: bool = True,
    ):
        super().__init__()
        self.d_motion = d_motion
        self.use_cam_relative = use_cam_relative

        # Input: velocity(3) + acceleration(3) + speed_scalar(1)
        #   + optionally cam_relative_velocity(3) + cam_rel_speed(1)
        input_dim = 7  # vel(3) + accel(3) + speed(1)
        if use_cam_relative:
            input_dim += 4  # cam_vel(3) + cam_speed(1)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_motion),
        )

        # Learnable "no motion" embedding for the first frame
        self.no_motion_embedding = nn.Parameter(torch.randn(d_motion) * 0.02)

    def forward(
        self,
        velocity: Optional[torch.Tensor] = None,
        acceleration: Optional[torch.Tensor] = None,
        camera_R: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            velocity: (N, 3) — world-frame velocity (center_t - center_{t-1}).
                      None for the first frame.
            acceleration: (N, 3) — world-frame acceleration (vel_t - vel_{t-1}).
                          None for the first two frames.
            camera_R: (3, 3) — camera rotation matrix. Used to compute
                      camera-relative velocity if use_cam_relative=True.
            valid_mask: (N,) bool — valid objects.

        Returns:
            motion_features: (N, d_motion) — per-object motion tokens.
        """
        if velocity is None:
            # First frame: return learnable "no motion" for all objects
            if valid_mask is not None:
                N = valid_mask.shape[0]
            else:
                return self.no_motion_embedding.unsqueeze(0)  # (1, d_motion)
            return self.no_motion_embedding.unsqueeze(0).expand(N, -1)  # (N, d_motion)

        N = velocity.shape[0]
        device = velocity.device

        # Speed scalar (magnitude of velocity)
        speed = velocity.norm(dim=-1, keepdim=True)  # (N, 1)

        # Acceleration (zeros if not available, e.g., second frame)
        if acceleration is None:
            acceleration = torch.zeros_like(velocity)  # (N, 3)

        # Base features
        feats = [velocity, acceleration, speed]  # 3 + 3 + 1 = 7

        # Camera-relative velocity
        if self.use_cam_relative and camera_R is not None:
            # Transform velocity to camera coordinates: v_cam = R^T @ v_world
            cam_vel = (camera_R.T @ velocity.T).T  # (N, 3)
            cam_speed = cam_vel.norm(dim=-1, keepdim=True)  # (N, 1)
            feats.extend([cam_vel, cam_speed])  # +3 +1 = 4
        elif self.use_cam_relative:
            # No camera rotation available — pad with zeros
            feats.extend([
                torch.zeros(N, 3, device=device),
                torch.zeros(N, 1, device=device),
            ])

        motion_input = torch.cat(feats, dim=-1)  # (N, 7 or 11)
        motion_features = self.mlp(motion_input)  # (N, d_motion)

        # Zero out invalid objects
        if valid_mask is not None:
            motion_features = motion_features * valid_mask.unsqueeze(-1).float()

        return motion_features

    @staticmethod
    def compute_velocity(
        curr_corners: torch.Tensor,
        prev_corners: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-object velocity from consecutive corner sets.

        Args:
            curr_corners: (N, 8, 3) — current frame corners.
            prev_corners: (N, 8, 3) — previous frame corners.

        Returns:
            velocity: (N, 3) — displacement of center between frames.
        """
        curr_centers = curr_corners.mean(dim=1)  # (N, 3)
        prev_centers = prev_corners.mean(dim=1)  # (N, 3)
        return curr_centers - prev_centers


# ============================================================================
# 13. Feature Aging
# ============================================================================

class FeatureAging(nn.Module):
    """
    Learned feature aging that blends stale visual features toward
    class prototypes based on staleness and camera pose delta.

    Key insight: DINO features are robust but NOT viewpoint-invariant.
    A feature captured from the front may be misleading when the camera
    is now behind the object. As staleness increases or the camera moves
    far from the capture pose, appearance should relax toward a
    view-independent class prototype.

    confidence = sigmoid( aging_mlp([log_staleness, pose_delta]) )
    aged_feat = confidence * stale_feat + (1-confidence) * prototype

    Args:
        d_visual: Visual feature dimension.
        n_classes: Number of object classes (for class prototypes).
    """

    def __init__(self, d_visual: int = 256, n_classes: int = 37):
        super().__init__()
        self.d_visual = d_visual

        # Learnable class prototype embeddings
        self.class_prototypes = nn.Embedding(n_classes, d_visual)

        # Aging function: [log_staleness(1), pose_delta(1)] → confidence scalar
        self.aging_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        stale_features: torch.Tensor,
        staleness: torch.Tensor,
        pose_delta: torch.Tensor,
        class_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply aging to stale features.

        Args:
            stale_features: (N, d_visual) — buffered visual features.
            staleness: (N,) long — frames since last capture.
            pose_delta: (N,) float — magnitude of camera pose change since capture.
            class_indices: (N,) long — object class indices (for prototype lookup).
            valid_mask: (N,) bool — valid objects.

        Returns:
            aged_features: (N, d_visual) — blended features.
        """
        N = stale_features.shape[0]

        # Log-scale staleness for better gradient behavior
        log_staleness = torch.log(staleness.float() + 1.0).unsqueeze(-1)  # (N, 1)
        pose_delta_input = pose_delta.unsqueeze(-1)  # (N, 1)

        aging_input = torch.cat([log_staleness, pose_delta_input], dim=-1)  # (N, 2)
        confidence = self.aging_mlp(aging_input)  # (N, 1)

        # Class prototypes
        prototypes = self.class_prototypes(class_indices)  # (N, d_visual)

        # Blend: high confidence → keep stale; low confidence → use prototype
        aged = confidence * stale_features + (1.0 - confidence) * prototypes

        if valid_mask is not None:
            aged = aged * valid_mask.unsqueeze(-1).float()

        return aged

    @staticmethod
    def compute_pose_delta(
        current_pose: torch.Tensor,
        capture_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scalar pose delta between current and capture poses.

        Combines rotation difference (Frobenius norm of R_diff - I)
        and translation distance.

        Args:
            current_pose: (4, 4) — current camera pose.
            capture_pose: (N, 4, 4) — per-object capture poses.

        Returns:
            pose_delta: (N,) — scalar pose delta per object.
        """
        N = capture_pose.shape[0]

        R_curr = current_pose[:3, :3]  # (3, 3)
        t_curr = current_pose[:3, 3]  # (3,)

        R_cap = capture_pose[:, :3, :3]  # (N, 3, 3)
        t_cap = capture_pose[:, :3, 3]  # (N, 3)

        # Rotation delta: Frobenius norm of (R_curr @ R_cap^T - I)
        R_diff = R_curr.unsqueeze(0) @ R_cap.transpose(-1, -2)  # (N, 3, 3)
        eye = torch.eye(3, device=R_diff.device).unsqueeze(0)  # (1, 3, 3)
        rot_delta = (R_diff - eye).flatten(start_dim=1).norm(dim=-1)  # (N,)

        # Translation delta
        trans_delta = (t_curr.unsqueeze(0) - t_cap).norm(dim=-1)  # (N,)

        # Combined (both contribute)
        return rot_delta + trans_delta
