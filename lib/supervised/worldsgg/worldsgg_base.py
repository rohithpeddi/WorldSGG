"""
WSGG Shared Architectural Components
======================================

All reusable modules shared across WSGG methods live here.
Method-specific modules (memory banks, tokenizers, retrievers) stay
in their respective directories.

Components:
  1. GlobalStructuralEncoder — Flattened local-centered 3D bbox → tokens
  2. SpatialPositionalEncoding — 3D geometry-aware PE
  3. SpatialGNN — Transformer encoder with spatial PE
  4. NodePredictor — Object class MLP
  5. RelationshipPredictor — Unified edge prediction (visual+union+CLIP→self-attn→3 heads)
  5b. TemporalEdgeAttention — Cross-temporal edge reasoning
  6. CameraPoseEncoder — Camera extrinsic → viewpoint tokens
  7. CameraTemporalEncoder — Ego-motion between consecutive frames
  8. LabelSmoother — Soft targets for VLM pseudo-labels
  9. ObservabilityClassifier — Per-object observability state from pose
  10. MotionFeatureEncoder — 3D velocity/acceleration → d_motion tokens
  11. FeatureAging — Staleness + pose-delta aware feature blending
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
    and a global summary token using flattened local-centered coordinates.

    Each bounding box's 8 corners are centered (translation-invariant local
    geometry) and flattened into a 24-dim vector, then concatenated with the
    3D absolute center to form a 27-dim input. This preserves the full rigid
    geometry (dimensions + orientation) that per-corner max-pooling discards.

    The global max-pool over N objects remains appropriate because objects
    in a scene ARE an unordered, variable-size set.

    Input:  corners (B, N, 8, 3), valid_mask (B, N)
    Output: object_tokens (B, N, d_struct), global_token (B, d_struct)
    """

    def __init__(self, d_struct: int = 256, d_hidden: int = 128):
        super().__init__()
        self.d_struct = d_struct

        # Input: 24 local corner coords (8*3) + 3 absolute center = 27
        self.object_mlp = nn.Sequential(
            nn.Linear(27, d_hidden),
            nn.ReLU(inplace=True),
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

        # 1. Extract local geometry (translation-invariant box shape)
        centers = corners.mean(dim=2, keepdim=True)               # (B, N, 1, 3)
        local_corners = corners - centers                         # (B, N, 8, 3)

        # 2. Flatten 8 local corners and concatenate absolute center
        local_flat = local_corners.view(B, N, 24)                 # (B, N, 24)
        x = torch.cat([local_flat, centers.squeeze(2)], dim=-1)   # (B, N, 27)

        # 3. Encode full bounding box geometry at once
        object_tokens = self.object_mlp(x)                        # (B, N, d_struct)
        object_tokens = object_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        # 4. Global pool over N objects (permutation-invariant set aggregation)
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

    Input:  corners (B, N, 8, 3), valid_mask (B, N)
    Output: spatial_pe (B, N, d_model)
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
        # 1. Handle Batch Dimension Properly
        B, N, C, D = corners.shape
        assert C == 8 and D == 3, "Expected corners shape (B, N, 8, 3)"

        # dim=2 corresponds to the 8 corners
        centers = corners.mean(dim=2)  # (B, N, 3)

        # Exact OBB volume from edge vectors (not AABB min/max which overestimates
        # for rotated boxes). Corner ordering: 0-3 bottom face perimeter, 4-7 top
        # face directly above (i.e., corner 4 is above corner 0).
        edge_a = corners[:, :, 1] - corners[:, :, 0]   # (B, N, 3) perimeter edge
        edge_b = corners[:, :, 3] - corners[:, :, 0]   # (B, N, 3) perimeter edge
        edge_c = corners[:, :, 4] - corners[:, :, 0]   # (B, N, 3) vertical edge
        len_a = torch.sqrt((edge_a ** 2).sum(dim=-1) + 1e-6)  # (B, N)
        len_b = torch.sqrt((edge_b ** 2).sum(dim=-1) + 1e-6)  # (B, N)
        len_c = torch.sqrt((edge_c ** 2).sum(dim=-1) + 1e-6)  # (B, N)
        volumes = len_a * len_b * len_c                        # (B, N)
        log_volumes = torch.log(volumes + 1e-6)                # (B, N)

        # 2. Pairwise Relationships (Broadcasting over B and N)
        # diff shape: (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N, 3)
        # points FROM neighbor 'j' TO target 'i'
        diff = centers.unsqueeze(2) - centers.unsqueeze(1)
        
        # CRITICAL FIX: Safe distance calculation (prevents NaN gradients)
        dist = torch.sqrt((diff ** 2).sum(dim=-1, keepdim=True) + 1e-6) # (B, N, N, 1)
        direction = diff / dist                                         # (B, N, N, 3)
        
        # Log volume ratio: log(Vi) - log(Vj)
        log_vol_ratio = (log_volumes.unsqueeze(2) - log_volumes.unsqueeze(1)).unsqueeze(-1) # (B, N, N, 1)

        # 3. Feed through MLP
        pair_feats = torch.cat([dist, direction, log_vol_ratio], dim=-1) # (B, N, N, 5)
        pair_encoded = self.pair_mlp(pair_feats)                         # (B, N, N, d_hidden)

        # 4. Masking Out Invalid Object Pairs
        # pair_valid shape: (B, N, 1) & (B, 1, N) -> (B, N, N)
        pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        pair_encoded = pair_encoded * pair_valid.unsqueeze(-1).float()

        # 5. Aggregation
        # Sum over the neighbor dimension (dim=2, or 'j')
        n_valid = pair_valid.float().sum(dim=2, keepdim=True).clamp(min=1) # (B, N, 1)
        agg = pair_encoded.sum(dim=2) / n_valid                            # (B, N, d_hidden)

        # 6. Final Projection & Post-Masking
        spatial_pe = self.out_proj(agg)                                    # (B, N, d_model)
        
        # STRICT MASKING: Ensure invalid objects don't output the linear layer bias
        spatial_pe = spatial_pe.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        return spatial_pe


# ============================================================================
# 3. Spatial GNN
# ============================================================================

class SpatialGNN(nn.Module):
    """
    Transformer encoder with 3D spatial positional encoding.
    Context propagation based on geometric proximity.

    Accepts batched (B, N, ...) inputs natively.
    Invalid (padding) objects are excluded via src_key_padding_mask and
    strictly zeroed out in the output.

    Args:
        d_model: Token / output dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        d_feedforward: FFN hidden dimension.
        dropout: Dropout probability.

    Input:  tokens (B, N, d_model), corners (B, N, 8, 3), valid_mask (B, N)
    Output: enriched (B, N, d_model)
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # Final LayerNorm required for pre-norm architecture
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Spatial Positional Encoding (natively batched)
        spatial_enc = self.spatial_pe(corners, valid_mask)  # (B, N, d_model)
        x = tokens + spatial_enc

        # 2. Padding mask: True = ignore for PyTorch transformer
        padding_mask = ~valid_mask  # (B, N)

        # 3. Failsafe: if an entire batch item is fully padded, MHA returns NaN.
        #    Temporarily unmask the first token of that sequence.
        all_invalid = padding_mask.all(dim=1)  # (B,)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        # 4. Global attention / context propagation
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 5. Strict zero-masking (FFN biases + residual leak into padded tokens)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

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
# 5. Relationship Predictor — Unified Edge Prediction Pipeline
# ============================================================================

class RelationshipPredictor(nn.Module):
    """
    Unified relationship prediction pipeline.

    Phase 2: Form (K, d_rel) relationship tokens by concatenating:
       - person visual features (d_model)
       - object visual features (d_model)
       - union ROI features (d_union, projected from d_union_roi)
       - CLIP text features for person + object labels (d_text × 2)

    Phase 3: Self-attention across all K relationship tokens.

    Phase 5: 3 simple MLP heads → att/spa/con logits.

    Phase 4 (temporal edge attention) is applied externally by each method
    between Phase 3 and Phase 5 via the decomposed interface:
        rel_tokens = predictor.form_rel_tokens(...)      # Phase 2
        rel_tokens = predictor.self_attend(rel_tokens)    # Phase 3
        rel_tokens = temporal_edge_attn(rel_tokens, ...)  # Phase 4 (external)
        edge_out   = predictor.predict_from_tokens(...)   # Phase 5

    Args:
        d_model: Per-object enriched token dimension.
        d_text: CLIP text projection dimension.
        d_rel: Relationship token dimension.
        d_union_roi: Raw union ROI feature dimension (e.g. 1024 from DINO).
        attention_class_num: Output size for attention head.
        spatial_class_num: Output size for spatial head.
        contact_class_num: Output size for contacting head.
        clip_embeddings_path: Path to precomputed CLIP .npy file.
        n_rel_layers: Number of self-attention layers.
        n_rel_heads: Number of self-attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_text: int = 128,
        d_rel: int = 256,
        d_union_roi: int = 1024,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
        clip_embeddings_path: str = "",
        n_rel_layers: int = 2,
        n_rel_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_rel = d_rel
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num

        # --- CLIP text embeddings (frozen, precomputed) ---
        if clip_embeddings_path:
            import numpy as np
            emb = np.load(clip_embeddings_path)  # (C, 512)
            self.register_buffer('clip_embeddings', torch.from_numpy(emb).float())
            d_clip = emb.shape[1]  # 512
        else:
            # Fallback: random init for testing without CLIP file
            self.register_buffer('clip_embeddings', torch.randn(37, 512))
            d_clip = 512
        self.text_proj = nn.Linear(d_clip, d_text)

        # --- Union feature projector ---
        d_union = d_rel // 4
        self.union_proj = nn.Sequential(
            nn.Linear(d_union_roi, d_union),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_union),
        )
        self.no_union_embedding = nn.Parameter(torch.randn(d_union) * 0.02)

        # --- Input projection: concat → d_rel ---
        # person(d_model) + object(d_model) + union(d_union) + person_text(d_text) + object_text(d_text)
        d_input = d_model * 2 + d_union + d_text * 2
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_rel),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_rel),
        )

        # --- Phase 3: Relationship self-attention ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_rel,
            nhead=n_rel_heads,
            dim_feedforward=d_rel * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.rel_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_rel_layers,
            norm=nn.LayerNorm(d_rel),
        )

        # --- Phase 5: 3 simple MLP heads ---
        self.att_head = nn.Sequential(
            nn.Linear(d_rel, d_rel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_rel // 2, attention_class_num),
        )
        self.spa_head = nn.Sequential(
            nn.Linear(d_rel, d_rel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_rel // 2, spatial_class_num),
        )
        self.con_head = nn.Sequential(
            nn.Linear(d_rel, d_rel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_rel // 2, contact_class_num),
        )

    def _get_text_features(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Look up precomputed CLIP embeddings and project.

        Args:
            class_indices: (K,) long — class indices.

        Returns:
            text_feat: (K, d_text)
        """
        # Clamp to valid range
        idx = class_indices.clamp(0, self.clip_embeddings.shape[0] - 1)
        raw = self.clip_embeddings[idx]  # (K, 512)
        return self.text_proj(raw)       # (K, d_text)

    def form_rel_tokens(
        self,
        enriched_states: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
        person_class_idx: torch.Tensor,
        object_class_idx: torch.Tensor,
        union_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Phase 2: Form relationship tokens.

        Args:
            enriched_states: (N, d_model) context-enriched object representations.
            person_idx: (K,) long — person indices.
            object_idx: (K,) long — object indices.
            person_class_idx: (K,) long — predicted class for each person.
            object_class_idx: (K,) long — predicted class for each object.
            union_features: (K, d_union_roi) or None.

        Returns:
            rel_tokens: (K, d_rel)
        """
        K = person_idx.shape[0]

        # Visual features
        person_vis = enriched_states[person_idx]   # (K, d_model)
        object_vis = enriched_states[object_idx]   # (K, d_model)

        # Union features
        if union_features is not None and union_features.shape[0] > 0:
            union_feat = self.union_proj(union_features)  # (K, d_union)
        else:
            union_feat = self.no_union_embedding.unsqueeze(0).expand(K, -1)

        # CLIP text features
        person_text = self._get_text_features(person_class_idx)  # (K, d_text)
        object_text = self._get_text_features(object_class_idx)  # (K, d_text)

        # Concatenate and project
        concat = torch.cat([person_vis, object_vis, union_feat,
                            person_text, object_text], dim=-1)
        return self.input_proj(concat)  # (K, d_rel)

    def self_attend(self, rel_tokens: torch.Tensor) -> torch.Tensor:
        """
        Phase 3: Relationship self-attention.

        Args:
            rel_tokens: (K, d_rel)

        Returns:
            attended: (K, d_rel)
        """
        if rel_tokens.shape[0] == 0:
            return rel_tokens
        # Add batch dim → (1, K, d_rel), attend, squeeze
        return self.rel_transformer(rel_tokens.unsqueeze(0)).squeeze(0)

    def predict_from_tokens(self, rel_tokens: torch.Tensor) -> dict:
        """
        Phase 5: 3 simple MLP heads.

        Args:
            rel_tokens: (K, d_rel) — final relationship representations.

        Returns:
            dict with attention/spatial/contacting distributions.
        """
        return {
            "attention_distribution": self.att_head(rel_tokens),
            "spatial_distribution": torch.sigmoid(self.spa_head(rel_tokens)),
            "contacting_distribution": torch.sigmoid(self.con_head(rel_tokens)),
        }

    def forward(
        self,
        enriched_states: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
        person_class_idx: torch.Tensor,
        object_class_idx: torch.Tensor,
        union_features: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full Phase 2 → 3 → 5 pipeline (no temporal).

        Returns:
            dict with:
                attention_distribution: (K, att_classes)
                spatial_distribution: (K, spa_classes)
                contacting_distribution: (K, con_classes)
                rel_tokens: (K, d_rel) — for external temporal attention
        """
        K = person_idx.shape[0]
        if K == 0:
            device = enriched_states.device
            return {
                "attention_distribution": torch.zeros(0, self.attention_class_num, device=device),
                "spatial_distribution": torch.zeros(0, self.spatial_class_num, device=device),
                "contacting_distribution": torch.zeros(0, self.contact_class_num, device=device),
                "rel_tokens": torch.zeros(0, self.d_rel, device=device),
            }

        # Phase 2
        rel_tokens = self.form_rel_tokens(
            enriched_states, person_idx, object_idx,
            person_class_idx, object_class_idx, union_features,
        )
        # Phase 3
        rel_tokens = self.self_attend(rel_tokens)
        # Phase 5
        preds = self.predict_from_tokens(rel_tokens)
        preds["rel_tokens"] = rel_tokens
        return preds


# ============================================================================
# 5b. Temporal Edge Attention — Cross-Temporal Relationship Reasoning
# ============================================================================

class TemporalEdgeAttention(nn.Module):
    """
    Cross-attention from current-frame relationship tokens to
    previous frames' cached relationship tokens.

    Allows temporal consistency: "person was holding cup last frame"
    informs current frame's edge predictions.

    Args:
        d_rel: Relationship token dimension.
        n_heads: Number of attention heads.
        n_layers: Number of decoder layers.
        dropout: Dropout probability.

    Input:
        current_rel: (K_t, d_rel) — current frame relationship tokens.
        prev_rels: (K_prev, d_rel) — cached from prior frame(s).
    Output:
        temporally_enriched: (K_t, d_rel)
    """

    def __init__(
        self,
        d_rel: int = 256,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_rel,
            nhead=n_heads,
            dim_feedforward=d_rel * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_rel),
        )

    def forward(
        self,
        current_rel: torch.Tensor,
        prev_rels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_rel: (K_t, d_rel) — current frame edges.
            prev_rels: (K_prev, d_rel) — previous frame edges.

        Returns:
            enriched: (K_t, d_rel) — temporally-informed edges.
        """
        if current_rel.shape[0] == 0 or prev_rels.shape[0] == 0:
            return current_rel

        # Add batch dim: (1, K, d_rel)
        out = self.decoder(
            current_rel.unsqueeze(0),
            prev_rels.unsqueeze(0),
        ).squeeze(0)
        return out



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
