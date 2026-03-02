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
  7. CameraTemporalEncoder — Full-temporal ego-motion self-attention
  8. LabelSmoother — Soft targets for VLM pseudo-labels
  9. MotionFeatureEncoder — 3D velocity/acceleration → d_motion tokens
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
            class_indices: (...) long — class indices (any shape).

        Returns:
            text_feat: (..., d_text)
        """
        # Clamp to valid range
        idx = class_indices.clamp(0, self.clip_embeddings.shape[0] - 1)
        raw = self.clip_embeddings[idx]  # (..., 512)
        return self.text_proj(raw)       # (..., d_text)

    def batched_form_and_attend(
        self,
        enriched_all: torch.Tensor,
        node_logits_all: torch.Tensor,
        person_idx_seq: List[torch.Tensor],
        object_idx_seq: List[torch.Tensor],
        union_features_seq: Optional[List[torch.Tensor]] = None,
    ) -> tuple:
        """
        Batched Phase 2 + Phase 3: form rel tokens + self-attend for ALL T frames at once.

        Pads variable K_t to K_max, runs batched gather/CLIP/projection/self-attend.

        Args:
            enriched_all: (T, N, d_model)
            node_logits_all: (T, N, num_classes)
            person_idx_seq: T-list of (K_t,) long
            object_idx_seq: T-list of (K_t,) long
            union_features_seq: T-list of (K_t, d_union_roi) or None

        Returns:
            rel_tokens: (T, K_max, d_rel) — padded relationship tokens
            pair_valid: (T, K_max) bool — True for real pairs
            padded_pidx: (T, K_max) long — padded person indices
            padded_oidx: (T, K_max) long — padded object indices
        """
        T = enriched_all.shape[0]
        device = enriched_all.device
        K_counts = [p.shape[0] for p in person_idx_seq]
        K_max = max(K_counts) if K_counts else 0

        if K_max == 0:
            return (
                torch.zeros(T, 0, self.d_rel, device=device),
                torch.zeros(T, 0, dtype=torch.bool, device=device),
                torch.zeros(T, 0, dtype=torch.long, device=device),
                torch.zeros(T, 0, dtype=torch.long, device=device),
            )

        d_union = self.no_union_embedding.shape[0]

        # Pad indices to K_max
        padded_pidx = torch.zeros(T, K_max, dtype=torch.long, device=device)
        padded_oidx = torch.zeros(T, K_max, dtype=torch.long, device=device)
        pair_valid = torch.zeros(T, K_max, dtype=torch.bool, device=device)

        for t in range(T):
            K_t = K_counts[t]
            if K_t > 0:
                padded_pidx[t, :K_t] = person_idx_seq[t]
                padded_oidx[t, :K_t] = object_idx_seq[t]
                pair_valid[t, :K_t] = True

        # Batched gather person/object representations: (T, K_max, d_model)
        pidx_exp = padded_pidx.unsqueeze(-1).expand(T, K_max, enriched_all.shape[-1])
        oidx_exp = padded_oidx.unsqueeze(-1).expand(T, K_max, enriched_all.shape[-1])
        person_repr = torch.gather(enriched_all, 1, pidx_exp)  # (T, K_max, d_model)
        object_repr = torch.gather(enriched_all, 1, oidx_exp)  # (T, K_max, d_model)

        # Batched class predictions
        person_class_idx = torch.gather(
            node_logits_all, 1,
            padded_pidx.unsqueeze(-1).expand(T, K_max, node_logits_all.shape[-1]),
        ).argmax(dim=-1)  # (T, K_max)
        object_class_idx = torch.gather(
            node_logits_all, 1,
            padded_oidx.unsqueeze(-1).expand(T, K_max, node_logits_all.shape[-1]),
        ).argmax(dim=-1)  # (T, K_max)

        # Batched CLIP text features: (T, K_max, d_text)
        person_text = self._get_text_features(person_class_idx)
        object_text = self._get_text_features(object_class_idx)

        # Batched union features: (T, K_max, d_union)
        if union_features_seq is not None:
            padded_union = torch.zeros(T, K_max, union_features_seq[0].shape[-1], device=device)
            for t in range(T):
                K_t = K_counts[t]
                if K_t > 0 and union_features_seq[t] is not None:
                    padded_union[t, :K_t] = union_features_seq[t]
            union_feat = self.union_proj(padded_union)  # (T, K_max, d_union)
        else:
            union_feat = self.no_union_embedding.view(1, 1, -1).expand(T, K_max, -1)

        # Concatenate and project: (T, K_max, d_input) → (T, K_max, d_rel)
        concat = torch.cat([person_repr, object_repr, union_feat,
                            person_text, object_text], dim=-1)
        rel_tokens = self.input_proj(concat)  # (T, K_max, d_rel)

        # Zero out padding
        rel_tokens = rel_tokens * pair_valid.unsqueeze(-1).float()

        # Batched self-attention with padding mask
        padding_mask = ~pair_valid  # (T, K_max), True = ignore
        # Failsafe: if all-padded frame, unmask first
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        rel_tokens = self.rel_transformer(
            rel_tokens, src_key_padding_mask=padding_mask,
        )  # (T, K_max, d_rel)

        # Re-zero padding
        rel_tokens = rel_tokens * pair_valid.unsqueeze(-1).float()

        return rel_tokens, pair_valid, padded_pidx, padded_oidx

    def batched_predict(
        self,
        rel_tokens: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> Dict[str, List]:
        """
        Batched Phase 5: run 3 MLP heads, split back to per-frame lists.

        Args:
            rel_tokens: (T, K_max, d_rel)
            pair_valid: (T, K_max) bool

        Returns:
            dict with T-lists of (K_t, C) tensors per relationship type.
        """
        T = rel_tokens.shape[0]

        # Batched heads: (T, K_max, C)
        att_all = self.att_head(rel_tokens)
        spa_all = torch.sigmoid(self.spa_head(rel_tokens))
        con_all = torch.sigmoid(self.con_head(rel_tokens))

        # Split per frame using pair_valid
        att_list, spa_list, con_list = [], [], []
        for t in range(T):
            mask_t = pair_valid[t]  # (K_max,)
            att_list.append(att_all[t][mask_t])
            spa_list.append(spa_all[t][mask_t])
            con_list.append(con_all[t][mask_t])

        return {
            "attention_distribution": att_list,
            "spatial_distribution": spa_list,
            "contacting_distribution": con_list,
        }

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
# 5b. Temporal Edge Attention — Per-Pair Temporal Self-Attention
# ============================================================================

class TemporalEdgeAttention(nn.Module):
    """
    Per-pair temporal self-attention over the full video.

    Receives relationship tokens from ALL T frames at once.
    Groups tokens by (person, object) pair, self-attends each pair's
    temporal sequence, and returns enriched tokens for all frames.

    Stateless — no internal buffer, single forward pass with full
    gradient flow through all timesteps.

    Args:
        d_rel: Relationship token dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        dropout: Dropout probability.
        max_time: Maximum number of frames (for temporal PE).

    Input:
        rel_tokens_seq: List[Tensor(K_t, d_rel)] for t=0..T-1
        person_idx_seq: List[Tensor(K_t,)] for t=0..T-1
        object_idx_seq: List[Tensor(K_t,)] for t=0..T-1
    Output:
        List[Tensor(K_t, d_rel)] — temporally-enriched tokens per frame
    """

    def __init__(
        self,
        d_rel: int = 256,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
        max_time: int = 300,
    ):
        super().__init__()
        self.d_rel = d_rel

        # Learnable temporal positional encoding
        self.temporal_pe = nn.Embedding(max_time, d_rel)

        # Self-attention encoder over temporal sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_rel,
            nhead=n_heads,
            dim_feedforward=d_rel * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_rel),
        )

    def forward(
        self,
        rel_tokens_all: torch.Tensor,
        pair_valid: torch.Tensor,
        padded_pidx: torch.Tensor,
        padded_oidx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized temporal edge attention.

        Args:
            rel_tokens_all: (T, K_max, d_rel) — padded rel tokens from batched_form_and_attend.
            pair_valid: (T, K_max) bool — True for real pairs.
            padded_pidx: (T, K_max) long — padded person indices.
            padded_oidx: (T, K_max) long — padded object indices.

        Returns:
            enriched: (T, K_max, d_rel) — temporally-enriched tokens (padded).
        """
        T, K_max = rel_tokens_all.shape[:2]
        device = rel_tokens_all.device

        if K_max == 0 or not pair_valid.any():
            return rel_tokens_all

        # --- Step 1: Vectorized pair grouping ---
        # Create unique pair keys: pidx * max_N + oidx
        max_N = max(padded_pidx.max().item(), padded_oidx.max().item()) + 1
        pair_keys = padded_pidx * max_N + padded_oidx  # (T, K_max)

        # Only consider valid pairs
        # Set invalid pair keys to a sentinel value
        SENTINEL = max_N * max_N + 1
        pair_keys_masked = torch.where(pair_valid, pair_keys, torch.full_like(pair_keys, SENTINEL))

        # Flatten to find unique pairs
        flat_keys = pair_keys_masked.reshape(-1)  # (T * K_max,)
        flat_valid = pair_valid.reshape(-1)        # (T * K_max,)
        flat_tokens = rel_tokens_all.reshape(-1, self.d_rel)  # (T * K_max, d_rel)

        # Frame index for each flat position
        flat_frame_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, K_max).reshape(-1)

        # Get only valid entries
        valid_indices = flat_valid.nonzero(as_tuple=True)[0]  # (total_valid,)
        if valid_indices.numel() == 0:
            return rel_tokens_all

        valid_keys = flat_keys[valid_indices]          # (total_valid,)
        valid_tokens = flat_tokens[valid_indices]      # (total_valid, d_rel)
        valid_frame_ids = flat_frame_ids[valid_indices] # (total_valid,)

        # Assign unique pair IDs
        unique_keys, inverse_ids = torch.unique(valid_keys, return_inverse=True)  # (num_pairs,), (total_valid,)
        num_pairs = unique_keys.shape[0]

        # Compute temporal position within each pair using cumsum trick
        # Sort by pair_id to group, then compute within-group position
        sort_order = torch.argsort(inverse_ids)
        sorted_pair_ids = inverse_ids[sort_order]

        # Mark boundaries between pairs
        boundaries = torch.ones(sorted_pair_ids.shape[0], dtype=torch.long, device=device)
        boundaries[1:] = (sorted_pair_ids[1:] != sorted_pair_ids[:-1]).long()
        temporal_pos_sorted = torch.cumsum(boundaries, dim=0) - torch.cumsum(boundaries, dim=0).gather(
            0, torch.zeros_like(sorted_pair_ids)  # placeholder
        )
        # Simpler: just count within each group
        pair_counts = torch.zeros(sorted_pair_ids.shape[0], dtype=torch.long, device=device)
        pair_starts = torch.cat([torch.tensor([0], device=device),
                                  (sorted_pair_ids[1:] != sorted_pair_ids[:-1]).nonzero(as_tuple=True)[0] + 1])
        for i, start in enumerate(pair_starts):
            end = pair_starts[i + 1] if i + 1 < len(pair_starts) else sorted_pair_ids.shape[0]
            pair_counts[start:end] = torch.arange(end - start, device=device)

        # Unsort temporal positions
        temporal_pos = torch.zeros_like(pair_counts)
        temporal_pos[sort_order] = pair_counts

        T_max = temporal_pos.max().item() + 1 if temporal_pos.numel() > 0 else 1

        # --- Step 2: Scatter into (num_pairs, T_max, d_rel) batch ---
        batch = torch.zeros(num_pairs, T_max, self.d_rel, device=device)
        batch[inverse_ids, temporal_pos] = valid_tokens  # gradient flows

        # Padding mask: True = ignore
        padding_mask = torch.ones(num_pairs, T_max, dtype=torch.bool, device=device)
        padding_mask[inverse_ids, temporal_pos] = False

        # Frame indices for temporal PE
        frame_indices = torch.zeros(num_pairs, T_max, dtype=torch.long, device=device)
        frame_indices[inverse_ids, temporal_pos] = valid_frame_ids

        # --- Step 3: Add temporal positional encoding ---
        batch = batch + self.temporal_pe(frame_indices)

        # --- Step 4: Self-attend ---
        attended = self.encoder(batch, src_key_padding_mask=padding_mask)

        # --- Step 5: Scatter back to (T, K_max, d_rel) ---
        enriched_valid = attended[inverse_ids, temporal_pos]  # (total_valid, d_rel)

        # Write back to flat tensor
        enriched_flat = torch.zeros_like(flat_tokens)
        enriched_flat[valid_indices] = enriched_valid

        # Reshape back to (T, K_max, d_rel)
        enriched_all = enriched_flat.reshape(T, K_max, self.d_rel)

        return enriched_all



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
      1. A global camera token (B, d_camera) — summarizes current viewpoint
      2. Per-object camera-relative features (B, N, d_camera) — how each
         object relates to the current camera position and viewing direction

    Accepts batched (B, ...) inputs natively.

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
        #   log_distance(1) + view_alignment(1) + azimuth_sin(1) + azimuth_cos(1) = 4
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
            camera_pose: (B, 4, 4) — camera-to-world extrinsic matrix.
            corners: (B, N, 8, 3) — 3D bbox corners for all objects.
            valid_mask: (B, N) bool — True for real objects.

        Returns:
            camera_token: (B, d_camera) — global viewpoint summary.
            per_object_cam_feats: (B, N, d_camera) — camera-relative per-object features.
        """
        B, N = corners.shape[0], corners.shape[1]

        # --- Extract rotation and translation from pose ---
        R = camera_pose[:, :3, :3]  # (B, 3, 3)
        t = camera_pose[:, :3, 3]   # (B, 3) — camera position in world

        # --- Global camera token ---
        rot_6d = _rotation_matrix_to_6d(R)  # (B, 6)
        global_input = torch.cat([rot_6d, t], dim=-1)  # (B, 9)
        camera_token = self.global_mlp(global_input)  # (B, d_camera)

        # --- Per-object camera-relative features ---
        centers = corners.mean(dim=2)  # (B, N, 3)

        # Camera viewing direction (negative Z axis)
        view_dir = -R[:, :, 2]  # (B, 3)
        view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # Vector from camera to each object
        cam_to_obj = centers - t.unsqueeze(1)  # (B, N, 3)
        dist = cam_to_obj.norm(dim=-1, keepdim=True)  # (B, N, 1)
        cam_to_obj_norm = cam_to_obj / (dist + 1e-8)  # (B, N, 3)

        # View alignment: dot product with viewing direction
        view_alignment = (cam_to_obj_norm * view_dir.unsqueeze(1)).sum(dim=-1, keepdim=True)  # (B, N, 1)

        # Azimuth (sin, cos) for rotational equivariance
        right_dir = R[:, :, 0]  # (B, 3)
        right_component = (cam_to_obj_norm * right_dir.unsqueeze(1)).sum(dim=-1, keepdim=True)  # (B, N, 1)
        azimuth_sin = right_component
        azimuth_cos = view_alignment

        log_dist = torch.log(dist + 1e-6)  # (B, N, 1)

        per_obj_input = torch.cat([
            log_dist, view_alignment, azimuth_sin, azimuth_cos,
        ], dim=-1)  # (B, N, 4)

        per_object_cam_feats = self.per_object_mlp(per_obj_input)  # (B, N, d_camera)

        # Zero out padding
        per_object_cam_feats = per_object_cam_feats * valid_mask.unsqueeze(-1).float()

        return camera_token, per_object_cam_feats


# ============================================================================
# 8. Camera Temporal Encoder (Ego-Motion)
# ============================================================================

class CameraTemporalEncoder(nn.Module):
    """
    Encodes the full sequence of camera poses into ego-motion tokens
    with full temporal context via self-attention.

    Each timestamp's raw ego-motion (relative pose change from previous frame)
    is encoded via MLP, then all T tokens self-attend to capture long-range
    camera motion patterns (e.g., panning back and forth, orbiting around).

    Input:  pose_seq (T, 4, 4)
    Output: ego_motion_tokens (T, d_camera)

    Args:
        d_camera: Output dimension for ego-motion tokens.
        d_hidden: Hidden dimension in the encoder MLP.
        n_attn_layers: Number of self-attention layers.
        n_heads: Number of attention heads.
    """

    def __init__(
        self,
        d_camera: int = 128,
        d_hidden: int = 64,
        n_attn_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.d_camera = d_camera

        # Per-step relative pose encoder: 6D rotation + 3D translation = 9
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

        # Learnable temporal positional encoding
        self.max_time = 300
        self.temporal_pe = nn.Embedding(self.max_time, d_camera)

        # Self-attention over full temporal context
        attn_layer = nn.TransformerEncoderLayer(
            d_model=d_camera,
            nhead=n_heads,
            dim_feedforward=d_camera * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(
            attn_layer,
            num_layers=n_attn_layers,
        )

    def forward(
        self,
        pose_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pose_seq: (T, 4, 4) — full sequence of camera poses.

        Returns:
            ego_motion_tokens: (T, d_camera) — temporally-contextualized
                               ego-motion tokens for each timestep.
        """
        T = pose_seq.shape[0]
        device = pose_seq.device

        # 1. Compute per-step relative poses → MLP → raw ego tokens
        ego_tokens = []
        for t in range(T):
            if t == 0:
                ego_tokens.append(self.no_prev_embedding.unsqueeze(0))
            else:
                T_rel = pose_seq[t] @ torch.inverse(pose_seq[t - 1])  # (4, 4)
                R_rel = T_rel[:3, :3]
                t_rel = T_rel[:3, 3]
                rot_6d = _rotation_matrix_to_6d(R_rel)  # (6,)
                ego_input = torch.cat([rot_6d, t_rel], dim=-1)  # (9,)
                ego_tokens.append(self.ego_mlp(ego_input).unsqueeze(0))

        ego_seq = torch.cat(ego_tokens, dim=0)  # (T, d_camera)

        # 2. Add temporal positional encoding
        positions = torch.arange(T, device=device).clamp(max=self.max_time - 1)
        ego_seq = ego_seq + self.temporal_pe(positions)

        # 3. Self-attend over full temporal context (batch dim = 1)
        ego_seq = self.temporal_attn(ego_seq.unsqueeze(0)).squeeze(0)  # (T, d_camera)

        return ego_seq



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
# 9. Motion Feature Encoder
# ============================================================================

class MotionFeatureEncoder(nn.Module):
    """
    Encodes 3D motion features (velocity, acceleration) into per-object tokens.

    Uses finite-difference on 3D bbox centers from consecutive frames.
    Optionally computes camera-relative velocity for parallax-aware motion.

    Accepts batched (B, ...) inputs natively.

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
            velocity: (B, N, 3) — world-frame velocity. None for first frame.
            acceleration: (B, N, 3) — world-frame acceleration. None for first two frames.
            camera_R: (B, 3, 3) — camera rotation matrix.
            valid_mask: (B, N) bool — valid objects.

        Returns:
            motion_features: (B, N, d_motion) — per-object motion tokens.
        """
        if velocity is None:
            # First frame: return learnable "no motion" for all objects
            if valid_mask is not None:
                B, N = valid_mask.shape
            else:
                return self.no_motion_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, d_motion)
            return self.no_motion_embedding.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # (B, N, d_motion)

        B, N = velocity.shape[0], velocity.shape[1]
        device = velocity.device

        # Speed scalar (magnitude of velocity)
        speed = velocity.norm(dim=-1, keepdim=True)  # (B, N, 1)

        # Acceleration (zeros if not available)
        if acceleration is None:
            acceleration = torch.zeros_like(velocity)  # (B, N, 3)

        # Base features
        feats = [velocity, acceleration, speed]  # 3 + 3 + 1 = 7

        # Camera-relative velocity
        if self.use_cam_relative and camera_R is not None:
            # Transform: v_cam = R^T @ v_world → batched einsum
            cam_vel = torch.einsum('bij,bnj->bni', camera_R.transpose(-1, -2), velocity)  # (B, N, 3)
            cam_speed = cam_vel.norm(dim=-1, keepdim=True)  # (B, N, 1)
            feats.extend([cam_vel, cam_speed])
        elif self.use_cam_relative:
            feats.extend([
                torch.zeros(B, N, 3, device=device),
                torch.zeros(B, N, 1, device=device),
            ])

        motion_input = torch.cat(feats, dim=-1)  # (B, N, 7 or 11)
        motion_features = self.mlp(motion_input)  # (B, N, d_motion)

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
            curr_corners: (B, N, 8, 3) or (N, 8, 3) — current frame corners.
            prev_corners: (B, N, 8, 3) or (N, 8, 3) — previous frame corners.

        Returns:
            velocity: (..., N, 3) — displacement of center between frames.
        """
        curr_centers = curr_corners.mean(dim=-2)  # (..., N, 3)
        prev_centers = prev_corners.mean(dim=-2)  # (..., N, 3)
        return curr_centers - prev_centers
