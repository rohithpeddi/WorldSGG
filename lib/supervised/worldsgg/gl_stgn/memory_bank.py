"""
Temporal Object Transformer
=============================

Per-object bidirectional self-attention over all T video frames.
Replaces the sequential GRU-based PersistentWorldMemoryBank.

For each object n, builds a sequence of T tokens (one per frame) from
fused visual, structural, camera, motion, and ego-motion features.
A learned visibility embedding distinguishes observed vs. unseen objects.
Bidirectional self-attention (no causal mask) lets each timestep see
the entire video, enabling global temporal reasoning for world scene graphs.

Pipeline:
  input_proj(cat[visual, struct, cam, motion, ego]) → (T, N, d_memory)
  + temporal_pe + visibility_embedding
  reshape → (N, T, d_memory)  [each object is a temporal sequence]
  TransformerEncoder(bidirectional) → (N, T, d_memory)
  reshape → (T, N, d_memory)
"""

import logging
import math
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class TemporalObjectTransformer(nn.Module):
    """
    Per-object bidirectional self-attention over all T frames.

    Each object's temporal sequence is processed independently via self-attention.
    Visibility embeddings inform the model which observations are direct (visible)
    vs. inferred (unseen), replacing the old GRU's seen/unseen branching.

    Args:
        d_visual: Projected visual feature dimension.
        d_struct: Structural token dimension.
        d_camera: Camera-relative feature dimension.
        d_motion: Motion feature dimension.
        d_detector_roi: Raw detector ROI dimension (before projection).
        d_memory: Output memory dimension.
        n_heads: Attention heads for temporal self-attention.
        n_layers: Number of Transformer encoder layers.
        dropout: Dropout probability.
        max_T: Maximum video length (for positional encoding).
    """

    def __init__(
        self,
        d_visual: int = 256,
        d_struct: int = 256,
        d_camera: int = 128,
        d_motion: int = 64,
        d_detector_roi: int = 1024,
        d_memory: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_T: int = 256,
    ):
        super().__init__()
        self.d_memory = d_memory
        self.d_visual = d_visual
        self.d_camera = d_camera
        self.d_motion = d_motion

        # Project raw DINO ROI features → d_visual
        self.visual_projector = nn.Sequential(
            nn.Linear(d_detector_roi, d_visual),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_visual),
        )

        # Fuse all per-object features into d_memory
        # Input: visual + struct + camera + motion + ego_camera = d_visual + d_struct + d_camera + d_motion + d_camera
        d_input = d_visual + d_struct + d_camera + d_motion + d_camera
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_memory),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_memory),
        )

        # Learned visibility embedding: 0 = unseen, 1 = visible
        self.visibility_emb = nn.Embedding(2, d_memory)

        # Sinusoidal temporal positional encoding
        self.register_buffer("temporal_pe", self._build_sinusoidal_pe(max_T, d_memory))

        # Bidirectional Transformer encoder (no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_memory,
            nhead=n_heads,
            dim_feedforward=d_memory * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_memory),
        )

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Build sinusoidal positional encoding table (max_len, d_model)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)

    def forward(
        self,
        visual_features: torch.Tensor,
        struct_tokens: torch.Tensor,
        cam_feats: torch.Tensor,
        motion_feats: torch.Tensor,
        ego_tokens: torch.Tensor,
        visibility_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process all T frames for all N objects in a single forward pass.

        Args:
            visual_features: (T, N, d_detector_roi) — raw DINO ROI features.
            struct_tokens:   (T, N, d_struct) — per-object structural tokens.
            cam_feats:       (T, N, d_camera) — per-object camera-relative features.
            motion_feats:    (T, N, d_motion) — per-object motion features.
            ego_tokens:      (T, d_camera) — global ego-motion tokens per frame.
            visibility_mask: (T, N) bool — True if object is visible this frame.
            valid_mask:      (T, N) bool — True for real (non-padding) objects.

        Returns:
            memory: (T, N, d_memory) — temporally-enriched object states.
        """
        T, N = visual_features.shape[:2]
        device = visual_features.device

        assert T <= self.temporal_pe.shape[0], (
            f"Video length T={T} exceeds max_T={self.temporal_pe.shape[0]}. "
            f"Increase max_T in TemporalObjectTransformer."
        )

        # --- Project visual features ---
        vis = self.visual_projector(visual_features)  # (T, N, d_visual)

        # --- Broadcast ego-motion tokens to per-object ---
        ego_broadcast = ego_tokens.unsqueeze(1).expand(T, N, -1)  # (T, N, d_camera)

        # --- Default zeros for missing features ---
        if cam_feats is None:
            cam_feats = torch.zeros(T, N, self.d_camera, device=device)
        if motion_feats is None:
            motion_feats = torch.zeros(T, N, self.d_motion, device=device)

        # --- Fuse all features ---
        fused = torch.cat([vis, struct_tokens, cam_feats, motion_feats, ego_broadcast], dim=-1)
        tokens = self.input_proj(fused)  # (T, N, d_memory)

        # --- Add temporal PE + visibility embedding ---
        tokens = tokens + self.temporal_pe[:T].unsqueeze(1)  # broadcast over N
        vis_ids = visibility_mask.long()  # (T, N): 0=unseen, 1=visible
        tokens = tokens + self.visibility_emb(vis_ids)  # (T, N, d_memory)

        # --- Zero out padding objects ---
        tokens = tokens * valid_mask.unsqueeze(-1).float()

        # --- Reshape for per-object temporal attention ---
        # (T, N, d) → (N, T, d): each object is a sequence of T steps
        tokens = tokens.permute(1, 0, 2)  # (N, T, d_memory)

        # --- Build padding mask ---
        # An object at time t is "padding" if it's invalid at that frame
        # PyTorch expects True = ignore
        padding_mask = ~valid_mask.permute(1, 0)  # (N, T)

        # Failsafe: if an entire object is fully padded, unmask first token
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        # --- Bidirectional self-attention over T (no causal mask) ---
        memory = self.transformer(tokens, src_key_padding_mask=padding_mask)  # (N, T, d_memory)

        # --- Strict zero-masking for padding ---
        # Use pristine valid_mask, NOT the failsafe-modified padding_mask,
        # to avoid leaking transformer garbage at t=0 for fully invalid objects.
        memory = memory * valid_mask.permute(1, 0).unsqueeze(-1).float()

        # --- Reshape back to (T, N, d_memory) ---
        memory = memory.permute(1, 0, 2)  # (T, N, d_memory)

        return memory
