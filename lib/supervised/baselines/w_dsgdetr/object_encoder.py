"""
Temporal Object Encoder
========================

Lightweight self-attention encoder over each object's temporal sequence.
Inspired by DSGDetr's ObjectClassifier.encoder_tran, but adapted for
the world-centric setting where persistent object slots provide natural
tracking (no Hungarian matcher needed).

Given (T, N, d_model) tokens, for each object slot n ∈ [0, N):
  - Gathers its T-frame sequence (using valid_mask to ignore padding)
  - Adds learnable temporal positional encoding
  - Applies self-attention across the temporal dimension
  - Scatters enriched features back to (T, N, d_model)

This provides per-object temporal context BEFORE relationship prediction,
analogous to DSGDetr's tracking-based object representation.
"""

import logging

import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class TemporalObjectEncoder(nn.Module):
    """
    Per-object temporal self-attention encoder.

    For each object slot, applies self-attention across all T timesteps
    where the object is valid. Uses learnable temporal positional encoding.

    Args:
        d_model: Token dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout rate.
        max_T: Maximum sequence length for temporal PE.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_T: int = 300,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_T = max_T

        # Learnable temporal positional encoding
        self.temporal_pe = nn.Embedding(max_T, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-object temporal self-attention.

        Args:
            tokens: (T, N, d_model) — per-object tokens across all frames.
            valid_mask: (T, N) bool — True for real objects.

        Returns:
            enriched: (T, N, d_model) — temporally-enriched tokens.
        """
        T, N, D = tokens.shape
        device = tokens.device

        # Transpose to (N, T, D) for per-object temporal attention
        tokens_nt = tokens.permute(1, 0, 2)  # (N, T, D)
        mask_nt = valid_mask.permute(1, 0)    # (N, T) bool — True = valid

        # Add temporal positional encoding
        positions = torch.arange(T, device=device).clamp(max=self.max_T - 1)
        tokens_nt = tokens_nt + self.temporal_pe(positions).unsqueeze(0)  # (N, T, D)

        # Padding mask for transformer: True = IGNORE
        padding_mask = ~mask_nt  # (N, T) — True = ignore

        # Self-attend: each object slot attends over its temporal sequence
        # batch_first=True, so input is (N, T, D)
        enriched_nt = self.encoder(
            tokens_nt,
            src_key_padding_mask=padding_mask,
        )  # (N, T, D)

        # Transpose back to (T, N, D)
        enriched = enriched_nt.permute(1, 0, 2)  # (T, N, D)

        # Zero out padding positions
        enriched = enriched * valid_mask.unsqueeze(-1).float()

        return enriched
