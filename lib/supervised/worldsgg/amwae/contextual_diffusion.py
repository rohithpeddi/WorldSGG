"""
Contextual Diffusion
=====================

Self-attention Graph Transformer that propagates context between all tokens
(visible evidence + memory-recovered) after associative retrieval.

Allows visible evidence to propagate contextual clues to hallucinated tokens
(e.g., if a visible person is throwing, the recovered unseen ball must update
its relational state to reflect the incoming trajectory).

Accepts batched (B, N, ...) inputs natively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.supervised.worldsgg.worldsgg_base import SpatialPositionalEncoding


class ContextualDiffusion(nn.Module):
    """
    Self-attention transformer over the full set of auto-completed tokens
    with 3D spatial positional encoding.

    Args:
        d_model: Token dimension.
        n_layers: Number of self-attention layers.
        n_heads: Attention heads.
        d_feedforward: FFN hidden dim.
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
        self.d_model = d_model

        self.spatial_pe = SpatialPositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
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
        """
        Args:
            tokens: (B, N, d_model) — auto-completed tokens.
            corners: (B, N, 8, 3) — 3D bbox corners for spatial PE.
            valid_mask: (B, N) bool — True for real objects.

        Returns:
            enriched: (B, N, d_model) — context-enriched representations.
        """
        # 1. Spatial PE (natively batched)
        spatial_enc = self.spatial_pe(corners, valid_mask)  # (B, N, d_model)
        x = tokens + spatial_enc

        # 2. Padding mask
        padding_mask = ~valid_mask  # (B, N)

        # 3. Failsafe for fully-padded batch items
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        # 4. Self-attention
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 5. Strict zero-masking
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        return x
