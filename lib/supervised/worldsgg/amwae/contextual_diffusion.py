"""
Contextual Diffusion
=====================

Self-attention Graph Transformer that propagates context between all tokens
(visible evidence + memory-recovered) after associative retrieval.

Allows visible evidence to propagate contextual clues to hallucinated tokens
(e.g., if a visible person is throwing, the recovered unseen ball must update
its relational state to reflect the incoming trajectory).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the SpatialPositionalEncoding from GL-STGN
from lib.supervised.worldsgg.worldsgg_base import SpatialPositionalEncoding


class ContextualDiffusion(nn.Module):
    """
    Self-attention transformer over the full set of auto-completed tokens
    with 3D spatial positional encoding.

    Input:  (N, d_model) — auto-completed global tokens
            (N, 8, 3) — 3D corners for spatial PE
            (N,) — valid mask
    Output: (N, d_model) — context-enriched representations

    Args:
        d_model: Token dimension.
        n_layers: Number of self-attention layers.
        n_heads: Attention heads.
        d_feedforward: FFN hidden dim.
        dropout: Dropout probability.
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

        # 3D spatial positional encoding (reused from GL-STGN)
        self.spatial_pe = SpatialPositionalEncoding(d_model=d_model)

        # Pre-norm
        self.pre_norm = nn.LayerNorm(d_model)

        # Self-attention transformer encoder
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
        )

        # Post-norm
        self.post_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (N, d_model) — auto-completed tokens from AssociativeRetriever.
            corners: (N, 8, 3) — 3D bbox corners for spatial PE.
            valid_mask: (N,) bool — True for real objects.

        Returns:
            enriched: (N, d_model) — context-enriched representations.
        """
        # Add spatial positional encoding
        spatial_enc = self.spatial_pe(corners, valid_mask)  # (N, d_model)
        x = tokens + spatial_enc

        # Pre-normalize
        x = self.pre_norm(x)

        # Add batch dim: (1, N, d_model)
        x = x.unsqueeze(0)

        # Padding mask: True = ignore
        padding_mask = ~valid_mask.unsqueeze(0)  # (1, N)

        # Self-attention
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = x.squeeze(0)  # (N, d_model)
        x = self.post_norm(x)

        # Zero out padding
        x = x * valid_mask.unsqueeze(-1).float()

        return x
