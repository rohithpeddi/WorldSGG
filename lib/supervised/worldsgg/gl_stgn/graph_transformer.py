"""
Relational Graph Transformer
=============================

Graph Transformer with 3D spatial positional encoding that propagates context
between all memory nodes (visible + unseen) to enable relational reasoning.

Accepts batched (B, N, ...) inputs natively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.supervised.worldsgg.worldsgg_base import SpatialPositionalEncoding


class RelationalGraphTransformer(nn.Module):
    """
    Graph Transformer operating on the full memory bank with 3D spatial
    positional encoding injected into the representations.

    Follows the DsgDETR pattern: spatial encoder + temporal/graph encoder.

    Args:
        d_model: Model dimension (= d_memory).
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        d_feedforward: Feed-forward hidden dimension.
        dropout: Dropout probability.

    Input:  memory_states (B, N, d_model), corners (B, N, 8, 3), valid_mask (B, N)
    Output: enriched (B, N, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 3,
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
        memory_states: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory_states: (B, N, d_model) object memory states.
            corners: (B, N, 8, 3) 3D bbox corners for spatial PE.
            valid_mask: (B, N) bool — True for real objects.

        Returns:
            enriched: (B, N, d_model) context-enriched representations.
        """
        # 1. Spatial PE (natively batched)
        spatial_enc = self.spatial_pe(corners, valid_mask)  # (B, N, d_model)
        x = memory_states + spatial_enc

        # 2. Padding mask
        padding_mask = ~valid_mask  # (B, N)

        # 3. Failsafe for fully-padded batch items
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        # 4. Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 5. Strict zero-masking
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        return x
