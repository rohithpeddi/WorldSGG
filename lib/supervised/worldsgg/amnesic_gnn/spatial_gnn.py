"""
Spatial GNN
============

Graph Transformer for spatial context propagation in the Amnesic GNN.
Allows visual features from visible nodes to "bleed" to unseen nodes
based on 3D proximity — the only mechanism for unseen object reasoning
when there is zero temporal memory.

Accepts batched (B, N, ...) inputs natively.
"""

import torch
import torch.nn as nn

from lib.supervised.worldsgg.worldsgg_base import SpatialPositionalEncoding


class SpatialGNN(nn.Module):
    """
    Self-attention Graph Transformer with 3D spatial positional encoding.

    Propagates spatial context between all nodes. For the Amnesic baseline,
    this is the ONLY way unseen nodes can get any information beyond their
    raw geometry — via attention edges to visible neighbors.

    Args:
        d_model: Token dimension.
        n_layers: Number of transformer encoder layers.
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
            tokens: (B, N, d_model) — hybrid tokens from AmnesicTokenizer.
            corners: (B, N, 8, 3) — 3D bbox corners.
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

        # 4. Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 5. Strict zero-masking
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        return x
