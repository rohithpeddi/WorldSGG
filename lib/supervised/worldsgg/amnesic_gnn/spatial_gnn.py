"""
Spatial GNN
============

Graph Transformer for spatial context propagation in the Amnesic GNN.
Allows visual features from visible nodes to "bleed" to unseen nodes
based on 3D proximity — the only mechanism for unseen object reasoning
when there is zero temporal memory.
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

        # 3D spatial positional encoding (reused from GL-STGN)
        self.spatial_pe = SpatialPositionalEncoding(d_model=d_model)

        # Pre-norm
        self.pre_norm = nn.LayerNorm(d_model)

        # Transformer encoder stack
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
            tokens: (N, d_model) — hybrid tokens from AmnesicTokenizer.
            corners: (N, 8, 3) — 3D bbox corners.
            valid_mask: (N,) bool — True for real objects.

        Returns:
            enriched: (N, d_model) — context-enriched representations.
        """
        # Add spatial positional encoding
        spatial_enc = self.spatial_pe(corners, valid_mask)  # (N, d_model)
        x = tokens + spatial_enc

        x = self.pre_norm(x)

        # Add batch dim: (1, N, d_model)
        x = x.unsqueeze(0)
        padding_mask = ~valid_mask.unsqueeze(0)  # (1, N) — True = ignore

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = x.squeeze(0)  # (N, d_model)
        x = self.post_norm(x)

        # Zero out padding
        x = x * valid_mask.unsqueeze(-1).float()

        return x
