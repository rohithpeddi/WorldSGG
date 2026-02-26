"""
Relational Graph Transformer
=============================

Graph Transformer with 3D spatial positional encoding that propagates context
between all memory nodes (visible + unseen) to enable relational reasoning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPositionalEncoding(nn.Module):
    """
    Compute 3D-geometry-aware positional encodings from object bounding boxes.

    For each pair of objects, computes:
      - Euclidean distance between 3D centers
      - Relative direction vector (normalized)
      - Relative log-scale (volume ratio)

    These are encoded via an MLP and added to object representations.
    """

    def __init__(self, d_model: int = 256, d_hidden: int = 64):
        super().__init__()
        # Pairwise spatial feature: distance(1) + direction(3) + log_vol_ratio(1) = 5
        self.pair_mlp = nn.Sequential(
            nn.Linear(5, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
        )

        # Per-object spatial encoding: aggregate pairwise features
        self.out_proj = nn.Linear(d_hidden, d_model)

    def forward(
        self,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            corners: (N, 8, 3) 3D bbox corners.
            valid_mask: (N,) bool.

        Returns:
            spatial_pe: (N, d_model) per-object spatial positional encoding.
        """
        N = corners.shape[0]
        device = corners.device

        # Compute centers and volumes
        centers = corners.mean(dim=1)  # (N, 3)
        # Approximate volume from bbox extents
        mins, _ = corners.min(dim=1)  # (N, 3)
        maxs, _ = corners.max(dim=1)  # (N, 3)
        extents = (maxs - mins).clamp(min=1e-6)  # (N, 3)
        volumes = extents.prod(dim=-1)  # (N,)
        log_volumes = torch.log(volumes + 1e-6)  # (N,)

        # Pairwise features: (N, N, 5)
        # Distance
        diff = centers.unsqueeze(1) - centers.unsqueeze(0)  # (N, N, 3)
        dist = diff.norm(dim=-1, keepdim=True)  # (N, N, 1)
        # Normalized direction
        direction = diff / (dist + 1e-6)  # (N, N, 3)
        # Log volume ratio
        log_vol_ratio = (log_volumes.unsqueeze(1) - log_volumes.unsqueeze(0)).unsqueeze(-1)  # (N, N, 1)

        pair_feats = torch.cat([dist, direction, log_vol_ratio], dim=-1)  # (N, N, 5)

        # Encode pairwise features
        pair_encoded = self.pair_mlp(pair_feats)  # (N, N, d_hidden)

        # Mask invalid pairs
        pair_valid = valid_mask.unsqueeze(0) & valid_mask.unsqueeze(1)  # (N, N)
        pair_encoded = pair_encoded * pair_valid.unsqueeze(-1).float()

        # Aggregate: mean pool over neighbors → per-object encoding
        n_valid = pair_valid.float().sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
        agg = pair_encoded.sum(dim=1) / n_valid  # (N, d_hidden)

        spatial_pe = self.out_proj(agg)  # (N, d_model)
        return spatial_pe


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

        # 3D spatial positional encoding
        self.spatial_pe = SpatialPositionalEncoding(d_model=d_model)

        # Pre-norm LayerNorm before transformer
        self.pre_norm = nn.LayerNorm(d_model)

        # Transformer encoder layers
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
        memory_states: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory_states: (N, d_model) object memory states from the memory bank.
            corners: (N, 8, 3) 3D bbox corners for spatial PE.
            valid_mask: (N,) bool — True for real objects.

        Returns:
            enriched: (N, d_model) context-enriched representations.
        """
        N = memory_states.shape[0]

        # Add spatial positional encoding
        spatial_enc = self.spatial_pe(corners, valid_mask)  # (N, d_model)
        x = memory_states + spatial_enc

        # Pre-normalize
        x = self.pre_norm(x)

        # Add batch dimension for transformer: (1, N, d_model)
        x = x.unsqueeze(0)

        # Create padding mask: True = ignore (inverted from valid_mask)
        padding_mask = ~valid_mask.unsqueeze(0)  # (1, N)

        # Run through transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (1, N, d_model)

        x = x.squeeze(0)  # (N, d_model)
        x = self.post_norm(x)

        # Zero out padding
        x = x * valid_mask.unsqueeze(-1).float()

        return x
