"""
LKS Tokenizer
===============

Fuses CURRENT wireframe geometry (live from the global wireframe)
with BUFFERED visual features (potentially stale from the LKS buffer).

X_k = Proj([G_k^t ⊕ M_t[k]])

Where M_t[k] is:
  - Fresh DINO features (if object was just seen)
  - Stale DINO features (if object was seen N frames ago)
  - Zeros (if object has never been seen — "fog of war")
"""

import torch
import torch.nn as nn


class LKSTokenizer(nn.Module):
    """
    Fuses live geometry with buffered (possibly stale) visual features.

    Unlike AmnesicTokenizer, this module has no learnable [UNSEEN] embedding.
    The visual component comes directly from the LKS buffer, which may
    contain fresh, stale, or zero features depending on history.

    Args:
        d_struct: Structural token dim.
        d_visual: Projected visual feature dim (buffer dim).
        d_model: Output token dim.
    """

    def __init__(
        self,
        d_struct: int = 256,
        d_visual: int = 256,
        d_model: int = 256,
    ):
        super().__init__()

        # Fuse geometry + buffered visual → d_model
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_struct + d_visual, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        buffer_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            geometry_tokens: (N, d_struct) — LIVE per-object structural tokens.
            buffer_features: (N, d_visual) — from LKS buffer (detached, possibly stale).
            valid_mask: (N,) bool — True for real objects.

        Returns:
            tokens: (N, d_model) — hybrid tokens.
        """
        # Concatenate current geometry with buffered visual
        fused = torch.cat([geometry_tokens, buffer_features], dim=-1)
        tokens = self.fusion_proj(fused)  # (N, d_model)

        # Zero out padding
        tokens = tokens * valid_mask.unsqueeze(-1).float()

        return tokens
