"""
Amnesic Tokenizer
==================

Binary feature assignment for the Amnesic Geometric GNN baseline.
Visible objects get real DINO features; unseen objects get a learnable
[UNSEEN] embedding. No memory recovery, no masking curriculum.

This is the "amnesic mask" — the network explicitly cannot recall what
unseen objects looked like. It can only rely on geometry and spatial
context propagation from visible neighbors.
"""

import torch
import torch.nn as nn


class AmnesicTokenizer(nn.Module):
    """
    Creates hybrid tokens from geometry and visual features.

    Visible:   token = Proj([geometry_i ⊕ DINO_i])
    Unseen:    token = Proj([geometry_j ⊕ E_unseen])

    Args:
        d_struct: Structural token dim (from GlobalStructuralEncoder).
        d_visual: Projected visual feature dim.
        d_model: Output token dim.
        d_detector_roi: Raw DINO ROI feature dim (1024).
    """

    def __init__(
        self,
        d_struct: int = 256,
        d_visual: int = 256,
        d_model: int = 256,
        d_detector_roi: int = 1024,
    ):
        super().__init__()
        self.d_visual = d_visual

        # Project raw DINO ROI features → d_visual
        self.visual_projector = nn.Sequential(
            nn.Linear(d_detector_roi, d_visual),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_visual),
        )

        # Learnable [UNSEEN] embedding — stands in for visual features
        # when an object is outside the camera FOV
        self.unseen_embedding = nn.Parameter(torch.randn(d_visual) * 0.02)

        # Fuse geometry + visual/unseen → d_model
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_struct + d_visual, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        visual_features: torch.Tensor,
        visibility_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            geometry_tokens: (N, d_struct) — per-object structural tokens.
            visual_features: (N, d_detector_roi) — raw DINO ROI features.
            visibility_mask: (N,) bool — True if in camera FOV.
            valid_mask: (N,) bool — True for real objects.

        Returns:
            tokens: (N, d_model) — hybrid tokens.
        """
        N = geometry_tokens.shape[0]

        # Project visual features
        visual_proj = self.visual_projector(visual_features)  # (N, d_visual)

        # Binary assignment: visible → DINO, unseen → [UNSEEN]
        unseen_emb = self.unseen_embedding.unsqueeze(0).expand(N, -1)  # (N, d_visual)
        visual_component = torch.where(
            visibility_mask.unsqueeze(-1),
            visual_proj,
            unseen_emb,
        )  # (N, d_visual)

        # Fuse geometry + visual component
        fused = torch.cat([geometry_tokens, visual_component], dim=-1)
        tokens = self.fusion_proj(fused)  # (N, d_model)

        # Zero out padding
        tokens = tokens * valid_mask.unsqueeze(-1).float()

        return tokens
