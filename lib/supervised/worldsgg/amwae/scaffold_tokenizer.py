"""
Geometric Scaffold Tokenizer
=============================

Top-down tokenization: initializes the complete global graph from the wireframe,
then binds DINO visual evidence to visible nodes and a learnable [MASK] embedding
to unseen nodes.

This is the paradigm shift from GL-STGN: instead of building outward from camera
detections, we start with the full structural scaffold and fill in the evidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaffoldTokenizer(nn.Module):
    """
    Converts wireframe geometry tokens + visual features into hybrid tokens.

    For each object at time t:
      - If VISIBLE:  token = Proj([geometry_token ⊕ projected_DINO])
      - If MASKED:   token = Proj([geometry_token ⊕ E_mask])

    During training, a fraction p_mask_visible of visible objects are
    *artificially* masked (their DINO features replaced with E_mask)
    to force the memory to learn associative retrieval.

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
        self.d_model = d_model

        # Project raw DINO ROI features → d_visual
        self.visual_projector = nn.Sequential(
            nn.Linear(d_detector_roi, d_visual),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_visual),
        )

        # Learnable [MASK] embedding — replaces DINO features for unseen objects
        self.mask_embedding = nn.Parameter(torch.randn(d_visual) * 0.02)

        # Fuse geometry + visual/mask → d_model
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
        p_mask_visible: float = 0.0,
    ) -> tuple:
        """
        Args:
            geometry_tokens: (N, d_struct) — per-object structural tokens.
            visual_features: (N, d_detector_roi) — raw DINO ROI features.
                             Zeros for objects without detections.
            visibility_mask: (N,) bool — True if in camera FOV.
            valid_mask: (N,) bool — True for real objects (not padding).
            p_mask_visible: Training masking prob for visible objects.

        Returns:
            tokens: (N, d_model) — hybrid tokens.
            is_masked: (N,) bool — True for tokens that received [MASK]
                       (including both naturally unseen AND artificially masked).
            original_visual: (N, d_visual) — projected DINO features BEFORE masking
                             (for reconstruction loss on artificially masked tokens).
        """
        N = geometry_tokens.shape[0]
        device = geometry_tokens.device

        # Project visual features
        visual_proj = self.visual_projector(visual_features)  # (N, d_visual)

        # Save original projected features for reconstruction loss
        original_visual = visual_proj.clone().detach()

        # Determine which tokens are masked
        # Start with: unseen objects are masked
        is_masked = ~visibility_mask  # (N,) — True for unseen

        # Simulated extreme masking during training
        if self.training and p_mask_visible > 0.0:
            # Randomly mask some visible objects
            rand = torch.rand(N, device=device)
            artificially_masked = (rand < p_mask_visible) & visibility_mask & valid_mask
            is_masked = is_masked | artificially_masked

        # Build visual component: DINO features for unmasked, [MASK] for masked
        mask_emb = self.mask_embedding.unsqueeze(0).expand(N, -1)  # (N, d_visual)
        visual_component = torch.where(
            is_masked.unsqueeze(-1),
            mask_emb,
            visual_proj,
        )  # (N, d_visual)

        # Fuse geometry + visual component
        fused = torch.cat([geometry_tokens, visual_component], dim=-1)  # (N, d_struct + d_visual)
        tokens = self.fusion_proj(fused)  # (N, d_model)

        # Zero out padding
        tokens = tokens * valid_mask.unsqueeze(-1).float()

        return tokens, is_masked, original_visual
