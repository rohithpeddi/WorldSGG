"""
Geometric Scaffold Tokenizer (Batched)
========================================

Top-down tokenization: initializes the complete global graph from the wireframe,
then binds DINO visual evidence to visible nodes and a learnable [MASK] embedding
to unseen nodes. Fuses camera, motion, and ego-motion features.

Accepts batched (B, N, ...) inputs natively (B=T for full video processing).
"""

import torch
import torch.nn as nn
from typing import Optional


class ScaffoldTokenizer(nn.Module):
    """
    Converts wireframe geometry + visual + camera + motion + ego features
    into hybrid tokens. Batched with (B, N, ...) inputs.

    For each object at time t:
      - If VISIBLE:  token = Proj([geometry ⊕ DINO ⊕ cam ⊕ motion ⊕ ego])
      - If MASKED:   token = Proj([geometry ⊕ E_mask ⊕ cam ⊕ motion ⊕ ego])

    During training, a fraction p_mask_visible of visible objects are
    artificially masked to force the retriever to learn associative recovery.

    Args:
        d_struct: Structural token dim.
        d_visual: Projected visual feature dim.
        d_model: Output token dim.
        d_detector_roi: Raw DINO ROI feature dim.
        d_camera: Camera-relative feature dim.
        d_motion: Motion feature dim.
    """

    def __init__(
        self,
        d_struct: int = 256,
        d_visual: int = 256,
        d_model: int = 256,
        d_detector_roi: int = 1024,
        d_camera: int = 128,
        d_motion: int = 64,
    ):
        super().__init__()
        self.d_visual = d_visual
        self.d_model = d_model
        self.d_camera = d_camera
        self.d_motion = d_motion

        # Project raw DINO ROI features → d_visual
        self.visual_projector = nn.Sequential(
            nn.Linear(d_detector_roi, d_visual),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_visual),
        )

        # Learnable [MASK] embedding — replaces DINO features for unseen objects
        self.mask_embedding = nn.Parameter(torch.randn(d_visual) * 0.02)

        # Fuse: geometry + visual/mask + camera + motion + ego
        fusion_input_dim = d_struct + d_visual + d_camera + d_motion + d_camera
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
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
        cam_feats: Optional[torch.Tensor] = None,
        motion_feats: Optional[torch.Tensor] = None,
        ego_tokens: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            geometry_tokens: (B, N, d_struct) — per-object structural tokens.
            visual_features: (B, N, d_detector_roi) — raw DINO ROI features.
            visibility_mask: (B, N) bool — True if in camera FOV.
            valid_mask: (B, N) bool — True for real objects (not padding).
            p_mask_visible: Training masking prob for visible objects.
            cam_feats: (B, N, d_camera) or None.
            motion_feats: (B, N, d_motion) or None — velocity/acceleration.
            ego_tokens: (B, d_camera) or None — global ego-motion per frame.

        Returns:
            tokens: (B, N, d_model) — hybrid tokens.
            is_masked: (B, N) bool — True for tokens that received [MASK].
            original_visual: (B, N, d_visual) — projected DINO features BEFORE masking.
        """
        B, N = geometry_tokens.shape[:2]
        device = geometry_tokens.device

        # Project visual features
        visual_proj = self.visual_projector(visual_features)  # (B, N, d_visual)

        # Save original projected features for reconstruction loss
        original_visual = visual_proj.clone().detach()

        # Determine which tokens are masked: unseen objects
        is_masked = ~visibility_mask  # (B, N)

        # Simulated masking during training
        if self.training and p_mask_visible > 0.0:
            rand = torch.rand(B, N, device=device)
            artificially_masked = (rand < p_mask_visible) & visibility_mask & valid_mask
            is_masked = is_masked | artificially_masked

        # Build visual component: DINO for unmasked, [MASK] for masked
        mask_emb = self.mask_embedding.view(1, 1, -1).expand(B, N, -1)
        visual_component = torch.where(
            is_masked.unsqueeze(-1), mask_emb, visual_proj,
        )  # (B, N, d_visual)

        # Defaults for missing features
        if cam_feats is None:
            cam_feats = torch.zeros(B, N, self.d_camera, device=device)
        if motion_feats is None:
            motion_feats = torch.zeros(B, N, self.d_motion, device=device)

        # Broadcast ego tokens to per-object: (B, d_camera) → (B, N, d_camera)
        if ego_tokens is not None:
            ego_broadcast = ego_tokens.unsqueeze(1).expand(B, N, -1)
        else:
            ego_broadcast = torch.zeros(B, N, self.d_camera, device=device)

        # Fuse all features
        fused = torch.cat([
            geometry_tokens, visual_component, cam_feats,
            motion_feats, ego_broadcast,
        ], dim=-1)
        tokens = self.fusion_proj(fused)  # (B, N, d_model)

        # Zero out padding
        tokens = tokens * valid_mask.unsqueeze(-1).float()

        return tokens, is_masked, original_visual
