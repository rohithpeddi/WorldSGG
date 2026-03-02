"""
LKS Tokenizer
===============

Fuses CURRENT wireframe geometry (live from the global wireframe)
with BUFFERED visual features (raw DINO, potentially stale from the LKS buffer)
and CAMERA-RELATIVE features.

X[t,n] = Proj([G[t,n] ⊕ M[t,n] ⊕ cam[t,n] ⊕ log_staleness[t,n]])

All inputs are 3D tensors shaped (T, N, D).
"""

import logging

import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class LKSTokenizer(nn.Module):
    """
    Fuses live geometry with buffered visual features, camera features,
    and staleness metadata. Inputs are (T, N, ...) tensors.

    The visual projection (DINO → d_visual) is done INSIDE the fusion MLP,
    so gradients from the downstream loss flow back through it.

    Args:
        d_struct: Structural token dim.
        d_detector_roi: Raw DINO ROI feature dim (buffer stores raw features).
        d_model: Output token dim.
        d_camera: Camera-relative feature dim (from CameraPoseEncoder).
    """

    def __init__(
        self,
        d_struct: int = 256,
        d_detector_roi: int = 1024,
        d_model: int = 256,
        d_camera: int = 128,
    ):
        super().__init__()
        self.d_camera = d_camera

        # Fuse: geometry + raw buffered visual + camera + log_staleness
        fusion_input_dim = d_struct + d_detector_roi + d_camera + 1
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        buffer_features: torch.Tensor,
        valid_mask: torch.Tensor,
        cam_feats: Optional[torch.Tensor] = None,
        staleness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            geometry_tokens: (T, N, d_struct) — LIVE per-object structural tokens.
            buffer_features: (T, N, d_detector_roi) — raw DINO from LKS buffer (detached).
            valid_mask: (T, N) bool — True for real objects.
            cam_feats: (T, N, d_camera) or None — per-object camera-relative features.
            staleness: (T, N) long or None — frames to nearest visible.

        Returns:
            tokens: (T, N, d_model) — hybrid tokens.
        """
        T, N = geometry_tokens.shape[:2]
        device = geometry_tokens.device

        # Camera-relative features (zeros if not provided)
        if cam_feats is None:
            cam_feats = torch.zeros(T, N, self.d_camera, device=device)

        # Log-scaled staleness (default: 0)
        if staleness is None:
            staleness = torch.zeros(T, N, dtype=torch.long, device=device)
        log_staleness = torch.log(staleness.float() + 1.0).unsqueeze(-1)  # (T, N, 1)

        # Concatenate all inputs (visual projection happens inside fusion_proj)
        fused = torch.cat([
            geometry_tokens, buffer_features, cam_feats,
            log_staleness,
        ], dim=-1)  # (B, N, fusion_input_dim)
        tokens = self.fusion_proj(fused)  # (B, N, d_model)

        # Zero out padding
        tokens = tokens * valid_mask.unsqueeze(-1).float()

        return tokens
