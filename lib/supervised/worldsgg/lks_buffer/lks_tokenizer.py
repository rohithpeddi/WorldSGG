"""
LKS Tokenizer
===============

Fuses CURRENT wireframe geometry (live from the global wireframe)
with BUFFERED visual features (potentially stale from the LKS buffer),
CAMERA-RELATIVE features, and OBSERVABILITY state.

X_k = Proj([G_k^t ⊕ M_t[k] ⊕ cam_feats_k ⊕ obs_onehot_k ⊕ log_staleness_k])

Where M_t[k] is:
  - Fresh DINO features (if object was just seen — obs_state=VISIBLE)
  - Stale DINO features (if object was seen N frames ago — obs_state=OCCLUDED or OUT_OF_FRUSTUM)
  - Zeros (if object has never been seen — obs_state=NEVER_SEEN)

Observability state telling the model WHY the visual component
may be unreliable: out-of-frustum, occluded, or never seen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LKSTokenizer(nn.Module):
    """
    Fuses live geometry with buffered visual features, camera features,
    and observability metadata.

    The observability state and staleness let the model know whether
    the buffered features are trustworthy (fresh visible), somewhat
    trustworthy (recently occluded), or unreliable (stale/never seen).

    Args:
        d_struct: Structural token dim.
        d_visual: Projected visual feature dim (buffer dim).
        d_model: Output token dim.
        d_camera: Camera-relative feature dim (from CameraPoseEncoder).
        n_obs_states: Number of observability states (4).
    """

    def __init__(
        self,
        d_struct: int = 256,
        d_visual: int = 256,
        d_model: int = 256,
        d_camera: int = 128,
        n_obs_states: int = 4,
    ):
        super().__init__()
        self.d_camera = d_camera
        self.n_obs_states = n_obs_states

        # Fuse: geometry + buffered visual + camera + obs_state_onehot + log_staleness
        fusion_input_dim = d_struct + d_visual + d_camera + n_obs_states + 1
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        buffer_features: torch.Tensor,
        valid_mask: torch.Tensor,
        cam_feats: Optional[torch.Tensor] = None,
        obs_state: Optional[torch.Tensor] = None,
        staleness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            geometry_tokens: (N, d_struct) — LIVE per-object structural tokens.
            buffer_features: (N, d_visual) — from LKS buffer (detached, possibly stale).
            valid_mask: (N,) bool — True for real objects.
            cam_feats: (N, d_camera) or None — per-object camera-relative features.
            obs_state: (N,) long or None — observability state per object (0-3).
            staleness: (N,) long or None — frames since last visible.

        Returns:
            tokens: (N, d_model) — hybrid tokens.
        """
        N = geometry_tokens.shape[0]
        device = geometry_tokens.device

        # Camera-relative features (zeros if not provided)
        if cam_feats is None:
            cam_feats = torch.zeros(N, self.d_camera, device=device)

        # Observability one-hot (default: all VISIBLE)
        if obs_state is None:
            obs_state = torch.full((N,), 3, dtype=torch.long, device=device)  # VISIBLE
        obs_onehot = F.one_hot(obs_state, num_classes=self.n_obs_states).float()  # (N, 4)

        # Log-scaled staleness (default: 0)
        if staleness is None:
            staleness = torch.zeros(N, dtype=torch.long, device=device)
        log_staleness = torch.log(staleness.float() + 1.0).unsqueeze(-1)  # (N, 1)

        # Concatenate all inputs
        fused = torch.cat([
            geometry_tokens, buffer_features, cam_feats,
            obs_onehot, log_staleness,
        ], dim=-1)
        tokens = self.fusion_proj(fused)  # (N, d_model)

        # Zero out padding
        tokens = tokens * valid_mask.unsqueeze(-1).float()

        return tokens


