"""
Amnesic Tokenizer
==================

Binary feature assignment for the Amnesic Geometric GNN baseline.
Visible objects get real DINO features; unseen objects get a **conditioned**
surrogate embedding generated from geometry, camera, observability type,
and staleness — NOT a single global vector.

This means out-of-frustum, occluded, and never-seen objects produce
distinct representations, reducing the burden on the downstream GNN.
"""

import torch
import torch.nn as nn
from typing import Optional


class UnseenSurrogateGenerator(nn.Module):
    """
    Generates per-object unseen visual surrogates conditioned on:
      - geometry_token: structural context from 3D bbox
      - cam_feats: camera-relative position/angle
      - obs_type_onehot: {NEVER_SEEN, OUT_OF_FRUSTUM, OCCLUDED, VISIBLE} one-hot
      - log_staleness: log(frames since last visible + 1)

    Replaces the old single learnable E_unseen vector with a context-aware
    generator that produces different embeddings for different reasons an
    object might be unseen.

    Args:
        d_struct: Structural token dimension.
        d_camera: Camera-relative feature dimension.
        d_visual: Output visual surrogate dimension.
        n_obs_states: Number of observability states (4).
        d_hidden: Hidden dimension in the generator MLP.
    """

    def __init__(
        self,
        d_struct: int = 256,
        d_camera: int = 128,
        d_visual: int = 256,
        n_obs_states: int = 4,
        d_hidden: int = 128,
    ):
        super().__init__()
        # Input: geometry(d_struct) + cam_feats(d_camera) + obs_type(n_obs_states) + staleness(1)
        input_dim = d_struct + d_camera + n_obs_states + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_visual),
        )

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        cam_feats: torch.Tensor,
        obs_type_onehot: torch.Tensor,
        log_staleness: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            geometry_tokens: (N, d_struct) — per-object structural tokens.
            cam_feats: (N, d_camera) — camera-relative features.
            obs_type_onehot: (N, 4) — observability type one-hot.
            log_staleness: (N, 1) — log(frames since last visible + 1).

        Returns:
            unseen_surrogates: (N, d_visual) — conditioned unseen embeddings.
        """
        x = torch.cat([geometry_tokens, cam_feats, obs_type_onehot, log_staleness], dim=-1)
        return self.mlp(x)


class AmnesicTokenizer(nn.Module):
    """
    Creates hybrid tokens from geometry, visual features, camera features,
    and observability-conditioned unseen surrogates.

    Visible:   token = Proj([geometry_i ⊕ DINO_i ⊕ cam_feats_i])
    Unseen:    token = Proj([geometry_j ⊕ Surrogate_j(geo, cam, obs, staleness) ⊕ cam_feats_j])

    Args:
        d_struct: Structural token dim (from GlobalStructuralEncoder).
        d_visual: Projected visual feature dim.
        d_model: Output token dim.
        d_detector_roi: Raw DINO ROI feature dim (1024).
        d_camera: Camera-relative feature dim (from CameraPoseEncoder).
    """

    def __init__(
        self,
        d_struct: int = 256,
        d_visual: int = 256,
        d_model: int = 256,
        d_detector_roi: int = 1024,
        d_camera: int = 128,
    ):
        super().__init__()
        self.d_visual = d_visual
        self.d_camera = d_camera

        # Project raw DINO ROI features → d_visual
        self.visual_projector = nn.Sequential(
            nn.Linear(d_detector_roi, d_visual),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_visual),
        )

        # Conditioned unseen surrogate generator (replaces global E_unseen)
        self.unseen_generator = UnseenSurrogateGenerator(
            d_struct=d_struct,
            d_camera=d_camera,
            d_visual=d_visual,
        )

        # Fuse geometry + visual/unseen + camera features → d_model
        fusion_input_dim = d_struct + d_visual + d_camera
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
        cam_feats: Optional[torch.Tensor] = None,
        obs_type: Optional[torch.Tensor] = None,
        staleness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            geometry_tokens: (N, d_struct) — per-object structural tokens.
            visual_features: (N, d_detector_roi) — raw DINO ROI features.
            visibility_mask: (N,) bool — True if in camera FOV.
            valid_mask: (N,) bool — True for real objects.
            cam_feats: (N, d_camera) or None — per-object camera-relative features.
            obs_type: (N,) long or None — observability state (0-3).
                      From ObservabilityClassifier. If None, uses binary visible/unseen.
            staleness: (N,) long or None — frames since last visible.
                       If None, defaults to 0 for visible, 1 for unseen.

        Returns:
            tokens: (N, d_model) — hybrid tokens.
        """
        N = geometry_tokens.shape[0]
        device = geometry_tokens.device

        # Project visual features
        visual_proj = self.visual_projector(visual_features)  # (N, d_visual)

        # Camera-relative features (zeros if not provided)
        if cam_feats is None:
            cam_feats = torch.zeros(N, self.d_camera, device=device)

        # Observability type (default: binary visible=3, unseen=0)
        if obs_type is None:
            obs_type = torch.where(
                visibility_mask,
                torch.tensor(3, device=device),  # VISIBLE
                torch.tensor(0, device=device),   # NEVER_SEEN (generic unseen)
            )

        # Staleness (default: 0 for visible, 1 for unseen)
        if staleness is None:
            staleness = torch.where(
                visibility_mask,
                torch.zeros(N, device=device, dtype=torch.long),
                torch.ones(N, device=device, dtype=torch.long),
            )

        # Generate unseen surrogates
        obs_onehot = torch.nn.functional.one_hot(obs_type, num_classes=4).float()  # (N, 4)
        log_stale = torch.log(staleness.float() + 1.0).unsqueeze(-1)  # (N, 1)
        unseen_surrogates = self.unseen_generator(
            geometry_tokens, cam_feats, obs_onehot, log_stale,
        )  # (N, d_visual)

        # Select: visible → projected DINO, unseen → conditioned surrogate
        visual_component = torch.where(
            visibility_mask.unsqueeze(-1),
            visual_proj,
            unseen_surrogates,
        )  # (N, d_visual)

        # Fuse geometry + visual component + camera features
        fused = torch.cat([geometry_tokens, visual_component, cam_feats], dim=-1)
        tokens = self.fusion_proj(fused)  # (N, d_model)

        # Zero out padding
        tokens = tokens * valid_mask.unsqueeze(-1).float()

        return tokens


