"""
Energy Diffusion (AMWAE++)
===========================

Weight-tied recurrent transformer that simulates a continuous dynamical
system descending an energy landscape. Replaces the standard fixed-depth
ContextualDiffusion with a single shared layer iterated until convergence.

Key properties:
  1. Weight tying — one shared TransformerEncoderLayer (75% param reduction)
  2. Spatial PE re-injection — 3D geometry acts as constant external field
  3. Dynamic stopping — at inference, exits early when L2 delta < epsilon
  4. Fixed unrolling at training — constant graph size for DDP compatibility

Returns both the final state and the penultimate state (for stability loss).
"""

import torch
import torch.nn as nn

from lib.supervised.worldsgg.worldsgg_base import SpatialPositionalEncoding


class EnergyDiffusion(nn.Module):
    """
    Weight-tied Energy Transformer for context propagation.

    Iterates a single shared self-attention layer with 3D spatial PE
    re-injected at every step. During inference, exits early when
    the representation converges (L2 delta < epsilon).

    Args:
        d_model: Token dimension.
        n_heads: Attention heads.
        d_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
        train_iters: Fixed number of iterations during training.
        eval_iters: Maximum iterations during inference.
        epsilon: Convergence threshold (max L2 delta across valid tokens).

    Input:  tokens (B, N, d_model), corners (B, N, 8, 3), valid_mask (B, N)
    Output: (enriched, h_prev) — enriched tokens + penultimate state
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_feedforward: int = 512,
        dropout: float = 0.1,
        train_iters: int = 4,
        eval_iters: int = 15,
        epsilon: float = 1e-3,
    ):
        super().__init__()
        self.d_model = d_model
        self.train_iters = train_iters
        self.eval_iters = eval_iters
        self.epsilon = epsilon

        # Constant spatial grounding (re-injected every iteration)
        self.spatial_pe = SpatialPositionalEncoding(d_model=d_model)

        # Single weight-tied layer — pre-LN is mathematically required
        # for stable recurrent energy landscapes
        self.shared_energy_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: critical for recurrent stability
        )

        # Post-norm applied after the final iteration
        self.post_norm = nn.LayerNorm(d_model)

        # Diagnostic: number of steps taken at last inference
        self.last_convergence_steps = 0

    def forward(
        self,
        tokens: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple:
        """
        Args:
            tokens: (B, N, d_model) — auto-completed tokens.
            corners: (B, N, 8, 3) — 3D bbox corners for spatial PE.
            valid_mask: (B, N) bool — True for real objects.

        Returns:
            enriched: (B, N, d_model) — converged representations.
            h_prev: (B, N, d_model) — penultimate state (detached,
                    for attractor stability loss during training).
        """
        # 1. Compute constant spatial grounding (external energy field)
        spatial_pe = self.spatial_pe(corners, valid_mask)  # (B, N, d_model)

        # 2. Padding mask
        padding_mask = ~valid_mask  # (B, N) — True = ignore

        # Failsafe for fully-padded batch items
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        # 3. Iterate the shared energy layer
        h_t = tokens
        h_prev = tokens
        max_iters = self.train_iters if self.training else self.eval_iters

        for step in range(max_iters):
            h_prev = h_t

            # Re-inject spatial PE at every step (constant external field)
            h_in = h_t + spatial_pe

            # One step down the energy gradient
            h_t = self.shared_energy_layer(h_in, src_key_padding_mask=padding_mask)

            # Dynamic convergence check (inference only)
            if not self.training:
                diff = torch.norm(h_t - h_prev, dim=-1)  # (B, N)
                diff = diff * valid_mask.float()
                max_delta = diff.max().item()

                if max_delta < self.epsilon:
                    self.last_convergence_steps = step + 1
                    break
        else:
            # Loop completed without break
            self.last_convergence_steps = max_iters

        # 4. Post-norm + strict zero-masking
        h_t = self.post_norm(h_t)
        h_t = h_t.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        return h_t, h_prev.detach()
