"""
Global Structural Encoder
=========================

Encodes the complete 3D bounding box layout (the "wireframe") of all objects
into dense structural tokens using a PointNet-style architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalStructuralEncoder(nn.Module):
    """
    Encodes world-frame 3D bounding boxes for ALL objects (visible + unseen)
    into per-object structural tokens and a global summary token.

    The "wireframe" is represented as the complete set of 3D bounding boxes
    across all objects at a given timestamp.

    Args:
        d_struct: Output structural token dimension.
        d_hidden: Hidden dimension for internal MLPs.

    Input:
        corners: (B, N, 8, 3) — 3D bounding box corners in world frame.
        valid_mask: (B, N) bool — True for real objects, False for padding.

    Output:
        object_tokens: (B, N, d_struct) — per-object structural token.
        global_token:  (B, d_struct) — aggregated global structural summary.
    """

    def __init__(self, d_struct: int = 256, d_hidden: int = 128):
        super().__init__()
        self.d_struct = d_struct

        # Per-corner point encoder: (3) → (64) → (d_hidden)
        self.corner_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_hidden),
            nn.ReLU(inplace=True),
        )

        # Per-object encoder (after max-pool over 8 corners): (d_hidden) → d_struct
        self.object_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden * 2),
            nn.Linear(d_hidden * 2, d_struct),
        )

        # Global summary: (d_struct) → (d_struct)
        self.global_mlp = nn.Sequential(
            nn.Linear(d_struct, d_struct),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_struct),
            nn.Linear(d_struct, d_struct),
        )

    def forward(
        self,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple:
        """
        Args:
            corners: (B, N, 8, 3) world-frame 3D bbox corners.
            valid_mask: (B, N) bool — True for real objects.

        Returns:
            object_tokens: (B, N, d_struct)
            global_token:  (B, d_struct)
        """
        B, N, C, D = corners.shape  # C=8 corners, D=3 dims
        assert C == 8 and D == 3, f"Expected (B, N, 8, 3), got {corners.shape}"

        # Encode each corner point independently
        # (B, N, 8, 3) → (B*N*8, 3) → MLP → (B*N*8, d_hidden) → (B, N, 8, d_hidden)
        x = corners.reshape(B * N * C, D)
        x = self.corner_mlp(x)
        x = x.view(B, N, C, -1)  # (B, N, 8, d_hidden)

        # Max-pool over corners → per-object feature
        # Mask invalid objects (set to -inf before max-pool)
        mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        x = x.masked_fill(~mask_expanded, float("-inf"))
        x, _ = x.max(dim=2)  # (B, N, d_hidden) — max over 8 corners

        # Replace -inf (from fully-invalid objects) with zeros
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        # Per-object structural token
        object_tokens = self.object_mlp(x)  # (B, N, d_struct)
        object_tokens = object_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        # Global summary: masked max-pool over all objects
        global_pool = object_tokens.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
        global_pool, _ = global_pool.max(dim=1)  # (B, d_struct)
        # If all objects are invalid (shouldn't happen), replace with zeros
        all_invalid = ~valid_mask.any(dim=1, keepdim=True)  # (B, 1)
        global_pool = global_pool.masked_fill(all_invalid, 0.0)
        global_token = self.global_mlp(global_pool)  # (B, d_struct)

        return object_tokens, global_token
