"""
LKS Memory Buffer
===================

Non-differentiable zero-order hold memory buffer.
The core mechanic: "objects freeze when the camera looks away."

Update rule (hard-coded, NO neural parameters):
  - Visible (i ∈ FOV):  M[i] ← V_i^t   (overwrite with fresh features)
  - Not visible:         M[i] ← M[i]    (keep stale, staleness++)
  - Never seen:          M[i] = zeros    (no prior info)

All operations use .detach() — NO gradients flow through time.
This is NOT an nn.Module. It is purely programmatic state.
"""

import torch
from typing import Optional


class LKSMemoryBuffer:
    """
    Persistent state buffer storing the "last known" visual feature
    for every object in the scene.

    Args:
        max_objects: Maximum number of object slots.
        d_visual: Dimension of projected visual features.
        device: Torch device.
    """

    def __init__(
        self,
        max_objects: int,
        d_visual: int,
        device: str = "cpu",
    ):
        self.max_objects = max_objects
        self.d_visual = d_visual
        self.device = torch.device(device)

        # The buffer: (N, d_visual) — initialized to zeros ("fog of war")
        self.buffer = torch.zeros(
            max_objects, d_visual,
            dtype=torch.float32,
            device=self.device,
        )

        # Track how many frames since each object was last seen
        self.staleness = torch.zeros(
            max_objects, dtype=torch.long, device=self.device,
        )

        # Track whether each object has EVER been seen
        self.ever_seen = torch.zeros(
            max_objects, dtype=torch.bool, device=self.device,
        )

    def reset(self, N: int = None):
        """
        Reset buffer to fog of war (call at start of each video).

        Args:
            N: Optional new max_objects count. If None, reuses current size.
        """
        if N is not None and N != self.max_objects:
            self.max_objects = N
            self.buffer = torch.zeros(
                N, self.d_visual,
                dtype=torch.float32,
                device=self.device,
            )
            self.staleness = torch.zeros(N, dtype=torch.long, device=self.device)
            self.ever_seen = torch.zeros(N, dtype=torch.bool, device=self.device)
        else:
            self.buffer.zero_()
            self.staleness.zero_()
            self.ever_seen.fill_(False)

    @torch.no_grad()
    def update(
        self,
        visual_features: torch.Tensor,
        visibility_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        camera_pose: Optional[torch.Tensor] = None,
        corners: Optional[torch.Tensor] = None,
    ):
        """
        Zero-order hold update.

        Args:
            visual_features: (N, d_visual) — projected DINO features.
            visibility_mask: (N,) bool — True if detected this frame.
            valid_mask: (N,) bool — True for real objects.
            camera_pose: (4, 4) or None — current camera extrinsic (unused, kept for API compat).
            corners: (N, 8, 3) or None — 3D bbox corners (unused, kept for API compat).
        """
        N = visual_features.shape[0]
        assert N <= self.max_objects, \
            f"Got {N} objects but buffer has {self.max_objects} slots"

        # Hard overwrite for visible + valid objects
        overwrite_mask = visibility_mask & valid_mask  # (N,)
        self.buffer[:N][overwrite_mask] = visual_features[overwrite_mask].detach()

        # Track staleness
        self.staleness[:N] += 1  # Everything gets one frame older
        self.staleness[:N][overwrite_mask] = 0  # Reset for freshly seen

        # Track ever-seen
        self.ever_seen[:N] |= overwrite_mask

    def get_features(self, N: int = None) -> torch.Tensor:
        """
        Return current buffer state (detached).

        Args:
            N: Number of objects to return (default: all).

        Returns:
            (N, d_visual) — detached visual features.
        """
        if N is None:
            N = self.max_objects
        return self.buffer[:N].detach().clone()

    def get_staleness(self, N: int = None) -> torch.Tensor:
        """Return staleness counts (frames since last seen)."""
        if N is None:
            N = self.max_objects
        return self.staleness[:N]

    def get_ever_seen(self, N: int = None) -> torch.Tensor:
        """Return ever-seen mask."""
        if N is None:
            N = self.max_objects
        return self.ever_seen[:N]
