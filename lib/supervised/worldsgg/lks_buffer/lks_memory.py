"""
LKS Memory Buffer
===================

Non-differentiable zero-order hold memory buffer with observability tracking.
The core mechanic: "objects freeze when the camera looks away," but now
with awareness of WHY they're unseen (out of frustum vs occluded).

Update rule (hard-coded, NO neural parameters):
  - Visible (i ∈ FOV):      M[i] ← V_i^t          (overwrite with fresh features)
  - Occluded in-frustum:     M[i] ← M[i]           (keep stale — ROI unreliable)
  - Out of frustum:          M[i] ← M[i]           (keep stale, staleness++)
  - Never seen:              M[i] = zeros           (no prior info)

All operations use .detach() — NO gradients flow through time.
This is NOT an nn.Module. It is purely programmatic state.

Observability states:
  0 = NEVER_SEEN:      Object has never had a GT/GDino bbox
  1 = OUT_OF_FRUSTUM:  Object's center is behind/beside the camera
  2 = OCCLUDED:        In camera frustum but not detected (blocked)
  3 = VISIBLE:         Object has a detection this frame
"""

import torch
from typing import Optional


class LKSMemoryBuffer:
    """
    Persistent state buffer storing the "last known" visual feature and
    observability state for every object in the scene.

    Args:
        max_objects: Maximum number of object slots.
        d_visual: Dimension of projected visual features.
        device: Torch device.
        frustum_thresh: View-alignment threshold for in-frustum classification.
    """

    # Observability state constants (match ObservabilityClassifier)
    NEVER_SEEN = 0
    OUT_OF_FRUSTUM = 1
    OCCLUDED = 2
    VISIBLE = 3

    def __init__(
        self,
        max_objects: int,
        d_visual: int,
        device: str = "cpu",
        frustum_thresh: float = -0.1,
    ):
        self.max_objects = max_objects
        self.d_visual = d_visual
        self.device = torch.device(device)
        self.frustum_thresh = frustum_thresh

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

        # Current observability state per object
        self.obs_state = torch.zeros(
            max_objects, dtype=torch.long, device=self.device,
        )

        # Camera pose at which each object's features were captured
        # Used by FeatureAging to compute pose delta
        self.capture_poses = torch.zeros(
            max_objects, 4, 4,
            dtype=torch.float32, device=self.device,
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
            self.obs_state = torch.zeros(N, dtype=torch.long, device=self.device)
            self.capture_poses = torch.zeros(N, 4, 4, dtype=torch.float32, device=self.device)
        else:
            self.buffer.zero_()
            self.staleness.zero_()
            self.ever_seen.fill_(False)
            self.obs_state.zero_()
            self.capture_poses.zero_()

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
        Zero-order hold update with observability state tracking.

        Args:
            visual_features: (N, d_visual) — projected DINO features.
            visibility_mask: (N,) bool — True if in camera FOV.
            valid_mask: (N,) bool — True for real objects.
            camera_pose: (4, 4) or None — current camera extrinsic.
            corners: (N, 8, 3) or None — 3D bbox corners (for frustum check).
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

        # Record the camera pose at which features were captured
        if camera_pose is not None:
            self.capture_poses[:N][overwrite_mask] = camera_pose.unsqueeze(0).expand(
                overwrite_mask.sum().item(), -1, -1,
            )

        # --- Classify observability state ---
        self.obs_state[:N] = 0  # Default: NEVER_SEEN

        if camera_pose is not None and corners is not None:
            # Compute view alignment for frustum check
            R = camera_pose[:3, :3]
            cam_pos = camera_pose[:3, 3]
            view_dir = -R[:, 2]
            view_dir = view_dir / (view_dir.norm() + 1e-8)

            centers = corners.mean(dim=1)  # (N, 3)
            cam_to_obj = centers - cam_pos.unsqueeze(0)
            cam_to_obj_norm = cam_to_obj / (cam_to_obj.norm(dim=-1, keepdim=True) + 1e-8)
            view_alignment = (cam_to_obj_norm * view_dir.unsqueeze(0)).sum(dim=-1)

            in_frustum = view_alignment > self.frustum_thresh

            # Seen before but not visible now
            seen_but_invisible = self.ever_seen[:N] & (~visibility_mask) & valid_mask
            self.obs_state[:N][seen_but_invisible & (~in_frustum)] = self.OUT_OF_FRUSTUM
            self.obs_state[:N][seen_but_invisible & in_frustum] = self.OCCLUDED
        else:
            # Fallback: no camera info → use binary visible/unseen
            seen_but_invisible = self.ever_seen[:N] & (~visibility_mask) & valid_mask
            self.obs_state[:N][seen_but_invisible] = self.OUT_OF_FRUSTUM

        # Visible objects
        self.obs_state[:N][overwrite_mask] = self.VISIBLE

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

    def get_obs_state(self, N: int = None) -> torch.Tensor:
        """Return per-object observability state (0-3)."""
        if N is None:
            N = self.max_objects
        return self.obs_state[:N]

    def get_capture_poses(self, N: int = None) -> torch.Tensor:
        """
        Return per-object capture poses (camera pose when features were stored).

        Returns:
            (N, 4, 4) — per-object capture camera poses.
        """
        if N is None:
            N = self.max_objects
        return self.capture_poses[:N]

