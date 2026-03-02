"""
LKS Memory Buffer (Vectorized)
================================

Non-differentiable zero-order hold memory — fully vectorized over T frames.
Replaces the sequential LKSMemoryBuffer class with a pure function.

Update rule per object n at frame t:
  - If visible at t:  buffer[t,n] = projected_visual[t,n]  (fresh)
  - If not visible:   buffer[t,n] = buffer[last_seen,n]    (stale)
  - If never seen:    buffer[t,n] = zeros                  (fog of war)

Implementation uses cummax to find the most recent visible frame for
each (t, n) pair, then advanced indexing to gather features in O(1).
"""

import torch


@torch.no_grad()
def vectorized_lks_buffer(
    projected_visual: torch.Tensor,
    visibility_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple:
    """
    Compute the LKS buffer state for ALL frames simultaneously.

    Args:
        projected_visual: (T, N, d_visual) — projected DINO features per frame.
        visibility_mask:  (T, N) bool — True if object detected this frame.
        valid_mask:       (T, N) bool — True for real (non-padding) objects.

    Returns:
        buffer_features: (T, N, d_visual) — detached buffered features.
        staleness:       (T, N) long — frames since last seen (0 = just seen).
    """
    T, N, D = projected_visual.shape
    device = projected_visual.device

    # Overwrite mask: visible AND valid
    overwrite = visibility_mask & valid_mask  # (T, N)

    # Frame indices: (T, 1) broadcast to (T, N)
    frame_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, N)  # (T, N)

    # For each (t, n): frame index if overwrite, else -1
    last_seen_raw = torch.where(overwrite, frame_ids, torch.full_like(frame_ids, -1))  # (T, N)

    # Cumulative max propagates the most recent visible frame forward in time
    last_seen_at, _ = last_seen_raw.cummax(dim=0)  # (T, N)

    # Staleness: current frame - last seen frame (0 = just seen)
    staleness = frame_ids - last_seen_at  # (T, N)

    # Never-seen objects have last_seen_at == -1
    never_seen = last_seen_at < 0  # (T, N)
    staleness = staleness.clamp(min=0)

    # Gather features from the last-seen frame
    # Clamp indices for safe gather (never-seen will be zeroed out below)
    gather_idx = last_seen_at.clamp(min=0)  # (T, N)

    # Advanced indexing: buffer_features[t, n] = projected_visual[gather_idx[t,n], n]
    n_idx = torch.arange(N, device=device).unsqueeze(0).expand(T, N)  # (T, N)
    buffer_features = projected_visual[gather_idx, n_idx]  # (T, N, D)

    # Zero out never-seen objects (fog of war)
    buffer_features = buffer_features.masked_fill(never_seen.unsqueeze(-1), 0.0)

    # Zero out invalid (padding) objects
    buffer_features = buffer_features * valid_mask.unsqueeze(-1).float()

    return buffer_features.detach(), staleness
