"""
LKS Memory Buffer (Bidirectional, Vectorized)
================================================

Non-differentiable zero-order hold memory — fully vectorized over T frames.
For each unseen object, picks features from the **nearest** visible frame
in either temporal direction (past or future).

Update rule per object n at frame t:
  - If visible at t:  buffer[t,n] = projected_visual[t,n]  (fresh)
  - If not visible:   buffer[t,n] = projected_visual[nearest_seen, n]
  - If never seen:    buffer[t,n] = zeros                  (fog of war)

Implementation uses forward cummax + reverse cummax to find the closest
visible frame in either direction, then picks the nearest.
"""

import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def vectorized_lks_buffer(
    projected_visual: torch.Tensor,
    visibility_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple:
    """
    Compute the bidirectional LKS buffer state for ALL frames simultaneously.

    For each unseen (t, n), finds the closest visible frame in either
    temporal direction and copies features from that frame.

    Args:
        projected_visual: (T, N, d_visual) — projected DINO features per frame.
        visibility_mask:  (T, N) bool — True if object detected this frame.
        valid_mask:       (T, N) bool — True for real (non-padding) objects.

    Returns:
        buffer_features: (T, N, d_visual) — detached buffered features.
        staleness:       (T, N) long — frames to nearest visible (0 = just seen).
    """
    T, N, D = projected_visual.shape
    device = projected_visual.device

    # Overwrite mask: visible AND valid
    overwrite = visibility_mask & valid_mask  # (T, N)

    # Frame indices: (T, N)
    frame_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, N)

    # ==================== Forward pass (past) ====================
    # For each (t, n): most recent frame ≤ t where object was visible
    past_raw = torch.where(overwrite, frame_ids, torch.full_like(frame_ids, -1))
    last_seen_at, _ = past_raw.cummax(dim=0)  # (T, N)
    past_staleness = frame_ids - last_seen_at  # ≥ 0 where last_seen_at ≥ 0
    never_seen_past = last_seen_at < 0

    # ==================== Backward pass (future) ====================
    # For each (t, n): earliest frame ≥ t where object is visible
    # Use negative trick: flip, cummax on negated values, flip back
    future_raw = torch.where(overwrite, -frame_ids, torch.full_like(frame_ids, -(T + 1)))
    next_seen_neg, _ = future_raw.flip(0).cummax(dim=0)
    next_seen_at = -next_seen_neg.flip(0)  # (T, N) — frame index of next visible
    future_staleness = next_seen_at - frame_ids  # ≥ 0 where next_seen_at ≤ T-1
    never_seen_future = next_seen_at > (T - 1)

    # ==================== Pick nearest direction ====================
    # Prefer future when closer; fall back to past; handle never-seen
    past_staleness_safe = past_staleness.clone()
    past_staleness_safe[never_seen_past] = T + 1  # sentinel: very stale

    future_staleness_safe = future_staleness.clone()
    future_staleness_safe[never_seen_future] = T + 1  # sentinel: very stale

    use_future = future_staleness_safe < past_staleness_safe
    gather_idx = torch.where(use_future, next_seen_at, last_seen_at).clamp(min=0, max=T - 1)
    staleness = torch.where(use_future, future_staleness_safe, past_staleness_safe)

    # Never-seen in BOTH directions
    never_seen = never_seen_past & never_seen_future
    staleness = staleness.clamp(min=0)

    # ==================== Gather features ====================
    n_idx = torch.arange(N, device=device).unsqueeze(0).expand(T, N)
    buffer_features = projected_visual[gather_idx, n_idx]  # (T, N, D)

    # Zero out never-seen objects (fog of war)
    buffer_features = buffer_features.masked_fill(never_seen.unsqueeze(-1), 0.0)

    # Zero out invalid (padding) objects
    buffer_features = buffer_features * valid_mask.unsqueeze(-1).float()

    return buffer_features.detach(), staleness
