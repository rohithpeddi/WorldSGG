"""
Per-Object Episodic Memory
============================

Replaces the FIFO-based EpisodicMemoryBank with per-object slot memory
that supports viewpoint diversity eviction.

Key improvements over FIFO:
  1. Each object gets K dedicated slots (no slot collision between objects)
  2. When slots fill, evict the entry whose capture pose is MOST SIMILAR
     to the new entry → retains viewpoint diversity
  3. Each entry stores both the feature token and capture camera pose

This enables view-aware retrieval: the model can query memory entries
from diverse viewpoints rather than just the most recent observations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PerObjectEpisodicMemory(nn.Module):
    """
    Per-object episodic memory bank with viewpoint diversity eviction.

    Structure: (max_objects, slots_per_object, d_memory) tensor
    Each object's memory is independent — no cross-object contamination.

    When all K slots for object i are full:
      → evict the entry whose capture pose is most similar to the new one
      → this retains diverse viewpoints in memory

    Args:
        max_objects: Maximum number of tracked objects.
        slots_per_object: Number of memory slots per object (K).
        d_memory: Feature dimension per memory entry.
    """

    def __init__(
        self,
        max_objects: int = 64,
        slots_per_object: int = 5,
        d_memory: int = 256,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.slots_per_object = slots_per_object
        self.d_memory = d_memory

        # Memory bank: (max_objects, K, d_memory)
        self.register_buffer(
            "memory",
            torch.zeros(max_objects, slots_per_object, d_memory),
        )
        # Capture poses: (max_objects, K, 4, 4) — pose when each entry was stored
        self.register_buffer(
            "capture_poses",
            torch.zeros(max_objects, slots_per_object, 4, 4),
        )
        # Slot occupancy: (max_objects, K) — True if slot is filled
        self.register_buffer(
            "slot_filled",
            torch.zeros(max_objects, slots_per_object, dtype=torch.bool),
        )
        # Slot count per object: (max_objects,) — number of filled slots
        self.register_buffer(
            "slot_count",
            torch.zeros(max_objects, dtype=torch.long),
        )

    def reset(self):
        """Reset all memory (call between videos)."""
        self.memory.zero_()
        self.capture_poses.zero_()
        self.slot_filled.fill_(False)
        self.slot_count.zero_()

    @torch.no_grad()
    def store(
        self,
        object_indices: torch.Tensor,
        features: torch.Tensor,
        camera_pose: torch.Tensor,
    ):
        """
        Store features for visible objects.

        If all slots are full for an object, evict the entry with
        the most similar capture pose (smallest delta) to preserve
        viewpoint diversity.

        Args:
            object_indices: (M,) long — indices of objects to store.
            features: (M, d_memory) — feature tokens to store.
            camera_pose: (4, 4) — current camera pose (same for all objects).
        """
        M = object_indices.shape[0]

        for i in range(M):
            obj_idx = object_indices[i].item()
            feat = features[i].detach()

            if self.slot_count[obj_idx] < self.slots_per_object:
                # Free slot available — append
                slot = self.slot_count[obj_idx].item()
                self.memory[obj_idx, slot] = feat
                self.capture_poses[obj_idx, slot] = camera_pose
                self.slot_filled[obj_idx, slot] = True
                self.slot_count[obj_idx] += 1
            else:
                # All slots full — evict most similar pose
                stored_poses = self.capture_poses[obj_idx]  # (K, 4, 4)

                # Compute pose similarity: smaller delta = more similar = evict
                deltas = self._compute_pose_deltas(camera_pose, stored_poses)  # (K,)

                # Evict the entry with SMALLEST delta (most similar viewpoint)
                evict_slot = deltas.argmin().item()
                self.memory[obj_idx, evict_slot] = feat
                self.capture_poses[obj_idx, evict_slot] = camera_pose

    def retrieve(
        self,
        object_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve all stored memory entries for given objects.

        Args:
            object_indices: (M,) long — object indices to retrieve

        Returns:
            memory_entries: (M, K, d_memory) — stored features per object
            entry_poses: (M, K, 4, 4) — capture poses per slot
            entry_mask: (M, K) bool — True for filled slots
        """
        memory_entries = self.memory[object_indices]      # (M, K, d_memory)
        entry_poses = self.capture_poses[object_indices]   # (M, K, 4, 4)
        entry_mask = self.slot_filled[object_indices]      # (M, K)

        return memory_entries, entry_poses, entry_mask

    def get_all_filled_count(self) -> torch.Tensor:
        """Return number of filled slots per object."""
        return self.slot_count

    @staticmethod
    def _compute_pose_deltas(
        current_pose: torch.Tensor,
        stored_poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scalar pose delta between current and each stored pose.

        Args:
            current_pose: (4, 4) — current camera pose.
            stored_poses: (K, 4, 4) — stored capture poses.

        Returns:
            deltas: (K,) — scalar pose delta per stored entry.
        """
        K = stored_poses.shape[0]
        R_curr = current_pose[:3, :3]
        t_curr = current_pose[:3, 3]

        R_stored = stored_poses[:, :3, :3]  # (K, 3, 3)
        t_stored = stored_poses[:, :3, 3]   # (K, 3)

        # Rotation delta: Frobenius norm of (R_curr @ R_stored^T - I)
        R_diff = R_curr.unsqueeze(0) @ R_stored.transpose(-1, -2)  # (K, 3, 3)
        eye = torch.eye(3, device=R_diff.device).unsqueeze(0)
        rot_delta = (R_diff - eye).flatten(start_dim=1).norm(dim=-1)  # (K,)

        # Translation delta
        trans_delta = (t_curr.unsqueeze(0) - t_stored).norm(dim=-1)  # (K,)

        return rot_delta + trans_delta
