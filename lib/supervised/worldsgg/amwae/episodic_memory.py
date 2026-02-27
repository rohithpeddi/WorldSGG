"""
Episodic Associative Memory Bank
==================================

FIFO queue storing unmasked (visible) tokens from recent frames.
Acts as an episodic temporal database of what objects looked like
and where they were before the camera panned away.

Unlike GL-STGN's GRU states, this is a raw token store — the memory
itself holds no trainable parameters. Only the retrieval (cross-attention
queries against this memory) is differentiable.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EpisodicMemoryBank(nn.Module):
    """
    External memory matrix maintaining unmasked tokens from recent frames.

    Structure:
        memory_tokens: (capacity, d_model) — stored visible object tokens
        object_ids:    (capacity,) — which object slot each entry belongs to
        frame_ids:     (capacity,) — which frame each entry came from
        valid:         (capacity,) — whether each slot is occupied

    Operations:
        store(): Push visible tokens from current frame into the FIFO
        retrieve(): Return all stored tokens for cross-attention K/V
        reset(): Clear the memory (between videos)

    Args:
        capacity: Maximum number of tokens to store (memory_bank_size * max_objects).
        d_model: Token dimension.
    """

    def __init__(self, capacity: int, d_model: int):
        super().__init__()
        self.capacity = capacity
        self.d_model = d_model

        # Buffers (non-trainable, persistent across forward calls)
        self.register_buffer(
            "memory_tokens",
            torch.zeros(capacity, d_model),
        )
        self.register_buffer(
            "object_ids",
            torch.full((capacity,), -1, dtype=torch.long),
        )
        self.register_buffer(
            "frame_ids",
            torch.full((capacity,), -1, dtype=torch.long),
        )
        self.register_buffer(
            "valid",
            torch.zeros(capacity, dtype=torch.bool),
        )
        self.register_buffer(
            "write_ptr",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "num_stored",
            torch.tensor(0, dtype=torch.long),
        )

    def reset(self):
        """Clear the memory bank (call between videos)."""
        self.memory_tokens.zero_()
        self.object_ids.fill_(-1)
        self.frame_ids.fill_(-1)
        self.valid.fill_(False)
        self.write_ptr.zero_()
        self.num_stored.zero_()

    @torch.no_grad()
    def store(
        self,
        tokens: torch.Tensor,
        object_slot_ids: torch.Tensor,
        visibility_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        frame_id: int,
    ):
        """
        Push visible tokens from the current frame into the FIFO queue.

        Only stores tokens that are both valid AND visible (unmasked).
        Tokens are detached — no gradients flow through storage.

        Args:
            tokens: (N, d_model) — hybrid tokens BEFORE masking
                    (i.e., with real DINO features, not [MASK]).
            object_slot_ids: (N,) long — object slot indices (0..N-1).
            visibility_mask: (N,) bool — True for objects in FOV.
            valid_mask: (N,) bool — True for real objects.
            frame_id: int — current frame index.
        """
        # Select visible + valid tokens
        store_mask = visibility_mask & valid_mask  # (N,)
        indices = torch.where(store_mask)[0]
        n_to_store = indices.shape[0]

        if n_to_store == 0:
            return

        # Detach tokens — memory stores snapshots, not computation graph nodes
        tokens_to_store = tokens[indices].detach()
        ids_to_store = object_slot_ids[indices]

        # Write into circular buffer
        for i in range(n_to_store):
            ptr = self.write_ptr.item()
            self.memory_tokens[ptr] = tokens_to_store[i]
            self.object_ids[ptr] = ids_to_store[i]
            self.frame_ids[ptr] = frame_id
            self.valid[ptr] = True
            self.write_ptr = (self.write_ptr + 1) % self.capacity

        self.num_stored = min(
            self.num_stored + n_to_store,
            torch.tensor(self.capacity, device=self.num_stored.device),
        )

    def retrieve(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return all stored tokens for cross-attention.

        Returns:
            memory_tokens: (M, d_model) — stored tokens.
            object_ids: (M,) long — which object slot each token belongs to.
            frame_ids: (M,) long — which frame each token came from.
            memory_valid: (M,) bool — all True (only valid entries returned).
        """
        valid_mask = self.valid  # (capacity,)
        n = valid_mask.sum().item()

        if n == 0:
            device = self.memory_tokens.device
            return (
                torch.zeros(0, self.d_model, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.bool, device=device),
            )

        indices = torch.where(valid_mask)[0]
        return (
            self.memory_tokens[indices],
            self.object_ids[indices],
            self.frame_ids[indices],
            torch.ones(n, dtype=torch.bool, device=self.memory_tokens.device),
        )

    @property
    def is_empty(self) -> bool:
        """Check if the memory bank has any stored entries."""
        return self.num_stored.item() == 0

    @property
    def fill_level(self) -> int:
        """Number of tokens currently stored."""
        return self.valid.sum().item()
