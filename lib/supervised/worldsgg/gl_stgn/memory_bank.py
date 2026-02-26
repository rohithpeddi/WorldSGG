"""
Persistent World Memory Bank
=============================

GRU-based memory bank that maintains latent states for ALL discovered objects
across time. Updates visible objects with visual features and structural tokens,
updates unseen objects via cross-attention to the global structural prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentWorldMemoryBank(nn.Module):
    """
    Maintains a memory bank M_t of per-object hidden states across time.

    For each timestep:
      1. Data Association: match incoming observations to existing memory nodes
         (GT track IDs during training / 3D IoU + cosine sim at inference).
      2. State Update:
         - Seen objects: GRU(prev_state, [visual_feat || struct_token])
         - Unseen objects: GRU(prev_state, [zeros || cross_attn(prev, global_struct)])
      3. New objects: Initialize memory from first observation.

    Args:
        d_memory: Hidden state dimension for memory bank.
        d_visual: Projected visual feature dimension.
        d_struct: Structural token dimension.
        d_detector_roi: Raw detector ROI feature dimension (before projection).
        n_heads: Number of attention heads for unseen cross-attention.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_memory: int = 256,
        d_visual: int = 256,
        d_struct: int = 256,
        d_detector_roi: int = 1024,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_memory = d_memory
        self.d_visual = d_visual
        self.d_struct = d_struct

        # Project raw DINO ROI features to d_visual
        self.visual_projector = nn.Sequential(
            nn.Linear(d_detector_roi, d_visual),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_visual),
        )

        # GRU for state update (input = visual + struct concatenated)
        self.update_gru = nn.GRUCell(
            input_size=d_visual + d_struct,
            hidden_size=d_memory,
        )

        # Cross-attention for unseen objects: query=prev_state, key/value=global_struct
        self.unseen_cross_attn = nn.MultiheadAttention(
            embed_dim=d_memory,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Project global structural tokens to d_memory for cross-attention K/V
        self.struct_to_kv = nn.Linear(d_struct, d_memory)

        # Initialize new object memory from first observation
        self.init_mlp = nn.Sequential(
            nn.Linear(d_visual + d_struct, d_memory),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_memory),
        )

        # For unseen objects, project cross-attention output to GRU input dim
        self.unseen_input_proj = nn.Linear(d_memory, d_visual)

    def initialize_memory(
        self,
        visual_features: torch.Tensor,
        struct_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize memory for the first frame.

        Args:
            visual_features: (N, d_detector_roi) raw DINO ROI features, or zeros for
                             objects without visual observations.
            struct_tokens: (N, d_struct) per-object structural tokens.
            valid_mask: (N,) bool — True for real objects.

        Returns:
            memory: (N, d_memory) initial memory states.
        """
        N = struct_tokens.shape[0]
        device = struct_tokens.device

        # Project visual features
        vis = self.visual_projector(visual_features)  # (N, d_visual)

        # Concatenate and initialize
        combined = torch.cat([vis, struct_tokens], dim=-1)  # (N, d_visual + d_struct)
        memory = self.init_mlp(combined)  # (N, d_memory)

        # Zero out padding slots
        memory = memory * valid_mask.unsqueeze(-1).float()
        return memory

    def step(
        self,
        memory: torch.Tensor,
        visual_features: torch.Tensor,
        struct_tokens: torch.Tensor,
        global_struct_tokens: torch.Tensor,
        visibility_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        p_mask_visual: float = 0.0,
    ) -> torch.Tensor:
        """
        Update memory for one timestep.

        Args:
            memory: (N, d_memory) previous memory states.
            visual_features: (N, d_detector_roi) raw DINO features.
                             Zeros for objects not detected this frame.
            struct_tokens: (N, d_struct) per-object structural tokens this frame.
            global_struct_tokens: (N_all, d_struct) ALL structural tokens (for cross-attn).
            visibility_mask: (N,) bool — True if object is in camera FOV this frame.
            valid_mask: (N,) bool — True for real objects (not padding).
            p_mask_visual: Probability of masking visual features (training regularization).

        Returns:
            updated_memory: (N, d_memory)
        """
        N = memory.shape[0]
        device = memory.device

        # --- Project visual features ---
        vis = self.visual_projector(visual_features)  # (N, d_visual)

        # --- Masked visual context training ---
        if self.training and p_mask_visual > 0.0:
            mask_drop = torch.rand(N, device=device) < p_mask_visual
            # Only mask objects that ARE visible (simulates not seeing them)
            mask_drop = mask_drop & visibility_mask
            vis = vis.masked_fill(mask_drop.unsqueeze(-1), 0.0)

        # --- Seen objects: direct visual + struct input ---
        seen_input = torch.cat([vis, struct_tokens], dim=-1)  # (N, d_visual + d_struct)

        # --- Unseen objects: cross-attention to global structure ---
        # Use global structural tokens as context for unseen objects
        # query = previous memory state, key/value = global structural tokens
        N_global = global_struct_tokens.shape[0]

        # Project global struct to d_memory space for K/V
        global_kv = self.struct_to_kv(global_struct_tokens)  # (N_global, d_memory)

        # Cross-attention: each unseen object queries the global structure
        # Reshape for batch attention: (1, N, d_memory) queries (1, N_global, d_memory)
        query = memory.unsqueeze(0)  # (1, N, d_memory)
        key_value = global_kv.unsqueeze(0)  # (1, N_global, d_memory)

        cross_attn_out, _ = self.unseen_cross_attn(
            query=query,
            key=key_value,
            value=key_value,
        )
        cross_attn_out = cross_attn_out.squeeze(0)  # (N, d_memory)

        # Project cross-attention output to visual dimension for unseen input
        unseen_vis = self.unseen_input_proj(cross_attn_out)  # (N, d_visual)
        unseen_input = torch.cat([unseen_vis, struct_tokens], dim=-1)  # (N, d_visual + d_struct)

        # --- Select input based on visibility ---
        # seen → direct visual+struct, unseen → cross-attn+struct
        vis_mask = visibility_mask.unsqueeze(-1).float()  # (N, 1)
        gru_input = vis_mask * seen_input + (1 - vis_mask) * unseen_input

        # --- GRU update ---
        updated_memory = self.update_gru(gru_input, memory)  # (N, d_memory)

        # Zero out padding slots
        updated_memory = updated_memory * valid_mask.unsqueeze(-1).float()

        return updated_memory

    def spawn_new_objects(
        self,
        memory: torch.Tensor,
        new_visual_features: torch.Tensor,
        new_struct_tokens: torch.Tensor,
        n_new: int,
    ) -> torch.Tensor:
        """
        Extend memory bank with newly discovered objects.

        Args:
            memory: (N_old, d_memory) existing memory.
            new_visual_features: (n_new, d_detector_roi) features for new objects.
            new_struct_tokens: (n_new, d_struct) structural tokens for new objects.
            n_new: Number of new objects.

        Returns:
            extended_memory: (N_old + n_new, d_memory)
        """
        if n_new == 0:
            return memory

        valid = torch.ones(n_new, device=memory.device, dtype=torch.bool)
        new_memory = self.initialize_memory(new_visual_features, new_struct_tokens, valid)
        return torch.cat([memory, new_memory], dim=0)
