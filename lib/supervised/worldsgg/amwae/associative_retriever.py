"""
Associative Retriever (Batched, Bidirectional)
================================================

Per-object cross-attention that auto-completes masked tokens by
retrieving visual features from ALL visible appearances of that
object across ALL T frames.

For each object n:
  Q = masked tokens across T (frames where unseen)
  K/V = visible tokens across T (frames where visible)

This replaces the sequential episodic memory store/retrieve pattern.
All T frames are available simultaneously — no causal constraint.

The retriever also supports view-aware biasing: camera pose features
are added to Q (current viewpoint) and K (capture viewpoint) so the
attention naturally favours memory entries from similar viewpoints.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AssociativeRetriever(nn.Module):
    """
    Bidirectional per-object cross-attention over all T frames.

    For each object, masked tokens query visible tokens from ANY frame.
    Visible tokens are also refined (self-contextualized) via the same
    cross-attention mechanism.

    Supports view-aware retrieval via camera pose feature bias on Q/K.

    Args:
        d_model: Token dimension.
        n_layers: Number of cross-attention layers.
        n_heads: Attention heads.
        d_feedforward: FFN hidden dim.
        dropout: Dropout probability.
        d_camera: Camera feature dim for view-aware Q/K bias.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        d_feedforward: int = 512,
        dropout: float = 0.1,
        d_camera: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_camera = d_camera

        # Cross-attention: Q = all tokens, K/V = visible tokens only
        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms1 = nn.ModuleList()
        self.cross_norms2 = nn.ModuleList()
        self.cross_ffns = nn.ModuleList()

        for _ in range(n_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=n_heads,
                    dropout=dropout, batch_first=True,
                )
            )
            self.cross_norms1.append(nn.LayerNorm(d_model))
            self.cross_norms2.append(nn.LayerNorm(d_model))
            self.cross_ffns.append(nn.Sequential(
                nn.Linear(d_model, d_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_feedforward, d_model),
                nn.Dropout(dropout),
            ))

        # View-aware pose projections
        self.query_pose_proj = nn.Linear(d_camera, d_model)
        self.key_pose_proj = nn.Linear(d_camera, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        visibility_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        cam_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Bidirectional per-object cross-attention over all T frames.

        Each object's masked tokens retrieve from that same object's
        visible tokens across all frames. Visible tokens are also
        refined by attending to other visible appearances.

        Args:
            tokens:          (T, N, d_model) — scaffold tokens (visible + masked).
            visibility_mask: (T, N) bool — True where object is visible.
            valid_mask:      (T, N) bool — True for real objects.
            cam_feats:       (T, N, d_camera) or None — camera-relative features.

        Returns:
            completed: (T, N, d_model) — auto-completed tokens.
        """
        T, N, D = tokens.shape
        device = tokens.device

        # --- Reshape: (T, N, D) → (N, T, D) — per-object temporal sequences ---
        x = tokens.permute(1, 0, 2)                  # (N, T, D)
        vis = visibility_mask.permute(1, 0).contiguous()  # (N, T)
        val = valid_mask.permute(1, 0).contiguous()       # (N, T)

        # --- View-aware bias ---
        if cam_feats is not None:
            cam = cam_feats.permute(1, 0, 2)          # (N, T, d_camera)
            q_bias = self.query_pose_proj(cam)        # (N, T, D)
            k_bias = self.key_pose_proj(cam)          # (N, T, D)
        else:
            q_bias = torch.zeros_like(x)
            k_bias = torch.zeros_like(x)

        # --- Build K/V from visible tokens only ---
        # Key padding mask for cross-attention: True = ignore
        # For each object, ignore frames where it's NOT visible (or invalid)
        kv_active = vis & val  # (N, T) — True where visible + valid
        kv_padding_mask = ~kv_active  # (N, T) — True = ignore

        # Failsafe: if an object has NO visible frames, unmask first frame
        all_masked = kv_padding_mask.all(dim=1)  # (N,)
        if all_masked.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked, 0] = False

        # --- Cross-attention layers ---
        query = x + q_bias  # Add view bias to queries
        kv = x + k_bias     # Add view bias to keys

        for i in range(len(self.cross_attn_layers)):
            # Cross-attention: Q = all tokens, K/V = visible tokens
            attn_out, _ = self.cross_attn_layers[i](
                query=query, key=kv, value=kv,
                key_padding_mask=kv_padding_mask,
            )
            query = self.cross_norms1[i](query + attn_out)
            ffn_out = self.cross_ffns[i](query)
            query = self.cross_norms2[i](query + ffn_out)

        # --- Zero out invalid ---
        query = query * val.unsqueeze(-1).float()

        # --- Reshape back: (N, T, D) → (T, N, D) ---
        completed = query.permute(1, 0, 2)  # (T, N, D)

        return completed
