"""
Associative Retriever
======================

Cross-attention module that retrieves visual features from the episodic
memory bank for masked tokens. Mathematically equivalent to dense
associative memory retrieval (continuous Hopfield network).

Masked tokens use their wireframe geometry as a query key to "unlock"
the visual memory of that specific 3D location from past frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with residual + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (1, N, d_model) — current tokens as queries.
            memory: (1, M, d_model) — memory bank tokens as K/V.
            memory_key_padding_mask: (1, M) bool — True = ignore.

        Returns:
            output: (1, N, d_model)
            attn_weights: (N, M) — attention weights (for contrastive loss).
        """
        # Cross-attention
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            average_attn_weights=True,  # Average across heads for contrastive loss
        )
        query = self.norm1(query + attn_out)

        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        # Remove batch dim from attn_weights
        attn_weights = attn_weights.squeeze(0)  # (N, M)

        return query, attn_weights


class AssociativeRetriever(nn.Module):
    """
    Multi-layer cross-attention that auto-completes masked tokens by
    retrieving features from the episodic memory bank.

    Q = all current tokens (masked tokens need retrieval, visible tokens
        are refined with temporal context)
    K, V = episodic memory bank tokens

    Returns completed tokens AND attention weights for contrastive loss.

    Args:
        d_model: Token dimension.
        n_layers: Number of cross-attention layers.
        n_heads: Attention heads.
        d_feedforward: FFN hidden dim.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        d_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, d_feedforward, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_valid: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (N, d_model) — current hybrid tokens.
            memory_tokens: (M, d_model) — episodic memory bank.
            memory_valid: (M,) bool — valid mask for memory entries.

        Returns:
            completed_tokens: (N, d_model) — auto-completed tokens.
            attn_weights: (N, M) — last layer's attention weights
                          (for contrastive loss supervision).
        """
        N = tokens.shape[0]
        M = memory_tokens.shape[0]
        device = tokens.device

        # Handle empty memory: return tokens unchanged with zero attention
        if M == 0:
            return tokens, torch.zeros(N, 0, device=device)

        # Add batch dimension: (1, N, d_model), (1, M, d_model)
        query = tokens.unsqueeze(0)
        memory = memory_tokens.unsqueeze(0)

        # Memory padding mask: True = ignore, so invert valid
        if memory_valid is not None:
            memory_pad_mask = ~memory_valid.unsqueeze(0)  # (1, M)
        else:
            memory_pad_mask = None

        # Run through cross-attention layers
        last_attn = None
        for layer in self.layers:
            query, attn_weights = layer(query, memory, memory_pad_mask)
            last_attn = attn_weights

        completed = query.squeeze(0)  # (N, d_model)

        return completed, last_attn
