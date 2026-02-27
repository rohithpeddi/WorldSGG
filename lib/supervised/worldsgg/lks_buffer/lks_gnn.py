"""
LKS GNN: Last-Known-State Graph Neural Network
=================================================

Main model for Baseline 1. Pairs a hard-coded, non-differentiable
LKS memory buffer with a stateless feed-forward GNN predictor.

Sequential processing:
  1. LKSMemoryBuffer.update(DINO, visibility)   → detached
  2. GlobalStructuralEncoder(corners)            → geometry tokens
  3. LKSTokenizer(geometry, buffer)              → hybrid tokens
  4. SpatialGNN(hybrid_tokens, corners)          → enriched tokens
  5. NodePredictor + EdgePredictor               → scene graph

Gradients only flow through steps 2-5 at the current frame.
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .lks_memory import LKSMemoryBuffer
from .lks_tokenizer import LKSTokenizer

# Shared components from worldsgg_base
from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, EdgePredictor, SpatialGNN,
)


class LKSGNN(nn.Module):
    """
    Last-Known-State GNN — passive memory baseline.

    Processes frames sequentially. The memory buffer is updated via
    hard copy-paste (non-differentiable), NOT learned routing. The
    GNN predictor at each frame is fully stateless.

    Args:
        config: LKSConfig.
        num_object_classes: Object categories.
        attention_class_num: Attention relationship classes.
        spatial_class_num: Spatial relationship classes.
        contact_class_num: Contacting relationship classes.
    """

    def __init__(
        self,
        config,
        num_object_classes: int = 37,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
    ):
        super().__init__()
        self.config = config

        # Module 1: Geometry encoder (reused from GL-STGN)
        self.structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Visual projector: raw DINO ROI → d_visual (for buffer storage)
        self.visual_projector = nn.Sequential(
            nn.Linear(config.d_detector_roi, config.d_visual),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.d_visual),
        )

        # Module 2: LKS Tokenizer (geometry + buffer fusion)
        self.tokenizer = LKSTokenizer(
            d_struct=config.d_struct,
            d_visual=config.d_visual,
            d_model=config.d_model,
        )

        # Module 3: Spatial GNN (reused from Amnesic GNN)
        self.spatial_gnn = SpatialGNN(
            d_model=config.d_model,
            n_layers=config.n_gnn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 4: Prediction heads (reused from GL-STGN)
        self.node_predictor = NodePredictor(
            d_memory=config.d_model,
            num_classes=num_object_classes,
        )
        self.edge_predictor = EdgePredictor(
            d_memory=config.d_model,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
        )

        # Non-differentiable memory buffer (not an nn.Module)
        self.memory_buffer = None  # Initialized in reset_memory()

    def reset_memory(self, device: torch.device = None):
        """Reset the LKS buffer (call at start of each video)."""
        if device is None:
            device = next(self.parameters()).device
        self.memory_buffer = LKSMemoryBuffer(
            max_objects=self.config.max_objects,
            d_visual=self.config.d_visual,
            device=str(device),
        )

    def forward_frame(
        self,
        visual_features: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
        visibility_mask: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a SINGLE frame with passive memory.

        Steps:
          1. Project visual features and update buffer (detached)
          2. Encode geometry
          3. Fuse geometry + buffered features
          4. GNN context propagation
          5. Predict scene graph

        Args:
            visual_features: (N, d_detector_roi) — raw DINO features.
            corners: (N, 8, 3) — 3D bbox corners.
            valid_mask: (N,) bool.
            visibility_mask: (N,) bool.
            person_idx: (K,) long.
            object_idx: (K,) long.

        Returns:
            dict with node_logits, attention/spatial/contacting distributions.
        """
        N = corners.shape[0]

        # --- Step 1: Project visual features and update buffer ---
        with torch.no_grad():
            projected_visual = self.visual_projector(visual_features)  # (N, d_visual)

        # Ensure buffer exists
        if self.memory_buffer is None:
            self.reset_memory(corners.device)

        # Handle size changes
        if N > self.memory_buffer.max_objects:
            self.memory_buffer.reset(N)

        # Zero-order hold update (all detached internally)
        self.memory_buffer.update(projected_visual, visibility_mask, valid_mask)

        # Get buffered features (detached)
        buffer_features = self.memory_buffer.get_features(N)  # (N, d_visual)

        # --- Step 2: Encode wireframe geometry ---
        struct_tokens, _ = self.structural_encoder(
            corners.unsqueeze(0),
            valid_mask.unsqueeze(0),
        )
        struct_tokens = struct_tokens.squeeze(0)  # (N, d_struct)

        # --- Step 3: Fuse geometry + buffered visual ---
        tokens = self.tokenizer(
            geometry_tokens=struct_tokens,
            buffer_features=buffer_features,
            valid_mask=valid_mask,
        )  # (N, d_model)

        # --- Step 4: Spatial GNN ---
        enriched = self.spatial_gnn(
            tokens=tokens,
            corners=corners,
            valid_mask=valid_mask,
        )  # (N, d_model)

        # --- Step 5: Predict scene graph ---
        node_logits = self.node_predictor(enriched)
        edge_out = self.edge_predictor(
            enriched_states=enriched,
            person_idx=person_idx,
            object_idx=object_idx,
            corners=corners,
        )

        return {
            "node_logits": node_logits,
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
        }
