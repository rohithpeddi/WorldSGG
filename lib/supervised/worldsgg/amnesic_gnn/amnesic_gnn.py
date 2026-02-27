"""
Amnesic Geometric GNN
======================

Stateless feed-forward baseline with zero temporal memory.
Processes each frame independently — pure geometric reasoning.

Pipeline:
  1. GlobalStructuralEncoder(corners) → geometry tokens
  2. AmnesicTokenizer(geometry, DINO, visibility) → hybrid tokens
  3. SpatialGNN(hybrid_tokens, corners) → enriched tokens
  4. NodePredictor + EdgePredictor → scene graph
"""

import torch
import torch.nn as nn
from typing import Dict

from .amnesic_tokenizer import AmnesicTokenizer
from .spatial_gnn import SpatialGNN

# Reuse from GL-STGN
from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, EdgePredictor,
)


class AmnesicGNN(nn.Module):
    """
    Amnesic Geometric GNN — zero-memory baseline.

    Each frame is processed as a completely independent puzzle.
    No temporal state, no memory bank, no recurrence.

    Args:
        config: AmnesicGNNConfig.
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

        # Module 2: Amnesic tokenizer (visible → DINO, unseen → [UNSEEN])
        self.tokenizer = AmnesicTokenizer(
            d_struct=config.d_struct,
            d_visual=config.d_visual,
            d_model=config.d_model,
            d_detector_roi=config.d_detector_roi,
        )

        # Module 3: Spatial GNN (context propagation)
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

    def forward(
        self,
        visual_features: torch.Tensor,
        corners: torch.Tensor,
        valid_mask: torch.Tensor,
        visibility_mask: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a SINGLE frame (stateless).

        Args:
            visual_features: (N, d_detector_roi) — DINO ROI features.
            corners: (N, 8, 3) — world-frame 3D bbox corners.
            valid_mask: (N,) bool — True for real objects.
            visibility_mask: (N,) bool — True if in camera FOV.
            person_idx: (K,) long — person indices in pairs.
            object_idx: (K,) long — object indices in pairs.

        Returns:
            dict:
                node_logits: (N, num_classes)
                attention_distribution: (K, 3)
                spatial_distribution: (K, 6)
                contacting_distribution: (K, 17)
        """
        # Step 1: Encode wireframe geometry
        struct_tokens, _ = self.structural_encoder(
            corners.unsqueeze(0),  # (1, N, 8, 3)
            valid_mask.unsqueeze(0),  # (1, N)
        )
        struct_tokens = struct_tokens.squeeze(0)  # (N, d_struct)

        # Step 2: Binary feature assignment
        tokens = self.tokenizer(
            geometry_tokens=struct_tokens,
            visual_features=visual_features,
            visibility_mask=visibility_mask,
            valid_mask=valid_mask,
        )  # (N, d_model)

        # Step 3: Spatial context propagation via GNN
        enriched = self.spatial_gnn(
            tokens=tokens,
            corners=corners,
            valid_mask=valid_mask,
        )  # (N, d_model)

        # Step 4: Predict scene graph
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
