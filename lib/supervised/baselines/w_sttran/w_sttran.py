"""
W-STTran: World-adapted STTran (Batched)
==========================================

Simplest world-centric baseline adaptation of STTran. Uses the LKS Buffer
for persistent object memory but with NO TemporalEdgeAttention and NO
camera/motion encoders — analogous to how STTran uses simple relationship
prediction without explicit object tracking.

Single-pass pipeline:
  1. vectorized_lks_buffer(visual, vis, valid)      → (T, N, d_roi), (T, N)
  2. GlobalStructuralEncoder(corners)                → (T, N, d_struct)
  3. LKSTokenizer(struct, buffer, staleness)         → (T, N, d_model)
  4. InterObjectTransformer(tokens, corners, valid)  → (T, N, d_model)
  5. NodePredictor(enriched)                         → (T, N, C)
  6. batched_form_and_attend(enriched, logits,...)   → (T, K_max, d_rel)
  7. batched_predict(rel_tokens, valid)              → distributions

Key differences from LKSGNN (PWG):
  - No TemporalEdgeAttention (simpler, like STTran)
  - No ObjectSpatialEncoder (no camera-relative features)
  - No ObjectMotionEncoder (no 3D motion features)

All differentiable except the LKS buffer (step 1).
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from lib.supervised.worldsgg.lks_buffer.lks_memory import vectorized_lks_buffer
from lib.supervised.worldsgg.lks_buffer.lks_tokenizer import LKSTokenizer

from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, RelationshipPredictor,
    SpatialGNN as InterObjectTransformer,
)


class WSTTran(nn.Module):
    """
    W-STTran — World-adapted STTran (batched, no temporal edge attention).

    Simplest baseline adaptation: uses LKS passive memory + InterObjectTransformer
    + RelationshipPredictor WITHOUT TemporalEdgeAttention. No camera or
    motion encoders. Relationship tokens are predicted directly after
    self-attention (Phase 3 of RelationshipPredictor).

    Args:
        config: Method config namespace.
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

        # Module 1: Global Structural Encoder — 3D OBB geometry in world frame
        self.global_structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: LKS Tokenizer (geometry + raw buffer fusion, NO camera)
        # We pass d_camera=0 since W-STTran doesn't use camera features.
        # The tokenizer will use zero camera features internally.
        self.tokenizer = LKSTokenizer(
            d_struct=config.d_struct,
            d_detector_roi=config.d_detector_roi,
            d_model=config.d_model,
            d_camera=getattr(config, 'd_camera', 128),
        )

        # Module 3: Inter-Object Transformer (vanilla transformer encoder across objects)
        self.inter_object_encoder = InterObjectTransformer(
            d_model=config.d_model,
            n_layers=config.n_gnn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 4: Node predictor
        self.node_predictor = NodePredictor(
            d_memory=config.d_model,
            num_classes=num_object_classes,
        )

        # Module 5: Relationship predictor
        clip_path = getattr(config, 'clip_embeddings_path', '')
        clip_path = Path(config.data_path) / clip_path if clip_path else None
        self.rel_predictor = RelationshipPredictor(
            d_model=config.d_model,
            d_text=config.d_text,
            d_rel=config.d_rel,
            d_union_roi=config.d_union_roi,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
            clip_embeddings_path=clip_path,
            n_rel_layers=config.n_rel_layers,
            n_rel_heads=config.n_rel_heads,
            dropout=config.dropout,
        )

        # NO TemporalEdgeAttention — this is the key difference from PWG

    def forward(
        self,
        visual_features_seq: torch.Tensor,
        corners_seq: torch.Tensor,
        valid_mask_seq: torch.Tensor,
        visibility_mask_seq: torch.Tensor,
        person_idx_seq: torch.Tensor,
        object_idx_seq: torch.Tensor,
        pair_valid: torch.Tensor,
        camera_pose_seq: Optional[torch.Tensor] = None,
        union_features_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a full video in a single batched forward pass.

        Args:
            visual_features_seq: (T, N_max, d_roi)
            corners_seq:         (T, N_max, 8, 3)
            valid_mask_seq:      (T, N_max) bool
            visibility_mask_seq: (T, N_max) bool
            person_idx_seq:      (T, K_max) long
            object_idx_seq:      (T, K_max) long
            pair_valid:          (T, K_max) bool
            camera_pose_seq:     (T, 4, 4) or None — accepted but unused
            union_features_seq:  (T, K_max, d_union_roi) or None

        Returns:
            dict with (T, ...) padded tensors.
        """
        # ==================== Step 1: LKS buffer (raw features) ====================
        buffer_all, staleness_all = vectorized_lks_buffer(
            raw_visual=visual_features_seq,
            visibility_mask=visibility_mask_seq,
            valid_mask=valid_mask_seq,
        )  # (T, N, d_detector_roi), (T, N)

        # ==================== Step 2: Global structural encoding ====================
        struct_all, _ = self.global_structural_encoder(
            corners_seq, valid_mask_seq,
        )  # (T, N, d_struct)

        # ==================== Step 3: Tokenizer (no camera features) ====================
        tokens_all = self.tokenizer(
            geometry_tokens=struct_all,
            buffer_features=buffer_all,
            valid_mask=valid_mask_seq,
            cam_feats=None,  # No camera encoder in W-STTran
            staleness=staleness_all,
        )  # (T, N, d_model)

        # ==================== Step 4: Inter-object transformer ====================
        enriched_all = self.inter_object_encoder(
            tokens=tokens_all,
            corners=corners_seq,
            valid_mask=valid_mask_seq,
        )  # (T, N, d_model)

        # ==================== Step 5: Node prediction ====================
        node_logits_all = self.node_predictor(enriched_all)  # (T, N, num_classes)

        # ==================== Step 6: Edge prediction (no temporal edge attn) ====================
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq,
            pair_valid, union_features_seq,
        )  # (T, K_max, d_rel), (T, K_max)

        # Skip TemporalEdgeAttention — predict directly from self-attended tokens
        edge_out = self.rel_predictor.batched_predict(rel_tokens, pair_valid_out)

        return {
            "node_logits": node_logits_all,
            "attention_logits": edge_out["attention_logits"],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
            "spatial_logits": edge_out["spatial_logits"],
            "contacting_logits": edge_out["contacting_logits"],
        }
