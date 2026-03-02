"""
LKS GNN: Last-Known-State Graph Neural Network (Batched)
==========================================================

Pairs a vectorized, non-differentiable LKS memory buffer with a
stateless feed-forward GNN predictor. All T frames are processed
in a single forward pass with B=T batching — no per-frame loops
except for relationship token formation (variable K per frame).

Single-pass pipeline:
  1. vectorized_lks_buffer(visual_all, vis, valid)   → (T, N, d_roi), (T, N)
  2. GlobalStructuralEncoder(corners_all)              → (T, N, d_struct)
  3. CameraPoseEncoder(pose_all, corners_all)          → (T, N, d_camera)
  4. LKSTokenizer(struct, buffer, cam, staleness)      → (T, N, d_model)
  5. SpatialGNN(tokens, corners, valid)                → (T, N, d_model)
  6. NodePredictor(enriched)                           → (T, N, C)
  7. batched_form_and_attend(enriched, logits, ...)    → (T, K_max, d_rel)
  8. TemporalEdgeAttention(rel, valid, pidx, oidx)     → (T, K_max, d_rel)
  9. batched_predict(enriched_rel, valid)              → distributions

All differentiable except the LKS buffer (step 1).
No per-frame loops — fully batched.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .lks_memory import vectorized_lks_buffer
from .lks_tokenizer import LKSTokenizer

from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, RelationshipPredictor, SpatialGNN,
    CameraPoseEncoder, TemporalEdgeAttention,
)


class LKSGNN(nn.Module):
    """
    Last-Known-State GNN — passive memory baseline (batched).

    Processes all T frames in a single forward pass. The memory buffer
    is computed via vectorized cummax (non-differentiable). The GNN
    predictor is fully stateless and batched over T.

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

        # Module 1: Geometry encoder
        self.structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: Camera pose encoder
        self.camera_encoder = CameraPoseEncoder(
            d_camera=config.d_camera,
        )

        # Module 3: LKS Tokenizer (geometry + raw buffer + camera fusion)
        # Visual projection (1024→d_model) happens inside tokenizer's fusion MLP
        self.tokenizer = LKSTokenizer(
            d_struct=config.d_struct,
            d_detector_roi=config.d_detector_roi,
            d_model=config.d_model,
            d_camera=config.d_camera,
        )

        # Module 4: Spatial GNN
        self.spatial_gnn = SpatialGNN(
            d_model=config.d_model,
            n_layers=config.n_gnn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 5: Node predictor
        self.node_predictor = NodePredictor(
            d_model=config.d_model,
            num_classes=num_object_classes,
        )

        # Module 6: Relationship predictor
        clip_path = getattr(config, 'clip_embeddings_path', '')
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

        # Module 7: Temporal edge attention
        self.temporal_edge_attn = TemporalEdgeAttention(
            d_rel=config.d_rel,
            n_heads=config.n_rel_heads,
            n_layers=config.n_temporal_edge_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        visual_features_seq: List[torch.Tensor],
        corners_seq: List[torch.Tensor],
        valid_mask_seq: List[torch.Tensor],
        visibility_mask_seq: List[torch.Tensor],
        person_idx_seq: List[torch.Tensor],
        object_idx_seq: List[torch.Tensor],
        camera_pose_seq: Optional[List[torch.Tensor]] = None,
        union_features_seq: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, List]:
        """
        Process a full video in a single batched forward pass.

        Args:
            visual_features_seq: T-list of (N, d_roi) tensors.
            corners_seq:         T-list of (N, 8, 3) tensors.
            valid_mask_seq:      T-list of (N,) bool tensors.
            visibility_mask_seq: T-list of (N,) bool tensors.
            person_idx_seq:      T-list of (K_t,) long tensors.
            object_idx_seq:      T-list of (K_t,) long tensors.
            camera_pose_seq:     T-list of (4,4) tensors or None.
            union_features_seq:  T-list of (K_t, d_union_roi) tensors or None.

        Returns:
            dict with per-frame lists: node_logits, att/spa/con distributions.
        """
        T = len(corners_seq)

        # ==================== Stack inputs (T, N, ...) ====================
        visual_all = torch.stack(visual_features_seq)       # (T, N, d_roi)
        corners_all = torch.stack(corners_seq)              # (T, N, 8, 3)
        valid_all = torch.stack(valid_mask_seq)              # (T, N)
        visibility_all = torch.stack(visibility_mask_seq)    # (T, N)

        # ==================== Step 1: Vectorized LKS buffer (raw features) ====================
        buffer_all, staleness_all = vectorized_lks_buffer(
            projected_visual=visual_all,
            visibility_mask=visibility_all,
            valid_mask=valid_all,
        )  # (T, N, d_detector_roi), (T, N)

        # ==================== Step 3: Batch structural encoding ====================
        struct_all, _ = self.structural_encoder(
            corners_all, valid_all,
        )  # (T, N, d_struct)

        # ==================== Step 4: Batch camera encoding ====================
        cam_all = None
        if camera_pose_seq is not None:
            camera_pose_all = torch.stack(camera_pose_seq)  # (T, 4, 4)
            _, cam_all = self.camera_encoder(
                camera_pose=camera_pose_all,
                corners=corners_all,
                valid_mask=valid_all,
            )  # (T, N, d_camera)

        # ==================== Step 5: Batch tokenizer ====================
        tokens_all = self.tokenizer(
            geometry_tokens=struct_all,
            buffer_features=buffer_all,
            valid_mask=valid_all,
            cam_feats=cam_all,
            staleness=staleness_all,
        )  # (T, N, d_model)

        # ==================== Step 6: Batch spatial GNN ====================
        enriched_all = self.spatial_gnn(
            tokens=tokens_all,
            corners=corners_all,
            valid_mask=valid_all,
        )  # (T, N, d_model)

        # ==================== Step 7: Node prediction (batched) ====================
        node_logits_all = self.node_predictor(enriched_all)  # (T, N, num_classes)

        # ==================== Step 7: Batched edge prediction ====================
        rel_tokens, pair_valid, padded_pidx, padded_oidx = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq, union_features_seq,
        )  # (T, K_max, d_rel), (T, K_max), (T, K_max), (T, K_max)

        # ==================== Step 8: Temporal edge attention ====================
        enriched_rel = self.temporal_edge_attn(rel_tokens, pair_valid, padded_pidx, padded_oidx)

        # ==================== Step 9: Predict distributions ====================
        edge_out = self.rel_predictor.batched_predict(enriched_rel, pair_valid)

        outputs = {
            "node_logits": [node_logits_all[t] for t in range(T)],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
        }

        return outputs
