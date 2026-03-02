"""
AMWAE: Associative Masked World Auto-Encoder (Batched)
========================================================

Single-pass pipeline with B=T batching:

  1. GlobalStructuralEncoder(corners)                → (T, N, d_struct)
  2. CameraPoseEncoder(pose, corners)                → (T, N, d_camera)
  3. CameraTemporalEncoder(pose_stack)               → (T, d_camera)
  4. MotionFeatureEncoder(vel, acc)                   → (T, N, d_motion)
  5. ScaffoldTokenizer(struct, vis, cam, motion, ego) → (T, N, d_model)
  6. AssociativeRetriever(tokens, vis, valid, cam)    → (T, N, d_model)
  7. + visibility_embedding                           → (T, N, d_model)
  8. SpatialGNN(tokens, corners, valid)               → (T, N, d_model)
  9. NodePredictor + RelationshipPredictor            → scene graph
 10. TemporalEdgeAttention                           → final distributions

No per-frame loops (except edge token formation — variable K per frame).
Associative retrieval uses bidirectional per-object cross-attention over
ALL T frames. Visibility embedding after retrieval informs downstream
modules about observation provenance (direct vs retrieved).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional

from .scaffold_tokenizer import ScaffoldTokenizer
from .associative_retriever import AssociativeRetriever

from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, RelationshipPredictor,
    SpatialGNN, CameraPoseEncoder, CameraTemporalEncoder,
    MotionFeatureEncoder, TemporalEdgeAttention,
)


class AMWAE(nn.Module):
    """
    Associative Masked World Auto-Encoder (batched).

    Processes all T frames in a single forward pass with B=T batching.
    Includes ego-motion encoding, object motion features, and a learned
    visibility embedding applied after associative retrieval.

    Args:
        config: Config with architecture hyperparameters.
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

        # Module 1: Structural Encoder — (B, N, 8, 3) → (B, N, d_struct)
        self.structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: Camera Pose Encoder — (B, 4, 4) → (B, N, d_camera)
        self.camera_encoder = CameraPoseEncoder(
            d_camera=config.d_camera,
        )

        # Module 3: Camera Temporal Encoder — (T, 4, 4) → (T, d_camera)
        self.camera_temporal_encoder = CameraTemporalEncoder(
            d_camera=config.d_camera,
        )

        # Module 4: Motion Feature Encoder — vel/acc → (B, N, d_motion)
        d_motion = config.d_motion
        self.motion_encoder = MotionFeatureEncoder(
            d_motion=d_motion,
        )

        # Module 5: Scaffold Tokenizer — bind evidence + mask + all features (B=T)
        self.scaffold_tokenizer = ScaffoldTokenizer(
            d_struct=config.d_struct,
            d_visual=config.d_visual,
            d_model=config.d_model,
            d_detector_roi=config.d_detector_roi,
            d_camera=config.d_camera,
            d_motion=d_motion,
        )

        # Module 6: Associative Retriever — bidirectional per-object cross-attention
        self.retriever = AssociativeRetriever(
            d_model=config.d_model,
            n_layers=config.n_cross_attn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
            d_camera=config.d_camera,
        )

        # Module 7: Visibility Embedding — applied after retrieval
        # Informs downstream modules: 0 = retrieved/inferred, 1 = directly observed
        self.visibility_emb = nn.Embedding(2, config.d_model)

        # Module 8: Spatial GNN — inter-object context propagation (B=T)
        self.spatial_gnn = SpatialGNN(
            d_model=config.d_model,
            n_layers=config.n_self_attn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 9: Node Predictor
        self.node_predictor = NodePredictor(
            d_memory=config.d_model,
            num_classes=num_object_classes,
        )

        # Module 10: Relationship Predictor
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

        # Module 11: Temporal Edge Attention
        self.temporal_edge_attn = TemporalEdgeAttention(
            d_rel=config.d_rel,
            n_heads=config.n_rel_heads,
            n_layers=config.n_temporal_edge_layers,
            dropout=config.dropout,
        )

        # Cross-view reconstruction projection
        self.reconstruction_proj = nn.Linear(config.d_model, config.d_visual)

    def forward(
        self,
        visual_features_seq: torch.Tensor,
        corners_seq: torch.Tensor,
        valid_mask_seq: torch.Tensor,
        visibility_mask_seq: torch.Tensor,
        person_idx_seq: torch.Tensor,
        object_idx_seq: torch.Tensor,
        pair_valid: torch.Tensor,
        p_mask_visible: float = 0.0,
        camera_pose_seq: Optional[torch.Tensor] = None,
        union_features_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a full video in a single batched forward pass.

        All inputs are pre-padded (T, N_max, ...) or (T, K_max, ...) tensors.

        Args:
            visual_features_seq: (T, N_max, d_detector_roi)
            corners_seq:         (T, N_max, 8, 3)
            valid_mask_seq:      (T, N_max) bool
            visibility_mask_seq: (T, N_max) bool
            person_idx_seq:      (T, K_max) long
            object_idx_seq:      (T, K_max) long
            pair_valid:          (T, K_max) bool
            p_mask_visible:      Training masking probability.
            camera_pose_seq:     (T, 4, 4) or None
            union_features_seq:  (T, K_max, d_union_roi) or None

        Returns:
            dict with (T, ...) padded tensors.
        """
        T = corners_seq.shape[0]
        device = corners_seq.device

        # Inputs already (T, N_max, ...) — no stacking needed
        visual_all = visual_features_seq
        corners_all = corners_seq
        valid_all = valid_mask_seq
        visibility_all = visibility_mask_seq

        # ==================== Step 1: Structural encoding (B=T) ====================
        struct_all, _ = self.structural_encoder(corners_all, valid_all)  # (T, N, d_struct)

        # ==================== Step 2-3: Camera encoding (B=T) ====================
        cam_all = None
        ego_tokens_all = None
        if camera_pose_seq is not None:
            camera_pose_all = camera_pose_seq
            _, cam_all = self.camera_encoder(camera_pose_all, corners_all, valid_all)
            ego_tokens_all = self.camera_temporal_encoder(camera_pose_all)  # (T, d_camera)

        # ==================== Step 4: Motion features (B=T) ====================
        centers_all = corners_all.mean(dim=2)  # (T, N, 3)
        velocity_all = torch.zeros_like(centers_all)
        accel_all = torch.zeros_like(centers_all)
        velocity_all[1:] = centers_all[1:] - centers_all[:-1]
        accel_all[2:] = velocity_all[2:] - velocity_all[1:-1]

        camera_R_all = None
        if camera_pose_seq is not None:
            camera_R_all = camera_pose_seq[:, :3, :3]

        motion_all = self.motion_encoder(
            velocity=velocity_all, acceleration=accel_all,
            camera_R=camera_R_all, valid_mask=valid_all,
        )  # (T, N, d_motion)

        # ==================== Step 5: Scaffold tokenization (B=T) ====================
        hybrid_tokens, is_masked_all, original_visual_all = self.scaffold_tokenizer(
            geometry_tokens=struct_all,
            visual_features=visual_all,
            visibility_mask=visibility_all,
            valid_mask=valid_all,
            p_mask_visible=p_mask_visible if self.training else 0.0,
            cam_feats=cam_all,
            motion_feats=motion_all,
            ego_tokens=ego_tokens_all,
        )  # (T, N, d_model), (T, N), (T, N, d_visual)

        # ==================== Step 6: Associative retrieval (bidirectional) ====================
        completed_tokens = self.retriever(
            tokens=hybrid_tokens,
            visibility_mask=visibility_all,
            valid_mask=valid_all,
            cam_feats=cam_all,
        )  # (T, N, d_model)

        # ==================== Step 7: Visibility embedding (after retrieval) ====================
        # 0 = masked/retrieved, 1 = directly observed
        vis_ids = visibility_all.long()  # (T, N)
        completed_tokens = completed_tokens + self.visibility_emb(vis_ids)

        # ==================== Step 8: Spatial GNN (B=T) ====================
        enriched_all = self.spatial_gnn(
            tokens=completed_tokens,
            corners=corners_all,
            valid_mask=valid_all,
        )  # (T, N, d_model)

        # ==================== Step 9: Node prediction (B=T) ====================
        node_logits_all = self.node_predictor(enriched_all)  # (T, N, C)

        # ==================== Step 10: Cross-view reconstruction ====================
        recon_pred_all = self.reconstruction_proj(completed_tokens)  # (T, N, d_visual)

        # ==================== Step 11: Batched edge prediction ====================
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq,
            pair_valid, union_features_seq,
        )  # (T, K_max, d_rel), (T, K_max)

        # ==================== Step 12: Temporal edge attention ====================
        enriched_rel = self.temporal_edge_attn(rel_tokens, pair_valid_out, person_idx_seq, object_idx_seq)

        # ==================== Step 13: Predict distributions ====================
        edge_out = self.rel_predictor.batched_predict(enriched_rel, pair_valid_out)

        # ==================== Build output ====================
        outputs = {
            "node_logits": node_logits_all,                              # (T, N, C)
            "attention_logits": edge_out["attention_logits"],              # (T, K_max, 3)
            "attention_distribution": edge_out["attention_distribution"],  # (T, K_max, 3)
            "spatial_distribution": edge_out["spatial_distribution"],      # (T, K_max, 6)
            "contacting_distribution": edge_out["contacting_distribution"],# (T, K_max, 17)
            "spatial_logits": edge_out["spatial_logits"],                  # (T, K_max, 6)
            "contacting_logits": edge_out["contacting_logits"],            # (T, K_max, 17)
            "is_masked": is_masked_all,                                    # (T, N)
            "original_visual": original_visual_all,                        # (T, N, d_visual)
            "reconstruction_predictions": recon_pred_all,                  # (T, N, d_visual)
            "reconstruction_targets": original_visual_all,                 # (T, N, d_visual)
        }

        return outputs



