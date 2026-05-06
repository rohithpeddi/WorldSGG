"""
WorldWise: MWAE-based World Scene Graph Generation (Batched)
==============================================================

The full WorldWise method built on the MWAE (Masked World Auto-Encoder)
architecture. Uses config flags to toggle individual components for
ablation experiments while keeping the MWAE core constant.

MWAE core (always on):
  - GlobalStructuralEncoder
  - ScaffoldTokenizer (occlusion-as-masking)
  - AssociativeRetriever (bidirectional cross-attention)
  - VisibilityEmbedding
  - InterObjectTransformer
  - NodePredictor
  - RelationshipPredictor

Togglable components (for ablations):
  - ObjectSpatialEncoder     (use_object_spatial_encoder)
  - CameraTemporalEncoder    (use_camera_temporal)
  - ObjectMotionEncoder      (use_object_motion_encoder)
  - TemporalEdgeAttention    (use_temporal_edge_attn)

Full WorldWise = MWAE + all togglable components enabled.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from lib.supervised.worldsgg.amwae.scaffold_tokenizer import ScaffoldTokenizer
from lib.supervised.worldsgg.amwae.associative_retriever import AssociativeRetriever

from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, RelationshipPredictor,
    SpatialGNN as InterObjectTransformer,
    CameraPoseEncoder as ObjectSpatialEncoder,
    CameraTemporalEncoder,
    MotionFeatureEncoder as ObjectMotionEncoder,
    TemporalEdgeAttention,
)


class WorldWise(nn.Module):
    """
    WorldWise — MWAE-based World Scene Graph Generation.

    Processes all T frames in a single forward pass with B=T batching.
    Individual components can be toggled via config flags for ablation.

    Args:
        config: Config namespace with architecture hyperparameters and
                ablation flags (use_object_spatial_encoder, use_camera_temporal,
                use_object_motion_encoder, use_temporal_edge_attn).
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

        # Ablation flags (default: all enabled for full WorldWise)
        self.use_object_spatial_encoder = getattr(config, 'use_object_spatial_encoder', True)
        self.use_camera_temporal = getattr(config, 'use_camera_temporal', True)
        self.use_object_motion_encoder = getattr(config, 'use_object_motion_encoder', True)
        self.use_temporal_edge_attn = getattr(config, 'use_temporal_edge_attn', True)

        # ====================== MWAE Core (always on) ======================

        # Module 1: Global Structural Encoder — 3D OBB geometry in world frame
        self.global_structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: Object Spatial Encoder — per-object spatial position in camera frame (togglable)
        if self.use_object_spatial_encoder:
            self.object_spatial_encoder = ObjectSpatialEncoder(
                d_camera=config.d_camera,
            )

        # Module 3: Camera Temporal Encoder — ego-motion in world frame (togglable)
        if self.use_camera_temporal:
            self.camera_temporal_encoder = CameraTemporalEncoder(
                d_camera=config.d_camera,
            )

        # Module 4: Object Motion Encoder — per-object motion in world frame (togglable)
        d_motion = getattr(config, 'd_motion', 64)
        if self.use_object_motion_encoder:
            self.object_motion_encoder = ObjectMotionEncoder(
                d_motion=d_motion,
            )

        # Module 5: Scaffold Tokenizer — handles masking + feature binding
        self.scaffold_tokenizer = ScaffoldTokenizer(
            d_struct=config.d_struct,
            d_visual=config.d_visual,
            d_model=config.d_model,
            d_detector_roi=config.d_detector_roi,
            d_camera=config.d_camera if self.use_object_spatial_encoder else 0,
            d_motion=d_motion if self.use_object_motion_encoder else 0,
        )

        # Module 6: Associative Retriever — bidirectional cross-attention
        self.retriever = AssociativeRetriever(
            d_model=config.d_model,
            n_layers=config.n_cross_attn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
            d_camera=config.d_camera if self.use_object_spatial_encoder else 0,
        )

        # Module 7: Visibility Embedding
        self.visibility_emb = nn.Embedding(2, config.d_model)

        # Module 8: Inter-Object Transformer (vanilla transformer encoder across objects)
        self.inter_object_encoder = InterObjectTransformer(
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

        # Module 11: Temporal Edge Attention (togglable)
        if self.use_temporal_edge_attn:
            self.temporal_edge_attn = TemporalEdgeAttention(
                d_rel=config.d_rel,
                n_heads=config.n_rel_heads,
                n_layers=config.n_temporal_edge_layers,
                dropout=config.dropout,
            )

        # Cross-view reconstruction projection (always on — MWAE core)
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

        # ==================== Step 1: Global structural encoding ====================
        struct_all, _ = self.global_structural_encoder(corners_seq, valid_mask_seq)

        # ==================== Step 2: Object spatial / camera temporal encoding (togglable) ====================
        cam_all = None
        ego_tokens_all = None
        if camera_pose_seq is not None:
            if self.use_object_spatial_encoder:
                _, cam_all = self.object_spatial_encoder(
                    camera_pose_seq, corners_seq, valid_mask_seq,
                )
            if self.use_camera_temporal:
                ego_tokens_all = self.camera_temporal_encoder(camera_pose_seq)

        # ==================== Step 3: Object motion features (togglable) ====================
        motion_all = None
        if self.use_object_motion_encoder:
            centers_all = corners_seq.mean(dim=2)
            velocity_all = torch.zeros_like(centers_all)
            accel_all = torch.zeros_like(centers_all)

            valid_vel = valid_mask_seq[1:] & valid_mask_seq[:-1]
            velocity_all[1:] = torch.where(
                valid_vel.unsqueeze(-1),
                centers_all[1:] - centers_all[:-1],
                torch.zeros_like(centers_all[1:]),
            )

            valid_acc = valid_mask_seq[2:] & valid_mask_seq[1:-1] & valid_mask_seq[:-2]
            accel_all[2:] = torch.where(
                valid_acc.unsqueeze(-1),
                velocity_all[2:] - velocity_all[1:-1],
                torch.zeros_like(velocity_all[2:]),
            )

            camera_R_all = camera_pose_seq[:, :3, :3] if camera_pose_seq is not None else None
            motion_all = self.object_motion_encoder(
                velocity=velocity_all, acceleration=accel_all,
                camera_R=camera_R_all, valid_mask=valid_mask_seq,
            )

        # ==================== Step 4: Scaffold tokenization ====================
        hybrid_tokens, is_masked_all, original_visual_all, artificially_masked_all = self.scaffold_tokenizer(
            geometry_tokens=struct_all,
            visual_features=visual_features_seq,
            visibility_mask=visibility_mask_seq,
            valid_mask=valid_mask_seq,
            p_mask_visible=p_mask_visible if self.training else 0.0,
            cam_feats=cam_all,
            motion_feats=motion_all,
            ego_tokens=ego_tokens_all,
        )

        # ==================== Step 5: Associative retrieval ====================
        completed_tokens = self.retriever(
            tokens=hybrid_tokens,
            visibility_mask=visibility_mask_seq,
            valid_mask=valid_mask_seq,
            cam_feats=cam_all,
        )

        # ==================== Step 6: Visibility embedding ====================
        effective_visible = visibility_mask_seq & (~is_masked_all)
        vis_ids = effective_visible.long()
        completed_tokens = completed_tokens + self.visibility_emb(vis_ids)
        completed_tokens = completed_tokens * valid_mask_seq.unsqueeze(-1).float()

        # ==================== Step 7: Inter-object transformer ====================
        enriched_all = self.inter_object_encoder(
            tokens=completed_tokens,
            corners=corners_seq,
            valid_mask=valid_mask_seq,
        )

        # ==================== Step 8: Node prediction ====================
        node_logits_all = self.node_predictor(enriched_all)

        # ==================== Step 9: Reconstruction ====================
        recon_pred_all = self.reconstruction_proj(completed_tokens)

        # ==================== Step 10: Edge prediction ====================
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq,
            pair_valid, union_features_seq,
        )

        # ==================== Step 11: Temporal edge attention (togglable) ====================
        if self.use_temporal_edge_attn:
            enriched_rel = self.temporal_edge_attn(
                rel_tokens, pair_valid_out, person_idx_seq, object_idx_seq,
            )
        else:
            enriched_rel = rel_tokens

        # ==================== Step 12: Predict distributions ====================
        edge_out = self.rel_predictor.batched_predict(enriched_rel, pair_valid_out)

        return {
            "node_logits": node_logits_all,
            "attention_logits": edge_out["attention_logits"],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
            "spatial_logits": edge_out["spatial_logits"],
            "contacting_logits": edge_out["contacting_logits"],
            "is_masked": is_masked_all,
            "artificially_masked": artificially_masked_all,
            "original_visual": original_visual_all,
            "reconstruction_predictions": recon_pred_all,
            "reconstruction_targets": original_visual_all,
        }
