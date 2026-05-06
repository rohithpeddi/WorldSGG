"""
W-DSGDetr++: Enhanced World-adapted DSGDetr (Batched)
=======================================================

Full enhancement of W-DSGDetr with all world-aware components:
  - ObjectSpatialEncoder (per-object spatial position in camera frame)
  - ObjectMotionEncoder (3D velocity/acceleration in world frame)
  - CameraTemporalEncoder (ego-motion sequence attention)

Pipeline adds object spatial, object motion, and ego-motion on top of W-DSGDetr's
TemporalObjectEncoder + TemporalEdgeAttention pipeline.

Key differences from W-DSGDetr:
  + ObjectSpatialEncoder
  + ObjectMotionEncoder
  + CameraTemporalEncoder (full ego-motion temporal context)
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from .object_encoder import TemporalObjectEncoder
from lib.supervised.worldsgg.lks_buffer.lks_memory import vectorized_lks_buffer
from lib.supervised.worldsgg.lks_buffer.lks_tokenizer import LKSTokenizer

from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, RelationshipPredictor,
    SpatialGNN as InterObjectTransformer,
    CameraPoseEncoder as ObjectSpatialEncoder,
    CameraTemporalEncoder,
    MotionFeatureEncoder as ObjectMotionEncoder,
    TemporalEdgeAttention,
)


class WDSGDetrPP(nn.Module):
    """
    W-DSGDetr++ — Enhanced World-adapted DSGDetr (batched).

    Full enhancement: temporal object encoder + object spatial + object motion
    + ego-motion + temporal edge attention. The most feature-rich baseline.

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

        # Module 2: Object Spatial Encoder — per-object spatial position in camera frame
        self.object_spatial_encoder = ObjectSpatialEncoder(
            d_camera=config.d_camera,
        )

        # Module 3: Camera Temporal Encoder — ego-motion in world frame
        self.camera_temporal_encoder = CameraTemporalEncoder(
            d_camera=config.d_camera,
        )

        # Module 4: Object Motion Encoder — per-object motion in world frame
        d_motion = getattr(config, 'd_motion', 64)
        self.object_motion_encoder = ObjectMotionEncoder(
            d_motion=d_motion,
        )

        # Module 5: LKS Tokenizer (with camera)
        self.tokenizer = LKSTokenizer(
            d_struct=config.d_struct,
            d_detector_roi=config.d_detector_roi,
            d_model=config.d_model,
            d_camera=config.d_camera,
        )

        # Module 6: Motion + ego-motion fusion
        self.motion_fusion = nn.Sequential(
            nn.Linear(config.d_model + d_motion + config.d_camera, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        # Module 7: Temporal Object Encoder
        self.temporal_obj_encoder = TemporalObjectEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=getattr(config, 'n_temporal_obj_layers', 2),
            dropout=config.dropout,
        )

        # Module 8: Inter-Object Transformer (vanilla transformer encoder across objects)
        self.inter_object_encoder = InterObjectTransformer(
            d_model=config.d_model,
            n_layers=config.n_gnn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 9: Node predictor
        self.node_predictor = NodePredictor(
            d_memory=config.d_model,
            num_classes=num_object_classes,
        )

        # Module 10: Relationship predictor
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

        # Module 11: Temporal edge attention
        self.temporal_edge_attn = TemporalEdgeAttention(
            d_rel=config.d_rel,
            n_heads=config.n_rel_heads,
            n_layers=config.n_temporal_edge_layers,
            dropout=config.dropout,
        )

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
            camera_pose_seq:     (T, 4, 4) or None
            union_features_seq:  (T, K_max, d_union_roi) or None

        Returns:
            dict with (T, ...) padded tensors.
        """
        T = corners_seq.shape[0]
        device = corners_seq.device

        # ==================== Step 1: LKS buffer ====================
        buffer_all, staleness_all = vectorized_lks_buffer(
            raw_visual=visual_features_seq,
            visibility_mask=visibility_mask_seq,
            valid_mask=valid_mask_seq,
        )

        # ==================== Step 2: Global structural encoding ====================
        struct_all, _ = self.global_structural_encoder(
            corners_seq, valid_mask_seq,
        )

        # ==================== Step 3: Object spatial encoding ====================
        cam_all = None
        if camera_pose_seq is not None:
            _, cam_all = self.object_spatial_encoder(
                camera_pose=camera_pose_seq,
                corners=corners_seq,
                valid_mask=valid_mask_seq,
            )

        # ==================== Step 4: Ego-motion encoding ====================
        ego_tokens = None
        if camera_pose_seq is not None:
            ego_tokens = self.camera_temporal_encoder(camera_pose_seq)  # (T, d_camera)

        # ==================== Step 5: Object motion encoding ====================
        motion_feats = self.object_motion_encoder(
            velocity=None,
            valid_mask=valid_mask_seq,
        )
        if T > 1:
            velocity = ObjectMotionEncoder.compute_velocity(
                corners_seq[1:].mean(dim=2), corners_seq[:-1].mean(dim=2),
            )
            vel_expanded = velocity.unsqueeze(0).expand(T - 1, -1, -1) if velocity.dim() == 2 else velocity
            camera_R = camera_pose_seq[1:, :3, :3] if camera_pose_seq is not None else None
            motion_from_t1 = self.object_motion_encoder(
                velocity=vel_expanded,
                camera_R=camera_R,
                valid_mask=valid_mask_seq[1:],
            )
            motion_feats = torch.cat([motion_feats[:1], motion_from_t1], dim=0)

        # ==================== Step 6: Tokenizer (with camera) ====================
        tokens_all = self.tokenizer(
            geometry_tokens=struct_all,
            buffer_features=buffer_all,
            valid_mask=valid_mask_seq,
            cam_feats=cam_all,
            staleness=staleness_all,
        )

        # ==================== Step 7: Fuse motion + ego-motion ====================
        N = tokens_all.shape[1]
        if ego_tokens is not None:
            ego_expanded = ego_tokens.unsqueeze(1).expand(-1, N, -1)  # (T, N, d_camera)
        else:
            ego_expanded = torch.zeros(T, N, self.config.d_camera, device=device)

        tokens_all = self.motion_fusion(
            torch.cat([tokens_all, motion_feats, ego_expanded], dim=-1)
        )
        tokens_all = tokens_all * valid_mask_seq.unsqueeze(-1).float()

        # ==================== Step 8: Temporal object encoder ====================
        tokens_all = self.temporal_obj_encoder(
            tokens=tokens_all,
            valid_mask=valid_mask_seq,
        )

        # ==================== Step 9: Inter-object transformer ====================
        enriched_all = self.inter_object_encoder(
            tokens=tokens_all,
            corners=corners_seq,
            valid_mask=valid_mask_seq,
        )

        # ==================== Step 10: Node prediction ====================
        node_logits_all = self.node_predictor(enriched_all)

        # ==================== Step 11: Edge prediction ====================
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq,
            pair_valid, union_features_seq,
        )

        # ==================== Step 12: Temporal edge attention ====================
        enriched_rel = self.temporal_edge_attn(
            rel_tokens, pair_valid_out, person_idx_seq, object_idx_seq,
        )

        # ==================== Step 13: Predict distributions ====================
        edge_out = self.rel_predictor.batched_predict(enriched_rel, pair_valid_out)

        return {
            "node_logits": node_logits_all,
            "attention_logits": edge_out["attention_logits"],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
            "spatial_logits": edge_out["spatial_logits"],
            "contacting_logits": edge_out["contacting_logits"],
        }
