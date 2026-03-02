"""
LKS GNN: Last-Known-State Graph Neural Network
=================================================

Main model for Baseline 1. Pairs a hard-coded, non-differentiable
LKS memory buffer with a stateless feed-forward GNN predictor.
Camera-aware tokenization and union feature edge prediction.

Sequential processing:
  1. LKSMemoryBuffer.update(DINO, visibility) → detached update
  2. GlobalStructuralEncoder(corners)          → geometry tokens
  3. CameraPoseEncoder(pose, corners)          → camera features
  4. LKSTokenizer(geometry, buffer, cam, stale) → hybrid tokens
  5. SpatialGNN(hybrid_tokens, corners)        → enriched tokens
  6. NodePredictor + EdgePredictor(+union)     → scene graph

Gradients only flow through steps 2-6 at the current frame.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .lks_memory import LKSMemoryBuffer
from .lks_tokenizer import LKSTokenizer

# Shared components from worldsgg_base
from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, RelationshipPredictor, SpatialGNN,
    CameraPoseEncoder, TemporalEdgeAttention,
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

        # Module 1: Geometry encoder
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

        # Module 2: Camera pose encoder
        self.camera_encoder = CameraPoseEncoder(
            d_camera=config.d_camera,
        )

        # Module 3: LKS Tokenizer (geometry + buffer + camera fusion)
        self.tokenizer = LKSTokenizer(
            d_struct=config.d_struct,
            d_visual=config.d_visual,
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

        # Module 5: Relationship predictor
        clip_path = getattr(config, 'clip_embeddings_path', '')
        self.rel_predictor = RelationshipPredictor(
            d_model=config.d_model,
            d_text=getattr(config, 'd_text', 128),
            d_rel=getattr(config, 'd_rel', 256),
            d_union_roi=getattr(config, 'd_union_roi', 1024),
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
            clip_embeddings_path=clip_path,
            n_rel_layers=getattr(config, 'n_rel_layers', 2),
            n_rel_heads=getattr(config, 'n_rel_heads', 4),
            dropout=config.dropout,
        )

        # Module 5b: Temporal edge attention
        self.temporal_edge_attn = TemporalEdgeAttention(
            d_rel=getattr(config, 'd_rel', 256),
            n_heads=getattr(config, 'n_rel_heads', 4),
            n_layers=getattr(config, 'n_temporal_edge_layers', 1),
            dropout=config.dropout,
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
        camera_pose: Optional[torch.Tensor] = None,
        union_features: Optional[torch.Tensor] = None,
        bboxes_2d: Optional[torch.Tensor] = None,
        class_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a SINGLE frame with passive memory.

        Args:
            visual_features: (N, d_detector_roi) — raw DINO features.
            corners: (N, 8, 3) — 3D bbox corners.
            valid_mask: (N,) bool.
            visibility_mask: (N,) bool.
            person_idx: (K,) long.
            object_idx: (K,) long.
            camera_pose: (4, 4) or None — camera extrinsic matrix.
            union_features: (K, 1024) or None — union ROI features per pair.
            bboxes_2d: (N, 4) or None — 2D bounding boxes xyxy.
            class_indices: (N,) long or None — object class indices.

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

        # Zero-order hold update
        self.memory_buffer.update(
            projected_visual, visibility_mask, valid_mask,
            camera_pose=camera_pose, corners=corners,
        )

        # Get buffered features (all detached)
        buffer_features = self.memory_buffer.get_features(N)  # (N, d_visual)
        staleness = self.memory_buffer.get_staleness(N)        # (N,) long

        # --- Step 3: Encode wireframe geometry ---
        struct_tokens, _ = self.structural_encoder(
            corners.unsqueeze(0),
            valid_mask.unsqueeze(0),
        )
        struct_tokens = struct_tokens.squeeze(0)  # (N, d_struct)

        # --- Step 3: Camera pose encoding ---
        cam_feats = None
        if camera_pose is not None:
            _, cam_feats = self.camera_encoder(
                camera_pose=camera_pose.unsqueeze(0),
                corners=corners.unsqueeze(0),
                valid_mask=valid_mask.unsqueeze(0),
            )  # cam_feats: (1, N, d_camera)
            cam_feats = cam_feats.squeeze(0)  # (N, d_camera)

        # --- Step 4: Fuse geometry + buffered visual + camera ---
        tokens = self.tokenizer(
            geometry_tokens=struct_tokens,
            buffer_features=buffer_features,
            valid_mask=valid_mask,
            cam_feats=cam_feats,
            staleness=staleness,
        )  # (N, d_model)

        # --- Step 5: Spatial GNN (batched interface) ---
        enriched = self.spatial_gnn(
            tokens=tokens.unsqueeze(0),
            corners=corners.unsqueeze(0),
            valid_mask=valid_mask.unsqueeze(0),
        ).squeeze(0)  # (N, d_model)

        # --- Step 6: Node prediction + form/self-attend rel tokens ---
        node_logits = self.node_predictor(enriched)

        person_class_idx = node_logits[person_idx].argmax(dim=-1)
        object_class_idx = node_logits[object_idx].argmax(dim=-1)

        rel_tokens = self.rel_predictor.form_rel_tokens(
            enriched_states=enriched,
            person_idx=person_idx,
            object_idx=object_idx,
            person_class_idx=person_class_idx,
            object_class_idx=object_class_idx,
            union_features=union_features,
        )
        rel_tokens = self.rel_predictor.self_attend(rel_tokens)

        return {
            "node_logits": node_logits,
            "rel_tokens": rel_tokens,
            "person_idx": person_idx,
            "object_idx": object_idx,
        }

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
        bboxes_2d_seq: Optional[List[torch.Tensor]] = None,
        class_indices_seq: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, List]:
        """
        Process a full video with per-pair temporal self-attention.

        Phase A: per-frame upstream + collect rel_tokens
        Phase B: temporal self-attention across all frames
        Phase C: predict from enriched tokens

        Returns:
            dict with per-frame lists: node_logits, att/spa/con distributions.
        """
        T = len(corners_seq)

        outputs = {
            "node_logits": [],
            "attention_distribution": [],
            "spatial_distribution": [],
            "contacting_distribution": [],
        }

        # ========== Phase A: Sequential per-frame processing ==========
        collected_rel = []
        collected_pidx = []
        collected_oidx = []

        for t in range(T):
            camera_pose_t = camera_pose_seq[t] if camera_pose_seq is not None else None
            union_feat_t = union_features_seq[t] if union_features_seq is not None else None
            bboxes_2d_t = bboxes_2d_seq[t] if bboxes_2d_seq is not None else None
            class_idx_t = class_indices_seq[t] if class_indices_seq is not None else None

            frame_out = self.forward_frame(
                visual_features=visual_features_seq[t],
                corners=corners_seq[t],
                valid_mask=valid_mask_seq[t],
                visibility_mask=visibility_mask_seq[t],
                person_idx=person_idx_seq[t],
                object_idx=object_idx_seq[t],
                camera_pose=camera_pose_t,
                union_features=union_feat_t,
                bboxes_2d=bboxes_2d_t,
                class_indices=class_idx_t,
            )

            outputs["node_logits"].append(frame_out["node_logits"])
            collected_rel.append(frame_out["rel_tokens"])
            collected_pidx.append(frame_out["person_idx"])
            collected_oidx.append(frame_out["object_idx"])

        # ========== Phase B: Per-pair temporal self-attention ==========
        enriched_rel = self.temporal_edge_attn(collected_rel, collected_pidx, collected_oidx)

        # ========== Phase C: Predict from enriched tokens ==========
        for t in range(T):
            edge_out = self.rel_predictor.predict_from_tokens(enriched_rel[t])
            outputs["attention_distribution"].append(edge_out["attention_distribution"])
            outputs["spatial_distribution"].append(edge_out["spatial_distribution"])
            outputs["contacting_distribution"].append(edge_out["contacting_distribution"])

        return outputs

