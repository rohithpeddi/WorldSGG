"""
LKS GNN: Last-Known-State Graph Neural Network
=================================================

Main model for Baseline 1. Pairs a hard-coded, non-differentiable
LKS memory buffer with a stateless feed-forward GNN predictor.
Camera-aware tokenization and union feature edge prediction.

Sequential processing:
  1. LKSMemoryBuffer.update(DINO, visibility, pose) → detached update + obs classification
  2. GlobalStructuralEncoder(corners)               → geometry tokens
  3. CameraPoseEncoder(pose, corners)               → camera features
  4. LKSTokenizer(geometry, buffer, cam, obs, stale) → hybrid tokens
  5. SpatialGNN(hybrid_tokens, corners)             → enriched tokens
  6. NodePredictor + EdgePredictor(+union, +2D)     → scene graph

Gradients only flow through steps 2-6 at the current frame.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .lks_memory import LKSMemoryBuffer
from .lks_tokenizer import LKSTokenizer

# Shared components from worldsgg_base
from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, EdgePredictor, SpatialGNN,
    CameraPoseEncoder, FeatureAging,
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

        # Module 5: Prediction heads
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

        # Module 6: Feature Aging (learned staleness + pose-delta blending)
        self.feature_aging = FeatureAging(
            d_visual=config.d_visual,
            n_classes=num_object_classes,
        )

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
            class_indices: (N,) long or None — object class indices (for FeatureAging).

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

        # Zero-order hold update + observability classification
        self.memory_buffer.update(
            projected_visual, visibility_mask, valid_mask,
            camera_pose=camera_pose, corners=corners,
        )

        # Get buffered features + observability metadata (all detached)
        buffer_features = self.memory_buffer.get_features(N)  # (N, d_visual)
        obs_state = self.memory_buffer.get_obs_state(N)        # (N,) long
        staleness = self.memory_buffer.get_staleness(N)        # (N,) long

        # --- Step 2: Apply feature aging ---
        # Blend stale features toward class prototypes based on staleness + pose delta
        if camera_pose is not None and class_indices is not None:
            capture_poses = self.memory_buffer.get_capture_poses(N)  # (N, 4, 4)
            pose_delta = FeatureAging.compute_pose_delta(camera_pose, capture_poses)  # (N,)
            buffer_features = self.feature_aging(
                stale_features=buffer_features,
                staleness=staleness,
                pose_delta=pose_delta,
                class_indices=class_indices,
                valid_mask=valid_mask,
            )  # (N, d_visual)

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
                camera_pose=camera_pose,
                corners=corners,
                valid_mask=valid_mask,
            )  # cam_feats: (N, d_camera)

        # --- Step 4: Fuse geometry + buffered visual + camera + observability ---
        tokens = self.tokenizer(
            geometry_tokens=struct_tokens,
            buffer_features=buffer_features,
            valid_mask=valid_mask,
            cam_feats=cam_feats,
            obs_state=obs_state,
            staleness=staleness,
        )  # (N, d_model)

        # --- Step 5: Spatial GNN ---
        enriched = self.spatial_gnn(
            tokens=tokens,
            corners=corners,
            valid_mask=valid_mask,
        )  # (N, d_model)

        # --- Step 6: Predict scene graph ---
        node_logits = self.node_predictor(enriched)
        edge_out = self.edge_predictor(
            enriched_states=enriched,
            person_idx=person_idx,
            object_idx=object_idx,
            corners=corners,
            union_features=union_features,
            bboxes_2d=bboxes_2d,
        )

        return {
            "node_logits": node_logits,
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
        }

