"""
Amnesic Geometric GNN
======================

Stateless feed-forward baseline with zero temporal memory.
Processes each frame independently — pure geometric reasoning
with camera-aware, observability-conditioned tokenization and
union feature edge prediction.

Pipeline:
  1. GlobalStructuralEncoder(corners) → geometry tokens
  2. CameraPoseEncoder(pose, corners) → camera token + per-object cam features
  3. ObservabilityClassifier(pose, corners, vis) → obs_type per object
  4. AmnesicTokenizer(geometry, DINO, cam_feats, obs_type, staleness) → hybrid tokens
  5. SpatialGNN(hybrid_tokens, corners) → enriched tokens
  6. NodePredictor + EdgePredictor(+union, +2D) → scene graph
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .amnesic_tokenizer import AmnesicTokenizer
from .spatial_gnn import SpatialGNN

from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, EdgePredictor,
    CameraPoseEncoder, ObservabilityClassifier,
)


class AmnesicGNN(nn.Module):
    """
    Amnesic Geometric GNN — zero-memory baseline.

    Each frame is processed as a completely independent puzzle.
    No temporal state, no memory bank, no recurrence.
    Unseen objects receive observability-conditioned surrogate
    embeddings instead of a single global [UNSEEN] vector.

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

        # Module 1: Geometry encoder
        self.structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: Camera pose encoder
        self.camera_encoder = CameraPoseEncoder(
            d_camera=config.d_camera,
        )

        # Module 3: Observability classifier (non-differentiable)
        self.obs_classifier = ObservabilityClassifier(
            frustum_thresh=getattr(config, 'frustum_thresh', -0.1),
        )

        # Module 4: Amnesic tokenizer (visible → DINO, unseen → conditioned surrogate)
        self.tokenizer = AmnesicTokenizer(
            d_struct=config.d_struct,
            d_visual=config.d_visual,
            d_model=config.d_model,
            d_detector_roi=config.d_detector_roi,
            d_camera=config.d_camera,
        )

        # Module 5: Spatial GNN (context propagation)
        self.spatial_gnn = SpatialGNN(
            d_model=config.d_model,
            n_layers=config.n_gnn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 6: Prediction heads
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
        camera_pose: Optional[torch.Tensor] = None,
        union_features: Optional[torch.Tensor] = None,
        bboxes_2d: Optional[torch.Tensor] = None,
        ever_seen_mask: Optional[torch.Tensor] = None,
        staleness: Optional[torch.Tensor] = None,
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
            camera_pose: (4, 4) or None — camera extrinsic matrix.
            union_features: (K, 1024) or None — union ROI features per pair.
            bboxes_2d: (N, 4) or None — 2D bounding boxes xyxy.
            ever_seen_mask: (N,) bool or None — True if object was ever visible
                           in any prior frame. If None, all valid objects treated as "ever seen."
            staleness: (N,) long or None — frames since last visible.
                       If None, defaults to 0 for visible, 1 for unseen.

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

        # Step 2: Camera pose encoding
        cam_feats = None
        obs_type = None
        if camera_pose is not None:
            _, cam_feats = self.camera_encoder(
                camera_pose=camera_pose,
                corners=corners,
                valid_mask=valid_mask,
            )  # cam_feats: (N, d_camera)

            # Step 3: Classify observability per object
            obs_type = self.obs_classifier(
                camera_pose=camera_pose,
                corners=corners,
                visibility_mask=visibility_mask,
                valid_mask=valid_mask,
                ever_seen_mask=ever_seen_mask,
            )  # (N,) long — 0=NEVER_SEEN, 1=OUT_OF_FRUSTUM, 2=OCCLUDED, 3=VISIBLE

        # Step 4: Conditioned feature assignment + camera features
        tokens = self.tokenizer(
            geometry_tokens=struct_tokens,
            visual_features=visual_features,
            visibility_mask=visibility_mask,
            valid_mask=valid_mask,
            cam_feats=cam_feats,
            obs_type=obs_type,
            staleness=staleness,
        )  # (N, d_model)

        # Step 5: Spatial context propagation via GNN (batched interface)
        enriched = self.spatial_gnn(
            tokens=tokens.unsqueeze(0),
            corners=corners.unsqueeze(0),
            valid_mask=valid_mask.unsqueeze(0),
        ).squeeze(0)  # (N, d_model)

        # Step 6: Predict scene graph
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


