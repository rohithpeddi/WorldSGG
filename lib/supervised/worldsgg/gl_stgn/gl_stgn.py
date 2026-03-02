"""
GL-STGN: Global-Local Spatio-Temporal Graph Network
======================================================

Simplified single-pass pipeline:

  1. GlobalStructuralEncoder(corners)           → (T, N, d_struct)
  2. CameraPoseEncoder(pose, corners)           → (T, N, d_camera)
  3. CameraTemporalEncoder(pose_stack)          → (T, d_camera)
  4. MotionFeatureEncoder(vel, acc)             → (T, N, d_motion)
  5. TemporalObjectTransformer(features, vis)   → (T, N, d_memory)
  6. SpatialGNN(memory, corners, valid)         → (T, N, d_memory)
  7. NodePredictor(enriched)                    → (T, N, num_classes)
  8. EdgeMLP(person ⊕ object)                  → (T, K_max, att+spa+con)

All batched with B=T. No per-frame loops. Temporal context flows
through step 5 (per-object bidirectional attention over all frames).
Spatial context propagates in step 6 (per-frame inter-object attention).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .memory_bank import TemporalObjectTransformer
from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, SpatialGNN,
    CameraPoseEncoder, CameraTemporalEncoder, MotionFeatureEncoder,
)


class GLSTGN(nn.Module):
    """
    Global-Local Spatio-Temporal Graph Network.

    Processes all T frames in a single forward pass with B=T batching.
    TemporalObjectTransformer provides per-object bidirectional attention
    across all frames. SpatialGNN provides per-frame inter-object context.
    Edge predictions use a simple MLP on concatenated person+object states
    (temporal context is already baked into both endpoints).

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
        self.num_object_classes = num_object_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num

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
        self.motion_encoder = MotionFeatureEncoder(
            d_motion=config.d_motion,
        )

        # Module 5: Temporal Object Transformer (world memory bank)
        self.temporal_transformer = TemporalObjectTransformer(
            d_visual=config.d_visual,
            d_struct=config.d_struct,
            d_camera=config.d_camera,
            d_motion=config.d_motion,
            d_detector_roi=config.d_detector_roi,
            d_memory=config.d_memory,
            n_heads=config.n_heads,
            n_layers=config.n_temporal_layers,
            dropout=config.dropout,
            max_T=config.max_T,
        )

        # Module 6: Spatial GNN — per-frame inter-object context (B=T)
        self.spatial_gnn = SpatialGNN(
            d_model=config.d_memory,
            n_layers=config.n_graph_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_memory * 2,
            dropout=config.dropout,
        )

        # Module 7: Node Predictor — (B, N, d_memory) → (B, N, num_classes)
        self.node_predictor = NodePredictor(
            d_memory=config.d_memory,
            num_classes=num_object_classes,
        )

        # Module 8: Edge MLP — [person ⊕ object] → [att + spa + con]
        total_edge_classes = attention_class_num + spatial_class_num + contact_class_num
        d_rel = getattr(config, 'd_rel', 256)
        self.edge_mlp = nn.Sequential(
            nn.Linear(config.d_memory * 2, d_rel),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_rel),
            nn.Linear(d_rel, d_rel),
            nn.ReLU(inplace=True),
            nn.Linear(d_rel, total_edge_classes),
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
            visual_features_seq: T-list of (N, d_detector_roi) tensors.
            corners_seq:         T-list of (N, 8, 3) tensors.
            valid_mask_seq:      T-list of (N,) bool tensors.
            visibility_mask_seq: T-list of (N,) bool tensors.
            person_idx_seq:      T-list of (K_t,) long tensors.
            object_idx_seq:      T-list of (K_t,) long tensors.
            camera_pose_seq:     T-list of (4, 4) tensors or None.
            union_features_seq:  Unused (kept for interface compat).

        Returns:
            dict with per-frame lists: node_logits, att/spa/con distributions.
        """
        T = len(corners_seq)
        device = corners_seq[0].device

        # ==================== Stack inputs (T, N, ...) ====================
        visual_all = torch.stack(visual_features_seq)       # (T, N, d_roi)
        corners_all = torch.stack(corners_seq)              # (T, N, 8, 3)
        valid_all = torch.stack(valid_mask_seq)              # (T, N)
        visibility_all = torch.stack(visibility_mask_seq)    # (T, N)

        # ==================== Step 1: Structural encoding (B=T) ====================
        struct_all, _ = self.structural_encoder(corners_all, valid_all)  # (T, N, d_struct)

        # ==================== Step 2-3: Camera encoding (B=T) ====================
        cam_all = None
        ego_tokens_all = None
        if camera_pose_seq is not None:
            camera_pose_all = torch.stack(camera_pose_seq)  # (T, 4, 4)
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
            camera_R_all = torch.stack(camera_pose_seq)[:, :3, :3]

        motion_all = self.motion_encoder(
            velocity=velocity_all, acceleration=accel_all,
            camera_R=camera_R_all, valid_mask=valid_all,
        )  # (T, N, d_motion)

        # ==================== Step 5: Temporal Object Transformer ====================
        if ego_tokens_all is None:
            ego_tokens_all = torch.zeros(T, self.config.d_camera, device=device)

        memory_all = self.temporal_transformer(
            visual_features=visual_all, struct_tokens=struct_all,
            cam_feats=cam_all, motion_feats=motion_all,
            ego_tokens=ego_tokens_all, visibility_mask=visibility_all,
            valid_mask=valid_all,
        )  # (T, N, d_memory)

        # ==================== Step 6: Spatial GNN (B=T) ====================
        enriched_all = self.spatial_gnn(
            tokens=memory_all, corners=corners_all, valid_mask=valid_all,
        )  # (T, N, d_memory)

        # ==================== Step 7: Node prediction (B=T) ====================
        node_logits_all = self.node_predictor(enriched_all)  # (T, N, num_classes)

        # ==================== Step 8: Edge prediction (batched) ====================
        # Pad person/object indices to K_max and batch-gather
        K_per_frame = [p.shape[0] for p in person_idx_seq]
        K_max = max(K_per_frame) if K_per_frame else 0

        if K_max > 0:
            # Pad indices to (T, K_max) for batched gather
            padded_pidx = torch.zeros(T, K_max, dtype=torch.long, device=device)
            padded_oidx = torch.zeros(T, K_max, dtype=torch.long, device=device)
            edge_valid = torch.zeros(T, K_max, dtype=torch.bool, device=device)

            for t in range(T):
                K_t = K_per_frame[t]
                if K_t > 0:
                    padded_pidx[t, :K_t] = person_idx_seq[t]
                    padded_oidx[t, :K_t] = object_idx_seq[t]
                    edge_valid[t, :K_t] = True

            # Gather person and object representations: (T, K_max, d_memory)
            person_repr = torch.gather(
                enriched_all, 1, padded_pidx.unsqueeze(-1).expand(-1, -1, enriched_all.shape[-1])
            )
            object_repr = torch.gather(
                enriched_all, 1, padded_oidx.unsqueeze(-1).expand(-1, -1, enriched_all.shape[-1])
            )

            # Concatenate and predict: (T, K_max, d_memory*2) → (T, K_max, total_classes)
            pair_input = torch.cat([person_repr, object_repr], dim=-1)
            edge_logits = self.edge_mlp(pair_input)  # (T, K_max, att+spa+con)

            # Split into att/spa/con
            att_end = self.attention_class_num
            spa_end = att_end + self.spatial_class_num
            att_logits = edge_logits[:, :, :att_end]           # (T, K_max, 3)
            spa_logits = edge_logits[:, :, att_end:spa_end]    # (T, K_max, 6)
            con_logits = edge_logits[:, :, spa_end:]           # (T, K_max, 17)

        # ==================== Build output lists ====================
        outputs: Dict[str, List] = {
            "node_logits": [],
            "attention_distribution": [],
            "spatial_distribution": [],
            "contacting_distribution": [],
        }

        for t in range(T):
            outputs["node_logits"].append(node_logits_all[t])  # (N, C)
            K_t = K_per_frame[t]
            if K_max > 0 and K_t > 0:
                outputs["attention_distribution"].append(att_logits[t, :K_t])
                outputs["spatial_distribution"].append(torch.sigmoid(spa_logits[t, :K_t]))
                outputs["contacting_distribution"].append(torch.sigmoid(con_logits[t, :K_t]))
            else:
                outputs["attention_distribution"].append(torch.zeros(0, self.attention_class_num, device=device))
                outputs["spatial_distribution"].append(torch.zeros(0, self.spatial_class_num, device=device))
                outputs["contacting_distribution"].append(torch.zeros(0, self.contact_class_num, device=device))

        return outputs
