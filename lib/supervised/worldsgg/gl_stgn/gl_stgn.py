"""
GL-STGN: Global-Local Spatio-Temporal Graph Network
=====================================================

Main model that wires together:
  1. GlobalStructuralEncoder — encodes 3D bbox layout ("wireframe")
  2. CameraPoseEncoder — camera viewpoint + per-object cam features
  3. CameraTemporalEncoder — ego-motion between consecutive frames
  4. MotionFeatureEncoder — velocity/acceleration from 4D trajectories
  5. PersistentWorldMemoryBank — GRU-based temporal memory for all objects
  6. RelationalGraphTransformer — spatial-aware context propagation
  7. NodePredictor + EdgePredictor(+union, +2D) — scene graph prediction

Processes a video's frames sequentially, maintaining the memory bank state,
and outputs per-frame scene graph predictions for ALL objects (visible + unseen).
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .memory_bank import PersistentWorldMemoryBank
from .graph_transformer import RelationalGraphTransformer
from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, EdgePredictor,
    CameraPoseEncoder, CameraTemporalEncoder, MotionFeatureEncoder,
    ObservabilityClassifier,
)


class GLSTGN(nn.Module):
    """
    Global-Local Spatio-Temporal Graph Network.

    Processes a sequence of frames from a video, maintaining persistent memory
    for all objects, and predicts the world scene graph at each timestep.
    Camera pose and ego-motion inform the memory update and tokenization.

    Args:
        config: GLSTGNConfig with architecture hyperparameters.
        num_object_classes: Number of object categories (incl. background).
        attention_class_num: Number of attention relationship classes.
        spatial_class_num: Number of spatial relationship classes.
        contact_class_num: Number of contacting relationship classes.
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

        # Module 1: Global Structural Encoder
        self.structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: Camera Pose Encoder
        self.camera_encoder = CameraPoseEncoder(
            d_camera=config.d_camera,
        )

        # Module 3: Camera Temporal Encoder (ego-motion)
        self.camera_temporal_encoder = CameraTemporalEncoder(
            d_camera=config.d_camera,
        )

        # Module 4: Persistent World Memory Bank
        self.memory_bank = PersistentWorldMemoryBank(
            d_memory=config.d_memory,
            d_visual=config.d_visual,
            d_struct=config.d_struct,
            d_detector_roi=config.d_detector_roi,
            n_heads=config.n_heads,
            dropout=config.dropout,
            d_camera=config.d_camera,
            d_motion=getattr(config, 'd_motion', 64),
        )

        # Module 5: Motion Feature Encoder (4D trajectory dynamics)
        d_motion = getattr(config, 'd_motion', 64)
        self.motion_encoder = MotionFeatureEncoder(
            d_motion=d_motion,
        )

        # Module 5b: Observability Classifier (non-differentiable, for graded shielding)
        self.obs_classifier = ObservabilityClassifier(
            frustum_thresh=getattr(config, 'frustum_thresh', -0.1),
        )

        # Module 6: Relational Graph Transformer
        self.graph_transformer = RelationalGraphTransformer(
            d_model=config.d_memory,
            n_layers=config.n_graph_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_memory * 2,
            dropout=config.dropout,
        )

        # Module 7: Prediction Heads
        self.node_predictor = NodePredictor(
            d_memory=config.d_memory,
            num_classes=num_object_classes,
        )

        self.edge_predictor = EdgePredictor(
            d_memory=config.d_memory,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
        )

    def forward(
        self,
        visual_features_seq: List[torch.Tensor],
        corners_seq: List[torch.Tensor],
        valid_mask_seq: List[torch.Tensor],
        visibility_mask_seq: List[torch.Tensor],
        person_idx_seq: List[torch.Tensor],
        object_idx_seq: List[torch.Tensor],
        p_mask_visual: float = 0.0,
        camera_pose_seq: Optional[List[torch.Tensor]] = None,
        union_features_seq: Optional[List[torch.Tensor]] = None,
        bboxes_2d_seq: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, List]:
        """
        Process a sequence of frames and predict scene graphs.

        Args:
            visual_features_seq: List of (N_t, d_detector_roi) tensors.
            corners_seq: List of (N_t, 8, 3) tensors.
            valid_mask_seq: List of (N_t,) bool tensors.
            visibility_mask_seq: List of (N_t,) bool tensors.
            person_idx_seq: List of (K_t,) long tensors.
            object_idx_seq: List of (K_t,) long tensors.
            p_mask_visual: Visual feature masking probability (training only).
            camera_pose_seq: List of (4, 4) tensors or None.
            union_features_seq: List of (K_t, 1024) tensors or None.
            bboxes_2d_seq: List of (N_t, 4) tensors or None.

        Returns:
            dict with lists (one per frame):
                node_logits, attention/spatial/contacting distributions,
                memory_states, detached edge predictions for memory shielding.
        """
        T = len(corners_seq)
        device = corners_seq[0].device

        outputs = {
            "node_logits": [],
            "attention_distribution": [],
            "spatial_distribution": [],
            "contacting_distribution": [],
            "memory_states": [],
            "obs_type": [],
        }

        memory = None
        prev_pose = None
        prev_corners = None
        prev_velocity = None

        for t in range(T):
            corners_t = corners_seq[t]          # (N_t, 8, 3)
            valid_t = valid_mask_seq[t]           # (N_t,)
            vis_t = visibility_mask_seq[t]        # (N_t,)
            visual_t = visual_features_seq[t]     # (N_t, d_detector_roi)
            person_idx_t = person_idx_seq[t]      # (K_t,)
            object_idx_t = object_idx_seq[t]      # (K_t,)

            # Optional per-frame data
            camera_pose_t = camera_pose_seq[t] if camera_pose_seq is not None else None
            union_feat_t = union_features_seq[t] if union_features_seq is not None else None
            bboxes_2d_t = bboxes_2d_seq[t] if bboxes_2d_seq is not None else None

            N_t = corners_t.shape[0]

            # --- Step 1: Encode global structure ---
            struct_tokens, global_token = self.structural_encoder(
                corners_t.unsqueeze(0),  # (1, N_t, 8, 3)
                valid_t.unsqueeze(0),    # (1, N_t)
            )
            struct_tokens = struct_tokens.squeeze(0)  # (N_t, d_struct)
            global_token = global_token.squeeze(0)    # (d_struct,)

            # --- Step 2: Camera pose encoding ---
            cam_feats = None
            ego_motion_token = None
            if camera_pose_t is not None:
                _, cam_feats = self.camera_encoder(
                    camera_pose=camera_pose_t,
                    corners=corners_t,
                    valid_mask=valid_t,
                )  # cam_feats: (N_t, d_camera)

                ego_motion_token = self.camera_temporal_encoder(
                    prev_pose=prev_pose,
                    curr_pose=camera_pose_t,
                )  # (d_camera,)

                prev_pose = camera_pose_t

            # --- Step 3: Compute motion features from 4D trajectories ---
            velocity = None
            acceleration = None
            if prev_corners is not None:
                velocity = MotionFeatureEncoder.compute_velocity(corners_t, prev_corners)
                if prev_velocity is not None:
                    acceleration = velocity - prev_velocity

            camera_R = camera_pose_t[:3, :3] if camera_pose_t is not None else None
            motion_feats = self.motion_encoder(
                velocity=velocity,
                acceleration=acceleration,
                camera_R=camera_R,
                valid_mask=valid_t,
            )  # (N_t, d_motion)

            # Track for next frame
            prev_corners = corners_t.detach()
            prev_velocity = velocity.detach() if velocity is not None else None

            # --- Step 4: Update memory bank ---
            if memory is None:
                # First frame: initialize memory
                memory = self.memory_bank.initialize_memory(
                    visual_features=visual_t,
                    struct_tokens=struct_tokens,
                    valid_mask=valid_t,
                    cam_feats=cam_feats,
                    motion_feats=motion_feats,
                )
            else:
                # Handle size changes (new objects appearing)
                N_prev = memory.shape[0]
                if N_t > N_prev:
                    new_mem = torch.zeros(
                        N_t - N_prev, self.config.d_memory,
                        device=device,
                    )
                    memory = torch.cat([memory, new_mem], dim=0)
                elif N_t < N_prev:
                    memory = memory[:N_t]

                # Update memory with ego-motion for unseen cross-attention
                memory = self.memory_bank.step(
                    memory=memory,
                    visual_features=visual_t,
                    struct_tokens=struct_tokens,
                    global_struct_tokens=struct_tokens,
                    visibility_mask=vis_t,
                    valid_mask=valid_t,
                    p_mask_visual=p_mask_visual if self.training else 0.0,
                    cam_feats=cam_feats,
                    ego_motion_token=ego_motion_token,
                    motion_feats=motion_feats,
                )

            # --- Step 5: Relational reasoning via Graph Transformer ---
            enriched = self.graph_transformer(
                memory_states=memory,
                corners=corners_t,
                valid_mask=valid_t,
            )  # (N_t, d_memory)

            # --- Step 6: Observability-Graded Memory Shielding ---
            # Classify each object's observability state
            obs_type_t = self.obs_classifier(
                camera_pose=camera_pose_t if camera_pose_t is not None else torch.eye(4, device=device),
                corners=corners_t,
                visibility_mask=vis_t,
                valid_mask=valid_t,
            )  # (N_t,) long — 0=NEVER_SEEN, 1=OUT_OF_FRUSTUM, 2=OCCLUDED, 3=VISIBLE

            # 3-tier gradient scaling:
            #   VISIBLE (3)        → 1.0 (full gradient)
            #   OCCLUDED (2)       → 0.5 (partial gradient — contextual cues exist)
            #   OUT_OF_FRUSTUM (1) → 0.0 (full detach — no visual evidence)
            #   NEVER_SEEN (0)     → 0.0 (full detach — no prior info)
            grad_scale = torch.zeros(N_t, 1, device=device)  # (N_t, 1)
            grad_scale[obs_type_t == ObservabilityClassifier.VISIBLE] = 1.0
            grad_scale[obs_type_t == ObservabilityClassifier.OCCLUDED] = 0.5
            # OUT_OF_FRUSTUM and NEVER_SEEN remain 0.0

            # Apply graded shielding: enriched_shielded = detached + grad_scale * (enriched - detached)
            # This allows partial gradients for occluded objects
            enriched_detached = enriched.detach()
            enriched_shielded = enriched_detached + grad_scale * (enriched - enriched_detached)

            # --- Step 7: Predict scene graph ---
            node_logits = self.node_predictor(enriched)  # (N_t, num_classes)

            # Full-gradient edge predictions (used for visible edges in loss)
            edge_out = self.edge_predictor(
                enriched_states=enriched,
                person_idx=person_idx_t,
                object_idx=object_idx_t,
                corners=corners_t,
                union_features=union_feat_t,
                bboxes_2d=bboxes_2d_t,
            )

            # Graded-shielded edge predictions (used for non-visible edges in loss)
            edge_out_shielded = self.edge_predictor(
                enriched_states=enriched_shielded,
                person_idx=person_idx_t,
                object_idx=object_idx_t,
                corners=corners_t,
                union_features=union_feat_t,
                bboxes_2d=bboxes_2d_t,
            )

            # Collect outputs
            outputs["node_logits"].append(node_logits)
            outputs["attention_distribution"].append(edge_out["attention_distribution"])
            outputs["spatial_distribution"].append(edge_out["spatial_distribution"])
            outputs["contacting_distribution"].append(edge_out["contacting_distribution"])
            outputs["memory_states"].append(memory.detach().clone())
            outputs["obs_type"].append(obs_type_t)

            # Graded-shielded predictions for non-visible edges
            if "attention_distribution_shielded" not in outputs:
                outputs["attention_distribution_shielded"] = []
                outputs["spatial_distribution_shielded"] = []
                outputs["contacting_distribution_shielded"] = []
            outputs["attention_distribution_shielded"].append(edge_out_shielded["attention_distribution"])
            outputs["spatial_distribution_shielded"].append(edge_out_shielded["spatial_distribution"])
            outputs["contacting_distribution_shielded"].append(edge_out_shielded["contacting_distribution"])

        return outputs

    def reset_memory(self):
        """Reset memory bank state (call between videos)."""
        pass

    def detach_memory(self):
        """Detach memory for BPTT truncation (called between chunks)."""
        pass


