"""
AMWAE: Associative Masked World Auto-Encoder
==============================================

Main model that wires together:
  1. GlobalStructuralEncoder — encodes wireframe into geometry tokens
  2. CameraPoseEncoder — camera viewpoint + per-object cam features
  3. ScaffoldTokenizer — binds visual evidence or [MASK] + cam features to geometry tokens
  4. PerObjectEpisodicMemory — K-slot per-object memory with viewpoint diversity eviction
  5. AssociativeRetriever — view-aware cross-attention retrieval to auto-complete masked tokens
  6. ContextualDiffusion — self-attention for context propagation
  7. NodePredictor + EdgePredictor(+union, +2D) — scene graph prediction
  8. ObservabilityClassifier — structured masking for simulated-unseen training

The model initializes the graph TOP-DOWN from the complete wireframe scaffold,
then auto-completes missing visual evidence via associative memory retrieval.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .scaffold_tokenizer import ScaffoldTokenizer
from .per_object_memory import PerObjectEpisodicMemory
from .associative_retriever import AssociativeRetriever
from .contextual_diffusion import ContextualDiffusion

from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, EdgePredictor,
    CameraPoseEncoder, ObservabilityClassifier,
)


class AMWAE(nn.Module):
    """
    Associative Masked World Auto-Encoder.

    Processes frames sequentially, but each frame is an independent
    masked auto-encoding step (no RNN/GRU recurrence). The episodic
    memory bank provides temporal context through cross-attention.
    Camera-aware tokenization enhances viewpoint reasoning.

    Args:
        config: AMWAEConfig with architecture hyperparameters.
        num_object_classes: Number of object categories.
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

        # Module 1: Global Structural Encoder
        self.structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: Camera Pose Encoder
        self.camera_encoder = CameraPoseEncoder(
            d_camera=config.d_camera,
        )

        # Module 3: Scaffold Tokenizer (evidence binding + masking + camera)
        self.scaffold_tokenizer = ScaffoldTokenizer(
            d_struct=config.d_struct,
            d_visual=config.d_visual,
            d_model=config.d_model,
            d_detector_roi=config.d_detector_roi,
            d_camera=config.d_camera,
        )

        # Module 4: Per-Object Episodic Memory (4A — replaces FIFO)
        self.memory_bank = PerObjectEpisodicMemory(
            max_objects=config.max_objects,
            slots_per_object=getattr(config, 'slots_per_object', 5),
            d_memory=config.d_model,
        )

        # Camera feature projector for memory storage
        self.mem_cam_proj = nn.Linear(config.d_camera, config.d_model)

        # Module 5: Associative Retriever (view-aware cross-attention, 4B)
        self.retriever = AssociativeRetriever(
            d_model=config.d_model,
            n_layers=config.n_cross_attn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
            d_camera=config.d_camera,
        )

        # Module 6: Contextual Diffusion (self-attention)
        self.diffusion = ContextualDiffusion(
            d_model=config.d_model,
            n_layers=config.n_self_attn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 7: Prediction Heads
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

        # Module 8: Observability Classifier (for structured masking, 4C)
        self.obs_classifier = ObservabilityClassifier(
            frustum_thresh=getattr(config, 'frustum_thresh', -0.1),
        )

        # Cross-view reconstruction projection (4D)
        self.reconstruction_proj = nn.Linear(config.d_model, config.d_visual)

    def forward(
        self,
        visual_features_seq: List[torch.Tensor],
        corners_seq: List[torch.Tensor],
        valid_mask_seq: List[torch.Tensor],
        visibility_mask_seq: List[torch.Tensor],
        person_idx_seq: List[torch.Tensor],
        object_idx_seq: List[torch.Tensor],
        p_mask_visible: float = 0.0,
        camera_pose_seq: Optional[List[torch.Tensor]] = None,
        union_features_seq: Optional[List[torch.Tensor]] = None,
        bboxes_2d_seq: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, List]:
        """
        Process a sequence of frames via masked auto-encoding.

        Args:
            visual_features_seq: List of (N_t, d_detector_roi) — DINO features.
            corners_seq: List of (N_t, 8, 3) — world-frame 3D corners.
            valid_mask_seq: List of (N_t,) bool — valid objects.
            visibility_mask_seq: List of (N_t,) bool — visible in FOV.
            person_idx_seq: List of (K_t,) — person pair indices.
            object_idx_seq: List of (K_t,) — object pair indices.
            p_mask_visible: Training masking probability for visible objects.
            camera_pose_seq: List of (4, 4) tensors or None.
            union_features_seq: List of (K_t, 1024) tensors or None.
            bboxes_2d_seq: List of (N_t, 4) tensors or None.

        Returns:
            dict with per-frame lists including node_logits,
            attention/spatial/contacting distributions, masking info,
            retrieved tokens, attention weights, and simulated-unseen outputs.
        """
        T = len(corners_seq)
        device = corners_seq[0].device

        outputs = {
            "node_logits": [],
            "attention_distribution": [],
            "spatial_distribution": [],
            "contacting_distribution": [],
            "is_masked": [],
            "original_visual": [],
            "retrieved_tokens": [],
            "attn_weights": [],
            "memory_object_ids": [],
            # Simulated-unseen clean fine-tuning outputs
            "simulated_unseen_predictions": [],
            "simulated_unseen_gt": [],
            # Cross-view reconstruction targets (4D)
            "reconstruction_targets": [],
            "reconstruction_predictions": [],
        }

        for t in range(T):
            corners_t = corners_seq[t]          # (N_t, 8, 3)
            valid_t = valid_mask_seq[t]           # (N_t,)
            vis_t = visibility_mask_seq[t]        # (N_t,)
            visual_t = visual_features_seq[t]     # (N_t, d_detector_roi)
            person_idx_t = person_idx_seq[t]
            object_idx_t = object_idx_seq[t]

            # Optional per-frame data
            camera_pose_t = camera_pose_seq[t] if camera_pose_seq is not None else None
            union_feat_t = union_features_seq[t] if union_features_seq is not None else None
            bboxes_2d_t = bboxes_2d_seq[t] if bboxes_2d_seq is not None else None

            N_t = corners_t.shape[0]

            # --- Step 1: Encode wireframe geometry ---
            struct_tokens, _ = self.structural_encoder(
                corners_t.unsqueeze(0),
                valid_t.unsqueeze(0),
            )
            struct_tokens = struct_tokens.squeeze(0)  # (N_t, d_struct)

            # --- Step 2: Camera pose encoding ---
            cam_feats = None
            if camera_pose_t is not None:
                _, cam_feats = self.camera_encoder(
                    camera_pose=camera_pose_t,
                    corners=corners_t,
                    valid_mask=valid_t,
                )  # cam_feats: (N_t, d_camera)

            # --- Step 3: Scaffold tokenization (bind evidence / apply mask + camera) ---
            hybrid_tokens, is_masked_t, original_visual_t = self.scaffold_tokenizer(
                geometry_tokens=struct_tokens,
                visual_features=visual_t,
                visibility_mask=vis_t,
                valid_mask=valid_t,
                p_mask_visible=p_mask_visible if self.training else 0.0,
                cam_feats=cam_feats,
            )

            # --- Step 4: Store visible tokens in per-object episodic memory ---
            with torch.no_grad():
                # Compute unmasked tokens for memory storage
                cam_for_storage = cam_feats if cam_feats is not None else torch.zeros(
                    N_t, self.config.d_camera, device=device,
                )
                unmasked_tokens = self.scaffold_tokenizer.fusion_proj(
                    torch.cat([
                        struct_tokens,
                        self.scaffold_tokenizer.visual_projector(visual_t),
                        cam_for_storage,
                    ], dim=-1)
                )

            # Store visible objects in per-object memory slots
            visible_indices = torch.where(vis_t & valid_t)[0]
            if visible_indices.numel() > 0 and camera_pose_t is not None:
                self.memory_bank.store(
                    object_indices=visible_indices,
                    features=unmasked_tokens[visible_indices],
                    camera_pose=camera_pose_t,
                )

            # --- Step 5: View-aware associative retrieval from per-object memory ---
            # Retrieve all stored memory entries
            all_indices = torch.arange(N_t, device=device)
            mem_entries, mem_poses, mem_mask = self.memory_bank.retrieve(all_indices)
            # mem_entries: (N_t, K, d_model), mem_mask: (N_t, K)

            # Flatten for cross-attention: (N_t * K, d_model)
            K_slots = mem_entries.shape[1]
            flat_mem = mem_entries.view(-1, self.config.d_model)  # (N_t * K, d_model)
            flat_mask = mem_mask.view(-1)  # (N_t * K,)

            # Compute camera features for memory entries (for view-aware retrieval)
            query_pose_feats_t = cam_feats  # (N_t, d_camera) or None
            memory_pose_feats_t = None
            if cam_feats is not None and camera_pose_t is not None:
                # For each memory entry, compute cam features from capture pose
                # Simplified: use stored pose positions as features
                mem_cam_raw = mem_poses.view(-1, 4, 4)  # (N_t*K, 4, 4)
                # Use translation + view direction as compact pose representation
                mem_positions = mem_cam_raw[:, :3, 3]  # (N_t*K, 3)
                mem_view_dirs = -mem_cam_raw[:, :3, 2]  # (N_t*K, 3)
                mem_pose_compact = torch.cat([mem_positions, mem_view_dirs], dim=-1)  # (N_t*K, 6)
                # Project to d_camera via zero-padding + linear
                mem_pose_padded = torch.zeros(mem_pose_compact.shape[0], self.config.d_camera, device=device)
                mem_pose_padded[:, :6] = mem_pose_compact
                memory_pose_feats_t = mem_pose_padded  # (N_t*K, d_camera)

            completed_tokens, attn_weights_t = self.retriever(
                tokens=hybrid_tokens,
                memory_tokens=flat_mem,
                memory_valid=flat_mask,
                query_pose_feats=query_pose_feats_t,
                memory_pose_feats=memory_pose_feats_t,
            )

            # --- Step 6: Contextual diffusion (self-attention) ---
            enriched = self.diffusion(
                tokens=completed_tokens,
                corners=corners_t,
                valid_mask=valid_t,
            )

            # --- Step 7: Scene graph prediction ---
            node_logits = self.node_predictor(enriched)

            edge_out = self.edge_predictor(
                enriched_states=enriched,
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
            outputs["is_masked"].append(is_masked_t)
            outputs["original_visual"].append(original_visual_t)
            outputs["retrieved_tokens"].append(completed_tokens)
            outputs["attn_weights"].append(attn_weights_t)
            outputs["memory_object_ids"].append(all_indices)

            # Cross-view reconstruction output (4D)
            recon_pred = self.reconstruction_proj(completed_tokens)  # (N_t, d_visual)
            outputs["reconstruction_predictions"].append(recon_pred)
            outputs["reconstruction_targets"].append(original_visual_t)

            # --- Step 8: Structured Simulated-Unseen (4C, training only) ---
            if self.training and getattr(self.config, 'p_simulate_unseen', 0) > 0:
                p_sim = self.config.p_simulate_unseen
                n_visible = vis_t.sum().item()
                n_to_mask = max(1, int(n_visible * p_sim))

                if n_visible > 1:
                    visible_indices = torch.where(vis_t & valid_t)[0]

                    # Structured masking (4C): classify observability to pick
                    # which visible objects to mask. Prefer objects near
                    # frustum boundary (more realistic unseen simulation).
                    if camera_pose_t is not None:
                        obs_type_t = self.obs_classifier(
                            camera_pose=camera_pose_t,
                            corners=corners_t,
                            visibility_mask=vis_t,
                            valid_mask=valid_t,
                        )
                        # Compute view alignment for prioritized masking
                        R = camera_pose_t[:3, :3]
                        cam_pos = camera_pose_t[:3, 3]
                        view_dir = -R[:, 2]
                        view_dir = view_dir / (view_dir.norm() + 1e-8)
                        centers = corners_t.mean(dim=1)
                        cam_to_obj = centers - cam_pos.unsqueeze(0)
                        cam_to_obj_norm = cam_to_obj / (cam_to_obj.norm(dim=-1, keepdim=True) + 1e-8)
                        view_align = (cam_to_obj_norm * view_dir.unsqueeze(0)).sum(dim=-1)

                        # Prefer masking objects near frustum edge (lower alignment)
                        vis_alignment = view_align[visible_indices]
                        _, sort_order = vis_alignment.sort()
                        sim_mask_indices = visible_indices[sort_order[:n_to_mask]]
                    else:
                        # Fallback: random masking
                        perm = torch.randperm(len(visible_indices), device=device)
                        sim_mask_indices = visible_indices[perm[:n_to_mask]]

                    sim_vis_t = vis_t.clone()
                    sim_vis_t[sim_mask_indices] = False

                    # Re-tokenize with simulated mask + camera features
                    sim_tokens, sim_is_masked, _ = self.scaffold_tokenizer(
                        geometry_tokens=struct_tokens,
                        visual_features=visual_t,
                        visibility_mask=sim_vis_t,
                        valid_mask=valid_t,
                        p_mask_visible=0.0,
                        cam_feats=cam_feats,
                    )

                    sim_completed, _ = self.retriever(
                        tokens=sim_tokens,
                        memory_tokens=flat_mem,
                        memory_valid=flat_mask,
                        query_pose_feats=query_pose_feats_t,
                        memory_pose_feats=memory_pose_feats_t,
                    )

                    sim_enriched = self.diffusion(
                        tokens=sim_completed,
                        corners=corners_t,
                        valid_mask=valid_t,
                    )

                    sim_edge_out = self.edge_predictor(
                        enriched_states=sim_enriched,
                        person_idx=person_idx_t,
                        object_idx=object_idx_t,
                        corners=corners_t,
                        union_features=union_feat_t,
                        bboxes_2d=bboxes_2d_t,
                    )

                    outputs["simulated_unseen_predictions"].append(sim_edge_out)
                    outputs["simulated_unseen_gt"].append(None)
                else:
                    outputs["simulated_unseen_predictions"].append(None)
                    outputs["simulated_unseen_gt"].append(None)
            else:
                outputs["simulated_unseen_predictions"].append(None)
                outputs["simulated_unseen_gt"].append(None)

        return outputs

    def reset_memory(self):
        """Reset episodic memory bank (call between videos)."""
        self.memory_bank.reset()


