"""
AMWAE++: Associative Masked World Auto-Encoder with Energy Transformer
=======================================================================

Subclass of AMWAE that replaces Module 8 (SpatialGNN) with
EnergyDiffusion — a weight-tied recurrent transformer that iterates
a single shared layer until the representation converges.

All other modules (scaffold tokenizer, memory, retriever, predictor,
temporal edge attention) are inherited unchanged from AMWAE.
"""

import logging
from typing import Dict, Optional

import torch

from .amwae import AMWAE
from .energy_diffusion import EnergyDiffusion

logger = logging.getLogger(__name__)


class AMWAEPP(AMWAE):
    """
    AMWAE++ with Energy Transformer diffusion.

    Differences from AMWAE:
      - Module 8 is EnergyDiffusion (weight-tied, convergence-based)
      - 75% fewer parameters in the diffusion module vs SpatialGNN
      - Adaptive inference compute (dynamic stopping when converged)
      - Returns h_prev for attractor stability loss

    Args:
        Same as AMWAE.
    """

    def __init__(
        self,
        config,
        num_object_classes: int = 37,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
    ):
        super().__init__(
            config=config,
            num_object_classes=num_object_classes,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
        )

        # Remove inherited SpatialGNN — AMWAE++ uses EnergyDiffusion instead.
        # Prevents DDP disconnected-graph crash and frees VRAM.
        del self.spatial_gnn

        # Override Module 8: Replace SpatialGNN with EnergyDiffusion
        self.diffusion = EnergyDiffusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
            train_iters=getattr(config, 'energy_train_iters', 4),
            eval_iters=getattr(config, 'energy_eval_iters', 15),
            epsilon=getattr(config, 'energy_epsilon', 1e-3),
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
        p_mask_visible: float = 0.0,
        camera_pose_seq: Optional[torch.Tensor] = None,
        union_features_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        AMWAE++ forward: same as AMWAE, but Step 8 uses iterative
        EnergyDiffusion instead of fixed-depth SpatialGNN.

        Returns extra keys for attractor stability loss:
            h_prev:   (T, N, d_model) — penultimate iteration state (detached)
            enriched: (T, N, d_model) — final converged representations
        """
        T = corners_seq.shape[0]
        device = corners_seq.device

        visual_all = visual_features_seq
        corners_all = corners_seq
        valid_all = valid_mask_seq
        visibility_all = visibility_mask_seq

        # ==================== Step 1: Structural encoding (B=T) ====================
        struct_all, _ = self.structural_encoder(corners_all, valid_all)

        # ==================== Step 2-3: Camera encoding (B=T) ====================
        cam_all = None
        ego_tokens_all = None
        if camera_pose_seq is not None:
            camera_pose_all = camera_pose_seq
            _, cam_all = self.camera_encoder(camera_pose_all, corners_all, valid_all)
            ego_tokens_all = self.camera_temporal_encoder(camera_pose_all)

        # ==================== Step 4: Motion features (B=T) ====================
        centers_all = corners_all.mean(dim=2)  # (T, N, 3)
        velocity_all = torch.zeros_like(centers_all)
        accel_all = torch.zeros_like(centers_all)

        # Only compute velocity where BOTH frames have valid objects
        valid_vel = valid_all[1:] & valid_all[:-1]  # (T-1, N)
        velocity_all[1:] = torch.where(
            valid_vel.unsqueeze(-1),
            centers_all[1:] - centers_all[:-1],
            torch.zeros_like(centers_all[1:]),
        )

        # Only compute acceleration where 3 consecutive frames are valid
        valid_acc = valid_all[2:] & valid_all[1:-1] & valid_all[:-2]  # (T-2, N)
        accel_all[2:] = torch.where(
            valid_acc.unsqueeze(-1),
            velocity_all[2:] - velocity_all[1:-1],
            torch.zeros_like(velocity_all[2:]),
        )

        camera_R_all = None
        if camera_pose_seq is not None:
            camera_R_all = camera_pose_seq[:, :3, :3]

        motion_all = self.motion_encoder(
            velocity=velocity_all, acceleration=accel_all,
            camera_R=camera_R_all, valid_mask=valid_all,
        )

        # ==================== Step 5: Scaffold tokenization (B=T) ====================
        hybrid_tokens, is_masked_all, original_visual_all, artificially_masked_all = self.scaffold_tokenizer(
            geometry_tokens=struct_all,
            visual_features=visual_all,
            visibility_mask=visibility_all,
            valid_mask=valid_all,
            p_mask_visible=p_mask_visible if self.training else 0.0,
            cam_feats=cam_all,
            motion_feats=motion_all,
            ego_tokens=ego_tokens_all,
        )

        # ==================== Step 6: Associative retrieval (bidirectional) ====================
        completed_tokens = self.retriever(
            tokens=hybrid_tokens,
            visibility_mask=visibility_all,
            valid_mask=valid_all,
            cam_feats=cam_all,
        )

        # ==================== Step 7: Visibility embedding (after retrieval) ====================
        # Use effective visibility: ground-truth visible AND not artificially masked.
        effective_visible = visibility_all & (~is_masked_all)  # (T, N)
        vis_ids = effective_visible.long()  # 0 = unseen/masked, 1 = directly observed
        completed_tokens = completed_tokens + self.visibility_emb(vis_ids)
        completed_tokens = completed_tokens * valid_all.unsqueeze(-1).float()  # re-zero padding

        # ==================== Step 8: EnergyDiffusion (iterative, weight-tied) ====================
        # This is the ONLY difference from AMWAE: SpatialGNN → EnergyDiffusion
        enriched_all, h_prev = self.diffusion(
            tokens=completed_tokens,
            corners=corners_all,
            valid_mask=valid_all,
        )  # (T, N, d_model), (T, N, d_model)

        # ==================== Step 9: Node prediction (B=T) ====================
        node_logits_all = self.node_predictor(enriched_all)

        # ==================== Step 10: Cross-view reconstruction ====================
        recon_pred_all = self.reconstruction_proj(completed_tokens)

        # ==================== Step 11: Batched edge prediction ====================
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq,
            pair_valid, union_features_seq,
        )

        # ==================== Step 12: Temporal edge attention ====================
        enriched_rel = self.temporal_edge_attn(rel_tokens, pair_valid_out, person_idx_seq, object_idx_seq)

        # ==================== Step 13: Predict distributions ====================
        edge_out = self.rel_predictor.batched_predict(enriched_rel, pair_valid_out)

        # ==================== Build output ====================
        outputs = {
            "node_logits": node_logits_all,
            "attention_logits": edge_out["attention_logits"],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
            "spatial_logits": edge_out["spatial_logits"],
            "contacting_logits": edge_out["contacting_logits"],
            "is_masked": is_masked_all,
            "artificially_masked": artificially_masked_all,                  # (T, N) — for reconstruction loss
            "original_visual": original_visual_all,
            "reconstruction_predictions": recon_pred_all,
            "reconstruction_targets": original_visual_all,
            # AMWAE++ specific: attractor stability loss inputs
            "h_prev": h_prev,            # (T, N, d_model) — penultimate, detached
            "enriched": enriched_all,     # (T, N, d_model) — final converged state
        }

        return outputs
