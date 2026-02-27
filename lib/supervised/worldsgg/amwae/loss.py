"""
AMWAE Triple-Objective Loss
=============================

L_total = L_vis + λ_mask * L_mask + λ_recon * L_recon + λ_contra * L_contrastive

Objective 1 — Feature Reconstruction (L_recon):
  MSE between memory-retrieved features and actual DINO features for
  tokens that were VISIBLE but artificially masked during training.

Objective 2 — Split Scene Graph Loss (L_vis + L_mask):
  CE/BCE for relationship pairs, split by visibility.

Objective 3 — Contrastive Memory Attention (L_contrastive):
  InfoNCE on cross-attention weights. Rewards attending to the correct
  object from past frames, penalizes attending to wrong objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from constants import Constants as const


class AMWAELoss(nn.Module):
    """
    Triple-objective loss for AMWAE training.

    Args:
        lambda_masked: Weight for masked-edge scene graph loss.
        lambda_recon: Feature reconstruction loss weight.
        lambda_contrastive: InfoNCE contrastive loss weight.
        temperature: InfoNCE temperature parameter.
        bce_loss: Use BCE (True) or MultiLabelMargin (False) for spatial/contacting.
        mode: Task mode (predcls/sgcls/sgdet).
    """

    def __init__(
        self,
        lambda_masked: float = 2.0,
        lambda_recon: float = 1.0,
        lambda_contrastive: float = 0.5,
        temperature: float = 0.07,
        bce_loss: bool = True,
        mode: str = "predcls",
    ):
        super().__init__()
        self.lambda_masked = lambda_masked
        self.lambda_recon = lambda_recon
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
        self.bce_loss = bce_loss
        self.mode = mode

        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCELoss()
        self._mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, List],
        gt_attention: List[torch.Tensor],
        gt_spatial: List[List],
        gt_contacting: List[List],
        gt_node_labels: Optional[List[torch.Tensor]] = None,
        visibility_mask_seq: Optional[List[torch.Tensor]] = None,
        person_idx_seq: Optional[List[torch.Tensor]] = None,
        object_idx_seq: Optional[List[torch.Tensor]] = None,
        valid_mask_seq: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the triple-objective loss.

        Returns dict of named losses ready for backpropagation.
        """
        device = predictions["attention_distribution"][0].device
        T = len(predictions["attention_distribution"])

        losses = {}

        # === Objective 1: Feature Reconstruction Loss ===
        loss_recon = self._compute_reconstruction_loss(predictions, device)
        losses["recon_loss"] = loss_recon * self.lambda_recon

        # === Objective 2: Split Scene Graph Loss ===
        sg_losses = self._compute_scene_graph_loss(
            predictions, gt_attention, gt_spatial, gt_contacting,
            gt_node_labels, visibility_mask_seq, person_idx_seq,
            object_idx_seq, valid_mask_seq, device,
        )
        losses.update(sg_losses)

        # === Objective 3: Contrastive Memory Attention Loss ===
        loss_contrastive = self._compute_contrastive_loss(predictions, device)
        losses["contrastive_loss"] = loss_contrastive * self.lambda_contrastive

        # Total
        losses["total"] = sum(losses.values())

        return losses

    # ------------------------------------------------------------------
    # Objective 1: Feature Reconstruction
    # ------------------------------------------------------------------

    def _compute_reconstruction_loss(
        self,
        predictions: Dict[str, List],
        device: torch.device,
    ) -> torch.Tensor:
        """
        MSE between retrieved tokens and original DINO features for
        artificially masked tokens (only during training).

        We compare the retrieved token representation with the original
        visual features for tokens that were masked during training but
        actually had DINO features available.
        """
        all_retrieved = []
        all_originals = []

        T = len(predictions["is_masked"])
        for t in range(T):
            is_masked_t = predictions["is_masked"][t]  # (N_t,) bool
            retrieved_t = predictions["retrieved_tokens"][t]  # (N_t, d_model)
            original_t = predictions["original_visual"][t]  # (N_t, d_visual)

            # Only compute recon loss for artificially masked tokens
            # (tokens that were visible but we masked during training)
            if is_masked_t.any():
                masked_indices = torch.where(is_masked_t)[0]
                # Truncate to match d_visual (retrieved is d_model, original is d_visual)
                d_visual = original_t.shape[-1]
                retrieved_proj = retrieved_t[masked_indices, :d_visual]
                original_proj = original_t[masked_indices]

                all_retrieved.append(retrieved_proj)
                all_originals.append(original_proj)

        if not all_retrieved:
            return torch.tensor(0.0, device=device, requires_grad=True)

        retrieved_cat = torch.cat(all_retrieved, dim=0)
        original_cat = torch.cat(all_originals, dim=0)

        return self._mse_loss(retrieved_cat, original_cat)

    # ------------------------------------------------------------------
    # Objective 2: Split Scene Graph Loss
    # ------------------------------------------------------------------

    def _compute_scene_graph_loss(
        self,
        predictions, gt_attention, gt_spatial, gt_contacting,
        gt_node_labels, visibility_mask_seq, person_idx_seq,
        object_idx_seq, valid_mask_seq, device,
    ) -> Dict[str, torch.Tensor]:
        """Split visible/masked scene graph loss (same structure as GL-STGN)."""
        T = len(predictions["attention_distribution"])

        # Accumulators for visible and masked pairs
        vis_att_p, vis_att_g = [], []
        vis_spa_p, vis_spa_g = [], []
        vis_con_p, vis_con_g = [], []
        mask_att_p, mask_att_g = [], []
        mask_spa_p, mask_spa_g = [], []
        mask_con_p, mask_con_g = [], []
        node_p, node_g = [], []

        for t in range(T):
            att_pred = predictions["attention_distribution"][t]
            spa_pred = predictions["spatial_distribution"][t]
            con_pred = predictions["contacting_distribution"][t]

            K_t = att_pred.shape[0]
            if K_t == 0:
                continue

            att_gt = gt_attention[t].to(device)
            spa_gt = self._build_multi_label_gt(gt_spatial[t], 6, device)
            con_gt = self._build_multi_label_gt(gt_contacting[t], 17, device)

            # Split by visibility
            if visibility_mask_seq is not None and person_idx_seq is not None:
                vis_t = visibility_mask_seq[t]
                p_idx = person_idx_seq[t]
                o_idx = object_idx_seq[t]
                pair_visible = vis_t[p_idx] & vis_t[o_idx]
                pair_masked = ~pair_visible

                if pair_visible.any():
                    vis_att_p.append(att_pred[pair_visible])
                    vis_att_g.append(att_gt[pair_visible])
                    vis_spa_p.append(spa_pred[pair_visible])
                    vis_spa_g.append(spa_gt[pair_visible])
                    vis_con_p.append(con_pred[pair_visible])
                    vis_con_g.append(con_gt[pair_visible])
                if pair_masked.any():
                    mask_att_p.append(att_pred[pair_masked])
                    mask_att_g.append(att_gt[pair_masked])
                    mask_spa_p.append(spa_pred[pair_masked])
                    mask_spa_g.append(spa_gt[pair_masked])
                    mask_con_p.append(con_pred[pair_masked])
                    mask_con_g.append(con_gt[pair_masked])
            else:
                vis_att_p.append(att_pred)
                vis_att_g.append(att_gt)
                vis_spa_p.append(spa_pred)
                vis_spa_g.append(spa_gt)
                vis_con_p.append(con_pred)
                vis_con_g.append(con_gt)

            # Node classification
            if gt_node_labels is not None and self.mode in [const.SGCLS, const.SGDET]:
                node_pred_t = predictions["node_logits"][t]
                node_gt_t = gt_node_labels[t].to(device)
                if valid_mask_seq is not None:
                    valid_t = valid_mask_seq[t]
                    node_pred_t = node_pred_t[valid_t]
                    node_gt_t = node_gt_t[valid_t]
                if len(node_gt_t) > 0:
                    node_p.append(node_pred_t)
                    node_g.append(node_gt_t)

        losses = {}
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        # Visible losses
        vis_losses = self._relationship_losses(vis_att_p, vis_att_g, vis_spa_p, vis_spa_g, vis_con_p, vis_con_g, device)
        for k, v in vis_losses.items():
            losses[f"vis_{k}"] = v

        # Masked losses (weighted)
        mask_losses = self._relationship_losses(mask_att_p, mask_att_g, mask_spa_p, mask_spa_g, mask_con_p, mask_con_g, device)
        for k, v in mask_losses.items():
            losses[f"masked_{k}"] = v * self.lambda_masked

        # Node loss
        if node_p:
            losses[const.OBJECT_LOSS] = self._ce_loss(
                torch.cat(node_p), torch.cat(node_g)
            )

        return losses

    # ------------------------------------------------------------------
    # Objective 3: Contrastive Memory Attention (InfoNCE)
    # ------------------------------------------------------------------

    def _compute_contrastive_loss(
        self,
        predictions: Dict[str, List],
        device: torch.device,
    ) -> torch.Tensor:
        """
        InfoNCE loss on cross-attention weights.

        For each masked token, the attention weight to memory slots
        belonging to the SAME object (positive) should be higher than
        to other objects (negatives).

        L = -log( sum(attn[positives]) / sum(attn[all]) )
        """
        all_losses = []

        T = len(predictions["attn_weights"])
        for t in range(T):
            attn_weights_t = predictions["attn_weights"][t]  # (N_t, M)
            is_masked_t = predictions["is_masked"][t]         # (N_t,)
            mem_obj_ids = predictions["memory_object_ids"][t]  # (M,)

            N_t = attn_weights_t.shape[0]
            M = attn_weights_t.shape[1] if attn_weights_t.dim() == 2 else 0

            if M == 0 or not is_masked_t.any():
                continue

            # Only compute for masked tokens
            masked_indices = torch.where(is_masked_t)[0]
            if len(masked_indices) == 0:
                continue

            masked_attn = attn_weights_t[masked_indices]  # (N_masked, M)

            # For each masked token [i], find memory slots with the same object ID
            # Object slot IDs of masked tokens
            masked_obj_ids = masked_indices  # Object slot index = position in the array

            # Build positive mask: (N_masked, M) — True where memory has same object
            positive_mask = (masked_obj_ids.unsqueeze(1) == mem_obj_ids.unsqueeze(0))  # (N_masked, M)

            # Skip if no positives exist in memory
            has_positives = positive_mask.any(dim=1)
            if not has_positives.any():
                continue

            # Filter to tokens with at least one positive in memory
            valid_masked = has_positives
            masked_attn = masked_attn[valid_masked]
            positive_mask = positive_mask[valid_masked]

            if masked_attn.shape[0] == 0:
                continue

            # InfoNCE: treat attention weights as logits
            # Scale by temperature
            logits = masked_attn / self.temperature  # (N_valid, M)

            # Positive sum (log-sum-exp of positives)
            # Mask negatives to -inf for logsumexp
            pos_logits = logits.masked_fill(~positive_mask, float("-inf"))
            log_pos = torch.logsumexp(pos_logits, dim=1)  # (N_valid,)

            # All logits logsumexp
            log_all = torch.logsumexp(logits, dim=1)  # (N_valid,)

            # InfoNCE: -log(pos / all) = log_all - log_pos
            nce_loss = (log_all - log_pos).mean()

            # Handle potential NaN/Inf from empty positives
            if torch.isfinite(nce_loss):
                all_losses.append(nce_loss)

        if not all_losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(all_losses).mean()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _relationship_losses(self, att_p, att_g, spa_p, spa_g, con_p, con_g, device):
        """Compute attention/spatial/contacting losses from accumulated pairs."""
        losses = {}
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        if att_p:
            losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(
                torch.cat(att_p), torch.cat(att_g)
            )
        else:
            losses[const.ATTENTION_RELATION_LOSS] = zero

        if spa_p:
            losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(
                torch.cat(spa_p), torch.cat(spa_g)
            )
        else:
            losses[const.SPATIAL_RELATION_LOSS] = zero

        if con_p:
            losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(
                torch.cat(con_p), torch.cat(con_g)
            )
        else:
            losses[const.CONTACTING_RELATION_LOSS] = zero

        return losses

    def _build_multi_label_gt(self, gt_list, num_classes, device):
        """Convert list of GT index lists to multi-hot BCE targets."""
        K = len(gt_list)
        if K == 0:
            return torch.zeros(0, num_classes, device=device)

        target = torch.zeros(K, num_classes, dtype=torch.float32, device=device)
        for i, indices in enumerate(gt_list):
            if isinstance(indices, torch.Tensor):
                idx = indices.long()
            elif isinstance(indices, (list, tuple)):
                idx = torch.tensor(indices, dtype=torch.long, device=device)
            else:
                idx = torch.tensor([indices], dtype=torch.long, device=device)
            if len(idx) > 0:
                target[i, idx] = 1.0
        return target
