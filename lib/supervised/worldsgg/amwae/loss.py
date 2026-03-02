"""
AMWAE Triple-Objective Loss — VLM Noisy Label Training (Padded Tensor API)
===========================================================================

Accepts pre-padded (T, K_max, C) tensors with pair_valid mask from dataset.
No per-frame loops or _build_multi_label_gt needed.

L_total = L_vis + λ_vlm * L_masked + λ_recon * λ_recon_dominance * L_recon
        + λ_contra * L_contrastive + L_simulated_unseen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from constants import Constants as const
from lib.supervised.worldsgg.worldsgg_base import LabelSmoother


class AMWAELoss(nn.Module):
    """Triple-objective loss with VLM noise handling + simulated-unseen.

    All edge-level inputs are pre-padded (T, K_max, ...) tensors with
    pair_valid masks. No _build_multi_label_gt or per-frame loops needed.
    """

    def __init__(
        self,
        lambda_vlm: float = 0.2,
        lambda_recon: float = 1.0,
        lambda_recon_dominance: float = 5.0,
        lambda_contrastive: float = 0.5,
        p_simulate_unseen: float = 0.25,
        label_smoothing: float = 0.2,
        physics_veto_thresh: float = 2.0,
        temperature: float = 0.07,
        bce_loss: bool = True,
        mode: str = "predcls",
        lambda_stability: float = 0.0,
    ):
        super().__init__()
        self.lambda_vlm = lambda_vlm
        self.lambda_recon = lambda_recon
        self.lambda_recon_dominance = lambda_recon_dominance
        self.lambda_contrastive = lambda_contrastive
        self.p_simulate_unseen = p_simulate_unseen
        self.physics_veto_thresh = physics_veto_thresh
        self.temperature = temperature
        self.bce_loss = bce_loss
        self.mode = mode
        self.lambda_stability = lambda_stability

        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCEWithLogitsLoss()
        self._kl_loss = nn.KLDivLoss(reduction='batchmean')

        self._label_smoother = LabelSmoother(epsilon=label_smoothing) if label_smoothing > 0 else None

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_attention: torch.Tensor,
        gt_spatial: torch.Tensor,
        gt_contacting: torch.Tensor,
        pair_valid: torch.Tensor,
        visibility_mask: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        corners: Optional[torch.Tensor] = None,
        gt_node_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute triple-objective loss on pre-padded tensors.

        Args:
            predictions: dict with (T, K_max, C) padded tensors
            gt_attention: (T, K_max) long
            gt_spatial: (T, K_max, 6) float multi-hot
            gt_contacting: (T, K_max, 17) float multi-hot
            pair_valid: (T, K_max) bool
            visibility_mask: (T, N_max) bool
            person_idx: (T, K_max) long
            object_idx: (T, K_max) long
        """
        device = predictions["attention_distribution"].device
        losses = {}

        # 1. Scene graph loss (split visible/masked)
        sg_losses = self._compute_scene_graph_loss(
            predictions, gt_attention, gt_spatial, gt_contacting,
            pair_valid, visibility_mask, person_idx, object_idx,
            valid_mask, gt_node_labels, device,
        )
        losses.update(sg_losses)

        # 2. Feature reconstruction loss
        recon_loss = self._compute_reconstruction_loss(predictions, device)
        losses["recon_loss"] = recon_loss * self.lambda_recon * self.lambda_recon_dominance

        # 3. Contrastive loss
        contra_loss = self._compute_contrastive_loss(predictions, device)
        losses["contrastive_loss"] = contra_loss * self.lambda_contrastive

        # 4. Simulated-unseen fine-tuning
        sim_loss = self._compute_simulated_unseen_loss(predictions, device)
        losses["simulated_unseen_loss"] = sim_loss

        # 5. Attractor stability loss (AMWAE++ only)
        if self.lambda_stability > 0 and "h_prev_seq" in predictions and predictions["h_prev_seq"]:
            h_final_all = torch.stack(predictions["enriched_seq"])
            h_prev_all = torch.stack(predictions["h_prev_seq"])
            L_stability = F.mse_loss(h_final_all, h_prev_all.detach())
            losses["stability_loss"] = self.lambda_stability * L_stability

        # Total
        losses["total"] = sum(v for v in losses.values() if isinstance(v, torch.Tensor) and v.requires_grad)

        return losses

    def _compute_scene_graph_loss(
        self, predictions, gt_attention, gt_spatial, gt_contacting,
        pair_valid, visibility_mask, person_idx, object_idx,
        valid_mask, gt_node_labels, device,
    ) -> Dict[str, torch.Tensor]:
        """Split visible/masked SG loss on pre-padded tensors."""
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        valid = pair_valid.bool()
        if not valid.any():
            return {
                const.ATTENTION_RELATION_LOSS: zero,
                const.SPATIAL_RELATION_LOSS: zero,
                const.CONTACTING_RELATION_LOSS: zero,
            }

        # Flatten all valid pairs across T
        att_pred = predictions["attention_distribution"][valid]
        spa_pred = predictions["spatial_logits"][valid]     # raw logits
        con_pred = predictions["contacting_logits"][valid]  # raw logits

        att_gt = gt_attention[valid].to(device)
        spa_gt = gt_spatial[valid].to(device)
        con_gt = gt_contacting[valid].to(device)

        # Split by visibility
        p_vis = visibility_mask.gather(1, person_idx)[valid]
        o_vis = visibility_mask.gather(1, object_idx)[valid]
        pair_visible = p_vis & o_vis
        pair_masked = ~pair_visible

        losses = {}

        # Visible losses (full weight, clean GT)
        if pair_visible.any():
            losses["vis_att"] = self._ce_loss(att_pred[pair_visible], att_gt[pair_visible])
            losses["vis_spa"] = self._bce_loss(spa_pred[pair_visible], spa_gt[pair_visible])
            losses["vis_con"] = self._bce_loss(con_pred[pair_visible], con_gt[pair_visible])
        else:
            losses["vis_att"] = zero
            losses["vis_spa"] = zero
            losses["vis_con"] = zero

        # Masked losses (λ_vlm weighted, smoothed)
        if pair_masked.any():
            m_att_pred = att_pred[pair_masked]
            m_att_gt = att_gt[pair_masked]
            m_spa_pred = spa_pred[pair_masked]
            m_spa_gt = spa_gt[pair_masked]
            m_con_pred = con_pred[pair_masked]
            m_con_gt = con_gt[pair_masked]

            if self._label_smoother is not None:
                m_spa_gt = self._label_smoother.smooth_bce_target(m_spa_gt)
                m_con_gt = self._label_smoother.smooth_bce_target(m_con_gt)
                smoothed = self._label_smoother.smooth_ce_target(m_att_gt, 3)
                losses["masked_att"] = self._kl_loss(
                    F.log_softmax(m_att_pred, dim=-1), smoothed
                ) * self.lambda_vlm
            else:
                losses["masked_att"] = self._ce_loss(m_att_pred, m_att_gt) * self.lambda_vlm

            losses["masked_spa"] = self._bce_loss(m_spa_pred, m_spa_gt) * self.lambda_vlm
            losses["masked_con"] = self._bce_loss(m_con_pred, m_con_gt) * self.lambda_vlm
        else:
            losses["masked_att"] = zero
            losses["masked_spa"] = zero
            losses["masked_con"] = zero

        losses[const.ATTENTION_RELATION_LOSS] = losses["vis_att"] + losses["masked_att"]
        losses[const.SPATIAL_RELATION_LOSS] = losses["vis_spa"] + losses["masked_spa"]
        losses[const.CONTACTING_RELATION_LOSS] = losses["vis_con"] + losses["masked_con"]

        # Node loss
        if gt_node_labels is not None and self.mode in [const.SGCLS, const.SGDET]:
            node_pred = predictions["node_logits"]
            node_gt = gt_node_labels.to(device)
            if valid_mask is not None:
                node_pred = node_pred[valid_mask]
                node_gt = node_gt[valid_mask]
            if len(node_gt) > 0:
                losses[const.OBJECT_LOSS] = self._ce_loss(node_pred, node_gt)

        return losses

    def _compute_reconstruction_loss(self, predictions, device):
        """MSE between retrieved tokens and original DINO features for masked tokens.

        Vectorized: stacks all T frames and computes masked MSE in one shot.
        """
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        recon_pred = predictions.get("reconstruction_predictions", None)
        recon_target = predictions.get("reconstruction_targets", None)
        is_masked = predictions.get("is_masked", None)

        if recon_pred is None or recon_target is None or is_masked is None:
            return zero

        # These should already be (T, N, D) tensors from the model
        if isinstance(recon_pred, list):
            if len(recon_pred) == 0:
                return zero
            recon_pred = torch.stack(recon_pred)
            recon_target = torch.stack(recon_target)
            is_masked = torch.stack(is_masked)

        # is_masked: (T, N) bool
        if not is_masked.any():
            return zero

        pred_masked = recon_pred[is_masked]
        target_masked = recon_target[is_masked]

        return F.mse_loss(pred_masked, target_masked)

    def _compute_contrastive_loss(self, predictions, device):
        """InfoNCE on cross-attention weights for masked tokens.

        For each masked token, attention to SAME object (positive) should be higher.
        """
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        attn_weights_list = predictions.get("attn_weights", None)
        is_masked_list = predictions.get("is_masked", None)
        mem_ids_list = predictions.get("memory_object_ids", None)

        if attn_weights_list is None or is_masked_list is None or mem_ids_list is None:
            return zero

        # These may be lists if model hasn't been updated yet
        if not isinstance(attn_weights_list, list):
            return zero

        T = len(attn_weights_list)
        if T == 0:
            return zero

        losses = []
        for t in range(T):
            is_masked = is_masked_list[t]
            attn_weights = attn_weights_list[t]
            mem_obj_ids = mem_ids_list[t]

            if not is_masked.any() or attn_weights is None:
                continue

            masked_indices = torch.where(is_masked)[0]

            for i in masked_indices:
                obj_id = i
                pos_mask = (mem_obj_ids == obj_id)
                if not pos_mask.any():
                    continue

                attn_i = attn_weights[i]
                pos_score = attn_i[pos_mask].sum()
                total_score = attn_i.sum()

                if total_score > 0:
                    ratio = (pos_score / (total_score + 1e-8)).clamp(min=1e-8)
                    losses.append(-torch.log(ratio))

        if not losses:
            return zero

        return torch.stack(losses).mean()

    def _compute_simulated_unseen_loss(self, predictions, device):
        """Loss for artificially masked visible objects — uses clean GT."""
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        sim_preds = predictions.get("simulated_unseen_predictions", None)
        sim_gt = predictions.get("simulated_unseen_gt", None)

        if sim_preds is None or sim_gt is None:
            return zero

        losses = []
        for t in range(len(sim_preds)):
            if sim_preds[t] is None or sim_gt[t] is None:
                continue

            pred = sim_preds[t]
            gt = sim_gt[t]

            if "attention" in pred and "attention" in gt:
                att_loss = self._ce_loss(pred["attention"], gt["attention"].to(device))
                losses.append(att_loss)
            if "spatial" in pred and "spatial" in gt:
                # Simulated GT should already be multi-hot tensors from the model
                spa_gt = gt["spatial"].to(device) if isinstance(gt["spatial"], torch.Tensor) else gt["spatial"]
                if isinstance(spa_gt, torch.Tensor):
                    spa_loss = self._bce_loss(pred["spatial"], spa_gt)
                    losses.append(spa_loss)
            if "contacting" in pred and "contacting" in gt:
                con_gt = gt["contacting"].to(device) if isinstance(gt["contacting"], torch.Tensor) else gt["contacting"]
                if isinstance(con_gt, torch.Tensor):
                    con_loss = self._bce_loss(pred["contacting"], con_gt)
                    losses.append(con_loss)

        if not losses:
            return zero

        return torch.stack(losses).mean()
