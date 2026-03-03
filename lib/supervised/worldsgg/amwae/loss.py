"""
AMWAE Triple-Objective Loss — VLM Noisy Label Training (Padded Tensor API)
===========================================================================

Accepts pre-padded (T, K_max, C) tensors with pair_valid mask from dataset.
No per-frame loops or _build_multi_label_gt needed.

L_total = L_vis + λ_vlm * L_masked + λ_recon * λ_recon_dominance * L_recon
        + λ_contra * L_contrastive + L_simulated_unseen

All edge losses use reduction='sum' with a shared global denominator
(N_total = total valid pairs across all buckets) so that each pair
contributes equally to the gradient regardless of bucket size.
λ_vlm is the sole down-weight for noisy unseen labels.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

logger = logging.getLogger(__name__)

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
        p_simulate_unseen: float = 0.25,
        label_smoothing: float = 0.2,
        physics_veto_thresh: float = 2.0,
        bce_loss: bool = True,
        mode: str = "predcls",
        lambda_stability: float = 0.0,
    ):
        super().__init__()
        self.lambda_vlm = lambda_vlm
        self.lambda_recon = lambda_recon
        self.lambda_recon_dominance = lambda_recon_dominance
        self.p_simulate_unseen = p_simulate_unseen
        self.physics_veto_thresh = physics_veto_thresh
        self.bce_loss = bce_loss
        self.mode = mode
        self.lambda_stability = lambda_stability

        # reduction='sum' — we normalize by N_total manually for uniform per-pair gradients
        self._ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self._bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self._kl_loss = nn.KLDivLoss(reduction='sum')

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
        device = predictions["attention_logits"].device
        losses = {}

        # DDP-safe zero: stays connected to the computation graph
        zero = predictions["attention_logits"].sum() * 0.0

        # 1. Scene graph loss (split visible/masked)
        sg_losses = self._compute_scene_graph_loss(
            predictions, gt_attention, gt_spatial, gt_contacting,
            pair_valid, visibility_mask, person_idx, object_idx,
            valid_mask, gt_node_labels, device, zero,
        )
        losses.update(sg_losses)

        # 2. Feature reconstruction loss
        recon_loss = self._compute_reconstruction_loss(predictions, device, zero)
        losses["recon_loss"] = recon_loss * self.lambda_recon * self.lambda_recon_dominance

        # 3. Simulated-unseen fine-tuning
        sim_loss = self._compute_simulated_unseen_loss(predictions, device, zero)
        losses["simulated_unseen_loss"] = sim_loss

        # 5. Attractor stability loss (AMWAE++ only)
        if self.lambda_stability > 0 and "h_prev" in predictions:
            h_final_all = predictions["enriched"]       # (T, N, d_model)
            h_prev_all = predictions["h_prev"]           # (T, N, d_model), already detached
            L_stability = F.mse_loss(h_final_all, h_prev_all.detach())
            losses["stability_loss"] = self.lambda_stability * L_stability

        # Total
        losses["total"] = sum(v for v in losses.values() if isinstance(v, torch.Tensor) and v.requires_grad)

        return losses

    def _compute_scene_graph_loss(
        self, predictions, gt_attention, gt_spatial, gt_contacting,
        pair_valid, visibility_mask, person_idx, object_idx,
        valid_mask, gt_node_labels, device, zero,
    ) -> Dict[str, torch.Tensor]:
        """Split visible/masked SG loss on pre-padded tensors."""

        valid = pair_valid.bool()
        if not valid.any():
            return {
                const.ATTENTION_RELATION_LOSS: zero,
                const.SPATIAL_RELATION_LOSS: zero,
                const.CONTACTING_RELATION_LOSS: zero,
            }

        # Flatten all valid pairs across T
        att_pred = predictions["attention_logits"][valid]
        spa_pred = predictions["spatial_logits"][valid]     # raw logits
        con_pred = predictions["contacting_logits"][valid]  # raw logits

        att_gt = gt_attention[valid].to(device)
        spa_gt = gt_spatial[valid].to(device)
        con_gt = gt_contacting[valid].to(device)

        # Global denominator: every pair contributes 1/N_total to the gradient
        N = valid.sum().float().clamp(min=1)

        # Split by visibility
        p_vis = visibility_mask.gather(1, person_idx)[valid]
        o_vis = visibility_mask.gather(1, object_idx)[valid]
        pair_visible = p_vis & o_vis
        pair_masked = ~pair_visible

        losses = {}

        # Visible losses (full weight, clean GT) — each pair contributes 1/N
        if pair_visible.any():
            losses["vis_att"] = self._ce_loss(att_pred[pair_visible], att_gt[pair_visible]) / N
            losses["vis_spa"] = self._bce_loss(spa_pred[pair_visible], spa_gt[pair_visible]) / N
            losses["vis_con"] = self._bce_loss(con_pred[pair_visible], con_gt[pair_visible]) / N
        else:
            losses["vis_att"] = zero
            losses["vis_spa"] = zero
            losses["vis_con"] = zero

        # Masked losses (λ_vlm weighted, smoothed) — each pair contributes λ_vlm/N
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
                ) * self.lambda_vlm / N
            else:
                losses["masked_att"] = self._ce_loss(m_att_pred, m_att_gt) * self.lambda_vlm / N

            losses["masked_spa"] = self._bce_loss(m_spa_pred, m_spa_gt) * self.lambda_vlm / N
            losses["masked_con"] = self._bce_loss(m_con_pred, m_con_gt) * self.lambda_vlm / N
        else:
            losses["masked_att"] = zero
            losses["masked_spa"] = zero
            losses["masked_con"] = zero

        losses[const.ATTENTION_RELATION_LOSS] = losses["vis_att"] + losses["masked_att"]
        losses[const.SPATIAL_RELATION_LOSS] = losses["vis_spa"] + losses["masked_spa"]
        losses[const.CONTACTING_RELATION_LOSS] = losses["vis_con"] + losses["masked_con"]

        # Node loss (separate normalization — not affected by pair buckets)
        if gt_node_labels is not None and self.mode in [const.SGCLS, const.SGDET]:
            node_pred = predictions["node_logits"]
            node_gt = gt_node_labels.to(device)
            if valid_mask is not None:
                node_pred = node_pred[valid_mask]
                node_gt = node_gt[valid_mask]
            else:
                # Flatten (T, N, C) → (T*N, C) so CE gets classes on dim=1
                node_pred = node_pred.reshape(-1, node_pred.shape[-1])
                node_gt = node_gt.reshape(-1)
            if len(node_gt) > 0:
                N_nodes = max((node_gt >= 0).sum().item(), 1)  # count real objects, not padding (-100)
                losses[const.OBJECT_LOSS] = self._ce_loss(node_pred, node_gt) / N_nodes

        return losses

    def _compute_reconstruction_loss(self, predictions, device, zero):
        """MSE between retrieved tokens and original DINO features for masked tokens.

        Vectorized: stacks all T frames and computes masked MSE in one shot.
        """

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



    def _compute_simulated_unseen_loss(self, predictions, device, zero):
        """Loss for artificially masked visible objects — uses clean GT.

        Each per-frame loss is normalized by its own pair count since these
        are independent simulation batches (not part of the main pair buckets).
        """

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
                att_target = gt["attention"].to(device)
                n_pairs = max(att_target.shape[0], 1)
                att_loss = self._ce_loss(pred["attention"], att_target) / n_pairs
                losses.append(att_loss)
            if "spatial" in pred and "spatial" in gt:
                spa_gt = gt["spatial"].to(device) if isinstance(gt["spatial"], torch.Tensor) else gt["spatial"]
                if isinstance(spa_gt, torch.Tensor):
                    n_pairs = max(spa_gt.shape[0], 1)
                    spa_loss = self._bce_loss(pred["spatial"], spa_gt) / n_pairs
                    losses.append(spa_loss)
            if "contacting" in pred and "contacting" in gt:
                con_gt = gt["contacting"].to(device) if isinstance(gt["contacting"], torch.Tensor) else gt["contacting"]
                if isinstance(con_gt, torch.Tensor):
                    n_pairs = max(con_gt.shape[0], 1)
                    con_loss = self._bce_loss(pred["contacting"], con_gt) / n_pairs
                    losses.append(con_loss)

        if not losses:
            return zero

        return torch.stack(losses).mean()
