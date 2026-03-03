"""
GL-STGN Loss — VLM Noisy Label Training (Padded Tensor API)
=============================================================

Accepts pre-padded (T, K_max, C) tensors with pair_valid mask from dataset.
No per-frame loops or _build_multi_label_gt needed.

L_total = L_manual(Visible) + λ_vlm * L_pseudo(Unseen) + λ_smooth * L_smooth

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


class GLSTGNLoss(nn.Module):
    """Split loss for GL-STGN training with VLM noisy label handling.

    All inputs are pre-padded (T, K_max, ...) tensors with pair_valid masks.
    """

    def __init__(
        self,
        lambda_vlm: float = 0.2,
        label_smoothing: float = 0.2,
        bce_loss: bool = True,
        mode: str = "predcls",
    ):
        super().__init__()
        self.lambda_vlm = lambda_vlm
        self.bce_loss = bce_loss
        self.mode = mode

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
        gt_node_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute split visible/unseen loss on pre-padded tensors.

        Args:
            predictions: dict with (T, K_max, C) padded prediction tensors
            gt_attention: (T, K_max) long — pre-encoded attention GT
            gt_spatial: (T, K_max, 6) float — pre-computed multi-hot spatial GT
            gt_contacting: (T, K_max, 17) float — pre-computed multi-hot contacting GT
            pair_valid: (T, K_max) bool — validity mask
            visibility_mask: (T, N_max) bool — per-object visibility
            person_idx: (T, K_max) long — person slot indices
            object_idx: (T, K_max) long — object slot indices
            gt_node_labels: (T, N_max) optional — node class labels
        """
        device = predictions["attention_logits"].device
        # DDP-safe zero: stays connected to the computation graph
        zero = predictions["attention_logits"].sum() * 0.0
        losses = {}

        valid = pair_valid.bool()
        if not valid.any():
            losses["total"] = zero
            return losses

        # Flatten all valid pairs across T
        att_pred = predictions["attention_logits"][valid]  # (K_total, 3)
        spa_pred = predictions["spatial_logits"][valid]           # (K_total, 6) raw logits
        con_pred = predictions["contacting_logits"][valid]        # (K_total, 17) raw logits

        att_gt = gt_attention[valid].to(device)
        spa_gt = gt_spatial[valid].to(device)
        con_gt = gt_contacting[valid].to(device)

        # Global denominator: every pair contributes 1/N_total to the gradient
        N = valid.sum().float().clamp(min=1)

        # Classify pairs by visibility
        p_vis = visibility_mask.gather(1, person_idx)[valid]
        o_vis = visibility_mask.gather(1, object_idx)[valid]
        pair_visible = p_vis & o_vis
        pair_unseen = ~pair_visible

        # --- Visible pairs: manual GT (pristine) — each pair contributes 1/N ---
        if pair_visible.any():
            vis_att_loss = self._ce_loss(att_pred[pair_visible], att_gt[pair_visible]) / N
            vis_spa_loss = self._bce_loss(spa_pred[pair_visible], spa_gt[pair_visible]) / N
            vis_con_loss = self._bce_loss(con_pred[pair_visible], con_gt[pair_visible]) / N
        else:
            vis_att_loss = zero
            vis_spa_loss = zero
            vis_con_loss = zero

        # --- Unseen pairs: VLM pseudo-labels (noisy) — each pair contributes λ_vlm/N ---
        if pair_unseen.any():
            u_att_pred = att_pred[pair_unseen]
            u_att_gt = att_gt[pair_unseen]
            u_spa_pred = spa_pred[pair_unseen]
            u_spa_gt = spa_gt[pair_unseen]
            u_con_pred = con_pred[pair_unseen]
            u_con_gt = con_gt[pair_unseen]

            if self._label_smoother is not None:
                u_spa_gt = self._label_smoother.smooth_bce_target(u_spa_gt)
                u_con_gt = self._label_smoother.smooth_bce_target(u_con_gt)
                smoothed = self._label_smoother.smooth_ce_target(u_att_gt, 3)
                unseen_att_loss = self._kl_loss(
                    F.log_softmax(u_att_pred, dim=-1), smoothed
                ) * self.lambda_vlm / N
            else:
                unseen_att_loss = self._ce_loss(u_att_pred, u_att_gt) * self.lambda_vlm / N

            unseen_spa_loss = self._bce_loss(u_spa_pred, u_spa_gt) * self.lambda_vlm / N
            unseen_con_loss = self._bce_loss(u_con_pred, u_con_gt) * self.lambda_vlm / N
        else:
            unseen_att_loss = zero
            unseen_spa_loss = zero
            unseen_con_loss = zero

        losses[const.ATTENTION_RELATION_LOSS] = vis_att_loss + unseen_att_loss
        losses[const.SPATIAL_RELATION_LOSS] = vis_spa_loss + unseen_spa_loss
        losses[const.CONTACTING_RELATION_LOSS] = vis_con_loss + unseen_con_loss

        # Node classification (separate normalization — not affected by pair buckets)
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

        total_keys = [const.ATTENTION_RELATION_LOSS, const.SPATIAL_RELATION_LOSS,
                      const.CONTACTING_RELATION_LOSS]
        if const.OBJECT_LOSS in losses:
            total_keys.append(const.OBJECT_LOSS)
        losses["total"] = sum(losses[k] for k in total_keys)

        return losses
