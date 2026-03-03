"""
LKS Buffer Loss — VLM Noisy Label Training (Padded Tensor API)
================================================================

Accepts pre-padded (T, K_max, C) tensors with pair_valid mask from dataset.
No per-frame loops or _build_multi_label_gt needed.

  Vis-Vis: full loss on clean manual labels
  Vis-Unseen / Unseen-Unseen: λ_vlm weighted, smoothed

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


class LKSLoss(nn.Module):

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
        Compute bucketed loss on pre-padded tensors.

        All inputs are (T, K_max, ...) or (T, N_max, ...) padded tensors.
        Uses pair_valid mask to select real pairs, then flattens across T.

        Args:
            predictions: dict with (T, K_max, C) tensors
            gt_attention: (T, K_max) long
            gt_spatial: (T, K_max, 6) float multi-hot
            gt_contacting: (T, K_max, 17) float multi-hot
            pair_valid: (T, K_max) bool
            visibility_mask: (T, N_max) bool
            person_idx: (T, K_max) long
            object_idx: (T, K_max) long
        """
        device = predictions["attention_logits"].device
        # DDP-safe zero: stays connected to the computation graph
        zero = predictions["attention_logits"].sum() * 0.0
        losses = {}

        # Flatten: select only valid pairs across all T frames
        valid = pair_valid.bool()  # (T, K_max)
        if not valid.any():
            losses["total"] = zero
            return losses

        att_pred = predictions["attention_logits"][valid]  # (K_total, 3)
        spa_pred = predictions["spatial_logits"][valid]           # (K_total, 6) raw logits
        con_pred = predictions["contacting_logits"][valid]        # (K_total, 17) raw logits

        att_gt = gt_attention[valid].to(device)       # (K_total,)
        spa_gt = gt_spatial[valid].to(device)          # (K_total, 6)
        con_gt = gt_contacting[valid].to(device)       # (K_total, 17)

        # Global denominator: every pair contributes 1/N_total to the gradient
        N = valid.sum().float().clamp(min=1)

        # Classify pairs into 3 buckets using visibility
        p_vis = visibility_mask.gather(1, person_idx)[valid]  # (K_total,)
        o_vis = visibility_mask.gather(1, object_idx)[valid]  # (K_total,)
        vis_vis = p_vis & o_vis
        vis_unseen = p_vis ^ o_vis
        unseen_unseen = ~p_vis & ~o_vis

        # Vis-Vis: full-weight, clean GT — each pair contributes 1/N
        if vis_vis.any():
            losses["vis_vis_att"] = self._ce_loss(att_pred[vis_vis], att_gt[vis_vis]) / N
            losses["vis_vis_spa"] = self._bce_loss(spa_pred[vis_vis], spa_gt[vis_vis]) / N
            losses["vis_vis_con"] = self._bce_loss(con_pred[vis_vis], con_gt[vis_vis]) / N
        else:
            losses["vis_vis_att"] = zero
            losses["vis_vis_spa"] = zero
            losses["vis_vis_con"] = zero

        # Vis-Unseen / Unseen-Unseen: VLM noise handling
        # Each unseen pair contributes λ_vlm/N — same denominator, λ_vlm is the sole down-weight
        for bucket_name, mask in [("vis_unseen", vis_unseen), ("unseen_unseen", unseen_unseen)]:
            if mask.any():
                b_att_pred = att_pred[mask]
                b_att_gt = att_gt[mask]
                b_spa_pred = spa_pred[mask]
                b_spa_gt = spa_gt[mask]
                b_con_pred = con_pred[mask]
                b_con_gt = con_gt[mask]

                # Label smoothing
                if self._label_smoother is not None:
                    b_spa_gt = self._label_smoother.smooth_bce_target(b_spa_gt)
                    b_con_gt = self._label_smoother.smooth_bce_target(b_con_gt)
                    smoothed = self._label_smoother.smooth_ce_target(b_att_gt, 3)
                    losses[f"{bucket_name}_att"] = self._kl_loss(
                        F.log_softmax(b_att_pred, dim=-1), smoothed
                    ) * self.lambda_vlm / N
                else:
                    losses[f"{bucket_name}_att"] = self._ce_loss(b_att_pred, b_att_gt) * self.lambda_vlm / N

                losses[f"{bucket_name}_spa"] = self._bce_loss(b_spa_pred, b_spa_gt) * self.lambda_vlm / N
                losses[f"{bucket_name}_con"] = self._bce_loss(b_con_pred, b_con_gt) * self.lambda_vlm / N
            else:
                losses[f"{bucket_name}_att"] = zero
                losses[f"{bucket_name}_spa"] = zero
                losses[f"{bucket_name}_con"] = zero

        # Aggregates
        losses[const.ATTENTION_RELATION_LOSS] = (
            losses["vis_vis_att"] + losses["vis_unseen_att"] + losses["unseen_unseen_att"]
        )
        losses[const.SPATIAL_RELATION_LOSS] = (
            losses["vis_vis_spa"] + losses["vis_unseen_spa"] + losses["unseen_unseen_spa"]
        )
        losses[const.CONTACTING_RELATION_LOSS] = (
            losses["vis_vis_con"] + losses["vis_unseen_con"] + losses["unseen_unseen_con"]
        )

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

        total_keys = [const.ATTENTION_RELATION_LOSS, const.SPATIAL_RELATION_LOSS, const.CONTACTING_RELATION_LOSS]
        if const.OBJECT_LOSS in losses:
            total_keys.append(const.OBJECT_LOSS)
        losses["total"] = sum(losses[k] for k in total_keys)

        return losses
