"""
GL-STGN Loss — VLM Noisy Label Training (Padded Tensor API)
=============================================================

Accepts pre-padded (T, K_max, C) tensors with pair_valid mask from dataset.
No per-frame loops or _build_multi_label_gt needed.

L_total = L_manual(Visible) + λ_vlm * L_pseudo(Unseen) + λ_smooth * L_smooth
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
        lambda_smooth: float = 0.1,
        movement_thresh: float = 0.3,
        bce_loss: bool = True,
        mode: str = "predcls",
    ):
        super().__init__()
        self.lambda_vlm = lambda_vlm
        self.lambda_smooth = lambda_smooth
        self.movement_thresh = movement_thresh
        self.bce_loss = bce_loss
        self.mode = mode

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
            corners: (T, N_max, 8, 3) optional — 3D corners for smoothness
            gt_node_labels: (T, N_max) optional — node class labels
        """
        device = predictions["attention_logits"].device
        zero = torch.tensor(0.0, device=device, requires_grad=True)
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

        # Classify pairs by visibility
        p_vis = visibility_mask.gather(1, person_idx)[valid]
        o_vis = visibility_mask.gather(1, object_idx)[valid]
        pair_visible = p_vis & o_vis
        pair_unseen = ~pair_visible

        # --- Visible pairs: manual GT (pristine) ---
        if pair_visible.any():
            vis_att_loss = self._ce_loss(att_pred[pair_visible], att_gt[pair_visible])
            vis_spa_loss = self._bce_loss(spa_pred[pair_visible], spa_gt[pair_visible])
            vis_con_loss = self._bce_loss(con_pred[pair_visible], con_gt[pair_visible])
        else:
            vis_att_loss = zero
            vis_spa_loss = zero
            vis_con_loss = zero

        # --- Unseen pairs: VLM pseudo-labels (noisy) ---
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
                ) * self.lambda_vlm
            else:
                unseen_att_loss = self._ce_loss(u_att_pred, u_att_gt) * self.lambda_vlm

            unseen_spa_loss = self._bce_loss(u_spa_pred, u_spa_gt) * self.lambda_vlm
            unseen_con_loss = self._bce_loss(u_con_pred, u_con_gt) * self.lambda_vlm
        else:
            unseen_att_loss = zero
            unseen_spa_loss = zero
            unseen_con_loss = zero

        losses[const.ATTENTION_RELATION_LOSS] = vis_att_loss + unseen_att_loss
        losses[const.SPATIAL_RELATION_LOSS] = vis_spa_loss + unseen_spa_loss
        losses[const.CONTACTING_RELATION_LOSS] = vis_con_loss + unseen_con_loss

        # Node classification
        if gt_node_labels is not None and self.mode in [const.SGCLS, const.SGDET]:
            node_pred = predictions["node_logits"]
            node_gt = gt_node_labels.to(device)
            if valid_mask is not None:
                node_pred = node_pred[valid_mask]
                node_gt = node_gt[valid_mask]
            if len(node_gt) > 0:
                losses[const.OBJECT_LOSS] = self._ce_loss(node_pred, node_gt)

        # Smoothness loss (temporal inertia)
        if self.lambda_smooth > 0:
            smooth = self._compute_smoothness_loss(predictions, pair_valid, corners)
            losses["smooth_loss"] = smooth * self.lambda_smooth

        total_keys = [const.ATTENTION_RELATION_LOSS, const.SPATIAL_RELATION_LOSS,
                      const.CONTACTING_RELATION_LOSS]
        if const.OBJECT_LOSS in losses:
            total_keys.append(const.OBJECT_LOSS)
        if "smooth_loss" in losses:
            total_keys.append("smooth_loss")
        losses["total"] = sum(losses[k] for k in total_keys)

        return losses

    def _compute_smoothness_loss(self, predictions, pair_valid, corners):
        """
        Temporal inertia: penalizes prediction changes between consecutive frames.

        Operates on padded (T, K_max, C) tensors. Computes KL between t and t-1
        only for positions valid in BOTH frames.
        """
        device = predictions["attention_logits"].device
        T = predictions["attention_logits"].shape[0]

        if T < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Get consecutive-frame valid masks: both frames must have valid pairs
        both_valid = pair_valid[1:] & pair_valid[:-1]  # (T-1, K_max)

        if not both_valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Use raw logits for numerically stable temporal consistency
        spa_logits_curr = predictions["spatial_logits"][1:]       # (T-1, K_max, 6)
        spa_prev_probs = torch.sigmoid(predictions["spatial_logits"][:-1]).detach()
        con_logits_curr = predictions["contacting_logits"][1:]   # (T-1, K_max, 17)
        con_prev_probs = torch.sigmoid(predictions["contacting_logits"][:-1]).detach()

        # Numerically stable BCE between current logits and previous probs
        spa_loss = F.binary_cross_entropy_with_logits(
            spa_logits_curr, spa_prev_probs, reduction='none',
        )  # (T-1, K_max, 6)
        con_loss = F.binary_cross_entropy_with_logits(
            con_logits_curr, con_prev_probs, reduction='none',
        )  # (T-1, K_max, 17)

        # Apply validity mask and average
        mask = both_valid.unsqueeze(-1).float()
        spa_smooth = (spa_loss * mask).sum() / mask.sum().clamp(min=1)
        con_smooth = (con_loss * mask).sum() / mask.sum().clamp(min=1)

        return spa_smooth + con_smooth
