"""
Amnesic GNN Loss — VLM Noisy Label Training
=============================================

Standard CE/BCE loss with:
  1. λ_vlm discounting for non-vis_vis buckets
  2. Label smoothing on VLM pseudo-labels

Three-bucket stratified evaluation (Vis-Vis, Vis-Unseen, Unseen-Unseen)
remains for ablation analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from constants import Constants as const
from lib.supervised.worldsgg.worldsgg_base import LabelSmoother


class AmnesicGNNLoss(nn.Module):
    """
    Stratified loss with VLM noise handling.

    Vis-Vis: full CE/BCE on clean manual labels.
    Vis-Unseen / Unseen-Unseen: λ_vlm weighted, smoothed.
    """

    def __init__(
        self,
        lambda_vlm: float = 0.2,
        label_smoothing: float = 0.2,
        use_physics_veto: bool = True,  # DEPRECATED — kept for backward compat, ignored
        physics_veto_thresh: float = 2.0,
        bce_loss: bool = True,
        mode: str = "predcls",
    ):
        super().__init__()
        self.lambda_vlm = lambda_vlm
        self.bce_loss = bce_loss
        self.mode = mode

        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCELoss()
        self._kl_loss = nn.KLDivLoss(reduction='batchmean')

        self._label_smoother = LabelSmoother(epsilon=label_smoothing) if label_smoothing > 0 else None


    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_attention: torch.Tensor,
        gt_spatial: List,
        gt_contacting: List,
        visibility_mask: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
        valid_mask: torch.Tensor,
        corners: torch.Tensor = None,
        gt_node_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a single frame with stratified VLM noise handling."""
        device = predictions["attention_distribution"].device
        K = predictions["attention_distribution"].shape[0]
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        losses = {}

        if K == 0:
            losses["total"] = zero
            return losses

        att_pred = predictions["attention_distribution"]
        spa_pred = predictions["spatial_distribution"]
        con_pred = predictions["contacting_distribution"]

        att_gt = gt_attention.to(device)
        spa_gt = self._build_multi_label_gt(gt_spatial, 6, device)
        con_gt = self._build_multi_label_gt(gt_contacting, 17, device)

        # --- Classify pairs into 3 buckets ---
        p_vis = visibility_mask[person_idx]
        o_vis = visibility_mask[object_idx]
        vis_vis = p_vis & o_vis
        vis_unseen = p_vis ^ o_vis
        unseen_unseen = ~p_vis & ~o_vis

        # --- Vis-Vis: standard CE/BCE on clean manual labels ---
        if vis_vis.any():
            losses["vis_vis_att"] = self._ce_loss(att_pred[vis_vis], att_gt[vis_vis])
            losses["vis_vis_spa"] = self._bce_loss(spa_pred[vis_vis], spa_gt[vis_vis])
            losses["vis_vis_con"] = self._bce_loss(con_pred[vis_vis], con_gt[vis_vis])
        else:
            losses["vis_vis_att"] = zero
            losses["vis_vis_spa"] = zero
            losses["vis_vis_con"] = zero

        # --- Vis-Unseen & Unseen-Unseen: VLM noise handling ---
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
                    # CE: use KL with smoothed target
                    smoothed_att = self._label_smoother.smooth_ce_target(b_att_gt, 3)
                    losses[f"{bucket_name}_att"] = self._kl_loss(
                        F.log_softmax(b_att_pred, dim=-1), smoothed_att
                    ) * self.lambda_vlm
                else:
                    losses[f"{bucket_name}_att"] = self._ce_loss(b_att_pred, b_att_gt) * self.lambda_vlm

                losses[f"{bucket_name}_spa"] = self._bce_loss(b_spa_pred, b_spa_gt) * self.lambda_vlm
                losses[f"{bucket_name}_con"] = self._bce_loss(b_con_pred, b_con_gt) * self.lambda_vlm
            else:
                losses[f"{bucket_name}_att"] = zero
                losses[f"{bucket_name}_spa"] = zero
                losses[f"{bucket_name}_con"] = zero

        # --- Aggregate relationship losses ---
        losses[const.ATTENTION_RELATION_LOSS] = (
            losses["vis_vis_att"] + losses["vis_unseen_att"] + losses["unseen_unseen_att"]
        )
        losses[const.SPATIAL_RELATION_LOSS] = (
            losses["vis_vis_spa"] + losses["vis_unseen_spa"] + losses["unseen_unseen_spa"]
        )
        losses[const.CONTACTING_RELATION_LOSS] = (
            losses["vis_vis_con"] + losses["vis_unseen_con"] + losses["unseen_unseen_con"]
        )

        # Node classification
        if gt_node_labels is not None and self.mode in [const.SGCLS, const.SGDET]:
            node_pred = predictions["node_logits"]
            node_gt = gt_node_labels.to(device)
            if valid_mask is not None:
                node_pred = node_pred[valid_mask]
                node_gt = node_gt[valid_mask]
            if len(node_gt) > 0:
                losses[const.OBJECT_LOSS] = self._ce_loss(node_pred, node_gt)

        # Total
        total_keys = [const.ATTENTION_RELATION_LOSS, const.SPATIAL_RELATION_LOSS, const.CONTACTING_RELATION_LOSS]
        if const.OBJECT_LOSS in losses:
            total_keys.append(const.OBJECT_LOSS)
        losses["total"] = sum(losses[k] for k in total_keys)

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
