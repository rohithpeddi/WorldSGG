"""
LKS Buffer Loss
=================

Standard CE/BCE loss with stratified Vis-Vis / Vis-Unseen / Unseen-Unseen
evaluation. Identical structure to AmnesicGNNLoss — reused for Baseline 1.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from constants import Constants as const


class LKSLoss(nn.Module):
    """
    Standard loss with three-bucket stratified evaluation.

    Same as AmnesicGNNLoss — no λ weighting, uniform loss across buckets.
    The stratified metrics reveal where passive memory helps vs. fails.

    Args:
        bce_loss: Use BCE for spatial/contacting.
        mode: Task mode.
    """

    def __init__(self, bce_loss: bool = True, mode: str = "predcls"):
        super().__init__()
        self.bce_loss = bce_loss
        self.mode = mode
        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCELoss()

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
        gt_node_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a single frame with stratified tracking."""
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

        # Classify pairs into 3 buckets
        p_vis = visibility_mask[person_idx]
        o_vis = visibility_mask[object_idx]
        vis_vis = p_vis & o_vis
        vis_unseen = p_vis ^ o_vis
        unseen_unseen = ~p_vis & ~o_vis

        for bucket_name, mask in [("vis_vis", vis_vis), ("vis_unseen", vis_unseen), ("unseen_unseen", unseen_unseen)]:
            if mask.any():
                losses[f"{bucket_name}_att"] = self._ce_loss(att_pred[mask], att_gt[mask])
                losses[f"{bucket_name}_spa"] = self._bce_loss(spa_pred[mask], spa_gt[mask])
                losses[f"{bucket_name}_con"] = self._bce_loss(con_pred[mask], con_gt[mask])
            else:
                losses[f"{bucket_name}_att"] = zero
                losses[f"{bucket_name}_spa"] = zero
                losses[f"{bucket_name}_con"] = zero

        # Total relationship loss
        losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(att_pred, att_gt)
        losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(spa_pred, spa_gt)
        losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(con_pred, con_gt)

        # Node classification
        if gt_node_labels is not None and self.mode in [const.SGCLS, const.SGDET]:
            node_pred = predictions["node_logits"]
            node_gt = gt_node_labels.to(device)
            if valid_mask is not None:
                node_pred = node_pred[valid_mask]
                node_gt = node_gt[valid_mask]
            if len(node_gt) > 0:
                losses[const.OBJECT_LOSS] = self._ce_loss(node_pred, node_gt)

        total_keys = [const.ATTENTION_RELATION_LOSS, const.SPATIAL_RELATION_LOSS, const.CONTACTING_RELATION_LOSS]
        if const.OBJECT_LOSS in losses:
            total_keys.append(const.OBJECT_LOSS)
        losses["total"] = sum(losses[k] for k in total_keys)

        return losses

    def _build_multi_label_gt(self, gt_list, num_classes, device):
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
