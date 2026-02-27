"""
Amnesic GNN Loss with Stratified Evaluation
=============================================

Standard CE/BCE loss for scene graph prediction, with explicit tracking
of three edge categories to quantify Baseline 2's limitations:

  1. Vis-Vis:       Both objects in camera FOV
  2. Vis-Unseen:    One visible, one outside FOV
  3. Unseen-Unseen: Both objects outside camera FOV

All three contribute equally to the total loss (no λ weighting).
The stratified metrics are the scientific output proving the need for
temporal memory in Methods 1 & 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from constants import Constants as const


class AmnesicGNNLoss(nn.Module):
    """
    Standard loss with three-bucket stratified evaluation.

    Args:
        bce_loss: Use BCE for spatial/contacting (True) or MultiLabelMargin (False).
        mode: Task mode (predcls/sgcls/sgdet).
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
        """
        Compute loss for a single frame with stratified tracking.

        Args:
            predictions: dict from AmnesicGNN.forward():
                node_logits: (N, num_classes)
                attention_distribution: (K, 3)
                spatial_distribution: (K, 6)
                contacting_distribution: (K, 17)
            gt_attention: (K,) long — attention class per pair.
            gt_spatial: List[List[int]] — spatial rel indices per pair.
            gt_contacting: List[List[int]] — contacting rel indices per pair.
            visibility_mask: (N,) bool — per-object visibility.
            person_idx: (K,) — person indices.
            object_idx: (K,) — object indices.
            valid_mask: (N,) bool — valid object mask.
            gt_node_labels: Optional (N,) long — GT object classes.

        Returns:
            dict of named losses + stratified breakdowns.
        """
        device = predictions["attention_distribution"].device
        K = predictions["attention_distribution"].shape[0]
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        losses = {}

        if K == 0:
            losses["total"] = zero
            return losses

        att_pred = predictions["attention_distribution"]  # (K, 3)
        spa_pred = predictions["spatial_distribution"]    # (K, 6)
        con_pred = predictions["contacting_distribution"]  # (K, 17)

        att_gt = gt_attention.to(device)
        spa_gt = self._build_multi_label_gt(gt_spatial, 6, device)
        con_gt = self._build_multi_label_gt(gt_contacting, 17, device)

        # --- Classify pairs into 3 buckets ---
        p_vis = visibility_mask[person_idx]   # (K,) — is person visible?
        o_vis = visibility_mask[object_idx]   # (K,) — is object visible?

        vis_vis = p_vis & o_vis               # Both visible
        vis_unseen = p_vis ^ o_vis            # Exactly one visible
        unseen_unseen = ~p_vis & ~o_vis       # Both unseen

        # --- Compute per-bucket losses ---
        for bucket_name, mask in [("vis_vis", vis_vis), ("vis_unseen", vis_unseen), ("unseen_unseen", unseen_unseen)]:
            if mask.any():
                losses[f"{bucket_name}_att"] = self._ce_loss(att_pred[mask], att_gt[mask])
                losses[f"{bucket_name}_spa"] = self._bce_loss(spa_pred[mask], spa_gt[mask])
                losses[f"{bucket_name}_con"] = self._bce_loss(con_pred[mask], con_gt[mask])
            else:
                losses[f"{bucket_name}_att"] = zero
                losses[f"{bucket_name}_spa"] = zero
                losses[f"{bucket_name}_con"] = zero

        # --- Total relationship loss (uniform — no λ weighting) ---
        losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(att_pred, att_gt) if K > 0 else zero
        losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(spa_pred, spa_gt) if K > 0 else zero
        losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(con_pred, con_gt) if K > 0 else zero

        # --- Node classification ---
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
