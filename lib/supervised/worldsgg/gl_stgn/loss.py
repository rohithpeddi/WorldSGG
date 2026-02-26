"""
GL-STGN Loss
=============

Split loss formulation for the GL-STGN model:
  L_total = L_visible + λ * L_unseen

Where:
  - L_visible: losses for relationship pairs where both objects are in FOV
  - L_unseen:  losses for pairs where ≥1 object is out of FOV
  - λ (lambda_unseen): weight to force the model to learn from the memory bank

Each loss component includes:
  - Attention: CrossEntropy (3 classes, single-label)
  - Spatial: BCE (6 classes, multi-label)
  - Contacting: BCE (17 classes, multi-label)
  - Optional: Node classification CE loss (for sgcls/sgdet modes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from constants import Constants as const


class GLSTGNLoss(nn.Module):
    """
    Split loss for GL-STGN training.

    Args:
        lambda_unseen: Weight multiplier for unseen-object relationship loss.
        bce_loss: If True, use BCE for spatial/contacting. Else, use MultiLabelMarginLoss.
        mode: Task mode (predcls/sgcls/sgdet) — controls whether object loss is included.
    """

    def __init__(
        self,
        lambda_unseen: float = 2.0,
        bce_loss: bool = True,
        mode: str = "predcls",
    ):
        super().__init__()
        self.lambda_unseen = lambda_unseen
        self.bce_loss = bce_loss
        self.mode = mode

        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCELoss()
        self._mlm_loss = nn.MultiLabelMarginLoss()

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
        Compute split visible/unseen loss.

        Args:
            predictions: dict from GLSTGN.forward() with per-frame lists:
                - attention_distribution: List of (K_t, 3)
                - spatial_distribution: List of (K_t, 6)
                - contacting_distribution: List of (K_t, 17)
                - node_logits: List of (N_t, num_classes)
            gt_attention: List of (K_t,) LongTensor — GT attention class per pair.
            gt_spatial: List of List[List[int]] — GT spatial rel indices per pair.
            gt_contacting: List of List[List[int]] — GT contacting rel indices per pair.
            gt_node_labels: Optional List of (N_t,) LongTensor — GT object classes.
            visibility_mask_seq: List of (N_t,) bool — per-object visibility.
            person_idx_seq: List of (K_t,) — person indices in pairs.
            object_idx_seq: List of (K_t,) — object indices in pairs.
            valid_mask_seq: List of (N_t,) bool — valid object mask.

        Returns:
            dict of named losses ready for backpropagation.
        """
        device = predictions["attention_distribution"][0].device
        T = len(predictions["attention_distribution"])

        # Accumulators
        all_att_pred_vis, all_att_gt_vis = [], []
        all_att_pred_unseen, all_att_gt_unseen = [], []
        all_spa_pred_vis, all_spa_gt_vis = [], []
        all_spa_pred_unseen, all_spa_gt_unseen = [], []
        all_con_pred_vis, all_con_gt_vis = [], []
        all_con_pred_unseen, all_con_gt_unseen = [], []
        all_node_pred, all_node_gt = [], []

        for t in range(T):
            att_pred_t = predictions["attention_distribution"][t]  # (K_t, 3)
            spa_pred_t = predictions["spatial_distribution"][t]    # (K_t, 6)
            con_pred_t = predictions["contacting_distribution"][t]  # (K_t, 17)

            K_t = att_pred_t.shape[0]
            if K_t == 0:
                continue

            att_gt_t = gt_attention[t].to(device)  # (K_t,)

            # Build spatial GT
            spa_gt_t = self._build_multi_label_gt(gt_spatial[t], 6, device)
            con_gt_t = self._build_multi_label_gt(gt_contacting[t], 17, device)

            # Split by visibility
            if visibility_mask_seq is not None and person_idx_seq is not None:
                vis_t = visibility_mask_seq[t]
                p_idx = person_idx_seq[t]
                o_idx = object_idx_seq[t]

                # A pair is "visible" if BOTH person and object are visible
                pair_visible = vis_t[p_idx] & vis_t[o_idx]  # (K_t,)
                pair_unseen = ~pair_visible

                if pair_visible.any():
                    all_att_pred_vis.append(att_pred_t[pair_visible])
                    all_att_gt_vis.append(att_gt_t[pair_visible])
                    all_spa_pred_vis.append(spa_pred_t[pair_visible])
                    all_spa_gt_vis.append(spa_gt_t[pair_visible])
                    all_con_pred_vis.append(con_pred_t[pair_visible])
                    all_con_gt_vis.append(con_gt_t[pair_visible])

                if pair_unseen.any():
                    all_att_pred_unseen.append(att_pred_t[pair_unseen])
                    all_att_gt_unseen.append(att_gt_t[pair_unseen])
                    all_spa_pred_unseen.append(spa_pred_t[pair_unseen])
                    all_spa_gt_unseen.append(spa_gt_t[pair_unseen])
                    all_con_pred_unseen.append(con_pred_t[pair_unseen])
                    all_con_gt_unseen.append(con_gt_t[pair_unseen])
            else:
                # No visibility info → treat all as visible
                all_att_pred_vis.append(att_pred_t)
                all_att_gt_vis.append(att_gt_t)
                all_spa_pred_vis.append(spa_pred_t)
                all_spa_gt_vis.append(spa_gt_t)
                all_con_pred_vis.append(con_pred_t)
                all_con_gt_vis.append(con_gt_t)

            # Node classification
            if gt_node_labels is not None and self.mode in [const.SGCLS, const.SGDET]:
                node_pred_t = predictions["node_logits"][t]
                node_gt_t = gt_node_labels[t].to(device)
                if valid_mask_seq is not None:
                    valid_t = valid_mask_seq[t]
                    node_pred_t = node_pred_t[valid_t]
                    node_gt_t = node_gt_t[valid_t]
                if len(node_gt_t) > 0:
                    all_node_pred.append(node_pred_t)
                    all_node_gt.append(node_gt_t)

        # --- Compute losses ---
        losses = {}

        # Visible losses
        loss_vis = self._compute_relationship_losses(
            all_att_pred_vis, all_att_gt_vis,
            all_spa_pred_vis, all_spa_gt_vis,
            all_con_pred_vis, all_con_gt_vis,
            device,
        )
        for k, v in loss_vis.items():
            losses[f"vis_{k}"] = v

        # Unseen losses
        loss_unseen = self._compute_relationship_losses(
            all_att_pred_unseen, all_att_gt_unseen,
            all_spa_pred_unseen, all_spa_gt_unseen,
            all_con_pred_unseen, all_con_gt_unseen,
            device,
        )
        for k, v in loss_unseen.items():
            losses[f"unseen_{k}"] = v * self.lambda_unseen

        # Node classification loss
        if all_node_pred:
            node_pred_cat = torch.cat(all_node_pred, dim=0)
            node_gt_cat = torch.cat(all_node_gt, dim=0)
            losses[const.OBJECT_LOSS] = self._ce_loss(node_pred_cat, node_gt_cat)

        # Total loss (for convenience)
        losses["total"] = sum(losses.values())

        return losses

    def _compute_relationship_losses(
        self,
        att_preds, att_gts,
        spa_preds, spa_gts,
        con_preds, con_gts,
        device,
    ) -> Dict[str, torch.Tensor]:
        """Compute attention/spatial/contacting losses from accumulated tensors."""
        losses = {}
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        # Attention (CE loss)
        if att_preds:
            att_p = torch.cat(att_preds, dim=0)
            att_g = torch.cat(att_gts, dim=0)
            if len(att_g) > 0:
                losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(att_p, att_g)
            else:
                losses[const.ATTENTION_RELATION_LOSS] = zero
        else:
            losses[const.ATTENTION_RELATION_LOSS] = zero

        # Spatial (BCE loss)
        if spa_preds:
            spa_p = torch.cat(spa_preds, dim=0)
            spa_g = torch.cat(spa_gts, dim=0)
            if len(spa_g) > 0:
                losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(spa_p, spa_g)
            else:
                losses[const.SPATIAL_RELATION_LOSS] = zero
        else:
            losses[const.SPATIAL_RELATION_LOSS] = zero

        # Contacting (BCE loss)
        if con_preds:
            con_p = torch.cat(con_preds, dim=0)
            con_g = torch.cat(con_gts, dim=0)
            if len(con_g) > 0:
                losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(con_p, con_g)
            else:
                losses[const.CONTACTING_RELATION_LOSS] = zero
        else:
            losses[const.CONTACTING_RELATION_LOSS] = zero

        return losses

    def _build_multi_label_gt(
        self,
        gt_list: List,
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert list of GT index lists to multi-hot BCE targets.

        Args:
            gt_list: List of K items, each a list of active class indices.
            num_classes: Total number of classes.
            device: Target device.

        Returns:
            (K, num_classes) float tensor with 1s at active positions.
        """
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
