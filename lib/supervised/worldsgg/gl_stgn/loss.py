"""
GL-STGN Loss — VLM Noisy Label Training
=========================================

L_total = L_manual(Visible) + λ_vlm * L_pseudo(Unseen) + λ_smooth * L_smooth

Changes from baseline loss:
  1. λ_vlm discounting (0.2 vs old 2.0) — VLM labels are gentle regularizers
  2. Label smoothing on unseen GT — tells model "VLM might be wrong"
  3. Physics veto — zero out loss for geometrically impossible VLM labels
  4. Memory shielding — .detach() handled in model forward, not here
  5. Temporal inertia — penalizes jittery unseen predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from constants import Constants as const
from lib.supervised.worldsgg.worldsgg_base import PhysicsVeto, LabelSmoother


class GLSTGNLoss(nn.Module):
    """
    Split loss for GL-STGN training with VLM noisy label handling.

    Args:
        lambda_vlm: Weight for unseen (VLM) edge loss (should be << 1.0).
        label_smoothing: Epsilon for VLM label smoothing.
        use_physics_veto: Enable geometric masking.
        physics_veto_thresh: Distance threshold for physics veto.
        lambda_smooth: Weight for temporal inertia regularization.
        movement_thresh: Skip smoothing if object moved > this (meters).
        bce_loss: Use BCE (True) or MultiLabelMarginLoss (False).
        mode: Task mode (predcls/sgcls/sgdet).
    """

    def __init__(
        self,
        lambda_vlm: float = 0.2,
        label_smoothing: float = 0.2,
        use_physics_veto: bool = True,
        physics_veto_thresh: float = 2.0,
        lambda_smooth: float = 0.1,
        movement_thresh: float = 0.3,
        bce_loss: bool = True,
        mode: str = "predcls",
    ):
        super().__init__()
        self.lambda_vlm = lambda_vlm
        self.label_smoothing = label_smoothing
        self.lambda_smooth = lambda_smooth
        self.movement_thresh = movement_thresh
        self.bce_loss = bce_loss
        self.mode = mode

        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCELoss()
        self._kl_loss = nn.KLDivLoss(reduction='batchmean')

        # VLM noisy label utilities
        self._label_smoother = LabelSmoother(epsilon=label_smoothing) if label_smoothing > 0 else None
        self._physics_veto = PhysicsVeto(dist_thresh=physics_veto_thresh) if use_physics_veto else None

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
        corners_seq: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute split visible/unseen loss with VLM noise handling.

        New vs baseline: corners_seq for physics veto, smoothed unseen targets,
        λ_vlm weighting, and temporal inertia.
        """
        device = predictions["attention_distribution"][0].device
        T = len(predictions["attention_distribution"])

        # Accumulators for visible (manual GT) and unseen (VLM pseudo) pairs
        vis_att_p, vis_att_g = [], []
        vis_spa_p, vis_spa_g = [], []
        vis_con_p, vis_con_g = [], []
        unseen_att_p, unseen_att_g = [], []
        unseen_spa_p, unseen_spa_g = [], []
        unseen_con_p, unseen_con_g = [], []
        all_node_pred, all_node_gt = [], []

        for t in range(T):
            att_pred_t = predictions["attention_distribution"][t]  # (K_t, 3)
            spa_pred_t = predictions["spatial_distribution"][t]    # (K_t, 6)
            con_pred_t = predictions["contacting_distribution"][t]  # (K_t, 17)

            K_t = att_pred_t.shape[0]
            if K_t == 0:
                continue

            att_gt_t = gt_attention[t].to(device)
            spa_gt_t = self._build_multi_label_gt(gt_spatial[t], 6, device)
            con_gt_t = self._build_multi_label_gt(gt_contacting[t], 17, device)

            # Split by visibility
            if visibility_mask_seq is not None and person_idx_seq is not None:
                vis_t = visibility_mask_seq[t]
                p_idx = person_idx_seq[t]
                o_idx = object_idx_seq[t]
                pair_visible = vis_t[p_idx] & vis_t[o_idx]
                pair_unseen = ~pair_visible

                # --- Visible pairs: manual GT (pristine) ---
                if pair_visible.any():
                    vis_att_p.append(att_pred_t[pair_visible])
                    vis_att_g.append(att_gt_t[pair_visible])
                    vis_spa_p.append(spa_pred_t[pair_visible])
                    vis_spa_g.append(spa_gt_t[pair_visible])
                    vis_con_p.append(con_pred_t[pair_visible])
                    vis_con_g.append(con_gt_t[pair_visible])

                # --- Unseen pairs: VLM pseudo-labels (noisy) ---
                if pair_unseen.any():
                    unseen_att_pred = att_pred_t[pair_unseen]
                    unseen_att_gt = att_gt_t[pair_unseen]
                    unseen_spa_pred = spa_pred_t[pair_unseen]
                    unseen_spa_gt = spa_gt_t[pair_unseen]
                    unseen_con_pred = con_pred_t[pair_unseen]
                    unseen_con_gt = con_gt_t[pair_unseen]

                    # Physics veto: zero out geometrically impossible edges
                    if self._physics_veto is not None and corners_seq is not None:
                        # Get unseen pair indices relative to full node set
                        unseen_pidx = p_idx[pair_unseen]
                        unseen_oidx = o_idx[pair_unseen]
                        keep = self._physics_veto.compute_veto_mask(
                            corners_seq[t], unseen_pidx, unseen_oidx, unseen_con_pred
                        )
                        if keep.any():
                            unseen_att_pred = unseen_att_pred[keep]
                            unseen_att_gt = unseen_att_gt[keep]
                            unseen_spa_pred = unseen_spa_pred[keep]
                            unseen_spa_gt = unseen_spa_gt[keep]
                            unseen_con_pred = unseen_con_pred[keep]
                            unseen_con_gt = unseen_con_gt[keep]
                        else:
                            # All vetoed → skip this frame's unseen
                            continue

                    # Label smoothing for VLM targets
                    if self._label_smoother is not None:
                        unseen_spa_gt = self._label_smoother.smooth_bce_target(unseen_spa_gt)
                        unseen_con_gt = self._label_smoother.smooth_bce_target(unseen_con_gt)

                    if len(unseen_att_pred) > 0:
                        unseen_att_p.append(unseen_att_pred)
                        unseen_att_g.append(unseen_att_gt)
                        unseen_spa_p.append(unseen_spa_pred)
                        unseen_spa_g.append(unseen_spa_gt)
                        unseen_con_p.append(unseen_con_pred)
                        unseen_con_g.append(unseen_con_gt)
            else:
                vis_att_p.append(att_pred_t)
                vis_att_g.append(att_gt_t)
                vis_spa_p.append(spa_pred_t)
                vis_spa_g.append(spa_gt_t)
                vis_con_p.append(con_pred_t)
                vis_con_g.append(con_gt_t)

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

        # Visible losses (standard CE/BCE on clean manual labels)
        loss_vis = self._compute_relationship_losses(
            vis_att_p, vis_att_g, vis_spa_p, vis_spa_g, vis_con_p, vis_con_g, device,
        )
        losses[const.ATTENTION_RELATION_LOSS] = loss_vis.get("att", torch.tensor(0.0, device=device))
        losses[const.SPATIAL_RELATION_LOSS] = loss_vis.get("spa", torch.tensor(0.0, device=device))
        losses[const.CONTACTING_RELATION_LOSS] = loss_vis.get("con", torch.tensor(0.0, device=device))

        # Unseen losses (λ_vlm weighted, smoothed, vetoed)
        loss_unseen = self._compute_relationship_losses(
            unseen_att_p, unseen_att_g, unseen_spa_p, unseen_spa_g,
            unseen_con_p, unseen_con_g, device, smoothed_bce=True,
        )
        losses["unseen_att"] = loss_unseen.get("att", torch.tensor(0.0, device=device)) * self.lambda_vlm
        losses["unseen_spa"] = loss_unseen.get("spa", torch.tensor(0.0, device=device)) * self.lambda_vlm
        losses["unseen_con"] = loss_unseen.get("con", torch.tensor(0.0, device=device)) * self.lambda_vlm

        # Node loss
        if all_node_pred:
            losses[const.OBJECT_LOSS] = self._ce_loss(
                torch.cat(all_node_pred), torch.cat(all_node_gt)
            )

        # Temporal inertia loss (anti-jitter for unseen predictions)
        if self.lambda_smooth > 0:
            losses["smooth"] = self._compute_smoothness_loss(predictions, visibility_mask_seq, corners_seq) * self.lambda_smooth

        # Total
        losses["total"] = sum(v for v in losses.values() if isinstance(v, torch.Tensor) and v.requires_grad)

        return losses

    def _compute_relationship_losses(self, att_p, att_g, spa_p, spa_g, con_p, con_g, device, smoothed_bce=False):
        """Compute attention/spatial/contacting losses from accumulated tensors."""
        losses = {}
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        if att_p:
            all_att_pred = torch.cat(att_p)
            all_att_gt = torch.cat(att_g)
            if self._label_smoother is not None and smoothed_bce:
                # For CE: use KL divergence with smoothed targets
                smoothed_att = self._label_smoother.smooth_ce_target(all_att_gt, 3)
                losses["att"] = self._kl_loss(F.log_softmax(all_att_pred, dim=-1), smoothed_att)
            else:
                losses["att"] = self._ce_loss(all_att_pred, all_att_gt)
        else:
            losses["att"] = zero

        if spa_p:
            all_spa_pred = torch.cat(spa_p)
            all_spa_gt = torch.cat(spa_g)
            # BCE targets may already be smoothed for unseen
            losses["spa"] = self._bce_loss(all_spa_pred, all_spa_gt)
        else:
            losses["spa"] = zero

        if con_p:
            all_con_pred = torch.cat(con_p)
            all_con_gt = torch.cat(con_g)
            losses["con"] = self._bce_loss(all_con_pred, all_con_gt)
        else:
            losses["con"] = zero

        return losses

    def _compute_smoothness_loss(self, predictions, visibility_mask_seq, corners_seq):
        """
        Temporal inertia: penalize distribution changes for unseen objects between frames.

        KL(P_t || P_{t-1}) for unseen objects, masked by physical movement.
        """
        device = predictions["attention_distribution"][0].device
        T = len(predictions["attention_distribution"])
        smooth_loss = torch.tensor(0.0, device=device)
        count = 0

        for t in range(1, T):
            att_curr = predictions["attention_distribution"][t]
            att_prev = predictions["attention_distribution"][t - 1]

            if att_curr.shape[0] == 0 or att_prev.shape[0] == 0:
                continue

            # Only penalize if same number of predictions (same object set)
            if att_curr.shape != att_prev.shape:
                continue

            # Movement masking: skip if wireframe proves physical movement
            if corners_seq is not None and t > 0:
                centroids_curr = corners_seq[t].mean(dim=1)
                centroids_prev = corners_seq[t - 1].mean(dim=1)
                if centroids_curr.shape == centroids_prev.shape:
                    movement = torch.norm(centroids_curr - centroids_prev, dim=-1)
                    # Average movement — skip entire frame if lots of motion
                    if movement.mean() > self.movement_thresh:
                        continue

            # KL divergence between consecutive distributions
            p_curr = F.log_softmax(att_curr, dim=-1)
            p_prev = F.softmax(att_prev.detach(), dim=-1)
            smooth_loss = smooth_loss + F.kl_div(p_curr, p_prev, reduction='batchmean')
            count += 1

        return smooth_loss / max(count, 1)

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
