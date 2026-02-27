"""
AMWAE Triple-Objective Loss — VLM Noisy Label Training
=======================================================

L_total = L_vis + λ_vlm * L_masked + λ_recon * λ_recon_dominance * L_recon
        + λ_contra * L_contrastive + L_simulated_unseen

Changes from baseline:
  1. λ_vlm discounting (0.2 vs old 2.0)
  2. Label smoothing on masked/unseen GT
  3. Physics veto for geometrically impossible VLM edges
  4. Feature reconstruction dominance (λ_recon_dominance = 5x)
  5. Simulated-unseen clean fine-tuning (the "silver bullet")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from constants import Constants as const
from lib.supervised.worldsgg.worldsgg_base import PhysicsVeto, LabelSmoother


class AMWAELoss(nn.Module):
    """
    Triple-objective loss with VLM noise handling + simulated-unseen.

    Args:
        lambda_vlm: Weight for masked/unseen (VLM) edges.
        lambda_recon: Feature reconstruction loss base weight.
        lambda_recon_dominance: Multiplier on reconstruction (>>1 emphasizes visual reality).
        lambda_contrastive: InfoNCE contrastive loss weight.
        p_simulate_unseen: Fraction of visible objects to artificially mask.
        label_smoothing: Epsilon for VLM label smoothing.
        use_physics_veto: Enable geometric masking.
        physics_veto_thresh: Distance threshold for veto.
        temperature: InfoNCE temperature.
        bce_loss: Use BCE for spatial/contacting.
        mode: Task mode.
    """

    def __init__(
        self,
        lambda_vlm: float = 0.2,
        lambda_recon: float = 1.0,
        lambda_recon_dominance: float = 5.0,
        lambda_contrastive: float = 0.5,
        p_simulate_unseen: float = 0.25,
        label_smoothing: float = 0.2,
        use_physics_veto: bool = True,
        physics_veto_thresh: float = 2.0,
        temperature: float = 0.07,
        bce_loss: bool = True,
        mode: str = "predcls",
    ):
        super().__init__()
        self.lambda_vlm = lambda_vlm
        self.lambda_recon = lambda_recon
        self.lambda_recon_dominance = lambda_recon_dominance
        self.lambda_contrastive = lambda_contrastive
        self.p_simulate_unseen = p_simulate_unseen
        self.temperature = temperature
        self.bce_loss = bce_loss
        self.mode = mode

        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCELoss()
        self._kl_loss = nn.KLDivLoss(reduction='batchmean')

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
        """Compute triple-objective loss with VLM noise handling."""
        device = predictions["attention_distribution"][0].device
        losses = {}

        # 1. Scene graph loss (split visible/masked with VLM handling)
        sg_losses = self._compute_scene_graph_loss(
            predictions, gt_attention, gt_spatial, gt_contacting,
            gt_node_labels, visibility_mask_seq, person_idx_seq,
            object_idx_seq, valid_mask_seq, corners_seq, device,
        )
        losses.update(sg_losses)

        # 2. Feature reconstruction loss (DOMINANT — drives visual fidelity)
        recon_loss = self._compute_reconstruction_loss(predictions, device)
        losses["recon_loss"] = recon_loss * self.lambda_recon * self.lambda_recon_dominance

        # 3. Contrastive loss (InfoNCE on cross-attention)
        contra_loss = self._compute_contrastive_loss(predictions, device)
        losses["contrastive_loss"] = contra_loss * self.lambda_contrastive

        # 4. Simulated-unseen clean fine-tuning (the "silver bullet")
        sim_loss = self._compute_simulated_unseen_loss(predictions, device)
        losses["simulated_unseen_loss"] = sim_loss

        # Total
        losses["total"] = sum(v for v in losses.values() if isinstance(v, torch.Tensor) and v.requires_grad)

        return losses

    # ------------------------------------------------------------------
    # Scene Graph Loss
    # ------------------------------------------------------------------
    def _compute_scene_graph_loss(
        self, predictions, gt_attention, gt_spatial, gt_contacting,
        gt_node_labels, visibility_mask_seq, person_idx_seq,
        object_idx_seq, valid_mask_seq, corners_seq, device,
    ) -> Dict[str, torch.Tensor]:
        """Split visible/masked SG loss with VLM noise handling."""
        T = len(predictions["attention_distribution"])

        vis_att_p, vis_att_g = [], []
        vis_spa_p, vis_spa_g = [], []
        vis_con_p, vis_con_g = [], []
        mask_att_p, mask_att_g = [], []
        mask_spa_p, mask_spa_g = [], []
        mask_con_p, mask_con_g = [], []
        node_p, node_g = [], []

        for t in range(T):
            att_pred = predictions["attention_distribution"][t]
            spa_pred = predictions["spatial_distribution"][t]
            con_pred = predictions["contacting_distribution"][t]

            K_t = att_pred.shape[0]
            if K_t == 0:
                continue

            att_gt = gt_attention[t].to(device)
            spa_gt = self._build_multi_label_gt(gt_spatial[t], 6, device)
            con_gt = self._build_multi_label_gt(gt_contacting[t], 17, device)

            if visibility_mask_seq is not None and person_idx_seq is not None:
                vis_t = visibility_mask_seq[t]
                p_idx = person_idx_seq[t]
                o_idx = object_idx_seq[t]
                pair_visible = vis_t[p_idx] & vis_t[o_idx]
                pair_masked = ~pair_visible

                # Visible: clean manual GT
                if pair_visible.any():
                    vis_att_p.append(att_pred[pair_visible])
                    vis_att_g.append(att_gt[pair_visible])
                    vis_spa_p.append(spa_pred[pair_visible])
                    vis_spa_g.append(spa_gt[pair_visible])
                    vis_con_p.append(con_pred[pair_visible])
                    vis_con_g.append(con_gt[pair_visible])

                # Masked: VLM pseudo-labels
                if pair_masked.any():
                    m_att_pred = att_pred[pair_masked]
                    m_att_gt = att_gt[pair_masked]
                    m_spa_pred = spa_pred[pair_masked]
                    m_spa_gt = spa_gt[pair_masked]
                    m_con_pred = con_pred[pair_masked]
                    m_con_gt = con_gt[pair_masked]

                    # Physics veto
                    if self._physics_veto is not None and corners_seq is not None:
                        keep = self._physics_veto.compute_veto_mask(
                            corners_seq[t], p_idx[pair_masked], o_idx[pair_masked], m_con_pred
                        )
                        if keep.any():
                            m_att_pred, m_att_gt = m_att_pred[keep], m_att_gt[keep]
                            m_spa_pred, m_spa_gt = m_spa_pred[keep], m_spa_gt[keep]
                            m_con_pred, m_con_gt = m_con_pred[keep], m_con_gt[keep]
                        else:
                            continue

                    # Label smoothing
                    if self._label_smoother is not None:
                        m_spa_gt = self._label_smoother.smooth_bce_target(m_spa_gt)
                        m_con_gt = self._label_smoother.smooth_bce_target(m_con_gt)

                    if len(m_att_pred) > 0:
                        mask_att_p.append(m_att_pred)
                        mask_att_g.append(m_att_gt)
                        mask_spa_p.append(m_spa_pred)
                        mask_spa_g.append(m_spa_gt)
                        mask_con_p.append(m_con_pred)
                        mask_con_g.append(m_con_gt)
            else:
                vis_att_p.append(att_pred)
                vis_att_g.append(att_gt)
                vis_spa_p.append(spa_pred)
                vis_spa_g.append(spa_gt)
                vis_con_p.append(con_pred)
                vis_con_g.append(con_gt)

            # Node classification
            if gt_node_labels is not None and self.mode in [const.SGCLS, const.SGDET]:
                node_pred_t = predictions["node_logits"][t]
                node_gt_t = gt_node_labels[t].to(device)
                if valid_mask_seq is not None:
                    valid_t = valid_mask_seq[t]
                    node_pred_t = node_pred_t[valid_t]
                    node_gt_t = node_gt_t[valid_t]
                if len(node_gt_t) > 0:
                    node_p.append(node_pred_t)
                    node_g.append(node_gt_t)

        losses = {}
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        # Visible losses (full weight, clean GT)
        vis_losses = self._relationship_losses(vis_att_p, vis_att_g, vis_spa_p, vis_spa_g, vis_con_p, vis_con_g, device)
        for k, v in vis_losses.items():
            losses[f"vis_{k}"] = v

        # Masked losses (λ_vlm weighted, smoothed)
        mask_losses = self._relationship_losses(
            mask_att_p, mask_att_g, mask_spa_p, mask_spa_g, mask_con_p, mask_con_g, device,
            smoothed_ce=True,
        )
        for k, v in mask_losses.items():
            losses[f"masked_{k}"] = v * self.lambda_vlm

        # Node loss
        if node_p:
            losses[const.OBJECT_LOSS] = self._ce_loss(torch.cat(node_p), torch.cat(node_g))

        return losses

    # ------------------------------------------------------------------
    # Feature Reconstruction Loss
    # ------------------------------------------------------------------
    def _compute_reconstruction_loss(self, predictions, device):
        """
        MSE between retrieved tokens and original DINO features for masked tokens.

        This is the dominant loss for M2 — drives visual fidelity over VLM text.
        """
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        T = len(predictions.get("is_masked", []))
        if T == 0:
            return zero

        recon_losses = []
        for t in range(T):
            is_masked = predictions["is_masked"][t]  # (N_t,) bool
            if not is_masked.any():
                continue

            retrieved = predictions["retrieved_tokens"][t]  # (N_t, d_model)
            original = predictions["original_visual"][t]    # (N_t, d_visual)

            # Compare masked token retrievals with originals
            masked_retrieved = retrieved[is_masked]
            masked_original = original[is_masked]

            if masked_retrieved.shape[-1] != masked_original.shape[-1]:
                # Project if dims differ
                masked_original = F.adaptive_avg_pool1d(
                    masked_original.unsqueeze(0), masked_retrieved.shape[-1]
                ).squeeze(0)

            recon_losses.append(F.mse_loss(masked_retrieved, masked_original))

        if not recon_losses:
            return zero

        return torch.stack(recon_losses).mean()

    # ------------------------------------------------------------------
    # Contrastive Loss (InfoNCE)
    # ------------------------------------------------------------------
    def _compute_contrastive_loss(self, predictions, device):
        """
        InfoNCE on cross-attention weights.

        For each masked token, attention to SAME object (positive) should be higher.
        """
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        T = len(predictions.get("attn_weights", []))
        if T == 0:
            return zero

        losses = []
        for t in range(T):
            is_masked = predictions["is_masked"][t]  # (N_t,) bool
            attn_weights = predictions["attn_weights"][t]  # (N_t, M)
            mem_obj_ids = predictions["memory_object_ids"][t]  # (M,)

            if not is_masked.any() or attn_weights is None:
                continue

            N_t = is_masked.shape[0]
            masked_indices = torch.where(is_masked)[0]

            for i in masked_indices:
                obj_id = i  # Object slot ID matches index
                # Positive: memory slots belonging to same object
                pos_mask = (mem_obj_ids == obj_id)
                if not pos_mask.any():
                    continue

                attn_i = attn_weights[i]  # (M,)
                pos_score = attn_i[pos_mask].sum()
                total_score = attn_i.sum()

                if total_score > 0:
                    losses.append(-torch.log(pos_score / (total_score + 1e-8)))

        if not losses:
            return zero

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Simulated-Unseen Clean Fine-Tuning (The Silver Bullet)
    # ------------------------------------------------------------------
    def _compute_simulated_unseen_loss(self, predictions, device):
        """
        Loss for artificially masked visible objects.

        During training, p_simulate_unseen fraction of VISIBLE objects get [MASK]ed.
        The model must retrieve features and predict SG using memory.
        Loss is against PERFECT MANUAL GT (not VLM labels).

        This teaches flawless retrieval without VLM noise.
        """
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        # Simulated-unseen predictions are stored separately by the model
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

            # Standard CE/BCE on perfect manual GT
            if "attention" in pred and "attention" in gt:
                att_loss = self._ce_loss(pred["attention"], gt["attention"].to(device))
                losses.append(att_loss)
            if "spatial" in pred and "spatial" in gt:
                spa_gt = self._build_multi_label_gt(gt["spatial"], 6, device)
                spa_loss = self._bce_loss(pred["spatial"], spa_gt)
                losses.append(spa_loss)
            if "contacting" in pred and "contacting" in gt:
                con_gt = self._build_multi_label_gt(gt["contacting"], 17, device)
                con_loss = self._bce_loss(pred["contacting"], con_gt)
                losses.append(con_loss)

        if not losses:
            return zero

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Shared Helpers
    # ------------------------------------------------------------------
    def _relationship_losses(self, att_p, att_g, spa_p, spa_g, con_p, con_g, device, smoothed_ce=False):
        """Compute relationship losses from accumulated tensors."""
        losses = {}
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        if att_p:
            all_att_pred = torch.cat(att_p)
            all_att_gt = torch.cat(att_g)
            if smoothed_ce and self._label_smoother is not None:
                smoothed = self._label_smoother.smooth_ce_target(all_att_gt, 3)
                losses["att"] = self._kl_loss(F.log_softmax(all_att_pred, dim=-1), smoothed)
            else:
                losses["att"] = self._ce_loss(all_att_pred, all_att_gt)
        else:
            losses["att"] = zero

        if spa_p:
            losses["spa"] = self._bce_loss(torch.cat(spa_p), torch.cat(spa_g))
        else:
            losses["spa"] = zero

        if con_p:
            losses["con"] = self._bce_loss(torch.cat(con_p), torch.cat(con_g))
        else:
            losses["con"] = zero

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
