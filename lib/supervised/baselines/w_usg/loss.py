"""
W-USG Loss — LKS bucketed loss + text-centric contrastive alignment.

Identical to the shared LKSLoss (vis-vis clean labels at full weight,
unseen buckets at lambda_vlm) so relation supervision matches the ladder
baselines exactly, plus USG-Par's text-centric term: a cross-entropy over
cosine similarities between projected object tokens and the frozen CLIP
class embeddings (`align_logits` from the model). One term per valid
object slot, weighted by lambda_align.

Relation-level text contrast is NOT included — the repo ships CLIP
embeddings for object classes only, and inventing predicate text
embeddings would add a data dependency the baselines don't have.
"""

import logging

import torch
import torch.nn as nn
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from lib.supervised.baselines.lks_buffer.loss import LKSLoss


class WUSGLoss(LKSLoss):

    def __init__(
        self,
        lambda_vlm: float = 0.2,
        label_smoothing: float = 0.2,
        bce_loss: bool = True,
        mode: str = "predcls",
        lambda_align: float = 0.1,
        **logit_adjustment_kwargs,
    ):
        super().__init__(
            lambda_vlm=lambda_vlm,
            label_smoothing=label_smoothing,
            bce_loss=bce_loss,
            mode=mode,
            **logit_adjustment_kwargs,
        )
        self.lambda_align = lambda_align
        self._align_ce = nn.CrossEntropyLoss()

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
        losses = super().forward(
            predictions=predictions,
            gt_attention=gt_attention,
            gt_spatial=gt_spatial,
            gt_contacting=gt_contacting,
            pair_valid=pair_valid,
            visibility_mask=visibility_mask,
            person_idx=person_idx,
            object_idx=object_idx,
            valid_mask=valid_mask,
            gt_node_labels=gt_node_labels,
        )

        # Text-centric alignment: CE over CLIP-space similarities per slot
        align_logits = predictions.get("align_logits")
        if (
            self.lambda_align > 0
            and align_logits is not None
            and gt_node_labels is not None
        ):
            labels = gt_node_labels.to(align_logits.device)
            if valid_mask is not None:
                sel = valid_mask.bool()
                logits = align_logits[sel]
                labels = labels[sel]
            else:
                logits = align_logits.reshape(-1, align_logits.shape[-1])
                labels = labels.reshape(-1)
            keep = (labels >= 0) & (labels < logits.shape[-1])
            if keep.any():
                align = self._align_ce(logits[keep], labels[keep])
                losses["alignment_loss"] = align * self.lambda_align
                losses["total"] = losses["total"] + losses["alignment_loss"]

        return losses
