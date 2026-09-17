"""
WorldFormer C1 — token-swap model.

WorldWise with its single appearance seam (``ScaffoldTokenizer.visual_projector``,
plus the relation head's ``union_proj``) replaced by a projector over frozen
foundation-model tokens instead of the FRCNN ``box_head`` output:

* ``dinov3_tok`` — ROI-pooled DINOv3-L patch tokens (layers L16|L20|L24n, 3072-d)
* ``pi3_tok``    — ROI-pooled Pi3 decoder latents (F4|F14|G14, 3072-d)
* ``fused``      — both, concatenated in the feature PKL (6144-d) and combined
                   here by a **gated fusion**: per-stream Linear -> d, a per-dimension
                   softmax gate over the streams (computed from the concatenated
                   projections), then ReLU + LayerNorm (the base
                   projector's non-linearity), so the single-stream case
                   reduces exactly to the base ``Linear -> ReLU -> LayerNorm``.

Everything else (geometry scaffold, MWAE masking, EMA reconstruction target,
associative retriever, relation head, temporal edges) is untouched, so C1 vs
WorldWise@dinov3l isolates "latents vs decoded outputs".

Config keys (see configs/methods/*/worldformer_c1_*.yaml):
    token_streams: [3072]            # single stream  (dinov3_tok / pi3_tok)
    token_streams: [3072, 3072]      # fused: dims of the concatenated blocks, in order
    fusion: gated | concat           # concat = one Linear over the full vector
    d_detector_roi / d_union_roi     # must equal sum(token_streams)
"""
from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn

from lib.supervised.worldwise.worldwise import WorldWise


class GatedFusionProjector(nn.Module):
    """Project K concatenated token streams (dims ``d_streams``) to ``d_out``.

    K == 1 or fusion == "concat":  Linear(sum d) -> ReLU -> LayerNorm (base form).
    K >= 2 and fusion == "gated":  h_k = Linear_k(x_k); g = softmax_k(W [h_1..h_K])
        (K-way softmax over streams, per output dim); h = sum_k g_k * h_k; ReLU; LN.
    The mean gate per stream is kept in ``last_gate_mean`` for logging.
    """

    def __init__(self, d_streams: List[int], d_out: int, fusion: str = "gated"):
        super().__init__()
        self.d_streams = list(int(d) for d in d_streams)
        self.d_out = d_out
        self.fusion = fusion if len(self.d_streams) > 1 else "concat"
        if self.fusion == "concat":
            self.proj = nn.Linear(sum(self.d_streams), d_out)
        elif self.fusion == "gated":
            self.proj = nn.ModuleList([nn.Linear(d, d_out) for d in self.d_streams])
            self.gate = nn.Linear(d_out * len(self.d_streams), d_out * len(self.d_streams))
        else:
            raise ValueError(f"unknown fusion {fusion!r}")
        self.post = nn.Sequential(nn.ReLU(inplace=True), nn.LayerNorm(d_out))
        self.register_buffer("last_gate_mean", torch.zeros(len(self.d_streams)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != sum(self.d_streams):
            raise ValueError(f"expected {sum(self.d_streams)}-d tokens, got {x.shape[-1]}")
        if self.fusion == "concat":
            return self.post(self.proj(x))
        parts = torch.split(x, self.d_streams, dim=-1)
        hs = [p(t) for p, t in zip(self.proj, parts)]                   # K x (..., d_out)
        H = torch.stack(hs, dim=-2)                                     # (..., K, d_out)
        logits = self.gate(torch.cat(hs, dim=-1))                       # (..., K*d_out)
        g = torch.softmax(logits.view(*logits.shape[:-1], len(hs), self.d_out), dim=-2)
        with torch.no_grad():
            self.last_gate_mean = g.detach().float().transpose(-2, -1).reshape(-1, len(hs)).mean(0)
        return self.post((g * H).sum(dim=-2))


class WorldFormerC1(WorldWise):
    """WorldWise whose appearance projectors consume cached foundation tokens."""

    def __init__(self, config, num_object_classes: int = 37, attention_class_num: int = 3,
                 spatial_class_num: int = 6, contact_class_num: int = 17):
        super().__init__(config, num_object_classes, attention_class_num,
                         spatial_class_num, contact_class_num)
        streams = [int(d) for d in getattr(config, "token_streams", [config.d_detector_roi])]
        fusion = getattr(config, "fusion", "gated")
        if sum(streams) != config.d_detector_roi:
            raise ValueError(f"token_streams {streams} must sum to d_detector_roi={config.d_detector_roi}")
        if config.d_union_roi != config.d_detector_roi:
            raise ValueError("C1 expects d_union_roi == d_detector_roi (same token layout for union boxes)")

        tok = self.scaffold_tokenizer
        tok.visual_projector = GatedFusionProjector(streams, tok.d_visual, fusion)
        if getattr(tok, "use_ema_target", False):
            tok.target_projector = copy.deepcopy(tok.visual_projector)
            for p in tok.target_projector.parameters():
                p.requires_grad_(False)

        rp = self.rel_predictor
        d_union = rp.d_rel // 4
        rp.union_proj = GatedFusionProjector(streams, d_union, fusion)
        self.token_streams = streams
        self.fusion = fusion

    def gate_summary(self) -> Optional[dict]:
        """Mean per-stream gate of the last forward (fused runs only)."""
        vp = self.scaffold_tokenizer.visual_projector
        if getattr(vp, "fusion", "concat") != "gated":
            return None
        return {"visual": vp.last_gate_mean.tolist(),
                "union": self.rel_predictor.union_proj.last_gate_mean.tolist()}
