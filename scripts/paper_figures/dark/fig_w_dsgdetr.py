"""W-DSGDetr method figure (hero style), self-contained, three stages:

    Stage 1  Perception, Geometry Scaffold And LKS Memory
    Stage 2  Tokenization And Spatio-Temporal Reasoning
    Stage 3  Relationship Heads And Bucketed Loss

W-STTran++ plus the TemporalObjectEncoder
(``lib/supervised/baselines/w_dsgdetr/object_encoder.py``): per-slot
self-attention across the T frames with a learnable temporal positional
encoding, applied before the inter-object transformer.  That encoder is the
orange module; every other module is repeated from W-STTran++.

Usage::

    python scripts/paper_figures/dark/fig_w_dsgdetr.py \
        --images outputs/intermediates/12XD3/w_dsgdetr --video 12XD3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import Tier, run  # noqa: E402

TIER = Tier(
    name="w_dsgdetr",
    title="W-DSGDetr · LKS Memory Plus Per-Slot Temporal Object Encoding",
    spatial=True, temporal_obj=True, motion=False, usg=False,
    new={"temporal_obj"},
    cards=[
        ("Remember By Copying",
         ["An unseen slot re-uses the ROI features of its nearest",
          "visible frame in either direction, with a staleness",
          "counter; never-seen slots get zeros and Δ = 1000.",
          "No parameters, no gradient. On 12XD3 the picture's",
          "ten unseen cells all copy frame 13 and Δ climbs to 10."]),
        ("Track Each Slot Through Time",
         ["A temporal object encoder self-attends over each",
          "slot's T-frame sequence (learnable temporal PE, two",
          "layers) before the inter-object transformer: the",
          "world-slot analogue of DSGDetr's tracking-based object",
          "encoding, with no Hungarian matching anywhere."]),
        ("What This Baseline Does Not Have",
         ["No motion encoder (W-DSGDetr++ adds it) and no",
          "ego-motion encoder; no artificial masking, no",
          "reconstruction target and no logit adjustment.",
          "Unseen pairs (13 of 92 on 12XD3) use VLM pseudo-",
          "labels at λ_vlm = 0.2 with label smoothing 0.2."]),
    ],
)

if __name__ == "__main__":
    run(TIER, __file__)
