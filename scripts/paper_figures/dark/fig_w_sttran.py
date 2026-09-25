"""W-STTran method figure (hero style), self-contained, three stages:

    Stage 1  Perception, Geometry Scaffold And LKS Memory
    Stage 2  Tokenization And Spatio-Temporal Reasoning
    Stage 3  Relationship Heads And Bucketed Loss

The base tier of the adapted-baseline ladder: frozen ResNet-50 Faster R-CNN
ROI features, the non-differentiable LKS zero-order-hold buffer, the global
structural encoder, the LKS tokenizer (no camera slice), the inter-object
transformer with a 3-D pairwise positional encoding, the node predictor, the
pair former, temporal edge attention and the bucketed LKS loss.  The orange
modules are the world-frame substrate this tier adds over the 2-D STTran.

Usage::

    python scripts/paper_figures/dark/fig_w_sttran.py \
        --images outputs/intermediates/12XD3/w_sttran --video 12XD3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import Tier, run  # noqa: E402

TIER = Tier(
    name="w_sttran",
    title="W-STTran · Passive LKS Memory And 3-D Spatial Attention",
    spatial=False, temporal_obj=False, motion=False, usg=False,
    new={"gse", "tokenizer", "spatial_pe"},
    legend_new="World-Frame Substrate (New Over STTran)",
    cards=[
        ("Remember By Copying",
         ["An unseen slot re-uses the ROI features of its nearest",
          "visible frame in either direction, with a staleness",
          "counter; never-seen slots get zeros and Δ = 1000.",
          "No parameters, no gradient. On 12XD3 the picture's",
          "ten unseen cells all copy frame 13 and Δ climbs to 10."]),
        ("A 3-D Scaffold Over STTran",
         ["World-frame OBB corners enter twice: as structural",
          "tokens fused with the buffered appearance and the",
          "log-staleness, and as a pairwise 3-D positional",
          "encoding inside the inter-object transformer.",
          "Persistent world slots make tracking free."]),
        ("What This Baseline Does Not Have",
         ["No camera-relative, motion or ego-motion encoder and",
          "no temporal object encoder (higher tiers add them);",
          "no artificial masking, no reconstruction target and",
          "no logit adjustment. Unseen pairs (13 of 92 on 12XD3)",
          "use VLM pseudo-labels at λ_vlm = 0.2, smoothing 0.2."]),
    ],
)

if __name__ == "__main__":
    run(TIER, __file__)
