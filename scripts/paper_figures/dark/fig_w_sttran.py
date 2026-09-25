"""W-STTran method figure (hero style), self-contained, drawn as four processing units (Observed Objects,
Unobserved Objects, Relationship, Decoders; see ``baseline_common.py``).

The base tier of the adapted-baseline ladder: frozen ResNet-50 Faster R-CNN
ROI features, the non-differentiable LKS zero-order-hold buffer, the global
structural encoder, the LKS tokenizer (no camera slice), the inter-object
transformer with a 3-D pairwise positional encoding, the node predictor, the
pair former, temporal edge attention and the bucketed LKS loss.  The orange
modules are the world-frame substrate this tier adds over the 2-D STTran.

Usage::

    python scripts/paper_figures/dark/fig_w_sttran.py \
        --images assets/figures/architecture/intermediates/00T1E/w_sttran --video 00T1E
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
)

if __name__ == "__main__":
    run(TIER, __file__)
