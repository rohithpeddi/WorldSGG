"""W-DSGDetr method figure (hero style), self-contained, drawn as four processing units (Observed Objects,
Unobserved Objects, Relationship, Decoders; see ``baseline_common.py``).

W-STTran++ plus the TemporalObjectEncoder
(``lib/supervised/baselines/w_dsgdetr/object_encoder.py``): per-slot
self-attention across the T frames with a learnable temporal positional
encoding, applied before the inter-object transformer.  That encoder is the
orange module; every other module is repeated from W-STTran++.

Usage::

    python scripts/paper_figures/dark/fig_w_dsgdetr.py \
        --images assets/figures/architecture/intermediates/00T1E/w_dsgdetr --video 00T1E
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
)

if __name__ == "__main__":
    run(TIER, __file__)
