"""W-STTran++ method figure (hero style), self-contained, drawn as four processing units (Observed Objects,
Unobserved Objects, Relationship, Decoders; see ``baseline_common.py``).

W-STTran plus the ObjectSpatialEncoder (``CameraPoseEncoder`` in
``components.py``): per-object camera-relative features (log-distance, view
alignment, azimuth sin / cos) from the frame's camera pose, which give the LKS
tokenizer a 128-d camera slice.  That encoder is the orange module; every other
module is repeated from W-STTran.

Usage::

    python scripts/paper_figures/dark/fig_w_sttran_pp.py \
        --images assets/figures/architecture/intermediates/00T1E/w_sttran_pp --video 00T1E
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import Tier, run  # noqa: E402

TIER = Tier(
    name="w_sttran_pp",
    title="W-STTran++ · LKS Memory Plus Camera-Relative Object Features",
    spatial=True, temporal_obj=False, motion=False, usg=False,
    new={"spatial"},
)

if __name__ == "__main__":
    run(TIER, __file__)
