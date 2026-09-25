"""W-DSGDetr++ method figure (hero style), self-contained, drawn as four processing units (Observed Objects,
Unobserved Objects, Relationship, Decoders; see ``baseline_common.py``).

W-DSGDetr plus the ObjectMotionEncoder (``MotionFeatureEncoder`` in
``components.py``) and its fusion MLP: world-frame velocity of the OBB centre,
its speed and the camera-relative velocity Rᵀv, gated on consecutive validity
and fused into every slot token before the temporal object encoder.  Those two
are the orange modules; every other module is repeated from W-DSGDetr.

Usage::

    python scripts/paper_figures/dark/fig_w_dsgdetr_pp.py \
        --images assets/figures/architecture/intermediates/00T1E/w_dsgdetr_pp --video 00T1E
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import Tier, run  # noqa: E402

TIER = Tier(
    name="w_dsgdetr_pp",
    title="W-DSGDetr++ · LKS Memory Plus World-Frame Object Motion",
    spatial=True, temporal_obj=True, motion=True, usg=False,
    new={"motion"},
)

if __name__ == "__main__":
    run(TIER, __file__)
