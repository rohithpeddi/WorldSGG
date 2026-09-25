"""W-DSGDetr++ method figure (hero style), self-contained, three stages:

    Stage 1  Perception, Geometry Scaffold And LKS Memory
    Stage 2  Tokenization And Spatio-Temporal Reasoning
    Stage 3  Relationship Heads And Bucketed Loss

W-DSGDetr plus the ObjectMotionEncoder (``MotionFeatureEncoder`` in
``components.py``) and its fusion MLP: world-frame velocity of the OBB centre,
its speed and the camera-relative velocity Rᵀv, gated on consecutive validity
and fused into every slot token before the temporal object encoder.  Those two
are the orange modules; every other module is repeated from W-DSGDetr.

Usage::

    python scripts/paper_figures/dark/fig_w_dsgdetr_pp.py \
        --images outputs/intermediates/12XD3/w_dsgdetr_pp --video 12XD3
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
    cards=[
        ("Remember By Copying",
         ["An unseen slot re-uses the ROI features of its nearest",
          "visible frame in either direction, with a staleness",
          "counter; never-seen slots get zeros and Δ = 1000.",
          "No parameters, no gradient. On 12XD3 the picture's",
          "ten unseen cells all copy frame 13 and Δ climbs to 10."]),
        ("World-Frame Motion In Every Token",
         ["The centre velocity v = cₜ − cₜ₋₁, its speed and the",
          "camera-relative velocity Rᵀv are encoded (64-d) and",
          "fused into each slot token by a small MLP, gated on",
          "valid[t] ∧ valid[t−1] so slot gaps add nothing. The",
          "acceleration input is wired but always zero."]),
        ("What This Baseline Does Not Have",
         ["No ego-motion encoder (the CameraTemporalEncoder is",
          "WorldWise-only), no artificial masking, no",
          "reconstruction target and no logit adjustment.",
          "Unseen pairs (13 of 92 on 12XD3) use VLM pseudo-",
          "labels at λ_vlm = 0.2 with label smoothing 0.2."]),
    ],
)

if __name__ == "__main__":
    run(TIER, __file__)
