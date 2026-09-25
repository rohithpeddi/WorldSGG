"""W-STTran++ method figure (hero style), self-contained, three stages:

    Stage 1  Perception, Geometry Scaffold And LKS Memory
    Stage 2  Tokenization And Spatio-Temporal Reasoning
    Stage 3  Relationship Heads And Bucketed Loss

W-STTran plus the ObjectSpatialEncoder (``CameraPoseEncoder`` in
``components.py``): per-object camera-relative features (log-distance, view
alignment, azimuth sin / cos) from the frame's camera pose, which give the LKS
tokenizer a 128-d camera slice.  That encoder is the orange module; every other
module is repeated from W-STTran.

Usage::

    python scripts/paper_figures/dark/fig_w_sttran_pp.py \
        --images outputs/intermediates/12XD3/w_sttran_pp --video 12XD3
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
    cards=[
        ("Remember By Copying",
         ["An unseen slot re-uses the ROI features of its nearest",
          "visible frame in either direction, with a staleness",
          "counter; never-seen slots get zeros and Δ = 1000.",
          "No parameters, no gradient. On 12XD3 the picture's",
          "ten unseen cells all copy frame 13 and Δ climbs to 10."]),
        ("See Each Object From The Camera",
         ["The object spatial encoder reads the frame's camera",
          "pose and encodes, per OBB centre, its log-distance to",
          "the camera, the alignment with the viewing axis and",
          "the azimuth (sin, cos). The tokenizer's camera slice",
          "grows from 0-d to 128-d; nothing else changes."]),
        ("What This Baseline Does Not Have",
         ["No motion or ego-motion encoder and no temporal",
          "object encoder (higher tiers add the first two);",
          "no artificial masking, no reconstruction target and",
          "no logit adjustment. Unseen pairs (13 of 92 on 12XD3)",
          "use VLM pseudo-labels at λ_vlm = 0.2, smoothing 0.2."]),
    ],
)

if __name__ == "__main__":
    run(TIER, __file__)
