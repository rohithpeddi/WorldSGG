"""W-USG method figure (hero style), self-contained, three stages:

    Stage 1  Perception, Geometry Scaffold And LKS Memory
    Stage 2  Tokenization And Spatio-Temporal Reasoning
    Stage 3  Relationship Heads And Bucketed Loss

The USG-Par relation stack on the W-STTran substrate
(``lib/supervised/baselines/w_usg/w_usg.py``): the LKS buffer, structural
encoder, tokenizer (no camera slice), node predictor and pair former are
repeated from W-STTran; the inter-object transformer becomes a plain per-frame
object context encoder (no 3-D PE), temporal edge attention is replaced by a
relation decoder whose pair queries self-attend and cross-attend to the frame's
object tokens, and an alignment head scores object tokens against the frozen
CLIP class embeddings for an extra cross-entropy (λ_align = 0.1).  Those three
are the orange modules.

Usage::

    python scripts/paper_figures/dark/fig_w_usg.py \
        --images outputs/intermediates/12XD3/w_usg --video 12XD3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import Tier, run  # noqa: E402

TIER = Tier(
    name="w_usg",
    title="W-USG · The USG-Par Relation Stack On The LKS Substrate",
    spatial=False, temporal_obj=False, motion=False, usg=True,
    new={"context", "usg_decoder", "align"},
    legend_new="USG-Par Relation Stack (New)",
    cards=[
        ("Same Substrate As W-STTran",
         ["LKS buffer, structural encoder, tokenizer without a",
          "camera slice, node predictor and pair former are",
          "byte-identical to W-STTran, so the comparison is",
          "about the relation stack alone. The buffer stays",
          "because USG-Par has no way to score unseen slots."]),
        ("The USG-Par Relation Stack",
         ["A plain per-frame object context encoder (no 3-D PE),",
          "a relation decoder in which pair queries self-attend",
          "and cross-attend to the frame's object tokens, and a",
          "text-centric alignment of object tokens to the CLIP",
          "class embeddings (cosine / 0.07, CE at λ_align = 0.1)."]),
        ("What This Baseline Does Not Have",
         ["No temporal edge attention (relations are per frame;",
          "slot identity is given), no 3-D positional encoding,",
          "no camera, motion or ego-motion encoder; no masking,",
          "reconstruction or logit adjustment. Unseen pairs (13 of",
          "92 on 12XD3) use VLM pseudo-labels at λ_vlm = 0.2."]),
    ],
)

if __name__ == "__main__":
    run(TIER, __file__)
