"""W-USG method figure (hero style), self-contained, drawn as four processing units (Observed Objects,
Unobserved Objects, Relationship, Decoders; see ``baseline_common.py``).

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
        --images assets/figures/architecture/intermediates/00T1E/w_usg --video 00T1E
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
)

if __name__ == "__main__":
    run(TIER, __file__)
