# `setup/` — per-method design documentation

One document per track and one per method: what it is, why it is built that way,
what it optimises or prompts for, what it actually scored, and what is wrong
with it.

Authoritative results for every table in this directory:
[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md).
No number appears here that is not traceable to that file, to a JSON under
`results/`, or to a `.tex` table.

---

## The three tracks

| Track | Document | What it is | Localization | SGDet matching |
|---|---|---|---|---|
| **1** | [TRACK1_TRAINING.md](TRACK1_TRAINING.md) | WorldWise lineage + supervised baselines | learned 3D boxes | 2D IoU 0.5 |
| **2** | [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) | training-free MLLMs emitting class-only graphs | none | class-only |
| **3** | [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) | training-free MLLMs that also place objects in 3D | predicted oriented boxes | 3D IoU 0.15 / 0.25 |

**PredCls is comparable across all three tracks** — the same stock WorldSGG
evaluator scores every row with ground-truth boxes supplied. **SGDet is not.**
Never put a track-1 and a track-3 SGDet number in one column.

---

## Track 1 — training-based

| Method | Document |
|---|---|
| W-STTran, W-STTran++, W-DSGDetr, W-DSGDetr++, W-USG | [BASELINES.md](BASELINES.md) |
| WorldWise | [WORLDWISE.md](WORLDWISE.md) |
| WorldWise+ | [WORLDWISE_PLUS.md](WORLDWISE_PLUS.md) |
| WorldWise++ | [WORLDWISE_PP.md](WORLDWISE_PP.md) |
| Monocular 3D detector (upstream of everything) | [MONOCULAR_3D_DETECTOR.md](MONOCULAR_3D_DETECTOR.md) |

Operational recipes: [RUN_WSGG.md](RUN_WSGG.md) (training + evaluation),
[RUN_MON3D.md](RUN_MON3D.md) (detector).
Lineage diagram: [worldwise_variants_overview.svg](worldwise_variants_overview.svg).

## Track 2 — unlocalized MLLM

A nested ladder; each rung adds one context source to the same per-object prompt.

| Method key | Document | Adds |
|---|---|---|
| `zero_shot` | [MLLM_ZERO_SHOT.md](MLLM_ZERO_SHOT.md) | nothing — frames + prompt |
| `caption_all` | [MLLM_CAPTION_ALL.md](MLLM_CAPTION_ALL.md) | Stage-1 clip captions |
| `rag_all` | [MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) | **Graph-RAG** retrieval over the Stage-1 video graph |
| `wsg_agent` | [MLLM_WSG_AGENT.md](MLLM_WSG_AGENT.md) | a per-object strategy router over `rag_all` |

The shared Stage-1 graph/caption build is documented in
[MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) §2. Full `wsg_agent` architecture:
[docs/WSG_AGENT.md](../docs/WSG_AGENT.md).

## Track 3 — localized MLLM

| Method key | Document | What it is |
|---|---|---|
| `track_a` | [MLLM_TRACK_A.md](MLLM_TRACK_A.md) | marked frames (set-of-mark) + marked BEV, one call per frame |
| `track_b` | [MLLM_TRACK_B.md](MLLM_TRACK_B.md) | fixed tool loop + retrieval + **geometric critic** + repair |
| `track_b --objects_key objects_pre` | [MLLM_TRACK_B.md](MLLM_TRACK_B.md) §5 | the critic ablation, scored from the same run |

---

## The four caveats that must survive into the paper

1. **Backbone confound.** Full-split unlocalized runs use `qwen25vl_7b`; Track A
   and Track B use `qwen3vl_8b`. The only backbone-matched MLLM comparison is
   the 150-video block on `splits/test_worldbbox_thinking150.txt`
   ([TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) §5).
2. **Thinking-budget defect.** The reasoning trace shares `max_new_tokens` with
   the answer, so array-style prompts (Track A, Track B) never reach their
   answer — `</think>` appears in 46.2 % of Track A PredCls responses and 0.7 %
   in SGDet. Per-object RAG prompting is immune. **The RAG thinking cell is the
   only valid one in the project**; every other thinking cell is a budget
   failure, not a model result.
3. **Metric-family split.** The backbone-breadth tables are per-predicate-group
   P/R/F1 with micro/macro F1; everything else is R@K / mR@K. They share the
   test videos, not the metric. The bridge is the `legacy uF1` column.
4. **Incomplete cells are incomplete.** `wsg_agent` PredCls is n = 1458 of
   1,511; the four unlocalized full-split SGDet cells and `wsg_agent` SGDet were
   still generating or queued as of 2026-09-22. Write "not run" / "not scored",
   never a plausible number.

Two further ones that apply to every row: **~37 % of annotation objects have no
feature slot** (never detected by GDino) and are therefore never evaluated by
any method in any track; and **completion must be measured by output-file count
against the split, never by exit code**, because the MLLM base processor catches
per-video failures and continues.

---

## One known naming error

[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) labels the
`zero_shot` rows "Graph-RAG (zero_shot)". The code does not support it:
`ActionGenomeZeroShotProcessor` never opens the graph pickle. Graph-RAG is
`rag_all`. The numbers are correctly attributed to the right runs; only the
display name is wrong. Fix it before submission.
