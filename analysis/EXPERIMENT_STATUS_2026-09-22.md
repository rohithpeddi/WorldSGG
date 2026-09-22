# Experiment status — all tracks (2026-09-22)

Test set for everything below: `world4d_rel_annotations_worldbbox/test`,
1,511 videos / 48,834 frames, `annotation_version = 40decd1785af3c3470290be4b67443e1`.
All scoring is **all frames**, best epoch. Values in %.

Two protocols, NOT comparable to each other:
* supervised rows — stock WorldSGG evaluator, sgdet matched by 2D IoU 0.5;
* MLLM rows — `lib/mllm/eval/`, sgdet matched by 3D-IoU on oriented boxes.

---

## Track 3 — learning / WorldWise lineage  (28 / 28 complete, all scored)

Training and scoring finished 2026-09-19. Table artifact:
`analysis/worldbbox_lineage_2026-09-19.md`; raw JSON under
`/data3/rohith/ag/runs/{rescore,worldformer/score,worldwise_pp/score}`.

### predcls (14 cells)

| # | method | backbone / stream | wc R@20 | wc mR@20 | nc mR@20 | OU-nt R@20 | OU-nt mR@20 | state |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | W-STTran | resnet50 | 68.2 | 37.6 | 70.0 | 56.9 | 30.7 | done |
| 2 | W-STTran++ | resnet50 | 66.9 | 34.0 | 63.6 | 67.0 | 29.1 | done |
| 3 | W-DSGDetr | resnet50 | 67.5 | 33.4 | 65.6 | 62.7 | 27.0 | done |
| 4 | W-DSGDetr++ | resnet50 | 68.5 | 38.9 | 71.0 | 55.3 | 26.1 | done |
| 5 | W-USG | resnet50 | 67.3 | 33.4 | 64.0 | 67.5 | 29.1 | done |
| 6 | WorldWise | resnet50 | 68.0 | 49.1 | 81.8 | 73.5 | 40.9 | done |
| 7 | WorldWise | dinov2b | 68.2 | 45.4 | 79.4 | 74.7 | 38.7 | done |
| 8 | WorldWise | dinov2l | 68.8 | 47.8 | 81.6 | 74.2 | 42.6 | done |
| 9 | WorldWise | dinov3l | 69.0 | 49.6 | 82.4 | 72.9 | 41.1 | done |
| 10 | WorldWise+ | dinov3tok | **73.5** | **54.7** | 85.6 | 76.0 | 45.2 | done |
| 11 | WorldWise+ | pi3tok | 71.9 | 51.5 | 84.4 | 75.4 | 46.0 | done |
| 12 | WorldWise+ | fused | 73.2 | 54.1 | 84.5 | 75.9 | 44.1 | done |
| 13 | WorldWise++ | dinov3 | **74.8** | 54.4 | **86.2** | **76.6** | **48.1** | done |
| 14 | WorldWise++ | dinov3_nodet (ablation) | 74.1 | 54.3 | 86.0 | 75.0 | 40.1 | done |

### sgdet (14 cells)

| # | method | backbone / stream | wc R@20 | wc mR@20 | wc mR@50 | nc mR@20 | OU-nt R@20 | OU-nt mR@20 | state |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 15 | W-STTran | resnet50 | 56.3 | 19.2 | 33.4 | 27.7 | 14.6 | 7.1 | done |
| 16 | W-STTran++ | resnet50 | 56.6 | 17.6 | 30.3 | 25.4 | 15.6 | 6.6 | done |
| 17 | W-DSGDetr | resnet50 | 56.6 | 19.2 | 33.5 | 28.0 | 12.3 | 6.0 | done |
| 18 | W-DSGDetr++ | resnet50 | 56.4 | 20.3 | 35.4 | 31.2 | 14.3 | 7.7 | done |
| 19 | W-USG | resnet50 | 56.3 | 19.1 | 34.1 | 25.9 | 14.5 | 6.7 | done |
| 20 | WorldWise | resnet50 | 53.6 | 20.4 | 38.5 | 36.1 | 25.4 | 20.6 | done |
| 21 | WorldWise | dinov2b | 54.1 | 23.4 | 40.1 | 37.9 | 29.6 | 21.1 | done |
| 22 | WorldWise | dinov2l | 53.5 | 22.3 | 38.1 | 37.2 | 30.4 | 20.2 | done |
| 23 | WorldWise | dinov3l | 52.6 | 21.5 | 38.8 | 37.5 | 28.1 | 21.3 | done |
| 24 | WorldWise+ | dinov3tok | 54.2 | 23.1 | 45.5 | 43.3 | 26.6 | 20.5 | done |
| 25 | WorldWise+ | pi3tok | 53.2 | 22.9 | 43.8 | 41.1 | 27.2 | **22.4** | done |
| 26 | WorldWise+ | fused | 54.2 | 23.9 | 46.9 | 43.5 | 27.1 | 21.1 | done |
| 27 | WorldWise++ | dinov3 | **54.6** | **24.5** | **48.6** | 44.1 | 27.8 | 22.3 | done |
| 28 | WorldWise++ | dinov3_nodet (ablation) | 54.8 | 24.1 | 48.4 | **43.9** | 27.7 | 21.9 | done |

---

## Track 2 — MLLM  (10 / 19 scored, 2 running, 3 queued, 4 generated-but-unscored)

Outputs under `/data3/rohith/ag/runs/mllm/<method>/<mode>/<model>/`.
`think150` = the fixed 150-video subset `splits/test_worldbbox_thinking150.txt`.

### predcls (10 cells)

| # | method | model | split | videos | run state | scoring | wc R@20 | wc mR@20 |
|---|---|---|---|---:|---|---|---:|---:|
| 29 | zero_shot (Graph-RAG, unlocalized) | qwen25vl_7b | full | 1511/1511 | done | scored | 46.9 | 25.1 |
| 30 | caption_all | qwen25vl_7b | full | 1511/1511 | done | scored | 45.3 | 26.1 |
| 31 | rag_all | qwen25vl_7b | full | 1511/1511 | done | scored | 46.9 | 26.0 |
| 32 | wsg_agent | qwen25vl_7b | full | **1442/1511** | done, 69 short | **NOT scored** | – | – |
| 33 | **Track A** (marked frames + BEV) | qwen3vl_8b | full | 1511/1511 | done | scored | **51.7** | **30.5** |
| 34 | Track A (same run, subset score) | qwen3vl_8b | think150 | 150/150 | done | scored | 51.4 | 29.9 |
| 35 | **Track B** (tool loop + geometric critic) | qwen3vl_8b | full | 1511/1511 | **done 2026-09-22 03:27** | **NOT scored** | – | – |
| 36 | rag_all (2×2: RAG × standard) | qwen3vl_8b | think150 | 150/150 | done | scored | 47.4 | 28.8 |
| 37 | rag_all (2×2: RAG × thinking) | qwen3vl_8b_thinking | think150 | 150/150 | done | **NOT scored** | – | – |
| 38 | Track A × thinking | qwen3vl_8b_thinking | think150 | 150/150 | done | scored | 30.5 | 19.3 |

Row 38 is **invalid, not a result**: the reasoning trace eats the shared
`max_new_tokens` budget, only 46.2 % of responses close `</think>`, pair coverage
10,090/25,673. Report it as a budget failure or re-run per-object, never as a
thinking-vs-standard comparison.

### sgdet (9 cells)

| # | method | model | split | videos | run state | scoring | unloc nc R@50 | IoU.15 nc R@50 | slots w/ 3D |
|---|---|---|---|---:|---|---|---:|---:|---:|
| 39 | **Track A** | qwen3vl_8b | full | 1511/1511 | done | scored | 33.3 | 11.0 | 80.8 |
| 40 | Track A (subset score) | qwen3vl_8b | think150 | 150/150 | done | scored | 33.4 | 10.5 | 80.2 |
| 41 | **Track B** | qwen3vl_8b | full | **1440/1511** | **RUNNING** (2 workers, ~71 left) | pending | – | – | – |
| 42 | rag_all (2×2 std) | qwen3vl_8b | think150 | 145/150 | done | scored | 22.7 | 0.0 | 0.0 |
| 43 | rag_all (2×2 thinking) | qwen3vl_8b_thinking | think150 | 143/150 | done | **NOT scored** | – | – | – |
| 44 | Track A × thinking | qwen3vl_8b_thinking | think150 | 150/150 | done | scored | 0.6 | 0.2 | 0.4 |
| 45 | zero_shot | qwen25vl_7b | full | **38/1511** | **RUNNING** (GPU 0, from 11:05) | pending | – | – | – |
| 46 | caption_all | qwen25vl_7b | full | 0/1511 | **QUEUED** | pending | – | – | – |
| 47 | rag_all | qwen25vl_7b | full | 0/1511 | **QUEUED** | pending | – | – | – |
| 48 | wsg_agent | qwen25vl_7b | full | 0/1511 | **QUEUED** | pending | – | – | – |

Row 44 is invalid for the same reason as row 38 (0.7 % trace closure; 114 predicted
pairs out of 25,673). Rows 42/43: `rag_all` emits no oriented boxes at all, so RAG
sgdet rows are class-only and can never be a localization comparison.

---

## Supporting runs (not paper cells)

| item | scope | state |
|---|---|---|
| WorldBBox alignment gate (predcls + sgdet) | 1,511 videos | done — 0.12 % zero-corner slots, gt-2D IoU 0.998 / 0.744 |
| Stage-1 graph fill | 1,069 of 1,511 videos | done (4.9 h) |
| DINOv3 token cache | train + test | done (914 GB / 254 GB) |
| π³ token cache | train + test | done |
| WorldWise++ PCA-256 grid cache | train + test | done (NVMe `/code`) |
| Smoke runs (Track A/B, RAG-thinking, 2 videos each) | – | done, excluded from tables |

---

## What is left to finish

| # | work | where | estimate |
|---|---|---|---|
| 1 | Track B sgdet — last ~71 videos | UTD GPU 1 + GPU 2 | ~1 h |
| 2 | zero_shot / caption_all / rag_all / wsg_agent sgdet | UTD queue, 3 GPUs | ~12 h wall (~35 GPU-h) |
| 3 | wsg_agent predcls — recover the 69 missing videos | UTD | ~30 min |
| 4 | Score rows 32, 35, 37, 41, 43 and then rows 45–48 | CPU, `eval/score_all` | ~20 min per run |
| 5 | Decide on rows 38 and 44 (thinking budget defect) | – | re-run per-object, or report as a negative result |
| 6 | Merge `exp/mllm-tracks` and `exp/worldformer` into `methods` | local | – |

The jobs file notes a split with the IITD pragya cluster (thinking jobs,
`wsg_agent` predcls, the scorers). Pragya is not resolvable from this machine
right now, so its side could not be verified in this pass.
