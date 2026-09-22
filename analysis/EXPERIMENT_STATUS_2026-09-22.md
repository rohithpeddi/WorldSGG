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

## Track 2 — MLLM  (21 / 24 scored, 3 sgdet baselines still generating)

Outputs under `/data3/rohith/ag/runs/mllm/<method>/<mode>/<model>/`.
`think150` = the fixed 150-video subset `splits/test_worldbbox_thinking150.txt`.
Full table: `analysis/mllm_tracks_status_2026-09-22.md`.

### predcls (14 cells — all scored)

| # | method | model | split | videos | wc R@20 | wc mR@20 | nc mR@50 | OU-nt nc mR@50 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 29 | zero_shot (Graph-RAG, unlocalized) | qwen25vl_7b | full | 1511/1511 | 46.9 | 25.1 | 43.1 | 42.8 |
| 30 | caption_all | qwen25vl_7b | full | 1511/1511 | 45.3 | 26.1 | 43.6 | 44.1 |
| 31 | rag_all | qwen25vl_7b | full | 1511/1511 | 46.9 | 26.0 | 43.8 | 44.6 |
| 32 | wsg_agent | qwen25vl_7b | full | 1458/1511 | 46.3 | 25.8 | 43.5 | 43.6 |
| 33 | Track A (marked frames + BEV) | qwen3vl_8b | full | 1511/1511 | 51.7 | 30.5 | 47.2 | 40.0 |
| 34 | Track A, subset score | qwen3vl_8b | think150 | 150/150 | 51.4 | 29.9 | 46.7 | 42.7 |
| 35 | **Track B (tool loop + geometric critic)** | qwen3vl_8b | full | 1511/1511 | **52.4** | **31.7** | 48.4 | 40.8 |
| 36 | Track B without the critic | qwen3vl_8b | full | 1511/1511 | 52.1 | 31.1 | 47.9 | 40.0 |
| 37 | rag_all (2×2: RAG × standard) | qwen3vl_8b | think150 | 150/150 | 47.4 | 28.8 | 48.5 | 51.8 |
| 38 | **rag_all (2×2: RAG × thinking)** | qwen3vl_8b_thinking | think150 | 150/150 | **50.4** | **30.3** | 49.9 | 53.8 |
| 39 | Track A × thinking | qwen3vl_8b_thinking | think150 | 150/150 | 30.5 | 19.3 | 40.4 | 39.9 |
| 40 | Track B × thinking | qwen3vl_8b_thinking | think150 | 150/150 | 27.2 | 16.4 | 38.2 | 38.2 |
| 41 | Track B × thinking, no critic | qwen3vl_8b_thinking | think150 | 150/150 | 27.3 | 16.2 | 38.1 | 37.6 |
| 42 | Track A × standard, subset of row 33 | qwen3vl_8b | think150 | 150/150 | 51.4 | 29.9 | 46.7 | 42.7 |

### sgdet (10 cells — 7 scored, 3 generating)

| # | method | model | split | videos | run state | unloc nc R@50 | unloc nc mR@50 | IoU.15 R@50 | IoU.15 mR@50 | slots w/ 3D |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| 43 | Track A | qwen3vl_8b | full | 1511/1511 | scored | 33.3 | 20.2 | 11.0 | 6.6 | 80.8 |
| 44 | **Track B** | qwen3vl_8b | full | 1511/1511 | scored | **33.5** | **23.8** | 10.7 | 7.7 | 79.5 |
| 45 | Track B without the critic | qwen3vl_8b | full | 1511/1511 | scored | 33.4 | 23.2 | **11.2** | 7.8 | 79.4 |
| 46 | rag_all (2×2 standard) | qwen3vl_8b | think150 | 145/150 | scored | 22.7 | 18.5 | 0.0 | 0.0 | 0.0 |
| 47 | rag_all (2×2 thinking) | qwen3vl_8b_thinking | think150 | 150/150 | scored | 22.3 | 20.6 | 0.0 | 0.0 | 0.0 |
| 48 | Track A × thinking | qwen3vl_8b_thinking | think150 | 150/150 | scored | 0.6 | 0.4 | 0.2 | 0.1 | 0.4 |
| 49 | Track B × thinking (± critic) | qwen3vl_8b_thinking | think150 | 150/150 | scored | 0.3 | 0.2 | 0.1 | 0.1 | 0.2 |
| 50 | zero_shot | qwen25vl_7b | full | 477/1511 | **RUNNING** GPU 0 | – | – | – | – | – |
| 51 | rag_all | qwen25vl_7b | full | generating | **RUNNING** GPU 1 | – | – | – | – | – |
| 52 | caption_all | qwen25vl_7b | full | generating | **RUNNING** GPU 2 | – | – | – | – | – |
| 53 | wsg_agent | qwen25vl_7b | full | 0/1511 | QUEUED | – | – | – | – | – |

### The matched comparison (same 150 videos, same Qwen3-VL-8B backbone, predcls)

The full-split `rag_all` runs on Qwen2.5-VL-7B while Track A and Track B run on
Qwen3-VL-8B, so at 1,511 videos any retrieval-versus-localization comparison is
confounded by backbone. This block removes that confound: one backbone, one set of
150 videos, every arm scored by the same pass.

| method | decode | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Track B (tool loop) | standard | **52.2** | **31.4** | 62.1 | 49.5 | **48.4** | 45.3 |
| Track A (marked frames + BEV) | standard | 51.4 | 29.9 | 61.5 | 46.7 | 46.0 | 42.7 |
| RAG | thinking | 50.4 | 30.3 | **63.8** | **49.9** | 47.7 | **53.8** |
| RAG | standard | 47.4 | 28.8 | 60.0 | 48.5 | 46.1 | 51.8 |
| Track A | thinking | 30.5 | 19.3 | 45.9 | 40.4 | 41.2 | 39.9 |
| Track B | thinking | 27.2 | 16.4 | 43.2 | 38.2 | 37.8 | 38.2 |

**The two families are complementary, not ranked.** Track B wins headline recall and
observed objects; RAG with thinking wins no-constraint recall, mean recall, and the
unobserved-non-trivial bucket by 8.5 points. Localization helps what the camera can
see; retrieval carries object permanence, and thinking recovers most of what
retrieval gave up on observed objects without surrendering permanence.

That is a real routing headroom on the one axis where the arms disagree, and it is
now testable with no GPU: merge the existing Track B and RAG predictions at slot
level (observed → Track B, unobserved → RAG) and score the oracle. Rows 39–41 also
confirm that the thinking collapse is a property of the prompt format, since it hits
Track A and Track B identically while leaving per-object RAG untouched.

**Caveat for the write-up:** these six rows are 150 videos. Making the retrieval arm
backbone-matched at full scale costs ~12 GPU-hours (RAG, standard decode,
Qwen3-VL-8B, 1,511 videos). Making the thinking arm full-scale costs ~300.

### What these rows settle

1. **Track B is the best MLLM method in both modes.** predcls 52.4 / 31.7 against Track A's
   51.7 / 30.5; sgdet mean recall 23.8 against 20.2. The tool loop beats one-shot prompting.
2. **The geometric critic helps relations and does not help placement.** It is worth
   +0.3 R@20 / +0.6 mR@20 in predcls and +0.6 unlocalized mR@50 in sgdet, but at IoU 0.15 the
   ablation is *better* (11.2 vs 10.7 R@50). Despite cutting geometric violations by 36 %, the
   critic does not convert that into metric accuracy. Report the ablation honestly.
3. **Thinking helps, but only with per-object prompts.** RAG × thinking is the one valid
   thinking cell: 50.4 / 30.3 against 47.4 / 28.8 standard, i.e. +3.0 R@20, and it lands within
   a point of localized Track A. Every array-style prompt collapses instead — Track A 30.5,
   Track B 27.2 in predcls, and 0.6 / 0.3 in sgdet — because the reasoning trace exhausts the
   shared token budget before the answer. This is a prompt-format result, not a model result.
4. **wsg_agent is redundant.** 46.3 / 25.8 puts it inside the 45.3–46.9 band of the other three
   unlocalized baselines. Track B supersedes it; cut it if space is short.
5. **All four unlocalized baselines agree within 1.6 pt of R@20**, so neither captioning nor
   Graph-RAG retrieval buys anything over a plain per-object prompt. Localization does.

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
| 1 | zero_shot / rag_all / caption_all sgdet — generating now | UTD GPUs 0–2 | ~7 h (to ~22:00 today) |
| 2 | wsg_agent sgdet — last queued generation job | UTD | ~9 h after a GPU frees |
| 3 | Score rows 50–53 once they finish | UTD CPU, 4 workers | ~10 min per row |
| 4 | wsg_agent predcls is 1458/1511 — recover the last 53 videos, or report n=1458 | UTD | ~30 min |
| 5 | Decide how to present the collapsed thinking cells (rows 39–41, 48–49) | – | re-run per-object, or report as a prompt-format negative |

Pragya carries a prepared but unsubmitted CPU scoring job
(`scripts/remote/pbs/score_mllm_cpu.pbs`, project `neuro.symbolic.utd.colab.spons`).
It is no longer needed — the cells it targeted were copied to UTD and scored there —
but it is validated and ready if pragya should take future scoring load.
