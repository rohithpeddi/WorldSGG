# SceneGraphVLM on the worldbbox test set, plus hallucination metrics for every MLLM row (2026-09-25)

Setup and commands: [setup/EXT_SCENEGRAPHVLM.md](../setup/EXT_SCENEGRAPHVLM.md). Branch
`exp/ext-sgvlm`, server worktree `~/CODE/Scene4Cast_sgvlm`. All numbers are percentages.
Every row is scored on **all annotated frames and all 231,639 GT slots** (45 % unobserved);
an unobserved slot the method does not name is a miss. The score JSONs are in
`~/CODE/Scene4Cast_sgvlm/results/` (`ext_sgvlm_{t150,full}.json`,
`ext_sgvlm_baselines{150,_full}_halluc.json`, `ext_prevgraph_{noctx,prevgraph}_t150.json`).

**Question** (Phase 3 of `docs/EXTERNAL_BASELINES_PLAN.md`): does a small SGG-tuned VLM that
sees its own previous graph beat our unlocalized MLLMs, and does it hallucinate less?

**Short answer.**
* On **observed** objects, yes. The 0.8B model has the best observed-object (OO) relation
  recall of any MLLM sgdet row: nc R@50 42.2 / mR@50 42.7 on the full split, against 38.7 / 27.2
  for Track B.
* On **unobserved** objects it collapses (OU-nt nc R@50 8.2, against 17–20), because it has
  no memory. That leaves it behind Track B overall (wc R@20 22.3 vs 26.7).
* It **hallucinates far less**: UOR 28 % vs 55–74 %; URR on GT pairs 21 % vs 41–50 %.
  Part of this is that it names about 1.9 objects per frame against 7–11 for the MLLM sgdet
  rows.
* The model ran at about 12 frames/s: 1.3 GPU-hours for the whole split, not the 14 budgeted.
* **Backbone-controlled ablation.** Adding the previous frame's predicted graph to our
  Qwen3-VL-8B zero_shot prompt gives **+2.2 wc R@20** (47.7 → 49.9; paired-bootstrap 95 % CI
  +1.6 to +2.8). The gain is on observed objects; OU-nt is unchanged. Previous-graph context is
  temporal smoothing, not object permanence (§5).

---

## 1. Run health

| tag | videos | frames | driver non-empty | parse rate | frames with rel rows | objects / frame (mapped, unique class) | unmapped object names |
|---|---:|---:|---:|---:|---:|---:|---|
| t150 | 150 | 5,049 | 5,049/5,049 | 100 % | 99.96 % | 1.84 | `drinking from` ×9 (a predicate emitted as an object name) |
| rest_00 | 341 | 10,769 | 10,769/10,769 | 100 % | 100 % | 1.88 | – |
| rest_01 | 340 | 10,561 | 10,561/10,561 | 100 % | 100 % | 1.87 | – |
| rest_02 | 340 | 10,798 | 10,798/10,798 | 100 % | 100 % | 1.81 | `shag`, `brefrigerator`, `shurm` (1 each) |
| rest_03 | 340 | 11,657 | 11,657/11,657 | 100 % | 99.99 % | 1.90 | `cset/cabinet`, `cag` (1 each) |
| **total** | **1,511** | **48,834** | all | 100 % | – | 1.86 | 14 of 139,877 raw object rows (0.01 %) |

* No synonym was needed: every mapped name was an exact AG class name.
* No predicate fell outside the 26-label vocabulary, and none landed in the wrong head.
* 1,511 PKLs = split size. Predicted pairs / GT pairs = 90,970 / 231,639 (39 %). Of those
  predicted pairs, 65,388 are GT pairs.
* The 1,511 worldbbox test videos are all in AG's `test` set (`metadata.set`), so the AG
  checkpoint never trained on them.

**Reproduction check.** On t150 the authors' own evaluator
(`metrics/sgbench/eval_classical_metrics_extended.py`, sgdet, IoU 0.5, observed GT) gives
with-constraint **R@50 26.0 / P@50 38.5**. Their paper reports R@50 27.84 / P@50 38.74 on the
AG test split.

---

## 2. 150-video set (matched to the RAG × thinking 2×2; Qwen3-VL-8B unless noted)

Predcls rows give the model the frame's GT objects. SceneGraphVLM has no predcls mode (§4),
so it is compared with the **sgdet (unloc)** rows, where every method proposes its own
objects and matching is class-only.

| run | mode | videos | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc R@50 | OO nc mR@50 | OU-nt nc R@50 | OU-nt nc mR@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_shot (qwen25vl_7b) | predcls | 150 | 45.8 | 25.2 | 59.3 | 42.2 | 54.3 | 39.7 | 38.1 | 47.5 |
| zero_shot (= `prev_graph` noctx arm) | predcls | 150 | 47.7 | 30.5 | 60.5 | 49.2 | 60.3 | 47.6 | 39.0 | 53.5 |
| RAG std | predcls | 150 | 47.4 | 28.8 | 60.0 | 48.5 | 59.2 | 46.1 | 38.2 | 51.8 |
| RAG × thinking | predcls | 150 | 50.4 | 30.3 | 63.8 | 49.9 | 63.7 | 47.7 | 42.6 | 53.8 |
| Track A | predcls | 150 | 51.4 | 29.9 | 61.5 | 46.7 | 56.5 | 46.0 | 29.0 | 42.7 |
| Track B | predcls | 150 | 52.2 | 31.4 | 62.1 | 49.5 | 57.8 | 48.4 | 29.9 | 45.3 |
| zero_shot (qwen25vl_7b) | sgdet (unloc) | 150 | 19.7 | 11.4 | 26.0 | 18.6 | 27.4 | 22.4 | 16.1 | 13.9 |
| RAG std | sgdet (unloc) | 145* | 18.1 | 12.5 | 22.7 | 18.5 | 25.5 | 19.1 | 13.4 | 16.4 |
| RAG × thinking | sgdet (unloc) | 150 | 17.8 | 11.6 | 22.3 | 20.6 | 26.2 | 21.0 | 13.0 | 16.0 |
| Track A | sgdet (unloc) | 150 | 24.0 | 13.4 | 33.4 | 21.7 | 38.3 | 25.2 | 19.2 | 14.2 |
| Track B | sgdet (unloc) | 150 | **27.4** | **17.6** | **33.7** | 25.8 | 39.0 | 30.2 | 17.2 | 12.3 |
| **SceneGraphVLM** (Qwen3.5-0.8B) | sgdet (unloc) | 150 | 21.6 | 13.7 | 27.3 | **30.5** | **40.3** | **38.1** | 7.5 | 12.7 |

\* 5 videos have no output ("no objects after sgdet filtering"). They are left out, not
scored as zero.

SceneGraphVLM, observed-only 2D SGDet (`sgdet2d`: class + predicate + 2D IoU ≥ 0.5 on both
endpoints, GT = OO pairs): wc R@20 **26.0** / mR@20 15.0; nc R@50 **32.4** / mR@50 29.3.
No other MLLM row emits 2D boxes, so every other row scores 0 here by design. The
with-constraint figure (26.0) equals the authors' R@50 above, as it should: both use the same
matching on the same frames.

## 3. Full split (1,511 videos, 48,834 frames)

| run | mode | videos | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc R@50 | OO nc mR@50 | OU-nt nc R@50 | OU-nt nc mR@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_shot (qwen25vl_7b) | predcls | 1511 | 46.9 | 25.1 | 60.9 | 43.1 | 55.4 | 40.3 | 41.4 | 42.8 |
| rag_all (qwen25vl_7b) | predcls | 1511 | 46.9 | 26.0 | 61.1 | 43.8 | 56.8 | 40.8 | 41.9 | 44.6 |
| Track A | predcls | 1511 | 51.7 | 30.5 | 62.8 | 47.2 | 58.0 | 46.5 | 30.8 | 40.0 |
| Track B | predcls | 1511 | 52.4 | 31.7 | 63.4 | 48.4 | 59.3 | 47.6 | 31.2 | 40.8 |
| zero_shot (qwen25vl_7b) | sgdet (unloc) | 1501* | 19.6 | 11.1 | 26.6 | 18.8 | 27.0 | 20.4 | 18.1 | 14.7 |
| rag_all (qwen25vl_7b) | sgdet (unloc) | 1480* | 20.4 | 11.6 | 29.2 | 20.9 | 29.7 | 22.0 | **20.3** | **18.8** |
| Track A | sgdet (unloc) | 1511 | 23.6 | 12.2 | 33.3 | 20.2 | 38.2 | 23.2 | 17.9 | 12.4 |
| Track B | sgdet (unloc) | 1511 | **26.7** | 16.8 | **33.5** | 23.8 | 38.7 | 27.2 | 16.9 | 13.3 |
| **SceneGraphVLM** | sgdet (unloc) | 1511 | 22.3 | **16.9** | 28.7 | **33.1** | **42.2** | **42.7** | 8.2 | 12.7 |

\* Videos without an output are left out of the row.

SceneGraphVLM `sgdet2d` (OO, 2D IoU ≥ 0.5): wc R@20 **28.1** / mR@20 19.3; nc R@50 **35.3** /
mR@50 35.3.

Bucket detail (SceneGraphVLM, full, wc@20): OO R 33.2 / mR 22.1; OU R 5.5 / mR 6.1; OU-nt R 5.6
/ mR 6.2.
* The OU credit is not zero because slots are keyed by class. When the model names a class
  that the annotation marks unobserved in that frame, the triplet can still match.

**Caveat on the nc columns.** "No constraint" ranks every (pair, predicate) cell, including the
zero-score ones. That is the stock rule, applied to every row. SceneGraphVLM emits about 1.9
pairs per frame, so its top 50 is mostly zero-score cells in index order. Those cells add
coverage of rare predicates and inflate nc **mR** (33.1 > nc R 28.7). Its wc mR@20 (16.9) ties
Track B's (16.8). Under the constraint the OO lead survives but shrinks (full split, sgdet
unloc, wc@20):

| run | OO R | OO mR | OU-nt R | OU-nt mR | objects / frame | triplets / frame |
|---|---:|---:|---:|---:|---:|---:|
| zero_shot (qwen25vl_7b) | 20.0 | 12.0 | 13.2 | 7.9 | 9.41 | 28.0 |
| rag_all (qwen25vl_7b) | 20.8 | 12.1 | 13.1 | 10.5 | 11.0 | 32.2 |
| Track A | 26.6 | 13.9 | 10.8 | 6.2 | 7.05 | 19.2 |
| Track B | 30.8 | 19.4 | 12.0 | 8.0 | 6.74 | 18.3 |
| **SceneGraphVLM** | **33.2** | **22.1** | 5.6 | 6.2 | 1.86 | 6.0 |

## 4. Hallucination (UOR / URR)

Definitions are in `lib/mllm/eval/hallucination.py`.
* **UOR** = predicted (frame, class) objects whose class is absent from the video's world
  inventory (all frames, observed or not) / predicted objects.
* **URR** = triplets whose (person, object) pair is in the frame's GT but whose predicate is not
  a GT predicate of that pair / all predicted triplets.
  * The "pairs in GT" column uses only those triplets as denominator.
  * The obs / unobs columns split them by the GT object's visibility.

AG is incomplete, so both rates are upper bounds; the WS4 human audit (200 per method) is
still to do. UOR is 0 by construction in predcls, where the object list is given.

**150-video set**

| run | mode | UOR | URR (all triplets) | URR pairs-in-GT | URR obs | URR unobs | objects / frame | triplets / frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| zero_shot (qwen25vl_7b) | predcls | 0.0 | 47.5 | 47.7 | 50.1 | 44.8 | 5.10 | 15.3 |
| zero_shot (qwen3vl_8b, noctx) | predcls | 0.0 | 44.3 | 44.4 | 41.7 | 47.8 | 5.10 | 14.7 |
| RAG std | predcls | 0.0 | 43.8 | 43.9 | 41.7 | 46.5 | 5.10 | 14.5 |
| RAG × thinking | predcls | 0.0 | 41.7 | 41.9 | 38.8 | 45.5 | 5.10 | 15.4 |
| Track A | predcls | 0.0 | 40.4 | 40.4 | 45.8 | 34.1 | 5.08 | 15.3 |
| Track B | predcls | 0.0 | 39.7 | 39.7 | 44.2 | 34.4 | 5.08 | 15.3 |
| zero_shot (qwen25vl_7b) | sgdet | 72.3 | 13.4 | 48.1 | 48.6 | 47.4 | 9.25 | 27.6 |
| RAG std | sgdet | 60.2 | 18.6 | 43.8 | 41.7 | 46.9 | 4.55 | 12.9 |
| RAG × thinking | sgdet | 52.8 | 18.0 | 37.6 | 34.8 | 41.6 | 3.45 | 10.3 |
| Track A | sgdet | 53.5 | 24.9 | 49.9 | 52.6 | 44.3 | 7.16 | 19.6 |
| Track B | sgdet | 52.6 | 23.6 | 46.7 | 49.7 | 40.5 | 6.90 | 18.9 |
| **SceneGraphVLM** | sgdet | **26.4** | 15.4 | **20.6** | **17.7** | **36.3** | 1.85 | 6.0 |

**Full split**

| run | mode | UOR | URR (all) | URR pairs-in-GT | URR obs | URR unobs |
|---|---|---:|---:|---:|---:|---:|
| zero_shot (qwen25vl_7b) | predcls | 0.0 | 47.2 | 47.3 | 50.2 | 43.8 |
| rag_all (qwen25vl_7b) | predcls | 0.0 | 47.0 | 47.1 | 48.3 | 45.7 |
| Track A | predcls | 0.0 | 41.6 | 41.6 | 45.9 | 36.3 |
| Track B | predcls | 0.0 | 40.9 | 40.9 | 44.4 | 36.6 |
| zero_shot (qwen25vl_7b) | sgdet | 73.6 | 12.9 | 48.1 | 49.9 | 45.6 |
| rag_all (qwen25vl_7b) | sgdet | 74.4 | 12.6 | 48.2 | 48.3 | 47.9 |
| Track A | sgdet | 56.4 | 23.8 | 50.5 | 52.9 | 45.3 |
| Track B | sgdet | 55.3 | 22.8 | 47.5 | 50.0 | 42.0 |
| **SceneGraphVLM** | sgdet | **28.1** | 15.0 | **20.7** | **17.8** | **36.9** |

How to read it:

* **URR "all triplets" misleads for sgdet rows.** Most sgdet triplets sit on objects that are
  not in the frame's GT; those are UOR's problem, not URR's. The denominator then shrinks
  URR: zero_shot sgdet shows 13 % because 72 % of its objects are invented. **Report
  "pairs in GT" as the URR headline**, with UOR beside it.
* **SceneGraphVLM halves both rates.** UOR is 28 % against 55–74 %. URR on GT pairs is 21 %
  against 38–50 % for every Qwen2.5/3-VL row, predcls rows included, which are given the true
  object list. On observed objects the gap is largest: 17.8 % against 44–53 % (full split).
* **Two confounds, both real.**
  1. It emits far less: about 1.9 objects and 6 triplets per frame, against 7–11 objects and
     18–32 triplets for the Track A/B, zero_shot and rag_all sgdet rows (full split). A precision-oriented emitter will
     look good on unsupported-rate metrics.
  2. It was SFT + GRPO-trained on AG train labels, so it has learned AG's labelling habits,
     for example which spatial label AG annotators use. "Unsupported by AG" partly measures
     agreement with the annotation style, not only hallucination.

  This is why the human audit matters before the paper calls it a hallucination result.
* **URR on unobserved objects is 37 % for SceneGraphVLM, against 18 % on observed ones.** When it
  names a class the annotation marks as out of view, its relations are usually wrong for the
  annotated (occluded) state.
* **Thinking lowers URR on the MLLM side.** RAG × thinking vs RAG std on the 150 set: sgdet UOR
  52.8 vs 60.2 and URR-GT 37.6 vs 43.8. The thinking model also proposes fewer objects
  (3.45 vs 4.55 per frame).

## 5. Backbone-controlled ablation: previous-frame graph in our prompt

The backbone is held fixed: Qwen3-VL-8B, the vendored zero_shot per-object unlocalized
prompt, predcls, `--skip-verification`, T = 0.2 (same settings as the RAG std cell), 150-video set.
The two arms differ only by one prompt block: the previous annotated frame's predicted graph, one
line per object ("attention; contacting; spatial"), capped at 600 characters. The measured block is
224 characters on average (max 477, i.e. ~60-120 tokens); frame 0 gets "(no previous frame)".
The previous graph is the `noctx` arm's own prediction for frame t-1 (two-pass, see §6).
Runner: `lib/mllm/methods/prev_graph/runner.py`; each arm took ~47-55 GPU-min.

| arm | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc R@50 | OO nc mR@50 | OU-nt nc R@50 | OU-nt nc mR@50 | URR pairs-in-GT | URR obs | URR unobs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noctx | 47.7 | 30.5 | 60.5 | 49.2 | 60.3 | 47.6 | 39.0 | 53.5 | 44.4 | 41.7 | 47.8 |
| prevgraph | **49.9** | **30.9** | **62.7** | **50.7** | **61.9** | **50.8** | 39.6 | 52.3 | 43.4 | 42.0 | 45.2 |
| delta | +2.2 | +0.4 | +2.2 | +1.5 | +1.6 | +3.2 | +0.6 | -1.2 | -1.0 | +0.3 | -2.6 |

Paired video bootstrap (2,000 resamples of the 150 videos, frame-weighted): wc R@20 +2.2
[95 % CI +1.6, +2.8], 73 % of videos improve; nc R@50 +2.2 [+1.6, +2.9], 63 % improve. This CI covers
video sampling only, not decoding noise at T = 0.2 (one run per arm).

Reading: with the backbone fixed, the preceding-frame graph is worth about +2 R points, which is
more than half of Track A's gain over the same Qwen3-VL-8B zero_shot arm on the 150 set (51.4 vs 47.7 wc R@20) and
larger than what Graph-RAG retrieval bought (RAG std 47.4 vs noctx 47.7). The gain sits on
**observed** objects (OO nc mR +3.2); unobserved non-trivial recall does not move (OU-nt R +0.6,
mR -1.2). A one-frame-back graph is temporal smoothing, not object permanence: it helps say the same
thing about what is still in view, and it does not recover what has left the view. That matches the
SceneGraphVLM result above: strong OO, weak OU-nt.

## 6. Deviations and caveats

* **Environment.** The conda env `sgvlm` lives at `/data3/datasets/OakInk_pilot/conda_envs/sgvlm`
  because the user's `.condarc` `envs_dirs` puts it there. It is not in `~/anaconda3/envs`.
  `scene4cast` and `wsg` are untouched.
* **ms-swift 4.5.3 instead of the authors' container** (a 9.8 GB Docker image; there is no
  Docker on the server path we use). This needed three shims:
  * `model_type` pinned to `qwen3_5` in `launch.py`;
  * `--template-type qwen3_5`;
  * the PyTorch sampler, because FlashInfer JIT needs `nvcc` ≥ 12.8 and the server has 12.4.

  The reproduction check (R@50 26.0 vs 27.84 published; P@50 38.5 vs 38.74) says the port is
  faithful. The remaining gap is plausibly the frame set. We use every annotated frame; their
  cleaned test JSONL drops frames without relations.
* **Batch size 256 / 384 instead of 64.** It changes throughput only; decoding is greedy.
* **No predcls mode.** The released model cannot be conditioned on the current frame's object
  list, so it has no predcls row.
* **Previous-graph chain = annotated frames**, as in their AG test JSONL. The chain restarts
  at frame 0 of each video.
* **The 150-set zero_shot baseline in the 2×2 is qwen25vl_7b**, because no Qwen3-VL-8B
  zero_shot run existed. The ablation's `noctx` arm supplies a Qwen3-VL-8B zero_shot predcls
  row (T=0.2, `--skip-verification`).
* **`sgdet2d` keeps one box per class per frame** (the first instance), the same
  label-keyed slot rule as every other regime. Duplicate instances merge their predicates.
* **rag_all / zero_shot sgdet rows have missing videos** (145/150; 1501 and 1480 of 1511).
  The missing videos are left out of those rows, not scored as zero.
* **Ablation is two-pass, not autoregressive.** The `prevgraph` arm reads the `noctx` arm's
  frame t-1 prediction instead of its own. The vendored runner batches all frames of a video
  in one call; a true chain would need per-frame sequential calls, about 30x the wall time. The
  block holds the same information SceneGraphVLM's `--prev-source model` carries (a model graph of
  t-1), without error compounding through the chain. It is also not padded to a fixed length: the
  `noctx` arm has no block at all, and the cap (600 characters) bounds the added context.
* **Time.** Inference took 1.3 GPU-hours on the A40 (about 12 frames/s at batch 384), not
  the budgeted 14. Frame resizing and JSONL building took about 20 CPU-minutes.
