# Track 2 — the unlocalized MLLM track

Training-free MLLM methods that emit **class-only** scene graphs: a person, a
set of named objects, and person→object predicates. No method in this track
emits a 3D box, so the "slots with 3D" column is 0 by construction and SGDet
can only be matched by class.

These are the rows that answer whether a prompted model can do the task at all.

Authoritative results: [analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md)
§2 · run status: [analysis/mllm_tracks_status_2026-09-22.md](../analysis/mllm_tracks_status_2026-09-22.md)

| Track | Document |
|---|---|
| 1 — Training-based | [TRACK1_TRAINING.md](TRACK1_TRAINING.md) |
| **2 — Unlocalized MLLM** | this file |
| 3 — Localized MLLM | [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) |

---

## 1. Methods in this track

A nested ladder of evidence assembly. Each rung adds one source of context to
the same per-object relationship prompt.

```
zero_shot     frames only                                  (no context)
caption_all   + per-clip video captions                    (text context)
rag_all       + retrieved scene-graph nodes per query       (Graph-RAG)
```

| Method key | Document | What it adds | Runner |
|---|---|---|---|
| `zero_shot` | [MLLM_ZERO_SHOT.md](MLLM_ZERO_SHOT.md) | nothing — frames + prompt | [lib/mllm/methods/zero_shot/runner.py](../lib/mllm/methods/zero_shot/runner.py) |
| `caption_all` | [MLLM_CAPTION_ALL.md](MLLM_CAPTION_ALL.md) | Stage-1 clip captions in the prompt prefix | [lib/mllm/methods/caption_all/runner.py](../lib/mllm/methods/caption_all/runner.py) |
| `rag_all` | [MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) | Graph-RAG: embedding retrieval over the Stage-1 video graph | [lib/mllm/methods/rag_all/runner.py](../lib/mllm/methods/rag_all/runner.py) |

Shared prerequisite: the **Stage-1 graph/caption build**
([lib/mllm/methods/graphs/runner.py](../lib/mllm/methods/graphs/runner.py)),
which segments a video into clips and asks the VLM for a per-clip caption plus a
JSON `{entities, actions, scenes}` graph. `caption_all` and `rag_all` read its
per-video pickle; `zero_shot` does not. It is documented in
[MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) §2.

All three share one base class,
[lib/mllm/base_processor.py](../lib/mllm/base_processor.py), which owns the
frame/clip loading, the SGDet object-estimation step, the response parser and
the Yes/No verification pass that turns categorical generations into the scored
distribution R@K needs.

> **Naming discrepancy — read before quoting the tables.**
> [analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) labels the
> `zero_shot` rows "Graph-RAG (zero_shot)". The code does not support that
> label: `ActionGenomeZeroShotProcessor` never loads the graph pickle and its
> docstring is explicit — "No caption or graph context is injected". Graph-RAG
> is `rag_all`. The **numbers** are correctly attributed to the `zero_shot`
> runs; only the display name is wrong. Fix it before submission.

---

## 2. Evaluation protocol

Scored by [lib/mllm/eval/score_run.py](../lib/mllm/eval/score_run.py) over the
locked split `/data3/rohith/ag/splits/test_worldbbox_1511.txt`. Protocol detail:
[lib/mllm/eval/README.md](../lib/mllm/eval/README.md).

| | |
|---|---|
| Test set | 1,511 videos / 48,834 frames, all annotated frames |
| Adapter | [lib/mllm/eval/dump_adapter.py](../lib/mllm/eval/dump_adapter.py) — converts a run's PKLs into the exact `tools/dump_predictions.py` record format |
| PredCls regime | `wsgg`: the **stock** `evaluate_wsgg_video` + `evaluate_wsgg_video_bucketed`, identical to track 1 |
| SGDet regime | `loc3d` at τ = 0 ("unloc"): class equality only, restricted to pairs the model emitted (`pred_pair_valid`) |
| Legacy regime | `legacy`: box-free per-predicate-group P/R/F1 ([lib/mllm/eval/legacy_f1.py](../lib/mllm/eval/legacy_f1.py)) |
| Missing videos | reported as `n_missing`, never dropped silently |

Three details that shape how the numbers read:

- **Unobserved objects get a full-frame placeholder box** for both GT and
  prediction, so PredCls IoU is 1 and they are scored on relations only. This
  is the WorldAG rule, applied identically in track 1.
- **Scores come from verification, not generation.** A predicted label carries
  its `yes_prob` from a single-token logprob; every other predicate is 0. With
  `--skip-verification` every predicted label gets 1.0, which makes the
  no-constraint ordering among pairs arbitrary-but-deterministic
  (`argsort_desc`). Prefer R@50 / R@100 for MLLM rows.
- **mR divides by all 26 predicates** (stock behaviour), so predicates an MLLM
  never emits count as 0. This is why MLLM mean recall is structurally low.

MLLM SGDet is **never** matched by 2D box. Methods in this track have
`has_pred_3d = False` and score 0 at any τ > 0 by design; they are comparable
only under `unloc`.

---

## 3. Results

### 3a. PredCls — full split, 1,511 videos, Qwen2.5-VL-7B

| method | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 | legacy uF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | **46.9** | 25.1 | 60.9 | 43.1 | 40.3 | 42.8 | – |
| `caption_all` | 45.3 | **26.1** | 59.9 | 43.6 | 40.6 | 44.1 | 48.2 |
| `rag_all` | 46.9 | 26.0 | **61.1** | **43.8** | **40.8** | **44.6** | **49.9** |

### 3b. PredCls — thinking cells, 150-video subset, Qwen3-VL-8B

| method | decode | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| RAG | standard | 47.4 | 28.8 | 60.0 | 48.5 | 46.1 | 51.8 |
| **RAG** | **thinking** | **50.4** | **30.3** | **63.8** | **49.9** | **47.7** | **53.8** |

### 3c. SGDet

| method | model | split | videos | unloc nc R@50 | unloc nc mR@50 | slots w/ 3D |
|---|---|---|---:|---:|---:|---:|
| RAG, standard | qwen3vl_8b | think150 | 145/150 | 22.7 | 18.5 | 0.0 |
| RAG, thinking | qwen3vl_8b_thinking | think150 | 150/150 | 22.3 | **20.6** | 0.0 |
| `zero_shot` | qwen25vl_7b | full | 1298/1511 | **not scored — generating** | | |
| `caption_all` | qwen25vl_7b | full | 634/1511 | **not scored — generating** | | |
| `rag_all` | qwen25vl_7b | full | 578/1511 | **not scored — generating** | | |

The three full-split unlocalized SGDet cells were still generating when
[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) was written
(2026-09-22). Those cells have **no number**; write "not run" rather than
interpolating from the 150-video rows.

### 3d. Legacy-protocol backbone breadth

A separate metric family — per-predicate-group precision / recall / F1 with
micro-F1 (μF1) and macro-F1 (MF1) — run across six backbones on the pragya
cluster. Same Action Genome test split and same GT + corrections annotation
basis as the rest of this document. The pragya rows evaluate 1,734 videos and
the WorldBBox rows 1,511; the 1,511 are a **strict subset** of the 1,734
(verified: zero WorldBBox videos absent from the pragya set), so the rows sit on
one axis and the tables combine.

**PredCls — F1 per predicate group**

| Model | Method | videos | Attention F1 | Contacting F1 | Spatial F1 | μF1 | MF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| InternVL2.5-8B | RAG | 1734 | 41.8 | 34.7 | 23.4 | 32.9 | 18.8 |
| InternVL2.5-8B | Caption | 1734 | 38.9 | 33.2 | 17.0 | 29.3 | 17.7 |
| KeyeVL-1.5-8B | RAG | 1734 | 38.3 | 34.6 | 49.2 | 41.0 | 24.4 |
| KeyeVL-1.5-8B | Caption | 1734 | 26.0 | 21.5 | 28.9 | 25.5 | 16.9 |
| MiniCPM-V4.5 | RAG | 1734 | 42.0 | 41.0 | 32.6 | 38.6 | 24.5 |
| MiniCPM-V4.5 | Caption | 1734 | 40.6 | 41.1 | 32.3 | 38.1 | 24.4 |
| Ovis2.5-9B | RAG | 1734 | 43.9 | 46.3 | **56.0** | 48.9 | 27.9 |
| Ovis2.5-9B | Caption | 1734 | 43.4 | 46.4 | 55.8 | 48.7 | 28.3 |
| Qwen2.5-VL-7B | RAG | 1734 | 52.8 | 47.8 | 49.0 | 49.8 | 23.3 |
| Qwen2.5-VL-7B | Caption | 1734 | 49.3 | 45.1 | 47.4 | 47.2 | 21.8 |
| Qwen3-VL-30B-A3B | RAG | 1734 | 55.7 | 45.3 | 35.1 | 44.5 | 28.6 |
| Qwen3-VL-30B-A3B | Caption | 1734 | **62.1** | **51.2** | 32.8 | 47.8 | **29.2** |
| Qwen2.5-VL-7B | `zero_shot` | 1511 | 55.8 | 48.3 | 45.9 | 49.8 | 23.2 |
| Qwen2.5-VL-7B | `rag_all` | 1511 | 53.6 | 49.7 | 46.7 | **49.9** | 24.2 |
| Qwen2.5-VL-7B | `caption_all` | 1511 | 53.2 | 47.3 | 44.7 | 48.3 | 23.2 |
| Qwen3-VL-8B | `rag_all` | 150 | 63.0 | 46.0 | 45.9 | 51.4 | 28.4 |

**SGDet — F1 per predicate group**

| Model | Method | videos | Attention F1 | Contacting F1 | Spatial F1 | μF1 | MF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| InternVL2.5-8B | RAG | 1734 | 29.0 | 30.2 | 14.1 | 24.2 | 11.0 |
| InternVL2.5-8B | Caption | 1734 | 29.6 | 28.5 | 14.1 | 23.8 | 9.8 |
| KeyeVL-1.5-8B | RAG | 1734 | **40.6** | 34.0 | 37.2 | 37.2 | 18.1 |
| KeyeVL-1.5-8B | Caption | 1734 | 20.4 | 14.7 | 22.7 | 19.2 | 9.9 |
| MiniCPM-V4.5 | RAG | 1734 | 33.1 | 27.8 | 28.8 | 29.6 | 16.4 |
| MiniCPM-V4.5 | Caption | 1734 | 31.4 | 25.3 | 26.4 | 27.4 | 15.4 |
| Ovis2.5-9B | RAG | 1734 | 32.3 | 37.3 | 33.8 | 34.5 | 13.5 |
| Ovis2.5-9B | Caption | 1734 | 33.2 | 40.3 | 33.1 | 35.6 | 13.9 |
| Qwen2.5-VL-7B | RAG | 1734 | 35.5 | **50.3** | 44.6 | **43.7** | 19.9 |
| Qwen2.5-VL-7B | Caption | 1734 | 34.2 | 42.4 | **50.4** | 42.7 | **20.5** |
| Qwen3-VL-8B | `rag_all` | 145 | 32.2 | 25.7 | 24.2 | 27.1 | 17.0 |

Sources: pragya rows from
`WorldSceneGraphAnnotationTool/assets/tex_files/tables/*_gt_plus_corrections_detailed.tex`
(generated May 2026; "Caption" = the `subtitle_all` pipeline, "RAG" = `rag_all`);
WorldBBox rows from `results/mllm_worldbbox_*.json`, key `legacy.gt_plus_corrections`.

**Incomplete cells, not tabulated:** Qwen3-VL-30B-A3B has PredCls only
(1,542 videos, no SGDet) and LLaVA-OneVision-7B never finished (19 PredCls /
147 SGDet). A corrections-only variant of both tables exists in the same
directory and tells the same story at uniformly lower absolute values.

**How much the 223 extra videos matter.** Qwen2.5-VL-7B with retrieval is the
one cell measured on both video sets:

| metric | 1,734 videos | 1,511 videos | delta |
|---|---:|---:|---:|
| μF1 | 49.8 | 49.9 | +0.1 |
| MF1 | 23.3 | 24.2 | +0.9 |
| Attention F1 | 52.8 | 53.6 | +0.8 |
| Contacting F1 | 47.8 | 49.7 | +1.9 |
| Spatial F1 | 49.0 | 46.7 | −2.3 |

Micro-F1 is stable to 0.1 across the two sets; per-group F1 moves by up to 2.3.
Compare backbones on μF1 freely; treat a per-group comparison across different
video counts as approximate.

---

## 4. Findings, ranked

1. **All four ladder rungs agree within 1.6 points of wc R@20.** 45.3 to 46.9.
   Neither captioning nor Graph-RAG retrieval buys anything over a plain
   per-object prompt on the full split. The ladder's premise — that richer
   evidence assembly helps — is not supported at this scale.
2. **Thinking is worth +3.0 wc R@20 on top of retrieval** (47.4 → 50.4) and
   this is the only valid thinking cell in the project. See §5.2.
3. **Retrieval beats captioning across backbones, but the margin is small and
   model-dependent.** RAG wins μF1 for five of six backbones in PredCls and
   four of five in SGDet, typically by 0.5 to 3.6 points.
4. **The one large retrieval gain is a captioning failure, not a retrieval
   success.** KeyeVL gains 15.5 μF1 in PredCls and 18.0 in SGDet purely
   because its captioning *recall* collapses — its captioning precision is
   43.7 / 37.6 / 55.8 against recall 18.5 / 15.0 / 19.5. The captioning path
   is not inaccurate, it is silent. Quote this as robustness for RAG, not as
   evidence that retrieval carries information.
5. **Two backbones invert the ordering.** Qwen3-VL prefers captioning in
   PredCls (48.7 vs 44.5 μF1) and Ovis2.5 prefers it in SGDet (35.6 vs 34.5).
   No method dominates across backbones.
6. **Spatial predicates separate the backbones far more than attention does.**
   Spatial F1 spans 23.4–56.0 under RAG in PredCls while attention spans
   38.3–55.7. The unlocalized setting is mostly measuring spatial competence —
   which is exactly what track 3 attacks and where it collapses at IoU 0.15.
7. **Rank order is not preserved between modes.** Ovis2.5 leads PredCls μF1
   among the non-Qwen backbones (48.9) and falls to 34.5 in SGDet. Single-mode
   backbone claims will not survive review.
8. **Backbone scale is not the story.** Qwen3-VL-30B-A3B, the largest model
   here, wins attention F1 by a wide margin (62.1) yet lands mid-table on μF1
   (47.8) because its spatial F1 is among the worst (32.8).

---

## 5. Protocol caveats

1. **The backbone confound.** Full-split rows in this track run
   **Qwen2.5-VL-7B**; track 3 runs **Qwen3-VL-8B**. Any track-2-versus-track-3
   claim on the full split is confounded by backbone. The only backbone-matched
   comparison is the 150-video block on `splits/test_worldbbox_thinking150.txt`
   — see [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) §5. Closing this properly
   costs ~12 h GPU (RAG, standard decode, Qwen3-VL-8B, full split) and is the
   cheapest high-value run left.
2. **The thinking-budget defect.** The reasoning trace shares `max_new_tokens`
   with the answer, so a prompt that asks for **one array covering every object
   in the frame** never reaches its answer. The closing `</think>` appears in
   46.2 % of Track A PredCls responses and 0.7 % in SGDet. **RAG is immune
   because it prompts per object** — each answer is a three-field JSON object,
   not an array. This is why §3b is the only valid thinking cell in the
   project: it is a prompt-format result, not a model result.
3. **Video counts differ by row.** Always print the videos column. Thinking
   cells are 150 videos because thinking costs 255 s per video for Track A and
   715 s for RAG, against 30 s for standard decode.
4. **The metric-family split.** §3d is per-predicate-group P/R/F1 with micro
   and macro F1; §3a–3c are R@K and mR@K. They share the test videos and the
   annotation basis but **not the metric**. The bridge is the `legacy uF1`
   column in §3a, which is the same micro-F1. Never merge a μF1 column and an
   R@K column into one table.
5. **Completion must be measured by PKL count against the split, never by exit
   code.** `ActionGenomeBaseProcessor.run()` wraps each video in try/except and
   continues, so a run can exit `rc = 0` having written fewer PKLs than
   requested. This has recurred across methods.

---

## 6. What is still missing in this track

| gap | cost |
|---|---|
| Three unlocalized SGDet baselines at full split | ~7 h GPU, scored automatically |
| RAG, standard decode, Qwen3-VL-8B, full split — makes the track-2 vs track-3 comparison backbone-matched at 1,511 instead of 150 | ~12 h GPU |
| Thinking at full split | ~300 h GPU — not worth it before the deadline |
| Re-score the pragya backbone cells on the 1,511-video list so §3d is exact at the per-group level | CPU only, PBS job on pragya |
