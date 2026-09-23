# WorldSGG — results by track (ICLR submission view, 2026-09-22)

Test set for every row: `world4d_rel_annotations_worldbbox/test`, 1,511 videos /
48,834 frames, `annotation_version = 40decd1785af3c3470290be4b67443e1`.
All rows are **all frames**, best epoch where training applies. Values in %.

Three tracks:

| track | what it is | localization | sgdet matching |
|---|---|---|---|
| **1 — Training-based** | WorldWise lineage and the supervised baselines | learned 3D boxes | 2D IoU 0.5 |
| **2 — Unlocalized MLLM** | Graph-RAG (rag_all), captions, frames-only prompting, thinking | none — emits no 3D boxes | class-only |
| **3 — Localized MLLM** | marked frames + BEV, tool loop with geometric critic | predicted oriented boxes | 3D IoU 0.15 / 0.25 |

> **Read the protocol note in section 5 before putting any two tracks in one table.**
> PredCls is directly comparable across all three tracks — the same stock
> WorldSGG evaluator scores every row, and ground-truth boxes are supplied.
> SGDet is **not**: track 1 matches in 2D at IoU 0.5, track 3 matches in 3D,
> and track 2 cannot match at all.

---

## 0. Cross-track headline — PredCls, all frames, 1,511 videos

The one table where the tracks may be compared directly.

| track | best method | wc R@20 | wc mR@20 |
|---|---|---:|---:|
| 1 — Training-based | WorldWise++ @ dinov3 | **74.8** | 54.4 |
| 1 — Training-based | WorldWise+ @ dinov3tok | 73.5 | **54.7** |
| 1 — best prior baseline | W-DSGDetr++ @ resnet50 | 68.5 | 38.9 |
| 3 — Localized MLLM | Track B (tool loop + critic) | 52.4 | 31.7 |
| 3 — Localized MLLM | Track A (marked frames + BEV) | 51.7 | 30.5 |
| 2 — Unlocalized MLLM | zero_shot (frames only) | 46.9 | 25.1 |

The supervised ceiling sits **22.4 points of R@20 above the best MLLM**, and the
localized MLLM sits **5.5 points above the unlocalized one**. Both gaps carry the
argument: the task is not solved by prompting, and localization is what moves a
prompted model.

---

## 1. Training-based track (28 cells, all trained and scored)

### 1a. PredCls

| method | backbone | wc R@20 | wc mR@20 | nc R@20 | nc mR@20 | OO R@20 | OU-nt R@20 | OU-nt mR@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W-STTran | resnet50 | 68.2 | 37.6 | 92.6 | 70.0 | 91.2 | 56.9 | 30.7 |
| W-STTran++ | resnet50 | 66.9 | 34.0 | 92.4 | 63.6 | 90.3 | 67.0 | 29.1 |
| W-DSGDetr | resnet50 | 67.5 | 33.4 | 92.2 | 65.6 | 90.1 | 62.7 | 27.0 |
| W-DSGDetr++ | resnet50 | 68.5 | 38.9 | 92.7 | 71.0 | 91.4 | 55.3 | 26.1 |
| W-USG | resnet50 | 67.3 | 33.4 | 92.6 | 64.0 | 90.6 | 67.5 | 29.1 |
| WorldWise | resnet50 | 68.0 | 49.1 | 92.5 | 81.8 | 90.6 | 73.5 | 40.9 |
| WorldWise | dinov2b | 68.2 | 45.4 | 92.5 | 79.4 | 90.6 | 74.7 | 38.7 |
| WorldWise | dinov2l | 68.8 | 47.8 | 92.6 | 81.6 | 90.6 | 74.2 | 42.6 |
| WorldWise | dinov3l | 69.0 | 49.6 | 92.6 | 82.4 | 90.9 | 72.9 | 41.1 |
| WorldWise+ | dinov3tok | 73.5 | **54.7** | 94.3 | 85.6 | 93.4 | 76.0 | 45.2 |
| WorldWise+ | pi3tok | 71.9 | 51.5 | 93.8 | 84.4 | 92.4 | 75.4 | 46.0 |
| WorldWise+ | fused | 73.2 | 54.1 | 94.2 | 84.5 | 93.0 | 75.9 | 44.1 |
| **WorldWise++** | dinov3 | **74.8** | 54.4 | **94.6** | **86.2** | **93.6** | **76.6** | **48.1** |
| WorldWise++ (ablation) | dinov3_nodet | 74.1 | 54.3 | 94.2 | 86.0 | 93.6 | 75.0 | 40.1 |

### 1b. SGDet (2D IoU 0.5)

| method | backbone | wc R@20 | wc mR@20 | wc mR@50 | nc mR@20 | OU-nt R@20 | OU-nt mR@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| W-STTran | resnet50 | 56.3 | 19.2 | 33.4 | 27.7 | 14.6 | 7.1 |
| W-STTran++ | resnet50 | **56.6** | 17.6 | 30.3 | 25.4 | 15.6 | 6.6 |
| W-DSGDetr | resnet50 | 56.6 | 19.2 | 33.5 | 28.0 | 12.3 | 6.0 |
| W-DSGDetr++ | resnet50 | 56.4 | 20.3 | 35.4 | 31.2 | 14.3 | 7.7 |
| W-USG | resnet50 | 56.3 | 19.1 | 34.1 | 25.9 | 14.5 | 6.7 |
| WorldWise | resnet50 | 53.6 | 20.4 | 38.5 | 36.1 | 25.4 | 20.6 |
| WorldWise | dinov2b | 54.1 | 23.4 | 40.1 | 37.9 | 29.6 | 21.1 |
| WorldWise | dinov2l | 53.5 | 22.3 | 38.1 | 37.2 | **30.4** | 20.2 |
| WorldWise | dinov3l | 52.6 | 21.5 | 38.8 | 37.5 | 28.1 | 21.3 |
| WorldWise+ | dinov3tok | 54.2 | 23.1 | 45.5 | 43.3 | 26.6 | 20.5 |
| WorldWise+ | pi3tok | 53.2 | 22.9 | 43.8 | 41.1 | 27.2 | **22.4** |
| WorldWise+ | fused | 54.2 | 23.9 | 46.9 | 43.5 | 27.1 | 21.1 |
| **WorldWise++** | dinov3 | 54.6 | **24.5** | **48.6** | **44.1** | 27.8 | 22.3 |
| WorldWise++ (ablation) | dinov3_nodet | 54.8 | 24.1 | 48.4 | 43.9 | 27.7 | 21.9 |

**Note on the baselines.** They are scored at resnet50 only, which is conservative:
their dinov3l predcls results were slightly worse (66.9 / 38.4). The baseline ladder
is within noise and non-monotone (W-STTran++ below W-STTran), so do not claim that
each baseline component helps.

---

## 2. Unlocalized MLLM track

No method here emits a 3D box, so sgdet is class-only and the slots-with-3D column
is 0 by construction. These are the rows that answer whether a prompted model can do
the task at all.

### 2a. PredCls — full split, 1,511 videos (Qwen2.5-VL-7B)

| method | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 | legacy uF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero_shot (frames only) | **46.9** | 25.1 | 60.9 | 43.1 | 40.3 | 42.8 | – |
| caption_all | 45.3 | **26.1** | 59.9 | 43.6 | 40.6 | 44.1 | 48.2 |
| rag_all | 46.9 | 26.0 | **61.1** | **43.8** | **40.8** | **44.6** | **49.9** |

**All three agree within 1.6 points of R@20.** Neither captioning nor Graph-RAG
retrieval buys anything over a plain frames-only per-object prompt.

### 2b. PredCls — thinking cells, 150-video subset (Qwen3-VL-8B)

| method | decode | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| RAG | standard | 47.4 | 28.8 | 60.0 | 48.5 | 46.1 | 51.8 |
| **RAG** | **thinking** | **50.4** | **30.3** | **63.8** | **49.9** | **47.7** | **53.8** |

**Thinking is worth +3.0 R@20 on top of retrieval** and this is the only valid
thinking cell in the project — see section 5.

### 2c. SGDet

| method | model | split | videos | unloc nc R@50 | unloc nc mR@50 | slots w/ 3D |
|---|---|---|---:|---:|---:|---:|
| RAG, standard | qwen3vl_8b | think150 | 145/150 | 22.7 | 18.5 | 0.0 |
| RAG, thinking | qwen3vl_8b_thinking | think150 | 150/150 | 22.3 | **20.6** | 0.0 |
| zero_shot (frames only) | qwen25vl_7b | full | 1298/1511 | generating | | |
| caption_all | qwen25vl_7b | full | 634/1511 | generating | | |
| rag_all | qwen25vl_7b | full | 578/1511 | generating | | |

### 2d. Legacy-protocol table — backbone breadth, combined

Same Action Genome test split as the rest of this document, same annotation basis
(GT + corrections), same three predicate groups. The pragya rows evaluate 1,734
videos and the WorldBBox rows evaluate 1,511; **the 1,511 are a strict subset of the
1,734** (verified: zero WorldBBox videos are absent from the pragya set), so these
rows sit on one axis and the tables combine.

What they do **not** share with sections 2a–2c and 3–4 is the metric family. Here it
is per-predicate-group precision / recall / F1 with micro-F1 and macro-F1; there it
is R@K and mR@K. Legacy micro-F1 already appears as the `legacy uF1` column in
section 2a, which is how these two bodies of work connect.

Sources: pragya rows from
`WorldSceneGraphAnnotationTool/assets/tex_files/tables/*_gt_plus_corrections_detailed.tex`
(generated May 2026, "Caption" = the `subtitle_all` pipeline, "RAG" = `rag_all`);
WorldBBox rows from `results/mllm_worldbbox_*.json`, `legacy.gt_plus_corrections`.

#### PredCls — F1 per predicate group, micro-F1, macro-F1

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
| Qwen2.5-VL-7B | zero_shot (frames only) | 1511 | 55.8 | 48.3 | 45.9 | 49.8 | 23.2 |
| Qwen2.5-VL-7B | rag_all | 1511 | 53.6 | 49.7 | 46.7 | **49.9** | 24.2 |
| Qwen2.5-VL-7B | caption_all | 1511 | 53.2 | 47.3 | 44.7 | 48.3 | 23.2 |
| Qwen3-VL-8B | rag_all | 150 | 63.0 | 46.0 | 45.9 | 51.4 | 28.4 |

#### SGDet — F1 per predicate group, micro-F1, macro-F1

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
| Qwen3-VL-8B | rag_all | 145 | 32.2 | 25.7 | 24.2 | 27.1 | 17.0 |

#### Full precision and recall for the pragya rows

The per-group precision and recall behind the F1 columns above are in
`assets/tex_files/tables/{predcls,sgdet}_gt_plus_corrections_detailed.tex`. They are
worth quoting for one row in particular: KeyeVL Caption holds precision of 43.7 /
37.6 / 55.8 against recall of 18.5 / 15.0 / 19.5 in predcls. Its captioning path is
not inaccurate, it is silent.

#### How much the 223 extra videos matter

Qwen2.5-VL-7B with retrieval is the one cell measured on both video sets, which
makes it the calibration point:

| metric | 1,734 videos | 1,511 videos | delta |
|---|---:|---:|---:|
| μF1 | 49.8 | 49.9 | +0.1 |
| MF1 | 23.3 | 24.2 | +0.9 |
| Attention F1 | 52.8 | 53.6 | +0.8 |
| Contacting F1 | 47.8 | 49.7 | +1.9 |
| Spatial F1 | 49.0 | 46.7 | −2.3 |

**Micro-F1 is stable to 0.1 across the two video sets; per-group F1 moves by up to
2.3.** Compare backbones on μF1 across the whole table without qualification. Treat a
per-group comparison between a 1,734-video row and a 1,511-video row as approximate
until the pragya cells are re-scored on the 1,511 list (section 6, item 7).

#### What the breadth adds

1. **Retrieval beats captioning, but the margin is small and model-dependent.**
   RAG wins μF1 for five of six backbones in predcls and four of five in sgdet, and
   the typical margin is 0.5 to 3.6 points. That matches section 2a, where retrieval
   and a plain per-object prompt tie at 46.9 R@20.
2. **The one large gain is a captioning failure, not a retrieval success.** KeyeVL
   gains 15.5 μF1 in predcls and 18.0 in sgdet purely because its captioning recall
   collapses. Quote it as robustness for RAG, not as evidence retrieval carries
   information.
3. **Two backbones invert the ordering.** Qwen3-VL prefers captioning in predcls,
   48.7 against 44.5 μF1, and Ovis2.5 prefers it in sgdet, 35.6 against 34.5. No
   method dominates across backbones.
4. **Spatial predicates separate the backbones far more than attention does.**
   Spatial F1 spans 23.4 to 56.0 under RAG in predcls while attention spans 38.3 to
   55.7. The unlocalized setting is mostly measuring spatial competence, which is
   what the localized track attacks and where it collapses at IoU 0.15.
5. **Rank order is not preserved between modes.** Ovis2.5 leads predcls μF1 among the
   non-Qwen backbones at 48.9 and falls to 34.5 in sgdet; Qwen2.5-VL leads both.
   Single-mode backbone claims will not survive review.
6. **Backbone scale is not the story.** Qwen3-VL-30B-A3B, the largest model here,
   wins attention F1 by a wide margin (62.1) yet lands mid-table on μF1 (47.8) because
   its spatial F1 is among the worst (32.8). Spatial competence does not track size.

**Incomplete cells, not tabulated:** Qwen3-VL-30B-A3B has predcls only (1,542 videos,
no sgdet) and LLaVA-OneVision-7B never finished (19 predcls / 147 sgdet). A
corrections-only variant of both tables exists in the same directory and tells the
same story at uniformly lower absolute values.

---

## 3. Localized MLLM track

Both methods emit oriented boxes on roughly 80 % of object slots, so sgdet here is a
genuine localization measurement, matched in 3D.

### 3a. PredCls — full split, 1,511 videos (Qwen3-VL-8B)

| method | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---:|---:|---:|---:|---:|---:|
| Track A — marked frames + BEV | 51.7 | 30.5 | 62.8 | 47.2 | 46.5 | 40.0 |
| **Track B — tool loop + geometric critic** | **52.4** | **31.7** | **63.4** | **48.4** | **47.6** | **40.8** |
| Track B, critic removed (ablation) | 52.1 | 31.1 | 63.2 | 47.9 | 47.3 | 40.0 |

### 3b. SGDet — full split, 1,511 videos, 3D-IoU matched

| method | unloc nc R@50 | unloc nc mR@50 | IoU.15 R@50 | IoU.15 mR@50 | IoU.25 R@50 | IoU.25 mR@50 | slots w/ 3D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Track A | 33.3 | 20.2 | 11.0 | 6.6 | 7.9 | 4.7 | **80.8** |
| **Track B** | **33.5** | **23.8** | 10.7 | 7.7 | 7.7 | 5.5 | 79.5 |
| Track B, critic removed | 33.4 | 23.2 | **11.2** | **7.8** | **8.0** | **5.6** | 79.4 |

**The localization collapse is the headline of this track.** Track B goes from 33.5
unlocalized to 10.7 at IoU 0.15 to 7.7 at IoU 0.25. Relations survive; metric
placement does not. This is the quantitative case for the training-based track.

### 3c. The geometric critic ablation

| mode | geometric violations, before → after | effect on recall |
|---|---|---|
| predcls | 41,077 → 31,075, down 24 % | +0.3 wc R@20, +0.6 wc mR@20 |
| sgdet | 36,370 → 22,819, down 37 % | +0.6 unlocalized mR@50, but **−0.5 R@50 at IoU 0.15** |

The critic demonstrably repairs geometry and that repair does **not** become metric
accuracy. Report it as a mechanism-versus-metric dissociation, not as a win.

---

## 4. The matched block — tracks 2 and 3 on one backbone and one video set

At full split the unlocalized track runs on Qwen2.5-VL-7B while the localized track
runs on Qwen3-VL-8B, so any track-2-versus-track-3 claim there is confounded by
backbone. This block removes the confound: same 150 videos, same Qwen3-VL-8B, same
scoring pass. **It is the only place the two MLLM tracks may be compared.**

| track | method | decode | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | Track B | standard | **52.2** | **31.4** | 62.1 | 49.5 | **48.4** | 45.3 |
| 3 | Track A | standard | 51.4 | 29.9 | 61.5 | 46.7 | 46.0 | 42.7 |
| 2 | RAG | thinking | 50.4 | 30.3 | **63.8** | **49.9** | 47.7 | **53.8** |
| 2 | RAG | standard | 47.4 | 28.8 | 60.0 | 48.5 | 46.1 | 51.8 |
| 3 | Track A | thinking | 30.5 | 19.3 | 45.9 | 40.4 | 41.2 | 39.9 |
| 3 | Track B | thinking | 27.2 | 16.4 | 43.2 | 38.2 | 37.8 | 38.2 |
| 3 | Track B, no critic | thinking | 27.3 | 16.2 | 43.3 | 38.1 | 37.8 | 37.6 |

**The two MLLM tracks are complementary, not ranked.** Track 3 wins headline recall
and observed objects, 48.4 against 47.7. Track 2 with thinking wins no-constraint
recall, mean recall, and the unobserved-non-trivial bucket by **8.5 points**, 53.8
against 45.3. Localization helps what the camera can see; retrieval carries object
permanence. That is real routing headroom on the one axis where the tracks disagree,
and testing it needs no GPU — merge the existing slot-level predictions and score
the oracle.

---

## 5. Protocol warnings — read before merging any two tables

1. **PredCls is cross-track comparable. SGDet is not.** Every predcls row in this
   document, supervised or prompted, is scored by the same stock WorldSGG evaluator
   with ground-truth boxes supplied. SGDet differs by track: 2D IoU 0.5 in track 1,
   3D IoU in track 3, class-only in track 2. Never put a track-1 and a track-3 sgdet
   number in one column.
2. **Track-1 sgdet is not comparable to anything produced before the `bbox_2d` fix.**
   The 2D ground-truth box lives under PKL key `bbox_2d` in original-frame pixels and
   is rescaled into Pi-3 space. Before that fix `gt_bboxes_2d` was all zero and sgdet
   matching was meaningless.
3. **Every thinking cell outside section 2b is invalid**, not merely weak. The
   reasoning trace shares `max_new_tokens` with the answer, so a prompt that asks for
   one array over every object in the frame never reaches its answer. The closing
   `</think>` appears in 46.2 % of Track A predcls responses and 0.7 % in sgdet.
   Track A at 30.5, Track B at 27.2, and 0.6 / 0.3 in sgdet are budget failures. RAG
   is immune because it prompts per object. This is a prompt-format result, not a
   model result.
4. **About 37 % of annotation objects have no feature slot** (never detected by
   GDino) in the old and the new annotation sets alike, so they are never evaluated
   by any row here. The paper should state this.
5. **Video counts differ by row** — check the videos column. The thinking cells are
   150 videos because thinking costs 255 s per video for Track A and 715 s for RAG,
   against 30 s for standard decode.

6. **Section 2d shares the test split but not the metric.** Its rows sit on the
   same Action Genome test videos and the same GT + corrections annotation basis,
   and the 1,511-video WorldBBox set is a strict subset of the 1,734 evaluated on
   pragya, so its rows combine with each other. What it does not share with
   sections 0-2c and 3-4 is the metric family: per-predicate-group P/R/F1 with
   micro and macro F1, rather than R@K. The bridge between the two is the
   `legacy uF1` column in section 2a, which is the same micro-F1. Compare
   backbones there on micro-F1 freely; treat a per-group comparison across
   different video counts as approximate, since micro-F1 moves 0.1 between the two
   sets but per-group F1 moves up to 2.3.

---

## 6. What is still missing

| # | gap | track | cost |
|---|---|---|---|
| 1 | Three unlocalized sgdet baselines, generating now | 2 | ~7 h GPU, then scored automatically |
| 2 | RAG, standard decode, Qwen3-VL-8B, full split — makes the track-2 versus track-3 comparison backbone-matched at 1,511 instead of 150 | 2 | **~12 h GPU, the cheapest high-value run left** |
| 3 | Oracle router over Track B and RAG slot predictions | 2 + 3 | **no GPU** |
| 4 | Thinking at full split | 2 | ~300 h GPU, not worth it before the deadline |
| 5 | Re-score the pragya backbone cells on the 1,511-video list so section 2d is exact rather than approximate at the per-group level | 2 | CPU only, PBS job on pragya |
