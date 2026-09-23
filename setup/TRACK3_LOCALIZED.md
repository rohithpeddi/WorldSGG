# Track 3 — the localized MLLM track

Training-free MLLM methods that also **place objects in 3D**: each emitted
object carries an oriented bounding box (centre, size, yaw) in the canonical
floor-aligned world frame, so its SGDet can be matched geometrically rather than
by class alone.

This is the track that tests whether a prompted model can localize. The answer
is the paper's quantitative case for the supervised track.

Authoritative results: [analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md)
§3–4 · design: [docs/ICLR_PLAN.md](../docs/ICLR_PLAN.md) WS2

| Track | Document |
|---|---|
| 1 — Training-based | [TRACK1_TRAINING.md](TRACK1_TRAINING.md) |
| 2 — Unlocalized MLLM | [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) |
| **3 — Localized MLLM** | this file |

---

## 1. Methods in this track

| Method key | Document | What it is | Runner |
|---|---|---|---|
| `track_a` | [MLLM_TRACK_A.md](MLLM_TRACK_A.md) | prompt engineering: marked frames (set-of-mark) + a marked BEV of the Pi-3 reconstruction, one VLM call per frame | [lib/mllm/methods/track_a_prompt/runner.py](../lib/mllm/methods/track_a_prompt/runner.py) |
| `track_b` | [MLLM_TRACK_B.md](MLLM_TRACK_B.md) | agentic: a fixed tool loop over Track A's payload plus retrieval, a **geometric critic**, and a repair call | [lib/mllm/methods/track_b_agent/runner.py](../lib/mllm/methods/track_b_agent/runner.py) |
| `track_b`, `objects_pre` | [MLLM_TRACK_B.md](MLLM_TRACK_B.md) §5 | the critic ablation, scored from the same run without regenerating | [critic.py](../lib/mllm/methods/track_b_agent/critic.py) |

Track B is Track A's payload plus two things: retrieved Stage-1 context, and
the verify → repair round. Both arms of the critic ablation come out of **one**
run — `frames[f]["objects"]` is post-repair (+critic) and
`frames[f]["objects_pre"]` is the first proposal (−critic) — and
`score_run --objects_key objects_pre` scores the latter through the usual
adapter. Keeping the tracks distinct by *what the model may call*, not by what
it sees, is the design rule from [docs/ICLR_PLAN.md](../docs/ICLR_PLAN.md) WS2.

`wsg_agent` is **not** in this track. It has "agent" in the name but emits no
boxes, so it is scored in [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md).

### Shared tool cache layer

Both methods build their payload from the same annotation-independent caches
([lib/mllm/tools/build_caches.py](../lib/mllm/tools/build_caches.py)):

| Tool | Module | Cache |
|---|---|---|
| `detect` (GDino proposals) | [tools/detections.py](../lib/mllm/tools/detections.py) | `cache/mllm/detections/` |
| `lift_to_3d` (Pi-3 + floor transform → OBB) | [tools/lift.py](../lib/mllm/tools/lift.py) | `cache/mllm/lifted3d/` |
| `render_bev` (top-down map of the reconstruction) | [tools/bev.py](../lib/mllm/tools/bev.py) | `cache/mllm/bev/` |
| `mark_frame` / `mark_bev` (set-of-mark ids) | [tools/marks.py](../lib/mllm/tools/marks.py) | — |
| VLM response cache, keyed on content hash | [tools/llm_cache.py](../lib/mllm/tools/llm_cache.py) | `cache/mllm/llm_cache/` |
| `query_graph` (Stage-1 graph/captions) — Track B only | [methods/graphs/runner.py](../lib/mllm/methods/graphs/runner.py) | `graphs/<model>/` |

Geometry conventions and the split/annotation join live in
[lib/mllm/data/worldbbox.py](../lib/mllm/data/worldbbox.py) and
[lib/mllm/data/geometry.py](../lib/mllm/data/geometry.py): z is up, the floor is
z = 0, an OBB is `(center, size, yaw)` and converts to and from 8 corners.
Everything operates in the **canonical floor-aligned frame** via the corrected
transform chain `T_XY ∘ T_delta ∘ T_auto` — Pi-3 output is scale- and
affine-ambiguous and reference-free, so anything emitting world geometry has to
go through that chain.

---

## 2. Evaluation protocol

Scored by [lib/mllm/eval/score_run.py](../lib/mllm/eval/score_run.py); protocol
detail in [lib/mllm/eval/README.md](../lib/mllm/eval/README.md).

| | |
|---|---|
| Test set | 1,511 videos / 48,834 frames, all annotated frames |
| Adapter | [lib/mllm/eval/dump_adapter.py](../lib/mllm/eval/dump_adapter.py) |
| PredCls regime | `wsgg`: the stock WorldSGG evaluator — **identical to track 1** |
| SGDet regime | `loc3d` ([lib/mllm/eval/recall3d.py](../lib/mllm/eval/recall3d.py)) at τ ∈ {0, 0.15, 0.25} |
| 3D matcher | [lib/mllm/eval/iou3d.py](../lib/mllm/eval/iou3d.py) — a verbatim copy of `compute_iou_3d_obb` from [lib/detector/monocular3d/evaluation/evaluate_3d.py](../lib/detector/monocular3d/evaluation/evaluate_3d.py): bottom-face polygon intersection × z-overlap |

`recall3d.py` keeps the stock GT-triplet construction, the with/no-constraint
ranking and the per-frame recall bookkeeping of `evaluate_from_dict`, and
changes exactly two things:

1. the triplet matcher requires class equality **and oriented 3D IoU ≥ τ on
   both endpoints**, instead of 2D IoU;
2. only pairs the model actually emitted (`pred_pair_valid`) produce predicted
   triplets — an MLLM that never mentions an object does not get a free argmax
   triplet for it.

At τ = 0 the matcher degenerates to class matching, which is the "unloc" column
and the only column comparable with track 2. In PredCls the predicted corners
*are* the GT corners, so `loc3d/unloc` equals `wsgg` restricted to emitted
pairs.

**Why 0.15 / 0.25 and not 0.5.** Relaxed thresholds are standard for monocular
3D detection; IoU@0.5 zeroes out every MLLM and measures nothing. The thresholds
were fixed in advance in [docs/ICLR_PLAN.md](../docs/ICLR_PLAN.md) WS2, not
chosen after seeing the results.

---

## 3. Results

Full split, 1,511 videos, Qwen3-VL-8B, standard decode. Values in %.

### 3a. PredCls

| method | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---:|---:|---:|---:|---:|---:|
| Track A — marked frames + BEV | 51.7 | 30.5 | 62.8 | 47.2 | 46.5 | 40.0 |
| **Track B — tool loop + geometric critic** | **52.4** | **31.7** | **63.4** | **48.4** | **47.6** | **40.8** |
| Track B, critic removed (ablation) | 52.1 | 31.1 | 63.2 | 47.9 | 47.3 | 40.0 |

### 3b. SGDet, 3D-IoU matched

| method | unloc nc R@50 | unloc nc mR@50 | IoU.15 R@50 | IoU.15 mR@50 | IoU.25 R@50 | IoU.25 mR@50 | slots w/ 3D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Track A | 33.3 | 20.2 | 11.0 | 6.6 | 7.9 | 4.7 | **80.8** |
| **Track B** | **33.5** | **23.8** | 10.7 | 7.7 | 7.7 | 5.5 | 79.5 |
| Track B, critic removed | 33.4 | 23.2 | **11.2** | **7.8** | **8.0** | **5.6** | 79.4 |

Both methods emit oriented boxes on roughly 80 % of object slots, so this is a
genuine localization measurement rather than a degenerate one.

### 3c. The geometric critic ablation

| mode | geometric violations, before → after | effect on recall |
|---|---|---|
| predcls | 41,077 → 31,075, down 24 % | +0.3 wc R@20, +0.6 wc mR@20 |
| sgdet | 36,370 → 22,819, down 37 % | +0.6 unlocalized mR@50, but **−0.5 R@50 at IoU 0.15** |

---

## 4. Findings, ranked

1. **The localization collapse is the headline of this track.** Track B goes
   from 33.5 unlocalized nc R@50 to **10.7 at IoU 0.15** to **7.7 at IoU
   0.25**. Relations survive; metric placement does not. A prompted model can
   say *what* the person is doing with *which* object and cannot say *where*
   the object is to within a loose box overlap. This is the quantitative case
   for the training-based track.
2. **Localization is what moves a prompted model on the relational metric
   too.** The best localized method sits 5.5 points of wc R@20 above the best
   unlocalized one (52.4 vs 46.9) — but see the backbone caveat in §5.1, which
   is why the matched block in §5 exists.
3. **The supervised ceiling remains 22.4 points of wc R@20 above the best
   MLLM** (74.8 vs 52.4). The task is not solved by prompting.
4. **The critic repairs geometry and that repair does not become metric
   accuracy.** It removes 24 % of violations in PredCls and 37 % in SGDet, and
   buys +0.3 / +0.6 R / mR in PredCls and +0.6 unlocalized mR@50 in SGDet —
   but **costs 0.5 R@50 at IoU 0.15**. Report this as a mechanism-versus-metric
   dissociation, not as a win. The honest reading: the violations the critic
   catches are real geometric errors, and fixing them moves boxes in directions
   that are not correlated with the ground truth.
5. **Track B beats Track A on every PredCls column** (52.4 / 31.7 vs 51.7 /
   30.5) and on unlocalized SGDet mean recall (23.8 vs 20.2), while Track A
   holds the slots-with-3D rate (80.8 vs 79.5) and the IoU-matched columns. The
   agentic arm buys relational quality; it does not buy placement.
6. **Track A and Track B are close enough that the ablation is the
   contribution.** A 0.7-point R@20 gap between the two full methods is not a
   result; the critic's measurable 24–37 % violation reduction with no metric
   payoff is.

---

## 5. The matched block — tracks 2 and 3 on one backbone and one video set

At full split the unlocalized track runs Qwen2.5-VL-7B while this track runs
Qwen3-VL-8B, so **any track-2-versus-track-3 claim at full split is confounded
by backbone**. This block removes the confound: same 150 videos
(`splits/test_worldbbox_thinking150.txt`), same Qwen3-VL-8B, same scoring pass.
It is the only place the two MLLM tracks may be compared.

| track | method | decode | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | Track B | standard | **52.2** | **31.4** | 62.1 | 49.5 | **48.4** | 45.3 |
| 3 | Track A | standard | 51.4 | 29.9 | 61.5 | 46.7 | 46.0 | 42.7 |
| 2 | RAG | thinking | 50.4 | 30.3 | **63.8** | **49.9** | 47.7 | **53.8** |
| 2 | RAG | standard | 47.4 | 28.8 | 60.0 | 48.5 | 46.1 | 51.8 |
| 3 | Track A | thinking | 30.5 | 19.3 | 45.9 | 40.4 | 41.2 | 39.9 |
| 3 | Track B | thinking | 27.2 | 16.4 | 43.2 | 38.2 | 37.8 | 38.2 |
| 3 | Track B, no critic | thinking | 27.3 | 16.2 | 43.3 | 38.1 | 37.8 | 37.6 |

**The two MLLM tracks are complementary, not ranked.** Track 3 wins headline
recall and the observed-object bucket (48.4 vs 47.7). Track 2 with thinking
wins no-constraint recall, mean recall, and the unobserved-non-trivial bucket
by **8.5 points** (53.8 vs 45.3). Localization helps what the camera can see;
retrieval carries object permanence.

That is real routing headroom on the one axis where the tracks disagree, and
testing it needs **no GPU** — merge the existing slot-level predictions of
Track B and RAG and score the oracle. It is the cheapest outstanding item in the
whole project.

---

## 6. Protocol caveats

1. **The backbone confound.** Full-split track-2 rows are Qwen2.5-VL-7B;
   track-3 rows are Qwen3-VL-8B. Cross-track MLLM claims belong in §5 or
   nowhere. Closing it properly costs ~12 h GPU (RAG, standard decode,
   Qwen3-VL-8B, full split).
2. **Every thinking cell in this track is invalid**, not merely weak. The
   reasoning trace shares `max_new_tokens` with the answer, and Track A and
   Track B both ask for **one JSON array covering every object in the frame**,
   so the model never reaches its answer. The closing `</think>` appears in
   46.2 % of Track A PredCls responses and 0.7 % in SGDet. Track A at 30.5,
   Track B at 27.2, and 0.6 / 0.3 in SGDet are **budget failures, not model
   failures**. RAG is immune because it prompts per object. The only valid
   thinking cell in the project is RAG in
   [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) §3b.
   * Partial mitigation already in the code: `salvage_objects` in
     [track_a_prompt/runner.py](../lib/mllm/methods/track_a_prompt/runner.py)
     recovers the complete entries of a truncated array so a cut-off response
     does not drop the whole frame. It cannot recover entries never emitted.
3. **PredCls is cross-track comparable; SGDet is not.** Track 1 matches in 2D
   at IoU 0.5, this track matches in 3D, track 2 cannot match at all. Never put
   a track-1 and a track-3 SGDet number in one column.
4. **About 37 % of annotation objects have no feature slot** (never detected by
   GDino) and are therefore never evaluated by any row in any track.
5. **The 3D boxes of unlifted proposals are absent, not zero.** Roughly 20 % of
   slots carry no 3D box; those slots are dropped from τ > 0 matching by
   `require_pred_3d` rather than scored as misses. The slots-with-3D column is
   what makes that visible — always print it next to an IoU-matched number.
6. **`--skip-verification` flattens the scores**, making no-constraint ordering
   among pairs arbitrary-but-deterministic. Prefer R@50 / R@100 for MLLM rows.
