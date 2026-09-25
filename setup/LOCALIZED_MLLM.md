# Localized MLLM — Architecture (Track A and Track B)

Track: [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) · method, results and runbooks:
[MLLM_TRACK_A.md](MLLM_TRACK_A.md), [MLLM_TRACK_B.md](MLLM_TRACK_B.md) ·
index: [README.md](README.md)

The localized MLLM is the training-free architecture that gives a prompted VLM a
metric world frame. Its scene graph carries an oriented 3D box per object, so it
can be matched geometrically and checked by a program. It has three layers:

1. **Perception layer.** Frozen tools and annotation-independent caches express the
   video in the canonical floor frame: a Pi-3 point cloud rendered as a top-down
   map, GDino proposals, and their lifts to floor-parallel OBBs. It makes no VLM
   calls.
2. **Localized prompt (Track A).** Per annotated frame, one multi-image VLM call in
   which the target frame, the map and a metric text table all carry the same
   object ids. The answer is one JSON array: predicates in PredCls, predicates
   plus boxes in SGDet.
3. **Agentic wrapper (Track B).** Track A's payload, plus retrieved Stage-1
   context, a program-checkable **geometric critic** on the proposed localized
   graph, and one repair call that returns the violations to the model.

Track B imports Track A's payload builder, parser and model loader unchanged, so
the two tracks are one architecture with an optional loop, not two designs.

**Main configuration.** Qwen3-VL-8B-Instruct (`qwen3vl_8b`) runs through vLLM with
native multi-image prompts. The headline runs use `--max_new_tokens 1024
--temperature 0.2`, with the default top_p 0.95 and seed 0, `--context_frames 2`,
four videos pooled per generate call, and `max_model_len` 24,576. Track B adds
`--max_repairs 1`. The 150-video thinking cells use `qwen3vl_8b_thinking` with
4,096 new tokens, temperature 0.6 and top_p 0.95. Track A's argparse defaults
(4,096 tokens, temperature 0.6) are these thinking settings; every standard run
overrides them from the job file
([scripts/remote/b3_baselines.jobs](../scripts/remote/b3_baselines.jobs)).

| Component | Code |
|---|---|
| Data join, vocabularies | [lib/mllm/data/worldbbox.py](../lib/mllm/data/worldbbox.py): `WorldBBoxTestSet`, `WorldBBoxVideo`, `FrameAnnot`, `ObjectAnnot` |
| Geometry | [lib/mllm/data/geometry.py](../lib/mllm/data/geometry.py): `obb_to_corners`, `corners_to_obb`, `obb_floor_parallel_from_points` |
| Map | [lib/mllm/tools/bev.py](../lib/mllm/tools/bev.py): `build_cloud`, `bev_window`, `render_base`, `mark_bev` |
| Proposals | [lib/mllm/tools/detections.py](../lib/mllm/tools/detections.py): `normalise_gdino_label`, `build_detections` |
| 2D → 3D lift | [lib/mllm/tools/lift.py](../lib/mllm/tools/lift.py): `lift_bbox`, `LiftCache` |
| Set-of-mark | [lib/mllm/tools/marks.py](../lib/mllm/tools/marks.py): `mark_frame` |
| Response cache | [lib/mllm/tools/llm_cache.py](../lib/mllm/tools/llm_cache.py): `LLMCache`, `CachedVLM` |
| Cache build | [lib/mllm/tools/build_caches.py](../lib/mllm/tools/build_caches.py) |
| Track A | [lib/mllm/methods/track_a_prompt/runner.py](../lib/mllm/methods/track_a_prompt/runner.py): `build_payload`, `extract_json`, `salvage_objects`, `parse_objects`; [prompts.py](../lib/mllm/methods/track_a_prompt/prompts.py) |
| Track B | [lib/mllm/methods/track_b_agent/runner.py](../lib/mllm/methods/track_b_agent/runner.py): `load_stage1`, `query_graph`, `repair_prompt`, `process_group`; [critic.py](../lib/mllm/methods/track_b_agent/critic.py): `check_geometry` |
| Backbone | [lib/mllm/models/qwen3vl_new.py](../lib/mllm/models/qwen3vl_new.py): multi-image prompts |

---

## 1. Architecture at a glance

```
PERCEPTION LAYER — offline, CPU, no VLM                  lib/mllm/tools/build_caches.py
  Pi-3 dynamic scene (points, confidences, cameras)
    └─ world_to_final ─► canonical frame (z up, floor z = 0, metres)
         ├─ build_cloud → bev_window → render_base ─► BEV base image + metric window
         └─ points_final(k) ─────────────────────────────┐
  GDino boxes ─► normalise labels, score ≥ 0.25 ─► proposals ─► lift_to_3d
                                                                  └─► OBB | none

TRACK A — one VLM call per annotated frame f             methods/track_a_prompt/
  object table  PredCls: GT objects of f (GT OBB; visible / NOT visible)
                SGDet:   proposals of f by score (lifted OBB; detector score)
  person P      PredCls: GT OBB and 2D box   SGDet: largest person detection, lifted
  images  [1]   mark_frame(f): numbered 2D boxes, P, legend of unseen ids   ≤640 px
          [2–3] first and last other annotated frames, unmarked             ≤320 px
          [4]   mark_bev: numbered footprints, P, camera arrow at f         ≤640 px
  text    rules + image legend + metric table (+ SGDet: map extent, vocabulary)
          + task + one-line JSON schema
      ▼
  Qwen3-VL-8B: 4 images + text → ≤1,024 new tokens at T = 0.2
      ▼
  extract_json → salvage_objects → parse_objects ─► G_f = {label: predicates, OBB, score}

TRACK B — Track A plus a fixed loop                      methods/track_b_agent/
  text += query_graph(Stage-1 segments nearest f, ±2)            [before "Task:"]
  call 1 ─► G_f^pre ─► check_geometry(G_f^pre, person box, map window) ─► violations
    └─ if violations and G_f^pre parsed:
         call 2: text + G_f^pre as JSON + ≤12 violations ─► G_f^post (re-checked)
  emit objects = G_f^post (else G_f^pre), objects_pre = G_f^pre, violations pre/post
```

---

## 2. The canonical frame

Every geometric quantity lives in one frame, in metres: points, cameras, boxes, the
map, the critic's checks and the 3D evaluator. The frame is z-up with the floor at
z = 0. Pi-3 output is scale- and affine-ambiguous and has no reference frame, so
everything passes through the corrected chain T_XY ∘ T_delta ∘ T_auto, exposed as
`corners_final = A_world_to_final (corners_world − origin)`.

- **Box.** A box is a floor-parallel OBB `(center, size = (l, w, h), yaw)`, where yaw
  rotates the length axis about +z. Its 8 corners are listed bottom face first.
- **Frame index.** Pi-3 frame k is annotation frame `sampled_frames_idx[start + k]`.
  `start` is recovered per video by pose matching.
- **Camera.** The camera's heading is the azimuth of its optical (+z) axis in this
  frame.

---

## 3. Perception layer (offline, CPU, annotation-independent)

`python -m lib.mllm.tools.build_caches --what bev,detections,lift` builds the caches
once per video and registers them in the cache manifest with
`annotation_version = None`. An annotation revision therefore only costs new prompts.

### 3.1 Map (`render_bev`, `mark_bev`)

| Step | Rule |
|---|---|
| Cloud | ≤40 Pi-3 frames, sampled uniformly. Keep points with confidence ≥ 0.05, every 2nd pixel, −2 m < z < 3 m. Average position and colour per 3 cm voxel. Keep all camera poses. |
| Window | 1st–99th percentile of point xy, extended by the camera positions, plus 0.5 m of padding, extent ≥ 1 m. px/m = min(768 / longest side, 200), and at least 60. |
| Base render | Top-down splat where the highest point wins, brightening with height over 0–2 m. 0.5 m grid with metre labels (`x=..`, `y=..`); camera path as grey dots. |
| Per-frame marks | Object footprints (OBB bottom faces) numbered with the prompt ids; the person footprint as `P`; the target-frame camera as a red dot with a 0.5 m arrow along its viewing direction. |

### 3.2 Proposals (`detect`)

The GDino boxes of each sampled frame are precomputed (`detection/gdino_bboxes`).
Their free-text labels are normalised onto the AG short vocabulary, either through
an alias table (`glass`/`bottle` → `cup`, `camera` → `phone`, `couch` → `sofa`, …)
or by exact match to a short name. Unknown labels and scores below 0.25 are dropped;
person is kept. Each box is stored in both original and Pi-3 pixels.

### 3.3 Lift (`lift_to_3d`)

The lift turns a 2D box into an OBB, using that frame's Pi-3 points in the canonical
frame:

1. **Depth anchoring.** Take the median camera depth d0 over the box's central half.
   Keep points within max(0.25 m, 0.15·d0, 0.6 × the box's physical size) of d0.
2. **Candidates.** For each erosion kernel in {0, 5, 11} px, take the points inside
   the eroded box and the depth band, with confidence ≥ 0.05. A candidate needs at
   least 25 points.
3. **Trim.** Drop points outside the 5th–95th percentile on each axis.
4. **Fit.** Fit a floor-parallel minimum-area rectangle in xy, spanning
   [z_min, z_max].
5. **Select.** Keep the smallest-volume candidate. With no candidate, the proposal
   has no box and the prompt says "3D box unknown".

Results are cached per (frame, label, rounded box).

### 3.4 Set-of-mark (`mark_frame`)

The target frame is downscaled to ≤640 px on its long side. Each 2D box is outlined
and tagged `<id> <label>` in a 15-colour palette, and the person is drawn in black as
`P person`. Objects without a 2D box, i.e. unobserved objects, are listed in a
top-left legend as `<id> <label> (not visible here)`.

### 3.5 Response cache (`CachedVLM`)

Each VLM call is keyed by sha256 over the model key, the prompt text, the PNG hashes
of its images, the temperature and `max_new_tokens`. Hits are served from disk. A
multi-image backbone (Qwen3-VL) receives the images as separate images, placed before
the text. Other wrappers receive them zero-padded to a common canvas and stacked as
one video.

---

## 4. Track A — the localized prompt

### 4.1 Object table

| | PredCls | SGDet |
|---|---|---|
| Ids 1..N | the frame's GT objects, first per short label | the frame's GDino proposals by descending score, person excluded |
| Box per id | GT OBB, from `corners_final` | lifted OBB, or "3D box unknown" |
| Visibility | "visible" / "NOT visible in the target frame"; observed means visible and not sourced from rag / gdino / correction | not stated |
| Person P | GT person OBB and 2D box | largest person detection, lifted |
| Extra text | — | map extent in metres; 35-name object vocabulary; detector score per proposal |

### 4.2 Images, in prompt order

| # | Content | Size |
|---|---|---|
| 1 | target frame with set-of-mark ids | ≤640 px long side |
| 2–3 | unmarked context frames: `linspace` over the video's other annotated frames, which with two frames means the first and the last | ≤320 px long side |
| 4 | marked BEV: footprints, P, camera arrow | ≤640 px long side |

The text includes an image legend that explains each image. The backbone is loaded
with `max_images = context_frames + 2`, which is 4.

### 4.3 Text

The text runs in this order:

1. System rules: the frame conventions, the OBB definition, "the person is the only
   subject", and the 3 / 17 / 6 predicate vocabulary with its cardinalities.
2. The image legend.
3. The scene table: the P person box; the camera position and heading; one line per
   id with label, visibility and `center=(x, y, z) m, size=(l, w, h) m, yaw=D deg`.
4. The task.
5. A one-line JSON schema.

**PredCls task.** The model decides the three relationship heads for every listed
id, using "the map and the boxes for objects that are not visible" and "the images
for what the person is doing". It answers
`{"objects": [{"id", "attention", "contacting": [...], "spatial": [...]}]}`.

**SGDet task.** The model lists every object the person interacts with or that is
relevant, "including objects that are out of view or hidden but whose location you
can infer from the video and the map (e.g. the chair the person sat on earlier)". It
copies or corrects a proposal's box and estimates one for objects without a proposal.
It keeps proposal ids and uses `"new"` for objects it adds. Each answer entry adds
`label`, `center`, `size` and `yaw_deg`.

### 4.4 Parsing

`extract_json` drops any reasoning trace up to the last `</think>`. It tries fenced
blocks, then the raw text, and takes the first balanced `{…}` that has an `objects`
key. If that fails, `salvage_objects` keeps the complete entries of an array cut off
at the token budget.

`parse_objects` then works as follows:

- **PredCls.** An entry binds by id to a payload object. The label and the box are
  the ground truth's; any box the model writes is ignored.
- **SGDet.** The label is normalised to the short vocabulary, and person is rejected.
  The box is the model's center/size/yaw when given (`src = model` or
  `proposal+model`), otherwise the proposal's lifted box (`src = proposal`). An object
  with neither is dropped, because it cannot be scored in 3D.
- Predicates are filtered to their vocabularies; attention is kept only if valid.
- **One object per label per frame**: the first entry wins.
- The score is the detector score for proposals and 1.0 otherwise.

### 4.5 Decoding and batching

Track A makes one call per annotated frame, with up to 4 images per prompt. The
prompts of four videos are pooled into one vLLM `generate()`. The headline setting
is temperature 0.2, top_p 0.95, seed 0 and 1,024 new tokens.

---

## 5. Track B — a fixed tool loop with a geometric critic

### 5.1 The loop

The steps are plan → perceive → retrieve → relate → verify → repair → emit. The
driver runs them in that order for every frame; the model never selects a tool.

| Step | Tool | Implementation |
|---|---|---|
| perceive | `detect`, `lift_to_3d`, `get_camera_pose`, `render_bev`, `mark_frame` | Track A's `build_payload`, imported |
| retrieve | `query_graph` | Stage-1 captions and entities around the frame, inserted before `Task:` |
| relate | VLM call 1 | the proposal, stored as `objects_pre` |
| verify | `check_geometry` | the critic (§5.3), a program |
| repair | VLM call 2 | only for frames whose critic fired and whose proposal parsed |
| emit | — | `objects` after repair, `objects_pre`, `violations_pre`, `violations_post`, `n_calls` |

### 5.2 `query_graph`

`query_graph` reads the Stage-1 pickles of the unlocalized Graph-RAG
([UNLOCALIZED_GRAPH_RAG.md](UNLOCALIZED_GRAPH_RAG.md) §3). It always reads the
Qwen2.5-VL-7B graphs, whatever the generation model. For target frame f it works as
follows:

1. Sort the segments by key frame, and take the one nearest f plus two on either
   side.
2. Render each as `- [earlier | at the target frame | later, frame k] <caption>
   (entities: …)`, listing the first six entity names of that segment's node in
   alphabetical order.
3. Cut the block at 900 characters, under the header "Retrieved context from a scene
   graph built over the whole video (may be noisy):".

No embeddings are involved. This is a temporal window, not a similarity search: the
opposite choice from Graph-RAG, whose retrieval is per object and ignores the frame.

### 5.3 The critic

`check_geometry` works in the canonical frame. Its inputs are the parsed graph (boxes
and predicates), the person box and the map window. The person box is the ground
truth in PredCls and a lifted detection in SGDet. Without a person box, only the
schema, floor, size and extent checks run.

| Kind | Fires when | Threshold |
|---|---|---|
| schema | attention is not exactly one valid label | — |
| floor | the box bottom is below the floor | z_min < −0.15 m |
| floor | a touching-type contact (next row) is claimed but the box floats above the person | z_min > person z_max + 0.3 m |
| size | any box side is implausible | > 3 m or < 1 cm |
| extent | the box centre is outside the map window | 1 m margin |
| contact | holding / touching / carrying / eating / drinking_from / wiping / writing_on / twisting / wearing / have_it_on_the_back is claimed and the object is far from the person | gap > 0.5 m |
| contact | sitting_on / lying_on / standing_on / leaning_on / covered_by is claimed without touching the person | gap > 0.25 m, or no vertical overlap |
| contact | not_contacting is claimed but the box lies inside the person box | ±5 cm |
| vertical | `above` with the object centre not higher than the person centre; `beneath` with it not lower; or both claimed | — |

Two notes on the table:

- **Gap** is the Euclidean gap between the axis-aligned envelopes of the two boxes,
  0 when they overlap.
- **The schema kind** also covers vocabulary and predicate-string rules, but
  `parse_objects` has already filtered those, so in practice only the attention rule
  fires.

Unchecked predicates: `in_front_of`, `behind`, `on_the_side_of` and `in` among the
spatial predicates, because the person's facing direction is unknown; also
`other_relationship`, and attention beyond its cardinality.

Each violation is a sentence such as
`cup: ['holding'] claimed but the box is 2.31 m away from the person`.

### 5.4 Repair

The repair prompt is built from:

1. the full Track B prompt;
2. `Your previous answer was:` followed by the proposal re-serialised as JSON;
3. `A geometric checker found these problems with it:` followed by at most 12
   violation lines;
4. an instruction to keep every correct object and relationship and to "correct the
   box or the predicate that is inconsistent with the 3D layout".

The images are the same four. With `--max_repairs 1` there is one round. If the
repair does not parse, the proposal is kept. The repaired graph is checked again,
giving `violations_post`.

### 5.5 What a repair can change

| | SGDet | PredCls |
|---|---|---|
| Boxes | yes: centre, size, yaw | no: `parse_objects` keeps the GT box for each id |
| Predicates | yes | yes |
| Object set | add, drop, relabel | drop only, because ids bind to GT objects |

As a consequence, floor, size and extent violations raised on **ground-truth** boxes
cannot be repaired in PredCls; they persist into `violations_post` by design. A
predicate violation can also be discharged by leaving the object out. Both routes can
be checked offline by comparing `objects_pre` with `objects`.

### 5.6 Ablation arms

- **−critic** is `objects_pre` from the same run, which is the same first sampled
  answer. Score it with `score_run --objects_key objects_pre`.
- **`--no_critic`** makes a single call per frame and writes under the `_nocritic`
  tag.
- **`--no_retrieval`** writes under the `_noretr` tag. It is implemented but has never
  been scored. With retrieval off, the proposal prompt is byte-identical to Track A's.
  At the same temperature and budget, the response cache (§3.5) would therefore serve
  Track A's answers for the whole proposal round, and only repair calls would need
  the GPU.

---

## 6. Calls and budgets

| | Track A | Track B |
|---|---|---|
| Calls per annotated frame | 1 | 1, plus 1 if the critic fired and the proposal parsed |
| Calls per video (32.3 annotated frames on average) | F | F + number of repaired frames |
| Images per call | 4 | 4 |
| Answer | one array over all ids | the same; a repair re-emits the whole array |
| New-token budget | 1,024 (thinking: 4,096) | 1,024 (thinking: 4,096 in the `b6_think150_*` jobs) |

Measured cost is about 30 s per video with standard decoding, and 255 s per video for
Track A with the thinking backbone.

The answer is one array covering every object of the frame, so its length grows with
the number of objects. This is the thinking-budget defect: `</think>` closes in only
46.2 % of Track A PredCls responses and 0.7 % in SGDet, so every thinking cell of
this family is a budget failure ([TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) §6.2).

---

## 7. Output and scoring interface

Each video produces one pickle,
`<outputs.track_a | track_b>/<mode>/<model><tag>/<video>.mp4.pkl`:

```
{video_id, mode, model_name, track, context_frames, n_frames, n_parsed,
 frames: {<frame>.png: {objects: {label: {attention, spatial, contacting,
                                          corners, obb, score, src}},
                        raw_response, ids, person_corners}}}
```

Track B adds, per frame: `objects_pre`, `raw_response_pre`, `violations_pre`,
`violations_post`, `retrieved` and `n_calls`. Per video it adds `critic`,
`retrieval`, `n_calls` and the violation totals.

[lib/mllm/eval/dump_adapter.py](../lib/mllm/eval/dump_adapter.py) builds the scoring
slots:

- slot 0 is the person; SGDet uses the track's own person box;
- the frame's GT objects follow;
- in SGDet, predicted objects absent from the ground truth come last.

`has_pred_3d` is set wherever corners exist. The object score is the parse score, and
`pred_pair_valid` marks the pairs the model emitted.
[recall3d](../lib/mllm/eval/recall3d.py) requires class equality and oriented 3D
IoU ≥ τ on both endpoints, with τ ∈ {0, 0.15, 0.25}. In PredCls the predicted
corners are the ground truth's.

---

## 8. Relation to the unlocalized Graph-RAG

| | Graph-RAG (`rag_all`) | Track A | Track B |
|---|---|---|---|
| Call unit | (frame, object) | frame | frame, plus ≤1 repair |
| Visual input | one 16-frame video: target + ≤15 key frames at 168×336 | 4 images: marked target ≤640 px, 2 context frames ≤320 px, marked BEV ≤640 px | same as Track A |
| Metric 3D input | none | BEV, metric OBB table, camera pose | same as Track A |
| Video-level memory | top-1 Stage-1 node per object, by embedding similarity, frame-independent | 2 context frames | Stage-1 captions of the nearest segments (±2): a temporal window |
| Answer | three-field JSON per object | one JSON array over all ids, plus OBBs in SGDet | the same, then a corrected array |
| SGDet objects | P6 discovery over captions ∩ vocabulary | GDino proposals kept, corrected or added | same as Track A |
| 3D boxes | never | SGDet | SGDet |
| Thinking budget | immune: short answers | truncates: long array | truncates: long array |
| Full-split backbone | Qwen2.5-VL-7B | Qwen3-VL-8B | Qwen3-VL-8B |

Graph-RAG and Track B read the same Stage-1 pickles but use them in opposite ways:
retrieval by object versus retrieval by time.

---

## 9. Notes for a paper figure

A faithful diagram should show:

1. **The perception layer as frozen tools and caches**, with no learning and no VLM,
   and the canonical frame as their shared space.
2. **One id space across three encodings**: marks on the target frame, footprints on
   the BEV, and rows of the metric table. The answer uses the same ids.
3. **The PredCls / SGDet split**: given GT boxes on one side; lifted proposals,
   model-written boxes and `"new"` objects on the other.
4. **Track B as a fixed pipeline with one loop-back**: critic (a program) → repair (the
   VLM) → re-check, with at most one repair. The −critic arm is tapped before verify.
5. **`query_graph` as a temporal window** over the same Stage-1 graph that Graph-RAG
   builds.

It should not imply:

- that the model chooses its tools;
- bearing checks (front / behind / side);
- temporally adjacent context frames, since they are the video's first and last
  annotated frames;
- that the critic sees images. It sees only boxes, predicates, the person box and the
  map window.

---

## 10. Architecture-level caveats

1. **Array answers meet the thinking budget.** See §6: all thinking cells of this
   family are invalid.
2. **One instance per class per frame.** The PredCls payload deduplicates by label,
   and parsing keeps the first entry per label.
3. **Context frames are fixed per video.** With `--context_frames 2`, every target
   sees the video's first and last annotated frames, not its temporal neighbours.
4. **The SGDet person anchor is a heuristic.** It is the largest person detection. If
   it is missing or fails to lift, the critic's contact and vertical checks are
   skipped, and every spatial predicate loses its reference.
5. **PredCls repairs are limited** to predicates and dropping objects (§5.5).
6. **The critic covers height, not bearing.** The largest family of spatial
   predicates is outside its reach, so the measured violation reduction applies to the
   checkable subset only.
7. **The retrieval corpus comes from another model.** `query_graph` reads the
   Qwen2.5-VL-7B graphs even when Qwen3-VL generates.
8. **Placement is bounded by the reconstruction and the lift.** About 20 % of slots
   carry no box. They are dropped from τ > 0 matching rather than counted as misses,
   so always print slots-with-3D next to an IoU-matched number.
9. **The response-cache key omits top_p and seed.** A re-run that changes only those
   is served the old answers. Change the temperature or budget, or clear the cache.
10. **Porting to other backbones changes the visual encoding.** Wrappers without
    multi-image support receive the four images padded to a common canvas and stacked
    as a video, which is a different encoding from Qwen3-VL's.

---

## 11. Measured results

From [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) §3: full split, 1,511 videos,
Qwen3-VL-8B, standard decoding, values in %.

**PredCls**

| method | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---:|---:|---:|---:|---:|---:|
| Track A | 51.7 | 30.5 | 62.8 | 47.2 | 46.5 | 40.0 |
| **Track B** | **52.4** | **31.7** | **63.4** | **48.4** | **47.6** | **40.8** |
| Track B, −critic (`objects_pre`) | 52.1 | 31.1 | 63.2 | 47.9 | 47.3 | 40.0 |

**SGDet, 3D-IoU matched**

| method | unloc nc R@50 | unloc nc mR@50 | IoU.15 R@50 | IoU.25 R@50 | slots w/ 3D |
|---|---:|---:|---:|---:|---:|
| Track A | 33.3 | 20.2 | 11.0 | 7.9 | **80.8** |
| **Track B** | **33.5** | **23.8** | 10.7 | 7.7 | 79.5 |
| Track B, −critic | 33.4 | 23.2 | **11.2** | **8.0** | 79.4 |

**Critic.** Violations fall from 41,077 to 31,075 in PredCls (−24 %) and from 36,370
to 22,819 in SGDet (−37 %). The metric effect is +0.3 wc R@20 in PredCls, but
−0.5 R@50 at IoU 0.15 in SGDet. The critic repairs geometry, and that repair does not
become placement accuracy. The backbone-matched comparison with Graph-RAG is in
[TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) §5.
