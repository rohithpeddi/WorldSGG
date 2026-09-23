# Track A — localized WSGG by prompt engineering

Marked frames plus a marked bird's-eye view of the Pi-3 reconstruction, one VLM
call per annotated frame, answering with a compact JSON that carries both the
predicates **and** an oriented 3D box per object in the canonical world frame.

This is the first method in the project that a prompted model can be scored on
geometrically.

Runner: [lib/mllm/methods/track_a_prompt/runner.py](../lib/mllm/methods/track_a_prompt/runner.py) ·
Prompts: [lib/mllm/methods/track_a_prompt/prompts.py](../lib/mllm/methods/track_a_prompt/prompts.py) ·
Design: [docs/ICLR_PLAN.md](../docs/ICLR_PLAN.md) WS2 Track A ·
Track: [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) ·
Method key `track_a`

---

## 1. Design, and why it is shaped this way

The problem with asking a VLM for metric 3D placement is that it has no frame of
reference. Track A supplies one in three redundant encodings and asks the model
to reconcile them.

**Set-of-mark on the target frame.** Numbered boxes are drawn on the target
frame, `P` for the person. The ids in the image, the ids in the text table and
the ids in the answer are the same integers, so the model never has to bind an
object by name across modalities — the GPT4Scene recipe from
[docs/ICLR_PLAN.md](../docs/ICLR_PLAN.md) WS2.

**A top-down map carrying the same ids.** `render_bev` draws the Pi-3
reconstruction of the whole room; `mark_bev` stamps the object footprints with
the same numbers, the person's footprint as `P`, the camera path as grey dots
and the camera's position and viewing direction at the target frame as a red
arrow. This is what gives the model a world frame at all, and it is what makes
objects *outside the camera view* addressable — the prompt says so explicitly:
"some are outside the camera view or hidden".

**The same objects again as metric text.** Every object is also listed as
`center=(x, y, z) m, size=(l, w, h) m, yaw=D deg`, so the answer can be grounded
numerically rather than pictorially. The system rules fix the convention in
words: z up, floor at z = 0, x/y matching the grid labels on the map, and an OBB
is `{center, size, yaw_deg}` with yaw the rotation of the length axis about z.

**Unmarked context frames in between.** `--context_frames` (default 2) frames
sampled uniformly across the video at 320 px, placed between the target frame
and the BEV, with an explicit legend saying what each image is.

**Coordinate-frame discipline.** Everything is in the canonical floor-aligned
frame via the corrected transform chain `T_XY ∘ T_delta ∘ T_auto`. Pi-3 output
is scale- and affine-ambiguous and reference-free, so a method that emits world
geometry without that chain is emitting numbers in an arbitrary frame.

**Everything the model sees is cached and annotation-independent.** Detections,
lifted OBBs and the BEV are built once by
[lib/mllm/tools/build_caches.py](../lib/mllm/tools/build_caches.py); every VLM
call goes through a content-hash cache
([tools/llm_cache.py](../lib/mllm/tools/llm_cache.py)), so a re-run after an
annotation revision only recomputes prompts whose content actually changed.

---

## 2. The two modes are genuinely different prompts

| | PredCls | SGDet |
|---|---|---|
| ids come from | the frame's GT objects (deduplicated by short label) | GDino proposals, sorted by score |
| boxes | **given** — GT OBBs, in the text and on the BEV | proposals' **lifted** OBBs, stated to "contain errors, duplicates and misses" |
| person | GT person OBB and 2D box | largest `person` detection, lifted |
| model returns | id → attention / contacting / spatial | id (or `"new"`) → label, `center`, `size`, `yaw_deg`, + the three predicate heads |
| extra context | — | the map extent in metres, and the full 37-class object vocabulary |

The SGDet task statement asks for objects the person interacts with *including
objects out of view or hidden whose location can be inferred from the video and
the map* — "e.g. the chair the person sat on earlier". That sentence is the
whole unobserved-reasoning claim expressed as a prompt.

Answers are a single line of JSON, no markdown:

```json
{"objects": [{"id": 1, "attention": "...", "contacting": ["..."], "spatial": ["..."]}]}
```

---

## 3. Pipeline

```
per annotated frame:
  build_payload(ctx, frame, n_context)
    ├─ objects        predcls: GT objects (unique short label), obb from corners_final
    │                 sgdet:   GDino dets → LiftCache → OBB; person = largest det
    ├─ target image   mark_frame(...)                      max side 640
    ├─ context images n_context frames, uniform over the video, max side 320
    ├─ BEV            mark_bev(bev_img, meta, objs, camera_pose, person_corners)
    │                                                       max side 640
    └─ text           predcls_prompt / sgdet_prompt
  → images = [target] + context + [bev]

  one VLM call, max_new_tokens 4096, temperature 0.6, top_p 0.95
  → extract_json → parse_objects (validate against the three label sets)
  → salvage_objects on truncation
  → {label: {attention, spatial, contacting, corners, obb, score, src}}

prompts from --videos_per_batch (default 4) videos pooled into one vLLM call
```

Output: `<outputs.track_a>/<mode>/<model><tag>/<video>.mp4.pkl`, read by
[lib/mllm/eval/dump_adapter.py](../lib/mllm/eval/dump_adapter.py).

`--dry_run` builds every prompt and image without loading a model, which is how
the payload is inspected without a GPU.

**Truncation salvage.** `salvage_objects` recovers the complete entries of a
response cut off mid-array. The models answer with one JSON array, so a response
cut at `max_new_tokens` is unparseable and the whole frame would otherwise be
dropped even though every entry before the cut is valid. This mitigates — but
does not fix — the thinking-budget defect in §6.2: entries never emitted cannot
be salvaged.

---

## 4. Results

Full split, 1,511 videos, Qwen3-VL-8B, standard decode. From
[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) §3.

**PredCls**

| method | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---:|---:|---:|---:|---:|---:|
| **Track A** | 51.7 | 30.5 | 62.8 | 47.2 | 46.5 | 40.0 |
| Track B | **52.4** | **31.7** | **63.4** | **48.4** | **47.6** | **40.8** |

**SGDet, 3D-IoU matched**

| method | unloc nc R@50 | unloc nc mR@50 | IoU.15 R@50 | IoU.15 mR@50 | IoU.25 R@50 | IoU.25 mR@50 | slots w/ 3D |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Track A** | 33.3 | 20.2 | 11.0 | 6.6 | 7.9 | 4.7 | **80.8** |
| Track B | **33.5** | **23.8** | 10.7 | 7.7 | 7.7 | 5.5 | 79.5 |

**Matched 150-video block, Qwen3-VL-8B** ([ICLR_THREE_TRACKS](../analysis/ICLR_THREE_TRACKS.md) §4):

| decode | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 51.4 | 29.9 | 61.5 | 46.7 | 46.0 | 42.7 |
| thinking | 30.5 | 19.3 | 45.9 | 40.4 | 41.2 | 39.9 |

The thinking row is **invalid** — see §6.2. Do not present it as a model result.

### What the results establish

1. **Prompted localization works often enough to be measured and not well
   enough to be useful.** Track A emits an oriented box on **80.8 %** of object
   slots — the highest rate of any method — and its recall falls from 33.3
   unlocalized to **11.0 at IoU 0.15** and **7.9 at IoU 0.25**. Relations
   survive; metric placement does not.
2. **Track A holds the IoU-matched columns and the box-emission rate; Track B
   holds everything relational.** Adding retrieval and a critic
   ([Track B](MLLM_TRACK_B.md)) improves every PredCls column and unlocalized
   SGDet mean recall, and slightly *reduces* both the slots-with-3D rate
   (80.8 → 79.5) and IoU.15 R@50 (11.0 → 10.7).
3. **The BEV pays off on relations, not on placement.** Track A's PredCls wc
   R@20 of 51.7 is 4.8 points above the best unlocalized method's 46.9 — though
   that comparison is backbone-confounded (§6.1), which is why the matched
   block in [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) §5 exists. In the matched
   block Track A at 51.4 still leads RAG-standard at 47.4.
4. **The supervised ceiling is 23.1 points of wc R@20 above Track A**
   (74.8 vs 51.7).

---

## 5. Running

```sh
CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.track_a_prompt.runner \
    --model_name qwen3vl_8b --mode predcls \
    --video_list /data3/rohith/ag/splits/test_worldbbox_1511.txt
```

No-GPU prompt inspection:

```sh
python -m lib.mllm.methods.track_a_prompt.runner --mode sgdet --dry_run --limit 2
```

Scoring:

```sh
python -m lib.mllm.eval.score_run --method track_a --model qwen3vl_8b --mode sgdet
```

Defaults worth knowing: `--context_frames 2`, `--videos_per_batch 4`,
`--max_new_tokens 4096`, `--max_model_len 24576`, `--temperature 0.6`,
`--top_p 0.95`, `--seed 0`. `--tag` suffixes the output directory for prompt
variants.

**Prerequisite:** the B4 caches must be built —
`python -m lib.mllm.tools.build_caches` populates `cache/mllm/{bev,detections,lifted3d}`.

---

## 6. Known defects and caveats

1. **Backbone confound.** Track A runs Qwen3-VL-8B; the full-split unlocalized
   rows run Qwen2.5-VL-7B. Any Track-A-versus-track-2 claim at full split is
   confounded. The only valid comparison is the 150-video matched block on
   `splits/test_worldbbox_thinking150.txt`.
2. **Every Track A thinking cell is invalid, not merely weak.** The reasoning
   trace shares `max_new_tokens` with the answer, and Track A asks for **one
   JSON array over every object in the frame**, so the model spends the budget
   on the trace and never reaches the answer. The closing `</think>` appears in
   **46.2 % of Track A PredCls responses and 0.7 % in SGDet**. The PredCls
   thinking row (30.5) and the SGDet thinking figures (0.6 / 0.3) are budget
   failures. `salvage_objects` limits the damage; it does not repair it. Fixing
   this needs a separate thinking budget or a per-object prompt format — which
   is exactly why [`rag_all`](MLLM_RAG_ALL.md) is immune.
3. **~19 % of slots carry no 3D box.** Those slots are dropped from τ > 0
   matching by `require_pred_3d` rather than counted as misses, so an
   IoU-matched number is only interpretable next to the slots-with-3D column.
   Always print both.
4. **SGDet person geometry is heuristic.** The person is the largest `person`
   detection in the frame, lifted through `LiftCache`. A frame with two people
   or a missed person detection silently loses the subject anchor, and every
   spatial predicate is defined relative to the person.
5. **Free-form boxes in SGDet are not constrained to the map.** The prompt gives
   the extent and asks the model to "estimate a plausible box" for objects
   without a proposal, but nothing enforces it at generation time. Enforcing it
   *after* generation is exactly what [Track B](MLLM_TRACK_B.md)'s critic does —
   and §3c of [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) shows that enforcement
   does not become metric accuracy.
6. **The BEV is a rendering of a Pi-3 reconstruction**, so its quality bounds
   the achievable placement. The localization collapse in §4.1 is a joint
   property of the model and the reconstruction; the experiment as run cannot
   separate the two.
