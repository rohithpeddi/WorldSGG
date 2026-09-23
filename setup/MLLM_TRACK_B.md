# Track B — agentic localized WSGG with a geometric critic

[Track A](MLLM_TRACK_A.md)'s payload, plus retrieved context from the Stage-1
video graph, plus the piece the track exists to test: a **geometric critic** that
inspects the proposed localized scene graph, and a repair call that feeds the
violations back to the model.

Runner: [lib/mllm/methods/track_b_agent/runner.py](../lib/mllm/methods/track_b_agent/runner.py) ·
Critic: [lib/mllm/methods/track_b_agent/critic.py](../lib/mllm/methods/track_b_agent/critic.py) ·
Design: [docs/ICLR_PLAN.md](../docs/ICLR_PLAN.md) WS2 Track B ·
Track: [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) ·
Method key `track_b`

---

## 1. The claim, and why the critic is the contribution

From [docs/ICLR_PLAN.md](../docs/ICLR_PLAN.md) WS2: **"The geometric-consistency
critic is the novel, ablatable claim — only localized predictions can be
geometry-checked, and the agent self-corrects from the check."**

That sentence is the whole design. An unlocalized method emits `holding(person,
cup)` and there is nothing to check it against. A localized method emits
`holding(person, cup)` *and* a box for the cup 2.3 m away from the person, and
that is a contradiction a program can find without a label. The critic is the
capability localization unlocks, so it is the thing the track must ablate.

Two design rules follow.

**Keep the tracks distinct by what the model may *call*, not by what it sees.**
What Track B's model *sees* differs from Track A only by the retrieved context
block. The separating variable is the tool loop.

**Both arms of the ablation come out of one run.** `frames[f]["objects"]` is
post-repair (+critic) and `frames[f]["objects_pre"]` is the first proposal
(−critic); `score_run --objects_key objects_pre` scores the latter through the
usual adapter without regenerating anything. That is not a convenience — it
removes sampling variance from the ablation entirely. The two arms answer the
*same* first call.

**The tool order is fixed, not model-chosen.** The plan is
`plan → perceive → relate → verify → repair → emit` and the driver executes it
in that order. An 8 B model choosing tools freely per frame would not fit the
budget. This is a stated limitation, not an oversight: it is a *fixed* tool
loop, and the paper should call it that rather than implying free-form agency.

---

## 2. The tools

| Step | Tool | Source |
|---|---|---|
| perceive | `detect` (GDino boxes) | [tools/detections.py](../lib/mllm/tools/detections.py) |
| perceive | `lift_to_3d` (Pi-3 OBBs) | [tools/lift.py](../lib/mllm/tools/lift.py) |
| perceive | `get_camera_pose`, `render_bev` | [tools/bev.py](../lib/mllm/tools/bev.py) |
| retrieve | `query_graph` — Stage-1 graph/captions around the frame | [methods/graphs/runner.py](../lib/mllm/methods/graphs/runner.py) |
| relate | VLM call 1 → proposal JSON | Track A's `build_payload` / prompts |
| verify | `check_geometry` / schema check | [critic.py](../lib/mllm/methods/track_b_agent/critic.py) |
| repair | VLM call 2 with the violation list | `repair_prompt` |
| emit | parsed graph in the Track output format | — |

The perceive block is literally Track A's payload builder, imported:
`from lib.mllm.methods.track_a_prompt.runner import TrackAContext, build_payload,
extract_json, load_model, parse_objects`.

**`query_graph`** takes the clips whose annotated frame is nearest the target
(`window = 2` either side), and renders each as a caption plus up to 8 entity
names, tagged `earlier` / `at the target frame` / `later`, capped at 900
characters. It is injected immediately before the `Task:` line as:

```
Retrieved context from a scene graph built over the whole video (may be noisy):
- [earlier, frame 000012] <caption> (entities: cup, table, ...)
- [at the target frame, frame 000048] ...
```

The temporal tagging is what carries object permanence: "the chair the person sat
on earlier" is retrievable as text even when it is off-camera.

---

## 3. The critic

[critic.py](../lib/mllm/methods/track_b_agent/critic.py), all checks in the
canonical z-up floor frame, metres.

| kind | check |
|---|---|
| `schema` | label in vocabulary; exactly one attention label; predicate strings valid |
| `floor` | box bottom below the floor (`z_min < −0.15`), or floating > 0.3 m above the person while claiming contact |
| `extent` | box centre outside the reconstructed room window + 1 m margin |
| `size` | any side > 3 m or < 1 cm |
| `contact` | `holding / touching / carrying / eating / drinking_from / wiping / writing_on / twisting / wearing / have_it_on_the_back` need the object within `contact_dist = 0.5 m` of the person box; `sitting_on / lying_on / standing_on / leaning_on / covered_by` need it within `support_dist = 0.25 m` **and** overlapping in z; `not_contacting` with the object box inside the person box is contradictory |
| `vertical` | `above` → object centre higher than the person's; `beneath` → lower; both at once is a violation |

Distances use the Euclidean gap between axis-aligned envelopes (0 when they
overlap), not the OBB gap — a deliberate simplification, and a conservative one
for the contact checks.

**`front` / `behind` / `side` are not checked**, because the person's facing
direction is unknown. That is the right call and it also means the critic covers
the *height* axis of spatial reasoning and not the *bearing* axis — which is
where most spatial predicates live. Worth stating when the ablation result is
discussed.

Violations become natural language (`"cup: ['holding'] claimed but the box is
2.31 m away from the person"`), the top 12 are appended to the original prompt
along with the model's own previous answer, and the repair instruction asks for
a corrected JSON in the same format, keeping every correct object. `--max_repairs`
(default 1) bounds the rounds; only frames whose critic fired are re-sent.

---

## 4. Pipeline

```
per group of --videos_per_batch videos, per annotated frame:

  payload  = build_payload(...)                  ← Track A, unchanged
  text     = with_context(payload.text, query_graph(stage1_clips, frame_num))

  ── proposal round (pooled vLLM call over the whole group) ──
  resp     → parse_objects → objs
  store      objects_pre, raw_response_pre, violations_pre

  ── verify ──
  viol     = check_geometry(objs, person_corners, room_extent)

  ── repair round(s), only frames with viol, up to --max_repairs ──
  text'    = repair_prompt(text, _proposal_json(objs), viol)
  resp'    → parse_objects → objs'   (re-checked)

  emit       objects, violations_post, n_calls
```

Output: `<outputs.track_b>/<mode>/<model><tag>/<video>.mp4.pkl` — Track A's
schema plus `objects_pre`, `violations_pre`, `violations_post`, `n_calls`.
Ablation tags are appended automatically: `--no_critic` → `_nocritic`,
`--no_retrieval` → `_noretr`.

---

## 5. Results

Full split, 1,511 videos, Qwen3-VL-8B, standard decode. From
[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) §3.

**PredCls**

| method | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---:|---:|---:|---:|---:|---:|
| Track A | 51.7 | 30.5 | 62.8 | 47.2 | 46.5 | 40.0 |
| **Track B (+critic)** | **52.4** | **31.7** | **63.4** | **48.4** | **47.6** | **40.8** |
| Track B, critic removed | 52.1 | 31.1 | 63.2 | 47.9 | 47.3 | 40.0 |

**SGDet, 3D-IoU matched**

| method | unloc nc R@50 | unloc nc mR@50 | IoU.15 R@50 | IoU.15 mR@50 | IoU.25 R@50 | IoU.25 mR@50 | slots w/ 3D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Track A | 33.3 | 20.2 | 11.0 | 6.6 | 7.9 | 4.7 | **80.8** |
| **Track B (+critic)** | **33.5** | **23.8** | 10.7 | 7.7 | 7.7 | 5.5 | 79.5 |
| Track B, critic removed | 33.4 | 23.2 | **11.2** | **7.8** | **8.0** | **5.6** | 79.4 |

**The critic ablation, directly**

| mode | geometric violations, before → after | effect on recall |
|---|---|---|
| predcls | 41,077 → 31,075, down **24 %** | +0.3 wc R@20, +0.6 wc mR@20 |
| sgdet | 36,370 → 22,819, down **37 %** | +0.6 unlocalized mR@50, but **−0.5 R@50 at IoU 0.15** |

**Matched 150-video block, Qwen3-VL-8B**
([ICLR_THREE_TRACKS](../analysis/ICLR_THREE_TRACKS.md) §4):

| arm | decode | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Track B | standard | **52.2** | **31.4** | 62.1 | 49.5 | **48.4** | 45.3 |
| Track B | thinking | 27.2 | 16.4 | 43.2 | 38.2 | 37.8 | 38.2 |
| Track B, no critic | thinking | 27.3 | 16.2 | 43.3 | 38.1 | 37.8 | 37.6 |

Both thinking rows are **invalid** — §7.2.

### What the results establish

1. **The localization collapse is the headline.** Track B falls from 33.5
   unlocalized nc R@50 to **10.7 at IoU 0.15** to **7.7 at IoU 0.25**, on ~80 %
   of slots carrying a box. Relations survive; metric placement does not. This
   is the quantitative case for the training-based track.
2. **The critic demonstrably repairs geometry and that repair does not become
   metric accuracy.** 24 % of violations removed in PredCls, 37 % in SGDet; the
   payoff is +0.3 / +0.6 R / mR in PredCls and +0.6 unlocalized mR@50 in SGDet,
   against **−0.5 R@50 at IoU 0.15**. Report this as a **mechanism-versus-metric
   dissociation, not as a win.** The violations the critic catches are real; the
   corrections it induces move boxes in directions uncorrelated with the ground
   truth. Two readings are available and the data does not separate them: either
   the model repairs the predicate rather than the box (cheaper, and the prompt
   permits it), or the critic's contact/vertical geometry is a weaker proxy for
   correctness than for plausibility. Both are worth saying.
3. **Track B beats Track A on every relational column and loses the placement
   ones.** +0.7 wc R@20, +1.2 wc mR@20, +3.6 unlocalized SGDet mR@50; −0.3
   IoU.15 R@50 and −1.3 slots-with-3D. The agentic arm buys relational quality
   and does not buy placement.
4. **Track B is the best MLLM method in the project on headline PredCls
   recall** — 52.4, against 46.9 for the best unlocalized method and 74.8 for
   WorldWise++. The supervised ceiling stands **22.4 points** above it.
5. **Track 2 and track 3 are complementary, not ranked.** In the matched block
   Track B wins headline recall and the observed-object bucket (48.4 vs 47.7)
   while RAG-with-thinking wins the unobserved-non-trivial bucket by **8.5
   points** (53.8 vs 45.3). Localization helps what the camera can see;
   retrieval carries object permanence. That is real routing headroom, and
   testing it needs no GPU — merge the slot-level predictions and score the
   oracle.
6. **The `--no_retrieval` ablation is implemented but not reported.** The flag
   exists (`_noretr` tag) and would isolate `query_graph`'s contribution
   separately from the critic's. No scored cell for it appears in
   [analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) — treat it
   as **not run**.

---

## 6. Running

```sh
CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.track_b_agent.runner \
    --model_name qwen3vl_8b --mode sgdet \
    --video_list /data3/rohith/ag/splits/test_worldbbox_1511.txt
```

Scoring both ablation arms from the one run:

```sh
python -m lib.mllm.eval.score_run --method track_b --model qwen3vl_8b --mode sgdet
python -m lib.mllm.eval.score_run --method track_b --model qwen3vl_8b --mode sgdet \
    --objects_key objects_pre
```

Defaults: `--context_frames 2`, `--videos_per_batch 4`, `--max_new_tokens 1024`,
`--max_model_len 24576`, `--temperature 0.2`, `--top_p 0.95`, `--max_repairs 1`.
Note the lower temperature and much smaller token budget than
[Track A](MLLM_TRACK_A.md)'s (0.6 / 4096) — Track B is run to be reproducible
and to fit two rounds in the budget.

**Prerequisites:** the B4 caches (`build_caches`) and the Stage-1 graphs for
`query_graph` (which defaults to reading `qwen25vl_7b` graphs regardless of the
generation model — see §7.5).

---

## 7. Known defects and caveats

1. **Backbone confound.** Track B runs Qwen3-VL-8B; the full-split unlocalized
   rows run Qwen2.5-VL-7B. Cross-track claims belong in the 150-video matched
   block or nowhere.
2. **Both Track B thinking cells are invalid, not merely weak.** The reasoning
   trace shares `max_new_tokens` with the answer and Track B asks for one JSON
   array over the whole frame, so at `--max_new_tokens 1024` the model spends
   the budget on the trace. Track B at 27.2 and 0.3 in SGDet are budget
   failures. `--max_new_tokens 1024` makes Track B *more* exposed to this than
   Track A at 4096.
3. **The repair prompt permits predicate edits as well as box edits.** "correct
   the box **or** the predicate that is inconsistent with the 3D layout" — so a
   violation can be discharged by weakening the relation rather than by fixing
   the geometry. Given finding §5.2, this is the first thing to check before
   the ablation is written up, and it is checkable offline from the stored
   `objects_pre` / `objects` pairs.
4. **The critic covers height, not bearing.** `front` / `behind` / `side` are
   unchecked because the person's facing is unknown, so the largest family of
   spatial predicates is outside the critic's reach. The 24–37 % violation
   reduction is a reduction in the *checkable* subset.
5. **`query_graph` hard-defaults to `qwen25vl_7b` Stage-1 graphs.**
   `load_stage1(video_id, graph_dir, model="qwen25vl_7b")` — so a Track B run on
   Qwen3-VL-8B retrieves over a graph built by a *different* model. That is
   defensible as a fixed retrieval corpus, but it is a cross-model dependency
   that a reviewer will notice, and it means the `--no_retrieval` ablation would
   also be ablating a backbone mismatch.
6. **The tool order is fixed by the driver.** The model does not choose tools.
   Call it a fixed tool loop.
7. **~20 % of slots carry no 3D box**, dropped from τ > 0 matching by
   `require_pred_3d` rather than scored as misses. Always print slots-with-3D
   next to an IoU-matched number.
8. **`--no_retrieval` has no scored cell.** Do not report a number for it.
