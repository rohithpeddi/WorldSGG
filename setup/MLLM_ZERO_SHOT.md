# `zero_shot` — the context-free per-object prompt

The floor of the unlocalized ladder. The VLM sees the annotated frames and one
relationship question per object, and nothing else: no captions, no scene graph,
no retrieval. Everything above it in [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md)
is this method plus a context source, so `zero_shot` is the control that says
how much any of that context is actually worth.

Runner: [lib/mllm/methods/zero_shot/runner.py](../lib/mllm/methods/zero_shot/runner.py) ·
Base: [lib/mllm/base_processor.py](../lib/mllm/base_processor.py) ·
Track: [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) ·
Method key `zero_shot`

> **Naming.** [analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md)
> labels these rows "Graph-RAG (zero_shot)". That label is wrong.
> `ActionGenomeZeroShotProcessor` never opens the graph pickle —
> `process_video`'s own docstring says "No caption or graph context is
> injected", and `graph_dir` is only passed through to the base class and left
> unused. Graph-RAG is [`rag_all`](MLLM_RAG_ALL.md). The numbers are correctly
> attributed; only the display name needs fixing.

---

## 1. Design, and why it is shaped this way

Three decisions define the method.

**One query per (frame, object), not one query per frame.** Every other
formulation the project tried asks the model for a JSON array covering the whole
frame. This one asks a single three-field question about a single object. That
costs F × O calls per video instead of F, and it buys two things: the answer is
short enough that a truncation never loses more than one object, and — as it
turned out — **immunity to the thinking-budget defect** that invalidates every
array-style thinking cell in the project
([TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) §5.2). The per-object format is
the reason RAG is the only valid thinking cell.

**All video-level objects on every annotated frame, not just the visible ones.**
The prompt states plainly that the object "may or may not be visible in the
target frame" and that "a person IS visible". An object the camera cannot see
still gets a question, and the model is expected to answer from the video
context. This is what makes the method scorable on the OU (observed→unobserved)
buckets at all — a method that only answered about visible objects would score
0 there by construction.

**Target frame first, context after.** The visual payload is
`[target_frame] ++ annotated_context`, and the prompt says so: "The first frame
is the specific moment to analyze. The remaining frames provide surrounding
context." That convention is load-bearing and is shared verbatim by
`caption_all` and `rag_all`, so the three rungs differ only in the text prefix,
never in the visual construction.

The three predicate heads are asked in one call with different cardinalities —
attention exactly one, contacting one-or-more, spatial one-or-more — matching the
annotation schema rather than flattening to a single multi-label question.

---

## 2. Pipeline

```
process_video(video_id)
  0. skip if <out>/<mode>/<model>/<vid>.pkl exists            ← resumability / sharding
  1. annotations              get_final_data_lite(vid)  → bbox_frames, objects
  2. video tensor             all frames, ::8 if > 120 frames
  3. frame-local clips        one per annotated frame — verification only,
                              skipped entirely under --skip-verification
  4. object set               predcls: annotation objects
                              sgdet:   VLM-estimated ∩ AG vocab, + Yes/No scores
                                       (estimation runs with an EMPTY caption list)
  5. queries                  frames × objects, one prompt each
  6. visual context           _build_annotated_context (≤ 15 frames, ≤ 512k px)
                              _build_query_context_map → one tensor per frame
  7. ONE batched vLLM call    max_new_tokens = 128 per prompt
  8. parse                    JSON → validate against the three label sets
  9. verify                   one Yes/No prompt per predicted label → yes_prob
 10. redistribute             flat list → per-frame dicts
 11. pickle.dump
```

**SGDet without captions.** `get_objects_for_mode` normally estimates the object
set from captions. `zero_shot` passes an empty caption list, so the estimation
prompt literally receives "(No captions available for this video.)" and the VLM
must estimate the object set from the frames alone. The estimate is then
intersected with the canonical AG vocabulary — **not** with the GT annotations —
and each surviving object gets a discriminative Yes/No score. That intersection
choice is what keeps SGDet honest: the method is never handed the answer.

**Verification is what produces scores.** Generation is categorical; R@K needs a
ranking. Each predicted label is re-asked as a Yes/No question against the target
frame and the single-token logprob becomes `yes_prob`. Every predicate the model
did not predict scores 0. With `--skip-verification` every predicted label gets
1.0, which flattens the ranking — prefer R@50 / R@100 for these rows.

---

## 3. Output schema

One pickle per video at `<output_dir>/<mode>/<model_name>/<video_id>.pkl`:

```python
{
  "video_id": str, "mode": str, "model_name": str,
  "estimation_meta": dict | None,          # sgdet only
  "video_objects": [str],
  "num_frames_processed": int,
  "frames": {"<frame_stem>": {
      "objects": [str],
      "predictions": [{"object", "raw_response",
                       "attention", "contacting", "spatial"}]   # {label: score}
  }}
}
```

Read by [lib/mllm/eval/dump_adapter.py](../lib/mllm/eval/dump_adapter.py), which
converts it into the exact `tools/dump_predictions.py` record format. No
adapter changes: the schema is shared with `caption_all` and `rag_all`.

---

## 4. Results

Full split, 1,511 videos, Qwen2.5-VL-7B, PredCls. From
[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) §2a.

| metric | `zero_shot` | best rung in the track |
|---|---:|---:|
| wc R@20 | **46.9** | 46.9 (`rag_all`, tie) |
| wc mR@20 | 25.1 | 26.1 (`caption_all`) |
| nc R@50 | 60.9 | 61.1 (`rag_all`) |
| nc mR@50 | 43.1 | 43.8 (`rag_all`) |
| OO nc mR@50 | 40.3 | 40.8 (`rag_all`) |
| OU-nt nc mR@50 | 42.8 | 44.6 (`rag_all`) |
| legacy uF1 | **not scored** | 49.9 (`rag_all`) |

Legacy-protocol PredCls, 1,511 videos, Qwen2.5-VL-7B
([ICLR_THREE_TRACKS](../analysis/ICLR_THREE_TRACKS.md) §2d):

| Attention F1 | Contacting F1 | Spatial F1 | μF1 | MF1 |
|---:|---:|---:|---:|---:|
| 55.8 | 48.3 | 45.9 | 49.8 | 23.2 |

**SGDet: not scored.** The full-split run had produced 1,298 of 1,511 videos as
of 2026-09-22 and was still generating. There is no `zero_shot` SGDet number.

### What the results establish

1. **`zero_shot` ties the best method in the track on headline recall.** 46.9 wc
   R@20, equal to `rag_all` and 1.6 above `caption_all`. Adding a caption
   transcript, a retrieved scene graph, or a strategy router over the top of
   both moves the headline by less than the noise floor.
2. **It is behind on every tail and bucket column, but only slightly.** −1.0
   wc mR@20 and −1.8 OU-nt nc mR@50 against `rag_all`. Context helps a little
   where the camera cannot see; it does not help where it can.
3. **Its legacy attention F1 is the best of the three full-split rows** (55.8
   vs 53.6 for `rag_all` and 53.2 for `caption_all`), while its micro-F1 ties
   `rag_all` at 49.8–49.9. The per-group picture is not a clean ordering
   either.

The honest framing: this is the row that makes the track's negative result
legible. Report it first, not last.

---

## 5. Running

```sh
CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.zero_shot.runner \
    --model_name qwen25vl_7b --mode predcls --tensor_parallel_size 1 \
    --skip-verification \
    --video_list /data3/rohith/ag/splits/test_worldbbox_1511.txt
```

Scoring:

```sh
python -m lib.mllm.eval.score_run --method zero_shot --model qwen25vl_7b --mode predcls
```

Long jobs go through `scripts/remote/run_mllm_job.sh` (setsid nohup +
`status.json`); the queue is `scripts/remote/run_mllm_sequence.sh`. Config:
[configs/mllm/server.yaml](../configs/mllm/server.yaml) (UTD) /
[configs/mllm/pragya.yaml](../configs/mllm/pragya.yaml) (IITD), selected with
`$WSGG_MLLM_CONFIG` or `--config`.

**Sharding.** Step 0 skips any video whose PKL already exists, so disjoint
`--video_list` shards can safely overlay one output directory.

---

## 6. Known defects and caveats

1. **Silent per-video failures produce false "done" runs.**
   `ActionGenomeBaseProcessor.run()` wraps each video in try/except and
   continues, so a run exits `rc = 0` having written fewer PKLs than requested.
   **Measure completion by PKL count against the split, never by exit code.**
2. **`graph_dir` is accepted and unused.** The constructor takes it and forwards
   it to the base class purely for signature compatibility with the other
   runners. This is the source of the "Graph-RAG (zero_shot)" mislabel.
3. **`max_new_tokens = 128` is hard-coded** for the relationship call. It is
   ample for the three-field JSON answer under standard decode, but it is not a
   thinking budget — a thinking model on this path would spend it on the trace.
   The per-object format makes the failure graceful (one object lost, not a
   frame), but the cell would still be invalid.
4. **The `::8` frame subsample above 120 frames** applies to the whole-video
   tensor. The per-query context comes from `_build_annotated_context`
   (≤ 15 frames, ≤ 512k pixels) instead, so the subsample affects the SGDet
   object-estimation call and little else. The pixel cap is deliberate: raw HD
   frames blow up the visual-token count and hang the vLLM scheduler at
   `Processed prompts: 0%`.
