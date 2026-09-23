# `caption_all` — per-object prompting with a caption transcript

The second rung of the unlocalized ladder: [`zero_shot`](MLLM_ZERO_SHOT.md) plus
a text transcript of what happens in the video. No retrieval, no graph, no
embedding model — the captions are simply prepended to every relationship
prompt.

Runner: [lib/mllm/methods/caption_all/runner.py](../lib/mllm/methods/caption_all/runner.py) ·
Base: [lib/mllm/base_processor.py](../lib/mllm/base_processor.py) ·
Stage-1 source: [lib/mllm/methods/graphs/runner.py](../lib/mllm/methods/graphs/runner.py) ·
Track: [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) ·
Method key `caption_all`

In the legacy backbone-breadth tables of
[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) §2d this
method appears under the name **"Caption"** (the pragya rows call the same
pipeline `subtitle_all`).

---

## 1. Design, and why it is shaped this way

`caption_all` isolates one variable: **does natural-language summary of the
video help a per-object relationship prompt?** It is `zero_shot` with a text
prefix and nothing else changed — same visual construction, same per-object
query format, same parser, same verification pass, same output schema. Any
difference in the tables is attributable to the captions.

The captions come from the **Stage-1 graph build**, not from a fresh pass. That
build (`methods/graphs/runner.py`) segments each video into clips aligned to the
annotated frames and asks the VLM twice per clip: once with
`CAPTION_GENERATION_PROMPT` for a concise description of the main action, and
once with `GRAPH_PROMPT` for a JSON `{entities, actions, scenes}` structure.
`caption_all` reads only the `caption` field of each clip and discards the graph;
[`rag_all`](MLLM_RAG_ALL.md) reads the graph. Sharing the Stage-1 artifact is
what makes the two rungs a clean comparison — they are looking at the same
underlying VLM description of the video, once as free text and once as a
retrievable structure.

**Captions are ordered by proximity to the target frame.** `_format_caption_context`
sorts by `abs(caption_frame - target_frame)` when a frame index is available and
falls back to time order otherwise. The prompt prefix is:

```
The following captions describe what happens in this video:
[Frame 000012] ...
[Frame 000048] ...
```

**No-caption videos still run.** A missing Stage-1 pickle logs a warning and the
method proceeds with an empty prefix, degrading exactly to `zero_shot`. This is
the safe direction and it means a partial Stage-1 build never silently drops
videos from the denominator.

---

## 2. Pipeline

```
process_video(video_id)
  0. skip if output pkl exists                            ← resumability / sharding
  1. annotations                 get_final_data_lite(vid)
  2. captions + clip intervals   from graphs/<model>/<vid>.pkl   (caption field only)
  3. video tensor                all frames, ::8 if > 120 frames
  4. object set                  predcls: annotation objects
                                 sgdet:   estimate_objects_from_captions
                                          ∩ AG vocab, + Yes/No scores
  5. queries                     frames × objects
  6. visual context              _build_annotated_context (≤ 15 frames, ≤ 512k px)
                                 _build_query_context_map → one tensor per frame
  7. prompts                     caption_ctx_cache[frame_idx] + query prompt
  8. ONE batched vLLM call
  9. parse → verify (Yes/No, caption context reused as the verification prefix)
 10. redistribute → pickle.dump
```

`caption_ctx_cache` memoises the formatted prefix per frame index, so the
caption list is sorted once per frame rather than once per (frame, object).

**SGDet differs from `zero_shot` here in a way that matters.** The object-set
estimation call receives the real caption transcript rather than
"(No captions available for this video.)". The SGDet object set is therefore a
genuinely different set from `zero_shot`'s, which is part of why the two methods
are not interchangeable in SGDet even though they tie in PredCls.

---

## 3. Output schema

Byte-identical to [`zero_shot`](MLLM_ZERO_SHOT.md) §3 — one pickle per video
with `frames[stem]["predictions"][i]` carrying `object`, `raw_response` and the
three `{label: score}` dicts. Scored through the same
[dump_adapter](../lib/mllm/eval/dump_adapter.py) with no method-specific code.

---

## 4. Results

Full split, 1,511 videos, Qwen2.5-VL-7B, PredCls
([ICLR_THREE_TRACKS](../analysis/ICLR_THREE_TRACKS.md) §2a):

| metric | `caption_all` | `zero_shot` | `rag_all` |
|---|---:|---:|---:|
| wc R@20 | 45.3 | **46.9** | **46.9** |
| wc mR@20 | **26.1** | 25.1 | 26.0 |
| nc R@50 | 59.9 | 60.9 | **61.1** |
| nc mR@50 | 43.6 | 43.1 | **43.8** |
| OO nc mR@50 | 40.6 | 40.3 | **40.8** |
| OU-nt nc mR@50 | 44.1 | 42.8 | **44.6** |
| legacy uF1 | 48.2 | not scored | **49.9** |

Legacy-protocol PredCls, 1,511 videos, Qwen2.5-VL-7B (§2d):

| Attention F1 | Contacting F1 | Spatial F1 | μF1 | MF1 |
|---:|---:|---:|---:|---:|
| 53.2 | 47.3 | 44.7 | 48.3 | 23.2 |

**SGDet: not scored.** The full-split run stood at 634 of 1,511 videos as of
2026-09-22 and was still generating.

### Cross-backbone, legacy protocol (§2d, "Caption" rows)

| Model | PredCls μF1 | SGDet μF1 |
|---|---:|---:|
| InternVL2.5-8B | 29.3 | 23.8 |
| KeyeVL-1.5-8B | 25.5 | 19.2 |
| MiniCPM-V4.5 | 38.1 | 27.4 |
| Ovis2.5-9B | 48.7 | **35.6** |
| Qwen2.5-VL-7B | 47.2 | 42.7 |
| Qwen3-VL-30B-A3B | **47.8** | not run |

(1,734 videos; RAG counterparts in [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) §3d.)

### What the results establish

1. **Captions cost headline recall and buy a little tail recall.** −1.6 wc R@20
   against both neighbours, +1.0 wc mR@20 against `zero_shot`. `caption_all`
   holds the best wc mR@20 of all four full-split rungs (26.1) by 0.1 over
   `rag_all` — inside noise, but it is the only column it wins.
2. **Captioning loses to retrieval on five of six backbones in PredCls μF1**
   and four of five in SGDet, typically by 0.5 to 3.6 points.
3. **The single large gap is a captioning failure, not a retrieval success.**
   KeyeVL-1.5-8B drops 15.5 μF1 in PredCls and 18.0 in SGDet relative to RAG
   purely because its captioning *recall* collapses: precision 43.7 / 37.6 /
   55.8 against recall 18.5 / 15.0 / 19.5. The captioning path is not
   inaccurate, it is silent. Do not quote KeyeVL as evidence that retrieval
   carries information.
4. **Two backbones prefer captioning.** Qwen3-VL-30B-A3B wins PredCls μF1 with
   captions (47.8 vs 44.5) and Ovis2.5-9B wins SGDet μF1 with captions
   (35.6 vs 34.5). The method ordering is not backbone-invariant.

---

## 5. Running

```sh
CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.caption_all.runner \
    --model_name qwen25vl_7b --mode predcls --skip-verification \
    --video_list /data3/rohith/ag/splits/test_worldbbox_1511.txt
```

Scoring:

```sh
python -m lib.mllm.eval.score_run --method caption_all --model qwen25vl_7b --mode predcls
```

**Hard prerequisite:** the Stage-1 build must have completed *for the same model
key*. Captions live at `graphs/<model_name>/<video_id>.pkl`; a missing pickle
degrades the video to `zero_shot` behaviour silently.

---

## 6. Known defects and caveats

1. **A missing Stage-1 pickle degrades silently to `zero_shot`.** It logs a
   warning and continues. If Stage-1 coverage is partial, `caption_all` is a
   mixture of two methods and the table will not say so. Check Stage-1 PKL
   count against the split before scoring.
2. **Silent per-video failures produce false "done" runs** — the shared
   `ActionGenomeBaseProcessor.run()` defect. Measure completion by PKL count.
3. **The captions are generated by the same model being evaluated.** Stage-1
   runs with `--model_name`, so a weak captioner is evaluated on its own weak
   captions. That is the intended design (it keeps the comparison training-free
   and self-contained) but it means the caption rows conflate captioning
   ability with relational ability — which is precisely what the KeyeVL cell
   shows.
4. **SGDet object estimation is caption-driven**, so a video with no captions
   gets a different (frames-only) object set. In a partially-built Stage-1 that
   heterogeneity lands inside a single SGDet row.
