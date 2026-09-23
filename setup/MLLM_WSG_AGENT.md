# `wsg_agent` — per-object strategy routing over Graph-RAG

The top rung of the unlocalized ladder. It takes [`rag_all`](MLLM_RAG_ALL.md)
and inserts one planning step in front of it: a **Strategy Router** that asks the
VLM to classify every object in the video into one of three inference
strategies, then executes each group through a different evidence-assembly path.

Runner: [lib/mllm/methods/wsg_agent/runner.py](../lib/mllm/methods/wsg_agent/runner.py) ·
**Full architecture document: [docs/WSG_AGENT.md](../docs/WSG_AGENT.md)** ·
Track: [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) ·
Method key `wsg_agent`

This file is the design-and-results summary for the setup set;
[docs/WSG_AGENT.md](../docs/WSG_AGENT.md) Part I is the line-by-line
architecture and Part II is the proposed v2 redesign.

**`wsg_agent` is in track 2, not track 3.** It has "agent" in the name and it
plans, but it emits no 3D box, so it is scored class-only. The agentic *localized*
method is [Track B](MLLM_TRACK_B.md).

---

## 1. The claim being tested

That a single retrieval policy is wrong for a heterogeneous object set — a held
cup needs no scene-graph retrieval, a background table does, and a
picked-up-then-put-down blanket needs temporal aggregation — and that letting
the model choose the policy per object beats applying one policy uniformly.

```
zero_shot      frames → relationships                       (no context)
caption_all    + video captions                             (text context)
rag_all        + scene-graph retrieval per query            (Graph-RAG)
wsg_agent      + per-object strategy routing over rag_all   ← this document
```

**`wsg_agent` is a strict superset of `rag_all`.** Objects routed `RAG` go
through the unmodified parent pipeline (`_batch_all_video_queries`, inherited,
not overridden). Only the `DIRECT` and `TEMPORAL` branches are new code. If the
router returned `RAG` for every object, `wsg_agent` would reduce exactly to
`rag_all` plus one wasted call. That nesting is deliberate: it makes the router
the only variable.

---

## 2. Design decisions worth defending

**The router is one text-only call per video.** No frames, built from the
caption transcript and the object list, 256 tokens. Negligible overhead against
a budget dominated by verification (typically 3–5× the answering cost).

| Strategy | Router's stated criterion |
|---|---|
| `DIRECT` | commonly held / worn / manipulated (phone, cup, food); direct visual inspection suffices |
| `RAG` | background, occluded, or needs broader scene understanding (table, door, window) |
| `TEMPORAL` | interaction changes over time or spans segments (picking up then putting down) |

**Failure handling is total and degrades to the parent.** Any JSON parse error,
or any object missing from the response, falls back to `RAG`. A router failure
produces `rag_all` behaviour rather than a crash — the safe direction, and the
reason the method never scores *below* its parent by much.

**Routing granularity is per object per video**, not per (frame, object). An
object routed `TEMPORAL` is treated temporally at every annotated frame.

**One visual construction is shared by all three branches**, computed once per
frame and reference-shared across every object queried on that frame:
`[target_annotated_frame] ++ annotated_context` (≤ 15 bbox-overlaid frames,
≤ 512k pixels). The pixel cap is deliberate and documented in-code: raw HD
frames blow up the visual-token count and hang the vLLM scheduler at
`Processed prompts: 0%`.

**`strategy_assignments` is written into the output** so routing distribution
against per-strategy recall is available as an ablation. That is the analysis
hook that would justify the router's existence.

---

## 3. What the three branches actually do

This is the part that must be read carefully, because **the implementation and
the module docstring disagree**.

| | generation text prefix | generation visual | verification visual |
|---|---|---|---|
| `DIRECT` | captions | `query_ctx_map[frame]` | annotation clip for this frame |
| `TEMPORAL` | captions | `query_ctx_map[frame]` | 3-clip temporal concat, ≤ 60 frames |
| `RAG` | retrieved graph node context | `query_ctx_map[frame]` | retrieved / annotation clip |

**As implemented, `DIRECT` and `TEMPORAL` issue byte-identical generation
calls.** `_process_temporal_entries` builds the neighbour-concatenated
`temporal_clip`, but on the normal path it then sends `query_ctx_map[frame_stem]`
as the visual input and `sub_ctx + q["prompt"]` as the text — the same two
values `_process_direct_entries` sends. The temporal clip is appended to
`verify_clips` and only reaches the model during the Yes/No verification pass.

The docstring claims `TEMPORAL` aggregates evidence from the target frame's clip
*plus* its temporal neighbours for the answer. It does not — the aggregation
affects verification only. **The three-way router collapses to a two-way split
at prediction time**: graph context (`RAG`) versus caption context
(`DIRECT`/`TEMPORAL`), with `TEMPORAL` differing only in how strictly its
predictions are re-scored.

This is not necessarily wrong as an ablation, but it must not be described as
three distinct inference paths in a paper without either fixing the branch or
restating the claim.

---

## 4. Pipeline

Condensed from [docs/WSG_AGENT.md](../docs/WSG_AGENT.md) §3.

```
process_video(video_id)
  0. skip if output pkl exists
  1. annotations → video_data, annotated_objects
  2. Stage-1 graph cache → video_graph, entity_graph, captions, clip_intervals
                                                            [HARD GATE]
  3. graph embeddings (entity keys + node content)
  4. visual tensors: full video (::8 if > 120 frames) + per-frame clips
  5. object set: predcls annotations, or sgdet VLM-estimated ∩ vocab
  ╔══════════════════════════════════════════════════════╗
  6. PLAN — route_strategies(): one text-only call, 256 tok
            {object → DIRECT | RAG | TEMPORAL}
  ╚══════════════════════════════════════════════════════╝
  7. query cross-product frames × objects, partitioned by strategy;
     entry_order[] records (strategy, index) for order-preserving reassembly
  8. shared visual context (one tensor per frame stem)
  ╔══════════════════════════════════════════════════════╗
  9. ACT — three executors batched independently
        9a DIRECT    _process_direct_entries
        9b RAG       _batch_all_video_queries   (parent, unmodified)
        9c TEMPORAL  _process_temporal_entries
  ╚══════════════════════════════════════════════════════╝
 10. reassemble into original (frame, object) order
 11. parse → validate → verify (Yes/No per label → yes_prob)
 12. redistribute per frame → pickle.dump
```

Output schema is `rag_all`'s plus `strategy_assignments` (video-level) and a
per-prediction `strategy` key, so it scores through the unmodified
[dump_adapter](../lib/mllm/eval/dump_adapter.py) —
[score_all.py](../lib/mllm/eval/score_all.py) lists `wsg_agent` in its
`METHODS` tuple.

---

## 5. Results

PredCls, full split, Qwen2.5-VL-7B, **n = 1458 of 1,511 videos**. From
[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) §2a.

| metric | `wsg_agent` (n=1458) | `rag_all` (parent) | `zero_shot` |
|---|---:|---:|---:|
| wc R@20 | 46.3 | **46.9** | **46.9** |
| wc mR@20 | 25.8 | **26.0** | 25.1 |
| nc R@50 | 60.7 | **61.1** | 60.9 |
| nc mR@50 | 43.5 | **43.8** | 43.1 |
| OO nc mR@50 | 40.7 | **40.8** | 40.3 |
| OU-nt nc mR@50 | 43.6 | **44.6** | 42.8 |

**SGDet: not run.** The job was queued, 0 of 1,511 videos generated as of
2026-09-22. Cost estimate ~9 h GPU.

**Legacy uF1: not scored.**

### What the results establish

1. **The router does not beat its own parent on any column.** `wsg_agent` is
   0.1 to 1.0 points *below* `rag_all` everywhere, with a strictly larger call
   budget. The heterogeneous-policy hypothesis is not supported by this
   implementation.
2. **It does not lose badly either, and that is a structural fact, not a
   finding.** The router's fallback is `RAG`, and §3 shows `DIRECT` and
   `TEMPORAL` collapse to a caption-prefixed variant of the same call. The
   method is bounded between `rag_all` and `caption_all`, which is exactly the
   band it lands in (45.3–46.9 wc R@20). A null result from a method that
   cannot structurally move is weak evidence against the hypothesis.
3. **It is the cell to cut if space is short.** It is redundant with
   [Track B](MLLM_TRACK_B.md), which is the project's real agentic arm, tests a
   genuinely ablatable component, and carries a localized measurement.
4. **The n = 1458 must be reported.** See §6.1.

The defensible write-up: report `wsg_agent` as a negative result *about this
router*, note that the TEMPORAL branch does not currently influence prediction,
and point at [docs/WSG_AGENT.md](../docs/WSG_AGENT.md) Part II (the v2
evidence-routed-slots design and its no-GPU oracle-headroom experiment) as the
version of the claim that is still open.

---

## 6. Known defects

Four, all documented at length in [docs/WSG_AGENT.md](../docs/WSG_AGENT.md) §9.

1. **TEMPORAL clip concatenation crashes — this is the missing 53 videos.**
   `_process_temporal_entries` does `torch.cat(clip_tensors_to_cat, dim=0)`
   with no spatial-dimension guard. Each per-annotation clip is resized by
   `resize_video`, whose target resolution is derived from the clip's **frame
   count**; neighbouring clips cover different intervals, so they have
   different `nframes`, so they are resized to different H×W, so the `cat`
   fails. Both sibling code paths guard against exactly this
   (`_build_annotated_context` interpolates to the first frame's dims,
   `_prepend_target_frame` interpolates before its own `cat`) and TEMPORAL does
   not. Fix: interpolate to a common H,W before concatenating.
   **Impact:** the 53 videos missing from the PredCls cell.
2. **Silent per-video failures produce false "done" runs.**
   `ActionGenomeBaseProcessor.run()` catches per video and continues, so the
   run exited `rc = 0` at 1458/1511. **Completion must be measured by PKL count
   against the split, never by exit code.** Defect 1 is only visible through
   this one.
3. **Broad exception clause in the router.**
   `except (json.JSONDecodeError, TypeError, Exception)` — `Exception` subsumes
   the others, so every ordinary error becomes silent all-`RAG` routing.
   Harmless to correctness (the fallback is sound) but a systematically broken
   router is indistinguishable from `rag_all`. Given §5.1, this cannot be ruled
   out from the numbers alone; check `strategy_assignments` before writing any
   claim about routing behaviour.
4. **Duplicated `scenes` in the embedding text** —
   `Vgent.precompute_graph_embeddings` concatenates `scenes` twice,
   double-weighting scene terms in retrieval ranking. Inherited from
   [`rag_all`](MLLM_RAG_ALL.md) §7.1.

Defects 1 and 3 are both localised to code paths that the results in §5 depend
on. Neither is fixed. State the n and the TEMPORAL-branch caveat in the paper.

---

## 7. Running

```sh
CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.wsg_agent.runner \
    --model_name qwen25vl_7b --mode predcls --randomize \
    --video_list /data3/rohith/ag/splits/test_worldbbox_1511.txt
```

Scoring:

```sh
python -m lib.mllm.eval.score_run --method wsg_agent --model qwen25vl_7b --mode predcls
```

`--randomize` time-seeds a shuffle so multiple GPUs can share one output
directory; step 0's skip-if-exists makes disjoint or overlapping shards safe.
`--skip-verification` substitutes `yes_prob = 1.0`.

**Hard prerequisite:** Stage-1 graphs must exist for the same model key. A
missing or zero-node graph skips the video entirely.
