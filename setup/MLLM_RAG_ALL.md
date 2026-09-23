# `rag_all` — Graph-RAG over a VLM-built video graph

The third rung of the unlocalized ladder and **the method the paper means by
"Graph-RAG"**. A first pass builds a structured graph of the video; a second
pass retrieves the graph nodes relevant to each object and prepends them to the
relationship prompt.

Runner: [lib/mllm/methods/rag_all/runner.py](../lib/mllm/methods/rag_all/runner.py) ·
Stage-1: [lib/mllm/methods/graphs/runner.py](../lib/mllm/methods/graphs/runner.py) ·
Retrieval: [lib/mllm/core/vgent.py](../lib/mllm/core/vgent.py), [lib/mllm/core/retrieval.py](../lib/mllm/core/retrieval.py) ·
Prompts: [lib/mllm/core/prompts.py](../lib/mllm/core/prompts.py) ·
Track: [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) ·
Method key `rag_all`

> [analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) labels the
> `zero_shot` rows "Graph-RAG". That is a mislabel — see
> [MLLM_ZERO_SHOT.md](MLLM_ZERO_SHOT.md). Graph-RAG is this document's method.
> In §2d of that file this method appears as **"RAG"**.

---

## 1. Design, and why it is shaped this way

The hypothesis is object permanence: a relationship question about an object the
camera cannot see at the target frame should be answerable from a *structured
record of what happened elsewhere in the video*. Captions
([`caption_all`](MLLM_CAPTION_ALL.md)) give that record as undifferentiated
prose and hand the model all of it; Graph-RAG gives it as a graph and hands the
model only the part that matches the query.

Four decisions follow from that.

**Stage-1 is a separate, cached build.** The graph is produced once per video by
`methods/graphs/runner.py` and stored; the relationship pass never regenerates
it. This is what lets `caption_all` and `rag_all` read the *same* Stage-1
artifact — the two rungs differ in how they consume it, not in what the VLM saw.

**Retrieval is embedding-based, not LLM-based.** Node selection runs entirely
through a BGE text encoder (`bge_large`, `BAAI/bge-large-en-v1.5`); no VLM call
is spent choosing nodes. Only keyword extraction (before) and node checking
(after) are LLM calls.

**The prompt is deduplicated to O(objects), the answer is not.** The
relationship prompt is a pure function of the object name, so across F frames ×
O objects there are only O unique prompts. Keyword extraction, node retrieval
and node checking therefore run on unique prompts only — roughly an F-fold
reduction — while the final answering call stays per (frame, object), which is
what makes the predictions actually vary across frames. Without that asymmetry
the method would emit one answer per object for the whole video.

**Per-object prompting, inherited from `zero_shot`.** The relationship question,
its three-head cardinality, the target-frame-first visual convention and the
Yes/No verification pass are unchanged. Everything new is the text prefix.

That per-object format is also why **`rag_all` is the only method in the project
whose thinking cell is valid**: the reasoning trace shares `max_new_tokens` with
the answer, and a three-field JSON object survives the squeeze where a
whole-frame JSON array does not.

---

## 2. Stage-1 — the graph and caption build

[lib/mllm/methods/graphs/runner.py](../lib/mllm/methods/graphs/runner.py),
output `graphs/<model_name>/<video_id>.pkl`, a list of per-clip records.

Each video is segmented into clips aligned to the annotated frames. Per clip the
VLM is called twice:

| Prompt | Output |
|---|---|
| `CAPTION_GENERATION_PROMPT` | one concise caption of the main action, visual content only |
| `GRAPH_PROMPT` | strict JSON `{"entities": [{entity name, description}], "actions": [{entity name, action description}], "scenes": [{location}]}` |

`GRAPH_PROMPT` constrains action entity names to the entity list, so the graph
is internally consistent, and asks for first-person description when the video
is shot from a first-person viewpoint.

`rag_all` then merges the per-clip graphs into one video-level `nx.DiGraph`
(`load_precomputed_graphs`):

```python
offset = len(video_graph)
video_graph.add_node(node_id + offset, **clip_graph.nodes[node_id],
                     source_annotated_frame=annotated_frame)
```

The ID offset keeps node IDs unique across clips and `source_annotated_frame`
is what later maps a retrieved node back to the right video clip. An
`entity_graph` (`entity name → {node ids}`) is built alongside as an inverted
index.

**Hard gate:** a missing or zero-node graph pickle **skips the video entirely**.
Unlike `caption_all`, `rag_all` cannot run standalone. Stage-1 must complete for
the model being evaluated before this method is launched, and a partial Stage-1
shows up as missing videos rather than as degraded ones.

---

## 3. Pipeline

```
process_video(video_id)
  0. skip if output pkl exists
  1. annotations              get_final_data_lite(vid)
  2. Stage-1 graph            merge clips → video_graph, entity_graph,
                              captions, clip_intervals   [HARD GATE]
  3. graph embeddings         precompute_graph_embeddings(video_graph, entity_graph)
                                 entity-key embeddings + node-content embeddings
  4. video tensor             all frames, ::8 if > 120 frames; split into
                              chunk_size clips
  5. object set               predcls: annotation objects
                              sgdet:   estimated from captions ∩ AG vocab
  6. queries                  frames × objects   →   O unique prompts
  7. visual context           _build_annotated_context + _build_query_context_map
     ┌──────────── deduplicated, unique prompts only ─────────────┐
  8. Step 1/4  keyword extraction        batch_extract_keywords   (1 LLM call)
  9. Step 2/4  node retrieval            retrieve_nodes_with_cache (0 LLM calls,
                                          embeddings only)
 10. Step 3/4  node checking             Q1 "is '<obj>' visible or interacted with
                                          in this segment?" / Q2 "is a person
                                          visible?"                (batched LLM)
     └────────────────────────────────────────────────────────────┘
 11. Step 4/4  per (frame, object) answer:
                  text   = node_context + relationship prompt
                  visual = query_ctx_map[frame]
                  max_new_tokens = 128
 12. parse → verify (Yes/No per label → yes_prob) → redistribute → pickle.dump
```

**What the retrieved context actually looks like.** Only the **top** node is
used, and only three of its fields:

```
Relevant scene context from video analysis:
<entities>; <actions>; <scenes>

<the standard relationship prompt>
```

**Node ranking** uses `count_and_sort_filtered` over the node-checking answers,
falling back to the raw retrieval order when nothing survives the check. The
winning node index also selects the clip tensor used for the **verification**
pass — so retrieval affects both what the model reads and what it re-examines.

Note the asymmetry worth stating in the paper: the RAG generation prompt carries
**graph context but no captions**; `caption_all`'s carries **captions but no
graph**. Captions re-enter the RAG path only as the verification context prefix.

---

## 4. Output schema

Identical to [`zero_shot`](MLLM_ZERO_SHOT.md) §3 and
[`caption_all`](MLLM_CAPTION_ALL.md) §3. Scored through the shared
[dump_adapter](../lib/mllm/eval/dump_adapter.py) with no method-specific code.

---

## 5. Results

### 5a. PredCls, full split, 1,511 videos, Qwen2.5-VL-7B

([ICLR_THREE_TRACKS](../analysis/ICLR_THREE_TRACKS.md) §2a)

| metric | `rag_all` | `zero_shot` | `caption_all` |
|---|---:|---:|---:|
| wc R@20 | **46.9** | **46.9** | 45.3 |
| wc mR@20 | 26.0 | 25.1 | **26.1** |
| nc R@50 | **61.1** | 60.9 | 59.9 |
| nc mR@50 | **43.8** | 43.1 | 43.6 |
| OO nc mR@50 | **40.8** | 40.3 | 40.6 |
| OU-nt nc mR@50 | **44.6** | 42.8 | 44.1 |
| legacy uF1 | **49.9** | not scored | 48.2 |

`rag_all` takes five of the seven columns. It is the best method in the
unlocalized track — by margins between 0.1 and 1.8 points.

### 5b. PredCls, thinking cells, 150-video subset, Qwen3-VL-8B

([ICLR_THREE_TRACKS](../analysis/ICLR_THREE_TRACKS.md) §2b)

| decode | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 47.4 | 28.8 | 60.0 | 48.5 | 46.1 | 51.8 |
| **thinking** | **50.4** | **30.3** | **63.8** | **49.9** | **47.7** | **53.8** |

### 5c. SGDet

| model | split | videos | unloc nc R@50 | unloc nc mR@50 | slots w/ 3D |
|---|---|---:|---:|---:|---:|
| qwen3vl_8b, standard | think150 | 145/150 | 22.7 | 18.5 | 0.0 |
| qwen3vl_8b_thinking | think150 | 150/150 | 22.3 | **20.6** | 0.0 |
| qwen25vl_7b | full | 578/1511 | **not scored — generating** | | |

### 5d. Legacy protocol

PredCls, per predicate group ([ICLR_THREE_TRACKS](../analysis/ICLR_THREE_TRACKS.md) §2d):

| Model | videos | Attention F1 | Contacting F1 | Spatial F1 | μF1 | MF1 |
|---|---:|---:|---:|---:|---:|---:|
| InternVL2.5-8B | 1734 | 41.8 | 34.7 | 23.4 | 32.9 | 18.8 |
| KeyeVL-1.5-8B | 1734 | 38.3 | 34.6 | 49.2 | 41.0 | 24.4 |
| MiniCPM-V4.5 | 1734 | 42.0 | 41.0 | 32.6 | 38.6 | 24.5 |
| Ovis2.5-9B | 1734 | 43.9 | 46.3 | **56.0** | 48.9 | 27.9 |
| Qwen2.5-VL-7B | 1734 | 52.8 | 47.8 | 49.0 | 49.8 | 23.3 |
| Qwen3-VL-30B-A3B | 1734 | 55.7 | 45.3 | 35.1 | 44.5 | **28.6** |
| Qwen2.5-VL-7B | 1511 | 53.6 | 49.7 | 46.7 | **49.9** | 24.2 |
| Qwen3-VL-8B | 150 | **63.0** | 46.0 | 45.9 | 51.4 | 28.4 |

SGDet, same protocol: Qwen2.5-VL-7B at 1,734 videos reads 35.5 / **50.3** / 44.6
per group with μF1 **43.7** and MF1 19.9 — the best SGDet μF1 of any backbone;
Qwen3-VL-8B at 145 videos reads 32.2 / 25.7 / 24.2, μF1 27.1, MF1 17.0.
InternVL2.5-8B, KeyeVL-1.5-8B, MiniCPM-V4.5 and Ovis2.5-9B read μF1 24.2, 37.2,
29.6 and 34.5.

### What the results establish

1. **Retrieval is the best unlocalized method and the margin is negligible.**
   `rag_all` ties `zero_shot` on wc R@20 at 46.9 and leads it by 0.2–1.8 on the
   remaining columns. **Graph-RAG buys essentially nothing over a plain
   per-object prompt** on the full split. State this as a negative result; it
   is the honest reading and it is what motivates the localized track.
2. **Retrieval's one consistent advantage is the unobserved bucket.** OU-nt nc
   mR@50 44.6, the best in the track, +1.8 over `zero_shot`. The object-permanence
   hypothesis is directionally right and quantitatively small.
3. **Thinking is worth +3.0 wc R@20 on top of retrieval** (47.4 → 50.4) and
   this is the **only valid thinking cell in the project**. RAG survives the
   `max_new_tokens` squeeze because it prompts per object; every array-style
   prompt (Track A, Track B) does not. This is a prompt-format result, not a
   model result, and it must be reported that way.
4. **Retrieval wins across backbones but never by much.** RAG takes PredCls μF1
   on five of six backbones and SGDet μF1 on four of five, typically by 0.5 to
   3.6 points. The one large gap (KeyeVL, +15.5 μF1) is a captioning collapse,
   not a retrieval success — see [MLLM_CAPTION_ALL.md](MLLM_CAPTION_ALL.md) §4.3.
5. **Spatial predicates dominate the backbone spread.** Under RAG in PredCls,
   spatial F1 spans 23.4 to 56.0 while attention spans 38.3 to 55.7. The
   unlocalized setting is largely measuring spatial competence — which is what
   the localized track attacks and where it collapses at IoU 0.15.
6. **Scale is not the story.** Qwen3-VL-30B-A3B, the largest backbone, lands
   mid-table on μF1 (44.5) because its spatial F1 (35.1) is among the worst,
   even though it is second-best on attention.

---

## 6. Running

Stage-1 first:

```sh
CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.graphs.runner \
    --model_name qwen25vl_7b --video_list /data3/rohith/ag/splits/test_worldbbox_1511.txt
```

Then the relationship pass:

```sh
CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.rag_all.runner \
    --model_name qwen25vl_7b --mode predcls --skip-verification \
    --video_list /data3/rohith/ag/splits/test_worldbbox_1511.txt
```

Scoring:

```sh
python -m lib.mllm.eval.score_run --method rag_all --model qwen25vl_7b --mode predcls
```

Graphs are registered in the cache manifest by
[lib/mllm/tools/register_graphs.py](../lib/mllm/tools/register_graphs.py).

---

## 7. Known defects and caveats

1. **`scenes` is double-counted in the node-content embedding.**
   `Vgent.precompute_graph_embeddings` concatenates `data.get("scenes", [])`
   twice when building the text a node is embedded from, double-weighting scene
   terms in retrieval ranking. Harmless to correctness, but it means the
   retrieval ranking reported here is not the ranking a clean implementation
   would produce.
2. **Only the top-1 node reaches the prompt**, and only its `entities`,
   `actions` and `scenes` fields. The retrieval machinery ranks a node list and
   then throws the tail away. If retrieval is to be defended as a contribution,
   top-k is the obvious untested knob.
3. **Stage-1 is a hard gate.** A missing graph pickle skips the video. Verify
   Stage-1 PKL coverage against the split before launching.
4. **Silent per-video failures produce false "done" runs** — the shared
   `ActionGenomeBaseProcessor.run()` defect. Measure completion by PKL count,
   never by exit code.
5. **The graph is generated by the model being evaluated**, so a weak backbone
   retrieves over its own weak graph. Intended (it keeps the method
   training-free and self-contained) but it conflates graph-building ability
   with relational ability.
6. **`retrieve_nodes_with_cache` carries dead branches from its Vgent
   benchmark origin** — special cases keyed on "caption", "beginning", "at the
   end of the video", and an `mlvu` task check. The WSGG relationship prompts
   never match them, so they are inert here, but they are visible in the code
   and will invite questions.
