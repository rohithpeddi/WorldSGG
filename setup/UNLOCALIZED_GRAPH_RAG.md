# Unlocalized Graph-RAG — Architecture (`rag_all`)

Track: [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) · method, results and runbook:
[MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) · index: [README.md](README.md)

Graph-RAG is the training-free MLLM architecture of the unlocalized track. The
repository README calls this pipeline **WorldRAG**, and its current figure is
[analysis/assets/UWorldSGGGraphRAG.png](../analysis/assets/UWorldSGGGraphRAG.png).
It never trains and never places an object. For every annotated frame it emits a
class-only scene graph of person → object attention, contacting and spatial
predicates. Two stages share one frozen VLM:

- **Stage 1** runs offline, once per video. It segments the video around its
  annotated key frames, captions each segment, and turns each segment into one
  event-graph node of entities, actions and scenes.
- **Stage 2** runs per experiment. It fixes the object set: ground truth in
  PredCls, VLM-discovered in SGDet. A frozen text encoder retrieves the event-graph
  node most relevant to each object. The VLM then answers one relationship
  question per (annotated frame, object). The retrieved node's text is prepended
  to the question, and the target frame comes first in the visual input.

This document describes the architecture as implemented, at the level needed for
a method section or a figure. Results, the running recipe and the full defect list
are in [MLLM_RAG_ALL.md](MLLM_RAG_ALL.md).

**Main configuration.** The full-split rows use Qwen2.5-VL-7B-Instruct
(`qwen25vl_7b`) for every VLM call in both stages. Retrieval uses
BGE-large-en-v1.5. Decoding is standard (temperature 0.2), with
`--skip-verification`. The matched 150-video block runs Stage 2 on Qwen3-VL-8B:
`qwen3vl_8b`, and `qwen3vl_8b_thinking` with
`--max_new_tokens 4096 --temperature 0.6 --top_p 0.95`. Both read the **same**
Qwen2.5-VL Stage-1 graphs, because `graphs/qwen3vl_8b{,_thinking}` are symlinks to
`graphs/qwen25vl_7b`. This keeps the retrieval corpus fixed across cells.

| Component | Code |
|---|---|
| Stage-1 driver | [lib/mllm/methods/graphs/runner.py](../lib/mllm/methods/graphs/runner.py): `get_clip_intervals`, `process_video_fast` (used for the test-split fill), `process_video` |
| Graph construction, keywords, retrieval | [lib/mllm/core/vgent.py](../lib/mllm/core/vgent.py): `construct_graph`, `_update_entity_graph`, `precompute_graph_embeddings`, `batch_extract_keywords`, `retrieve_nodes_with_cache` |
| Node selection and ranking | [lib/mllm/core/retrieval.py](../lib/mllm/core/retrieval.py): `allocate_node_batched`, `count_and_sort_filtered` |
| Stage-2 driver | [lib/mllm/methods/rag_all/runner.py](../lib/mllm/methods/rag_all/runner.py): `load_precomputed_graphs`, `_batch_all_video_queries`, `process_video` |
| Object discovery, tensors, parsing, verification | [lib/mllm/base_processor.py](../lib/mllm/base_processor.py) |
| Prompts | [lib/mllm/core/prompts.py](../lib/mllm/core/prompts.py) (P0–P3), `rag_all/runner.py` (P4), `base_processor.py` (P5, P6) |
| Backbone wrappers | [lib/mllm/models/qwenvl.py](../lib/mllm/models/qwenvl.py), [lib/mllm/models/qwen3vl_new.py](../lib/mllm/models/qwen3vl_new.py) |
| Paths and vLLM engine settings | [configs/mllm/server.yaml](../configs/mllm/server.yaml) |

---

## 1. Architecture at a glance

The panel letters follow the current figure. The prompt ids P0–P6 are defined in §7.

| Module (figure panel) | Stage | What it does | VLM calls | Runs in |
|---|---|---|---|---|
| (a) Coarse event-graph construction | 1 | key-frame segment → caption → one `{entities, actions, scenes}` node | 2 per segment (P1, then P0) | both modes, once per video |
| (b) Object discovery | 2 | proposes the video's objects from captions and frames, restricted to the AG vocabulary, then scores each | 1 (P6), plus 1 Yes/No per candidate (P5) | SGDet only |
| (c) Graph RAG | 2 | per object: keywords, embedding retrieval over the event graph, node relevance check, context block | P2 once per object; none for retrieval; P3 once per (object, retrieved node) | both modes |
| (d) Relationship prediction | 2 | per (frame, object): context + question → three-head JSON | P4 once per (frame, object) | both modes |
| (e) Relationship verification | 2 | one Yes/No per predicted label → calibrated score | P5 per label | **off in every reported run** (`--skip-verification`) |

The same frozen VLM serves every call. The only other network is the frozen
BGE-large text encoder, which module (c) uses instead of LLM calls to select nodes.

---

## 2. Forward pipeline

```
STAGE 1 — offline, once per video                        methods/graphs/runner.py
  annotated key frames k_1 < … < k_F                      (frames_annotated/<video>/)
    │ segment i is bounded by the midpoints to k_{i-1} and k_{i+1};
    │ the first and last segments are clipped to 30 frames around their key frame
    ▼
  segment i: every 2nd raw frame ─► resize_video ─► S_i   (≤19 frames, ≤13.5k px each)
    ├─ P1  caption prompt ───────────────────────────► caption_i
    └─ P0  "Context: <caption_i>" + graph prompt ─────► entities_i, actions_i, scenes_i
  node i = {entities_i, actions_i, scenes_i, captions: [caption_i], key frame k_i}
  → graphs/qwen25vl_7b/<video>.mp4.pkl                  (one record per segment)

STAGE 2 — per video, per run                             methods/rag_all/runner.py
  0  merge the F nodes into one graph; entity index  name → {nodes}      [hard gate]
  1  BGE-large: embed entity names and node texts (entities; actions; scenes; caption)
  2  tensors   V   = whole video, ≤19 frames                 (discovery, fallback)
               S_i = Stage-1 segment of key frame k_i        (node check, verification)
               Q_f = [target frame f ; ≤15 key frames]       (answering, 16 frames)
  3  objects   PredCls: GT objects of the video
               SGDet:   P6(V, all captions, 36-class list) ∩ vocabulary, P5-scored
  4  queries   F frames × |O| objects; the P4 text depends only on the object
  ┌─ once per unique object o ──────────────────────────────────────────────────┐
  5  P2   keywords_o                                          (text only, no image)
  6  retrieve: queries = keywords_o ∪ {P4(o)}; keep nodes whose entity name or
     text has mean cosine > 0.5; rank by mean cosine to node text; keep top 20
  7  P3   per retrieved node n: (S_n, "is <o> visible or interacted with?",
          "is a person visible?") → count yes → rank             [unused by step 8]
     context_o = "Relevant scene context from video analysis:\n"
                 + entities; actions; scenes of the TOP node from step 6
  └──────────────────────────────────────────────────────────────────────────────┘
  8  per (f, o):  context_o + P4(o), visual Q_f  → {attention, contacting, spatial}
                                                                (≤128 new tokens)
  9  parse, validate against the 3 / 17 / 6 label sets
 10  P5 verification per predicted label           [skipped: every label scores 1.0]
 11  emit per frame: all |O| objects with predicates — no boxes
```

---

## 3. Stage 1 — coarse event-graph construction (module a)

**Key frames are the annotated frames.** No model selects key frames.
`get_clip_intervals` takes the frames listed in `frames_annotated/<video>/` and cuts
the raw frame range at the midpoints between consecutive key frames. Segment *i* is
therefore centred on key frame *k_i*. The first segment starts at most 30 frames
before *k_1*, and the last ends at most 30 frames after *k_F*.

**Segment tensors are tiny.** The loader keeps every second raw frame of the segment.
`resize_video` (total_pixels = 128,000) then caps the segment at **19 frames**, each
84×140 px for a 480×270 source. So the 30-frame subsampling in `construct_graph`
never triggers, and each segment becomes exactly one node.

**Two calls per segment, caption first.**

- **P1** (`CAPTION_GENERATION_PROMPT`, ≤100 new tokens) asks for one concise caption
  of the main action.
- **P0** (`GRAPH_PROMPT`, ≤512 new tokens) is sent with that caption as a prefix:
  `Context: The following caption describes this video clip: "<caption>"`. It
  returns strict JSON of the form `{"entities": [{entity name, description}],
  "actions": [{entity name, action description}], "scenes": [{location}]}`. Action
  entity names must come from the entity list.

The fast path, which generated the test-split fill, makes two batched calls per
video: all captions in one, all graph prompts in the next. It retries an
unparseable graph response up to four times.

**Node record.** Each segment is saved as a record:
`{"clip_metadata": {annotated_frame, start_frame, end_frame}, "caption": str,
"graph": nx.DiGraph}`. Its single node carries four lists:

- `entities` as `"name, description"` strings;
- `actions` as `"name, action description"` strings;
- `scenes` as location strings;
- `captions` = `[caption]`.

**Edges.** `_update_entity_graph` embeds every item with BGE and links it to an
existing entity key when their cosine similarity exceeds 0.7. Each segment is a
single node, so those links are self-loops, and Stage 2 never reads the graph's
edges. The links between events that a diagram draws exist only as Stage 2's
**entity index** (§4.1): two nodes are linked when they mention the same entity
name.

**Provenance.** Every Stage-1 graph comes from Qwen2.5-VL-7B.
[configs/mllm/server.yaml](../configs/mllm/server.yaml) merges 2,667 pre-existing
graphs with the missing test videos, which the `b3_graphs_fill` job generated with
`--fast`. Stage 2 skips any video whose pickle is missing or empty.

---

## 4. Stage-2 inputs

### 4.1 Event graph and entity index

`load_precomputed_graphs` concatenates the per-segment graphs. It offsets node ids by
the running node count and tags each node with `source_annotated_frame`. It then
builds the **entity index**. For every entity, action and scene string of a node,
the lower-cased text before the first comma becomes a key that maps to that node.
The function also returns the caption list `[(k_i, caption_i)]` and the segment
intervals. The intervals are reloaded as tensors S_i with the Stage-1 sampling rule.

`precompute_graph_embeddings` runs once per video. It embeds every entity key, and
every node text formed as `entities; actions; scenes; scenes; captions`. The scenes
appear twice, a weighting defect recorded in [MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) §7.1.
Embeddings are BGE-large CLS vectors, L2-normalised, so cosine similarity is a dot
product.

### 4.2 Visual tensors

| Tensor | Frames | Per-frame size (480×270 source) | Built by | Consumed by |
|---|---|---|---|---|
| V, whole video | all raw frames (stride 8 above 120), subsampled uniformly to ≤19 | 84×140 | `load_video_clip` | P6 and P5 object checks; fallback visual |
| S_i, key-frame segment | the Stage-1 rule: every 2nd frame of segment i, ≤19 | 84×140 | `load_video_clip` | P3 node checks; P5 relation verification, with the target frame prepended |
| Q_f, query tensor | target frame f, then ≤15 key frames (linspace over all key frames) | 168×336 | `_build_annotated_context` (total_pixels 512,000) + `_prepend_target_frame` | P4 answering |

Each tensor reaches the backbone as a single **video**, one per prompt
(`limit_mm_per_prompt = {"video": 1}`). Three properties of Q_f matter when the
method is described:

- The target frame is resized to the context resolution before it is prepended, so
  the model sees the target frame at 168×336, not at native resolution.
- Qwen-VL video encoders patch frames in temporal pairs. The target frame therefore
  shares its first token group with the first context frame, so the "first frame is
  the target" convention is only partly realised at the token level.
- The 15 context frames are the same for every target frame of a video.

### 4.3 Object set — module (b), object discovery

**PredCls** uses `get_video_objects`: every non-person label annotated anywhere in
the video, in short form (e.g. `cup`). Each object is queried at every frame,
including frames where it is not annotated. The scorer reads only the frame's
ground-truth slots.

**SGDet** uses `get_objects_for_mode`, in three steps:

1. **P6** (`AG_OBJECT_ESTIMATION_PROMPT`) receives the whole-video tensor V and every
   Stage-1 caption as `[Frame k_i] caption_i` lines. Its candidate list is the
   36-class AG list, with the instruction to exclude "person"; it may also add
   objects that are not on the list. It returns a JSON list (≤256 new tokens).
2. The answer is lower-cased and intersected with the vocabulary by exact string
   match. The ground-truth list is never consulted.
3. **P5** checks each surviving object with
   `Is there a <object> in this scene? Answer only Yes or No.`, captions prefixed and
   V attached. It decodes one token at temperature 0 and computes
   `P(Yes) = p_yes / (p_yes + p_no)` from the top-20 logprobs.

The P(Yes) values are stored in `estimation_meta.object_scores`. They do not filter
objects. Instead, they become the object confidence that SGDet scoring multiplies
into every triplet score. This object check runs even with `--skip-verification`.
Object discovery involves no embedding matching, contrary to the README's
description of module (b).

---

## 5. Graph RAG — module (c)

### 5.1 Deduplication

The P4 text depends only on the object name, so the F × |O| queries reduce to |O|
unique prompts. Keyword extraction (P2), retrieval and the node check (P3) run once
per unique object. Only answering (§6) runs per (frame, object).

As a result, **retrieval is per object, not per frame**. Object *o* receives the same
context block at every frame of the video. Temporal specificity enters only through
the target frame in Q_f.

### 5.2 Keyword extraction (P2)

`batch_extract_keywords` sends `REASONING_PROMPT`, inherited from the Vgent long-video
QA pipeline, as a text-only call with ≤256 new tokens. The whole P4 text is the
"question" and the candidate list is empty. The response is a JSON object with
`keywords`, `multiple`, `time`, `tool`, `candidates_necessary` and `global`; only
`keywords` affects WSGG retrieval.

### 5.3 Embedding retrieval (no VLM)

`retrieve_nodes_with_cache` calls `allocate_node_batched`, which works in four steps:

1. The query set is Q = keywords_o ∪ {the full P4 text}.
2. **Entity route.** Each entity key gets a score: its mean cosine similarity to the
   queries in Q. Every node indexed under a key scoring above 0.5 is selected.
3. **Content route.** Any other node whose text has a mean cosine above 0.5 is added.
4. The selected nodes are ranked by mean cosine between Q and the node text, and the
   top `n_retrieval = 20` are kept.

If nothing clears the threshold, the object gets no context and no node checks.
Vgent also has special-case branches for quoted captions, "beginning", "at the end
of the video" and "order". None of them fires on the WSGG prompt; this was checked
against all 35 object names.

### 5.4 Node relevance check (P3), and what it feeds

P3 runs for each unique object and each retrieved node *n*. It uses
`SQL_ANSWER_PROMPT` with two templated questions and the node's caption, over the
node's own segment tensor S_n, with ≤256 new tokens:

- Q1: `Is the '<o>' visible or interacted with in this video segment?`
- Q2: `Is there a person visible in this video segment?`

`count_and_sort_filtered` counts the answers that are not "no" and ranks the nodes
that have at least one.

The winner of this ranking has exactly one use. It selects the verification clip
for a frame that has no Stage-1 segment of its own. **The context block comes from
the embedding ranking of §5.3, not from this one.** Every reported run skips
verification, and every annotated frame has its own segment. So **P3 never changes a
reported prediction**, even though it is the largest call family after answering: up
to 20 calls per object.

### 5.5 The context block

The block reads
`Relevant scene context from video analysis:\n<entities>; <actions>; <scenes>\n\n`,
joining the three lists of the top-1 node with "; ". The node's caption is not
included. In Stage 2, captions enter only through the retrieval embeddings, P3, P6,
and the verification prompts, which are skipped.

---

## 6. Relationship prediction and verification — modules (d) and (e)

**P4** (`AG_RELATIONSHIP_QUERY_PROMPT`) tells the model that the first frame is the
moment to analyse and the remaining frames are context. It adds that the object "may
or may not be visible in the target frame", and that "A person IS visible". It then
asks three questions:

- attention: exactly one of 3 labels;
- contacting: one or more of 17;
- spatial: one or more of 6, the object's position relative to the person.

The answer format is `{"attention": "<label>", "contacting": [...], "spatial": [...]}`.
The text is context_o + P4(o), the visual input is Q_f, and the budget is ≤128 new
tokens (4,096 for thinking). `zero_shot` uses the same prompt and visual input, so
in PredCls the retrieved block is the only difference between the two methods.

`parse_relationship_response` validates every label against its vocabulary, with a
greedy `{…}` search as fallback. An invalid head becomes "unknown". With
verification skipped, `_default_scored_from_parsed` gives each predicted label
`yes_prob = 1.0`, and every other predicate scores 0.

**Verification** is off in all reported runs. It would ask one Yes/No question per
predicted label, such as `At the target frame (the first frame of this clip), is the
person <label> the <object>? Answer only Yes or No.` Each question is prefixed with
all captions and context_o and uses the visual input [target frame ; S_f]. The
resulting P(Yes) would replace the 1.0.

---

## 7. Prompt ledger

| Id | Constant | Module | Unit | Visual input | Output | New tokens |
|---|---|---|---|---|---|---:|
| P0 | `GRAPH_PROMPT`, caption-prefixed | (a) | segment | S_i | JSON entities / actions / scenes | 512 |
| P1 | `CAPTION_GENERATION_PROMPT` | (a) | segment | S_i | one caption | 100 |
| P2 | `REASONING_PROMPT` | (c) | object | none | JSON with `keywords` | 256 |
| P3 | `SQL_ANSWER_PROMPT` + Q1/Q2 + node caption | (c) | object × retrieved node | S_n | JSON yes / no | 256 |
| P4 | `AG_RELATIONSHIP_QUERY_PROMPT` + context block | (d) | frame × object | Q_f | JSON, three heads | 128 |
| P5 | Yes/No templates (object; relation) | (b), (e) | object; predicted label | V; [f ; S_f] | 1 token + logprobs | 1 |
| P6 | `AG_OBJECT_ESTIMATION_PROMPT` | (b) | video | V | JSON object list | 256 |

The current figure labels P0, P1, P3, P4 and P6. The two prompt families it leaves
unnamed are keyword extraction (P2) and the Yes/No checks (P5).

In thinking runs, `--max_new_tokens 4096` overrides the budgets of P3, P4 and P6.
P2 keeps its hard-coded 256 tokens, and Stage 1 is not re-run.

---

## 8. Calls per video

Here F is the number of annotated frames (32.3 on average: 48,834 frames over 1,511
videos) and |O| the number of objects.

| Call family | Count | Note |
|---|---|---|
| Stage 1: P1 + P0 | 2F, plus retries | once per video, shared by `caption_all`, `rag_all` and Track B |
| P6 + P5 object checks | 1 + number of candidates | SGDet only; P5 decodes one token |
| P2 keywords | \|O\| | text only |
| Retrieval | 0 | BGE forward passes only |
| P3 node checks | ≤ 20 \|O\| | no effect on reported outputs (§5.4) |
| P4 answers | F · \|O\| | the only per-frame call |
| P5 relation verification | one per predicted label | skipped |

Prompts go through vLLM in chunks of 64 (`BATCH_CHUNK_SIZE`, equal to
`max_num_seqs`). Each chunk decodes with the largest budget among its prompts.
Measured cost is about 30 s per video with standard decoding and 715 s per video
with the thinking backbone ([TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) §5).

---

## 9. Output and scoring interface

Each video produces one pickle, `<outputs.rag_all>/<mode>/<model><tag>/<video>.mp4.pkl`:

```
{video_id, mode, model_name, estimation_meta, video_objects, num_frames_processed,
 frames: {<frame>.png: {objects, predictions: [{object, raw_response,
          attention: {label, yes_prob}, contacting: [{label, yes_prob}], spatial: [...]}]}}}
```

[lib/mllm/eval/dump_adapter.py](../lib/mllm/eval/dump_adapter.py) maps each
prediction onto the frame's ground-truth slots. In SGDet, discovered objects without
a ground-truth slot are added as extra slots. `has_pred_3d` is False everywhere.
PredCls is scored by the stock WorldSGG evaluator. SGDet can be scored only under
class matching (`loc3d` at τ = 0, "unloc"), where a triplet's score is the P5 object
score times the predicate score.

---

## 10. Notes for the paper figure

The current figure and README §8 describe the intended design. The code differs from
them in the places below. A redrawn figure should follow the code.

1. **Key frames are the annotated frames.** The "key frame selection" of panel (a) is
   the annotation schedule. Segments are windows bounded by midpoints, one node each.
2. **P1 feeds P0.** The caption is a prefix of the graph prompt, so draw an arrow from
   caption to P0.
3. **Links between events are shared entity names.** The edges between nodes
   correspond to the Stage-2 entity index. The stored graph edges are self-loops
   inside a node and are never used.
4. **Object discovery has no embedding matching.** It is P6 over the subsampled
   video, all captions and the 36-class list, followed by an exact vocabulary
   intersection and a P5 Yes/No score per object. PredCls bypasses the module.
5. **Keywords come from text alone.** P2 reads the relationship question, not the
   query image.
6. **Retrieval is a BGE embedding match** with a 0.5 mean-cosine threshold and a
   top-20 cut. Label it as a text encoder, not as the VLM.
7. **The re-ranked nodes do not reach the answer.** In panel (c), the arrow from the
   P3 re-ranking to the VLM implies that the re-ranked node feeds prediction. The
   code prepends the top node of the embedding ranking instead. Either the arrow
   goes, or the code changes (§11.3).
8. **Answering takes one visual input:** [target frame ; ≤15 key frames] as one
   video. The subsampled full video feeds object discovery, not answering.
9. **The output is class-only:** F × |O| per-object answers per video.

---

## 11. Architecture-level caveats

1. **Retrieval ignores the frame.** The context for object *o* is identical at every
   frame (§5.1). Track B makes the opposite choice: its retrieval is a temporal window
   around the target frame ([LOCALIZED_MLLM.md](LOCALIZED_MLLM.md) §5.2).
2. **Only the top-1 node reaches the prompt, without its caption** (§5.5). Top-k is
   untested ([MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) §7.2).
3. **P3 is dead compute in every reported run** (§5.4). It costs up to 20 calls per
   object, against about 32 answering calls per object on an average video. Removing
   it, or feeding its ranking into the context block, is the cheapest change
   available. With the thinking backbone, each of those calls carries a 4,096-token
   budget.
4. **The thinking cell uses a different retrieval query from the standard cell.**
   P2 parses the raw response with `json.loads`, never strips a reasoning trace, and
   keeps its 256-token budget. Under `qwen3vl_8b_thinking` the keyword JSON cannot
   parse, so the keyword list is empty and the query set reduces to the P4 text alone.
   The +3.0 wc R@20 of RAG-thinking over RAG-standard therefore changes the retrieval
   query as well as the decoding. The outputs do not record this, because `llm_info`
   is not saved.
5. **SGDet discovery can silently drop five classes (unverified).** The vocabulary
   stores merged names (`cup/glass/bottle`, `closet/cabinet`, `paper/notebook`,
   `phone/camera`, `sofa/couch`) and the intersection is exact, yet P6's own example
   answers with `"cup"`. A short name would be dropped. Compare
   `estimation_meta.raw_estimated` with `vocab_filtered` in the SGDet pickles before
   relying on per-class SGDet numbers. Similarly, P5 decodes one token at temperature
   0, which is ill-defined for a thinking backbone; check the thinking cell's
   `object_scores`.
6. **Low visual resolution.** V and S_i are 84×140 per frame, and Q_f is 168×336 for
   a 16:9 source. The target frame is downscaled to the context resolution and
   shares a temporal token group with the first context frame (§4.2).
7. **The graph is not always built by the model under test.** Stage 1 is
   Qwen2.5-VL-7B for every cell. The Qwen3-VL cells therefore retrieve over another
   model's graph, deliberately, to fix the corpus. The statement in
   [MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) §7.5 that "the graph is generated by the model
   being evaluated" holds for the full-split rows only.
8. **Operational.** Stage 1 is a hard gate. Completion must be measured by pickle
   count, not exit code. The Qwen2.5-VL wrapper ignores `--temperature`; it
   hard-codes 0.2, which equals the default. See
   [MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) §7 for the rest.

---

## 12. Measured results

From [MLLM_RAG_ALL.md](MLLM_RAG_ALL.md) §5 and
[TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) §3. Values are percentages over all
annotated frames.

**PredCls**

| Backbone, decode | videos | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU-nt nc mR@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B, standard | 1,511 | 46.9 | 26.0 | 61.1 | 43.8 | 40.8 | 44.6 |
| Qwen3-VL-8B, standard | 150 | 47.4 | 28.8 | 60.0 | 48.5 | 46.1 | 51.8 |
| Qwen3-VL-8B, thinking | 150 | **50.4** | **30.3** | **63.8** | **49.9** | **47.7** | **53.8** |

**SGDet**, class-matched only: unloc nc R@50 / mR@50 is 22.7 / 18.5 with standard
decoding (145/150 videos) and 22.3 / 20.6 with thinking (150/150). The full-split
SGDet cell has not been scored. The backbone confound and the comparison with the
localized track are covered in [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) §5.
