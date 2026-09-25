# SGR3-style retrieval inside WorldRAG (`rag_sgr3`)

Phase 4 of [docs/EXTERNAL_BASELINES_PLAN.md](../docs/EXTERNAL_BASELINES_PLAN.md). The question: is
our retrieval (BGE text embedding of the video's own Stage-1 graph, top-1 node) the right design,
compared with SGR3's visual multi-vector retrieval of *reference* scene graphs from other scenes?

SGR3 (arXiv 2603.04614) has no public code. We reimplement **only its retrieval strategy** and call
it **"SGR3-style"** everywhere. Its published numbers (Qwen3-VL-32B, 3RScan) go in related work, not
in our tables.

Index/query: [lib/mllm/tools/sgr3_index.py](../lib/mllm/tools/sgr3_index.py) ·
Runner: [lib/mllm/methods/rag_sgr3/runner.py](../lib/mllm/methods/rag_sgr3/runner.py) ·
Queue: [scripts/remote/sgr3_arms.sh](../scripts/remote/sgr3_arms.sh) ·
Context stats: [lib/mllm/tools/sgr3_ctx_stats.py](../lib/mllm/tools/sgr3_ctx_stats.py) ·
Run check: [lib/mllm/tools/sgr3_check_run.py](../lib/mllm/tools/sgr3_check_run.py) ·
Base pipeline: [MLLM_RAG_ALL.md](MLLM_RAG_ALL.md)

---

## 1. Arms

Every arm is the `rag_all` predcls pipeline with **only the retrieved context changed**:
Qwen3-VL-8B-Instruct, the 150-video list `test_worldbbox_thinking150.txt`, the same per-object
prompt, the same visual input (target frame + annotated context frames), the same object inventory
(GT, predcls), `max_new_tokens` 128, temperature 0.2, `--skip-verification`.

| Arm | `--arm` | Retrieved context |
|---|---|---|
| R0 | `none` | none (zero-shot inside the `rag_all` prompt) |
| R1 | — | existing `rag_all/predcls/qwen3vl_8b_150` run (2026-09-20), uncapped |
| R1@B | `bge` | R1's retrieval (Steps 1–2 of `rag_all`) truncated to the budget |
| R2-k | `sgr3 --k k` | SGR3-style visual retrieval of top-k **train** reference scenes, k ∈ {1, 3, 5} |
| R3 | `hybrid --k 1` | R1 context truncated to B/2, then R2-1 fills the rest of B |

R2 = R2-1 (SGR3 uses only the top-1 scene). R3's k was fixed to 1 *a priori*, not chosen on
test results.

**R0 is new.** No Qwen3-VL-8B zero-shot run existed on the 150 set (only `zero_shot/qwen25vl_7b`, a
different model and prompt), so R0 is the `rag_all` prompt with an empty context: the cleanest
"retrieval off" control.

## 2. The context budget

- **Measured on R1.** `--arm bge --context_only` rebuilt R1's contexts for all 150 videos
  (25,755 prompts; keyword extraction is sampled at T=0.2, so this is a re-draw of the same
  distribution, not a byte copy of the 2026-09-20 run). Qwen3-VL tokenizer, header included:
  mean 90.3, median 82, p5 40, p95 164, max 207 tokens; no empty contexts.
- **B = 90 tokens** (≈ R1's mean). Every new arm is capped at B; R2 is filled up to B edge by
  edge (never over), so its per-prompt length is ≈ B. R1 as run is uncapped with mean 90, so its
  *average* budget matches; R1@B is the strictly capped version.
- Each run writes every prompt's context and token count to
  `/data3/rohith/ag/runs/mllm/rag_sgr3/ctx/<model+tag>/<video>.json`.

## 3. The bank (keys and values)

- **Values = WorldBBox train scene graphs.** `world4d_rel_annotations/train` (7,516 videos,
  175,751 annotated frames with at least one object and an image). A frame's graph is its
  person→object edges in the AG vocabulary (attention ∪ contacting ∪ spatial), with the object's
  `visible` flag. These train labels include the pipeline's RAG-augmented unobserved objects
  (`source` ∈ {`gt`, `rag`, …}), i.e. the same labels the supervised models train on.
- **Leakage check** (`/data3/rohith/ag/cache/sgr3/leakage.json`): bank ∩ `test_worldbbox_1511` = ∅
  and bank ∩ AG `video_splits.json` test (1,750) = ∅. It is asserted when the bank is built, when it
  is queried, and when the runner loads it. The 150 query videos are a subset of the 1,511.
- **Keys = SigLIP2 patch embeddings.** `google/siglip2-base-patch16-224`: 14×14 = 196 patch tokens
  of 768 dimensions (SGR3's "768-dimensional" SigLIP2), taken from the vision tower's last hidden
  state and L2-normalised; frames are resized to 224×224.
- **Key-frame filter** (SGR3 drops a frame whose token-wise similarity to the kept buffer exceeds
  a threshold). We use the same late-interaction similarity (mean over patches of the max cosine)
  on SigLIP2 patches, in stream order per video. SGR3's 0.5 is for ColQwen embeddings and does not
  transfer: on SigLIP2, adjacent annotated frames of one video score 0.77–0.97 (5th–95th
  percentile) and first frames of different videos 0.50–0.65. **Threshold 0.8**, at most 8 key
  frames per video: 28,841 key frames (mean 3.8/video) → 5,652,836 patch vectors.
- **FAISS** `IndexIVFScalarQuantizer` (IVF4096, SQ8, inner product), trained on 400k patch vectors.
  Search: nprobe 32, k = 64 nearest neighbours per query patch. Index 4.1 GB,
  key-frame patches 8.1 GB (fp16), both under `/data3/rohith/ag/cache/sgr3/`.

## 4. Scoring a query (SGR3 §III, as described in the paper)

For a query frame q with patches p_1..p_P:

1. Self-similarity weights: μ_i = mean_{t≠i} cos(p_i, p_t), w_i = exp(−μ_i/τ) / Σ_t exp(−μ_t/τ),
   τ = 0.1. This down-weights uninformative patches.
2. Each patch retrieves its k = 64 nearest bank patches. For a bank frame f, a_f[i] is the best
   similarity among patch i's neighbours that lie in f, and Ω_f is the set of query patches with at
   least one neighbour in f. Then Score(q, f) = Σ_{i∈Ω_f} w_i · a_f[i].
3. Window W = the target annotated frame ± 1 annotated frame. Scene score
   S(s) = Σ_{q∈W} max_{f∈F(s)} Score(q, f). Scenes (train videos) are ranked by S. Within a
   scene, frames are ranked by Σ_{q∈W} Score(q, f).
4. The retrieved reference graph of a scene is the de-duplicated union of the edges of its top-3
   frames (SGR3 merges the "top-ranked frames" of the top scene and removes duplicate edges).

Offline output: `/data3/rohith/ag/cache/sgr3/retrieval/<video>.json`, holding per test frame the
window, the top-10 scenes with scores, and each scene's top-5 frames.

## 5. Deviations from SGR3 (all deliberate; the paper leaves the rest unspecified)

1. **Artifact-token mask (necessary).** SigLIP2 emits about 3 artifact ("sink") tokens per image at
   near-fixed positions (13, 55, 1, 183, …). They are near-identical across unrelated images
   (cross-image max cosine ≈ 0.999), and they are the patches *least* similar to the rest of their
   own frame. So SGR3's weighting gives them about 95% of the total weight, and the literal
   retrieval collapses to chance. Evidence (`sgr3_index selfcheck`): a train key frame used as
   the query retrieves its own video top-1 **2.1%** of the time without the mask and **100%** with
   it (non-key frames: 3.8% → 94.7%). We take 4 sink prototypes from bank key frames (patches whose
   best match in 100 random other frames exceeds 0.98) and drop query patches with cosine > 0.9 to
   a prototype before weighting and search. The unmasked ("literal") retrieval is kept at
   `retrieval_literal/` and `diag_literal.json` for the record.
2. **Encoder and resolution.** SGR3 names SigLIP2 and 768-d but not the variant, so we use base/16
   at 224 px. Key frames are selected with SigLIP2 token-wise similarity, not ColQwen (§3).
3. **Unspecified hyper-parameters, chosen a priori and not tuned on test:** k = 64 neighbours per
   patch, window ±1 annotated frame, 3 merged frames per scene, IVF-SQ8 with nprobe 32.
4. **Serialization under a token budget.** SGR3 gives no prompt template. We write
   `Reference n: person -> <object>[ (unseen)]: rel, rel; …` under a one-line header, one line per
   scene. Because every arm is capped at B = 90 tokens, the order in which edges enter matters:
   (a) edges of the queried object's class come first within each scene (the `rag_all` prompt is
   per object, and R1's retrieval is conditioned on the object prompt too);
   (b) for k > 1 the budget is filled round-robin across scenes;
   (c) if a scene's merged top-3-frame graph is shorter than its share, it is **padded from the
   same reference video**: first its remaining retrieved frames, then its other annotated frames,
   nearest in time to the top frame first. This is how "pad to the budget" is implemented; the
   context never mixes in unretrieved videos.
5. **The MLLM, task and data are ours**: Qwen3-VL-8B (not 32B), AG/WorldBBox videos (not 3RScan),
   person–object predicates in the AG vocabulary, and per-object prompts at every annotated frame.

## 6. Retrieval sanity check against test GT (diagnostic only, never used by an arm)

`sgr3_index diag`: for each (test frame, GT object), is that object's class present in the merged
reference graph of the top-k scenes? The baseline is k random bank scenes (their first 3 frames).
25,673 GT objects.

| | k=1 | k=3 | k=5 |
|---|---:|---:|---:|
| retrieved (masked) | 19.2% | 40.9% | 55.2% |
| retrieved (literal, unmasked) | 14.7% | 36.7% | 51.4% |
| random scenes | 11.4% | 30.2% | 43.8% |
| retrieved (masked), unseen GT objects only | 15.5% | 35.5% | 50.4% |

Visual retrieval across *different homes* is weak at the object level. The top-1 reference scene
contains the queried object's class for about 1 in 5 objects, 1.7× chance. (SGR3 retrieves across
3RScan rescans of the *same* rooms, where this is much easier.)

## 7. Reproduce

```sh
# env: --system-site-packages venv over wsg + faiss-cpu (nothing installed into wsg)
~/anaconda3/envs/wsg/bin/python -m venv --system-site-packages /data3/rohith/ag/envs/sgr3
/data3/rohith/ag/envs/sgr3/bin/pip install --no-deps faiss-cpu
cd ~/CODE/Scene4Cast_sgr3; export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data3/rohith/hf OMP_NUM_THREADS=8
PY=/data3/rohith/ag/envs/sgr3/bin/python
$PY -m lib.mllm.tools.sgr3_index bank                     # + leakage.json
$PY -m lib.mllm.tools.sgr3_index embed --thr 0.8 --cap 8  # ~12 min
$PY -m lib.mllm.tools.sgr3_index index --threads 16       # ~15 min
$PY -m lib.mllm.tools.sgr3_index sinks
$PY -m lib.mllm.tools.sgr3_index selfcheck
$PY -m lib.mllm.tools.sgr3_index query --threads 16       # 150 videos, ~15 min
$PY -m lib.mllm.tools.sgr3_index diag
# budget: R1 context lengths
sh scripts/remote/sgr3_job.sh r1ctx --arm bge --context_only --tag _R1ctx
~/anaconda3/envs/wsg/bin/python -m lib.mllm.tools.sgr3_ctx_stats /data3/rohith/ag/runs/mllm/rag_sgr3/ctx/qwen3vl_8b_R1ctx
# arms + check + score (GPU 1, sequential)
setsid nohup sh scripts/remote/sgr3_arms.sh > /data3/rohith/ag/logs/ext/sgr3_arms.log 2>&1 < /dev/null &
```

Runs: `/data3/rohith/ag/runs/mllm/rag_sgr3/predcls/qwen3vl_8b_<ARM>/`. Scores:
`/data3/rohith/ag/runs/mllm/rag_sgr3/scores/<model dir>.json` (`score_run --method rag_all --pred_dir
<root> --video_list <150> --no_legacy`). Run checks: `scores/checks.jsonl`.
