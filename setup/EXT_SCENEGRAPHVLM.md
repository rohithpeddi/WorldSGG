# External baseline — SceneGraphVLM (hallucination-aware small VLM, previous-graph prompting)

Paper: *SceneGraphVLM: Dynamic Scene Graph Generation from Video with Vision-Language
Models* (Makarov, Gizetdinov, Yudin; arXiv 2605.13667). Code:
[markus0440/SceneGraphVLM](https://github.com/markus0440/SceneGraphVLM) (MIT), pinned at
`a6197da`. Released weights: Zenodo record 20511274 (`checkpoints.zip`, 5,283,563,855 bytes,
md5 `9e3c8f6ccce2bf632894053d74c74fc6`; contains `AG/`, `PSG/`, `PVSG/`, each a 2.2 GB
Qwen3.5-0.8B `model.safetensors`). We use `checkpoints/AG` (SFT + GRPO on Action Genome
train).

Runner: [lib/mllm/methods/scenegraphvlm/runner.py](../lib/mllm/methods/scenegraphvlm/runner.py) ·
Vendored driver: [lib/external/scenegraphvlm/](../lib/external/scenegraphvlm/) ·
Plan: `docs/EXTERNAL_BASELINES_PLAN.md` Phase 3 ·
Method key `scenegraphvlm`, model key `sgvlm_ag` ·
Results: [analysis/ext_scenegraphvlm_2026-09-25.md](../analysis/ext_scenegraphvlm_2026-09-25.md)

---

## 1. What the model does and how we feed it

One Qwen3.5-0.8B call per frame. The prompt holds the closed AG vocabulary and a TOON
schema; the model answers with its own object list (class + 2D box on a 640×480 canvas)
and one `rel_pairs` row per person–object pair (attention / spatial / contacting lists).
For frame *t* > 0 of a video the prompt also carries the model's **own graph for frame
*t*−1** (`--prev-source model`, the authors' "GEN" deployment mode). Frame 0 gets the
no-context prompt.

We change nothing about the model, prompt or decoding:

| Step | What we do | Authors' reference |
|---|---|---|
| frames | every annotated frame of the worldbbox annotation (`WorldBBoxVideo.frames`, 48,834 on the full split) — the same frames every MLLM row is scored on; chain order = annotated-frame order | AG test JSONL = consecutive annotated frames |
| image | original frame resized to 640×480, bilinear, PNG | `prepare_original_ag_sft.py` |
| prompt | `USER_PROMPT_FIRST` / `build_user_prompt_follow` imported from the vendored `sft_to_jsonl_ag.py` | same |
| inference | the vendored `infer_swift_gen_prompt.py`, run unmodified through `launch.py`: ms-swift `VllmEngine`, greedy (T=0), `max_new_tokens` 2048, `max_model_len` 8192, response prefix `<answer>\n`, stop `</answer>`, `IMAGE_MAX_TOKEN_NUM=1024` | their AG GEN command in `metrics/metrics.md` |
| boxes back | 640×480 → original pixels (per-axis scale) | — |

Compatibility shims (all in the launcher/runner, none in the driver):

* ms-swift 4.5 cannot auto-resolve the checkpoint's `model_type` / template (it matches
  `qwen3_5`, `ovis_ocr2`, … and `qwen3_5`/`qwen3_8`), so `launch.py` pins
  `model_type='qwen3_5'` and the runner passes the driver's own `--template-type qwen3_5`.
  The checkpoint config is `Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5`.
* `VLLM_USE_FLASHINFER_SAMPLER=0`: CS93371's system `nvcc` 12.4 cannot JIT-build FlashInfer
  0.6's sampling kernels. Decoding is greedy, so vLLM's PyTorch sampler emits the same tokens.
* Batch size 256–384 instead of 64 (the driver batches one frame per video per wave, so a
  bigger batch only changes throughput).

**Reproduction check.** The authors' own AG evaluator
(`metrics/sgbench/eval_classical_metrics_extended.py --mode sgdet --iou-thr 0.5`) on our
150-video set, against the observed-GT TOON we write into the JSONL, gives with-constraint
**R@50 26.0 / P@50 38.5**; their paper reports R@50 27.84 / P@50 38.74 on the AG test split.
The port is faithful.

**No predcls mode.** The released model has no input for the current frame's object
list — its only conditioning is the previous frame's graph — so feeding GT objects would be
an untrained prompt, not their method. Only the sgdet-style run exists.

---

## 2. Parsing and vocabulary mapping

`runner.py convert` parses each `predict` with the same line grammar as the authors' parser
(`obj[N]{id,name,x1,y1,x2,y2}` rows, `rel_pairs[M]{subj,attention,spatial,contacting,obj}`
rows), but clips out-of-canvas boxes instead of dropping the object. Then:

* object names → our 36 classes: exact AG name (`cup/glass/bottle` → short `cup`), else a
  logged synonym table (`cabinet`, `glass`, `couch`, `tv`, …), else **unmapped** (kept for UOR,
  never scored). The name `person` marks the subject.
* predicates → our 26 labels by exact name after lower-casing and `' '`/`-` → `_`; a label in
  the wrong head is dropped (counted).
* rows whose subject is not a person or whose object id is unknown are dropped (counted).
* one entry per class per frame (the WorldAG slot rule): duplicate instances merge their
  predicates; the first instance's box is kept. Predicates keep the model's order via scores
  `1 − 1e-4·j` (the authors' evaluator does the same).
* objects the model named but gave no relation row go to `objects_norel`: they count as
  predicted objects for UOR but never emit a pair.

Stats per tag: `/data3/rohith/ag/runs/mllm/scenegraphvlm/stats/sgvlm_ag__<tag>.json`
(parse rate, unmapped objects/predicates, objects per frame).

---

## 3. Scoring

The run is an sgdet-style run: `score_run --method scenegraphvlm --model sgvlm_ag --mode sgdet`.

* **Unlocalized (class-only)**: `loc3d.unloc` — the Track-2 column. Full GT, all 231,639 slots;
  the model has no memory, so unobserved slots it does not name are misses.
* **`sgdet2d` (new, optional `--sgdet2d`)**: observed-only 2D SGDet
  ([lib/mllm/eval/sgdet2d.py](../lib/mllm/eval/sgdet2d.py)). GT = OO pairs with annotated
  boxes on both endpoints; a triplet matches on class + predicate + 2D IoU ≥ 0.5 on both
  endpoints. Same ranking as the other regimes. Methods without 2D boxes score 0 by design.
* 3D IoU regimes are 0 (no OBBs).
* **Hallucination (new, optional `--halluc`)**: UOR / URR, computed for any method
  ([lib/mllm/eval/hallucination.py](../lib/mllm/eval/hallucination.py)):
  * UOR = predicted (frame, class) objects whose class is absent from the video's **world**
    GT inventory (every object annotated in any frame, observed or not) / predicted objects.
  * URR = triplets whose (person, object) pair is in the frame's GT but whose predicate is not
    among that pair's GT predicates / all predicted triplets; also split by the GT object's
    visibility (`urr_observed`, `urr_unobserved`) and restricted to pairs in GT
    (`urr_pairs_in_gt`). Both are upper bounds (AG labels are incomplete).

`score_all` takes `--sgdet2d --halluc` and `--no_full` (score only `--subset`); its markdown
gets an extra sgdet2d / hallucination table.

---

## 4. Environment

New conda env **`sgvlm`**, created by the user's `.condarc` under
`/data3/datasets/OakInk_pilot/conda_envs/sgvlm` (not `~/anaconda3/envs`): Python 3.12,
vLLM 0.30.0, torch 2.13.0, transformers 5.16.1, ms-swift 4.5.3, qwen_vl_utils. `scene4cast`
and `wsg` are untouched. pip needs `--isolated`: the user pip config lists the dead extra index
`pypi.ngc.nvidia.com`, and every package lookup otherwise waits through five DNS retries.

```sh
E=/data3/datasets/OakInk_pilot/conda_envs/sgvlm
~/anaconda3/bin/conda create -y -n sgvlm python=3.12
$E/bin/python -m pip --isolated install vllm==0.30.0
$E/bin/python -m pip --isolated install "ms-swift==4.5.3" qwen_vl_utils
```

Weights: `/data3/rohith/hf/sgvlm/checkpoints/{AG,PSG,PVSG}` (unzipped `checkpoints.zip`, kept
next to it). Source clone: `/data3/rohith/ext/SceneGraphVLM` (`a6197da`).

---

## 5. Commands (server, GPU 0, worktree `~/CODE/Scene4Cast_sgvlm`)

```sh
cd ~/CODE/Scene4Cast_sgvlm
W=~/anaconda3/envs/wsg/bin/python
S=/data3/datasets/OakInk_pilot/conda_envs/sgvlm/bin/python
# 1. frames (640x480) + Swift JSONL; CPU, any env with PIL
$W -m lib.mllm.methods.scenegraphvlm.runner prepare --tag t150 \
    --video_list /data3/rohith/ag/splits/test_worldbbox_thinking150.txt
# 2. inference; ~10-12 frames/s on an A40 (150 videos: 9.6 min)
CUDA_VISIBLE_DEVICES=0 $S -m lib.mllm.methods.scenegraphvlm.runner infer --tag t150 --batch_size 256
# 3. TOON -> PKLs + stats
$W -m lib.mllm.methods.scenegraphvlm.runner convert --tag t150
# 4. score
$W -m lib.mllm.eval.score_run --method scenegraphvlm --model sgvlm_ag --mode sgdet \
    --video_list /data3/rohith/ag/splits/test_worldbbox_thinking150.txt --subset_tag t150 \
    --no_legacy --sgdet2d --halluc --out results/ext_sgvlm_t150.json
```

The full split runs as `t150` + `rest_00..03` (the 1,361 other videos in four lists under
`/data3/rohith/ag/runs/mllm/scenegraphvlm/lists/`), queued by
`/data3/rohith/ag/logs/ext/sgvlm_full.sh`. All tags write PKLs into the same
`sgdet/sgvlm_ag/` directory.

Layout under `/data3/rohith/ag/runs/mllm/scenegraphvlm/`: `jsonl/<tag>.jsonl` (+`.meta.json`),
`raw/sgvlm_ag/<tag>.jsonl` (driver output), `sgdet/sgvlm_ag/<vid>.mp4.pkl`,
`stats/`, `authors_eval/`. Frames: `/data3/rohith/ag/cache/mllm/sgvlm_frames/`. Logs:
`/data3/rohith/ag/logs/ext/`.

**Checks before scoring.** The driver writes an error row and carries on when a batch fails,
so check `[done] non-empty predictions: N/N` in the infer log, then `raw_rows_missing: 0`,
`driver_errors: 0` and `parse_rate` in the convert stats, and the PKL count against the split.

---

## 6. Backbone-controlled ablation (`prev_graph`)

[lib/mllm/methods/prev_graph/runner.py](../lib/mllm/methods/prev_graph/runner.py): the
vendored `zero_shot` per-object prompt on Qwen3-VL-8B (predcls, `--skip-verification`,
T=0.2, as the `rag_all` std150 cell), in two arms: `noctx` (plain) and `prevgraph` (plus the
previous annotated frame's predicted graph, one line per object, capped at 600 characters ≈
150 tokens). The previous graph is the `noctx` arm's own prediction for frame *t*−1, so the
chain is two-pass, not autoregressive. The vendored runner batches every frame of a video in
one call; a true chain would need per-frame sequential calls. Outputs
`/data3/rohith/ag/runs/mllm/prev_graph/<arm>/predcls/qwen3vl_8b/`; score with
`score_run --method prev_graph --pred_dir <that root>/<arm>/ --model qwen3vl_8b`.
