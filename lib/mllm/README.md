# lib/mllm — MLLM tracks for the worldbbox test set

Vendored from **`WorldSceneGraphAnnotationTool@c686be1`** (`backend/pseudo` and
`backend/evaluation`), which is not checked out on the server. Provenance:

| Here | Source (`backend/…`) | Change |
|---|---|---|
| `core/` | `pseudo/core/{ag_data,config_loader,logger_utils,prompts,retrieval,vgent}.py` | package-relative imports; `config_loader` defaults to `configs/mllm/server.yaml` (`$WSGG_MLLM_CONFIG` / `--config` override); `ag_data` no longer `mkdir`s under `/data` |
| `models/` | `pseudo/models/*.py` | unchanged (already relative imports) |
| `base_processor.py` | `pseudo/process_ag_base.py` | imports; `run(video_list=…)` filter |
| `methods/{zero_shot,caption_all,rag_all}/runner.py` | `pseudo/process_ag_{zero_shot,caption_all,rag_all}.py` | imports; `--video_list` |
| `methods/graphs/runner.py` | `pseudo/process_ag_graphs.py` | imports; `--video_list`; sorted order |
| `eval/legacy/` | `evaluation/{evaluate_relationships,load_method_outputs,label_constants}.py` | imports; log file no longer written into the repo |
| `video_splits.json`, `LICENSE`, `requirements.vendored.txt` | `pseudo/` | copied |
| `configs/mllm/server.yaml` | `pseudo/config_utd.yaml` | outputs → `/data3/rohith/ag/runs/mllm`, graphs → merged cache dir, worldbbox block, `qwen3vl_8b_thinking` |

Dropped: `core/{config,data}.py` (Vgent benchmark helpers), `demo.py`, box sync,
frame dumping, visualisers — not needed for the worldbbox tracks.

New code (this repo): `data/worldbbox.py` (B1), `eval/{dump_adapter,legacy_f1}.py`
(B2), `methods/track_a_prompt/`, `methods/track_b_agent/` (B5/B6). See
`docs/EXECUTION_PLAN_WORLDBBOX.md` §3.

## Running (server, GPU 0 only, `wsg` env)

```sh
cd ~/CODE/Scene4Cast_mllm
CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/wsg/bin/python -m lib.mllm.methods.zero_shot.runner \
    --model_name qwen25vl_7b --mode predcls --tensor_parallel_size 1 --skip-verification \
    --video_list /data3/rohith/ag/splits/test_worldbbox_1511.txt
```
Long jobs go through `scripts/remote/run_mllm_job.sh` (setsid nohup + `status.json`).

## Layout of the new code (B1-B7)

| Path | Step | What |
|---|---|---|
| `data/worldbbox.py`, `data/geometry.py` | B1 | worldbbox split/annotation/Pi3 join in the canonical frame; OBB (center,size,yaw) <-> corners; gate `tools/check_worldbbox_mllm_adapter.py` |
| `eval/dump_adapter.py`, `eval/recall3d.py`, `eval/iou3d.py`, `eval/legacy_f1.py`, `eval/score_run.py`, `eval/score_all.py` | B2/B7 | run -> dump records -> stock WorldSGG evaluator (+ buckets), 3D-IoU regimes, legacy F1; protocol in `eval/README.md` |
| `tools/bev.py`, `tools/detections.py`, `tools/lift.py`, `tools/marks.py`, `tools/llm_cache.py`, `tools/build_caches.py`, `tools/register_graphs.py` | B4 | annotation-independent caches (`/data3/rohith/ag/cache/mllm/{bev,detections,lifted3d,llm_cache,graphs}`) |
| `methods/track_a_prompt/` | B5 | marked frames + BEV -> compact JSON (predicates, OBBs); `--dry_run` builds prompts without a model |
| `methods/track_b_agent/` | B6 | fixed tool loop + `critic.py` (`check_geometry`) + repair; `objects_pre` = -critic arm |

Server queue: `scripts/remote/run_mllm_sequence.sh <worker> /data3/rohith/ag/logs/b3.jobs` (jobs in
`scripts/remote/b3_baselines.jobs`; per-job `logs/<job>.log` + `logs/<job>.status.json`; lock dirs allow a
second worker on another GPU with `MLLM_GPU=1`). Scoring: `python -m lib.mllm.eval.score_run --method
<m> --model <k> --mode <predcls|sgdet> [--video_list ... --subset_tag ...] [--objects_key objects_pre]`
per run, `python -m lib.mllm.eval.score_all --out_prefix results/mllm_worldbbox_<date>` for the tables.
