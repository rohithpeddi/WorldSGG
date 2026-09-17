# lib/mllm — MLLM tracks for the worldbbox test set

Vendored from **`WorldSceneGraphAnnotationTool@c686be1`** (`backend/pseudo` and
`backend/evaluation`), which is not checked out on the server. Provenance:

| Here | Source (`backend/…`) | Change |
|---|---|---|
| `core/` | `pseudo/core/{ag_data,config_loader,logger_utils,prompts,retrieval,vgent}.py` | package-relative imports; `config_loader` defaults to `configs/mllm/server.yaml` (`$WSGG_MLLM_CONFIG` / `--config` override); `ag_data` no longer `mkdir`s under `/data` |
| `models/` | `pseudo/models/*.py` | unchanged (already relative imports) |
| `base_processor.py` | `pseudo/process_ag_base.py` | imports; `run(video_list=…)` filter |
| `methods/{zero_shot,caption_all,rag_all,wsg_agent}/runner.py` | `pseudo/process_ag_{zero_shot,caption_all,rag_all,wsg_agent}.py` | imports; `--video_list` |
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
