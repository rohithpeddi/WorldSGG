# Running WSGG Training & Evaluation (Baselines + WorldWise)

End-to-end recipe for the supervised WSGG campaign. Architecture references:
[BASELINES.md](BASELINES.md), [WORLDWISE.md](WORLDWISE.md). A deeper
feature-extraction walkthrough lives in `docs/METHODS_README.md` (note:
`docs/` is git-ignored — that file exists on machines where it was authored,
not in fresh clones).

**Final campaign design:** WorldWise's main configuration is the round-2
winner **v2e** (pair geometry + τ=0.5 + λ_vlm=0 + mask 0.3; experiment stem
`worldwise_v2e`, so historical v2e cells are reused). Three parts × 2 tasks
(predcls, sgdet), 1 seed (0), identical budget everywhere:

1. **Baselines, unchanged** @ resnet50 — the method-comparison controls.
2. **WorldWise (v2e)** @ all four backbones — main method + scaling.
3. **Component-wise ablations** of the main config @ dinov3l — one change
   per tier (see [WORLDWISE.md](WORLDWISE.md) for the 10-tier table).

## 0. Prerequisites

- ActionGenome4D at `{data_path}` (default `/data/rohith/ag`) with `frames/`,
  `annotations/`, `world_annotations/`, `world4d_rel_annotations/{train,test}/`,
  and `features/clip_features/clip_text_embeddings.npy`.
- Trained detector checkpoints per backbone (see [RUN_MON3D.md](RUN_MON3D.md)).
- **Annotation folders per split** are config keys (2026-09-17): `train_annot_dir`
  (default `world4d_rel_annotations`) and `test_annot_dir` (all method configs now
  set `world4d_rel_annotations_worldbbox`, the 1,511-video WorldBBox release set).
  Both are CLI-overridable. Before reporting numbers on a new test folder run
  `python tools/check_worldbbox_alignment.py --annot_dir <folder> --ref_annot_dir world4d_rel_annotations`
  (feature/annotation object alignment, zero-corner slots, floor height) and lock the
  set with `python tools/make_test_split_worldbbox.py` (writes the video list and an
  `annotation_version` into `/data3/rohith/ag/cache/manifest.json`; see `tools/cache_manifest.py`).
- Environment: torch + CUDA.
- **wandb** (`use_wandb: true` in every config): all cells log to one shared
  project `wandb_project` (default `worldsgg-v2`), each run named by
  `experiment_name` and grouped by `method_name`, so the ladder and plugin
  tiers overlay on one dashboard. **Before a multi-process launch, authenticate
  once** (`wandb login` or `export WANDB_API_KEY=…`) — an unauthenticated
  `wandb.init` can block on a prompt and hang every concurrent slot. To skip
  the network entirely: `export WANDB_MODE=offline`, or set `use_wandb: false`
  (metrics still land in `results/*_metrics.jsonl`, the source of truth).

## 1. One-time feature extraction (per backbone × mode)

Converts frames + detector checkpoints into per-video PKLs
(`{data_path}/features/roi_features/<mode>/<backbone>/{train,test}/`) holding
ROI features, boxes, 3D corners, labels, pair indices, union features.

```bash
# predcls (GT boxes) / sgdet (detector boxes) — one config per backbone
python datasets/preprocess/features/extract_roi_features_predcls.py \
    --config configs/features/predcls/ex_roi_feat_resnet50_predcls_rohith.yaml
python datasets/preprocess/features/extract_roi_features_sgdet.py \
    --config configs/features/sgdet/ex_roi_feat_resnet50_sgdet_rohith.yaml
# ... repeat with the dinov2b / dinov2l / dinov3l feature configs
```

Needed combinations for this campaign: **resnet50 × both modes** (all five
methods) and **dinov2b/dinov2l/dinov3l × both modes** (WorldWise only).

## 2. One-time setup on the training server

```bash
# a. Smoke-test every model + plugin flag (forward/backward, no data needed)
python tests/test_wsgg_forward.py

# b. Param-count freeze artifact (asserts the ladder is monotone)
python tools/dump_param_counts.py --out docs/param_counts.md

# c. Regenerate configs (16 base + 16 tier = 32, deletes stale ones)
python tools/gen_grid_configs.py --tiers

# d. Predicate priors (used by WorldWise's tail-aware logit adjustment)
python tools/compute_predicate_priors.py --data_path /data/rohith/ag \
    --feature_model dinov2b --mode predcls   # and --mode sgdet
```

## 3. Training

### Single run

```bash
python train_wsgg_methods.py --config configs/methods/predcls/worldwise_predcls_dinov3l.yaml
# any YAML key can be overridden on the CLI, e.g. --lr 5e-5 --nepoch 10
```

Checkpoints: `{save_path}/{experiment_name}/checkpoint_N/checkpoint_state.pth`
(+ `best_model.pth` at the best wc/R@20). Resume with `--ckpt checkpoint_N`.

### Full campaign on 3 GPUs (recommended)

```bash
python tools/run_grid_multigpu.py --gpus 0 1 2 --compute-priors
```

`--per-gpu` worker slots per GPU (default 3) pull from a shared queue.
Completed cells are skip-detected — historical v2e / v2a / v2f / v2g results
are reused via their experiment names, so the command only launches what is
missing:

| Stage | Cells | Purpose |
|---|---|---|
| `table_a` | 4 baselines + worldwise × resnet50 × 2 modes (10) | The method ladder |
| `scaling` | worldwise × dinov2b/2l/3l × 2 modes (6) | Table B backbone scaling |
| `abl` | 10 ablation tiers × dinov3l × 2 modes (20) | Table C component ablations |

(Earlier exploratory rounds — plugin stages, τ sweep, subtraction study, v2
candidates — are concluded; their verdicts live in `docs/DECISION_LOG.md` and
their configs were removed. Historical results remain in `results/`.)

- Per-run logs: `logs/grid/<experiment>.log`; live status:
  `results/grid_run_status.csv`.
- **Resume-safe**: completed runs (final-epoch row present in the metrics
  jsonl) are skipped — just re-run the same command after interruptions.
- Subsets: `--stages table_a scaling stage_a`, `--modes predcls`, `--dry-run`.
- Sequential single-GPU alternative: `tools/run_grid.py` (same cells,
  `--tiers plus1 ...` for tier configs).

## 4. Evaluation

Every training run already evaluates after each epoch (with-constraint +
no-constraint + occlusion-stratified). Standalone testing of a checkpoint:

```bash
python test_wsgg_methods.py \
    --config configs/methods/predcls/worldwise_predcls_dinov3l.yaml \
    --ckpt checkpoint_19
```

## 5. Results & aggregation

Each run appends to `results/<experiment_name>_metrics.jsonl`
(experiment names carry the `_v2` suffix — `_v1` results predate the
July 2026 fixes and are not comparable):

- a one-time **header row** (git commit, full config incl. plugin flags,
  param counts);
- one row per epoch: `wc|nc / R|mR|hR @ {10,20,50,100}`, the per-predicate
  recall vector at K=20 (`wc/per_predicate_R@20`), occlusion-stratified
  `vispair/*` and `occpair/*` metrics, all loss sub-terms (`loss/*`),
  `epoch_time_s`, `peak_vram_gb`.

Render the three markdown tables (best epoch by wc/R@20 per cell):

```bash
python tools/aggregate_results.py --mode predcls --tiers   # Tables A, B, C + CSV
python tools/aggregate_results.py --mode sgdet  --tiers
python tools/report_gate_metrics.py --mode predcls         # per-head + occlusion split
```

- **Table A** — 4 baselines + WorldWise (v2e) @ resnet50 (the ladder) ·
  **Table B** — WorldWise across backbones · **Table C** — component-wise
  ablations @ dinov3l (each row = main config minus/altering one thing).

**Paper-ready LaTeX tables** (the 4-way R/mR × with/no-constraint layout):

```bash
python tools/gen_paper_tables.py   # → results_tables/{predcls,sgdet}_{main,ablation}.tex + preview.tex
```

Compile a proof: `cd results_tables && pdflatex preview.tex`. A one-paragraph
reading of the current numbers (predcls win, sgdet trade, ablation effects,
backbone inversion) is in [WORLDWISE.md](WORLDWISE.md#measured-results).

> **Known gap:** the `abl_nospatial` ablation (`− ObjectSpatialEncoder`)
> produced no result in either mode (blank row in the ablation tables). Check
> `logs/grid/worldwise_abl_nospatial_*_dinov3l.log` and rerun the two cells:
> `python tools/run_grid_multigpu.py --gpus 0 1 2 --stages abl` (skip-detect
> leaves the completed ablations alone).

## 6. Decision log

Every keep/drop/composition decision of the campaign — including how v2e was
selected — is recorded in `docs/DECISION_LOG.md` with hypothesis, run IDs,
deltas, and verdicts. New ablation findings append there.

## Troubleshooting

- `worldwise_conf_*` (I-5) warns "no vlm_confidence tensor" — expected: the
  plugin is inert until features are regenerated with per-label VLM
  confidences; it falls back to the global λ_vlm.
- All-plugin-flags-off WorldWise must match the I-0 reference —
  `tests/test_wsgg_forward.py` asserts no plugin modules exist in that case.
- Three concurrent runs share CPU comfortably (`batch_size=1`,
  `num_workers=0`) but each holds its feature set in RAM — watch memory at
  campaign start.
