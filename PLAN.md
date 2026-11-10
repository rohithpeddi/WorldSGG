# ICLR 2027 Rebuttal: Experiment Plan (submission 46061)

Drafted 2026-11-06 from the three reviews (SRrf 2, wqgS 2, y9Am 3) and a read-only inventory of CS93371.
Nothing has been run or changed yet; the only files written on the server are three read-only probe scripts under
`/data3/rohith/ag/runs/rebuttal/probe/`. Edit this file, then run it top to bottom.

Steps marked **NEW** need a script or code change first; every other command uses code already in the repo.

---

## 0. Decisions for you (defaults in bold; change any of them)

| # | Decision | Default | Alternatives |
|---|---|---|---|
| D1 | Methods in the no-training re-scores (A1, A3, A4) | **The 8 Table 2 rows** | Add WorldWise@ResNet-50 and the WorldWise+ pi3tok/fused cells |
| D2 | Validation split | **750 train videos (10%), fixed by video-id hash; train on the other 6,766** | 500 videos / stratified by #objects |
| D3 | Retraining scope (B1) | **Full: W-DSGDetr++ and WorldWise++ × {PredCls, SGDet} × seeds {0,1,2} = 12 runs, about 126 GPU-h** | Minimum: 3 seeds PredCls + seed 0 SGDet = 8 runs, about 72 GPU-h |
| D4 | Add WorldWise (base) to B1 | **No** | Yes: +3×(4.0+9.2) GPU-h |
| D5 | Feature for added unobserved slots (A2) | **Run both: zero and last-visible "hold"** | Only one |
| D6 | GraphRAG | **Concede and narrow; no new GPU** | Also run C2 (backbone-matched full split) |
| D7 | Blind relation audit (D-phase) | **600 unobserved object-frames, 2 annotators, about 20-30 h** | Smaller sample / skip |
| D8 | Baselines at DINOv3-L (B4) | **Skip** | Train W-DSGDetr++ @DINOv3-L, about 9 GPU-h |

---

## 1. What is on CS93371 (2026-11-06)

**Machine.** `ssh utd` (HostName 100.121.176.32). 3× A40 46 GB. Another user (txr110830) holds 14-22 GB on every
GPU at 1-4% utilisation, so about 24-31 GB is free per GPU. Disks: `/data` is full (115 GB free); `/data3` has 683 GB
free; `/code` (NVMe) has 231 GB free. **Write new runs to `/data3/rohith/ag/runs/rebuttal/`, not `/data`.**

**Repo.** Server `~/CODE/Scene4Cast` is on `methods` @ `8cb4558`, with `tools/rescore_common_gt.py` untracked and
`build_pp_grid_cache.py` modified. Local `methods` @ `8e3ad0c` has the rescorer committed.

**Checkpoints: every epoch was kept.** This is what makes the checkpoint-selection answer cheap.

| Table 2 row | Experiment (`<mode>` = predcls / sgdet) | Location | Epochs |
|---|---|---|---|
| W-STTran | `w_sttran_<mode>_resnet50_v2` | `/data/rohith/ag/checkpoints/` | 20 + best |
| W-STTran++ | `w_sttran_pp_<mode>_resnet50_v2` | same | 20 + best |
| W-DSGDetr | `w_dsgdetr_<mode>_resnet50_v2` | same | 20 + best |
| W-DSGDetr++ | `w_dsgdetr_pp_<mode>_resnet50_v2` | same | 20 + best |
| W-USG | `w_usg_<mode>_resnet50_v2` | same | 15 / 11 + best |
| WorldWise | `worldwise_<mode>_dinov3l_v2` | same | 20 + best |
| WorldWise+ | `worldformer_c1_dinov3tok_<mode>` (config `worldwise_plus_dinov3tok_<mode>`) | `/data3/rohith/ag/runs/worldformer/` | 20 + best |
| WorldWise++ | `worldwise_pp_dinov3_<mode>` | `/data3/rohith/ag/runs/worldwise_pp/` | 20 + best |

**Ablations already trained (20 epochs, both modes):**
`worldwise_abl_{notau,nomask,noema,noego,nomotion,notempedge}_<mode>_dinov3l_v2`.
`notau` = no logit adjustment, `nomask` = `p_simulate_unseen = p_mask_visible = 0`. The `nospatial` folders are empty.
The supplement's ablation table (`wsgg_predcls_ablation.tex`) scored these on the **older 1,526-video annotation, All
bucket only**:
- Full model: WC R@20 69.2 / mR@20 49.5
- Without logit adjustment: 70.5 / 41.7
- Without artificial masking: 70.0 / 49.0

They have never been scored by visibility bucket or on the 1,511-video split.

**Prediction dumps.** `/data3/rohith/ag/runs/{rescore,worldformer/score,worldwise_pp/score}/dumps/*__all.pkl`: all
frames, `best_model.pth`. `tools/dump_predictions.py` **drops `bboxes_3d` and `gt_corners`**, so 3D matching needs
new dumps.

**Coverage, measured on 150 test videos (4,558 frames) with `probe/probe_inventory.py` and `probe/probe_history.py`:**
- Visible objects: 11,753 / 11,753 have a PredCls slot (source `gt`).
- Unobserved objects: all 10,322 have annotation source `correction`, a 3D box (`corners_final`) and no 2D box.
  Only **2,001 (19.4%) have a slot, and that slot is always a GroundingDINO box (`gdino`) of the same class in that frame.**
  Triplets covered: 6,548 / 35,827 = **18.3%** (Table 20 reports 18.8%).
- Every uncovered unobserved object is visible in some frame of its video: 61% of its triplets have an *earlier*
  sighting and 39% only a *later* one. None is never visible. The reviewers' "count omitted objects" request is
  therefore also runnable as a full-inventory input experiment (A2), not only as a re-score (A1).

**SGDet unobserved slots are current-frame detections** (`probe/probe_sgdet_uo.py`, W-DSGDetr++ dump, all
48,834 frames):
- 47,942 slots are flagged unobserved, and every one has a 2D box in that frame. All test SGDet slots have source
  `detector` (`use_supply: False`).
- Only 11,667 (24%) share a class with an unobserved ground-truth object in that frame.

So in SGDet, persistence comes only from class matching. This is exactly SRrf W1 / wqgS W1 / y9Am W1; say it
plainly in the response.

**SGDet 3D boxes.**
- `build_pred_pkl` exports the **input** corners (detector boxes moved by camera pose), not a model prediction.
- Only WorldWise++ predicts 3D boxes (`pred["slot_corners"]`, `worldwise_pp/model.py:461`).
- The annotation-corner fallback (`world_ag_dataset.py` about line 585) fired **0 times in a 40-video probe**
  (12,352 ResNet-50 and 13,921 DINOv3-L slots all carry `boxes_3d`). Confirm on all 1,511 videos in A3.

**Training cost** (`t10_compute.tex`), per run, PredCls / SGDet:
- Baselines: 3.1-3.6 h / 4.0-4.4 h
- WorldWise @DINOv3-L: 4.0 / 9.2 h
- WorldWise++: 11.6 / 22.3 h
- Add 10-35 min start-up per job for the train video list.

Trainers are data-loader bound: set `num_workers: 4`. GPU memory is small (WorldWise++ peak about 0.7 GB), so two
trainers per GPU fit next to the other user's jobs.

**Geometric agreement numbers.** No script or result for the main text's "mean IoU3D 0.385" or "inter-annotator 3D IoU
>0.9" (`scene_construction.tex` L108-116) exists in `~/CODE/3DBBoxAnnotationTool` @ `51c4ca2` or on the server.
See D1 below.

---

## 2. Phase 0: Setup (about 30 min)

```bash
# local: branch for all rebuttal code
git switch -c exp/rebuttal methods && git push -u origin exp/rebuttal
```

```bash
# server: separate worktree so the main checkout stays untouched
ssh utd 'cd ~/CODE/Scene4Cast && sha1sum tools/rescore_common_gt.py && git fetch origin && git worktree add ~/CODE/Scene4Cast_rebuttal exp/rebuttal && sha1sum ~/CODE/Scene4Cast_rebuttal/tools/rescore_common_gt.py'
```

Gate: both rescorer hashes match (otherwise diff before scoring anything). All later commands run in
`~/CODE/Scene4Cast_rebuttal` with `PY=~/anaconda3/envs/scene4cast/bin/python`, CPU jobs with
`OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""`, long jobs with `setsid nohup ... </dev/null &`.

Output root: `/data3/rohith/ag/runs/rebuttal/{full_inv,inv_expand,dumps3d,score3d,epochs,abl,train,attr}`.

---

## 3. Phase A: no training (week 1, post results early)

### A1. Score against the full inventory, counting omitted objects as misses (wqgS Q1, SRrf Q1)
- **NEW** `tools/rescore_full_inventory.py`, derived from `rescore_common_gt.py`:
  - Build ground truth per frame from `world4d_rel_annotations_worldbbox/test/<vid>.mp4.pkl` (`object_info_list`,
    every object, visible or not).
  - Map each dump slot to an annotation object by short label (AG has one instance per class per frame; assert it).
  - A prediction hits if (person, object, predicate) is in that frame's full ground truth; everything else is a miss.
  - Score PredCls and SGDet-common, buckets All / OO / UO, per-frame R and mR, same as Table 2.
- Frame alignment: dump order is `(video, frame_t)`; annotation keys are `<vid>.mp4/<frame>.png`. Assert per frame
  that the dump's visible classes equal the annotation's visible classes.
- Gates:
  - (1) the covered subset reproduces every `common_gt` cell to 0.1 pt;
  - (2) totals are 773,236 triplets / 358,793 UO / 48,834 frames.
- Run, CPU, about 2-5 min per experiment:
  ```bash
  for e in w_sttran w_sttran_pp w_dsgdetr w_dsgdetr_pp w_usg worldwise_dinov3l worldwise_plus_dinov3tok worldwise_pp_dinov3; do $PY tools/rescore_full_inventory.py --exp $e --out /data3/rohith/ag/runs/rebuttal/full_inv; done
  ```
- Deliverables:
  - Table "Covered vs Full" for both modes.
  - One table putting supervised and zero-shot methods on the same 773,236-triplet ground truth.
- Expect: PredCls ranking unchanged (all methods miss the same objects); supervised UO recall bounded by the
  roughly 19% coverage. Report it anyway.

### A2. Full-inventory PredCls: give the models every object (SRrf Q1, y9Am W1)
- **NEW** `WorldAG(expand_inventory=True)`, PredCls only. For each annotation object without a slot in frame t,
  append a slot with:
  - class from `label`, `corners_final` (PredCls provides GT OBBs), `visibility=False`, zero 2D box;
  - visual feature = zero **and** (second run) the last visible feature of the same object (D5);
  - zero union feature; the (person, object) pair carries its GT relations.
  - For WorldWise++, the grid read must tolerate a zero box.
- Check in the model code that WorldWise / + / ++ replace invisible-slot features before the encoder. If they do,
  the fill does not matter for them and only baselines see the difference.
- **NEW** `tools/dump_predictions.py --expand-inventory {zero,hold}` (passes the flag to `make_test_dataset`).
- Gates:
  - (1) the dump's GT has 773,236 triplets;
  - (2) all pre-existing slots are byte-identical to the current dumps;
  - (3) A1 scorer on these dumps gives Covered = current numbers.
- Run: 8 methods × 2 fills, best checkpoints, about 2-3 GPU-h total.
  ```bash
  $PY tools/dump_predictions.py --config configs/methods/predcls/<cfg>.yaml --ckpt best --frames all --expand-inventory zero --out /data3/rohith/ag/runs/rebuttal/inv_expand/<exp>_zero.pkl
  ```
- Caveat to state: models were trained with the 19% inventory, so this is a train/test shift. If the drop is large,
  a retrain with an expanded train inventory is the follow-up (not budgeted).

### A3. Match objects in 3D (SRrf Q2, wqgS Q1)
- **NEW** `dump_predictions.py --keep-3d`: keep `bboxes_3d` and `gt_corners`; for `worldwise_pp` also export
  `pred["slot_corners"]` through the same camera transform as `bboxes_3d`.
- Re-dump SGDet for the 8 Table 2 methods (best checkpoint) to `rebuttal/dumps3d/`, about 2-4 GPU-h.
- **NEW** `tools/rescore_3d.py`:
  - Same as `sgdet_common`, but a slot matches a ground-truth object only if the class agrees **and** oriented 3D
    IoU ≥ τ, applied to visible and unobserved objects alike. Use `lib/mllm/eval/iou3d.py` or `compute_iou_3d_obb`.
  - τ ∈ {0.10, 0.25, 0.50}, plus a centre-distance variant. Coordinates are reconstruction units, not metres; say so.
  - Also emit WorldWise++ 3D-IoU histograms: input box vs refined `slot_corners`, visible vs unobserved (SRrf Q2).
- Gates:
  - (1) **frame convention**: visible slots that match at 2D IoU ≥ 0.5 must have a median 3D IoU clearly above 0. If
    it is about 0, the exported transform is wrong; this is the open SGDet camera/world issue in
    `WSGG_Paper/updated_submission/HIGH_PRIORITY_REVIEW_FIXES.md`. Stop and fix it before scoring.
  - (2) count annotation-corner fallbacks over all 1,511 videos (expect 0).
- Expect UO SGDet recall to fall for every method, because those slots are current-frame detections (Section 1).
  The interesting number is whether WorldWise++'s refinement keeps more of it.

### A4. Checkpoint selection without test scores (wqgS Q2, y9Am W3)
- **NEW** `tools/epoch_sweep.py`:
  - Per run, load each test video once, run all 20 epoch checkpoints (4-11 M parameters each, so all fit on one GPU),
    and score every epoch with the common-GT scorer (All / OO / UO, WC / NC, R / mR @10/20).
  - Score separately on two fixed halves of the test split (hash of video id).
- Report per method:
  - (a) **final epoch** (`checkpoint_19`; W-USG uses its last epoch);
  - (b) **cross-half selection**: select by WC R@20 on half A, report on half B, and the reverse;
  - (c) current `best_model.pth`;
  - (d) per-epoch curves.
- Gate: the epoch whose weights equal `best_model.pth` reproduces the Table 2 cells.
- Run: 8 methods × 2 modes = 16 sweeps, about 8-12 GPU-h.
  ```bash
  $PY tools/epoch_sweep.py --config configs/methods/<mode>/<cfg>.yaml --epochs 0-19 --out /data3/rohith/ag/runs/rebuttal/epochs/<exp>.json
  ```

### A5. Score the existing ablations on the reporting protocol (y9Am W2)
- Add the six `worldwise_abl_*` experiments to `DUMPS` in `rescore_common_gt.py`.
- Dump each × 2 modes (best checkpoint; final epoch via A4) to `rebuttal/abl/`: 12 dumps, about 3 GPU-h.
  ```bash
  $PY tools/dump_predictions.py --config configs/methods/<mode>/worldwise_abl_<x>_<mode>_dinov3l.yaml --ckpt best --frames all --out /data3/rohith/ag/runs/rebuttal/abl/worldwise_abl_<x>_<mode>__all.pkl
  ```
- Score with buckets, plus A1 (full inventory) and A3 (SGDet 3D).
- Key question: **does removing artificial masking hurt the UO bucket?** The All-bucket numbers say it barely
  matters (49.0 vs 49.5 mR@20). If UO shows no effect either, narrow the completion claim.

### A6. Feature-matched comparison (y9Am W2), assembly only
- WorldWise @ResNet-50 vs the ResNet-50 baselines, both modes, from `common_gt/results/worldwise_resnet50.json`
  (already scored). From my notes, SGDet WC R@20 is 22.9 vs 23.4-24.3, NC mR 24.2 vs 16.3-20.5. Re-read these from
  the JSON.
- There are **no baseline DINOv3-L checkpoints on the server**; the other direction needs B4 (D8).

---

## 4. Phase B: training (launch on day 2, runs in the background)

Launcher for all of these (existing): `tools/run_configs_multigpu.py --gpus <g> --per-gpu 2 --python $PY --configs <yaml...>`.
New configs go in `configs/methods/<mode>/rebuttal/`, with `save_path: /data3/rohith/ag/runs/rebuttal/train`,
`num_workers: 4`, and the seed set explicitly.

### B1. Validation split and seeds (wqgS Q2, y9Am W3)
- **NEW**:
  - `splits/rebuttal_val750.txt`: 750 train video ids, chosen by `sha1(id) mod 10 == 0`, then truncated or filled
    to 750 deterministically.
  - `WorldAG(video_list=...)` filter.
  - `train_wsgg_base._make_dataset("val")`, with selection on val (same WC R@20 last-frame criterion for parity).
  - Fix: save `_best_score` in `checkpoint_state.pth` (the resume bug noted in `t22_manifest.tex`).
- Train on the other 6,766 videos, select on val, then score the 1,511 test videos **once** with the common-GT scorer
  and with A1, A3.
- Runs (D3 default): W-DSGDetr++ and WorldWise++ × {PredCls, SGDet} × seeds {0,1,2}.
  - About 3×(3.6+4.4) + 3×(11.6+22.3) ≈ 126 GPU-h; with 2 trainers/GPU on 3 GPUs, about 1.5-3 days wall-clock.
  - Start the three WorldWise++ SGDet seeds first (longest).
- Report: mean ± std over seeds, and the val-selected vs final-epoch gap.
- Compare B1 rows with each other, not with Table 2 (10% less training data).

### B2. Baseline with the tail-aware loss (y9Am W2)
- **NEW**: `use_logit_adjustment` / `logit_adjustment_tau` in the baseline loss. Port the per-class log-prior offset
  on relation logits (train only) from `lib/supervised/worldwise/loss.py`; priors
  `features/predicate_priors_<mode>.json`, τ = 0.5.
- Train W-DSGDetr++ + LA × {PredCls, SGDet}, about 8 GPU-h.
- Together with `worldwise_abl_notau` (A5) this gives a 2×2 {baseline, WorldWise} × {±LA}. That table answers
  "architecture or loss?" head-on.

### B3. Completion mechanism vs simpler memory (y9Am W2)
- **NEW**: WorldWise flag `completion: {mwae, hold_last_visible, mask_token}`.
  - `hold_last_visible` replaces the camera-pose-conditioned retrieval for invisible slots with the last visible
    feature of the same object.
  - `mask_token` uses a learned token with no retrieval.
  - Everything else, including the loss, is unchanged.
- Train @DINOv3-L × {PredCls, SGDet} × 2 variants, about 2×(4.0+9.2) ≈ 26 GPU-h.
- Score with buckets and on the A2 full inventory (where completion has the most to do).
- This is the decisive experiment for "does the completion architecture matter". The PUF-style rows already in
  Appendix G are a training-free memory comparison.

### B4. (optional, D8) W-DSGDetr++ @DINOv3-L × {PredCls, SGDet}, about 9 GPU-h.

**GPU schedule** (for the D3 default):

| GPU | Runs |
|---|---|
| 0 | WW++ SGDet s0, s1 |
| 1 | WW++ SGDet s2, then WW++ PredCls s0 |
| 2 | WW++ PredCls s1, s2 |
| Any free slot | DSGDetr++ ×6, B2 ×2, B3 ×4 |

Check `ps -o stat=` for `D` state and `nvidia-smi` utilisation after launch: if everything sits in I/O wait, drop to
one trainer per GPU.

---

## 5. Phase C: zero-shot

- **C1 (default, no GPU): concede and narrow GraphRAG.** Table 30 (`ext_mllm_context150.tex`), WC R@20 / mR@20:

  | Context | R@20 | mR@20 |
  |---|---|---|
  | GraphRAG | 47.4 | 28.8 |
  | No context | 47.8 | 30.6 |
  | No context, second draw | 47.8 | 30.7 |
  | Random training graph | 50.9 | 29.4 |
  | SGR3-style k=3 | 52.7 | 30.6 |

  The mR loss exceeds draw-to-draw noise, and the gains from retrieved text come mostly from vocabulary exposure.
  Reframe the zero-shot contribution as the benchmark, protocol and findings (localization and the preceding-frame
  graph help; retrieved graph text does not).
- **C2 (optional, D6):** backbone-matched full-split `rag_all` on Qwen3-VL-8B, standard decoding, under the 1,511 R/mR
  protocol, about 12 GPU-h per mode. Blockers:
  - (1) verify the caption-key defect is fixed (`caption_all/runner.py:197` and `rag_all/runner.py:203` read only
    `caption`; 442 videos ran with 0 captions);
  - (2) vLLM at `gpu_memory_utilization 0.85` needs about 39 GB, but only 24-31 GB is free while the other user's jobs
    run. Use about 0.55 or wait.
- **C3 (optional):** Table 3 (`uwsgg_gt_plus_corrections.tex`) is P/R/F1 on a different video subset. Either re-score
  its runs on the 1,511 R/mR protocol or state the difference explicitly (y9Am's "unified test protocol").

---

## 6. Phase D: annotation (people, not GPUs)

- **D1 (blocking for the SRrf reply): provenance of 0.385 and >0.9.** The main text says correction "raises the
  inter-annotator agreement measured by 3D IoU to >0.9, ensuring metrically accurate and temporally consistent bounding
  boxes across all frames".
  - No code or result for either number was found.
  - A 2026-09-25 check found the released 1,511 test boxes are automatic refits with no annotator box edits (Firebase
    edited overlays were not checked).
  - Owner: whoever ran the study.
  - Outcome A: a documented study (sample size, reference boxes, who annotated) → report it.
  - Outcome B: not reproducible → correct the claim in the revision and say so ourselves in the response.
- **D2: blind relation audit (SRrf Q3).**
  - Stratified sample of about 600 unobserved object-frames (the probe gives about 69 per video).
  - Two annotators, VLM proposals hidden.
  - Report agreement with released labels and Cohen's κ per axis (attention / spatial / contacting).
  - About 20-30 annotator-hours.
- **D3: rule for relations that cannot be determined from the video.** Find it in `sup_manual_correction_wsg.tex`; if
  it is not written down, write it.

---

## 7. Phase E: writing

- **E1 (end of week 1):** general response with A1 + A3 + A4 tables (all no-training). Posting early starts the
  discussion while B runs.
- **E2:** per-reviewer replies.
  - wqgS: A1, A3, A4, B1.
  - SRrf: A1, A2, A3, C1, D1, D2.
  - y9Am: A2, A5, A6, B1, B2, B3, `t10_compute`.
- **E3:** revised PDF.
  - Move protocol (coverage, matching, selection) into the main text.
  - Table 2 with Covered and Full GT.
  - Ablations on the reporting protocol.
  - GraphRAG narrowed.
  - Compute numbers in the main text.
  - Correct the geometric-agreement sentence per D1.

## 8. Timeline (check the discussion deadline in the author console)

| Days | Work |
|---|---|
| 1 | Phase 0; write `rescore_full_inventory.py`, `--keep-3d`, `epoch_sweep.py`; B1/B2 code + configs; launch B1 |
| 2-3 | A1, A3, A4, A5 runs; launch B2, B3; start D1 and D2 |
| 4-5 | A2 (inventory expansion); post E1 |
| 6-10 | B results land; per-reviewer replies; D2 results |
| 11+ | Revised PDF, final replies |

## 9. Traps (from earlier sessions)

- The non-interactive SSH shell is `sh`: call the env's python directly; `source`/`conda activate` do not work.
- `pkill -f <pattern>` over ssh kills the ssh shell itself if the pattern appears in the command; use the `patter[n]`
  trick.
- Trainers default to `num_workers=0`; every job start spends 10-35 min building the train video list.
- `_best_score` is not checkpointed, so resumed runs select among post-resume epochs only (fixed in B1).
- GroundingDINO pickles hold CUDA tensors: set `CUDA_VISIBLE_DEVICES=""` for CPU-only reads.
- MLLM score JSONs store fractions; the tables are percent.
- Quote training numbers only from the common-GT scorer; never reuse the old SGDet or pooled-bucket numbers.
