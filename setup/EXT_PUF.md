# PUF adapted to monocular per-timestamp WSGG

This note is the design record for the Phase 2 external baseline of `docs/EXTERNAL_BASELINES_PLAN.md`.
PUF is arXiv 2607.07170, with code at https://github.com/yyyyangyi/PUF (Apache-2.0).
In the paper, call it "PUF adapted to monocular per-timestamp WSGG".
Do not compare it with PUF's published 3DSSG or ReplicaSSG numbers.

- Code:
  - `lib/external/puf/`: the vendored PUF math in `puf_core.py`, plus `geometry.py`, `prior.py` and `fusion.py`. The license and attribution are in `NOTICE`.
  - `tools/ext_puf.py`: the driver, with the subcommands `geom`, `prior`, `run`, `subset` and `score`.
  - `scripts/remote/ext_puf.sh`: the runner, with the stages `smoke`, `s150` and `full`.
- Outputs are under `/data3/rohith/ag/runs/ext/puf/`:
  - `geom/<vid>.pkl`: the Pi3 geometry cache;
  - `prior/ag_train_prior.npz`: the fitted prior;
  - `<stage>/dumps/*.pkl`: the Track-1 dump records;
  - `<stage>/score/`: one reeval-style JSON and one bucketed JSON per dump.
- Logs are in `/data3/rohith/ag/logs/ext/`.
- The results table is in `analysis/ext_puf_<date>.md`.

## What PUF does (the parts we keep)

PUF runs online over the frames. Each frame goes through the same steps:

1. The 2D SGG front-end's detections are lifted to 3D Gaussians.
2. Each observation is associated with the global nodes. It gates on centroid L2 distance, then computes a likelihood:
   - spatial factor: `1 - Hellinger^2`;
   - semantic factor: `exp(-JSD(class Dirichlet means) / sigma)`.
3. JPDA marginals are computed as `beta = L / (lambda_birth + sum L)`. If `beta_birth > tau`, a new node is born.
4. Class and relation Dirichlet evidence is spread over every candidate with `beta >= beta_min`. Only the argmax node takes the geometry, by moment matching.

An optional class-conditional prior `s * P_exist * normalize(P_class * P_spatial)` then completes the edges that were never observed. It is also added to the observed edges.

FROSS, PUF's baseline, uses hard association instead: the same class plus Hellinger < 0.85, with vote counting.

We keep all of this. We use the defaults from PUF's README command for 3DSSG:

| Parameter | Value |
|---|---|
| `lambda_birth` | 0.4 |
| `sigma_jsd` | 0.3 |
| `tau_birth` | 0.5 |
| `beta_min` | 0.05 |
| `alpha_strength` | 1 |
| FROSS Hellinger threshold | 0.85 |

## Inputs (held fixed across methods)

**Front-end.** The front-end is the already-trained W-DSGDetr++ (resnet50), taken from its all-frame dumps. We use its outputs on observed pairs only:
- `/data3/rohith/ag/runs/rescore/dumps/w_dsgdetr_pp_{predcls,sgdet}_resnet50__all.pkl`
- 1,511 videos, 48,834 frames.

We never use WorldWise outputs. W-DSGDetr++'s own predictions for unobserved slots are discarded.

**Slots.** Every GT field is copied unchanged from the front-end dump. The scored slots, pairs, GT and detector boxes are therefore exactly the ones W-DSGDetr++ was scored on. These are Track 1's feature-covered slots, the "Covered" set of Phase 0.

**Geometry.** We read `points`, `conf` and `camera_poses` from `pi3_dynamic/<vid>_10/predictions.npz` and map them into the floor-aligned frame with `world_to_final`. Record `frame_t` maps to a Pi3 frame through the same `common_frames` as `WorldAG` and `WorldBBoxVideo.pi3_index_for_frame`. Pi3 covered 100% of the frames we checked.

**Detections.**
- predcls: GT boxes and classes on observed frames.
- sgdet: the feature-PKL detector boxes, labels and scores.

**Prior.** The prior is fitted on `world4d_rel_annotations/train` (7,516 videos). These are the training annotations the front-end was trained on. WorldBBox has no separate train split.

## Adaptation decisions and every deviation from PUF

1. **Monocular geometry.** PUF back-projects the box centre through a depth map and propagates the 2D box covariance through the pinhole Jacobian (its Eqs. 1–7). Pi3 gives dense points but no intrinsics, so we compute the two moments from the points instead:
   - mean: the median of the Pi3 points in the central 50% × 50% of the box (the same region PUF uses for its point cloud), after a foreground filter at camera distance median ± 2.5 MAD;
   - covariance: the empirical covariance of those points, plus a floor of `(0.03 · scale)^2 I`.

   The feature-PKL boxes live in the feature `target_size` pixel space, which differs from the Pi3 grid (for example 672×352 vs 672×378). They are rescaled per axis.
2. **Scale.** Pi3 is only defined up to scale; a typical person box is about 0.15 units tall. Every metric PUF constant is therefore expressed in units of `scale`, the median camera-to-point distance of the video:
   - the L2 gate is `1.5 · scale` (PUF: 3 m);
   - the spatial prior `sigma_d` is `0.5 · scale` (PUF: 1 m);
   - the covariance floor is `0.03 · scale`.
3. **Depth-less observations.** PUF drops them. We keep them and use the semantic likelihood only. In practice this never triggered: Pi3 lifted every observed slot.
4. **Class evidence.**
   - predcls: a one-hot class vector (PUF's GT-SG mode).
   - sgdet: the dump has no class softmax, so the detector score `s` goes on the predicted label and `(1 - s)` is spread uniformly over the other 34 object classes.
5. **Relation evidence for AG's heads.** PUF has a single softmax over predicates. AG has three heads:
   - attention (softmax, 3 classes): one categorical Dirichlet;
   - spatial (multi-label, 6) and contacting (multi-label, 17): one Beta per predicate, with evidence `(p, 1 - p)`.

   For FROSS vote counting, attention gets one vote at its argmax, and each multi-label predicate gets a +1 positive or negative vote at the 0.5 threshold. The read-out is the Dirichlet mean or the Beta mean.
6. **Dynamic person.** The person is a per-frame node that is never associated. Object nodes persist. Relation evidence accumulates on the (person track, object node) edge. AG videos have one person.

   **Observed pairs** are read from the per-frame person node's edge, `observed_readout = frame`. This is the evidence fused into the slot's node at frame t, including PUF's soft JPDA shares from the other observations of that frame. The alternatives are sensitivity rows:
   - `track`: the static-PUF accumulated edge;
   - `slot`: the observation's own edge, which equals the front-end.
7. **Causal per-timestamp read-out.** At each annotated frame t, the graph built from frames ≤ t is read out once for every valid pair `(person, slot)`:
   - **Slot observed at t:** fused evidence (point 6).
   - **Slot unobserved at t:** the slot is mapped to the node carrying the most class-c evidence. This is the protocol's own object identity, since AG pair keys are object classes and a frame never repeats a class in predcls. The output is that node's accumulated Dirichlet or Beta ("memory").
   - **No node for class c** (the object was never observed up to t): the prior in the prior arms, zeros otherwise.

   PUF itself evaluates a static global graph at the end of the scan. The per-timestamp snapshot is our addition.
8. **Prior.** In AG the subject is always the person, so `P_class` is indexed by the object class:
   - `P_att[c]`: categorical, Laplace 0.01;
   - `P_spa[c]` and `P_con[c]`: per-predicate Bernoulli rates;
   - `P_exist[c]`: the fraction of annotated (person, c) slots with at least one relation. This is about 1.0 for every class, so the gate is nearly inert in AG.

   PUF's spatial factor is hand-built for ScanNet support relations, which have no AG counterpart. We replace it with log-likelihood-ratio shifts on the Bernoulli predicates:
   - the contact predicates get proximity `1 - d/sigma_d`;
   - `not_contacting` gets the opposite shift;
   - `above` and `beneath` get the vertical offset;
   - `in` gets proximity;
   - there is no shift for front, behind and side (no facing direction) or for attention.

   The spatial factor needs the object node's centroid and the person's centroid at t. It is dropped for objects that were never observed. As in PUF, the prior is also added to observed edges (`prior_on_observed`). The `puf_prior_noobs` row switches that off.
9. **Completion threshold.** PUF completes an unobserved edge only if `max P > 0.8`, a precision guard for graph-level edge existence. In AG every person–object slot carries attention, spatial and contacting labels, and the metric is recall. The headline therefore uses 0 (always complete). The `puf_prior_ct08` row applies 0.8 to the attention head.
10. **Keyframes.** PUF can subsample keyframes. We fuse every annotated frame: all frames are scored, and AG frames are already sparse.
11. **Boxes.** For each node, we accumulate up to 1,500 foreground points from its boxes, trim them to the 5–95 percentile per axis, and fit an oriented floor-parallel box (`obb_floor_parallel_from_points`). The box is emitted as `pred_corners` for every slot mapped to a node. The run JSON reports the 3D IoU against the slot's GT `corners_final` (matched by label) as a side statistic. It is not a Phase-0 `sgdet3d` score.

## Arms

| Arm | Name in dumps | What it isolates |
|---|---|---|
| (ref) W-DSGDetr++ | `ref_wdsgdetrpp` | The front-end's own full output, including its trained unobserved-slot head |
| no memory | `puf_frontend` | Observed = front-end, unobserved = 0 (lower bound) |
| (a) naive LKS | `puf_lks` | Causal last-known state: last observed front-end output of the same object |
| (a') LKS bidirectional | `puf_lks_bi` | Nearest observed frame in either direction, the `lks_buffer` zero-order hold (non-causal) |
| (b) PUF − prior − uncertainty | `puf_fross` | FROSS hard association + vote counting |
| (c) PUF − prior | `puf_puf` | Likelihood association + soft Dirichlet evidence |
| (d) PUF full | `puf_puf_prior` | Plus the class-conditional prior |
| extension (not PUF) | `puf_puf_prior_vis` | The prior fitted on unobserved slots, used for unobserved objects |

The sensitivity rows (150 set) are `puf_track`, `puf_slot`, `puf_prior_slot`, `puf_prior_noobs`, `puf_prior_ct08`, `puf_prior_nospatial` and `puf_decay09`.

## Scoring

We use `tools/ext_puf.py score`. It applies `evaluate_wsgg_video` to every frame, with and without constraint, giving R, mR and hR at 10, 20, 50 and 100, in the same JSON layout as `tools/reeval_test.py` (`schemes.all`). `tools/bucketed_breakdown.process_dump` supplies the OO/OU/OU-nt buckets (no constraint, K = 20, as in the reference tables).

We score all frames in both predcls and sgdet. The slots match the reference rows, so the numbers are comparable to WorldWise++ and W-DSGDetr++ in `/data3/rohith/ag/runs/{rescore,worldwise_pp/score}`.

On the 150-video set, the reference W-DSGDetr++ row is re-scored on the same subset (`ref_wdsgdetrpp`). WorldWise++ is available on 1,511 videos only.

## Known protocol caveats (reported, not fixed)

- The bucketed SGDet evaluator does not enforce 2D IoU (`evaluation_recall_bucketed.py:207`). Its sgdet bucket numbers are class/slot matches on detector boxes.
- Track 1 covers only the 63% of slots that are feature-covered. The Phase-0 "Full" set is not used here.
- The sgdet slots are W-DSGDetr++'s resnet50 detector slots. WorldWise++ sgdet uses its own (dinov3) feature slots, so sgdet denominators differ across those two rows. The predcls GT is shared.
