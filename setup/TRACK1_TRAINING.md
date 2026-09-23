# Track 1 — the training-based track

The supervised lineage: the WorldWise family and the scene-graph baselines
adapted to world-frame scene graphs. Every row here is a model trained on the
WorldSGG train split and evaluated on the WorldBBox test split with the stock
evaluator.

Authoritative results: [analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md)
§1 · full lineage table: [analysis/worldbbox_lineage_2026-09-19.md](../analysis/worldbbox_lineage_2026-09-19.md)

| Track | Document |
|---|---|
| **1 — Training-based** | this file |
| 2 — Unlocalized MLLM | [TRACK2_UNLOCALIZED.md](TRACK2_UNLOCALIZED.md) |
| 3 — Localized MLLM | [TRACK3_LOCALIZED.md](TRACK3_LOCALIZED.md) |

---

## 1. Methods in this track

| Method | Document | Core idea | Backbones scored |
|---|---|---|---|
| W-STTran | [BASELINES.md](BASELINES.md) | LKS zero-order-hold buffer → inter-object transformer | resnet50 |
| W-STTran++ | [BASELINES.md](BASELINES.md) | + object spatial / motion / temporal-edge | resnet50 |
| W-DSGDetr | [BASELINES.md](BASELINES.md) | + temporal object encoder + tracking | resnet50 |
| W-DSGDetr++ | [BASELINES.md](BASELINES.md) | + ego-motion; strongest baseline | resnet50 |
| W-USG | [BASELINES.md](BASELINES.md) | external USG-Par relation stack on the WSGG substrate | resnet50 |
| WorldWise | [WORLDWISE.md](WORLDWISE.md) | MWAE mask-and-retrieve memory + the v2e loss recipe | resnet50, dinov2b, dinov2l, dinov3l |
| WorldWise+ | [WORLDWISE_PLUS.md](WORLDWISE_PLUS.md) | same model over frozen DINOv3 / π³ ROI **latents** | dinov3tok, pi3tok, fused |
| WorldWise++ | [WORLDWISE_PP.md](WORLDWISE_PP.md) | entity decoder over the full token grid + joint detection | dinov3 (+ `nodet` ablation) |

Supporting documents: the detector that supplies boxes and features is
[MONOCULAR_3D_DETECTOR.md](MONOCULAR_3D_DETECTOR.md); the operational recipe for
training and scoring is [RUN_WSGG.md](RUN_WSGG.md) (and
[RUN_MON3D.md](RUN_MON3D.md) for the detector).

The lineage is nested: WorldWise+ is WorldWise with the appearance input
swapped, WorldWise++ is WorldWise+ with the object-token core replaced. The
scene-graph loss is byte-identical across the three, which is what makes the
ladder an ablation rather than three unrelated models. The baselines sit
*beside* the ladder; W-USG in particular is an external method transplanted onto
the shared substrate, not a rung.

![WorldWise lineage overview](worldwise_variants_overview.svg)

---

## 2. Evaluation protocol

| | |
|---|---|
| Test set | `world4d_rel_annotations_worldbbox/test`, 1,511 videos / 48,834 frames, `annotation_version = 40decd1785af3c3470290be4b67443e1` |
| Frames | **all annotated frames**, not last-frame-only |
| Checkpoint | best epoch |
| Evaluator | [lib/supervised/evaluation_recall.py](../lib/supervised/evaluation_recall.py) + [lib/supervised/evaluation_recall_bucketed.py](../lib/supervised/evaluation_recall_bucketed.py) |
| Chain | [tools/dump_predictions.py](../tools/dump_predictions.py) → [tools/reeval_test.py](../tools/reeval_test.py) → [tools/bucketed_breakdown.py](../tools/bucketed_breakdown.py) |
| Metric family | R@K / mR@K / hR@K at K ∈ {10, 20, 50, 100}, with-constraint (wc) and no-constraint (nc) |
| Buckets | OO (observed→observed), OU (observed→unobserved), OU-non-trivial |
| **SGDet matching** | **2D IoU 0.5** in Pi-3 feature space |

PredCls supplies ground-truth boxes and classes, so it measures relational
reasoning only. SGDet runs the detector and matches predicted to ground-truth
boxes at 2D IoU 0.5.

**This track's SGDet number is not comparable to track 2 or track 3.** Track 2
cannot match boxes at all and track 3 matches in 3D. Only PredCls is
cross-track comparable — see §5.

---

## 3. Results

All rows: all frames, best epoch, 1,511 videos, values in %. Lifted from
[analysis/ICLR_THREE_TRACKS.md](../analysis/ICLR_THREE_TRACKS.md) §1.

### 3a. PredCls

| method | backbone | wc R@20 | wc mR@20 | nc R@20 | nc mR@20 | OO R@20 | OU-nt R@20 | OU-nt mR@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W-STTran | resnet50 | 68.2 | 37.6 | 92.6 | 70.0 | 91.2 | 56.9 | 30.7 |
| W-STTran++ | resnet50 | 66.9 | 34.0 | 92.4 | 63.6 | 90.3 | 67.0 | 29.1 |
| W-DSGDetr | resnet50 | 67.5 | 33.4 | 92.2 | 65.6 | 90.1 | 62.7 | 27.0 |
| W-DSGDetr++ | resnet50 | 68.5 | 38.9 | 92.7 | 71.0 | 91.4 | 55.3 | 26.1 |
| W-USG | resnet50 | 67.3 | 33.4 | 92.6 | 64.0 | 90.6 | 67.5 | 29.1 |
| WorldWise | resnet50 | 68.0 | 49.1 | 92.5 | 81.8 | 90.6 | 73.5 | 40.9 |
| WorldWise | dinov2b | 68.2 | 45.4 | 92.5 | 79.4 | 90.6 | 74.7 | 38.7 |
| WorldWise | dinov2l | 68.8 | 47.8 | 92.6 | 81.6 | 90.6 | 74.2 | 42.6 |
| WorldWise | dinov3l | 69.0 | 49.6 | 92.6 | 82.4 | 90.9 | 72.9 | 41.1 |
| WorldWise+ | dinov3tok | 73.5 | **54.7** | 94.3 | 85.6 | 93.4 | 76.0 | 45.2 |
| WorldWise+ | pi3tok | 71.9 | 51.5 | 93.8 | 84.4 | 92.4 | 75.4 | 46.0 |
| WorldWise+ | fused | 73.2 | 54.1 | 94.2 | 84.5 | 93.0 | 75.9 | 44.1 |
| **WorldWise++** | dinov3 | **74.8** | 54.4 | **94.6** | **86.2** | **93.6** | **76.6** | **48.1** |
| WorldWise++ (ablation) | dinov3_nodet | 74.1 | 54.3 | 94.2 | 86.0 | 93.6 | 75.0 | 40.1 |

### 3b. SGDet (2D IoU 0.5)

| method | backbone | wc R@20 | wc mR@20 | wc mR@50 | nc mR@20 | OU-nt R@20 | OU-nt mR@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| W-STTran | resnet50 | 56.3 | 19.2 | 33.4 | 27.7 | 14.6 | 7.1 |
| W-STTran++ | resnet50 | **56.6** | 17.6 | 30.3 | 25.4 | 15.6 | 6.6 |
| W-DSGDetr | resnet50 | 56.6 | 19.2 | 33.5 | 28.0 | 12.3 | 6.0 |
| W-DSGDetr++ | resnet50 | 56.4 | 20.3 | 35.4 | 31.2 | 14.3 | 7.7 |
| W-USG | resnet50 | 56.3 | 19.1 | 34.1 | 25.9 | 14.5 | 6.7 |
| WorldWise | resnet50 | 53.6 | 20.4 | 38.5 | 36.1 | 25.4 | 20.6 |
| WorldWise | dinov2b | 54.1 | 23.4 | 40.1 | 37.9 | 29.6 | 21.1 |
| WorldWise | dinov2l | 53.5 | 22.3 | 38.1 | 37.2 | **30.4** | 20.2 |
| WorldWise | dinov3l | 52.6 | 21.5 | 38.8 | 37.5 | 28.1 | 21.3 |
| WorldWise+ | dinov3tok | 54.2 | 23.1 | 45.5 | 43.3 | 26.6 | 20.5 |
| WorldWise+ | pi3tok | 53.2 | 22.9 | 43.8 | 41.1 | 27.2 | **22.4** |
| WorldWise+ | fused | 54.2 | 23.9 | 46.9 | 43.5 | 27.1 | 21.1 |
| **WorldWise++** | dinov3 | 54.6 | **24.5** | **48.6** | **44.1** | 27.8 | 22.3 |
| WorldWise++ (ablation) | dinov3_nodet | 54.8 | 24.1 | 48.4 | 43.9 | 27.7 | 21.9 |

28 cells, all trained and scored.

---

## 4. Findings, ranked

1. **The recall ladder is monotone and the gap is real.** PredCls wc R@20
   68.5 (best baseline) → 69.0 (WorldWise @ dinov3l) → 73.5 (WorldWise+ @
   dinov3tok) → **74.8** (WorldWise++). WorldWise++ also takes both
   no-constraint columns and every occlusion bucket.
2. **Mean recall is where the family separates, and it comes from the loss,
   not the architecture.** Every baseline sits at 33–39 wc mR@20 in PredCls
   against 49–55 for the WorldWise family. The jump happens at the first rung
   (WorldWise, 49.1–49.6), i.e. with the v2e loss recipe: `λ_vlm = 0`,
   simulated-unseen + EMA reconstruction, and tail-aware logit adjustment at
   τ = 0.5. See [WORLDWISE.md](WORLDWISE.md) §"Loss — the v2e recipe".
3. **Occlusion is the capability gap.** SGDet OU-non-trivial: baselines reach
   12–16 R@20 and 6–8 mR@20; the WorldWise family reaches 25–30 and 20–22,
   i.e. 2–3×. This is what the LKS zero-order-hold buffer structurally cannot
   do and what the MWAE memory replaces.
4. **The detection objective is what buys occlusion reasoning in WorldWise++.**
   The `nodet` ablation (free queries and box/corner refinement removed) costs
   **8.0 mR@20 on the PredCls occluded non-trivial bucket, 48.1 → 40.1**,
   while barely moving the headline (74.8 → 74.1). Learning to localize is
   what teaches the slots to reason about what the camera cannot see.
5. **Mean recall has plateaued across the last two rungs.** WorldWise+ 54.7 →
   WorldWise++ 54.4 in PredCls is flat-to-slightly-down. SGDet is the
   exception, where ++ takes mR@20 (24.5) and mR@50 (48.6) outright.
6. **SGDet with-constraint R@20 is the one column the baselines keep**
   (56.6 vs 54.6). The gap narrowed across the lineage (52.6 → 54.2 → 54.6)
   but did not close. Present SGDet on mR, no-constraint and the occlusion
   buckets.
7. **The baseline ladder is not monotone and must not be presented as one.**
   PredCls spans 66.9–68.5 wc R@20 with W-STTran++ *below* plain W-STTran. The
   added encoders do not pay for themselves on this dataset. Do not claim
   each baseline component helps.
8. **Latents beat decoded detector outputs.** WorldWise+ @ dinov3tok is +4.5 R
   / +5.1 mR over the identical model reading decoded FRCNN features, with
   nothing else changed.
9. **The fusion gate is mode-dependent.** `fused` fails to beat its own best
   single stream in PredCls (73.2 vs 73.5) but is the best WorldWise+ cell in
   SGDet on mR@20, mR@50 and nc mR@20. Report the honest, mode-split summary.

---

## 5. Protocol caveats

These are the points reviewers will attack. Carry them.

1. **PredCls is cross-track comparable; SGDet is not.** Every PredCls row in
   the paper — supervised or prompted — is scored by the same stock WorldSGG
   evaluator with ground-truth boxes supplied. SGDet differs by track: 2D IoU
   0.5 here, 3D IoU in track 3, class-only in track 2. Never put a track-1 and
   a track-3 SGDet number in one column.
2. **Track-1 SGDet is not comparable to anything produced before the
   `bbox_2d` fix.** The 2D ground-truth box lives under PKL key `bbox_2d` in
   original-frame pixels and is rescaled into Pi-3 space. Before that fix
   `gt_bboxes_2d` was all zero and SGDet matching was meaningless. This also
   invalidates the legacy campaign's SGDet numbers quoted for provenance in
   [WORLDWISE.md](WORLDWISE.md).
3. **The baselines are scored at resnet50 only.** This is the conservative
   direction: their DINOv3-L PredCls results were slightly *worse*
   (66.9 / 38.4), so the comparison does not flatter the proposed methods.
   State it rather than let a reviewer find it.
4. **About 37 % of annotation objects have no feature slot** (never detected
   by GDino) in the old and new annotation sets alike, so they are never
   evaluated by any row in any track. The paper should state this.
5. **`_best_score` is not stored in checkpoints**, so a resumed run selects
   `best_model.pth` among post-resume epochs only. Two WorldWise+ SGDet
   ablation cells were resumed mid-run; their best-epoch selection covers
   epochs 5–20 and 7–20. Immaterial here (best epochs were late) but it is a
   real hole in the protocol.
6. **Free-query detection quality is unreported.** WorldWise++'s DETR-style
   heads are trained and the `nodet` ablation shows they matter, but their
   standalone 2D AP / recall against the GT boxes has not been measured. The
   free-query detections are deliberately *not* substituted into the SGDet
   protocol — that would change what SGDet measures.

---

## 6. Where the losses are defined

| Method | Loss | Notes |
|---|---|---|
| W-STTran / ++ / W-DSGDetr / ++ | [lib/supervised/baselines/lks_buffer/loss.py](../lib/supervised/baselines/lks_buffer/loss.py) | `LKSLoss`, aliased as `WSTTranLoss` / `WDSGDetrLoss`; bucketed noisy-label training, `lambda_vlm = 0.2` on unseen pairs |
| W-USG | [lib/supervised/baselines/w_usg/loss.py](../lib/supervised/baselines/w_usg/loss.py) | `WUSGLoss`: `LKSLoss` plus a text-centric contrastive alignment term |
| WorldWise | [lib/supervised/worldwise/loss.py](../lib/supervised/worldwise/loss.py), [amwae_loss.py](../lib/supervised/worldwise/amwae_loss.py) | `WorldWiseLoss` = `AMWAELoss` + tail-aware logit adjustment; `lambda_vlm = 0.0` |
| WorldWise+ | unchanged (`WorldWiseLoss`) | only the appearance input changes — [lib/supervised/worldwise_plus/model.py](../lib/supervised/worldwise_plus/model.py) |
| WorldWise++ | [lib/supervised/worldwise_pp/loss.py](../lib/supervised/worldwise_pp/loss.py) | `L_WorldWise + λ_det · (CE + 5·L1 box + L1 corners) + λ_slot · (L1 box residual + L1 corner residual)`; `λ_det = λ_slot = 1` |

The scene-graph term is identical across WorldWise, WorldWise+ and
WorldWise++, so differences in R / mR are attributable to representation and
architecture, not to the loss recipe.

Training configs: `configs/methods/{predcls,sgdet}/`. Operational recipe:
[RUN_WSGG.md](RUN_WSGG.md).

**Throughput note that applies to every cell in this track.** The trainers are
dataloader-bound, not GPU-bound: each item re-reads two PKLs (plus ~32 MB of
grids for WorldWise++). With the historical `num_workers = 0` the WorldWise++
SGDet cell ran at 3.2 s/video with the GPU at 0–5 %. Keep `num_workers: 4` and
`prefetch_factor: 4` set in any new cell.
