# WorldWise Architecture — MWAE-based World Scene Graph Generation (v2e)

Track: [TRACK1_TRAINING.md](TRACK1_TRAINING.md) · index: [README.md](README.md)

WorldWise is the proposed method: the top tier of the method ladder (see
[BASELINES.md](BASELINES.md)). It replaces the baselines' passive LKS buffer
with a fully differentiable **Masked World Auto-Encoder (MWAE)** — occlusion
is treated as *masking*, and occluded objects are recovered by attention over
their visible appearances — plus ego-motion encoding, explicit pair geometry,
and a tuned tail-aware objective.

**The main configuration is the round-2 winner "v2e"** (experiment stem
`worldwise_v2e`): MWAE core + pair geometry + τ=0.5 logit adjustment +
**no noisy VLM supervision** (λ_vlm=0) + 0.3 artificial masking. See
`docs/DECISION_LOG.md` for how it was selected and the "Measured results"
section below for the final campaign numbers.

Model: [lib/supervised/worldwise/worldwise.py](../lib/supervised/worldwise/worldwise.py) ·
Loss: [loss.py](../lib/supervised/worldwise/loss.py) →
[amwae_loss.py](../lib/supervised/worldwise/amwae_loss.py) ·
Config source of truth: `tools/gen_grid_configs.py` (WORLDWISE_EXTRA)

## The lineage: WorldWise → WorldWise+ → WorldWise++

Two variants build on this model without touching its loss recipe or the
evaluation protocol; each changes exactly one layer of the story.

| Variant | Appearance input | Object-token core | Detection | Doc |
|---|---|---|---|---|
| **WorldWise** | decoded FRCNN box-head (1024-d) + fitted OBBs | scaffold → associative retriever → inter-object transformer | none (external detector) | this file |
| **WorldWise+** | frozen DINOv3 / π³ *latent* ROI tokens, gated fusion | unchanged | none | [WORLDWISE_PLUS.md](WORLDWISE_PLUS.md) |
| **WorldWise++** | + the full DINOv3 / π³ token grids | entity decoder: temporal slot attention · spatial self-attention · cross-attention to the grid | joint (free DETR queries + slot refinement) | [WORLDWISE_PP.md](WORLDWISE_PP.md) |

WorldWise+ tests *latents vs decoded outputs* with the architecture held
fixed; WorldWise++ tests *image-grounded, jointly-detecting reasoning* with
the representation held fixed. Overview figure:
[worldwise_variants_overview.svg](worldwise_variants_overview.svg).

## Forward pipeline (single batched pass, B = T frames)

```
corners ──► 1. GlobalStructuralEncoder ─────────────┐
poses ────► 2. ObjectSpatialEncoder (per-object)    │
        └─► 3. CameraTemporalEncoder (ego-motion)   ├─► 5. ScaffoldTokenizer
corners ──► 4. ObjectMotionEncoder (vel/accel)      │      (mask + fuse)
ROI feats ──────────────────────────────────────────┘          │
                                                               ▼
                                            6. AssociativeRetriever (per-object
                                               cross-attn over visible frames)
                                                               ▼
                             7. + VisibilityEmbedding ─► 8. InterObjectTransformer
                                                               ▼
                      9. NodePredictor        10. reconstruction_proj (MWAE)
                                                               ▼
                 11. RelationshipPredictor (pair tokens ⊕ PAIR GEOMETRY)
                                                               ▼
                 12. TemporalEdgeAttention ─► 13. att / spatial / contacting
```

## The MWAE core

- **ScaffoldTokenizer** ([scaffold_tokenizer.py](../lib/supervised/worldwise/scaffold_tokenizer.py)) —
  top-down tokenization: every object gets a token at every frame. Visible
  objects carry their projected ROI features; unseen objects a learnable
  **[MASK]** embedding; geometry/camera/motion/ego features are fused in
  either case. During training, `p_mask_visible = 0.3` of *visible* objects
  are artificially masked — this is a load-bearing choice: the masking acts
  as a tail-class augmentation as well as occlusion training (reducing it to
  0.1 cost ~5 mR in round 2). Reconstruction targets come from an **EMA copy
  of the visual projector** (`use_ema_recon_target: true`).
- **AssociativeRetriever** ([associative_retriever.py](../lib/supervised/worldwise/associative_retriever.py)) —
  per-object bidirectional cross-attention: each slot's masked tokens query
  that slot's *visible* tokens across all T frames, with view-aware Q/K
  biases from camera features. The differentiable replacement for the
  baselines' zero-order-hold copy.
- **Visibility embedding** — flags observed vs recovered tokens downstream.
- **Ego-motion (`CameraTemporalEncoder`)** — relative camera poses (6D rot +
  translation) per step, self-attended over the sequence, broadcast into
  every object token.
- **Pair geometry (I-1)** — an explicit relative-3D-geometry vector (unit
  direction, log center distance, both log-volumes, ratio, log min
  corner-gap) appended to every relation token (`use_pair_geometry: true`).

## Loss — the v2e recipe

`WorldWiseLoss` = `AMWAELoss` + tail-aware logit adjustment, with two
decisive settings found in round 2:

1. **Scene-graph loss on visible pairs only** (`lambda_vlm: 0.0`). The noisy
   VLM pseudo-labels on unseen pairs are **not used** — round 2 showed they
   were actively mis-teaching (removing them: +4 R, +9 mR, and **+9.5
   occluded-pair recall**). Unseen pairs receive *no direct edge
   supervision*; the recovery pathway is trained entirely by:
2. **Simulated-unseen + reconstruction objectives** — clean-GT supervision on
   the artificially masked (visible) pairs, plus MSE reconstruction of their
   EMA-target features. Simulated occlusion with clean labels generalizes to
   real occlusion better than direct-but-noisy supervision.
3. **Tail-aware logit adjustment at τ=0.5** (Menon et al., 2021): per-class
   log-prior offsets added to relation logits at train time only (priors from
   `tools/compute_predicate_priors.py`, mode-specific files). τ traces a
   clean R↔mR Pareto front: 0.5 is the balanced point; **0.75 is the
   published mR-max operating point** (`v2f`); 1.0 is unstable (round-1
   collapse on occluded pairs at resnet50).

## Component-wise ablation set (Table C)

Every ablation changes exactly one thing relative to the main config; all run
@ dinov3l, both modes (`--stages abl`). Three keep historical tier names so
already-trained cells are reused:

| Tier | Changes | Ablates |
|---|---|---|
| `v2a` | λ_vlm 0 → 0.2 | value of REMOVING noisy VLM supervision |
| `v2f` | τ 0.5 → 0.75 | the R↔mR operating point |
| `abl_notau` | logit adjustment off | the tail-aware objective |
| `abl_nomask` | p_mask 0.3 → 0 (recon/sim vanish) | the MWAE self-supervision |
| `abl_noema` | EMA target → live projector | reconstruction-target stability |
| `v2g` | pair geometry off | I-1's contribution |
| `abl_nospatial` | ObjectSpatialEncoder off | camera-frame features |
| `abl_noego` | CameraTemporalEncoder off | ego-motion |
| `abl_nomotion` | ObjectMotionEncoder off | object dynamics |
| `abl_notempedge` | TemporalEdgeAttention off | cross-frame edge reasoning |

## Retired plugins (round-2 gate, docs/DECISION_LOG.md)

Historical context — these were evaluated and failed; their flags remain in
the code (all `false` in configs) and their modules in the tree:
soft text embedding (I-2, unobservable in predcls / hurt sgdet), geometric
attention bias (I-3, −4 R), confidence-weighted VLM (I-5, no data),
predicate prototypes (I-6, catastrophic collapse), cross-object retrieval
(I-7, hurt its own occpair target), energy refinement (I-8, −5 R). The
campaign's conclusion: **the wins came from the loss side, not from adding
architecture.**

## What separates WorldWise (v2e) from W-DSGDetr++ (strongest baseline)

1. MWAE mask-and-retrieve memory (differentiable) vs zero-order-hold copy.
2. Ego-motion encoding.
3. Simulated-unseen + reconstruction objectives *instead of* noisy VLM
   supervision on unseen pairs (baselines keep λ_vlm=0.2).
4. Tail-aware logit adjustment (τ=0.5).
5. Explicit pair geometry in relation tokens.

## Measured results

### WorldBBox test set (current, authoritative)

1,511 videos, **all frames**, best epoch, identical
`dump_predictions -> reeval_test -> bucketed_breakdown` chain for every row.
Full table across all methods and backbones:
[analysis/worldbbox_lineage_2026-09-19.md](../analysis/worldbbox_lineage_2026-09-19.md).

| Mode | Backbone | wc R@20 | wc mR@20 | nc R@20 | nc mR@20 | OO R@20 | OU-nt R@20 | OU-nt mR@20 |
|---|---|---|---|---|---|---|---|---|
| predcls | resnet50 | 68.0 | 49.1 | 92.5 | 81.8 | 90.6 | 73.5 | 40.9 |
| predcls | dinov2b | 68.2 | 45.4 | 92.5 | 79.4 | 90.6 | 74.7 | 38.7 |
| predcls | dinov2l | 68.8 | 47.8 | 92.6 | 81.6 | 90.6 | 74.2 | 42.6 |
| predcls | **dinov3l** | **69.0** | **49.6** | 92.6 | **82.4** | 90.9 | 72.9 | 41.1 |
| sgdet | resnet50 | 53.6 | 20.4 | 58.3 | 36.1 | 55.5 | 25.4 | 20.6 |
| sgdet | **dinov2b** | **54.1** | **23.4** | **58.8** | 37.9 | 53.0 | 29.6 | 21.1 |
| sgdet | dinov2l | 53.5 | 22.3 | 57.8 | 37.2 | 51.7 | **30.4** | 20.2 |
| sgdet | dinov3l | 52.6 | 21.5 | 57.1 | **37.5** | 53.3 | 28.1 | **21.3** |

**PredCls — WorldWise leads the baseline ladder, and the margin is almost
entirely mean-recall.** At DINOv3-L, wc R@20 69.0 against the best baseline's
68.5 (W-DSGDetr++) is within noise; wc mR@20 **49.6 vs 38.9** is not, and
the occluded non-trivial bucket (72.9 R / 41.1 mR vs 55.3 / 26.1) is the
decisive gap. State the claim as tail-recall and occlusion, not headline R.

**SGDet — a different trade, read the right columns.** WorldWise does *not*
win with-constraint R@20 (52.6-54.1 vs baselines' ~56.5) but wins
with-constraint mR@20 (21.5-23.4 vs ~19), no-constraint mR@20 (37.5 vs ~28)
and dominates the occluded buckets (OU-nt 28.1 R / 21.3 mR vs 12-16 / 6-8,
i.e. 2-3x). Backbone order **inverts** here: DINOv2-B is the strongest
WorldWise backbone on wc metrics, decreasing with larger backbones - the
frozen ViT does not help once the detector drives localization.

**Where WorldWise sits in the lineage** (both @ DINOv3, predcls wc R@20 /
mR@20): WorldWise 69.0 / 49.6 -> [WorldWise+](WORLDWISE_PLUS.md) 73.5 / 54.7
-> [WorldWise++](WORLDWISE_PP.md) 74.8 / 54.4. The representation swap buys
recall *and* tail recall; the architecture change buys recall and the
occluded buckets (OU-nt mR 41.1 -> 45.2 -> 48.1) while mean recall plateaus.

### Legacy campaign (previous test split - kept for provenance)

Superseded by the table above; the numbers below were measured on the older
1,734-video split and before the `gt_bboxes_2d` fix, so **sgdet values in
particular are not comparable** (that fix changed sgdet from predcls-style
matching to true 2D-IoU matching). PredCls @ DINOv3-L read 68.9 / 49.7 with
the best baseline at 66.9 / 38.4; sgdet read 54.7-55.8 wc R@20 against
baselines' ~59.8. Full tables: `results_tables/*.tex`.

**Ablation takeaways** (@ DINOv3-L, vs the full config):

| Component removed / changed | PredCls effect | Verdict |
|---|---|---|
| + noisy VLM supervision back (λ_vlm 0→0.2) | R −3.8, mR −8.7 | λ_vlm=0 strongly confirmed (predcls); mixed on sgdet (slightly helps wc-R) |
| τ 0.5 → 0.75 | R −1.7, **mR +2.0** | Pareto point — the published mR-max operating config |
| − logit adjustment | **R +1.0**, mR −8.1 | The R↔mR knob; drop it only if R is the sole target |
| − artificial masking | R ≈0, mR −3.1 | Masking is a real tail-class augmentation — keep 0.3 |
| − pair geometry (I-1) | R ≈0, mR −3.5 | I-1 justified — it lifts tail recall on the v2e base |
| − ego-motion encoder | R ≈0, mR −3.0 | Contributes to tail recall |
| − object motion encoder | R +0.4, mR −1.7 | Marginal — smallest contribution of the encoders |
| − temporal edge attention | R −1.5, mR −1.3 | Helps both mildly |
| − ObjectSpatialEncoder | *no result* | **Cell missing — needs a rerun (see RUN_WSGG.md)** |

Net reading: the tail-recall (mR) gains come from the loss recipe
(λ_vlm=0, logit adjustment, masking) plus pair geometry and ego-motion;
object-motion is the one encoder that could be dropped with little cost.
