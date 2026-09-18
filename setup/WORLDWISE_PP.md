# WorldWise++ — an image-grounded entity decoder with joint detection

WorldWise++ is the architectural step of the lineage. WorldWise and
WorldWise+ never see an image: they reason over a fixed set of object slots
whose appearance was pooled by someone else, and object permanence lives in a
separate module (the associative retriever) bolted between the tokenizer and
the inter-object transformer. WorldWise++ keeps the WorldWise scaffold, loss
recipe and evaluation protocol, and replaces the object-token core with a
single **entity decoder** that

1. **sees the image** — cross-attends to the fused DINOv3 / π³ token grid of
   every frame (WorldWise+'s latents, but the *whole* grid, not only the
   pooled ROI);
2. **does permanence and inter-object reasoning in one place** — each decoder
   layer attends over a slot's own frames (temporal), over all entities of the
   frame (spatial), and into the grid (cross), so a masked slot is recovered
   from its own visible frames *and* from the current image;
3. **detects jointly** — free DETR queries are trained with Hungarian matching
   to detect objects (class, 2-D box, 3-D OBB) from the same grid, and the
   world slots refine their own boxes;
4. **reads relations out of attention** — the pair features for the relation
   head are the decoder's own spatial attention weights (EGTR-style), so no
   union-box ROI features are needed anywhere.

| Variant | Appearance input | Object-token core | Detection |
|---|---|---|---|
| WorldWise | decoded FRCNN box-head (1024-d) | scaffold → retriever → inter-object transformer | none (external detector) |
| WorldWise+ | frozen DINOv3 / π³ ROI latents | unchanged | none |
| **WorldWise++** | + full DINOv3 / π³ token grids | **entity decoder** (temporal · spatial · cross) | **joint** (free queries + slot refinement) |

![WorldWise lineage overview](worldwise_variants_overview.svg)

Model: [lib/supervised/worldwise_pp/model.py](../lib/supervised/worldwise_pp/model.py) ·
Loss: [lib/supervised/worldwise_pp/loss.py](../lib/supervised/worldwise_pp/loss.py) ·
Dataset: [lib/supervised/worldwise_pp/dataset.py](../lib/supervised/worldwise_pp/dataset.py) ·
Grid cache: [datasets/preprocess/tokens/build_pp_grid_cache.py](../datasets/preprocess/tokens/build_pp_grid_cache.py) ·
Method key `worldwise_pp` · Base: [WORLDWISE_PLUS.md](WORLDWISE_PLUS.md), [WORLDWISE.md](WORLDWISE.md)

---

## 1. Forward pipeline

```
                 ┌──────────────── WorldWise / WorldWise+ front-end, unchanged ────────────────┐
corners ──► GlobalStructuralEncoder ─┐                                                          │
poses ───► ObjectSpatialEncoder      ├─► ScaffoldTokenizer ──► N world-slot tokens  s_t,n         │
       └─► CameraTemporalEncoder     │     visible slot : gated projector(DINOv3 ROI latent)      │
corners ──► ObjectMotionEncoder      │     unseen / p_mask=0.3 : [MASK]      (EMA recon target)   │
DINOv3 ROI latents (3072-d) ─────────┘                                                          │
                 └──────────────────────────────────────────────────────────────────────────────┘

DINOv3 L24n grid ─┐  PCA-256, π³ lattice          ┌────────────────────────────────────────────┐
                  ├─► TokenGridFusion ──► M_t     │  Entity decoder  × L                        │
π³ G14 grid ──────┘  per-cell gate + 2-D sine PE  │   (a) temporal self-attn   slot n over t   │
                                                  │   (b) spatial  self-attn   all queries of t │  weights kept
Q free queries q_t,q  ────────────────────────────┤   (c) cross-attn           queries → M_t   │
N slot tokens  s_t,n  ────────────────────────────┤   (d) FFN                                   │
                                                  └───────────────┬────────────────────────────┘
                                                                  │
      free queries ───► class (C+1) · 2-D box · 3-D OBB  ──► Hungarian + one-to-many  (detection)
      world slots  ───► + VisibilityEmbedding ──► NodePredictor · reconstruction_proj · box/corner refinement
      slot pairs   ───► [ attention weights p↔o over all layers ; q_p ⊙ q_o ] ──► replaces union features
                        ──► RelationshipPredictor (text ⊕ pair geometry ⊕ rel self-attn)
                        ──► TemporalEdgeAttention ──► attention / spatial / contacting
```

Everything above the first box and to the right of the decoder is WorldWise
code, called with the same tensors; the output dict has the same keys as
`WorldWise.forward` (plus a `det` sub-dict), so `WorldWiseLoss`, the trainer,
`tools/dump_predictions.py` and `tools/reeval_test.py` run unchanged.

---

## 2. Grid inputs — the PCA-256 training cache

The Tier-2 grids ([WORLDWISE_PLUS.md](WORLDWISE_PLUS.md) §2) are 2.6 TB on a
spinning disk that reads at ~140 MB/s; one training epoch over them would be
~2 h of I/O. WorldWise++ therefore trains from a compact cache built once by
`build_pp_grid_cache.py`:

- **one layer per stream** — DINOv3 `L24n` (final, normalised) for semantics,
  π³ `G14` (global-attention layer: geometry with cross-frame context);
- **one lattice** — the DINOv3 grid is cropped to its unpadded region and
  bilinearly resampled onto the π³ grid, exactly as the online fusion would
  have done (both streams live in π³ pixel space, so the cells coincide);
- **frozen PCA 1024 → 256 per stream**, fit on train tokens only (150 videos,
  ~500 k tokens per stream); the retained variance is stored with the basis
  (`pca_dinov3_L24n.npz`, `pca_pi3_G14.npz`) and reported in the appendix;
- only the frames the loader uses, fp16, on NVMe:

```
/data3/rohith/ag/cache/pp_grids/<split>/<video>.npz    (→ NVMe /code)
   frames (T,)  dino (T,Hp,Wp,256) f16  pi3 (T,Hp,Wp,256) f16  grid_hw (Hp,Wp)  target_size (H,W)
```

≈ 32 MB per video, ≈ 290 GB for train + test, a few minutes of I/O per
epoch. The cache is annotation-independent (registered in the manifest): an
annotation revision changes object sets and pairs, never these grids.

`WorldAGGrid(WorldAG)` attaches `grid_dino`, `grid_pi3`, `image_hw` to the
standard per-video batch (rows selected by `frame_names`, in loader order).

---

## 3. Components

**Front-end (unchanged from WorldWise+).** Geometry, camera, ego-motion and
motion encoders; `ScaffoldTokenizer` with the `GatedFusionProjector` over the
DINOv3 Tier-1 tokens (`feature_model: dinov3_tok`, `token_streams: [3072]`);
`[MASK]` for unseen slots and `p_mask_visible = 0.3` artificial masking; EMA
reconstruction target. The slot tokens *are* the world-slot queries.

**TokenGridFusion.** Per stream `LayerNorm → Linear(256 → d_model)`; a
per-cell 2-way softmax gate mixes the two projections; LayerNorm; plus a 2-D
sine positional encoding computed for the video's own grid size (grids vary
with aspect ratio). Output: memory `M_t ∈ R^{Hp·Wp × d}` per frame.

**Queries.** `n_free_queries` learnable (content + position) free queries per
frame, and the N world slots with a positional embedding derived from the
structural token (`Linear(GlobalStructuralEncoder(corners))`). Padded slots are
masked out of every attention and zeroed after every layer, as in WorldWise.

**Entity decoder (× `n_decoder_layers`, pre-norm).**

| step | attends | purpose | replaces |
|---|---|---|---|
| (a) temporal self-attention | slot *n* across the T frames (+ learned frame embedding), slots only | a masked / unseen slot reaches its own visible frames | `AssociativeRetriever` |
| (b) spatial self-attention | all free queries + slots of frame *t*; weights returned per head | inter-object reasoning; **relation by-product** | `InterObjectTransformer` |
| (c) cross-attention | queries → `M_t` | image grounding: appearance beyond the ROI, context, detection | — (new) |
| (d) FFN | | | |

**Heads.**

- *Free queries* → class (C+1 with no-object), 2-D box (cxcywh, sigmoid),
  3-D OBB through the factorised parameterisation of the existing monocular
  detector (`_compute_3d_corners`: dims / yaw sin-cos / depth / centre offset,
  pinhole with f = max(H, W)).
- *World slots* → `+ VisibilityEmbedding` → `NodePredictor` (object classes;
  GT override in predcls exactly as WorldWise), `reconstruction_proj` (MWAE
  reconstruction now passes through the decoder), and a **refinement head**
  predicting residuals on the slot's 2-D box and 3-D corners — the joint
  detection on the persistent world graph.
- *Relations* — for each (person, object) pair the spatial attention weights
  in both directions from every layer (2·L·heads values) and `q_p ⊙ q_o` are
  projected to `d_rel / 4` and fed to `RelationshipPredictor` **in place of
  the union-box features**; text pathway, pair geometry, relation
  self-attention, `TemporalEdgeAttention` and the three heads are WorldWise's.

**Matching.** Per frame, Hungarian assignment of free queries to the frame's
GT objects (valid & visible slots with a GT box: cost = class prob + 5·L1 box
+ L1 corners), plus the Hydra-SGG one-to-many auxiliary assignment (every
unmatched query whose box IoU with a GT exceeds 0.5) at train time — the
standard remedy for DETR's slow convergence on a small dataset.

---

## 4. Loss

```
L = L_WorldWise                                   (unchanged: λ_vlm = 0, τ = 0.5 logit adjustment,
                                                    simulated-unseen + EMA reconstruction, node CE)
  + λ_det  · [ CE(class, no-object × 0.1) + 5·L1(box) + L1(corners) ]      free queries, matched
  + λ_slot · [ L1(box residual) + L1(corner residual) ]                   valid & visible slots
```

`λ_det = λ_slot = 1` in the main cells. The scene-graph part of the objective
is byte-identical to WorldWise and WorldWise+, so differences in R / mR are
attributable to the architecture and the detection objective, not to the
loss recipe.

---

## 5. Evaluation — same protocol, plus detection

Scene graphs are scored exactly as for every other method: the same object
slots and person–object pairs from the feature PKLs, the same
`dump_predictions → reeval_test → bucketed_breakdown` chain on the WorldBBox
test split (1,511 videos, all frames, wc/nc R@K and mR@K, OO / OU /
OU-non-trivial buckets). WorldWise++'s free-query detections are **not**
substituted into the protocol — that would change what sgdet measures — they
are reported separately (2-D AP / recall against the GT boxes) as the joint
detection result.

---

## 6. Cells and compute

| Experiment | Mode | GPU | What it isolates |
|---|---|---|---|
| `worldwise_pp_dinov3_predcls` | predcls | 0 | the headline — WorldWise++ @ DINOv3 |
| `worldwise_pp_dinov3_sgdet` | sgdet | 1 | idem |
| `worldwise_pp_dinov3_nodet_{predcls,sgdet}` | both | 2 | `n_free_queries = 0`, `λ_det = λ_slot = 0`: the decoder without the detection objective |

Config source: `tools/gen_worldwise_variant_configs.py` (from the
`worldwise_plus_dinov3tok_*` cells: 20 epochs, lr 1e-4, AMP, `d_model 256`,
`n_decoder_layers 4`, `n_decoder_heads 8`, `n_free_queries 30`,
`det_one_to_many_iou 0.5`, `save_path /data3/rohith/ag/runs/worldwise_pp`).
Launcher `scripts/remote/run_worldwise_pp_train.sh`; scoring
`scripts/remote/score_worldwise_variants.sh` (CPU, best epoch, dump → reeval
→ buckets) feeding the comparison table below.

---

## 7. Results

*(filled in as the cells finish — the comparison against all 20 reference
checkpoints of [analysis/worldbbox_reference_2026-09-17.md](../analysis/worldbbox_reference_2026-09-17.md),
WorldWise+ and WorldWise++, all-frame, both constraints, visibility buckets.)*

| Method | backbone | predcls wc R@20 / mR@20 | sgdet wc R@20 / mR@20 | sgdet nc R@20 / mR@20 | OU-nt R@20 |
|---|---|---|---|---|---|
| WorldWise (v2e) | dinov3l | 69.0 / 49.6 | 52.6 / 21.5 | — | 28.1 |
| WorldWise+ | DINOv3 latents | scoring | scoring | scoring | scoring |
| WorldWise++ | DINOv3 latents + grids | training | training | training | training |

---

## 8. Ablations the design supports, and open items

Free with the cells above: with / without the detection objective (`nodet`)
· WorldWise++ vs WorldWise+ (grid + decoder) vs WorldWise (representation).
Cheap to add: cross-attention off (decoder without the image → a
WorldWise+-like model with a different core), temporal step off (permanence
from the image only), EGTR readout vs union tokens, `n_free_queries`.

Open:

1. **Detection is trained from frozen features on ~184 k frames** — DETR-style
   heads converge slowly; the one-to-many assignment is there for that reason,
   but 20 epochs may under-train the free queries. Report detection honestly
   even if the scene-graph numbers are the headline.
2. **PCA-256 is a compromise** forced by disk throughput; the retained
   variance is logged and the full-rank grids remain cached for a re-run on
   faster storage.
3. **3-D OBBs of the free queries are camera-frame** (pinhole back-projection
   with an assumed focal length); the world-frame slots are the ones scored.
