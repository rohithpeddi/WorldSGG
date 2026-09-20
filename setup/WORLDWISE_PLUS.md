# WorldWise+ — WorldWise over frozen foundation-model latents

WorldWise+ is WorldWise ([WORLDWISE.md](WORLDWISE.md)) with **one seam
re-plumbed**: the appearance input. Instead of the *decoded* output of the
perception stack (the 1024-d FRCNN `box_head` vector of the monocular-3D
detector), the model reads the *latent* tokens of two frozen foundation
models — semantics from **DINOv3-L**, geometry from **π³** — ROI-pooled onto
the very same object and union boxes. Nothing else changes, so

> **WorldWise+ vs WorldWise is a clean A/B on the representation, not on the
> architecture.** The claim it tests: *latents beat decoded outputs.*

| Variant | Appearance input | Object-token core | Doc |
|---|---|---|---|
| WorldWise | decoded FRCNN box-head (1024-d) | scaffold → retriever → inter-object transformer | [WORLDWISE.md](WORLDWISE.md) |
| **WorldWise+** | frozen DINOv3 / π³ ROI latents, gated fusion | **unchanged** | this file |
| WorldWise++ | + the full DINOv3 / π³ token grids | entity decoder, joint detection | [WORLDWISE_PP.md](WORLDWISE_PP.md) |

![WorldWise lineage overview](worldwise_variants_overview.svg)

Model: [lib/supervised/worldwise_plus/model.py](../lib/supervised/worldwise_plus/model.py)
(`WorldWisePlus(WorldWise)` + `GatedFusionProjector`; the former
`lib/supervised/worldformer/c1_tokenswap` path is a compatibility shim) ·
Caches: [datasets/preprocess/tokens/](../datasets/preprocess/tokens/) ·
Method key `worldwise_plus` (alias `worldformer_c1`).

---

## 1. Why the seam is where it is

WorldWise reads everything through pre-extracted per-video PKLs, and all
appearance enters through exactly two projectors:

```
ScaffoldTokenizer.visual_projector : Linear(d_detector_roi -> d_visual) -> ReLU -> LayerNorm
RelationshipPredictor.union_proj   : Linear(d_union_roi    -> d_rel/4)  -> ReLU -> LayerNorm
```

In the reference configuration those consume the **1024-d FRCNN box-head
output** of the monocular-3D detector ([MONOCULAR_3D_DETECTOR.md](MONOCULAR_3D_DETECTOR.md)),
i.e. a *decoded* representation: DINOv3 → FPN → RPN → ROI head → 1024-d. The
geometry the model sees (`corners_final`) is likewise decoded — π³ point maps
that the annotation pipeline already fitted into oriented boxes.

WorldWise+ changes only what flows through those two projectors. The geometry
scaffold, MWAE masking (`p_mask_visible = 0.3`), EMA reconstruction target,
associative retriever, visibility embedding, inter-object transformer,
relation head with pair geometry, temporal edge attention, τ = 0.5 logit
adjustment and λ_vlm = 0 are inherited unchanged from WorldWise v2e.

---

## 2. Token caches (annotation-independent)

Both streams are cached once per video over *every* frame of
`frames_annotated/<video>.mp4` — a superset of what any feature or annotation
PKL uses — so the caches survive an annotation revision untouched.

```
/data3/rohith/ag/cache/tokens/<stream>/<split>/<video>.npz
                              + done_shard<k>of<n>.txt, status_shard<k>of<n>.json
```

| Stream | Model | Grid (Tier-2) | Hooked layers (Tier-1) |
|---|---|---|---|
| `dinov3l` | `facebook/dinov3-vitl16-pretrain-lvd1689m` (frozen) | `L16`, `L24n` — (T, 24, 42, 1024) fp16 on the /16-padded 672×384 image | `L16`, `L20`, `L24n` |
| `pi3` | `yyfz233/Pi3` (frozen) | `F14`, `G14` — (T, 27, 48, 1024) fp16 on the 672×378 π³ image, 5 register tokens stripped | frame `F4,F12..F16`, global `G13..G15` |

**π³ layer convention** (arXiv 2511.22686 §B.1, 0-indexed per attention type):
the decoder alternates frame (even block) and global (odd block) attention, so
frame layer *j* → decoder block *2j* and global layer *j* → block *2j+1*. The
resolved map is stored in every npz as `layer_blocks`.

**Image geometry is shared by construction.** Both streams resize with the
same `compute_target_size` (pixel limit 255 000, patch 14) used by
`extract_roi_features_base` and `pi3.utils.basic.load_images_as_tensor`, so a
π³ cell and a DINOv3 cell address the same pixels and ROI boxes pool
identically from either.

**Two tiers, deliberately:**

- **Tier-1** — ROI-pooled (`roi_align` 7×7, mean) per object *and* per
  person–object union box, for the boxes of the **existing feature PKLs**.
  Those boxes come from raw AG GT + GDino fill (predcls) or the DINOv3
  detector (sgdet) — neither depends on the world4d annotation revision.
  Kilobytes per object; this is what WorldWise+ consumes.
- **Tier-2** — the full patch grids for two layers per stream. Gigabytes;
  this is what WorldWise++ consumes (through a PCA-compressed training
  cache, see [WORLDWISE_PP.md](WORLDWISE_PP.md) §2), and what a future
  annotation revision with *different object boxes* would re-pool from.

**π³ frame indexing — deliberate off-by-one.** The existing π³ reconstructions
(`pi3_dynamic/<video>_10/predictions.npz`, the geometry the annotations were
built from) were produced by a loader that indexed the sorted frame list by
the sampled frame id, so clip index *k* corresponds to file `{id+1:06d}.png`.
`pi3_tokens.py` **reproduces that mapping** (verified MAE 0.0 against the
stored npz images, camera poses within 0.04) so cached latents stay in
register with the annotations. Both id lists are stored in each npz. Do not
"fix" this without regenerating the reconstructions.

Realised sizes: DINOv3 237 GB (test) + 852 GB (train); π³ 318 GB (test) +
1.2 TB (train).

---

## 3. Derived streams — `derive_roi_features.py`

Tier-1 tokens are assembled into drop-in replacements for the feature PKLs,
written in the **exact same schema** (`roi_features`, `union_features`,
`bboxes_xyxy`, `labels`, `label_ids`, `sources`, `pair_indices`,
`target_size`, and for sgdet `boxes_3d` / `assigned_labels` copied from the
reference PKLs), fp16:

| Stream (`feature_model`) | Contents | dim |
|---|---|---|
| `dinov3_tok` | concat(`L16`, `L20`, `L24n`) | 3072 |
| `pi3_tok` | concat(`F4`, `F14`, `G14`) | 3072 |
| `fused` | both, in that order | 6144 |

```
/data3/rohith/ag/cache/roi_derived/<mode>/<stream>/{train,test_worldbbox}/<video>.pkl
  symlinked to /data/rohith/ag/features/roi_features/<mode>/<stream>/{train,test,test_worldbbox}
```

so `WorldAG(feature_model="dinov3_tok")` picks them up with no loader change.
This is also the **fast path for future annotation revisions**: the token
grids never move, only this derivation re-runs (CPU, ~1 h for both splits).

---

## 4. The model — `WorldWisePlus`

`WorldWisePlus` subclasses `WorldWise` and replaces the two projectors with a
`GatedFusionProjector` over K concatenated streams (`token_streams` config
key, which must sum to `d_detector_roi == d_union_roi`).

**Single stream** (`dinov3_tok`, `pi3_tok`) reduces *exactly* to the base form,
so the only difference from WorldWise is the input vector:

```
h = LayerNorm(ReLU(Linear_{3072 -> d}(x)))
```

**Two streams** (`fused`, `fusion: gated`) — a per-dimension softmax gate over
streams, so the model chooses geometry vs semantics per feature dimension
rather than committing to a fixed mixture:

```
h_k = Linear_k(x_k)                        k = 1..K,  h_k in R^d
g   = softmax_k( W [h_1 ; … ; h_K] )       g in R^{K x d}, softmax over K
h   = LayerNorm(ReLU( sum_k g_k * h_k ))
```

The mean per-stream gate of the last forward is kept in `last_gate_mean` and
exposed by `gate_summary()` — a free interpretability read on *how much of the
representation is geometric vs semantic*, per mode. `fusion: concat` (one
`Linear` over the whole 6144-d vector) is the ablation control for the gate.

The EMA reconstruction target projector is deep-copied from the new projector
(so MWAE reconstruction targets live in the same space) and frozen.

**Cells** — `configs/methods/{predcls,sgdet}/worldwise_plus_{dinov3tok,pi3tok,fused}_*.yaml`
(the trained runs carry their original `worldformer_c1_*` experiment names
and are exposed under the new names by symlinks): 3 streams × 2 modes, 20
epochs, lr 1e-4, otherwise identical to `worldwise_*_dinov3l.yaml`;
`train_annot_dir: world4d_rel_annotations`,
`test_annot_dir: world4d_rel_annotations_worldbbox`.

**The headline cell is `dinov3tok`** — WorldWise+ with the DINOv3 backbone,
the direct counterpart of WorldWise@dinov3l. `pi3tok` and `fused` are the
stream ablations.

---

## 5. Evaluation

Identical to WorldWise: the same object slots and person–object pairs from the
feature PKLs, the same `tools/dump_predictions.py → tools/reeval_test.py →
tools/bucketed_breakdown.py` chain on the WorldBBox test split (1,511 videos,
all frames, with/no constraint R@K and mR@K, visibility buckets OO / OU /
OU-non-trivial). Numbers land next to the 20 baseline/WorldWise checkpoints
of [analysis/worldbbox_reference_2026-09-17.md](../analysis/worldbbox_reference_2026-09-17.md)
via `scripts/remote/score_worldwise_variants.sh`.

---

## 6. Results

**All-frame, best epoch, WorldBBox test (1,511 videos)** — same protocol and
same scripts as the 20-checkpoint reference table, so these are directly
comparable to it. Reference = WorldWise v2e @ dinov3l: the identical model
reading the decoded FRCNN features.

**predcls — WorldWise+ wins every column.**

| Model | wc R@10 | wc R@20 | wc mR@10 | wc mR@20 | wc mR@50 | nc R@20 | nc mR@20 |
|---|---|---|---|---|---|---|---|
| WorldWise @ dinov3l | 63.2 | 69.0 | 40.1 | 49.6 | 49.7 | 92.6 | 82.4 |
| **WorldWise+ @ DINOv3** | **67.6** | **73.5** | **44.3** | **54.7** | **54.9** | **94.3** | **85.6** |
| Δ | +4.4 | **+4.5** | +4.2 | **+5.1** | +5.2 | +1.7 | +3.2 |

**sgdet — wins overall, and the tail gains are the large ones.**

| Model | wc R@20 | wc R@50 | wc mR@20 | wc mR@50 | nc R@20 | nc mR@20 |
|---|---|---|---|---|---|---|
| WorldWise @ dinov3l | 52.6 | 68.6 | 21.5 | 38.8 | 57.1 | 37.5 |
| **WorldWise+ @ DINOv3** | **54.2** | **70.5** | **23.1** | **45.5** | **58.6** | **43.3** |
| Δ | +1.6 | +1.9 | +1.6 | **+6.7** | +1.5 | **+5.8** |

**Visibility buckets (no-constraint, K=20)** — and here is the one place the
swap does *not* win:

| Model | mode | OO R | OU R | OU-nt R | OU-nt mR |
|---|---|---|---|---|---|
| WorldWise | predcls | 90.9 | 80.5 | 72.9 | 41.1 |
| **WorldWise+** | predcls | **93.4** | **81.4** | **76.0** | **45.2** |
| WorldWise | sgdet | 53.3 | **25.6** | **28.1** | **21.3** |
| **WorldWise+** | sgdet | **54.9** | 24.7 | 26.6 | 20.5 |

In predcls the latents help most exactly where the method's claim lives — the
occluded, non-trivial pairs (+3.1 R, +4.1 mR). **In sgdet they do not**: the
observed bucket improves (+1.6) while the unobserved buckets lose 0.8–1.5.
This should be reported, not hidden. The plausible reading is that richer
appearance latents sharpen recognition of what is *visible*, while unobserved
pairs in sgdet hinge on detector-supplied boxes the token swap never touches —
which is precisely the gap WorldWise++ is built to close, since its slots
reach the image directly.

Trainer-time last-frame metrics for the in-progress stream ablations (not
comparable to the tables above): `pi3tok` 71.2 / 53.7 predcls @ epoch 7,
`fused` 72.2 / 52.5 predcls @ epoch 5 — both early (`dinov3tok` itself read
68.2 / 36.3 at epoch 1 and finished at 73.5 / 54.7 all-frame).

The full cross-method table is built with
`tools/aggregate_rescore.py --root <rescore> <plus> <pp>`.

**Ablations the design supports** (mostly free, since the cells exist):
stream identity (`dinov3tok` vs `pi3tok` vs `fused`) · fusion mechanism
(gated vs concat) · gate statistics per mode (`gate_summary()`) · layer
selection within a stream (re-derive only) · WorldWise+ vs WorldWise@dinov3l
vs the resnet50/dinov2 ladder · with/without MWAE on the token substrate.

---

## 7. Files, and what is still open

| Component | Path |
|---|---|
| Cache helpers / video + frame lists / Tier-1 pooling | `datasets/preprocess/tokens/token_common.py` |
| DINOv3 cacher | `datasets/preprocess/tokens/dinov3_tokens.py` |
| π³ cacher (hooks, register-token strip, off-by-one) | `datasets/preprocess/tokens/pi3_tokens.py` |
| Derived feature PKLs + symlinks | `datasets/preprocess/tokens/derive_roi_features.py` |
| Model | `lib/supervised/worldwise_plus/model.py` (shim: `lib/supervised/worldformer/c1_tokenswap`) |
| Trainer | `train_wsgg_methods.py::TrainWorldWisePlus` (`worldwise_plus`, alias `worldformer_c1`) |
| Configs | `tools/gen_worldwise_variant_configs.py` → `configs/methods/*/worldwise_plus_*.yaml` |
| Training launcher (multi-GPU, one job per cell) | `tools/run_configs_multigpu.py`, `scripts/remote/run_worldformer_c1_train.sh` |
| Scoring chain (dump → reeval → buckets → tables) | `scripts/remote/score_worldwise_variants.sh` |
| Manifest registration | `tools/register_token_cache.py`, `tools/cache_manifest.py` |
| Tests | `tests/test_worldformer_c1_{smoke,realdata}.py` |

**Open items**

1. **Tier-2 re-pooling path is not implemented.** `derive_roi_features.py`
   pools Tier-1 vectors that were cached against the *existing* feature-PKL
   boxes. If the annotation audit changes object sets, new boxes need pooling
   from the stored grids — only `L16/L24n` and `F14/G14` grids exist, so a
   re-pool is restricted to those layers unless the cachers re-run.
2. **π³ caching is GPU-bound** (~4.5 clip-frames/s): the pip package ships
   without the CUDA `curope` extension, so RoPE falls back to PyTorch.
3. **`fused` must beat its own single-stream cells, not just the reference** —
   if it does not beat `max(dinov3tok, pi3tok)`, the gate is not earning its
   parameters and `concat` should be reported instead.
