# Baseline Architectures — W-STTran / W-STTran++ / W-DSGDetr / W-DSGDetr++ / W-USG

Track: [TRACK1_TRAINING.md](TRACK1_TRAINING.md) · index: [README.md](README.md)

The four ladder baselines form a **strict nested capability ladder**: each tier is an
architectural superset of the one below, realised in separate model files (not
flags), so every performance delta is attributable to exactly one added
component. All baselines run at the **resnet50** backbone only (the common
backbone for the method-comparison table).

```
W-STTran      = GSE + LKS buffer + inter-object transformer + temporal-edge attention
W-STTran++    = + ObjectSpatialEncoder          (camera-frame position)
W-DSGDetr     = + TemporalObjectEncoder         (per-slot temporal self-attention)
W-DSGDetr++   = + ObjectMotionEncoder           (world-frame velocity/acceleration)
WorldWise(v2e)= + ego-motion + MWAE + pair geometry + tuned loss  → see WORLDWISE.md

W-USG         = external method (USG-Par relation stack) on the shared substrate
                — sits BESIDE the ladder, neither superset nor subset  → §"W-USG"
```

**The baselines are frozen** — they are the unchanged controls of the final
campaign; every improvement round happened on WorldWise only. Their training
loss keeps the original noisy-label recipe (λ_vlm = 0.2 on unseen pairs),
which WorldWise-v2e deliberately abandons — that asymmetry is a *finding*
(the noisy supervision hurts; see `docs/DECISION_LOG.md`), not an oversight.

**Where the baselines remain competitive.** On PredCls they sit clearly below
the WorldWise family on every metric. On **SGDet they win with-constraint
R@K** — the LKS zero-order-hold memory is a strong, low-variance prior when
the detector already drives localization. The WorldWise family's SGDet
advantage is on mean-recall, the no-constraint protocol and the occluded
buckets, not wc-R. Report SGDet with all column groups so this trade is
visible. Final numbers below.

## Measured results — WorldBBox test set

Full table (all methods, all backbones, both modes, visibility buckets):
[analysis/worldbbox_lineage_2026-09-19.md](../analysis/worldbbox_lineage_2026-09-19.md).
1,511 videos, **all frames**, best epoch, identical
`dump_predictions -> reeval_test -> bucketed_breakdown` chain for every row.
Values in %.

**PredCls**

| Method | wc R@20 | wc mR@20 | nc R@20 | nc mR@20 | OU-nt R@20 | OU-nt mR@20 |
|---|---|---|---|---|---|---|
| W-STTran | 68.2 | 37.6 | 92.6 | 70.0 | 56.9 | 30.7 |
| W-STTran++ | 66.9 | 34.0 | 92.4 | 63.6 | 67.0 | 29.1 |
| W-DSGDetr | 67.5 | 33.4 | 92.2 | 65.6 | 62.7 | 27.0 |
| W-DSGDetr++ | **68.5** | **38.9** | **92.7** | **71.0** | 55.3 | 26.1 |
| W-USG | 67.3 | 33.4 | 92.6 | 64.0 | 67.5 | 29.1 |
| *WorldWise @ dinov3l* | *69.0* | *49.6* | *92.6* | *82.4* | *72.9* | *41.1* |
| *WorldWise++ (best)* | *74.8* | *54.4* | *94.6* | *86.2* | *76.6* | *48.1* |

**SGDet**

| Method | wc R@20 | wc mR@20 | wc mR@50 | nc R@20 | nc mR@20 | OU-nt R@20 | OU-nt mR@20 |
|---|---|---|---|---|---|---|---|
| W-STTran | 56.3 | 19.2 | 33.4 | 63.1 | 27.7 | 14.6 | 7.1 |
| W-STTran++ | **56.6** | 17.6 | 30.3 | 62.9 | 25.4 | 15.6 | 6.6 |
| W-DSGDetr | **56.6** | 19.2 | 33.5 | 63.5 | 28.0 | 12.3 | 6.0 |
| W-DSGDetr++ | 56.4 | **20.3** | **35.4** | **63.8** | **31.2** | 14.3 | 7.7 |
| W-USG | 56.3 | 19.1 | 34.1 | 62.4 | 25.9 | 14.5 | 6.7 |
| *WorldWise @ dinov3l* | *52.6* | *21.5* | *38.8* | *57.1* | *37.5* | *28.1* | *21.3* |
| *WorldWise++ (best)* | *54.6* | *24.5* | *48.6* | *59.4* | *44.1* | *27.8* | *22.3* |

Three things to read off this:

1. **The four tiers are within noise of each other.** PredCls spans 66.9-68.5
   wc R@20 and the ladder is *not* monotone (W-STTran++ is the weakest, below
   plain W-STTran). The added encoders do not pay for themselves on this
   dataset; only W-DSGDetr++ edges ahead on mean-recall. Do not present the
   baseline ladder as evidence that each component helps.
2. **Mean recall is where the family separates**: every baseline sits at
   33-39 mR@20 in PredCls, versus 49.6-54.7 for WorldWise and its variants.
   That gap comes from the loss recipe, not from architecture.
3. **The occluded buckets are the real gap.** SGDet OU-non-trivial: baselines
   reach 12-16 R@20 and 6-8 mR@20; the WorldWise family reaches 27-28 and
   21-22, i.e. 2-3x. This is the capability the baselines structurally lack,
   and the reason the LKS buffer is the thing WorldWise replaces.

**Backbone caveat:** the baselines are scored at **resnet50** only. This is
the conservative direction - in the earlier campaign their best DINOv3-L
PredCls result (66.9 / 38.4) was slightly *worse* than these resnet50
numbers, so the comparison does not flatter the proposed methods.

## Task & I/O contract

Given a video of detected objects with 3D oriented bounding boxes (OBBs) in a
world frame, predict per frame **(a)** object classes and **(b)** the
person↔object relationships, decomposed into three heads: **attention**
(3 classes, softmax), **spatial** (6, sigmoid multi-label), **contacting**
(17, sigmoid multi-label). Every model processes a full video in one batched
forward pass where the batch dimension *is* time (`B = T` frames), over padded
tensors:

| Input | Shape | Meaning |
|---|---|---|
| `visual_features_seq` | (T, N, 1024) | detector ROI features per world slot |
| `corners_seq` | (T, N, 8, 3) | world-frame OBB corners |
| `valid_mask_seq` / `visibility_mask_seq` | (T, N) | real slot / in-camera-FOV |
| `person_idx_seq`, `object_idx_seq`, `pair_valid` | (T, K) | person–object pairs |
| `camera_pose_seq` | (T, 4, 4) | camera-to-world extrinsics |
| `union_features_seq` | (T, K, 1024) | union-ROI features |
| `node_labels_seq` | (T, N) | GT classes — predcls only (task input) |

Objects live in **persistent world slots**: the same slot index refers to the
same physical object across all frames, so temporal identity is free — no
Hungarian matching / tracking is needed anywhere in the ladder.

## Shared pipeline (all four tiers)

Components come from [lib/supervised/components.py](../lib/supervised/components.py);
the memory pieces from [lib/supervised/baselines/lks_buffer/](../lib/supervised/baselines/lks_buffer/).

1. **`vectorized_lks_buffer`** (non-differentiable, `@torch.no_grad`) — a
   bidirectional zero-order-hold memory. For every unseen `(t, n)` it copies
   the raw ROI features from the *nearest visible frame* in either temporal
   direction (forward + reverse `cummax`), returning buffered features and a
   per-slot **staleness** counter (frames since last seen; fixed sentinel 1000
   for never-seen "fog-of-war" slots, which get zero features).
2. **`GlobalStructuralEncoder` (GSE)** — flattens each OBB's 8 centered corners
   (24-d, translation-invariant local shape) ⊕ absolute 3D center (3-d) → MLP →
   per-object structural token (d_struct).
3. **`LKSTokenizer`** — fuses [geometry ⊕ buffered visual (raw 1024-d) ⊕
   camera features (tier ≥ 2) ⊕ log-staleness] → d_model token per slot.
4. **`SpatialGNN`** (inter-object transformer) — transformer encoder over the N
   slots per frame, with a 3D spatial positional encoding (pairwise distance /
   direction / log-volume-ratio features, neighbor-averaged per object).
5. **`NodePredictor`** — MLP → object-class logits.
6. **`RelationshipPredictor.batched_form_and_attend`** — pair token =
   [person token ⊕ object token ⊕ projected union-ROI ⊕ CLIP text embeddings
   of both classes] → self-attention over the K pairs. In **predcls** the text
   pathway uses the GT labels (task inputs, matching the original
   STTran/DSGDetr protocol); otherwise argmax of the node logits.
7. **`TemporalEdgeAttention`** — groups pair tokens by (person, object) pair
   identity and self-attends each pair's sequence across all T frames
   (vectorized scatter/gather, learnable temporal PE).
8. **`RelationshipPredictor.batched_predict`** — three MLP heads →
   attention/spatial/contacting logits + distributions.

## Per-tier additions

| Tier | Added module | What it contributes |
|---|---|---|
| **W-STTran** ([w_sttran.py](../lib/supervised/baselines/w_sttran/w_sttran.py)) | — (base) | LKS memory + spatial transformer + temporal edge attention. `LKSTokenizer` gets `d_camera=0` — no dead camera parameters. |
| **W-STTran++** ([w_sttran_pp.py](../lib/supervised/baselines/w_sttran/w_sttran_pp.py)) | `CameraPoseEncoder` (as ObjectSpatialEncoder) | Per-object camera-relative features: log-distance to camera, view alignment, azimuth sin/cos — how each object sits in the *current* view. |
| **W-DSGDetr** ([w_dsgdetr.py](../lib/supervised/baselines/w_dsgdetr/w_dsgdetr.py)) | `TemporalObjectEncoder` ([object_encoder.py](../lib/supervised/baselines/w_dsgdetr/object_encoder.py)) | Per-slot temporal self-attention over each object's T-frame sequence (learnable temporal PE) — the world-slot analogue of DSGDetr's tracking-based object encoding, applied *before* the inter-object transformer. |
| **W-DSGDetr++** ([w_dsgdetr_pp.py](../lib/supervised/baselines/w_dsgdetr/w_dsgdetr_pp.py)) | `MotionFeatureEncoder` | World-frame finite-difference velocity (+ camera-relative velocity via Rᵀv) from OBB centers, gated by `valid[t] & valid[t-1]` so slot gaps produce no garbage; fused into tokens via a small MLP. |

## W-USG — the external baseline beside the ladder

[lib/supervised/baselines/w_usg/w_usg.py](../lib/supervised/baselines/w_usg/w_usg.py) ·
loss [lib/supervised/baselines/w_usg/loss.py](../lib/supervised/baselines/w_usg/loss.py) ·
config `configs/methods/{predcls,sgdet}/w_usg_*_resnet50.yaml`

The relation machinery of **USG-Par** (Universal Scene Graph Generation, Wu et
al., CVPR 2025) transplanted onto the WSGG substrate. It is *not* a rung: it is
neither a superset nor a subset of any tier, and it exists to show that the
WorldWise result is not an artefact of comparing against one family of
temporal-SGG designs.

**Kept from the shared substrate (identical to W-STTran)** so the comparison is
about the relation stack and nothing else: `vectorized_lks_buffer`,
`GlobalStructuralEncoder` over world-frame OBB corners, `LKSTokenizer` fusion
with `d_camera = 0`, `NodePredictor`, and the pair-token forming of
`RelationshipPredictor` (person ⊕ object ⊕ union-ROI ⊕ CLIP text — USG-Par
likewise fuses visual queries with text embeddings).

The LKS buffer in particular is kept **because USG-Par cannot run without it
here**: USG-Par is pixels-only and has no mechanism for emitting predictions on
unobserved slots, so without the buffer it would score 0 on the occluded-pair
buckets by construction rather than by performance.

**Replaced, following USG-Par's design:**

| Ladder component | W-USG replacement | USG-Par analogue |
|---|---|---|
| `SpatialGNN` (3D positional encoding) | `ObjectContextEncoder` — plain per-frame transformer over object tokens, **no 3D PE** | the shared mask decoder refining object queries (USG-Par's 2D pipeline has no 3D PE) |
| `TemporalEdgeAttention` | *nothing* — no per-pair temporal attention | USG-Par predicts video relations per-frame after object association; slot identity is given in WSGG, so its object associator collapses to the identity map |
| — | `USGRelationDecoder` — pair queries self-attend and cross-attend to the frame's object-token memory | relation proposal constructor + relation decoder |
| — | text-centric alignment logits: cosine similarity between projected object tokens and the frozen CLIP class embeddings | text-centric scene contrastive learning |

**Loss — `WUSGLoss`.** `LKSLoss` unchanged (so relation supervision matches the
ladder baselines exactly: clean visible-pair labels at full weight, unseen
buckets at `lambda_vlm = 0.2`) plus one term: a cross-entropy over the alignment
logits, one per valid object slot, weighted by `lambda_align` (default 0.1).
Relation-level text contrast is deliberately **not** included — the repo ships
CLIP embeddings for object classes only, and inventing predicate text embeddings
would add a data dependency the other baselines do not have.

**Results** (in the tables above): PredCls 67.3 wc R@20 / 33.4 wc mR@20, SGDet
56.3 / 19.1. It lands inside the baseline band on both modes and is tied with
W-DSGDetr for the worst PredCls mean recall (33.4). Its one distinguishing
column is PredCls OU-non-trivial R@20 at **67.5, the best of any baseline** —
ahead of W-STTran++ (67.0) and well ahead of W-DSGDetr++ (55.3), which wins the
headline. That is the LKS buffer doing the work, not the USG relation stack, and
it is a useful reminder that the occluded-bucket R column and the mean-recall
column disagree across the baselines.

## Loss — `LKSLoss` (shared by the four ladder tiers)

[baselines/lks_buffer/loss.py](../lib/supervised/baselines/lks_buffer/loss.py),
aliased per method as `WSTTranLoss` / `WDSGDetrLoss`. Bucketed noisy-label
training over valid pairs:

- **Visible pairs** (both endpoints in camera FOV): full-weight CE (attention) +
  BCE (spatial/contacting) on clean manual labels.
- **Unseen pairs** (≥ 1 endpoint out of view — labels come from a VLM):
  down-weighted by `lambda_vlm` (default 0.2) with label smoothing
  (`label_smoothing_vlm`, default 0.2; KL against a smoothed target for the
  single-label attention head).
- All edge losses use a shared global denominator (total valid pairs) so every
  pair contributes equally to the gradient regardless of bucket size.
- Node CE loss added in sgcls/sgdet modes.

## What baselines deliberately do NOT have

Ego-motion encoding (`CameraTemporalEncoder`), the MWAE mask-and-retrieve
memory (differentiable occlusion handling), the reconstruction / simulated-
unseen objectives, and the tail-aware logit adjustment — all exclusive to
WorldWise. The LKS buffer is intentionally a *passive*, non-differentiable
memory: the baselines "remember" occluded objects only by copying stale
features, which is exactly the limitation WorldWise's MWAE addresses.
