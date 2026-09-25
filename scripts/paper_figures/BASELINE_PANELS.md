# Baseline and detector panels (`dump_intermediates.py`)

Panels for the five adapted training baselines (`w_sttran`, `w_sttran_pp`,
`w_dsgdetr`, `w_dsgdetr_pp`, `w_usg`; resnet50 predcls) and for the monocular
3-D detector (`mono3d`), produced by the same dumper as the WorldWise panels:

```sh
~/anaconda3/envs/scene4cast/bin/python scripts/paper_figures/dump_intermediates.py \
    --video 12XD3 --methods w_sttran w_sttran_pp w_dsgdetr w_dsgdetr_pp w_usg mono3d \
    --theme light --caps title --out-dir /data3/rohith/ag/runs/intermediates_light
scp -r utd:/data3/rohith/ag/runs/intermediates_light/12XD3/<method> outputs/intermediates/12XD3/
```

Every panel is read out of one forward pass of the best checkpoint on the real
predcls test item; nothing is drawn that the model did not compute. The
checkpoints are resolved from the configs in `BASELINE_CELLS`
(`/data/rohith/ag/checkpoints/<experiment>_v2/best_model.pth`); the detector's
from `MONO3D` (`v1_dinov3l_separate/checkpoint_30`, the DINOv3-L `separate`
head the feature-extraction configs point at).

## How the baselines are hooked

`install_baseline_hooks` registers forward hooks on the real submodules and
forces `need_weights=True` on every `nn.MultiheadAttention` the models discard
weights from (`Capture.patch_mha_weights`). Those attentions live inside
`nn.TransformerEncoder` / `nn.TransformerDecoder` stacks whose fused fast path
would bypass the Python `forward`; the hooked pass therefore runs under
`slow_attention()` (`torch.backends.mha.set_fastpath_enabled(False)`), and a
second plain pass checks that the distributions are unchanged
(`meta.json: hooked_vs_plain_max_abs_diff`, 0 to 1e-7 on 12XD3).

The baselines train **without artificial masking**: the training forward is the
inference forward (dropout off), so the loss panels use the same pass. The LKS
buffer is a `@torch.no_grad` function, not a module; the dumper records the
buffer and staleness the tokenizer actually receives (keyword-argument pre-hook)
and verifies its own NumPy replica of the index arithmetic against them
(`meta.json: lks.{staleness_matches, copies_match, fog_zero}`).

Key frames, tracked slot and target size follow the existing rule
(`pick_keyframes`, `target_size`); on 12XD3: T = 23, six slots (person,
sandwich, food, dish, picture, picture), tracked slot 4 = picture, key frames
1 / 19 / 23 (visible / unseen / last, the picture never reappears).

## Panels per baseline method

| Panel | Tensor / hook | Notes |
|---|---|---|
| `frame_0..2.png` | item `bboxes_2d`, `visibility_mask` | existing `render_frames` (input 2-D boxes; unseen slots as chips) |
| `visibility.png` | `valid_mask`, `visibility_mask` | existing renderer, key frames dotted |
| `obb_0..2.png`, `scene3d_0..2.png` | item `corners`, `camera_poses` + π³ scene | existing geometry renderers (fitted focal in `meta.json`) |
| `camera_path.png` | `camera_poses` | **only** `w_sttran_pp`, `w_dsgdetr`, `w_dsgdetr_pp` (the tiers whose ObjectSpatialEncoder consumes the poses) |
| `motion.png` | `corners` centres | **only** `w_dsgdetr_pp` (ObjectMotionEncoder input) |
| `lks_buffer.png` | tokenizer kwarg `buffer_features` + `lks_sources` | slot × frame; filled = own ROI features, number = frame copied from (teal: held from the past, violet: from the future), hatched = fog (zero features, staleness 1000); copy arrows on the tracked row |
| `staleness.png` | tokenizer kwarg `staleness` | per-slot counter over time, log axis; `log(1+staleness)` is the tokenizer's scalar |
| `appearance_in.png` | item `visual_features` (1024-d resnet50 ROI) | PCA colours, visible cells only |
| `struct_tokens.png` | `global_structural_encoder` output[0] | PCA, all valid cells (geometry exists for unseen slots) |
| `tokens_in.png` | `tokenizer` output | PCA basis shared with `tokens_temporal` / `tokens_out`; unseen cells outlined |
| `spatial_feats.png` | `object_spatial_encoder` output[1] | spatial tiers only |
| `motion_feats.png` | `object_motion_encoder` outputs (frame 1 = no-motion embedding, then frames 2..T) | `w_dsgdetr_pp` only |
| `temporal_obj_attn.png` | `temporal_obj_encoder.encoder.layers[i].self_attn` weights, (N, T, T), mean over heads and layers, tracked slot | `w_dsgdetr`, `w_dsgdetr_pp` |
| `tokens_temporal.png` | `temporal_obj_encoder` output | same tiers |
| `inter_object_attn_0..2.png` | `inter_object_encoder.transformer.layers[i].self_attn` (W-USG: `object_context_encoder.encoder.layers[i].self_attn`), (T, N, N), mean over layers, valid slots at each key frame | rows/cols named; tracked slot orange, unseen tagged |
| `tokens_out.png` | that encoder's output | same PCA basis as `tokens_in` |
| `rel_attn_0..2.png` | `rel_predictor.rel_transformer.layers[i].self_attn`, (T, K, K), mean over layers | pairs named `person → object` |
| `temporal_edge_attn.png` | `temporal_edge_attn.encoder.layers[i].self_attn`, (pairs, T_max, T_max), group of the tracked pair mapped back to frames through the temporal-PE input | not for `w_usg` (it has no temporal edge attention) |
| `preds_0..2.png`, `preds.json` | output distributions vs GT | existing `render_preds` |
| `loss_pairs_0..2.png` | `baseline_pair_losses` on the model's logits | per pair: bucket (visible / unseen one endpoint / unseen both), weight (1 or λ_vlm = 0.2), ℒatt ℒspa ℒcon (each already /N, the pair's contribution to the batch loss), predicted top contacting predicate, the label it is scored against (clean GT or VLM pseudo-label). The terms are reproduced from `LKSLoss` (CE / BCE / KL against the `LabelSmoother` targets, ε = 0.2) and their sums are checked against the real loss class (`WSTTranLoss`/`WDSGDetrLoss` = `LKSLoss`, `WUSGLoss`): `meta.json: loss.{real, reproduced, max_abs_diff, buckets, bucket_counts, alignment_loss}` (max diff ≈ 6e-8 on 12XD3) |
| `usg_rel_self_attn_1.png` | `relation_decoder.decoder.layers[i].self_attn`, (T, K, K) at the unseen key frame | `w_usg` |
| `usg_rel_cross_attn_1.png` | `relation_decoder.decoder.layers[i].multihead_attn`, (T, K, N): pair query × object-token memory | `w_usg` |
| `align_logits.png` | output `align_logits` × temperature = cosine(object token, CLIP class embedding), unseen key frame, GT class outlined | `w_usg` |
| `meta.json`, `tensors.npz` | all captured arrays (`lks_buffer`, `lks_staleness`, `lks_gather`, `struct_tokens`, `tokens_*`, `*_attn_*`, `loss_*`, …) | |

Not produced, and why: `camera_path.png` for `w_sttran` / `w_usg` and
`motion.png` for every tier but `w_dsgdetr_pp` (those models do not consume
the quantity, so the panel would not be empirical for them);
`temporal_edge_attn.png` and `temporal_obj_attn.png` for `w_usg` (no such
modules); `train_mask.png` / `recon_sim.png` (no masking or reconstruction in
the baselines). The RPN scores and 3-D head parameters of the detector are not
returned by the model, so they are recovered by patching `rpn.filter_proposals`
and by re-running the 3-D layers on the hooked inputs (checked equal to the
model's own corners: `meta.json: per_frame.*.rerun_matches_model_output`).

## Panels for `mono3d/`

Run on the three key frames of the predcls item (same rule, same slot), at the
detector's resolution (`pixel_limit 255000`, aspect-preserving, multiples of the
training config's `patch_size 14`; 12XD3: 270×480 → 378×672, padded to 384 wide
by the batching transform).

| Panel | Tensor / hook |
|---|---|
| `frame_0..2.png` | the resized, normalised input (de-normalised for display) |
| `fpn_0..2.png` | `model.backbone` output dict p2..p6 (256 channels; 168×96 … 11×6 on 12XD3), channel L2 norm per cell |
| `proposals_0..2.png` | `rpn.filter_proposals` output (boxes, objectness) — top-50 of the 1000 post-NMS proposals |
| `det2d_0..2.png` | final detections (`boxes`, `labels`, `scores`), score ≥ 0.3, colour per class |
| `det3d_0..2.png` | predicted 8-corner camera-frame boxes projected with the frame's pinhole; annotation boxes (camera frame, `monocular3d_bbox_annotations/<video>.pkl`) dashed |
| `bev_0..2.png` | top-down x (lateral) vs z (depth), predicted solid vs annotation dashed, camera at the origin |
| `head_0..2.png` | per detection: dims l, w, h and depth as bars, yaw (atan2 of the normalised sin/cos) and the uncertainty μ |
| `meta.json`, `dets.json`, `tensors.npz` | checkpoint, resolution, intrinsics, thresholds; per-frame detections with dims / yaw / depth / offset / μ / corners and the annotation boxes |

**Intrinsics.** In inference the model's `separate` head is fed the image-size
default (f = max(H, W), principal point at the centre; `_gather_intrinsics`
with `targets=None`, exactly as the ROI-feature extraction runs it). For 12XD3
this equals the annotation's stored intrinsics (fx = fy = 672, cx = 189,
cy = 336, i.e. already in the 378×672 frame), which reproject the annotation's
3-D boxes onto the 2-D boxes. The training dataset (`ag_dataset_3d.py`) would
scale those by the resize factor (940.8, 264.6, 470.4); the dumper re-runs the
3-D head under both conventions and keeps the one whose box centres are closer
to the annotation (`meta.json: per_frame.*.centre_error_m_vs_annotation`; on
12XD3 the image-default wins, 5–9 cm vs 16–17 cm), and always projects with the
annotation pinhole. Both conventions and the choice are recorded per frame.

## Observations on 12XD3

- The tracked picture is visible in frames 1–13 (slot absent at 4 and 6) and
  never again; every unseen cell from frame 14 on copies frame 13 (zero-order
  hold from the past, staleness climbing to 10 at frame 23). The dish borrows
  from the future twice (frames 15 and 21). No slot is ever in fog.
- 92 valid pairs: 79 visible (clean labels, w = 1) and 13 with the unseen
  picture as object (VLM pseudo-label, w = 0.2); no unseen–unseen pair. The
  picture's pseudo-labels (`carrying, holding, touching`) contradict the
  prediction (`not_contacting`), which is where its down-weighted BCE comes from.
- After the tokenizer the picture's unseen cells keep the colour of its visible
  cells (the buffer copy), unlike WorldWise's `[MASK]` cells; the inter-object
  attention of the unseen picture at frame 19 is spread over sandwich/food/dish,
  and the temporal edge attention of person→picture forms a block over the
  held frames.
- The detector finds the person (1.00), dish, food and sandwich at the hands
  (depth 0.8–1.0 m, sizes 3–11 cm) but not the picture or the window; μ is
  strongly negative for the small objects (the head is confident there).
