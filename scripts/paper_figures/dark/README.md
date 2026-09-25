# Dark "hero" method figures with real inference panels

One standalone, self-contained script per method, in the visual language of a
project-page hero figure: serif stage titles on rules, left-to-right data flow with
every arrow headed and every intermediate named as a tensor node, three module
kinds (frozen with a snowflake, learnable in purple, the component the variant
introduces in orange), script-L loss nodes with their formula, Title Case on every
component, and three numbered cards on the right. Each figure has three stages and
each stage carries panels rendered from one forward pass of the trained checkpoint
on a real test video.

| Script | Stages | Real panels per stage |
|---|---|---|
| `fig_worldwise.py` | Perception And Geometry Scaffold · Masked World Auto-Encoding · Prediction Heads And Losses | frames, projected OBBs, canonical scene, camera path, object motion, token grid · scaffold / retrieved / enriched tokens, retriever attention · artificial masks, reconstruction similarity, loss per pair, predictions |
| `fig_worldwise_plus.py` | Frozen Foundation-Model Latents · Geometry Scaffold And MWAE · Heads And Losses | DINOv3 / π³ grids, Tier-1 tokens · OBBs, camera, token grid, retrieval · as above |
| `fig_worldwise_pp.py` | Frozen Token Grids And Memory · Geometry Scaffold And Entity Decoder · Heads, Joint Detection And Losses | grids, gate, memory · slots, temporal / spatial / cross attention, decoded slots · Hungarian match, refined boxes, BEV, masks, loss per pair, predictions |

`common.py` holds the palette and primitives (`flow`, `tensor`, `loss_node`,
`stage`, `strip`, `image_slot`). Every image slot falls back to a labelled
placeholder when its PNG is missing, so a layout can be reviewed before the panels
exist; slot sizes follow each PNG's aspect ratio (portrait videos work).

## 1. Dump the intermediates (server, GPU)

```sh
~/anaconda3/envs/scene4cast/bin/python scripts/paper_figures/dump_intermediates.py \
    --video 12XD3 --out-dir /data3/rohith/ag/runs/intermediates
```

writes `<out-dir>/<video>/{worldwise,worldwise_plus,worldwise_pp}/` with one PNG per
panel, `tensors.npz`, `meta.json` (key frames, tracked slot, checkpoint, fitted
focal) and `preds.json`. The predcls best checkpoints are resolved from the configs
in `dump_intermediates.CELLS`.

Two forward passes are made. The **inference pass** (hooks on the tokenizer,
retriever / decoder, grid fusion and the attention modules whose weights the model
discards) feeds the architecture panels. A second **training-mode pass** with the
configured artificial masking (p = 0.3) and every dropout off, on the same
checkpoint and with the EMA update disabled, feeds the loss panels: which visible
tokens were masked, the cosine between the reconstruction and its EMA target, and
the loss each person-object pair receives (visible → scene-graph loss with logit
adjustment; artificially masked → simulated-unseen with clean GT; unseen → none).
For WorldWise++ the detection loss's Hungarian assignment (cost = −p(class) +
5·L1 box; the corner term is omitted in the picture) and the one-to-many
auxiliary matches are drawn on the frame.

The tracked slot is the non-person object with the most unseen frames (seen at
least three times); the key frames are its first visible frame, one where it is
unseen, and the last frame. **Pick a video with unseen slots** (run
`scripts/paper_figures/count_unseen.py`): HI75B, AQQQ5 and QI0EL have none in
predcls (their unobserved objects are absent slots), 12XD3, UG4M2, GFK4S, IDXZK
and R4SJJ do. Dumps exist on the server for all eight.

Geometry facts the dumper relies on (predcls): the item's `corners` are the
annotation's **canonical-frame** corners, `camera_poses[t]` is the 4×4
camera→canonical pose, and π³ ships no intrinsics, so the focal length used for
projection is fitted per video from π³'s own camera-frame points (`fit_focal`;
≈800 px on the 378×672 image of 12XD3). The projected boxes land within a few
pixels of the annotation 2-D boxes, which is the check to repeat on a new video.
The reader in `lib/mllm/data/worldbbox.py` needs the NumPy ≥ 2.3 shim it now has,
and `lib/mllm` + `configs/mllm` must exist in the server checkout that runs the dump.

### A video outside the WorldBBox split (e.g. 00T1E)

Such a video has a legacy annotation under `world4d_rel_annotations/test` with no
camera poses and no stored canonical transform, and none of the frozen-token
caches. The chain that ran for 00T1E (all steps accept explicit ids):

```sh
python datasets/preprocess/tokens/dinov3_tokens.py --split test --videos 00T1E          # scene4cast env
~/anaconda3/envs/pi3/bin/python datasets/preprocess/tokens/pi3_tokens.py --split test --videos 00T1E
python datasets/preprocess/tokens/derive_roi_features.py --split test --mode predcls --stream dinov3_tok --videos 00T1E
python datasets/preprocess/tokens/build_pp_grid_cache.py build --split test --videos 00T1E
python scripts/paper_figures/dump_intermediates.py --video 00T1E --annot-dir world4d_rel_annotations
```

The dumper then reads the π³ scene directly (`LegacyScene`), recovers the rigid
world→canonical transform by Kabsch alignment of the π³ point centroids inside
each annotated 2-D box with the stored box centroids (00T1E: 80 pairs, 7 cm median
residual), carries the π³ camera poses into that frame and injects them into the
item so the ego-motion and spatial encoders run as trained. The projected boxes
then overlap the annotation's 2-D boxes at IoU ≈ 0.45, which reflects the legacy
annotation's own precision rather than the transform.

## 2. Build the figures (local)

```sh
scp -r utd:/data3/rohith/ag/runs/intermediates/12XD3 outputs/intermediates/
python scripts/paper_figures/dark/fig_worldwise.py      --images outputs/intermediates/12XD3/worldwise      --video 12XD3
python scripts/paper_figures/dark/fig_worldwise_plus.py --images outputs/intermediates/12XD3/worldwise_plus --grid-images outputs/intermediates/12XD3/worldwise_pp --video 12XD3
python scripts/paper_figures/dark/fig_worldwise_pp.py   --images outputs/intermediates/12XD3/worldwise_pp   --video 12XD3
```

`build_all.py` runs the three for every video under `outputs/intermediates/` into
`outputs/paper_figures/dark/<video>/`. Outputs are self-contained SVGs (PNGs
embedded as base64). To rasterise on Windows without extra dependencies:

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu --hide-scrollbars --window-size=1720,1400 --screenshot=out.png file:///C:/path/to/worldwise.svg
```

(heights: worldwise 1400, worldwise_plus 1460, worldwise_pp 1560.) A light variant
is a palette swap at the top of `common.py`.

## Reading the 12XD3 panels

- **Stage 1**: the projected OBB corners show the unseen picture's dashed box at
  the hands at frame 19 although no 2-D box exists for it; the camera spans 2 cm.
- **Stage 2**: the retriever's key axis is black outside the picture's visible
  frames; after retrieval the [MASK] cells take the colour of the visible cells in
  their row. In WorldWise++ the unseen slot's cross-attention lands on the hands.
- **Stage 3**: the training-mode pass masks 24 of 102 visible tokens; masked cells
  reconstruct their EMA target at cosine 0.70 vs 0.44 for unmasked ones; the
  Hungarian assignment matches the person at IoU 0.97 and the sandwich at 0.58.
