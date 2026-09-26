# "Hero" method figures with real inference panels

One standalone, self-contained script per method, in the visual language of a
project-page hero figure: serif stage titles on rules, left-to-right data flow with
every arrow headed and every intermediate named as a tensor node, three module
kinds (frozen with a snowflake, learnable in purple, the component the variant
introduces in orange), script-L loss nodes with their formula, Title Case on every
text, and three numbered cards on the right. Each figure has three stages and
each stage carries panels rendered from one forward pass of the trained checkpoint
on a real test video.

## Theme and capitalisation

Two palettes live in `common.py`: **light** (white background, the default) and
**dark** (the hero palette the figures were designed in; the folder keeps its
name). Every figure script takes `--theme light|dark` and `--caps title|upper|none`
(`common.py` reads them at import, or `FIG_THEME` / `FIG_CAPS`). With `title`
(default) `Canvas.text` capitalises the first letter of every word; tokens that
are not plain words (formulas, subscripts like `sₜ,ₙ`, numbers, units, single-letter
variables, `[MASK]`, `DINOv3-L`, `IoU`) are left as written, and monospace text
(prompt templates, schemas) is verbatim. The panels carry their own palette:
`dump_intermediates.py --theme light|dark --caps ...` renders them to match, and
the same rule capitalises every matplotlib text (titles, annotations, legends,
object-name tick labels). A light figure therefore needs a **light dump**; the
two are kept apart on the server as `intermediates/` (dark) and
`intermediates_light/`.

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
    --video 12XD3 --out-dir /data3/rohith/ag/runs/intermediates_light --theme light   # --theme dark → intermediates/
```

(`scripts/paper_figures/run_light.sh` on the server loops over the nine dumped
videos.) It writes `<out-dir>/<video>/{worldwise,worldwise_plus,worldwise_pp}/` with one PNG per
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
scp -r utd:/data3/rohith/ag/runs/intermediates_light/12XD3 outputs/intermediates/     # light panels (default theme)
python scripts/paper_figures/dark/fig_worldwise.py      --images outputs/intermediates/12XD3/worldwise      --video 12XD3
python scripts/paper_figures/dark/fig_worldwise_plus.py --images outputs/intermediates/12XD3/worldwise_plus --grid-images outputs/intermediates/12XD3/worldwise_pp --video 12XD3
python scripts/paper_figures/dark/fig_worldwise_pp.py   --images outputs/intermediates/12XD3/worldwise_pp   --video 12XD3
```

`build_all.py [--theme ..] [--caps ..] [--png] [--mllm]` runs the three for every
video under `outputs/intermediates/` into `outputs/paper_figures/dark/<video>/`
(12XD3 also at the top level), optionally the five MLLM figures, and with `--png`
rasterises everything. Outputs are self-contained SVGs (PNGs embedded as base64).
`rasterize.ps1` rasterises on Windows without extra dependencies (headless Edge;
the canvas size is read from each SVG):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/paper_figures/dark/rasterize.ps1 -Dir outputs/paper_figures/dark
```

It gives every render a fresh Edge profile and waits for that launch's processes
to exit: a lingering instance adopts the next launch and exits 0 without writing
the screenshot. (Its parameters are named-only; `$args` / `$profile` are reserved
PowerShell names and must not be reused in such a script.)

## Processing-unit layout (training-based methods)

The WorldWise family (`fig_worldwise*.py`, units 2-4 of WorldWise / WorldWise+
shared through `ww_units.py`) and the adapted baselines (`baseline_common.py`)
are drawn as four processing units, top to bottom, each a tinted enclosure with a
numbered header (`Canvas.begin_unit` / `end_unit` in `common.py`) and a card
beside it (`Canvas.unit_card`):

| Unit | WorldWise / WorldWise+ | WorldWise++ | Baselines |
|---|---|---|---|
| 1 Observed Objects | detector (WW) or frozen DINOv3 / π³ latents + gated fusion (WW+), scaffold, encoders, scaffold tokenizer | token grids + memory, scaffold, encoders, tokenizer, world slots + free queries | detector, world-frame scaffold, structural / spatial / motion encoders |
| 1 + 2 shared | – | entity decoder × 4 (`u12`) | – |
| 2 Unobserved Objects | associative retriever, visibility embedding, reconstruction vs EMA target | visibility embedding, completed slots, reconstruction | LKS buffer + staleness, LKS tokenizer, motion fusion, temporal object encoder |
| 3 Relationship | inter-object transformer + 3-D PE, pair encoder, temporal edge attention | pair readout from the decoder's spatial attention, pair encoder, temporal edge attention | inter-object transformer + 3-D PE (object context encoder in W-USG), pair former, temporal edge attention (USG relation decoder) |
| 4 Decoders | node head, predicate heads, losses | + detection heads, box / corner refinement | node predictor, predicate heads, bucketed loss (+ alignment head in W-USG) |

Arrows that cross units carry a port outlined in the source unit's colour
("From Unit 1"); the one backward arrow (object classes, Unit 4 → the CLIP text
pathway of Unit 3) is labelled as such. An overview strip under the title
(`Canvas.unit_overview`) summarises the units with the video on the left and the
predicted graph on the right. Captions and card numbers (key frames, tracked
object, LKS copies, bucket counts) come from the dump's `meta.json`.

These figures are drawn for **00T1E** (baseline dumps: `run_baselines_light.sh`):

```sh
python scripts/paper_figures/dark/build_all.py --videos 00T1E --png
python scripts/paper_figures/dark/crop_stages.py assets/figures/architecture/paper_figures/dark/00T1E/{worldwise,worldwise_plus,worldwise_pp,w_sttran,w_sttran_pp,w_dsgdetr,w_dsgdetr_pp,w_usg}.svg --out-dir assets/figures/architecture/paper_figures/dark/units
```

Every unit writes a `<!-- crop NAME x0 y0 x1 y1 -->` marker; `crop_stages.py`
turns them into `<name>_overview`, `_u1` .. `_u4` (and `_u12`) pages, falling back
to the band rules for the stage figures (MLLM methods, detector).

### Compact main-paper variants

`fig_main_units.py` draws a short-form diagram of each of the same eight methods
for the main paper, as one left-to-right forward pass (2.1-2.6k x 0.7-0.87k px):

* top row: video → the four units as tinted enclosures, one column per module, so
  every arrow is long; arrows that cross into the next unit name their tensor, and
  long-range inputs (union features, the enriched tokens read by the node head, the
  decoder's spatial attention, free queries) run on labelled buses entering boxes
  from the top or bottom. WorldWise++'s entity decoder sits in the overlap of units
  1 and 2. The node and predicate heads feed a plain "World Scene Graph" output box on
  the right (user request: no predicted graph and no predicate panel; `output_graph`
  is kept but unused); each unit's losses at its foot, named as in the supplement's
  equations (ℒ_vis/ℒ_vlm/ℒ_align, ℒ_recon/ℒ_SG/ℒ_sim, ℒ_det/ℒ_slot);
* bottom row: real PredCls intermediates of the same video, each numbered with the
  badge of the module that produced it (all dumps are PredCls checkpoints).

Specs (modules with column / lane, edges with their routes, panels with badges) are
at the top of the script. Output goes to `assets/figures/architecture/paper_figures/main/`
(`.svg`, `.png`, vector `.pdf`) and never touches the long-form figures in `dark/`.

```sh
python scripts/paper_figures/dark/fig_main_units.py --png --pdf        # all eight, TM0BV (default)
cp assets/figures/architecture/paper_figures/main/*.pdf <paper>/sup_images/architectures/compact/
```

The showcase video is **TM0BV** (dumps: `run_showcase_light.sh`, key frames forced with the
dumper's `--keyframes 13 21 23`): `find_showcase.py` / `video_detail.py` scanned the all-frame
PredCls prediction dumps for videos where WorldWise++ is right on every unseen-object pair; on
TM0BV it is right on all 11 unseen pair cells and on every pair at the three key frames, the only
method that is. `--video 00T1E` still rebuilds the earlier set.

`--portrait` folds each diagram into two tiers for portrait pages (units 1-2 on top,
units 3-4 and the scene graph below; top-to-bottom edges run down a channel right of
unit 2 or straight down, then along their own corridor lane, drawn with a white halo so
crossings read as bridges; intermediates in two rows) and writes
`paper_figures/main_portrait/`. The supplement uses these portrait PDFs
(`<paper>/sup_images/architectures/portrait/`); the one-row versions stay for the main paper.

The supplement's training-method subsections (`sup_methods_baselines.tex`,
`sup_methods_worldwise.tex`) use these PDFs as their one landscape figure per
method; their text follows the same four units.

## 3. Into the paper (vector PDFs)

`svg2pdf.ps1` prints an SVG to a vector PDF with headless Edge (an HTML wrapper
with `@page { size: <w>px <h>px }`; text stays text, panels stay raster; the
1720-px canvas becomes a 1290-pt page). `export_paper.py` maps every figure to
its canonical source (00T1E for the WorldWise family, the baselines and the
MLLM methods; 12XD3 for the detector) and writes `<paper>/sup_images/architectures/<name>.pdf`;
`--stages` / `--stages-only` also cuts each figure into its three stage bands
with `crop_stages.py` (`<name>_s1..3.pdf`, 1280 px wide, cards column dropped)
because a whole canvas at text width is unreadable in print; the supplementary
places one stage per landscape page (`sidewaysfigure`).

```powershell
python scripts/paper_figures/dark/export_paper.py --stages          # all 14 figures + 42 stage pages
python scripts/paper_figures/dark/export_paper.py --only worldwise --stages-only
```

`crop_stages.py` finds the four band rules every script draws through
`Canvas.stage` (x = 30..1290, stroke-width 1.5), so a new figure script must keep
using `stage()` for its bands to be croppable.

## Reading the 12XD3 panels

- **Stage 1**: the projected OBB corners show the unseen picture's dashed box at
  the hands at frame 19 although no 2-D box exists for it; the camera spans 2 cm.
- **Stage 2**: the retriever's key axis is black outside the picture's visible
  frames; after retrieval the [MASK] cells take the colour of the visible cells in
  their row. In WorldWise++ the unseen slot's cross-attention lands on the hands.
- **Stage 3**: the training-mode pass masks 24 of 102 visible tokens; masked cells
  reconstruct their EMA target at cosine 0.70 vs 0.44 for unmasked ones; the
  Hungarian assignment matches the person at IoU 0.97 and the sandwich at 0.58.

## Adapted baselines and the detector

Six more figures in the same visual language: one per adapted training baseline
(`setup/BASELINES.md`, facts read off `lib/supervised/baselines/` and
`lib/supervised/components.py`) and one for the monocular 3-D detector
(`setup/MONOCULAR_3D_DETECTOR.md`, `lib/detector/monocular3d/`). Every figure is
complete on its own: the shared modules are repeated in each, the component the
tier adds over the tier below is orange, the non-differentiable LKS buffer is a
teal program ("No Gradient"), and card 3 says what the baseline does not have.
The five baselines share `baseline_common.py` (stages, strips, cards, CLI); the
panels come from `--images outputs/intermediates/<video>/<method>/` and fall back
to labelled placeholders until the dump exists. `strip_fit` shrinks a panel row so
it always fits the band, whatever the PNG aspects turn out to be.

| Script | Stages | Orange | Panels used | Canvas |
|---|---|---|---|---|
| `fig_w_sttran.py` | Perception, Geometry Scaffold And LKS Memory · Tokenization And Spatio-Temporal Reasoning · Relationship Heads And Bucketed Loss | structural encoder, LKS tokenizer, 3-D spatial PE (the world-frame substrate over STTran) | frame_0-2, obb_1, scene3d_1, visibility, lks_buffer, staleness · appearance_in, struct_tokens, tokens_in, inter_object_attn_1, tokens_out · rel_attn_1, temporal_edge_attn, loss_pairs_1, preds_1 | 1720 × 1522 |
| `fig_w_sttran_pp.py` | same | object spatial encoder (camera-relative features, 128-d camera slice) | + camera_path, spatial_feats | 1720 × 1588 |
| `fig_w_dsgdetr.py` | same | temporal object encoder (per-slot attention across T, before the inter-object transformer) | + camera_path, temporal_obj_attn, tokens_temporal (struct_tokens and spatial_feats are shown only where they are new) | 1720 × 1591 |
| `fig_w_dsgdetr_pp.py` | same | object motion encoder + motion fusion | + motion, motion_feats | 1720 × 1605 |
| `fig_w_usg.py` | same | object context encoder (no 3-D PE), USG relation decoder, text-centric alignment head + ℒ_txt | usg_rel_self_attn_1, usg_rel_cross_attn_1, align_logits replace rel_attn_1 / temporal_edge_attn | 1720 × 1553 |
| `fig_mono3d.py` | Frozen Backbone And Feature Pyramid · Region Proposals And 2-D Detection · Factorized 3-D Head And Disentangled Loss | the factorized 3-D head (context FC + five linear heads) | frame_0-2, fpn_0-2 · proposals_1, det2d_0-2 · head_1, det3d_0-2, bev_1 | 1720 × 1513 |

```sh
for m in w_sttran w_sttran_pp w_dsgdetr w_dsgdetr_pp w_usg mono3d; do
  python scripts/paper_figures/dark/fig_$m.py --images outputs/intermediates/12XD3/$m --video 12XD3 \
      --out outputs/paper_figures/dark/12XD3/$m.svg
done
```

Where the code overrides the older method notes (`docs/tex_from_sources/w_*_method.tex`,
the table in `setup/TRACK1_TRAINING.md`): W-STTran++ adds *only* the object spatial
encoder (temporal edge attention is already in W-STTran, there is no motion
encoder); W-DSGDetr keeps that spatial encoder and its temporal PE is a learnable
embedding, not sinusoidal; W-DSGDetr++ has no ego-motion encoder and its motion
fusion is `[x ⊕ μ] → Linear → LN → GELU`, with the acceleration input always zero;
the detector's 3-D context FC reads 1024 + 4 + 4 = 1032 inputs (the tex says
1036) and the V1 configs run with the weight ramp off.

## MLLM architecture figures (training-free tracks)

Five more figures in the same visual language, one per MLLM method, drawn from
the method docs (`setup/TRACK2_UNLOCALIZED.md`, `MLLM_ZERO_SHOT.md`,
`MLLM_CAPTION_ALL.md`, `UNLOCALIZED_GRAPH_RAG.md`, `TRACK3_LOCALIZED.md`,
`LOCALIZED_MLLM.md`, `MLLM_TRACK_A.md`, `MLLM_TRACK_B.md`) and, where the two
differ, from the runners in `lib/mllm/methods/`. Every figure is complete on its
own: the modules a method shares with its neighbours (the visual input Q(f), the
SGDet object discovery, the parser, the perception layer, Track A's payload
builder) are drawn again in each figure rather than referred to.

| Script | Method | Stages | Orange | Canvas |
|---|---|---|---|---|
| `fig_zero_shot.py` | U-WSGG-Zero (`zero_shot`) | Object Set And Visual Input Construction (Frames Only) · Per-Object Relationship Query (One Batched Call) · Parsing And Scene Graph | nothing: it is the control (only the target frame / Q(f) is orange) | 1720 × 1264 |
| `fig_caption_all.py` | U-WSGG-Sub (`caption_all`) | Caption Transcript Generation (Stage-1 Build, Once Per Video) · Caption-Prefixed Per-Object Query · Parsing And Scene Graph | the caption-context program c(f) (captions sorted by \|kᵢ − f\|, prepended to P₄) | 1720 × 1548 |
| `fig_graph_rag.py` | Unlocalized Graph-RAG (`rag_all`) | Coarse Event-Graph Construction (Offline, Once Per Video) · Object Discovery And Per-Object Graph RAG · Per-Frame Relationship Prediction | mean-cosine node retrieval | 1720 × 1180 |
| `fig_track_a.py` | Localized Track A (`track_a`) | Perception Layer In The Canonical Frame (Offline, No VLM) · Localized Prompt With Shared Ids (The Payload Builder) · One Call Per Frame, Parsing And Localized Scene Graph | the payload builder (set-of-mark frame, marked BEV, metric table in one id space) | 1720 × 1236 |
| `fig_track_b.py` | Localized Track B (`track_b`) | Perception Layer In The Canonical Frame (Offline, No VLM) · Perceive And Retrieve: Track A's Payload Plus A Temporal Window Of The Event Graph · Relate, Verify, Repair, Emit (The Critic Is A Program) | the geometric critic | 1720 × 1420 |

Nothing in these pipelines is trained, so the module kinds change: every network
is **frozen** (snowflake: the VLM, BGE-large, Pi-3, GDino), deterministic code is
a **program** (teal), the component the method introduces is **orange**, and calls
that exist in the code but change no reported number are **ghosted**: the Yes/No
relation verification and the clip it would read (every reported unlocalized run
uses `--skip-verification`), Graph-RAG's P₃ node relevance check, and the P₀ graph
node that `caption_all` builds but never reads. The P₀–P₆ badges are the prompt
ledger of `setup/UNLOCALIZED_GRAPH_RAG.md` §7 (zero_shot uses P₄, P₅, P₆;
caption_all adds P₁ and the ghosted P₀; the localized tracks have no ledger). The
decoding settings are written on each figure: Qwen2.5-VL-7B through vLLM at
T = 0.2 (fixed in the wrapper), ≤ 128 new tokens for P₄, ≤ 256 for P₂ / P₆, one
video per prompt, chunks of 64 prompts, for the unlocalized three; Qwen3-VL-8B
with 4 images, ≤ 1,024 new tokens, T = 0.2, top_p 0.95, seed 0, four videos pooled
per `generate()`, a content-hash response cache, for the localized two. Shared
primitives, the schematic placeholders and the repeated blocks
(`visual_input_block`, `object_set_block`, `parse_block`, `perception_block`,
`payload_images`) live in `mllm_common.py`.

```sh
for m in zero_shot caption_all graph_rag track_a track_b; do
  python scripts/paper_figures/dark/fig_$m.py [--images outputs/intermediates/00T1E/$m] [--video 00T1E] \
      [--out outputs/paper_figures/dark/00T1E/$m.svg] [--theme light|dark] [--caps title|upper|none]
done
```

Without `--images` every intermediate slot draws a schematic of what it will
hold. The schematics are built only from the prompt templates and output schemas
in the code, never from invented results. A PNG named `<slot>.png` in `--images`
replaces its schematic, sized by aspect ratio as in the WorldWise figures.

### Real panels from one video (step 2)

A video outside the WorldBBox split has no MLLM runs, so the figure runs are
one-off, sandboxed runs of the headline commands. On the server (the shared MLLM
checkout `~/CODE/Scene4Cast_mllm` is used as it is; nothing is written into it):

```sh
SB=/data3/rohith/ag/runs/intermediates/mllm           # scp scripts/paper_figures/mllm_sandbox/* and
PY=~/anaconda3/envs/wsg/bin/python                     # scripts/paper_figures/dump_mllm_intermediates.py to $SB/tools/
$PY $SB/tools/make_sandbox.py $SB 00T1E=/data/rohith/ag/world4d_rel_annotations/test_bak/00T1E.mp4.pkl \
                                 0DJ6R=/data/rohith/ag/world4d_rel_annotations/train/0DJ6R.mp4.pkl
WSGG_MLLM_CONFIG=$SB/config.yaml $PY $SB/tools/precheck.py        # CPU: loader + every payload, fills the map caches
SB=$SB GPU=2 setsid nohup sh $SB/tools/run_chain.sh     > $SB/logs/chain.out     2>&1 < /dev/null &   # graph, rag_all, Track A/B
SB=$SB GPU=2 setsid nohup sh $SB/tools/run_ctx_chain.sh > $SB/logs/ctx_chain.out 2>&1 < /dev/null &   # zero_shot, caption_all (after the graph)
```

`make_sandbox.py` writes a config whose annotation dir, split file, graphs, run
outputs, map / detection / lift / LLM caches, logs and status files all live under
`$SB`, so the shared WorldBBox runs, caches and manifest are never touched.
`run_chain.sh` runs the headline commands of `scripts/remote/b3_baselines.jobs` in
order: Stage-1 graph (qwen25vl_7b, `--fast`), `rag_all` PredCls / SGDet
(qwen25vl_7b, `--skip-verification`), Track A and Track B PredCls / SGDet
(qwen3vl_8b, 1,024 tokens, T = 0.2); two videos took 14 min on one A40.
`run_ctx_chain.sh` runs `zero_shot` and `caption_all` PredCls / SGDet the same way
(qwen25vl_7b, `--skip-verification`; it needs the Stage-1 graphs of the first
chain): 5 min 45 s for the two videos (1:03, 1:21, 1:06, 2:15 per step). Each step
runs in its own process group, so a leftover `VLLM::EngineCore` is reaped with it.
Two capture wrappers run the runners unchanged (every original is called first)
and record what the pickles drop: `capture_rag.py` the P₂ reasoning prompt, raw
response and keywords, the mean BGE cosine of every entity key and node
(recomputed from the runner's own embedding cache), the returned top-20 nodes, the
context blocks, and the V and Q(f) tensors; `capture_ctx.py` (for `zero_shot` and
`caption_all`) the exact text of every P₄ prompt mapped back to (frame, object),
the P₆ discovery prompt and answer, the P₅ prompts, the caption list, and the same
V, C and Q(f) tensors.

Then bundle, copy and render:

```sh
WSGG_MLLM_CONFIG=$SB/config.yaml $PY $SB/tools/dump_mllm_intermediates.py --video 00T1E --capture $SB/capture \
    --out /data3/rohith/ag/runs/intermediates/00T1E/mllm_bundle [--only runs]         # server, CPU
ssh utd "tar -C /data3/rohith/ag/runs/intermediates/00T1E -cf - mllm_bundle" | tar -C outputs/intermediates/00T1E -xf -
python scripts/paper_figures/dark/mllm_panels.py --bundle outputs/intermediates/00T1E/mllm_bundle \
    --out outputs/intermediates/00T1E [--theme dark] [--frame 000038.png] [--object shoe] [--figures track_b] [--only tb_graph]
python scripts/paper_figures/dark/build_all.py --png            # builds the five MLLM figures for 00T1E
```

The bundle is plain JSON / PNG / NPZ: ground truth in the canonical frame, key
frames, raw-frame thumbnails, the Stage-1 graph, every run output
(`{zero_shot,caption_all,rag_all,track_a,track_b}_<mode>.json`), the captures
(`capture_<method>_<mode>.json/.npz`), the caches, and the exact Track A inputs of
every frame (regenerated with `build_payload`) plus the Track B prompt.
`--only runs` rewrites just the run outputs and captures after a new chain.
`mllm_panels.py` writes `zero_shot/`, `caption_all/`, `graph_rag/`, `track_a/` and
`track_b/` (`_dark` suffix for the dark palette; one PNG per slot at 4× the slot
size) and `mllm_picks.json`. What the panels show is chosen by fixed rules, never
by whether a prediction is right:

- object o: the one with parsed P₂ keywords and the most unseen frames; the three
  unlocalized figures all show the same (f, o), so their prompts and answers compare;
- frame f: a frame where o is unseen, maximising #visible × #unseen objects
  (middle of the ties); Track A's Stage 2 and 3 show its PredCls prompt and answer;
- Stage-1 segment: the segment of o's top-1 retrieved node, so the node card, the
  orange bar and the context block are the same node (caption_all shows the same
  segment and caption);
- critic example: a Track B frame whose repair moved a flagged box and lowered the
  violation count (SGDet first, nearest to f), else any repair that lowered it,
  else any flagged frame. Every Track B panel (proposals, lift, payload, retrieved
  block, both calls, violations, before / after, emitted graph) shows that one
  frame in that mode, so the figure follows one frame through the whole loop.

| Figure · slot | Bundle source |
|---|---|
| all · `frame_0/1/2` | `frames_annotated/` (the segment's key frame and the first / last eighth of the video) |
| zero_shot, caption_all · `context_frames`, `query_tensor`, `video_v` | `capture_<method>_predcls.npz` (C, Q(f)) and `_sgdet.npz` (V): the tensors the VLM received |
| zero_shot, caption_all · `objects`, `discovery_card` | `<method>_sgdet.json` `estimation_meta`; `capture_<method>_sgdet.json` `estimation` (the P₆ prompt and raw answer) |
| zero_shot, caption_all · `prompt_card` | `capture_<method>_predcls.json` `prompts`: the exact P₄ text of (f, o); for caption_all with the caption prefix in the runner's order |
| zero_shot, caption_all · `answer_card`, `scene_graph` | `<method>_predcls.json` + `video.json` (GT) |
| caption_all · `segments`, `caption_card`, `transcript` | `graph.json` + `raw_thumbs.npz`; `capture_caption_all_predcls.json` `captions` |
| graph_rag · `segments`, `caption_card`, `node_card` | `graph.json` + `raw_thumbs.npz` |
| graph_rag · `event_graph`, `keywords_card`, `ranked_nodes`, `context_card` | `capture_rag_all_predcls.json` (entity index, P₂, cosines, contexts) |
| graph_rag · `video_v`, `query_tensor`, `objects`, `answer_card`, `scene_graph` | `capture_rag_all_*.npz`; `rag_all_sgdet.json` `estimation_meta`; `rag_all_predcls.json` + `video.json` |
| track_a, track_b · `cloud`, `bev_base` | `cache/` (cloud.npz, bev.png, meta.json) |
| track_a · `proposals_2d`, `lifted_obbs` | `cache/detections.json` + `payload/sgdet/<f>_objects.json` (frame f) |
| track_a · `marked_frame`, `context_0/1`, `marked_bev`, `prompt_card` | `payload/predcls/<f>_img*.png`, `_prompt_a.txt` |
| track_a · `answer_card`, `localized_graph` | `track_a_predcls.json` + `video.json` |
| track_b · `proposals_2d`, `lifted_obbs`, `marked_frame`, `context_0/1`, `marked_bev` | the same, at the critic frame and mode |
| track_b · `prompt_card`, `retrieved_card` | `payload/<mode>/<frame>_prompt_b.txt` (the retrieved block before "Task:"); `track_b_<mode>.json` `retrieved` |
| track_b · `proposal_card`, `violations_card`, `repair_card` | `track_b_<mode>.json`: `raw_response_pre`, `violations_pre`, `raw_response` (call 2) |
| track_b · `repaired_graph`, `critic_before_after` | `track_b_<mode>.json` `objects` / `objects_pre` + `payload/<mode>/` + `cache/cloud.npz` |

### Reading the 00T1E panels

- 29 annotated frames with bed, doorway, laptop and shoe: 51 visible and 65 unseen
  slots. The annotation is `world4d_rel_annotations/test_bak`; the `test` copy has
  no canonical-frame fields. The rules pick o = shoe and f = 38 (shoe unseen; GT
  not_looking_at / not_contacting / in_front_of).
- zero_shot: the P₄ text for shoe is 1,298 characters and identical at every frame;
  the visual input is [f ; 15 of the 29 key frames] at 168×336. At (38, shoe) it
  answered looking_at / wearing / on_the_side_of. SGDet, told "(No captions
  available for this video.)", named 12 classes, only bed and doorway among the GT
  four; the P(Yes) ordering is floor, bed, pillow, blanket, …
- caption_all: the prefix is all 36 captions sorted by |kᵢ − 38| (6,167
  characters; the whole prompt 7,467), the caption of frame 38 first. At (38, shoe)
  it answered not_looking_at / not_contacting / on_the_side_of. SGDet, with the
  transcript, named 32 classes, all four GT objects among them: the same set as
  rag_all's, which uses the same P₆ inputs.
- Graph-RAG: P₂'s JSON did not parse for bed and doorway (the model wrote `'none'`
  in single quotes), so their retrieval ran on the question alone. The entity key
  `person` scores above 0.5 for every question and appears in all 36 nodes, so every
  node is a candidate and the kept 20 are simply the closest by node text. At
  (38, shoe) it answered looking_at / holding / in_front_of.
- Track A (PredCls, frame 38): the answer covers the 4 ids, then invents entries up
  to id 31 until it is cut at 1,024 tokens (26 of 29 answers are cut this way; the
  parser keeps only the real ids).
- Track B: in SGDet the critic flagged 20 of 29 frames (37 violations: 19 size,
  9 contact, 9 vertical); one repair lowered the count in 8, cleared it in 1 and
  raised it in 2 (29 left). In PredCls it flagged 3 frames, all contact, and each
  repair cleared its flag. The figure follows SGDet frame 65 (one GDino proposal,
  the laptop): call 1 proposes 6 objects, a bed box floating at z = 1.80 m is
  flagged twice (lying_on at a 1.07 m gap; 'beneath' with the bed above the
  person), the repair lowers it to 0.43 m, which clears the contact flag while
  'beneath' stays flagged.

Rasterise with `rasterize.ps1` as above. All five scripts take the same
`--theme` / `--caps` flags; `mllm_common.py` draws its schematics from the palette
names, so the light theme needs no further changes.
