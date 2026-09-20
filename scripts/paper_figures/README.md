# WorldWise paper architecture figures

Three deterministic Matplotlib diagrams, one for each design document. These
are architecture illustrations, not empirical plots. They require no dataset,
checkpoint, GPU, network request, external artwork, or LaTeX installation.

## Compact paper figures

The separate `generate_publication.py` entry point produces three compact
architecture figures, preserving the original detailed reference diagrams:

```sh
python scripts/paper_figures/generate_publication.py
python scripts/paper_figures/generate_publication.py --variant worldwise_pp
```

Outputs are `worldwise_paper`, `worldwise_plus_paper`, and
`worldwise_pp_paper`, each in PDF, SVG and 600-dpi PNG in the default output
directory. The same `--formats`, `--dpi`, and `--output-dir` options apply.
Use the vector PDF for the manuscript and the SVG for further design edits.

These figures are 7.2 inches wide and approximately 4.4–4.6 inches tall.
They focus on the model contribution, key projections and data flow. The
larger reference figures retain the complete dimension ledger. Review the
final manuscript scaling before submission.

- **WorldWise:** a persistent-slot masking illustration, explicit
  reconstruction branch and distinct node/relation heads. The dashed purple
  branch is the training reconstruction pathway (after visibility embedding,
  before inter-object attention). The mini-graph is schematic, not a dataset
  prediction. The structure/camera/motion/ego concatenation is 576-d before
  adding the 256-d appearance component.
- **WorldWise+:** parallel object/union projectors and an optional gated-fusion
  inset. The dashed Pi3 stream denotes an optional configuration, not a
  train-only branch. The headline DINOv3-only configuration is stated in the
  footer. The lower bypass connects object tokens to node classification.
- **WorldWise++:** persistent slots, free queries, schematic grid memories,
  a repeated decoder block and three readout branches. Dashed purple arrows
  carry spatial-attention weights collected from every decoder layer.
  Grid icons indicate token lattices, not actual spatial dimensions.

In all figures, `3 / 6 / 17` denotes attention, spatial and contacting class
counts, respectively. Block interiors show feature widths unless a full
tensor shape is written. Normalization, activation functions and some
auxiliary operations are condensed; the reference figures and source docs
give the fuller specification. The existing suggested captions below also
describe the compact figures; adjust panel references to the new layouts.

## Generate

From the repository root, in a Python environment with Matplotlib:

```sh
python -m pip install -r scripts/paper_figures/requirements.txt
python scripts/paper_figures/generate_all.py
```

Or generate independently:

```sh
python scripts/paper_figures/generate_worldwise.py
python scripts/paper_figures/generate_worldwise_plus.py
python scripts/paper_figures/generate_worldwise_pp.py
```

All entry points support `--output-dir PATH`, `--formats pdf svg png` (one or
more formats), and `--dpi 600`. The default output directory is
`outputs/paper_figures`, resolved relative to the repository regardless of the
working directory. Each run replaces only the selected output files.

Example:

```sh
python scripts/paper_figures/generate_all.py --formats pdf png --dpi 600
```

On the authoring Windows machine, the base Anaconda environment has a
NumPy/Matplotlib binary incompatibility. Validation used this existing working
environment without modifying base or installing project dependencies:

```powershell
& C:\Users\rohit\anaconda3\envs\wsgtool\python.exe scripts/paper_figures/generate_all.py
```

## Files and editing

| Generator | Figure | Source sections |
|---|---|---|
| `generate_worldwise.py` | `worldwise.{pdf,svg,png}` | `setup/WORLDWISE.md`: Forward pipeline; MWAE core; v2e loss |
| `generate_worldwise_plus.py` | `worldwise_plus.{pdf,svg,png}` | `setup/WORLDWISE_PLUS.md`: sections 1–4 |
| `generate_worldwise_pp.py` | `worldwise_pp.{pdf,svg,png}` | `setup/WORLDWISE_PP.md`: sections 1–5 |

`common.py` contains the shared palette, typography, routing helpers and all
three layouts. Edit labels and coordinates in `worldwise()`, `worldwise_plus()`
and `worldwise_pp()`. Run the exporter again after edits: it checks labels
against their boxes and the canvas. Visual review is still necessary for
connector collisions and semantics.

PDF is the preferred paper inclusion format: vector shapes with embedded
TrueType text. SVG keeps editable text; PNG is a 600-dpi preview/export.
The design width is 7.2 inches with 7.2-point body text. Figures are intended
for full text width; check readability at the final manuscript scale. Colors
reinforce written labels rather than encoding information alone. These are
first architecture figures for manuscript integration, not a certification of
conference formatting compliance.

```latex
% In a single-column manuscript; use figure* if spanning a two-column layout.
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/worldwise.pdf}
  \caption{Insert the corresponding caption below.}
  \label{fig:worldwise}
\end{figure}
```

## Suggested captions

**WorldWise.** Geometry-conditioned masked world auto-encoding. The scaffold
combines structural, camera-relative, ego-motion and object-motion cues with
visible ROI features or learned mask tokens. A per-object associative
retriever recovers masked appearances from visible frames before inter-object
reasoning. Node and reconstruction heads read the resulting object features;
relation prediction combines pair tokens, union features and relative 3-D
geometry, followed by temporal edge attention. Training uses clean simulated
occlusion, EMA feature reconstruction and tail-aware logit adjustment.

**WorldWise+.** A representation substitution at the object and union appearance
projectors. Frozen foundation-model tokens are ROI-pooled at the same reference
boxes used by WorldWise. The headline configuration uses DINOv3 alone; the
dashed geometry stream indicates the optional Pi3 stream used in ablations.
Two-stream fusion uses a per-dimension softmax gate, whereas single-stream
projection reduces to Linear–ReLU–LayerNorm. Geometry, masked recovery, object
reasoning, prediction heads and the scene-graph loss recipe remain unchanged.
Panel (b) condenses the inherited node and relation heads shown in the
WorldWise diagram; node classification is a parallel readout of object tokens.

**WorldWise++.** An image-grounded entity decoder replaces associative retrieval
and inter-object reasoning. Each repeated layer applies temporal attention to
persistent slots, spatial attention to slots and free queries, image
cross-attention to fused token-grid memory, and a feed-forward network.
World slots provide node predictions, reconstruction and box refinement;
free queries provide jointly trained detections. Spatial attention weights
from all layers and pairwise query products replace union ROI features in
the relation pathway. The dashed readout carries spatial attention weights.
Detection and slot-refinement losses supplement the inherited WorldWise
objectives. Free-query detections do not replace the fixed slots used by the
scene-graph evaluation protocol.

## Fidelity choices

- WorldWise uses the documented v2e recipe: 30% artificial visible masking,
  EMA reconstruction, train-time tau 0.5 and lambda_vlm 0.
- WorldWise+ distinguishes the DINOv3 headline cell from the optional fused
  DINOv3/Pi3 configuration. It does not imply fusion is the measured winner.
- WorldWise++ uses DINOv3 ROI slots and both full-grid streams, compressed
  independently to 256 dimensions with PCA fitted only on training tokens.
- Temporal attention operates on slots only. Spatial and image attention
  operate on slots and free queries. Decoder outputs are read after L layers;
  the stacked blocks illustrate one repeated layer, not four different models.
- Visibility embedding is shown with the slot heads as a compact grouping;
  the documented box-refinement branch need not use that embedding.
- No quantitative performance or standalone detection AP is asserted.
  The docs state that free-query detection quality is not yet reported.
- Residual paths, padding masks, tensor dimensions and some auxiliary heads
  are condensed for readability. These diagrams summarize the design docs,
  not a separate audit of executable model behavior.

Source snapshot: the three `setup/WORLDWISE*.md` documents read on 2026-09-19.

## Dimension annotations (2026-09-20 script update)

The scripts now label stage outputs directly and append a stage-by-stage
dimension ledger with input shapes, projection widths and output shapes.
All three figures were subsequently regenerated on request with these
annotations in PDF, SVG and 600-dpi PNG. The expanded layouts were visually
reviewed, and a crowded projector label was shortened before final export.

Dimensions refer to the main configurations below, rather than generic model
defaults or every ablation. They are explicit figure annotations, not loaded
dynamically from YAML. If a model width changes, update the annotations too.

- `configs/methods/predcls/worldwise_predcls_dinov3l.yaml`
- `configs/methods/predcls/worldwise_plus_dinov3tok_predcls.yaml`
- `configs/methods/predcls/worldwise_pp_dinov3_predcls.yaml`
- `lib/supervised/worldwise/scaffold_tokenizer.py`: projected appearance,
  masking, EMA targets and the 832-to-256 scaffold fusion.
- `lib/supervised/components.py`: node heads, 512-to-128 text projection,
  8-to-64 pair geometry, 896-to-256 relation fusion and predicate heads.
- `lib/supervised/worldwise_plus/model.py`: object output width 256,
  union output width 64 and per-dimension fusion gates.
- `lib/supervised/worldwise_pp/model.py`: grid memory, decoder shapes,
  320-to-64 attention readout, detection heads and slot refinement.

Notation: `T` is the number of frames (the code batches frames as `B=T`),
`N` is the padded world-slot count, `K` the padded pair count per frame,
and `C` the number of object-class logits (default 37). For WorldWise++,
`Q=30`, `L=4`, `h=8`, and `P=Hp*Wp`. Spatial resolution is variable;
the full grid is aligned to the Pi3 lattice before the per-stream PCA.
Object, relation and decoder tokens are 256-dimensional; their sequence
lengths differ. Adding position or visibility embeddings preserves width.

WorldWise and WorldWise+ reconstruct from completed retrieval tokens after
visibility embedding, before inter-object attention; node prediction uses
the enriched inter-object tokens. Both tensors have shape `(T, N, 256)`.
WorldWise++ reconstructs from decoder slots after visibility embedding.
The compact node/MWAE box groups these readouts without implying a shared
branch point. `C+1` in free-query detection is the additional no-object logit.

The dimension ledger extends the figure vertically to preserve the original
font size rather than squeeze the extra detail into the existing boxes.
