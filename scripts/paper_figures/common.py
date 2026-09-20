"""Deterministic, vector-first architecture figures from setup/WORLDWISE*.md.

No model checkpoints, GPU, external artwork, or LaTeX installation required.
Coordinates are in diagram units; typography is sized for a 7.2-inch figure.
Shapes use the main DINOv3 configurations, not all possible ablations.
See README.md for the code/config provenance of each dimension.
"""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#182B3A"
COLORS = {"blue": ("#EAF2FA", "#326A9B"),
          "teal": ("#E6F3EF", "#267D70"),
          "orange": ("#FFF1DF", "#AB6C22"),
          "gray": ("#F3F5F7", "#687782"),
          "purple": ("#F0ECF7", "#78609A")}


class Diagram:
    def __init__(self, title, subtitle, height=10.8):
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                             "pdf.fonttype": 42, "ps.fonttype": 42,
                             "svg.fonttype": "none", "axes.unicode_minus": False})
        self.fig = plt.figure(figsize=(7.2, 7.2 * height / 16), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, 16), ylim=(0, height))
        self.ax.axis("off")
        self.text(.4, height - .48, title, size=12, weight="bold", ha="left")
        self.text(.4, height - .99, subtitle, size=7.5, ha="left", color="#516271")
        self.text_boxes = []

    def text(self, x, y, text, size=7.2, weight="normal", ha="center", color=INK):
        return self.ax.text(x, y, text, fontsize=size, fontweight=weight,
                            ha=ha, va="center", color=color, linespacing=1.45)

    def box(self, x, y, w, h, title, body="", color="blue", dashed=False):
        fill, edge = COLORS[color]
        self.ax.add_patch(FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0,rounding_size=0.12",
                          linewidth=.9, edgecolor=edge, facecolor=fill,
                          linestyle="--" if dashed else "-", zorder=2))
        if body:
            head = self.text(x+w/2, y+h-.35, title, weight="bold", size=7.7)
            content = self.text(x+w/2, y+(h-.52)/2, body)
            self.text_boxes.append((x, y, w, h, head, content))
        else:
            head = self.text(x+w/2, y+h/2, title, weight="bold", size=7.5)
            self.text_boxes.append((x, y, w, h, head))
        return (x, y, w, h)

    def arrow(self, points, dashed=False, color=INK):
        for a, b in zip(points[:-2], points[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=.8,
                         ls="--" if dashed else "-", zorder=1)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1],
                          arrowstyle="-|>", mutation_scale=7, lw=.8,
                          color=color, linestyle="--" if dashed else "-",
                          shrinkA=0, shrinkB=1, zorder=1))

    def section(self, x, y, label):
        self.text(x, y, label, size=7.5, weight="bold", ha="left")

    def dimensions(self, rows):
        """Add an aligned shape ledger below the flow without crowding its boxes.

        Each row names a stage and its input/output shape. This only defines
        artists; drawing/export remains exclusively in save().
        """
        height = self.ax.get_ylim()[1]
        depth = 1.65 + .43 * len(rows)
        self.ax.set_ylim(-depth, height)
        self.fig.set_size_inches(7.2, 7.2 * (height + depth) / 16)
        self.section(.4, -.1, "Token dimensions | main configurations")
        self.text(.4, -.58,
                  "T: frames; N: padded slots; K: padded pairs/frame; C: object classes (37).",
                  ha="left", size=7)
        self.text(.4, -.98,
                  "Q = 30 free queries; P = HₚWₚ grid cells; L = 4 decoder layers; h = 8 heads (WorldWise++).",
                  ha="left", size=7)
        for i, (stage, shape) in enumerate(rows):
            y = -1.5 - .43 * i
            self.text(.5, y, stage, ha="left", weight="bold", size=7)
            self.text(4.65, y, shape, ha="left", size=7)

    def save(self, name, args):
        # Catch clipped labels at every export, including after wording edits.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        canvas_bounds = self.fig.bbox
        for label in self.ax.texts:
            bounds = label.get_window_extent(renderer)
            if not (canvas_bounds.contains(bounds.x0, bounds.y0)
                    and canvas_bounds.contains(bounds.x1, bounds.y1)):
                raise ValueError(f"Label exceeds the canvas: {label.get_text()!r}")
        for x, y, w, h, *labels in self.text_boxes:
            low = self.ax.transData.transform((x, y))
            high = self.ax.transData.transform((x+w, y+h))
            for label in labels:
                bounds = label.get_window_extent(renderer)
                if (bounds.x0 < low[0] or bounds.x1 > high[0]
                        or bounds.y0 < low[1] or bounds.y1 > high[1]):
                    raise ValueError(f"Label exceeds its box: {label.get_text()!r}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in args.formats:
            target = args.output_dir / f"{name}.{fmt}"
            self.fig.savefig(target, dpi=args.dpi, facecolor="white")
            print(target.resolve())
        plt.close(self.fig)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "outputs" / "paper_figures")
    p.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"),
                   default=["pdf", "svg", "png"])
    p.add_argument("--dpi", type=int, default=600, help="Raster export resolution (default: 600)")
    return p


def worldwise():
    d = Diagram("WorldWise", "Masked world auto-encoding for persistent scene graphs  |  v2e")
    d.section(.4, 9.18, "(a) Geometry-conditioned object recovery")
    d.box(.4, 7.2, 3.0, 1.6, "World scaffold", "Structure: 256; cam: 128\nEgo: 128; motion: 64\nPer slot, per frame", "gray")
    d.box(4, 7.2, 3.2, 1.6, "Scaffold tokenizer", "ROI / [MASK]: 256\nFuse: 832 → 256\nTokens: (T, N, 256)")
    d.box(7.9, 7.2, 3.8, 1.6, "Associative retriever", "Masked queries; visible keys\nView-aware cross-attention\n(T, N, 256) → (T, N, 256)", "teal")
    d.box(12.4, 7.2, 3.2, 1.6, "Object reasoning", "Visibility + spatial attn\n256 → 256 per token\nOutput: (T, N, 256)", "teal")
    for a,b in [(3.4,4),(7.2,7.9),(11.7,12.4)]:
        d.arrow([(a,8),(b,8)])
    d.box(4, 5.75, 3.2, .85, "Decoded ROI features\n1024 → 256 projection", color="gray")
    d.arrow([(5.6,6.6),(5.6,7.2)])
    d.section(.4, 5.15, "(b) Node and relation prediction")
    d.box(.4, 3.1, 3.2, 1.65, "Node / MWAE heads", "Classes: (T, N, C)\nReconstruction:\n(T, N, 256)")
    d.box(4.25, 3.1, 4.45, 1.65, "Relationship predictor", "Union: 1024 → 64; geometry: 8 → 64\nPair fusion: 896 → 256\nRelation attention: (T, K, 256)")
    d.box(9.35, 3.1, 2.8, 1.65, "Temporal edges", "256 → 256 per pair\nOutput: (T, K, 256)", "teal")
    d.box(12.8, 3.1, 2.8, 1.65, "World scene graph", "Attention: (T, K, 3)\nSpatial: (T, K, 6)\nContact: (T, K, 17)", "orange")
    d.arrow([(14,7.2),(14,5.4),(.2,5.4),(.2,3.93),(.4,3.93)])
    d.arrow([(6.5,5.4),(6.5,4.75)])
    d.arrow([(8.7,3.93),(9.35,3.93)])
    d.arrow([(12.15,3.93),(12.8,3.93)])
    d.box(.4, .55, 15.2, 1.85, "(c) Training: clean simulated occlusion", 
          "Artificially mask 30% of visible objects; supervise simulated-unseen pairs with clean labels.\n"
          "Reconstruct masked features against an EMA visual projector; supervise originally visible pairs.\n"
          "Train-time logit adjustment: τ = 0.5   •   No noisy VLM edge supervision: λ_vlm = 0", "purple")
    d.dimensions(shared_dimensions(1024) + [
        ("Associative retrieval", "(T, N, 256) → (T, N, 256); attends along T independently for each slot"),
        ("Visibility / spatial attention", "Add 256-d visibility embedding; (T, N, 256) → (T, N, 256)"),
    ] + output_dimensions())
    return d


def worldwise_plus():
    d = Diagram("WorldWise+", "Replace decoded appearance with frozen foundation-model ROI latents")
    d.section(.4, 9.18, "(a) Appearance replacement at two projector inputs")
    d.box(.4, 6.95, 3.3, 1.85, "Frozen DINOv3-L", "L16 / L20 / L24n\n1024-d tokens per layer\nROI concat: 3072-d", "gray")
    d.box(.4, 4.65, 3.3, 1.85, "Frozen π³", "F4 / F14 / G14\n1024-d tokens per layer\nROI concat: 3072-d", "gray", dashed=True)
    d.box(4.4, 6.95, 3.5, 1.85, "Object + union ROIs", "7 × 7 × 1024 → 1024 / layer\nObjects: (T, N, 3072)\nUnions: (T, K, 3072)")
    d.box(4.4, 4.65, 3.5, 1.85, "Stream configurations", "Headline: DINOv3 only\nAblations: π³ only / fused\nFused input: 6144-d", "orange")
    d.arrow([(3.7,7.87),(4.4,7.87)])
    d.arrow([(3.7,5.57),(4.05,5.57),(4.05,7.3),(4.4,7.3)], dashed=True)
    d.box(8.6, 6.95, 7, 1.85, "Separate object and union projectors", "Input: 3072 (single) / 6144 (fused) → object 256 / union 64\nFused gates per ROI: (2, 256) object / (2, 64) union\nOutputs: (T, N, 256) object / (T, K, 64) union", "teal")
    d.arrow([(7.9,7.87),(8.6,7.87)])
    d.text(12.1, 6.1, "Same reasoning and scene-graph loss", weight="bold")
    d.text(12.1, 5.35, "EMA target copies the new visual projector.\nFrozen backbones; trainable projectors.", size=7.1)
    d.section(.4, 4.12, "(b) Unchanged WorldWise core")
    d.box(.4, 2.25, 3.3, 1.45, "Scaffold tokenizer", "Fuse: 832 → 256\nTokens: (T, N, 256)")
    d.box(4.4, 2.25, 3.5, 1.45, "Recover + reason", "Retriever / spatial attn\n(T, N, 256) at each stage", "teal")
    d.box(8.6, 2.25, 3.5, 1.45, "Relation prediction", "Fusion: 896 → 256\nSpatial / time: (T, K, 256)")
    d.box(12.8, 2.25, 2.8, 1.45, "World graph", "Nodes: (T, N, C)\nEdges: (T, K, 3/6/17)", "orange")
    d.arrow([(10.0,6.95),(10.0,6.65),(8.25,6.65),(8.25,3.93),(2.05,3.93),(2.05,3.7)])
    d.arrow([(14.4,6.95),(14.4,6.72),(15.8,6.72),(15.8,4.42),(10.35,4.42),(10.35,3.7)])
    for a,b in [(3.7,4.4),(7.9,8.6),(12.1,12.8)]:
        d.arrow([(a,2.98),(b,2.98)])
    d.box(.4, .4, 15.2, 1.2, "(c) Inherited training recipe", "30% artificial masking + simulated-unseen supervision + EMA reconstruction\nTrain-time τ = 0.5 logit adjustment; λ_vlm = 0; same slots, pairs and evaluation protocol", "purple")
    d.dimensions([
        ("Backbone / ROI pooling", "Each layer: 1024-d; ROIAlign (7, 7, 1024) → mean (1024); 3 layers → 3072"),
        ("Optional two-stream fusion", "3072 + 3072 = 6144 input; object gates (T, N, 2, 256), union (T, K, 2, 64)"),
    ] + shared_dimensions(3072) + [
        ("Retrieve / visibility / spatial", "Each output: (T, N, 256); the inherited object-token width is unchanged"),
    ] + output_dimensions())
    return d


def worldwise_pp():
    d = Diagram("WorldWise++", "Image-grounded entity decoder with joint detection", height=12.0)
    d.section(.4, 10.36, "(a) Inputs")
    d.section(5.2, 10.36, "(b) Entity decoder × L")
    d.section(11.25, 10.36, "(c) Prediction heads")
    d.box(.4, 7.8, 4.1, 2.15, "N persistent world slots", "ROI: (T, N, 3072) → (T, N, 256)\nScaffold fusion: 832 → 256\nSlots + position: each (T, N, 256)\n[MASK]: 256-d")
    d.box(.4, 6.3, 4.1, 1, "Q free queries", "(Q, 256) → (T, Q, 256); Q = 30", "orange")
    d.box(.4, 2.95, 4.1, 2.8, "Frozen token grids", "DINOv3 L24n + π³ G14\nPer stream: (T, Hₚ, Wₚ, 1024)\nPCA → (T, Hₚ, Wₚ, 256)\n2-way cell gate + 256-d position\nMemory: (T, P, 256)", "gray")
    d.box(5.2, 8.5, 5.25, 1.45, "1  Temporal self-attention", "Slot sequence: (N, T, 256) → (N, T, 256)\nReturn (T, N, 256); free queries bypass", "teal")
    d.box(5.2, 6.55, 5.25, 1.45, "2  Spatial self-attention", "Queries: (T, N+Q, 256) → (T, N+Q, 256)\nWeights / layer: (T, h, N+Q, N+Q)", "teal")
    d.box(5.2, 4.6, 5.25, 1.45, "3  Image cross-attention", "Q: (T, N+Q, 256); K/V: (T, P, 256)\nOutput: (T, N+Q, 256)", "teal")
    d.box(5.2, 2.95, 5.25, 1.15, "4  Feed-forward network", "256 → 1024 → 256; output: (T, N+Q, 256)", "teal")
    for y1,y2 in [(8.5,8),(6.55,6.05),(4.6,4.1)]:
        d.arrow([(7.825,y1),(7.825,y2)])
    d.arrow([(4.5,8.9),(4.85,8.9),(4.85,9.22),(5.2,9.22)])
    d.arrow([(4.5,6.8),(4.85,6.8),(4.85,7.27),(5.2,7.27)])
    d.arrow([(4.5,4.35),(4.85,4.35),(4.85,5.32),(5.2,5.32)])
    d.box(11.25, 8.1, 4.35, 1.85, "World-slot heads", "Class: (T, N, C); recon: (T, N, 256)\nBox residual: (T, N, 4)\nCorner residual: (T, N, 8, 3)")
    d.box(11.25, 5.7, 4.35, 1.85, "Free-query detection", "Class + no-object: (T, Q, C+1)\n2-D boxes: (T, Q, 4)\n3-D OBB corners: (T, Q, 8, 3)", "orange")
    d.box(11.25, 2.95, 4.35, 2.2, "Relation readout", "Weights: 2Lh = 64; product: 256\nPair concat: 320 → 64\nRelation fusion: 896 → 256\nTemporal output: (T, K, 256)")
    d.arrow([(10.45,3.45),(10.85,3.45),(10.85,9.02),(11.25,9.02)])
    d.arrow([(10.85,6.62),(11.25,6.62)])
    d.arrow([(10.85,3.8),(11.25,3.8)])
    d.arrow([(10.45,7.27),(10.65,7.27),(10.65,4.6),(11.25,4.6)], dashed=True)
    d.text(7.825, 2.49, "Spatial weights replace union-box ROI features.", size=7.0)
    d.box(.4, .45, 15.2, 1.55, "(d) Joint objective", "WorldWise scene-graph / simulated-unseen / EMA losses (τ = 0.5; λ_vlm = 0)\n+ matched free-query detection: class CE + 5 × box L1 + corner L1\n+ visible-slot refinement: box L1 + corner L1   |   λ_det = λ_slot = 1", "purple")
    d.dimensions(shared_dimensions(3072, union=False) + [
        ("Grid alignment / PCA", "Native grids: 1024-d; align to (Hₚ, Wₚ); per-stream PCA 1024 → 256"),
        ("Grid fusion / position", "Each stream: 256 → 256; gate (T, P, 2); memory + PE: (T, P, 256)"),
        ("Decoder queries", "(T, N+30, 256); h = 8, head width = 32; FFN 256 → 1024 → 256"),
        ("All-layer attention readout", "(T, L, h, N+Q, N+Q) → pair weights (T, K, 2Lh = 64)"),
        ("Pair readout projection", "Weights 64 + query product 256 = 320 → 64; output (T, K, 64)"),
        ("Slot / free-query split", "After L layers + output norm: slots (T, N, 256); free (T, Q, 256)"),
        ("Slot refinement", "256 → 256 → 4 box residuals; 256 → 256 → 24 corner residuals (= 8 × 3)"),
        ("Free-query class / box", "256 → C+1 logits; box MLP 256 → 256 → 4; sigmoid coordinates"),
        ("Free-query 3-D OBB", "Query 256 + box 4 → 512 → dims 3 / rotation 2 / depth 1 / offset 2 → 8 × 3"),
    ] + output_dimensions())
    return d


def shared_dimensions(roi_width, union=True):
    """Widths verified against the main configs and ScaffoldTokenizer."""
    rows = [
        ("Geometry / camera inputs", "Corners (T, N, 8, 3); camera poses (T, 4, 4); velocity / acceleration (T, N, 3)"),
        ("Scaffold encoders", "Structure (T, N, 256); camera (T, N, 128); motion (T, N, 64)"),
        ("Ego-motion encoder", "(T, 128) → broadcast (T, N, 128)"),
        ("Visual projector / mask", f"ROI (T, N, {roi_width}) → (T, N, 256); learned mask (256)"),
        ("Scaffold concatenation", "256 structure + 256 visual + 128 camera + 64 motion + 128 ego = 832 → 256"),
    ]
    if union:
        rows.append(("Union projector", f"(T, K, {roi_width}) → (T, K, 64); fused WorldWise+ input width is 6144" if roi_width == 3072
                     else f"(T, K, {roi_width}) → (T, K, 64)"))
    return rows


def output_dimensions():
    """Shared readouts; relation geometry is enabled in the main configs."""
    return [
        ("Node / reconstruction heads", "Node: 256 → 256 → C; reconstruction: 256 → 256; EMA target (T, N, 256)"),
        ("Text / pair geometry", "Each label: CLIP 512 → 128; pair geometry: 8 → 64"),
        ("Relation concatenation", "256 person + 256 object + 64 union/readout + 128 + 128 text + 64 geometry = 896"),
        ("Relation / temporal attention", "896 → 256; relation self-attention and temporal edge attention each keep (T, K, 256)"),
        ("Predicate heads", "256 → 128 → 3 / 6 / 17; outputs (T, K, 3), (T, K, 6), (T, K, 17)"),
    ]


