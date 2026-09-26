"""Compact processing-unit diagrams for the main paper, one per training-based method.

    python scripts/paper_figures/dark/fig_main_units.py [--only worldwise ...] [--video TM0BV] [--png] [--pdf]
    python scripts/paper_figures/dark/fig_main_units.py --portrait --png --pdf      # two-tier, portrait pages

The long-form supplementary figures (``fig_worldwise*.py``, ``baseline_common.py``)
draw every module with its tensors and real panels.  These are their summaries for
the main paper, laid out as one left-to-right forward pass:

* top row: the video, then the four processing units (1 Observed Objects,
  2 Unobserved Objects, 3 Relationship, 4 Decoders) as tinted enclosures; every
  module is a box in its own column, so each arrow is long enough to read, and an
  arrow that crosses into the next unit names the tensor it carries.  Long-range
  inputs (union features, the enriched tokens read by the node head) run on a bus
  above or below the modules.  WorldWise++'s entity decoder straddles units 1 and 2.
  On the right, the scene graph the method predicts at the key frame where the
  tracked object is out of view (its own ``preds.json``);
* bottom row: real PredCls intermediates of the same video (the dumps of
  ``dump_intermediates.py``), each numbered with the badge of the module that
  produced it.

``--portrait`` folds the same diagram into two tiers for a portrait page: units 1-2 on
top, units 3-4 and the scene graph below.  An edge from the top tier into the bottom
one leaves through a channel right of unit 2 (or straight down) and runs along a
corridor between the tiers, each edge on its own lane; the intermediates take two rows.
Output of that variant: ``paper_figures/main_portrait/<name>.*``.

Output: ``assets/figures/architecture/paper_figures/main/<name>.svg`` (+ ``.png`` with
``--png`` via rasterize.ps1, + a vector ``.pdf`` with ``--pdf`` via svg2pdf.ps1).
The long-form figures in ``paper_figures/dark/`` are not touched.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BG, FRAME_B, KINDS, MUTED, ORANGE, RED, SERIF, TXT, UNIT,  # noqa: E402
                    Canvas, cap)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARCH = REPO / "assets/figures/architecture"
SANS = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"

MW, MH = 150, 56          # module box
CG, UG, PAD = 44, 96, 18  # gap between columns, extra gap between units, unit padding
LP = 78                   # lane pitch
MOD_FONT = 17.5
X_UNITS = 150             # first unit column starts here (video to the left)
XB, UG_B = 90, 130
OUT_W, OUT_H = 170, 70     # the "World Scene Graph" output box        # portrait: bottom-tier origin and unit gap (room for the entry drops and labels)


@dataclass
class Mod:
    id: str
    label: str            # "\n" splits over two lines
    kind: str             # learn | frozen | new | tool
    col: int              # global column
    lane: float           # 0 = middle row
    badge: Optional[int] = None


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    route: str = "late"   # late | early | down | up | bus | side (bus below, then into the left side)
    bus: float = 0.0      # lane of the bus for route="bus"
    col: str = MUTED


@dataclass
class Panel:
    key: str
    caption: str
    badge: int
    aspect: float         # used when the PNG is missing
    src: str = ""         # dump directory override (e.g. worldwise_pp for the token grids)


@dataclass
class Method:
    name: str
    title: str
    tagline: str
    units: List[Tuple[int, int]]          # (first column, last column) of units 1..4
    mods: List[Mod]
    edges: List[Edge]
    panels: List[Panel]
    losses: Dict[int, List[Tuple[str, str]]] = field(default_factory=dict)   # unit index -> [(sub, colour)]
    new_label: str = "New In This Method"
    tool: bool = False


# ---------------------------------------------------------------------------
# method specs
# ---------------------------------------------------------------------------
def baseline(name, title, tagline, spatial=False, motion=False, temporal=False, usg=False, new=()) -> Method:
    k = lambda mid: "new" if mid in new else "learn"          # noqa: E731
    mods = [Mod("det", "Faster R-CNN\n(ResNet-50)", "frozen", 0, -1.0, 1),
            Mod("scaf", "World OBB\nScaffold", "frozen", 0, 0.5, 2),
            Mod("gse", "Structural\nEncoder", k("gse"), 1, 0.5)]
    edges = [Edge("scaf", "gse")]
    enc_lanes = {"gse": 0.5}
    if spatial:
        mods.append(Mod("spa", "Spatial\nEncoder", k("spatial"), 1, 1.5))
        edges.append(Edge("scaf", "spa", route="early"))
        enc_lanes["spa"] = 1.5
    if motion:
        mods.append(Mod("mot", "Motion\nEncoder", k("motion"), 1, 2.5))
        edges.append(Edge("scaf", "mot", route="early"))
    # unit 2: LKS memory and tokenization
    c = 2
    mods.append(Mod("lks", "LKS Buffer\n(Copy + Δ)", "tool", c, -1.0, 3))
    edges.append(Edge("det", "lks", "ROI Features", col=MUTED))
    mods.append(Mod("tok", "LKS\nTokenizer", k("tokenizer"), c + 1, 0.0, 4))
    edges += [Edge("lks", "tok"), Edge("gse", "tok", "Structural Tokens")]
    if spatial:
        edges.append(Edge("spa", "tok", "Camera Feats"))
    last = "tok"
    c += 2
    if motion:
        mods.append(Mod("fus", "Motion\nFusion", k("motion"), c, 0.0))
        edges += [Edge(last, "fus"), Edge("mot", "fus", "Motion Feats")]
        last, c = "fus", c + 1
    if temporal:
        mods.append(Mod("toe", "Temporal Object\nEncoder", k("temporal_obj"), c, 0.0, 5))
        edges.append(Edge(last, "toe"))
        last, c = "toe", c + 1
    u2 = (2, c - 1)
    # unit 3: relationships
    u3s = c
    if usg:
        mods += [Mod("iot", "Object Context\nEncoder", "new", c, 0.0, 6),
                 Mod("pair", "Pair\nFormer", "learn", c + 1, 0.0),
                 Mod("rel", "USG Relation\nDecoder", "new", c + 2, 0.0, 7)]
        edges.append(Edge("iot", "rel", "Object Memory", route="bus", bus=1.0, col=UNIT[2][0]))
    else:
        mods += [Mod("pe3", "3-D Spatial\nPE", k("spatial_pe"), c, 1.1),
                 Mod("iot", "Inter-Object\nTransformer", "learn", c, 0.0, 6),
                 Mod("pair", "Pair\nFormer", "learn", c + 1, 0.0),
                 Mod("rel", "Temporal Edge\nAttention", "learn", c + 2, 0.0, 7)]
        edges.append(Edge("pe3", "iot", route="up"))
    edges += [Edge(last, "iot", "Slot Tokens"), Edge("iot", "pair"), Edge("pair", "rel"),
              Edge("det", "pair", "Union ROI Features", route="bus", bus=-1.9, col=MUTED)]
    c += 3
    u3 = (u3s, c - 1)
    # unit 4: decoders
    mods += [Mod("node", "Node\nPredictor", "learn", c, -1.0),
             Mod("pred", "Predicate\nHeads × 3", "learn", c, 0.0, 8)]
    edges += [Edge("iot", "node", "Enriched Tokens", route="bus", bus=-1.0, col=UNIT[2][0]),
              Edge("rel", "pred", "Relation Tokens")]
    losses = {3: [("vis", RED), ("vlm", RED)]}
    if usg:
        mods.append(Mod("align", "CLIP Alignment\nHead", "new", c, 1.0))
        edges.append(Edge("iot", "align", route="bus", bus=1.0, col=UNIT[2][0]))
        losses[3].append(("align", ORANGE))
    units = [(0, 1), u2, u3, (c, c)]
    panels = [Panel("obb_1", "OBB Corners", 2, 0.57), Panel("appearance_in", "ROI Features, Visible Cells", 1, 2.76),
              Panel("lks_buffer", "LKS Buffer: Copied Cells", 3, 3.8), Panel("staleness", "Staleness Δ", 3, 1.8),
              Panel("tokens_in", "Slot Tokens (PCA)", 4, 2.76)]
    if temporal:
        panels.append(Panel("temporal_obj_attn", "Temporal Object Attn.", 5, 1.27))
    panels += [Panel("inter_object_attn_1", ("Context" if usg else "Inter-Object") + " Attn., t = {t}", 6, 1.6),
               Panel("usg_rel_cross_attn_1" if usg else "temporal_edge_attn",
                     "Pair × Object Cross-Attn." if usg else "Edge Attn., Person → {obj}", 7, 1.9 if usg else 1.19)]
    return Method(name, title, tagline, units, mods, edges, panels, losses, tool=True,
                  new_label="USG-Par Relation Stack" if usg else "New Over The Tier Below")


def worldwise_family(name) -> Method:
    mods: List[Mod] = []
    edges: List[Edge] = []
    if name == "worldwise":
        title, tagline = "WorldWise", "Masked World Auto-Encoding: Retrieve Instead Of Copy"
        mods += [Mod("app", "Mono-3D\nDetector", "frozen", 0, -1.0, 1),
                 Mod("vp", "Visual\nProjector", "learn", 1, -1.0)]
        edges += [Edge("app", "vp", "ROI")]
        new_label = "New Over The LKS Baselines"
    else:
        title, tagline = "WorldWise+", "Frozen Foundation-Model Latents At The Appearance Seam"
        mods += [Mod("app", "DINOv3-L + π³\n(Frozen)", "frozen", 0, -1.0, 1),
                 Mod("vp", "ROI Gated\nFusion", "new", 1, -1.0, 2)]
        edges += [Edge("app", "vp", "Latents")]
        new_label = "New Over WorldWise"
    retr_kind = "new" if name == "worldwise" else "learn"
    mods += [Mod("scaf", "Pose + OBB\nScaffold", "frozen", 0, 0.6, 3),
             Mod("geo", "Geometry\nEncoders", "learn", 1, 0.6),
             Mod("tok", "Scaffold\nTokenizer", "learn", 2, -0.2, 4),
             Mod("ret", "Associative\nRetriever", retr_kind, 3, -0.2, 5),
             Mod("vis", "+ Visibility\nEmbedding", "learn", 4, -0.2, 6),
             Mod("rec", "EMA\nReconstruction", retr_kind, 4, 1.0, 7),
             Mod("iot", "Inter-Object\nTransformer", "learn", 5, -0.2, 8),
             Mod("pair", "Pair\nEncoder", "learn", 6, -0.2),
             Mod("tea", "Temporal Edge\nAttention", "learn", 7, -0.2),
             Mod("node", "Node\nHead", "learn", 8, -1.2),
             Mod("pred", "Predicate\nHeads × 3", "learn", 8, -0.2, 9)]
    edges += [Edge("scaf", "geo"), Edge("vp", "tok"), Edge("geo", "tok"),
              Edge("tok", "ret", "Tokens + [MASK]"), Edge("ret", "vis"), Edge("vis", "rec", route="down"),
              Edge("vis", "iot", "Completed Tokens"), Edge("iot", "pair"), Edge("pair", "tea"),
              Edge("iot", "node", "Enriched Tokens", route="bus", bus=-1.2, col=UNIT[2][0]),
              Edge("tea", "pred", "Relation Tokens"),
              Edge("app", "pair", "Union Appearance", route="bus", bus=-2.1, col=MUTED)]
    panels = []
    if name == "worldwise_plus":
        panels.append(Panel("grid_dino_1", "DINOv3 Grid", 1, 0.57, src="worldwise_pp"))
        panels.append(Panel("appearance_in", "Tier-1 Tokens (PCA)", 2, 2.76))
    else:
        panels.append(Panel("frame_1", "2-D Boxes, t = {t}", 1, 0.57))
    panels += [Panel("obb_1", "OBB Corners", 3, 0.57), Panel("visibility", "Tokens: Visible / [MASK]", 4, 2.76),
               Panel("retriever_attn", "Retriever Attn., {obj}", 5, 1.06),
               Panel("tokens_out", "Completed Tokens (PCA)", 6, 2.76),
               Panel("recon_sim", "Reconstruction vs Target", 7, 2.67),
               Panel("tokens_enriched", "Enriched Tokens (PCA)", 8, 2.76)]
    return Method(name, title, tagline, [(0, 2), (3, 4), (5, 7), (8, 8)], mods, edges, panels,
                  {1: [("recon", ORANGE)], 3: [("SG", RED), ("sim", RED)]}, new_label=new_label)


def worldwise_pp() -> Method:
    mods = [Mod("app", "DINOv3-L + π³\nGrids (Frozen)", "frozen", 0, -1.0, 1),
            Mod("fus", "Token-Grid\nFusion", "new", 1, -1.0, 2),
            Mod("scaf", "Pose + OBB\nScaffold", "frozen", 0, 0.6, 3),
            Mod("geo", "Geometry\nEncoders", "learn", 1, 0.6),
            Mod("tok", "Scaffold\nTokenizer", "learn", 2, 0.6, 4),
            Mod("dec", "Entity\nDecoder × 4", "new", 3, -0.2, 5),
            Mod("vis", "+ Visibility\nEmbedding", "learn", 4, -0.2, 6),
            Mod("rec", "EMA\nReconstruction", "learn", 4, 1.0, 7),
            Mod("read", "Pair\nReadout", "new", 5, -0.2, 8),
            Mod("pair", "Pair\nEncoder", "learn", 6, -0.2),
            Mod("tea", "Temporal Edge\nAttention", "learn", 7, -0.2),
            Mod("node", "Node\nHead", "learn", 8, -1.2),
            Mod("pred", "Predicate\nHeads × 3", "learn", 8, -0.2, 9),
            Mod("box", "Box\nRefinement", "new", 8, 0.8, 10),
            Mod("det", "Detection\nHeads", "new", 8, 1.8)]
    edges = [Edge("app", "fus"), Edge("scaf", "geo"), Edge("geo", "tok"),
             Edge("fus", "dec", "Memory Mₜ"), Edge("tok", "dec"),
             Edge("dec", "vis", "Slots"), Edge("vis", "rec", route="down"),
             Edge("dec", "read", "Spatial Attn.", route="bus", bus=-1.2, col=UNIT[2][0]),
             Edge("vis", "read", "Completed Slots"), Edge("read", "pair"), Edge("pair", "tea"),
             Edge("vis", "node", route="bus", bus=-2.0, col=UNIT[2][0]),
             Edge("tea", "pred", "Relation Tokens"),
             Edge("dec", "det", "Free Queries", route="bus", bus=2.95, col=ORANGE),
             Edge("dec", "box", "Decoded Slots", route="side", bus=2.6, col=UNIT[0][0])]
    panels = [Panel("memory_1", "Memory Mₜ", 2, 0.57), Panel("visibility", "Slots: Visible / [MASK]", 4, 2.76),
              Panel("temporal_attn", "(a) Temporal Attn.", 5, 1.15), Panel("spatial_attn", "(b) Spatial Attn.", 5, 1.24),
              Panel("cross_attn_1", "(c) Cross-Attn.", 5, 0.57), Panel("tokens_out", "Decoded Slots (PCA)", 6, 2.76),
              Panel("recon_sim", "Reconstruction vs Target", 7, 2.67), Panel("det_1", "Refined Slot Boxes", 10, 0.57)]
    return Method("worldwise_pp", "WorldWise++", "Image-Grounded Entity Decoder With Joint Detection",
                  [(0, 3), (3, 4), (5, 7), (8, 8)], mods, edges, panels,
                  {1: [("recon", ORANGE)], 3: [("SG", RED), ("det", ORANGE), ("slot", ORANGE)]},
                  new_label="New Over WorldWise+")


METHODS: Dict[str, Method] = {m.name: m for m in [
    baseline("w_sttran", "W-STTran", "Last-Known-State Memory And 3-D Spatial Attention",
             new={"gse", "tokenizer", "spatial_pe"}),
    baseline("w_sttran_pp", "W-STTran++", "Adds Camera-Relative Object Features", spatial=True, new={"spatial"}),
    baseline("w_dsgdetr", "W-DSGDetr", "Adds Per-Slot Temporal Object Encoding", spatial=True, temporal=True,
             new={"temporal_obj"}),
    baseline("w_dsgdetr_pp", "W-DSGDetr++", "Adds World-Frame Object Motion", spatial=True, temporal=True, motion=True,
             new={"motion"}),
    baseline("w_usg", "W-USG", "The USG-Par Relation Stack On The LKS Substrate", usg=True),
    worldwise_family("worldwise"), worldwise_family("worldwise_plus"), worldwise_pp(),
]}


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
class Layout:
    def __init__(self, m: Method):
        self.m = m
        starts = sorted({u[0] for u in m.units})
        # an extra unit gap before every unit that starts after the previous one ends
        self.gaps = {}
        ends = {u[1] for u in m.units}
        for g in range(0, max(u[1] for u in m.units) + 1):
            self.gaps[g] = sum(1 for s in starts if 0 < s <= g and (s - 1) in ends)
        lanes = [md.lane for md in m.mods] + [e.bus for e in m.edges if e.route in ("bus", "side")]
        self.lmin, self.lmax = min(lanes), max(lanes)
        self.top = 56
        self.y0 = self.top + 58 + MH / 2 - self.lmin * LP + 4     # lane-0 centre
        self.bottom = self.y(self.lmax) + MH / 2 + 44           # unit enclosures end here

    cur = 0                      # tier whose lanes y() refers to (portrait only)
    cross: List["Edge"] = []     # edges from the top tier into the bottom one (portrait only)
    left_cols: set = set()

    def tier(self, col: int) -> int:
        return 0

    def unit_y(self, k: int) -> Tuple[float, float]:
        return self.top, self.bottom

    def x(self, col: int) -> float:
        return X_UNITS + col * (MW + CG) + self.gaps.get(col, 0) * UG

    def y(self, lane: float) -> float:
        return self.y0 + lane * LP

    def box(self, md: Mod):
        return self.x(md.col), self.y(md.lane) - MH / 2, MW, MH

    def unit_x(self, k: int) -> Tuple[float, float]:
        a, b = self.m.units[k]
        x0, x1 = self.x(a) - PAD, self.x(b) + MW + PAD
        # WorldWise++: units 1 and 2 share the decoder column; each enclosure reaches its middle
        if k == 0 and self.m.units[1][0] == b:
            x1 = self.x(b) + MW * 0.62
        if k == 1 and self.m.units[0][1] == a:
            x0 = self.x(a) + MW * 0.38
        return x0, x1

    @property
    def width(self) -> float:
        return self.unit_x(3)[1] + 50 + OUT_W + 24


class PortraitLayout(Layout):
    """Two tiers: units 1-2 on top, units 3-4 (and the scene graph) below."""

    def __init__(self, m: Method):
        self.m = m
        self.split = m.units[2][0]
        starts = sorted({u[0] for u in m.units})
        ends = {u[1] for u in m.units}
        self.gaps = {g: sum(1 for s_ in starts if 0 < s_ <= g and (s_ - 1) in ends)
                     for g in range(0, max(u[1] for u in m.units) + 1)}
        mods = {md.id: md for md in m.mods}
        top_tier = lambda col: col < self.split          # noqa: E731
        la = [md.lane for md in m.mods if top_tier(md.col)]
        lb = [md.lane for md in m.mods if not top_tier(md.col)]
        self.cross = []
        for e in m.edges:
            a, b = mods[e.src], mods[e.dst]
            if top_tier(a.col) and not top_tier(b.col):
                self.cross.append(e)
                if e.route in ("bus", "side") and e.bus < a.lane:      # leaves upwards: its bus stays in the top tier
                    la.append(e.bus)
            elif e.route in ("bus", "side"):
                (la if top_tier(a.col) else lb).append(e.bus)
        self.cur = 0
        self.topA = 56
        self.y0A = self.topA + 58 + MH / 2 - min(la) * LP + 4
        self.botA = self.y0A + max(la) * LP + MH / 2 + 44
        self.cor0 = self.botA + 30
        self.topB = self.cor0 + max(len(self.cross) - 1, 0) * 22 + 30
        self.y0B = self.topB + 58 + MH / 2 - min(lb) * LP + 4
        self.botB = self.y0B + max(lb) * LP + MH / 2 + 44
        self.top, self.bottom = self.topA, self.botB
        self.plan_cross(mods)

    def tier(self, col: int) -> int:
        return 0 if col < self.split else 1

    def x(self, col: int) -> float:
        if col < self.split:
            return X_UNITS + col * (MW + CG) + self.gaps.get(col, 0) * UG
        return XB + (col - self.split) * (MW + CG) + (self.gaps.get(col, 0) - self.gaps.get(self.split, 0)) * UG_B

    def y(self, lane: float) -> float:
        return (self.y0A if self.cur == 0 else self.y0B) + lane * LP

    def box(self, md: Mod):
        self.cur = self.tier(md.col)
        return super().box(md)

    def unit_y(self, k: int) -> Tuple[float, float]:
        return (self.topA, self.botA) if k < 2 else (self.topB, self.botB)

    def plan_cross(self, mods: Dict[str, Mod]):
        """Entry side, drop x, corridor lane and channel of every top-to-bottom edge."""
        self.plans = []
        for e in self.cross:
            a, b = mods[e.src], mods[e.dst]
            chain = e.route not in ("bus", "side")
            exit_ = "right" if chain else ("up" if e.bus < a.lane else "down")
            blocked = any(md.col == b.col and md.lane < b.lane for md in mods.values())
            # left entry for chains, for blocked columns and where a top drop would cross a unit header
            entry = "left" if (chain or blocked or b.col == self.split or b.col >= self.m.units[3][0]) else "top"
            self.plans.append(dict(e=e, a=a, b=b, chain=chain, exit=exit_, entry=entry))
        by_col: Dict[int, list] = {}
        for p in self.plans:
            if p["entry"] == "left":
                by_col.setdefault(p["b"].col, []).append(p)
        self.left_cols = set(by_col)
        for col, ps in by_col.items():
            ps.sort(key=lambda p: (p["b"].lane, p["chain"]))     # the inner drop enters highest
            for k, p in enumerate(ps):
                p["xd"] = self.x(col) - 20 - 14 * k
            per_dst: Dict[str, list] = {}
            for p in ps:
                per_dst.setdefault(p["b"].id, []).append(p)
            for group in per_dst.values():
                offs = [0] if len(group) == 1 else [-10 + 20 * i / (len(group) - 1) for i in range(len(group))]
                for p, o in zip(group, offs):
                    p["dy"] = o
        for p in self.plans:
            if p["entry"] == "top":
                p["xd"] = self.x(p["b"].col) + MW / 2 - 8
        self.plans.sort(key=lambda p: p["xd"])
        ch = 0
        for i, p in enumerate(self.plans):
            p["yc"] = self.cor0 + 22 * i
            if p["exit"] in ("right", "up"):
                p["xch"] = self.unit_x(1)[1] + 22 + 14 * ch
                ch += 1
        self.n_channels = ch

    @property
    def width(self) -> float:
        return max(self.unit_x(1)[1] + 22 + 14 * self.n_channels + 24, self.unit_x(3)[1] + 50 + OUT_W + 24)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw_module(c: Canvas, L: Layout, md: Mod):
    x, y, w, h = L.box(md)
    f, s = KINDS[md.kind]
    dash = ' stroke-dasharray="6 3"' if md.kind == "tool" else ""
    c.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{f}" stroke="{s}" stroke-width="2"{dash}/>')
    lines = md.label.split("\n")
    for i, ln in enumerate(lines):
        c.text(x + w / 2, y + h / 2 + 6.5 - (len(lines) - 1) * 10 + i * 20, ln, size=MOD_FONT, fill=TXT,
               anchor="middle", weight="600")
    if md.kind == "frozen":
        c.snow(x + w - 9, y + 9, r=5)
    if md.badge is not None and md.badge in REMAP and SHOW_BADGES:
        badge(c, x + 2, y + 2, REMAP[md.badge])


REMAP: Dict[int, int] = {}      # spec badge -> displayed number (only badges that have a panel)
SHOW_BADGES = True              # off with --no-panels (no intermediates to point at)


def badge(c: Canvas, x, y, n: int, r=10.5):
    c.a(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{TXT}" stroke="{BG}" stroke-width="1.5"/>')
    c.text(x, y + 4.5, str(n), size=12.5, fill=BG, anchor="middle", weight="700")


def draw_edge(c: Canvas, L: Layout, e: Edge, mods: Dict[str, Mod]):
    a, b = mods[e.src], mods[e.dst]
    ax, ay, aw, ah = L.box(a)
    bx, by, bw, bh = L.box(b)
    L.cur = L.tier(a.col)
    sx, sy = ax + aw, ay + ah / 2
    tx, ty = bx, by + bh / 2
    lab_x = lab_y = None
    if e.route == "down":
        path = f"M{ax + aw / 2} {ay + ah} V{by - 1}"
    elif e.route == "up":
        path = f"M{ax + aw / 2} {ay} V{by + bh + 1}"
    elif e.route == "bus":
        # leave through the top / bottom centre of the source, run along the bus lane, enter the
        # destination through its top / bottom centre (or its left side when it sits on the bus)
        yb = L.y(e.bus)
        cxa = ax + aw / 2 + (8 if yb < sy else -8)
        start = f"M{cxa} {ay if yb < sy else ay + ah} V{yb}"
        if abs(ty - yb) < 1:
            path = start + f" H{tx - 1}"
            xe = tx
        else:
            cxb = bx + bw / 2 - 8
            path = start + f" H{cxb} V{(by - 1) if yb < ty else (by + bh + 1)}"
            xe = cxb
        lab_x, lab_y = (cxa + xe) / 2, yb - 7
    elif e.route == "side":
        # down to a bus below, along it, then up the gap in front of the destination into its left side
        yb = L.y(e.bus)
        cxa = ax + aw / 2 - 20
        xm = tx - CG / 2
        path = f"M{cxa} {ay + ah} V{yb - 10} H{xm} V{ty} H{tx - 1}"
        lab_x, lab_y = (cxa + xm) / 2, yb - 17
    elif abs(sy - ty) < 1:
        path = f"M{sx} {sy} H{tx - 1}"
        lab_x, lab_y = (sx + tx) / 2, sy - 8
        if b.col in L.left_cols and e.label:        # portrait: keep clear of the drops entering this column
            lab_x = sx + (len(cap(e.label)) * 7.0 + 10) / 2 + 6
    else:
        xm = (sx + CG / 2) if e.route == "early" else (tx - CG / 2)
        path = f"M{sx} {sy} H{xm} V{ty} H{tx - 1}"
        lab_x, lab_y = ((sx + xm) / 2, sy - 8) if (xm - sx) >= (tx - xm) else ((xm + tx) / 2, ty - 8)
    c.arrow(0, 0, 0, 0, col=e.col, width=2.0, curve=path)
    if e.label and lab_x is not None:
        tw = len(cap(e.label)) * 7.0 + 10
        c.a(f'<rect x="{lab_x - tw / 2}" y="{lab_y - 13}" width="{tw}" height="17" rx="4" fill="{BG}" '
            f'fill-opacity="0.85"/>')
        c.text(lab_x, lab_y, e.label, size=13, fill=e.col if e.col != MUTED else TXT, anchor="middle", weight="600")


NAMES = [("Observed Objects", "Processing Unit"), ("Unobserved Objects", "Processing Unit"),
         ("Relationship", "Processing Unit"), ("Decoders", "Prediction Heads")]


def draw_units(c: Canvas, L: Layout):
    shared = L.m.units[0][1] == L.m.units[1][0]
    for k in range(4):
        x0, x1 = L.unit_x(k)
        y0, y1 = L.unit_y(k)
        col, fill = UNIT[k]
        op = ' fill-opacity="0.8"' if shared and k < 2 else ""
        c.a(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="16" fill="{fill}"{op} '
            f'stroke="{col}" stroke-width="2.2"/>')
    for k in range(4):
        x0, x1 = L.unit_x(k)
        y0, y1 = L.unit_y(k)
        col = UNIT[k][0]
        hx = x0 + 12
        c.a(f'<circle cx="{hx + 13}" cy="{y0 + 22}" r="12.5" fill="{col}"/>')
        c.text(hx + 13, y0 + 27, str(k + 1), size=14, fill=BG, anchor="middle", weight="700")
        c.text(hx + 32, y0 + 28, NAMES[k][0], size=19, fill=col, weight="700", font=SERIF)
        c.text(hx + 32, y0 + 46, NAMES[k][1], size=13.5, fill=MUTED, font=SANS)
        xl = x0 + 16 + (MW * 0.62 if (shared and k == 1) else 0)
        for i, (sub, lc) in enumerate(L.m.losses.get(k, [])):
            c.loss(xl + i * 56, y1 - 14, sub, lc, size=20)


def edge_label(c: Canvas, x, y, text, col):
    tw = len(cap(text)) * 7.0 + 10
    c.a(f'<rect x="{x - tw / 2}" y="{y - 13}" width="{tw}" height="17" rx="4" fill="{BG}" fill-opacity="0.85"/>')
    c.text(x, y, text, size=13, fill=col if col != MUTED else TXT, anchor="middle", weight="600")


def draw_cross(c: Canvas, L: "PortraitLayout"):
    """Top-tier -> bottom-tier edges: out of the source (right side, up to its bus and over to a
    channel right of unit 2, or straight down), along its corridor lane, then down into the
    destination's top or into its left side.  A white halo under each path makes crossings read
    as one line passing over another."""
    downs = [p["xd"] for p in L.plans if p["exit"] == "down"]
    n_down = 0
    for p in L.plans:
        e, a, b = p["e"], p["a"], p["b"]
        ax, ay, aw, ah = L.box(a)
        bx, by, bw, bh = L.box(b)
        L.cur = 0
        if p["exit"] == "right":
            path = f"M{ax + aw} {ay + ah / 2} H{p['xch']} V{p['yc']}"
        elif p["exit"] == "up":
            cxa = ax + aw / 2 + 8
            path = f"M{cxa} {ay} V{L.y(e.bus)} H{p['xch']} V{p['yc']}"
        else:
            cxa = ax + aw / 2 + (-10 if n_down == 0 else 10)
            n_down += 1
            path = f"M{cxa} {ay + ah} V{p['yc']}"
        path += f" H{p['xd']}"
        if p["entry"] == "top":
            path += f" V{by - 1}"
        else:
            path += f" V{by + bh / 2 + p.get('dy', 0)} H{bx - 1}"
        c.a(f'<path d="{path}" fill="none" stroke="{BG}" stroke-width="7" stroke-linejoin="round"/>')
        c.arrow(0, 0, 0, 0, col=e.col, width=2.0, curve=path)
        if e.label:
            tw = len(cap(e.label)) * 7.0 + 10
            lx = (min(downs) - 8 - tw / 2) if p["exit"] == "down" else (p["xd"] + tw / 2 + 10)
            edge_label(c, lx, p["yc"] - 6, e.label, e.col)


def output_box(c: Canvas, x, yc) -> float:
    """The method's output: a plain box standing for the world scene graph."""
    f, s_ = KINDS["frozen"]
    c.a(f'<rect x="{x}" y="{yc - OUT_H / 2}" width="{OUT_W}" height="{OUT_H}" rx="12" fill="{BG}" stroke="{TXT}" '
        f'stroke-width="2.4"/>')
    c.text(x + OUT_W / 2, yc - 4, "World Scene", size=MOD_FONT + 1, fill=TXT, anchor="middle", weight="700")
    c.text(x + OUT_W / 2, yc + 18, "Graph", size=MOD_FONT + 1, fill=TXT, anchor="middle", weight="700")
    return yc + OUT_H / 2


def output_graph(c: Canvas, x, y, w, preds: Optional[dict], key: Optional[str], t: Optional[int]) -> float:
    rows = []
    if preds and key in preds:
        for r in preds[key]:
            # top-1 contacting predicate; when that is "not_contacting", the top-1 spatial one
            con = (r.get("pred_contacting") or ["not_contacting"])[0]
            pred = con if con != "not_contacting" else (r.get("pred_spatial") or [r.get("pred_attention", "")])[0]
            rows.append((r["object"], pred.replace("_", " "), bool(r.get("visible", True))))
    if not rows:
        rows = [("object", "relation", True), ("object", "relation", False)]
    px, py = x + 30, y + 30
    f, s = KINDS["learn"]
    c.a(f'<circle cx="{px}" cy="{py}" r="25" fill="{f}" stroke="{s}" stroke-width="2"/>')
    c.text(px, py + 5, "Person", size=13.5, fill=TXT, anchor="middle", weight="600")
    for i, (obj, pred, vis) in enumerate(rows):
        oy = y + 76 + i * 64
        col = MUTED if vis else ORANGE
        ox = x + 64
        c.arrow(0, 0, 0, 0, col=col, width=1.7, curve=f"M{px} {py + 25} V{oy + 24} H{ox - 1}")
        dash = "" if vis else ' stroke-dasharray="6 3"'
        c.a(f'<rect x="{ox}" y="{oy}" width="{w - 64}" height="48" rx="9" fill="{BG}" stroke="{col}" stroke-width="1.8"{dash}/>')
        c.text(ox + (w - 64) / 2, oy + 20, obj + ("" if vis else " (Unseen)"), size=14, fill=TXT if vis else ORANGE,
               anchor="middle", weight="600")
        c.text(ox + (w - 64) / 2, oy + 39, pred, size=13.5, fill=MUTED if vis else ORANGE, anchor="middle")
    c.text(x + w / 2, y + 76 + len(rows) * 64 + 12, f"Scene Graph, t = {t}" if t else "Scene Graph", size=14,
           fill=MUTED, anchor="middle")
    return y + 76 + len(rows) * 64 + 18


def draw_panels(c: Canvas, L: Layout, W: float, y: float, images: Dict[str, Path], ts, obj, rows: int = 1) -> float:
    items = []
    for p in sorted(L.m.panels, key=lambda q: q.badge):
        items.append((p, p.caption.format(t=ts[1], obj=obj)))
    x0, x1, gap, h0 = 24, W - 24, 18, 150

    def aspect(p):
        size = c.img_size(p.key)
        return min(size[0] / size[1], 3.6) if size else p.aspect
    asp = [aspect(p) for p, _ in items]
    groups = [list(range(len(items)))]
    if rows == 2 and len(items) > 1:          # split where the two rows are closest in total width
        k = min(range(1, len(items)), key=lambda k_: abs(sum(asp[:k_]) - sum(asp) / 2))
        groups = [list(range(k)), list(range(k, len(items)))]
    for g in groups:
        h = h0
        if sum(asp[i] for i in g) * h + gap * (len(g) - 1) > x1 - x0:
            h = (x1 - x0 - gap * (len(g) - 1)) / sum(asp[i] for i in g)
        ws = [asp[i] * h for i in g]
        x = x0 + (x1 - x0 - sum(ws) - gap * (len(g) - 1)) / 2
        wrapped = False
        for i, w in zip(g, ws):
            p, caption = items[i]
            c.image_slot(x, y, w, h, p.key)
            badge(c, x + 2, y + 2, REMAP[p.badge])
            lines = [caption]
            if len(cap(caption)) * 7.2 > w + gap - 4 and " " in caption:      # wrap under a narrow panel
                words = caption.split(" ")
                k = max(1, min(range(1, len(words)), key=lambda i_: abs(len(" ".join(words[:i_])) - len(caption) / 2)))
                lines = [" ".join(words[:k]), " ".join(words[k:])]
                wrapped = True
            for j, ln in enumerate(lines):
                c.text(x, y + h + 17 + j * 16, ln, size=13.5, fill=MUTED)
            x += w + gap
        y += h + 40 + (16 if wrapped and g is not groups[-1] else 0)
    return y


def build(m: Method, images: Dict[str, Path], preds: Optional[dict], meta: dict, portrait: bool = False,
          panels: bool = True) -> Canvas:
    L = PortraitLayout(m) if portrait else Layout(m)
    W = int(L.width)
    global SHOW_BADGES
    SHOW_BADGES = panels
    REMAP.clear()
    REMAP.update({b: i + 1 for i, b in enumerate(sorted({p.badge for p in m.panels}))})
    ks = meta.get("keyframes") or [0, 1, 2]
    ts = [k + 1 for k in ks]
    obj = str(meta.get("tracked_label", "object")).title()
    panel_y = L.bottom + 40
    H = int(panel_y + 2 * (150 + 40) + 50)
    c = Canvas(W, H + 400, images)
    c.text(24, 36, m.title, size=28, fill=TXT, weight="700", font=SERIF)
    c.text(24 + sum(20 if (ch.isupper() or ch in '+-') else 15 for ch in m.title) + 24, 34, m.tagline, size=17, fill=MUTED)
    draw_units(c, L)
    L.cur = 0
    # video
    fw = 70
    vy = L.y(0) - 70
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        _, fh = c.fit(key, w=fw, default=(fw, 120), max_h=130)
        c.image_slot(20 + i * 7, vy - 30 + i * 16, fw, fh, key, border=FRAME_B)
    c.text(20 + fw / 2 + 7, vy + 150, "Video", size=15, fill=MUTED, anchor="middle")
    mods = {md.id: md for md in m.mods}
    firsts = [md for md in m.mods if md.col == 0]
    for md in firsts:
        x, y, w, h = L.box(md)
        c.arrow(0, 0, 0, 0, col=MUTED, width=2.0, curve=f"M{20 + fw + 16} {L.y(0) + 10} H{x - 22} V{y + h / 2} H{x - 1}")
    for e in m.edges:
        if e not in L.cross:
            draw_edge(c, L, e, mods)
    if L.cross:
        draw_cross(c, L)
    for md in m.mods:
        draw_module(c, L, md)
    # output
    ox = L.unit_x(3)[1] + 50
    last = [md for md in m.mods if md.id in ("node", "pred")]
    centres = [L.box(md)[1] + MH / 2 for md in last]
    yc = sum(centres) / len(centres)
    for md in last:
        x, y, w, h = L.box(md)
        c.arrow(0, 0, 0, 0, col=MUTED, width=2.0, curve=f"M{x + w} {y + h / 2} H{ox - 26} V{yc} H{ox - 4}")
    L.cur = L.tier(m.units[3][0])
    panel_y = max(panel_y, output_box(c, ox, yc) + 34)
    # intermediates
    if panels:
        c.text(24, panel_y - 8, "PredCls Intermediates On " + meta.get("video", ""), size=15, fill=TXT, weight="700")
        yb = draw_panels(c, L, W, panel_y + 6, images, ts, obj, rows=2 if portrait else 1)
    else:
        yb = panel_y - 24
    # legend
    ly = yb + 26
    items = [("frozen", "Frozen"), ("learn", "Learnable"), ("new", m.new_label)]
    if m.tool:
        items.append(("tool", "Non-Differentiable"))
    x = 24
    for kind, label in items:
        f, s = KINDS[kind]
        dash = ' stroke-dasharray="4 2"' if kind == "tool" else ""
        c.a(f'<rect x="{x}" y="{ly - 14}" width="28" height="18" rx="4" fill="{f}" stroke="{s}" stroke-width="1.6"{dash}/>')
        if kind == "frozen":
            c.snow(x + 14, ly - 5, r=5)
        c.text(x + 36, ly, label, size=15, fill=MUTED)
        x += 36 + len(cap(label)) * 8.2 + 28
    c.text(x, ly + 1, "ℒ", size=19, fill=RED, font=SERIF, style="italic")
    c.text(x + 18, ly, "Loss Trained In The Unit", size=15, fill=MUTED)
    x += 18 + 24 * 8.2 + 28
    if panels:
        badge(c, x + 10, ly - 5, 1)
        c.text(x + 26, ly, "Module → Its Intermediate Below", size=15, fill=MUTED)
    c.set_height(int(ly + 18))
    return c


def thumb(src: Path, tmp: Path, width: int = 360) -> Path:
    """A downscaled copy of a panel: the thumbnails print small, so full-resolution
    PNGs would only bloat the PDF."""
    if not src.exists():
        return src
    from PIL import Image
    dst = tmp / f"{src.parent.name}_{src.name}"
    with Image.open(src) as im:
        if im.width > width:
            im = im.convert("RGB").resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--video", default="TM0BV")
    ap.add_argument("--out-dir", default=None, help="default: paper_figures/main (main_portrait with --portrait)")
    ap.add_argument("--portrait", action="store_true", help="two-tier layout for portrait pages")
    ap.add_argument("--no-panels", action="store_true", help="architecture only: no intermediates rows (main paper)")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"])
    args = ap.parse_args()
    out = Path(args.out_dir or ARCH / ("paper_figures/main_portrait" if args.portrait else "paper_figures/main"))
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="main_units_"))
    svgs = []
    for name in args.only or list(METHODS):
        m = METHODS[name]
        d = ARCH / "intermediates" / args.video / name
        keys = {"frame_0", "frame_1", "frame_2"} | {p.key for p in m.panels}
        images = {}
        for k in keys:
            src_dir = next((ARCH / "intermediates" / args.video / p.src for p in m.panels if p.key == k and p.src), d)
            images[k] = thumb(src_dir / f"{k}.png", tmp, 240 if k.startswith("frame") else 520)
        preds = json.loads((d / "preds.json").read_text(encoding="utf-8")) if (d / "preds.json").exists() else None
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        c = build(m, images, preds, meta, portrait=args.portrait, panels=not args.no_panels)
        p = c.write(out / f"{name}.svg")
        svgs.append(str(p))
        print(f"wrote {p} ({c.w} x {c.h})" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))
    if args.png:
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "rasterize.ps1")] + svgs, check=True)
    if args.pdf:
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "svg2pdf.ps1"), "-OutDir", str(out)]
                       + svgs, check=True)


if __name__ == "__main__":
    main()
