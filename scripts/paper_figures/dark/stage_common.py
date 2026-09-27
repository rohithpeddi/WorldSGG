"""Shared builder for the adapted-baseline method figures
(``fig_w_sttran.py``, ``fig_w_sttran_pp.py``, ``fig_w_dsgdetr.py``,
``fig_w_dsgdetr_pp.py``, ``fig_w_usg.py``), in the visual language of the
WorldWise hero figures in ``common.py``.

Every figure is complete on its own: the three stages repeat every shared
module (LKS buffer, structural encoder, tokenizer, inter-object transformer,
node predictor, pair former, temporal edge attention, bucketed loss) and colour
the component this tier adds over the tier below in the orange "new" kind.
The facts (dimensions, layer counts, loss weights) are read off the model code
in ``lib/supervised/baselines/`` and ``lib/supervised/components.py`` and the
``configs/methods/predcls/w_*_predcls_resnet50.yaml`` configs.

    Stage 1  Perception, Geometry Scaffold And LKS Memory
    Stage 2  Tokenization And Spatio-Temporal Reasoning
    Stage 3  Relationship Heads And Bucketed Loss
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BG, BLUE, CARD_F, CARD_S, DIM, FRAME_B, GREEN, LAT, MUTED, ORANGE, RED, TEAL, TOOL_F, TOOL_S,  # noqa: E402
                    TXT, Canvas, image_map)

X0, X1 = 30, 1290            # the stage band
CX, CW = 1320, 380           # the card column
MONO = "Consolas, 'Courier New', monospace"


@dataclass
class Tier:
    """What one baseline tier draws."""
    name: str                       # dump directory / method name, e.g. "w_sttran"
    title: str                      # band title
    spatial: bool                   # ObjectSpatialEncoder present
    temporal_obj: bool              # TemporalObjectEncoder present
    motion: bool                    # ObjectMotionEncoder + motion fusion present
    usg: bool                       # W-USG relation stack instead of the ladder's
    new: Set[str]                   # module ids drawn in the orange "new" kind
    cards: List[Tuple[str, List[str]]] = field(default_factory=list)
    legend_new: str = "Introduced By This Variant"

    def kind(self, mid: str) -> str:
        return "new" if mid in self.new else "learn"

    @property
    def d_cam(self) -> int:
        return 128 if self.spatial else 0

    @property
    def images(self) -> List[str]:
        keys = ["frame_0", "frame_1", "frame_2", "visibility", "obb_0", "obb_1", "obb_2", "scene3d_1",
                "lks_buffer", "staleness", "appearance_in", "struct_tokens", "tokens_in",
                "inter_object_attn_0", "inter_object_attn_1", "inter_object_attn_2", "tokens_out",
                "rel_attn_0", "rel_attn_1", "rel_attn_2", "preds_0", "preds_1", "preds_2",
                "loss_pairs_0", "loss_pairs_1", "loss_pairs_2"]
        if self.spatial:
            keys += ["camera_path", "spatial_feats"]
        if self.motion:
            keys += ["motion", "motion_feats"]
        if self.temporal_obj:
            keys += ["temporal_obj_attn", "tokens_temporal"]
        if self.usg:
            keys += ["usg_rel_self_attn_1", "usg_rel_cross_attn_1", "align_logits"]
        else:
            keys += ["temporal_edge_attn"]
        return keys


# ---------------------------------------------------------------------------
# small drawing helpers on top of Canvas
# ---------------------------------------------------------------------------
def vflow(c: Canvas, x1, y1, x2, y2, ym, label: str = "", col=MUTED, lsize=8.5, loff=(0, -5), anchor="middle",
          dashed=False, lx: Optional[float] = None):
    """A vertical-first elbow (down / up, across, down / up) with a label on the
    horizontal run."""
    c.arrow(x1, y1, x2, y2, col=col, dashed=dashed, curve=f"M{x1} {y1} V{ym} H{x2} V{y2}")
    if label:
        c.text(((x1 + x2) / 2 if lx is None else lx) + loff[0], ym + loff[1], label, size=lsize, fill=col, anchor=anchor)


def strip_fit(c: Canvas, x, y, h, items: Sequence[Tuple[str, str, float]], gap=14, limit=X1, max_aspect=3.4,
              caption_size=9) -> float:
    """A row of image slots of common height that always fits in [x, limit].
    ``items`` are (key, caption, default_aspect); a real PNG supplies its own
    aspect (capped at ``max_aspect``).  When the row would overflow, the height
    is reduced so that it fits.  Returns the y below the captions."""
    def widths(hh):
        ws = []
        for key, _, asp in items:
            size = c.img_size(key)
            a = min(size[0] / size[1], max_aspect) if size else asp
            ws.append(a * hh)
        return ws
    ws = widths(h)
    gaps = gap * (len(items) - 1)
    if sum(ws) + gaps > limit - x:
        h = (limit - x - gaps) / (sum(ws) / h)
        ws = widths(h)
    xx = x
    for (key, cap_, _), w in zip(items, ws):
        c.image_slot(xx, y, w, h, key, caption=cap_, caption_size=caption_size)
        xx += w + gap
    return y + h + 16


def card(c: Canvas, x, y, w, h, n, title, body_lines, size=11.5, tsize=19):
    """``Canvas.card`` with an adjustable title size (long titles)."""
    c.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{CARD_F}" stroke="{CARD_S}" stroke-width="1.5"/>')
    cx, cy = x + 34, y + h / 2 - 8 * len(body_lines) + 14
    c.a(f'<circle cx="{cx}" cy="{cy}" r="13" fill="{ORANGE}"/>')
    c.text(cx, cy + 4.5, str(n), size=13, fill="#1a0f05", anchor="middle", weight="700")
    c.text(x + 62, cy + 5, title, size=tsize, fill=TXT, weight="700")
    c.wrap(x + 62, cy + 30, body_lines, size=size)


def note_panel(c: Canvas, x, y, w, lines: Sequence[str], title: str, lh=13, size=8.5):
    """A rounded outline holding verbatim (monospace) lines, for rules and formulas."""
    h = 22 + lh * len(lines) + 6
    c.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{TOOL_F}" stroke="{TOOL_S}" stroke-width="1.2"/>')
    c.text(x + 10, y + 15, title, size=9.5, fill=TXT, weight="600")
    for i, ln in enumerate(lines):
        c.text(x + 10, y + 30 + i * lh, ln, size=size, fill=MUTED, font=MONO)
    return y + h


# ---------------------------------------------------------------------------
# Stage 1: perception, geometry scaffold and LKS memory
# ---------------------------------------------------------------------------
def stage1(c: Canvas, t: Tier, video: str) -> float:
    y = c.stage(88, "Stage 1 · Perception, Geometry Scaffold And LKS Memory")
    fy = y + 20
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border=FRAME_B)
    c.text(130, fy + 126, "T Frames", size=10.5, fill=MUTED, anchor="middle")
    c.flow(236, fy + 55, 262, fy + 55, "Video", lsize=8.5, loff=(0, -5))
    c.box(264, fy + 22, 136, 68, "Frozen Detector", "Faster R-CNN, ResNet-50", kind="frozen", frozen=True,
          sub2="ROI Head, 1024-d", tsize=11, ssize=8.5)

    # detector outputs and the world-frame scaffold
    TX, TW, TH = 440, 140, 36
    c.tensor(TX, fy - 4, TW, TH, "ROI Features", "T × N × 1024", col=BLUE)
    c.tensor(TX, fy + 42, TW, TH, "Union ROI Features", "T × K × 1024 → Stage 3", col=BLUE)
    c.flow(400, fy + 48, 438, fy + 14, "", col=BLUE)
    c.flow(400, fy + 62, 438, fy + 60, "", col=BLUE)
    c.text(TX, fy + 98, "World-Frame Scaffold", size=8.5, fill=DIM)
    c.tensor(TX, fy + 104, TW, TH, "OBB Corners", "T × N × 8 × 3")
    c.tensor(TX, fy + 150, TW, TH, "Visibility Mask", "T × N, In Camera View")
    geo_bottom = fy + 186
    if t.spatial or t.motion:
        c.tensor(TX, fy + 196, TW, TH, "Camera Poses", "T × 4 × 4, Camera → World")
        geo_bottom = fy + 232
    c.text(TX, geo_bottom + 14, "GT Boxes In PredCls; The Detector's", size=8, fill=DIM)
    c.text(TX, geo_bottom + 25, "Lifted OBBs In SGDet", size=8, fill=DIM)

    # LKS buffer (non-differentiable program)
    EX, EW = 620, 210
    OX, OW = 870, 170
    c.box(EX, fy + 6, EW, 80, "LKS Buffer · Zero-Order Hold", "Copy The Features Of The Nearest", kind="tool",
          sub2="Visible Frame, Past Or Future · No Gradient", tsize=10.5, ssize=8)
    c.flow(580, fy + 14, EX - 2, fy + 34, "", col=BLUE)
    c.flow(580, fy + 168, EX - 2, fy + 62, "")
    c.tensor(OX, fy + 2, OW, 40, "Buffered Features", "T × N × 1024 → Stage 2", col=TEAL)
    c.tensor(OX, fy + 50, OW, 40, "Staleness Δ", "T × N; 1000 If Never Seen → Stage 2", col=TEAL)
    c.flow(EX + EW, fy + 36, OX - 2, fy + 22, "", col=TEAL)
    c.flow(EX + EW, fy + 56, OX - 2, fy + 70, "", col=TEAL)
    c.text(EX, fy + 100, "Never-Seen Slots: Zero Features, Δ = 1000 (Fog Of War)", size=8, fill=DIM)

    # structural encoder
    c.box(EX, fy + 114, EW, 44, "Global Structural Encoder", "8 Centred Corners ⊕ Centre (27-d) → MLP", kind=t.kind("gse"),
          tsize=10.5, ssize=8)
    c.flow(580, fy + 122, EX - 2, fy + 136, "")
    c.tensor(OX, fy + 118, OW, 36, "Structural Tokens", "T × N × 256 → Stage 2",
             col=ORANGE if "gse" in t.new else MUTED)
    c.flow(EX + EW, fy + 136, OX - 2, fy + 136, "")
    bottom = fy + 158

    if t.spatial:
        c.box(EX, fy + 166, EW, 44, "Object Spatial Encoder", "log‖c − t‖, View Alignment, Azimuth sin / cos → MLP",
              kind=t.kind("spatial"), tsize=10.5, ssize=7.5)
        c.flow(580, fy + 126, EX - 2, fy + 180, "")
        c.flow(580, fy + 214, EX - 2, fy + 196, "")
        c.tensor(OX, fy + 170, OW, 36, "Camera-Relative Features", "T × N × 128 → Stage 2",
                 col=ORANGE if "spatial" in t.new else MUTED)
        c.flow(EX + EW, fy + 188, OX - 2, fy + 188, "")
        bottom = fy + 210
    if t.motion:
        c.box(EX, fy + 218, EW, 44, "Object Motion Encoder", "v = cₜ − cₜ₋₁, ‖v‖, Rᵀv, ‖Rᵀv‖ → MLP",
              kind=t.kind("motion"), tsize=10.5, ssize=8)
        c.flow(580, fy + 130, EX - 2, fy + 232, "")
        c.flow(580, fy + 218, EX - 2, fy + 250, "")
        c.tensor(OX, fy + 222, OW, 36, "Motion Features", "T × N × 64 → Stage 2 (Fusion)",
                 col=ORANGE if "motion" in t.new else MUTED)
        c.flow(EX + EW, fy + 240, OX - 2, fy + 240, "")
        c.text(EX, fy + 276, "Gated On valid[t] ∧ valid[t−1]; Frame 0 Gets A Learnable No-Motion Token; Acceleration Slot = 0",
               size=8, fill=DIM)
        bottom = fy + 280

    # memory rule
    note_panel(c, 1070, fy - 4, 220, [
        "m[t,n] = f[t,n]        visible at t",
        "m[t,n] = f[t*,n]       t* = nearest visible",
        "m[t,n] = 0, D = 1000   never seen",
        "D[t,n] = |t - t*|  ->  log(D + 1)",
        "Chosen with forward + reverse cummax",
    ], "Memory Rule Per Slot n, Frame t")
    c.text(1070, fy + 104, "The Same Stale Vector Is Re-Used Until", size=8.5, fill=MUTED)
    c.text(1070, fy + 116, "The Object Is Seen Again; Nothing Is", size=8.5, fill=MUTED)
    c.text(1070, fy + 128, "Learned Or Recovered Here.", size=8.5, fill=MUTED)

    py = max(bottom, geo_bottom + 30, fy + 200) + 24
    items = [("obb_1", "OBBs, t = 19", 0.5625),
             ("scene3d_1", "π³ Scene, OBBs, Camera", 1.1)]
    if t.spatial:
        items.append(("camera_path", "Camera Poses → Spatial Enc.", 1.1))
    if t.motion:
        items.append(("motion", "Object Centres → Motion Enc.", 1.3))
    items += [("visibility", "Slot Visibility; Picture Unseen From Frame 14", 1.9),
              ("lks_buffer", "LKS Buffer: 13 Copied Cells, 0 Fog; Picture Holds Frame 13", 1.5),
              ("staleness", "Staleness Δ (Log): Picture → 10", 1.5)]
    return strip_fit(c, X0, py, 112, items)


# ---------------------------------------------------------------------------
# Stage 2: tokenization and spatio-temporal reasoning
# ---------------------------------------------------------------------------
def stage2(c: Canvas, t: Tier, y0: float) -> float:
    y = c.stage(y0, "Stage 2 · Tokenization And Spatio-Temporal Reasoning")
    ry = y + 22
    IX, IW, IH = 30, 150, 36
    rows = [("Structural Tokens", "T × N × 256, Stage 1", ORANGE if "gse" in t.new else MUTED),
            ("Buffered Features", "T × N × 1024, Stage 1", TEAL),
            ("log(Δ + 1)", "T × N × 1, Stage 1", TEAL)]
    if t.spatial:
        rows.append(("Camera-Relative Features", "T × N × 128, Stage 1", ORANGE if "spatial" in t.new else MUTED))
    for i, (a, b, col) in enumerate(rows):
        c.tensor(IX, ry + i * 46, IW, IH, a, b, col=col)
    n_tok = len(rows)
    tok_h = 100
    tok_y = ry + (n_tok * 46 - 10) / 2 - tok_h / 2
    tok_y = max(tok_y, ry - 6)
    TKX, TKW = 214, 176
    d_in = 256 + 1024 + t.d_cam + 1
    c.box(TKX, tok_y, TKW, tok_h, "LKS Tokenizer", "Concat → Linear → LN → GELU", kind=t.kind("tokenizer"),
          sub2=f"{d_in} → 256 Per Slot", tsize=11, ssize=8.5)
    for i in range(n_tok):
        c.flow(IX + IW, ry + i * 46 + 18, TKX - 2, tok_y + tok_h / 2, "", width=1.2)
    c.text(TKX, tok_y + tok_h + 14, ("Camera Slice: 0-d (No Camera Encoder)" if t.d_cam == 0
                                     else "Camera Slice: 128-d From The Spatial Encoder"), size=8, fill=DIM)
    if t.motion:
        c.tensor(IX, ry + n_tok * 46, IW, IH, "Motion Features", "T × N × 64, Stage 1",
                 col=ORANGE if "motion" in t.new else MUTED)

    # chain to the right of the tokenizer
    mid = tok_y + tok_h / 2
    x = 420
    gap = 26
    c.flow(TKX + TKW, mid, x - 2, mid, "")
    c.tensor(x, mid - 20, 130, 40, "Slot Tokens", "T × N × 256", col=BLUE)
    x += 130 + gap
    if t.motion:
        c.flow(x - gap, mid, x - 2, mid, "")
        c.box(x, mid - 30, 140, 60, "Motion Fusion", "[x ⊕ μ] → Linear → LN → GELU", kind=t.kind("motion"),
              sub2="320 → 256", tsize=10.5, ssize=8)
        my = ry + n_tok * 46 + 18
        vflow(c, IX + IW, my, x + 70, mid + 32, my, "Motion Features", col=ORANGE if "motion" in t.new else MUTED,
              lsize=8, loff=(0, -5), lx=x + 30)
        x += 140 + gap
    if t.temporal_obj:
        c.flow(x - gap, mid, x - 2, mid, "")
        c.box(x, mid - 38, 160, 76, "Temporal Object Encoder", "Per-Slot Self-Attention Across", kind=t.kind("temporal_obj"),
              sub2="T Frames; Learnable PE; 2 Layers", tsize=10.5, ssize=8)
        c.text(x, mid + 52, "The Slot Is The Track: No Hungarian Matching", size=8, fill=DIM)
        x += 160 + gap
    iot_x = x
    c.flow(x - gap, mid, x - 2, mid, "")
    if t.usg:
        c.box(x, mid - 38, 190, 76, "Object Context Encoder", "Plain Per-Frame Transformer,", kind=t.kind("context"),
              sub2="No 3-D PE; 3 Layers, 4 Heads", tsize=10.5, ssize=8)
        c.text(x, mid + 52, "Replaces The Inter-Object Transformer And Its 3-D PE", size=8, fill=DIM)
        x += 190 + gap
    else:
        c.box(x, mid - 38, 170, 76, "Inter-Object Transformer", "Self-Attention Over The Slots", kind="learn",
              sub2="Of Each Frame; 3 Layers, 4 Heads", tsize=10.5, ssize=8)
        # 3-D pairwise positional encoding below it
        pe_y = mid + 62
        c.box(iot_x, pe_y, 170, 44, "3-D Spatial Positional Encoding", "Pairwise Dist., Direction, log Vol. Ratio",
              kind=t.kind("spatial_pe"), sub2="→ MLP → Mean Over Neighbours", tsize=9.5, ssize=7.5)
        c.tensor(iot_x - 172, pe_y + 4, 140, 36, "OBB Corners", "T × N × 8 × 3, Stage 1")
        c.flow(iot_x - 32, pe_y + 22, iot_x - 2, pe_y + 22, "")
        c.flow(iot_x + 85, pe_y - 2, iot_x + 85, mid + 40, "+ PE", lsize=8, loff=(12, 4), anchor="start")
        x += 170 + gap
    c.flow(x - gap, mid, x - 2, mid, "")
    c.tensor(x, mid - 20, 140, 40, "Enriched Tokens", "T × N × 256 → Stage 3", col=LAT[2])

    py = ry + max(n_tok + (1 if t.motion else 0), 3) * 46 + 8
    py = max(py, mid + 62 + 44 + 20 if not t.usg else mid + 66)
    items = [("appearance_in", "Input ROI Features, Visible Cells (PCA)", 1.9)]
    if not t.temporal_obj:
        items.append(("struct_tokens", "Structural Tokens (PCA)", 1.9))
    if "spatial" in t.new:
        items.append(("spatial_feats", "Camera-Relative Feats", 1.4))
    if t.motion:
        items.append(("motion_feats", "Motion Feats", 1.4))
    items.append(("tokens_in", "Slot Tokens (PCA); Copies Outlined", 1.9))
    if t.temporal_obj:
        items += [("temporal_obj_attn", "Temporal Attn, Picture", 1.0),
                  ("tokens_temporal", "After Temporal Enc. (PCA)", 1.9)]
    items += [("inter_object_attn_1", "Inter-Obj. Attn, t = 19" if not t.usg else "Context Attn, t = 19", 1.0),
              ("tokens_out", "Enriched Tokens (PCA)", 1.9)]
    return strip_fit(c, X0, py, 104, items)


# ---------------------------------------------------------------------------
# Stage 3: relationship heads and bucketed loss
# ---------------------------------------------------------------------------
def stage3(c: Canvas, t: Tier, y0: float) -> float:
    y = c.stage(y0, "Stage 3 · Relationship Heads And Bucketed Loss")
    hy = y + 24
    # row A: node predictor and the text pathway
    c.tensor(30, hy, 150, 40, "Enriched Tokens", "T × N × 256, Stage 2", col=LAT[2])
    c.flow(180, hy + 20, 208, hy + 20, "")
    c.box(210, hy - 2, 140, 44, "Node Predictor", "MLP 256 → 256 → 37", kind="learn", tsize=10.5, ssize=8)
    c.flow(350, hy + 20, 378, hy + 20, "")
    c.tensor(380, hy, 130, 40, "Node Logits", "T × N × 37", col=MUTED)
    c.flow(510, hy + 20, 538, hy + 20, "")
    c.tensor(540, hy, 140, 40, "Object Classes", "GT In PredCls, Else argmax", col=MUTED)
    c.flow(680, hy + 20, 708, hy + 20, "Lookup", lsize=7.5, loff=(0, -5))
    c.box(710, hy - 2, 150, 44, "CLIP Text Embeddings", "37 × 512, Frozen", kind="frozen", frozen=True, tsize=10.5, ssize=8)
    c.flow(860, hy + 20, 888, hy + 20, "")
    c.box(890, hy - 2, 130, 44, "Text Projector", "Linear 512 → 128", kind="learn", tsize=10.5, ssize=8)
    c.flow(1020, hy + 20, 1048, hy + 20, "")
    c.tensor(1050, hy, 140, 40, "Text Features", "T × K × (128 + 128)", col=MUTED)

    # row C: pair former → temporal edge attention / relation decoder → heads
    row_c = hy + 84
    PFX, PFW, PFH = 380, 200, 96
    c.box(PFX, row_c, PFW, PFH, "Pair Former", "Person ⊕ Object ⊕ Union ⊕ Textₚ ⊕ Textₒ", kind="learn",
          sub2="832 → 256; 2 Self-Attn Layers Over K Pairs", tsize=11, ssize=8)
    vflow(c, 110, hy + 40, PFX - 2, row_c + 12, row_c + 12, "Gather Person And Object By Pair Index", col=LAT[2],
          lsize=8, loff=(0, -5), lx=245)
    c.tensor(200, row_c + 26, 150, 40, "Union ROI Features", "T × K × 1024, Stage 1", col=BLUE)
    c.flow(350, row_c + 46, PFX - 2, row_c + 48, "", col=BLUE)
    c.text(200, row_c + 80, "Union: Linear 1024 → 64 → ReLU → LN", size=7.5, fill=DIM)
    c.tensor(200, row_c + 90, 150, 34, "Pair Indices", "T × K (Person, Object)", col=MUTED)
    c.flow(350, row_c + 107, PFX - 2, row_c + 84, "")
    vflow(c, 1120, hy + 40, PFX + 100, row_c - 2, row_c - 16, "Text Features Of Both Classes", lsize=8, loff=(0, -5), lx=760)
    c.flow(PFX + PFW, row_c + 48, 598, row_c + 48, "")
    c.tensor(600, row_c + 28, 130, 40, "Pair Tokens", "T × K × 256", col=MUTED)
    c.flow(730, row_c + 48, 758, row_c + 48, "")
    if t.usg:
        c.box(760, row_c + 18, 180, 60, "USG Relation Decoder", "Pair Queries Self-Attend, Then", kind=t.kind("usg_decoder"),
              sub2="Cross-Attend To Object Tokens; 2 Layers", tsize=10.5, ssize=7.5)
        vflow(c, 45, hy + 40, 850, row_c + 80, row_c + 132, "Memory: The Frame's Enriched Object Tokens", col=LAT[2],
              lsize=8, loff=(0, 12), lx=470)
        c.text(760, row_c + 92, "Replaces Temporal Edge Attention: Per-Frame Relations", size=7.5, fill=DIM)
    else:
        c.box(760, row_c + 18, 180, 60, "Temporal Edge Attention", "Same (Person, Object) Pair Across", kind="learn",
              sub2="Frames; Learnable PE; 1 Layer", tsize=10.5, ssize=8)
    c.flow(940, row_c + 48, 958, row_c + 48, "")
    c.box(960, row_c + 18, 130, 60, "Predicate Heads × 3", "MLP 256 → 128 → C", kind="learn", tsize=10.5, ssize=8)
    c.flow(1090, row_c + 48, 1108, row_c + 48, "")
    c.tensor(1110, row_c + 14, 168, 68, "Predicate Distributions", "Attention 3 · Spatial 6 · Contacting 17", col=RED)
    c.text(1110, row_c + 94, "Softmax · Sigmoid · Sigmoid", size=7.5, fill=DIM)

    # losses: buckets on the left, the loss nodes on the right, a prediction bus down the right edge
    ly = row_c + (152 if t.usg else 132)
    c.tensor(70, ly + 30, 140, 40, "Visibility Mask", "T × N, Stage 1", col=MUTED)
    c.flow(210, ly + 50, 228, ly + 24, "", width=1.2)
    c.flow(210, ly + 52, 228, ly + 72, "", width=1.2)
    c.text(70, ly + 88, "Bucket Each Pair By The Visibility", size=8, fill=DIM)
    c.text(70, ly + 99, "Of Its Person And Its Object", size=8, fill=DIM)
    c.tensor(230, ly, 320, 40, "Visible Pairs", "Both Endpoints In View · Manual Labels", col=GREEN)
    c.tensor(230, ly + 52, 320, 40, "Unseen Pairs", "≥ 1 Endpoint Out Of View · VLM Pseudo-Labels", col=ORANGE)
    c.tensor(230, ly + 104, 320, 40, "GT Object Classes", "Node Targets; SGDet / SGCls Only", col=GREEN)
    lx, lw = 600, 680
    c.loss_node(lx, ly - 4, lw, "vis", "CE(Attention) + BCE(Spatial) + BCE(Contacting) On Visible Pairs, Weight 1",
                note="Clean Manual Labels; Every Term Summed, Then Divided By N = All Valid Pairs")
    c.loss_node(lx, ly + 50, lw, "vlm", "λ_vlm · [ KL(Smoothed Attention, ε = 0.2) + BCE(Smoothed Spatial / Contacting) ]",
                note="VLM Pseudo-Labels On Vis-Unseen And Unseen-Unseen Pairs; λ_vlm = 0.2, Label Smoothing 0.2, Same N")
    c.loss_node(lx, ly + 104, lw, "node", "CE(Node Logits, GT Class) Over Valid Slots — SGDet / SGCls Only",
                note="In PredCls The GT Classes Are Inputs, So This Term Is Off")
    c.flow(550, ly + 20, lx - 2, ly + 20, "", col=GREEN)
    c.flow(550, ly + 72, lx - 2, ly + 74, "", col=ORANGE)
    c.flow(550, ly + 124, lx - 2, ly + 128, "", col=GREEN)
    bx = 1285
    c.arrow(1278, row_c + 48, bx, ly + 74, col=RED, head=False, curve=f"M1278 {row_c + 48} H{bx} V{ly + 74}")
    c.arrow(bx, ly + 16, lx + lw + 1, ly + 16, col=RED)
    c.arrow(bx, ly + 74, lx + lw + 1, ly + 74, col=RED)
    c.text(bx - 5, row_c + 106, "Predictions", size=8, fill=RED, anchor="end")
    total_y = ly + 162
    if t.usg:
        # text-centric alignment head, fed by the same enriched tokens as the decoder memory
        c.arrow(45, row_c + 132, 45, ly + 176, col=LAT[2], head=False)
        c.flow(45, ly + 176, 228, ly + 176, "Enriched Tokens", col=LAT[2], lsize=8, loff=(0, -5))
        c.box(230, ly + 156, 150, 40, "Alignment Projector", "Linear 256 → 512, L2-Norm", kind=t.kind("align"),
              tsize=10, ssize=7.5)
        c.flow(380, ly + 176, 398, ly + 176, "")
        c.tensor(400, ly + 156, 150, 40, "Alignment Logits", "cos(·, CLIP Class) / 0.07; T × N × 37", col=ORANGE)
        c.text(400, ly + 208, "Same Frozen CLIP Class Embeddings", size=7.5, fill=DIM)
        c.loss_node(lx, ly + 158, lw, "txt", "λ_align · CE(Alignment Logits, GT Class) Over Valid Slots", col=ORANGE,
                    note="Text-Centric Contrast Of USG-Par; λ_align = 0.1, Temperature 0.07; Object Classes Only")
        c.flow(550, ly + 176, lx - 2, ly + 180, "", col=ORANGE)
        total_y = ly + 220
    parts = [("Total:  ℒ = ( ℒ", {"fill": TXT, "serif": True}), ("vis", {"fill": RED, "sub": True}),
             (" + ℒ", {"fill": TXT, "serif": True}), ("vlm", {"fill": RED, "sub": True}),
             (" ) / N + ℒ", {"fill": TXT, "serif": True}), ("node", {"fill": RED, "sub": True})]
    if t.usg:
        parts += [(" + ℒ", {"fill": TXT, "serif": True}), ("txt", {"fill": ORANGE, "sub": True})]
    parts.append(("      λ_vlm = 0.2 Is The Only Down-Weight; N Is Shared By Every Bucket", {"fill": DIM}))
    c.rich(lx, total_y, parts, size=11)
    c.text(lx, total_y + 16, "No Artificial Masking · No Reconstruction Target · No Logit Adjustment · No Ego-Motion Encoder",
           size=8.5, fill=DIM)
    py = total_y + 28
    if t.usg:
        items = [("usg_rel_self_attn_1", "Pair Self-Attn, t = 19", 1.0),
                 ("usg_rel_cross_attn_1", "Pair × Object Cross-Attn, t = 19", 1.2),
                 ("align_logits", "cos(Object Token, CLIP Class), t = 19", 1.6)]
    else:
        items = [("rel_attn_1", "Pair Self-Attn, t = 19", 1.0),
                 ("temporal_edge_attn", "Edge Attn, Person → Picture", 1.4)]
    items += [("loss_pairs_1", "Loss Per Pair, t = 19: Three Visible (w = 1), Picture (w = 0.2)", 2.0),
              ("preds_1", "Predicates At t = 19; Picture Unseen", 2.0)]
    return strip_fit(c, X0, py, 104, items)


# ---------------------------------------------------------------------------
# cards, legend and the CLI
# ---------------------------------------------------------------------------
def finish(c: Canvas, t: Tier, y_after_strip: float) -> None:
    legend_y = y_after_strip + 30
    c.legend(legend_y, [("frozen", "Frozen"), ("learn", "Learnable"), ("new", t.legend_new),
                        ("tool", "Non-Differentiable Program")],
             extra_swatches=[(BLUE, "Detector ROI Token"), (TEAL, "LKS Buffer Output"), (LAT[2], "Enriched Token"),
                             (GREEN, "Manual Labels"), (ORANGE, "VLM Pseudo-Labels"), (RED, "Loss")])
    top, bottom, gap = 88, legend_y - 30, 20
    ch = (bottom - top - 2 * gap) / 3
    for i, (title, lines) in enumerate(t.cards):
        card(c, CX, top + i * (ch + gap), CW, ch, i + 1, title, lines)


def build(t: Tier, images: Dict[str, Path], video: str) -> Canvas:
    # a tall scratch canvas; the real height is set once the content is placed
    c = Canvas(1720, 1700, images)
    c.band_title(46, t.title)
    y1 = stage1(c, t, video)
    y2 = stage2(c, t, y1 + 20)
    y3 = stage3(c, t, y2 + 20)
    finish(c, t, y3)
    set_height(c, int(y3 + 30 + 22))
    return c


def set_height(c: Canvas, height: int) -> None:
    """Shrink the scratch canvas to the placed content: rewrite the root
    element (parts[0]) and the background rectangle (parts[2])."""
    c.parts[0] = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                  f'viewBox="0 0 {c.w} {height}" width="{c.w}" height="{height}">')
    c.parts[2] = f'<rect width="{c.w}" height="{height}" fill="{BG}"/>'
    c.h = height


def run(t: Tier, script: str) -> None:
    ap = argparse.ArgumentParser(description=f"{t.title} method figure")
    ap.add_argument("--images", default=None, help=f"dump directory for the {t.name} cell (outputs/intermediates/<video>/{t.name})")
    ap.add_argument("--video", default="12XD3")
    ap.add_argument("--out", default=str(Path(script).resolve().parents[3] / f"outputs/paper_figures/dark/{t.name}.svg"))
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="white background (light) or the dark hero palette; read by common.py at import")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every text: Title Case, UPPER CASE or as written; read by common.py at import")
    args = ap.parse_args()
    c = build(t, image_map(args.images, t.images), args.video)
    p = c.write(Path(args.out))
    print(f"wrote {p} ({c.w} x {c.h})" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))
