"""Shared builder for the adapted-baseline method figures
(``fig_w_sttran.py``, ``fig_w_sttran_pp.py``, ``fig_w_dsgdetr.py``,
``fig_w_dsgdetr_pp.py``, ``fig_w_usg.py``), drawn as the four processing units
of the WorldWise figures:

    Unit 1  Observed Objects Processing Unit     frozen detector, world-frame scaffold, structural /
                                                 spatial / motion encoders
    Unit 2  Unobserved Objects Processing Unit   LKS buffer (zero-order hold) + staleness, LKS tokenizer,
                                                 motion fusion, temporal object encoder
    Unit 3  Relationship Processing Unit         inter-object transformer + 3-D PE (object context
                                                 encoder in W-USG), pair former, temporal edge attention
                                                 (USG relation decoder in W-USG)
    Unit 4  Decoders                             node predictor, predicate heads, bucketed loss
                                                 (+ the CLIP alignment head in W-USG)

Every figure is complete on its own: each unit repeats every module it holds and
colours the component a tier adds over the tier below in the orange "new" kind.
The facts (dimensions, layer counts, loss weights) are read off the model code
in ``lib/supervised/baselines/`` and ``lib/supervised/components.py`` and the
``configs/methods/predcls/w_*_predcls_resnet50.yaml`` configs; the per-video
numbers in the captions and cards come from the dump's ``meta.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUE, DIM, FRAME_B, GREEN, LAT, MUTED, ORANGE, RED, TEAL, TOOL_F, TOOL_S,  # noqa: E402
                    TXT, UNIT, Canvas, image_map)

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
    legend_new: str = "Introduced By This Variant"
    cards: List = field(default_factory=list)   # unused; the unit cards are written from the tier + meta.json

    def kind(self, mid: str) -> str:
        return "new" if mid in self.new else "learn"

    def col(self, mid: str, default=MUTED) -> str:
        return ORANGE if mid in self.new else default

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
# per-video facts from the dump
# ---------------------------------------------------------------------------
@dataclass
class Facts:
    video: str
    ts: List[int]                   # the three key frames, 1-indexed
    tracked: str                    # tracked (occluded) object label, title case
    first_unseen: Optional[int]     # first frame the tracked slot is copied, 1-indexed
    holds: Optional[int]            # the frame whose features it holds, 1-indexed
    max_stale: Optional[int]
    n_copied: int
    n_fog: int
    n_vis: int                      # visible pairs
    n_unseen: int                   # pairs with >= 1 unseen endpoint


def facts(meta: dict, video: str) -> Facts:
    ks = meta.get("keyframes") or [0, 1, 2]
    lks = meta.get("lks", {})
    src = lks.get("tracked_sources") or []
    copied = [(i + 1, s) for i, s in enumerate(src) if isinstance(s, int)]
    first = copied[0][0] if copied else None
    holds = copied[0][1] if copied else None
    stale = max((i - s for i, s in copied), default=None)
    bc = meta.get("loss", {}).get("bucket_counts", {})
    return Facts(video=video, ts=[k + 1 for k in ks], tracked=str(meta.get("tracked_label", "object")).title(),
                 first_unseen=first, holds=holds, max_stale=stale, n_copied=int(lks.get("n_copied", 0)),
                 n_fog=int(lks.get("n_fog", 0)), n_vis=int(bc.get("vis_vis", 0)),
                 n_unseen=int(bc.get("vis_unseen", 0)) + int(bc.get("unseen_unseen", 0)))


# ---------------------------------------------------------------------------
# small drawing helpers on top of Canvas
# ---------------------------------------------------------------------------
def vflow(c: Canvas, x1, y1, x2, y2, ym, label: str = "", col=MUTED, lsize=8.5, loff=(0, -5), anchor="middle",
          dashed=False, lx: Optional[float] = None):
    """A vertical-first elbow (down / up, across, down / up) with a label on the horizontal run."""
    c.arrow(x1, y1, x2, y2, col=col, dashed=dashed, curve=f"M{x1} {y1} V{ym} H{x2} V{y2}")
    if label:
        c.text(((x1 + x2) / 2 if lx is None else lx) + loff[0], ym + loff[1], label, size=lsize, fill=col, anchor=anchor)


def note_panel(c: Canvas, x, y, w, lines: Sequence[str], title: str, lh=13, size=8.5):
    """A rounded outline holding verbatim (monospace) lines, for rules and formulas."""
    h = 22 + lh * len(lines) + 6
    c.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{TOOL_F}" stroke="{TOOL_S}" stroke-width="1.2"/>')
    c.text(x + 10, y + 15, title, size=9.5, fill=TXT, weight="600")
    for i, ln in enumerate(lines):
        c.text(x + 10, y + 30 + i * lh, ln, size=size, fill=MUTED, font=MONO)
    return y + h


# ---------------------------------------------------------------------------
# Unit 1: observed objects
# ---------------------------------------------------------------------------
def unit1(c: Canvas, t: Tier, f: Facts, y0: float) -> float:
    y = c.begin_unit(y0, 1, role="Detector Appearance Where Visible, World-Frame Geometry At Every Frame")
    fy = y + 18
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border=FRAME_B)
    c.text(130, fy + 126, "T Frames", size=10.5, fill=MUTED, anchor="middle")
    c.flow(236, fy + 40, 262, fy + 40, "Video", lsize=8.5, loff=(0, -5))
    c.box(264, fy + 6, 136, 68, "Frozen Detector", "Faster R-CNN, ResNet-50", kind="frozen", frozen=True,
          sub2="ROI Head, 1024-d", tsize=11, ssize=8.5)
    c.flow(236, fy + 80, 262, fy + 132, "")
    c.box(264, fy + 104, 136, 62, "World-Frame Scaffold", "GT OBBs (PredCls); π³ Poses", kind="frozen", frozen=True,
          sub2="(Lifted OBBs In SGDet)", tsize=10.5, ssize=8)
    TX, TW, TH = 440, 150, 36
    c.tensor(TX, fy - 4, TW, TH, "ROI Features", "T × N × 1024 → Unit 2", col=BLUE)
    c.tensor(TX, fy + 42, TW, TH, "Union ROI Features", "T × K × 1024 → Unit 3", col=BLUE)
    c.flow(400, fy + 30, TX - 2, fy + 14, "", col=BLUE)
    c.flow(400, fy + 50, TX - 2, fy + 60, "", col=BLUE)
    c.tensor(TX, fy + 96, TW, TH, "OBB Corners", "T × N × 8 × 3")
    c.tensor(TX, fy + 142, TW, TH, "Visibility Mask", "T × N, In View → Units 2, 4")
    c.flow(400, fy + 124, TX - 2, fy + 114, "")
    c.flow(400, fy + 140, TX - 2, fy + 158, "")
    cam = t.spatial or t.motion
    if cam:
        c.tensor(TX, fy + 188, TW, TH, "Camera Poses", "T × 4 × 4, Camera → World")
        c.flow(400, fy + 156, TX - 2, fy + 202, "")
    # encoders
    EX, EW = 640, 230
    OX, OW = 910, 200
    c.box(EX, fy + 4, EW, 48, "Global Structural Encoder", "8 Centred Corners ⊕ Centre (27-d) → MLP",
          kind=t.kind("gse"), tsize=10.5, ssize=8)
    c.flow(TX + TW, fy + 110, EX - 2, fy + 28, "")
    c.tensor(OX, fy + 8, OW, 40, "Structural Tokens", "T × N × 256 → Unit 2", col=t.col("gse"))
    c.flow(EX + EW, fy + 28, OX - 2, fy + 28, "")
    bottom = fy + 186
    if t.spatial:
        c.box(EX, fy + 66, EW, 48, "Object Spatial Encoder", "log‖c − t‖, View Alignment, Azimuth → MLP",
              kind=t.kind("spatial"), tsize=10.5, ssize=8)
        c.flow(TX + TW, fy + 114, EX - 2, fy + 86, "")
        c.flow(TX + TW, fy + 202, EX - 2, fy + 98, "")
        c.tensor(OX, fy + 70, OW, 40, "Camera-Relative Features", "T × N × 128 → Unit 2", col=t.col("spatial"))
        c.flow(EX + EW, fy + 90, OX - 2, fy + 90, "")
    if t.motion:
        c.box(EX, fy + 128, EW, 48, "Object Motion Encoder", "v = cₜ − cₜ₋₁, ‖v‖, Rᵀv, ‖Rᵀv‖ → MLP",
              kind=t.kind("motion"), tsize=10.5, ssize=8)
        c.flow(TX + TW, fy + 118, EX - 2, fy + 146, "")
        c.flow(TX + TW, fy + 208, EX - 2, fy + 160, "")
        c.tensor(OX, fy + 132, OW, 40, "Motion Features", "T × N × 64 → Unit 2 (Fusion)", col=t.col("motion"))
        c.flow(EX + EW, fy + 152, OX - 2, fy + 152, "")
        c.text(EX, fy + 192, "Gated On valid[t] ∧ valid[t−1]; Frame 0 Gets A Learnable No-Motion Token;", size=8,
               fill=DIM)
        c.text(EX, fy + 203, "The Acceleration Input Is Wired But Always Zero", size=8, fill=DIM)
    if cam:
        bottom = fy + 226
    missing = [n for n, on in (("Camera-Relative", t.spatial), ("Motion", t.motion)) if not on]
    if missing:
        c.text(EX, fy + (126 if not t.spatial else 188), "No " + " Or ".join(missing) + " Encoder In This Tier",
               size=8.5, fill=DIM)
    items = [("frame_1", f"Frame t = {f.ts[1]}", 0.57), ("obb_1", "OBB Corners", 0.57),
             ("scene3d_1", "π³ Scene, OBBs, Camera", 0.96)]
    if t.spatial:
        items.append(("camera_path", "Camera Poses → Spatial Enc.", 1.62))
    if t.motion:
        items.append(("motion", "Object Centres → Motion Enc.", 1.42))
    items += [("appearance_in", "Input ROI Features, Visible Cells (PCA)", 2.76)]
    if not t.temporal_obj:
        items.append(("struct_tokens", "Structural Tokens (PCA)", 2.76))
    if "spatial" in t.new:
        items.append(("spatial_feats", "Camera-Relative Feats", 1.4))
    if t.motion:
        items.append(("motion_feats", "Motion Feats", 1.4))
    out = c.strip_fit(30, bottom + 14, 120, items)
    c.end_unit(out + 6, crop="u1")
    return out + 6


# ---------------------------------------------------------------------------
# Unit 2: unobserved objects (LKS memory and tokenization)
# ---------------------------------------------------------------------------
def unit2(c: Canvas, t: Tier, f: Facts, y0: float) -> float:
    y = c.begin_unit(y0, 2, role="Last-Known-State Memory: An Unseen Cell Copies Its Nearest Visible Frame")
    ry = y + 10
    c.unit_port(30, ry, 170, 1, "ROI Features", "From Unit 1, T × N × 1024")
    c.unit_port(30, ry + 48, 170, 1, "Visibility Mask", "From Unit 1, T × N")
    c.box(240, ry, 230, 86, "LKS Buffer · Zero-Order Hold", "Copy The Features Of The Nearest", kind="tool",
          sub2="Visible Frame, Past Or Future · No Gradient", tsize=11, ssize=8.5)
    c.flow(200, ry + 19, 238, ry + 30, "", col=BLUE)
    c.flow(200, ry + 67, 238, ry + 56, "")
    c.tensor(510, ry, 180, 38, "Buffered Features", "T × N × 1024", col=TEAL)
    c.tensor(510, ry + 48, 180, 38, "Staleness Δ", "T × N; 1000 If Never Seen", col=TEAL)
    c.flow(470, ry + 30, 508, ry + 19, "", col=TEAL)
    c.flow(470, ry + 56, 508, ry + 67, "", col=TEAL)
    c.text(240, ry + 100, "Never-Seen Slots: Zero Features, Δ = 1000 (Fog Of War)", size=8, fill=DIM)
    note_panel(c, 1050, ry - 4, 232, [
        "m[t,n] = f[t,n]        visible at t",
        "m[t,n] = f[t*,n]       t* = nearest visible",
        "m[t,n] = 0, D = 1000   never seen",
        "D[t,n] = |t - t*|  ->  log(D + 1)",
        "Chosen with forward + reverse cummax",
    ], "Memory Rule Per Slot n, Frame t")
    c.text(1050, ry + 112, "The Same Stale Vector Is Re-Used Until The", size=8.5, fill=MUTED)
    c.text(1050, ry + 124, "Object Is Seen Again; Nothing Is Recovered.", size=8.5, fill=MUTED)
    # tokenizer
    yb = ry + 134
    c.unit_port(30, yb, 170, 1, "Structural Tokens", "From Unit 1, T × N × 256")
    if t.spatial:
        c.unit_port(30, yb + 48, 170, 1, "Camera-Relative Features", "From Unit 1, T × N × 128")
    TKX, TKW, TKH = 740, 190, 96
    d_in = 256 + 1024 + t.d_cam + 1
    c.box(TKX, yb - 6, TKW, TKH, "LKS Tokenizer", "Concat → Linear → LN → GELU", kind=t.kind("tokenizer"),
          sub2=f"{d_in} → 256 Per Slot", tsize=11.5, ssize=8.5)
    c.flow(690, ry + 19, TKX - 2, yb + 10, "", col=TEAL)
    c.flow(690, ry + 67, TKX - 2, yb + 26, "log(Δ + 1)", col=TEAL, lsize=8, loff=(12, -2), lpos=0.5, anchor="start")
    c.flow(200, yb + 19, TKX - 2, yb + 44, "", width=1.3)
    if t.spatial:
        c.flow(200, yb + 67, TKX - 2, yb + 62, "", width=1.3)
    c.text(TKX, yb + TKH + 6, ("Camera Slice: 0-d (No Camera Encoder)" if t.d_cam == 0
                               else "Camera Slice: 128-d From The Spatial Encoder"), size=8, fill=DIM)
    c.flow(TKX + TKW, yb + 42, 968, yb + 42, "")
    later = t.motion or t.temporal_obj
    c.tensor(970, yb + 22, 150, 40, "Slot Tokens", "T × N × 256" + ("" if later else " → Unit 3"), col=BLUE)
    if not later:
        c.flow(1120, yb + 42, 1180, yb + 42, "To Unit 3", col=UNIT[2][0], lsize=9)
        end = yb + 120
    else:
        yc = yb + 150
        x = 480
        vflow(c, 1045, yb + 62, x + 60, yc - (6 if t.motion else 12), yc - 30, "")
        if t.motion:
            c.unit_port(30, yc + 8, 170, 1, "Motion Features", "From Unit 1, T × N × 64")
            c.box(x, yc - 4, 190, 64, "Motion Fusion", "[x ⊕ μ] → Linear → LN → GELU", kind=t.kind("motion"),
                  sub2="320 → 256", tsize=10.5, ssize=8.5)
            c.flow(200, yc + 27, x - 2, yc + 28, "", col=t.col("motion"))
            x += 190 + 36
            c.flow(x - 36, yc + 28, x - 2, yc + 28, "")
        if t.temporal_obj:
            c.box(x, yc - 10, 220, 76, "Temporal Object Encoder", "Per-Slot Self-Attention Across T Frames",
                  kind=t.kind("temporal_obj"), sub2="Learnable PE; 2 Layers", tsize=10.5, ssize=8.5)
            c.text(x, yc + 80, "The Slot Is The Track: No Hungarian Matching", size=8, fill=DIM)
            x += 220 + 36
            c.flow(x - 36, yc + 28, x - 2, yc + 28, "")
        c.tensor(x, yc + 8, 150, 40, "Memory Tokens", "T × N × 256 → Unit 3", col=BLUE)
        c.flow(x + 150, yc + 28, x + 200, yc + 28, "To Unit 3", col=UNIT[2][0], lsize=9)
        end = yc + 96
    unseen = (f"{f.tracked} Unseen From t = {f.first_unseen}" if f.first_unseen else "Slot Visibility")
    items = [("visibility", f"Slot Visibility; {unseen}", 2.76),
             ("lks_buffer", f"LKS Buffer: {f.n_copied} Copied Cells, {f.n_fog} Fog"
                            + (f"; {f.tracked} Holds t = {f.holds}" if f.holds else ""), 2.2),
             ("staleness", f"Staleness Δ (Log)" + (f": {f.tracked} → {f.max_stale}" if f.max_stale else ""), 2.2),
             ("tokens_in", "Slot Tokens (PCA); Copies Outlined", 2.76)]
    if t.temporal_obj:
        items += [("temporal_obj_attn", f"Temporal Attn, {f.tracked}", 1.0),
                  ("tokens_temporal", "After Temporal Enc. (PCA)", 2.76)]
    out = c.strip_fit(30, end, 110, items)
    c.end_unit(out + 6, crop="u2")
    return out + 6


# ---------------------------------------------------------------------------
# Unit 3: relationships
# ---------------------------------------------------------------------------
def unit3(c: Canvas, t: Tier, f: Facts, y0: float) -> float:
    role = ("Per-Frame Object Context, Then USG Pair Queries Over The Frame's Objects" if t.usg
            else "Contextualises The Slots Of Each Frame, Then Every Person-Object Pair Across Time")
    y = c.begin_unit(y0, 3, role=role)
    ry = y + 8
    c.unit_port(30, ry + 8, 170, 2, "Memory Tokens", "From Unit 2, T × N × 256", h=40)
    if t.usg:
        c.box(470, ry, 210, 96, "Object Context Encoder", "Plain Per-Frame Transformer,", kind=t.kind("context"),
              sub2="No 3-D PE; 3 Layers, 4 Heads", tsize=11.5, ssize=9)
        c.flow(200, ry + 28, 468, ry + 40, "")
        c.text(470, ry + 112, "Replaces The Inter-Object Transformer And Its 3-D PE", size=8, fill=DIM)
        ex = 690
    else:
        c.unit_port(30, ry + 62, 170, 1, "OBB Corners", "From Unit 1, T × N × 8 × 3")
        c.box(230, ry + 58, 200, 48, "3-D Spatial Positional Encoding", "Distance, Direction, Log-Volume Ratio",
              kind=t.kind("spatial_pe"), sub2="5 → 64 → 64, Mean Over Neighbours", tsize=10, ssize=8)
        c.flow(200, ry + 81, 228, ry + 81, "")
        c.box(470, ry, 200, 96, "Inter-Object Transformer", "Self-Attention Over The Slots", kind="learn",
              sub2="Of Each Frame; 3 Layers, 4 Heads", tsize=11.5, ssize=9)
        c.flow(200, ry + 28, 468, ry + 28, "")
        c.flow(430, ry + 82, 468, ry + 72, "+ PE", lsize=8, loff=(0, -6))
        ex = 680
    c.flow(ex, ry + 48, 708, ry + 48, "")
    c.tensor(710, ry + 28, 150, 40, "Enriched Tokens", "T × N × 256", col=LAT[2])
    c.flow(860, ry + 48, 930, ry + 48, "To Unit 4", col=UNIT[3][0], lsize=9)
    c.text(936, ry + 52, "(Node Predictor" + (", Alignment Head)" if t.usg else ")"), size=8.5, fill=UNIT[3][0])
    # pair former
    yb = ry + 150
    c.unit_port(30, yb, 170, 1, "Union ROI Features", "From Unit 1, T × K × 1024")
    c.box(230, yb, 170, 38, "Union Projector", "Linear 1024 → 64, ReLU, LN", kind="learn", tsize=10, ssize=8)
    c.flow(200, yb + 19, 228, yb + 19, "", col=BLUE)
    c.flow(400, yb + 19, 618, yb + 30, "", col=BLUE)
    c.unit_port(30, yb + 50, 170, 4, "Object Classes", "From Unit 4; GT In PredCls")
    c.box(230, yb + 50, 170, 38, "CLIP Text Embeddings", "37 × 512", kind="frozen", frozen=True, tsize=10, ssize=8)
    c.flow(200, yb + 69, 228, yb + 69, "")
    c.box(430, yb + 50, 150, 38, "Text Projector", "Linear 512 → 128", kind="learn", tsize=10, ssize=8)
    c.flow(400, yb + 69, 428, yb + 69, "")
    c.flow(580, yb + 69, 618, yb + 62, "")
    c.box(620, yb - 6, 220, 96, "Pair Former", "Person ⊕ Object ⊕ Union ⊕ Textₚ ⊕ Textₒ", kind="learn",
          sub2="832 → 256; 2 Self-Attn Layers, 4 Heads", tsize=11.5, ssize=8.5)
    c.arrow(740, ry + 68, 700, yb - 8, curve=f"M740 {ry + 68} V{yb - 26} H700 V{yb - 8}")
    c.text(746, ry + 86, "Gather Person / Object", size=8, fill=LAT[2])
    c.text(746, ry + 97, "By Pair Index", size=8, fill=LAT[2])
    c.flow(840, yb + 42, 868, yb + 42, "")
    c.tensor(870, yb + 22, 104, 40, "Pair Tokens", "T × K × 256", col=MUTED)
    c.flow(974, yb + 42, 998, yb + 42, "")
    if t.usg:
        c.box(1000, yb + 8, 166, 68, "USG Relation Decoder", "Pair Queries Self-Attend, Then", kind=t.kind("usg_decoder"),
              sub2="Cross-Attend To Objects; 2 Layers", tsize=10.5, ssize=8)
        c.arrow(850, ry + 68, 1083, yb + 6, col=LAT[2], curve=f"M850 {ry + 68} V{yb - 40} H1083 V{yb + 6}")
        c.text(1090, yb - 44, "Memory: The Frame's Enriched Object Tokens", size=8, fill=LAT[2], anchor="end")
        c.text(1000, yb + 92, "Replaces Temporal Edge Attention: Relations Are Per Frame", size=8, fill=DIM)
    else:
        c.box(1000, yb + 12, 166, 60, "Temporal Edge Attention", "Same (Person, Object) Pair", kind="learn",
              sub2="Across Frames; Learnable PE; 1 Layer", tsize=10.5, ssize=8)
    c.flow(1166, yb + 42, 1180, yb + 42, "")
    c.tensor(1182, yb + 22, 102, 40, "Relation Tokens", "→ Unit 4", col=UNIT[3][0])
    items = [("inter_object_attn_1", ("Context Attn" if t.usg else "Inter-Object Attn") + f", t = {f.ts[1]}", 1.0),
             ("tokens_out", "Enriched Tokens (PCA)", 2.76)]
    if t.usg:
        items += [("usg_rel_self_attn_1", f"Pair Self-Attn, t = {f.ts[1]}", 1.0),
                  ("usg_rel_cross_attn_1", f"Pair × Object Cross-Attn, t = {f.ts[1]}", 1.2)]
    else:
        items += [("rel_attn_1", f"Pair Self-Attn, t = {f.ts[1]}", 1.0),
                  ("temporal_edge_attn", f"Edge Attn, Person → {f.tracked}", 1.4)]
    out = c.strip_fit(30, yb + 110, 120, items)
    c.end_unit(out + 6, crop="u3")
    return out + 6


# ---------------------------------------------------------------------------
# Unit 4: decoders and the bucketed loss
# ---------------------------------------------------------------------------
def unit4(c: Canvas, t: Tier, f: Facts, y0: float) -> float:
    y = c.begin_unit(y0, 4, role="Object Classes, Predicate Distributions And The Bucketed Training Loss")
    ry = y + 8
    c.unit_port(30, ry, 170, 3, "Enriched Tokens", "From Unit 3, T × N × 256", h=40)
    c.flow(200, ry + 20, 228, ry + 20, "")
    c.box(230, ry + 2, 150, 36, "Node Predictor", "MLP 256 → 256 → 37", kind="learn", tsize=10.5, ssize=8)
    c.flow(380, ry + 20, 408, ry + 20, "")
    c.tensor(410, ry, 190, 40, "Node Logits", "T × N × 37", col=MUTED)
    c.flow(600, ry + 20, 628, ry + 20, "")
    c.tensor(630, ry, 190, 40, "Object Classes", "GT In PredCls, argmax Else", col=MUTED)
    c.flow(820, ry + 20, 870, ry + 20, "", col=UNIT[2][0])
    c.text(876, ry + 24, "Back To Unit 3: CLIP Text Pathway", size=9, fill=UNIT[2][0])
    c.unit_port(30, ry + 60, 170, 3, "Relation Tokens", "From Unit 3, T × K × 256", h=40)
    c.flow(200, ry + 80, 228, ry + 80, "")
    c.box(230, ry + 58, 150, 44, "Predicate Heads × 3", "MLP 256 → 128 → C", kind="learn", tsize=10.5, ssize=8)
    c.flow(380, ry + 80, 408, ry + 80, "")
    c.tensor(410, ry + 54, 210, 52, "Predicate Distributions", "Attention 3 · Spatial 6 · Contacting 17", col=RED)
    c.text(630, ry + 84, "Softmax · Sigmoid · Sigmoid", size=8, fill=DIM)
    ly = ry + 136
    if t.usg:
        c.unit_port(30, ry + 120, 170, 3, "Enriched Tokens", "From Unit 3", h=40)
        c.flow(200, ry + 140, 228, ry + 140, "", col=LAT[2])
        c.box(230, ry + 120, 150, 40, "Alignment Projector", "Linear 256 → 512, L2-Norm", kind=t.kind("align"),
              tsize=10, ssize=8)
        c.flow(380, ry + 140, 408, ry + 140, "")
        c.tensor(410, ry + 120, 210, 40, "Alignment Logits", "cos(·, CLIP) / 0.07; T × N × 37", col=ORANGE)
        ly = ry + 196
    # buckets and losses
    c.unit_port(30, ly + 30, 150, 1, "Visibility Mask", "From Unit 1, T × N", h=40)
    c.flow(180, ly + 50, 208, ly + 22, "", width=1.2)
    c.flow(180, ly + 52, 208, ly + 74, "", width=1.2)
    c.text(30, ly + 88, "Bucket Each Pair By The Visibility", size=8, fill=DIM)
    c.text(30, ly + 99, "Of Its Person And Its Object", size=8, fill=DIM)
    c.tensor(210, ly, 330, 40, "Visible Pairs", f"Both Endpoints In View · Manual Labels ({f.n_vis} On {f.video})",
             col=GREEN)
    c.tensor(210, ly + 52, 330, 40, "Unseen Pairs",
             f"≥ 1 Endpoint Out Of View · VLM Pseudo-Labels ({f.n_unseen} On {f.video})", col=ORANGE)
    c.tensor(210, ly + 104, 330, 40, "GT Object Classes", "Node Targets; SGDet / SGCls Only", col=GREEN)
    lx, lw = 600, 650
    c.loss_node(lx, ly - 4, lw, "vis", "CE(Attention) + BCE(Spatial) + BCE(Contacting) On Visible Pairs, Weight 1",
                note="Clean Manual Labels; Every Term Summed, Then Divided By N = All Valid Pairs")
    c.loss_node(lx, ly + 50, lw, "vlm", "λ_vlm · [ KL(Smoothed Attention, ε = 0.2) + BCE(Smoothed Spatial / Contacting) ]",
                note="VLM Pseudo-Labels On Vis-Unseen And Unseen-Unseen Pairs; λ_vlm = 0.2, Label Smoothing 0.2, Same N")
    c.loss_node(lx, ly + 104, lw, "node", "CE(Node Logits, GT Class) Over Valid Slots — SGDet / SGCls Only",
                note="In PredCls The GT Classes Are Inputs, So This Term Is Off")
    c.flow(540, ly + 20, lx - 2, ly + 20, "", col=GREEN)
    c.flow(540, ly + 72, lx - 2, ly + 74, "", col=ORANGE)
    c.flow(540, ly + 124, lx - 2, ly + 128, "", col=GREEN)
    bx = 1275
    c.arrow(620, ry + 72, bx, ly + 74, col=RED, head=False, curve=f"M620 {ry + 72} H{bx} V{ly + 74}")
    c.arrow(bx, ly + 16, lx + lw + 1, ly + 16, col=RED)
    c.arrow(bx, ly + 74, lx + lw + 1, ly + 74, col=RED)
    c.text(bx - 5, ry + 66, "Predictions", size=8, fill=RED, anchor="end")
    total_y = ly + 172
    if t.usg:
        c.loss_node(lx, ly + 158, lw, "txt", "λ_align · CE(Alignment Logits, GT Class) Over Valid Slots", col=ORANGE,
                    note="Text-Centric Contrast Of USG-Par; λ_align = 0.1, Temperature 0.07; Object Classes Only")
        c.arrow(620, ry + 140, lx - 2, ly + 178, col=ORANGE, curve=f"M620 {ry + 140} H{lx - 30} V{ly + 178} H{lx - 2}")
        total_y = ly + 226
    parts = [("Total:  ℒ = ( ℒ", {"fill": TXT, "serif": True}), ("vis", {"fill": RED, "sub": True}),
             (" + ℒ", {"fill": TXT, "serif": True}), ("vlm", {"fill": RED, "sub": True}),
             (" ) / N + ℒ", {"fill": TXT, "serif": True}), ("node", {"fill": RED, "sub": True})]
    if t.usg:
        parts += [(" + ℒ", {"fill": TXT, "serif": True}), ("txt", {"fill": ORANGE, "sub": True})]
    parts.append(("      λ_vlm = 0.2 Is The Only Down-Weight; N Is Shared By Every Bucket", {"fill": DIM}))
    c.rich(lx, total_y, parts, size=11)
    c.text(lx, total_y + 16, "No Artificial Masking · No Reconstruction Target · No Logit Adjustment · No Ego-Motion Encoder",
           size=8.5, fill=DIM)
    items = []
    if t.usg:
        items.append(("align_logits", f"cos(Object Token, CLIP Class), t = {f.ts[1]}", 1.6))
    items += [("loss_pairs_1", f"Loss Per Pair, t = {f.ts[1]}: Visible (w = 1), {f.tracked} (w = 0.2)", 3.4),
              ("preds_0", f"Predicates, t = {f.ts[0]}", 3.15),
              ("preds_1", f"Predicates, t = {f.ts[1]}; {f.tracked} Unseen", 3.15)]
    out = c.strip_fit(30, total_y + 30, 110, items)
    c.end_unit(out + 6)
    legend_y = out + 36
    c.legend(legend_y, [("frozen", "Frozen"), ("learn", "Learnable"), ("new", t.legend_new),
                        ("tool", "Non-Differentiable Program")],
             extra_swatches=[(BLUE, "Detector ROI Token"), (TEAL, "LKS Buffer Output"), (LAT[2], "Enriched Token"),
                             (GREEN, "Manual Labels"), (ORANGE, "VLM Pseudo-Labels"), (RED, "Loss")])
    c.crop_mark("u4", y0 - 10, legend_y + 12)
    return legend_y + 22


# ---------------------------------------------------------------------------
# overview, cards and the CLI
# ---------------------------------------------------------------------------
def overview(t: Tier) -> List[List[str]]:
    u1 = ["Frozen Faster R-CNN (ROI)", "World-Frame OBB Scaffold", "Global Structural Encoder"]
    if t.spatial:
        u1.append("Object Spatial Encoder")
    if t.motion:
        u1.append("Object Motion Encoder")
    u2 = ["LKS Buffer (Zero-Order Hold)", "Staleness Counter Δ", "LKS Tokenizer"]
    if t.motion:
        u2.append("Motion Fusion")
    if t.temporal_obj:
        u2.append("Temporal Object Encoder")
    if t.usg:
        u3 = ["Object Context Encoder", "  (No 3-D PE)", "Pair Former (Union, Text)", "USG Relation Decoder"]
        u4 = ["Node Predictor", "Predicate Heads × 3", "CLIP Alignment Head", "Bucketed Visible / VLM Loss"]
    else:
        u3 = ["Inter-Object Transformer", "  With 3-D PE", "Pair Former (Union, Text)", "Temporal Edge Attention"]
        u4 = ["Node Predictor", "Predicate Heads × 3", "Bucketed Visible / VLM Loss"]
    return [u1, u2, u3, u4]


def unit_cards(t: Tier, f: Facts):
    c1 = ["A frozen Faster R-CNN gives 1024-d ROI features",
          "only where an object is in view; the world-frame",
          "scaffold gives OBB corners at every frame, which",
          "the structural encoder turns into 256-d tokens."]
    if t.spatial:
        c1 += ["A spatial encoder adds 128-d camera-relative",
               "features (distance, view alignment, azimuth)."]
    if t.motion:
        c1 += ["A motion encoder adds 64-d world-frame velocity."]
    c2 = ["An unseen slot re-uses the ROI features of its",
          "nearest visible frame, with a staleness counter;",
          "never-seen slots get zeros and Δ = 1000. No",
          "parameters, no gradient."]
    if f.holds:
        c2 += [f"On {f.video} the {f.tracked.lower()} copies frame {f.holds} and Δ",
               f"climbs to {f.max_stale}; the tokenizer fuses it with geometry."]
    if t.temporal_obj:
        c2 += ["A temporal object encoder then attends along",
               "each slot's own frames (the slot is the track)."]
    if t.usg:
        c3 = ["A plain per-frame context encoder (no 3-D PE)",
              "relates the slots; pair queries built from union",
              "appearance and CLIP class text self-attend and",
              "cross-attend to the frame's object tokens.",
              "Relations are per frame: no temporal edge attention."]
    else:
        c3 = ["The slots of a frame attend to one another with a",
              "3-D positional encoding; each person-object pair",
              "joins its union appearance and the CLIP text of",
              "both classes, then temporal edge attention follows",
              "the same pair across frames."]
    c4 = ["A node predictor and three predicate heads read",
          f"the tokens. Visible pairs ({f.n_vis} on {f.video}) get manual",
          f"labels; unseen pairs ({f.n_unseen}) get VLM pseudo-labels at",
          "λ_vlm = 0.2 with label smoothing 0.2. No masking,",
          "reconstruction or logit adjustment."]
    if t.usg:
        c4 += ["An alignment head adds CE to CLIP classes (0.1)."]
    return [("Detect What Is Visible", c1), ("Remember By Copying", c2),
            ("Relate Objects And Pairs" if not t.usg else "The USG-Par Relation Stack", c3),
            ("Read Out And Supervise", c4)]


def build(t: Tier, images: Dict[str, Path], video: str, meta: dict) -> Canvas:
    f = facts(meta, video)
    c = Canvas(1720, 3400, images)
    c.band_title(46, t.title)
    ov_end = c.unit_overview(80, overview(t), right_caption=f"Output: Predicates At t = {f.ts[1]}")
    c.crop_mark("overview", 14, ov_end)
    tops, y = [], ov_end + 10
    for draw in (unit1, unit2, unit3):
        tops.append(y)
        y = draw(c, t, f, y) + 18
    tops.append(y)
    end = unit4(c, t, f, y)
    bottoms = [tops[1] - 18, tops[2] - 18, tops[3] - 18, end - 36]
    for k, (title, lines) in enumerate(unit_cards(t, f)):
        c.unit_card(tops[k], bottoms[k], k + 1, title, lines)
    c.set_height(int(end + 10))
    return c


def run(t: Tier, script: str) -> None:
    ap = argparse.ArgumentParser(description=f"{t.title} method figure")
    ap.add_argument("--images", default=None,
                    help=f"dump directory for the {t.name} cell (assets/figures/architecture/intermediates/<video>/{t.name})")
    ap.add_argument("--video", default="00T1E")
    ap.add_argument("--out", default=str(Path(script).resolve().parents[3]
                                         / f"assets/figures/architecture/paper_figures/dark/{t.name}.svg"))
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="white background (light) or the dark hero palette; read by common.py at import")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every text: Title Case, UPPER CASE or as written; read by common.py at import")
    args = ap.parse_args()
    meta = {}
    if args.images and (Path(args.images) / "meta.json").exists():
        meta = json.loads((Path(args.images) / "meta.json").read_text(encoding="utf-8"))
    c = build(t, image_map(args.images, t.images), args.video, meta)
    p = c.write(Path(args.out))
    print(f"wrote {p} ({c.w} x {c.h})" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))
