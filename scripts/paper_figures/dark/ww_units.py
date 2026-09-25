"""Units 2-4 of the WorldWise and WorldWise+ figures, drawn as processing units.

WorldWise and WorldWise+ differ only in their appearance source (Unit 1): the
detector's decoded 1024-d ROI vector against gated fusions of frozen
foundation-model tokens.  Units 2 (associative retrieval + reconstruction),
3 (inter-object transformer, pair encoder, temporal edge attention) and 4 (node
and predicate heads, losses) are the same modules, so both figure scripts draw
them with these functions; a :class:`Seams` value names what differs (the
appearance tensor, the union input and the EMA target).  Every module is drawn
in full in both figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from common import (DIM, GREEN, LAT, MUTED, ORANGE, RED, TXT, UNIT, Canvas)


@dataclass
class Seams:
    appearance: str             # name of the per-object appearance input, e.g. "ROI Features"
    appearance_sub: str         # its sub-line on the Unit 2 port
    ema_sub: str                # what the EMA target projector copies
    union: str                  # name of the union input of the pair encoder
    union_sub: str
    union_projector: Optional[str]   # sub-line of the union projector box, None if already projected in Unit 1
    union_col: str


def keyframe_labels(meta: dict):
    """(t of the three key frames, 1-indexed; tracked label) from a dump's meta.json."""
    ks = meta.get("keyframes") or [0, 1, 2]
    return [k + 1 for k in ks], meta.get("tracked_label", "object")


def unit2(c: Canvas, y0: float, s: Seams, meta: dict) -> float:
    ts, tracked = keyframe_labels(meta)
    y = c.begin_unit(y0, 2, role="Recovers The Tokens Of Objects That Are Out Of View At A Frame")
    ry = y + 8
    c.unit_port(30, ry + 22, 150, 1, "Object Tokens sₜ,ₙ", "From Unit 1: Visible + [MASK]", h=40)
    c.unit_port(30, ry + 74, 150, 1, "Camera Features", "From Unit 1 Spatial Encoder")
    c.unit_port(30, ry + 124, 150, 1, "Visibility Mask", "From Unit 1, T × N")
    c.flow(180, ry + 42, 214, ry + 42, "")
    c.flow(180, ry + 93, 214, ry + 76, "")
    c.box(216, ry, 180, 90, "Associative Retriever", "Per-Object Cross-Attention", kind="learn",
          sub2="Masked Frames ← Own Visible Frames", tsize=11.5, ssize=9)
    c.arrow(180, ry + 136, 250, ry + 92, curve=f"M180 {ry + 136} H250 V{ry + 92}")
    c.text(256, ry + 132, "Key Mask", size=8, fill=MUTED)
    c.text(262, ry + 104, "View-Aware Q / K Bias From Camera Features (128 → 256)", size=8.5, fill=MUTED)
    c.text(262, ry + 116, "Keys / Values: Naturally Visible Frames Only; 2 Layers, 4 Heads", size=8.5, fill=MUTED)
    c.flow(396, ry + 45, 424, ry + 45, "")
    c.tensor(426, ry + 25, 116, 40, "Retrieved Tokens", "T × N × 256", col=LAT[0])
    c.flow(542, ry + 45, 570, ry + 45, "")
    c.box(572, ry + 30, 150, 30, "+ Visibility Embedding", kind="learn", tsize=10)
    c.arrow(180, ry + 150, 647, ry + 62, curve=f"M180 {ry + 150} H647 V{ry + 62}")
    c.text(655, ry + 100, "Observed / Recovered Flag", size=8, fill=MUTED)
    c.text(655, ry + 111, "(Artificial Masks Count As Recovered)", size=8, fill=DIM)
    c.flow(722, ry + 45, 750, ry + 45, "")
    c.tensor(752, ry + 25, 130, 40, "Completed Tokens", "T × N × 256", col=LAT[0])
    c.flow(882, ry + 45, 950, ry + 45, "To Unit 3", col=UNIT[2][0], lsize=9)
    # training-only reconstruction branch
    yb = ry + 196
    c.text(30, yb - 8, "Training Only: Reconstruct The Hidden Appearance Of The Artificially Masked Cells",
           size=9.5, fill=ORANGE, weight="600")
    c.flow(817, ry + 65, 817, yb - 2, "Completed", lsize=8, loff=(8, 30), anchor="start")
    c.box(732, yb, 170, 36, "Reconstruction Projector", "Linear 256 → 256", kind="learn", tsize=10.5, ssize=8.5)
    c.unit_port(30, yb + 44, 150, 1, s.appearance, s.appearance_sub, h=40)
    c.box(300, yb + 40, 210, 48, "EMA Target Projector", s.ema_sub, kind="frozen", frozen=True, tsize=10.5, ssize=8)
    c.flow(180, yb + 64, 298, yb + 64, "")
    lx, lw = 960, 320
    c.loss_node(lx, yb + 18, lw, "rec", "MSE( Projector(Completed), EMA Target )", col=ORANGE,
                note="Artificially Masked Cells; λ = 0.5 × 0.1 = 0.05")
    c.flow(902, yb + 18, lx - 2, yb + 32, "Prediction", col=ORANGE, lsize=8, loff=(0, -6))
    c.elbow(510, yb + 64, lx - 2, yb + 52, col=ORANGE, xm=930)
    c.text(720, yb + 58, "Target", size=8, fill=ORANGE, anchor="middle")
    py = yb + 106
    bottom = c.strip_fit(30, py, 104, [
        ("tokens_in", "Scaffold Tokens (PCA); [MASK] Outlined", 2.76),
        ("retriever_attn", f"Retriever Attention, {tracked.title()} Slot", 1.06),
        ("tokens_out", "Completed Tokens (Same PCA Basis)", 2.76),
        ("train_mask", "Training Pass: Artificial Masks (p = 0.3)", 2.77),
        ("recon_sim", "Reconstruction vs EMA Target (Cosine)", 2.67),
    ])
    c.end_unit(bottom + 6, crop="u2")
    return bottom + 6


def unit3(c: Canvas, y0: float, s: Seams, meta: dict) -> float:
    y = c.begin_unit(y0, 3, role="Contextualises Every Object, Then Every Person-Object Pair Across Time")
    ry = y + 8
    c.unit_port(30, ry + 8, 150, 2, "Completed Tokens", "From Unit 2, T × N × 256", h=40)
    c.unit_port(30, ry + 62, 150, 1, "OBB Corners", "From Unit 1, T × N × 8 × 3")
    c.box(210, ry + 58, 200, 48, "3-D Spatial Positional Encoding", "Distance, Direction, Log-Volume Ratio",
          kind="learn", sub2="5 → 64 → 64, Mean Over Neighbours", tsize=10, ssize=8)
    c.flow(180, ry + 81, 208, ry + 81, "")
    c.box(450, ry, 200, 96, "Inter-Object Transformer", "Self-Attention Over The Objects", kind="learn",
          sub2="Of Each Frame; 3 Layers, 4 Heads", tsize=11.5, ssize=9)
    c.flow(180, ry + 28, 448, ry + 28, "")
    c.flow(410, ry + 82, 448, ry + 72, "+ PE", lsize=8, loff=(0, -6))
    c.flow(650, ry + 48, 678, ry + 48, "")
    c.tensor(680, ry + 28, 150, 40, "Enriched Tokens", "T × N × 256", col=LAT[2])
    c.flow(830, ry + 48, 900, ry + 48, "To Unit 4", col=UNIT[3][0], lsize=9)
    c.text(906, ry + 52, "(Node Head)", size=8.5, fill=UNIT[3][0])
    w, h = c.fit("tokens_enriched", h=104, default=(287, 104), max_w=300)
    c.image_slot(1290 - w, ry - 4, w, h, "tokens_enriched", caption="Enriched Tokens (PCA)", caption_size=9)
    # pair encoder
    yb = ry + 142
    c.unit_port(30, yb, 150, 1, s.union, s.union_sub)
    if s.union_projector:
        c.box(210, yb, 170, 38, "Union Projector", s.union_projector, kind="learn", tsize=10, ssize=8)
        c.flow(180, yb + 19, 208, yb + 19, "", col=s.union_col)
        c.flow(380, yb + 19, 588, yb + 30, "", col=s.union_col)
    else:
        c.flow(180, yb + 19, 588, yb + 30, "Already 64-d (Unit 1 Gated Fusion)", col=s.union_col, lsize=8,
               loff=(0, -6), lpos=0.4)
    c.unit_port(30, yb + 48, 150, 1, "Pair Geometry", "From Unit 1 OBB Corners, 8-D")
    c.box(210, yb + 48, 170, 38, "Geometry Projector", "Linear 8 → 64", kind="learn", tsize=10, ssize=8)
    c.flow(180, yb + 67, 208, yb + 67, "")
    c.flow(380, yb + 67, 588, yb + 58, "")
    c.unit_port(30, yb + 96, 150, 4, "Object Classes", "From Unit 4; GT In PredCls")
    c.box(210, yb + 96, 170, 38, "CLIP Text Embeddings", "37 × 512", kind="frozen", frozen=True, tsize=10, ssize=8)
    c.flow(180, yb + 115, 208, yb + 115, "")
    c.box(410, yb + 96, 150, 38, "Text Projector", "Linear 512 → 128", kind="learn", tsize=10, ssize=8)
    c.flow(380, yb + 115, 408, yb + 115, "")
    c.flow(560, yb + 115, 588, yb + 86, "")
    c.box(590, yb + 4, 230, 104, "Pair Encoder", "Person ⊕ Object ⊕ Union ⊕ Textₚ ⊕ Textₒ ⊕ Geo", kind="learn",
          sub2="896 → 256; 2 Self-Attn Layers, 4 Heads", tsize=11.5, ssize=8.5)
    c.arrow(755, ry + 68, 705, yb + 2, curve=f"M755 {ry + 68} V{yb - 16} H705 V{yb + 2}")
    c.text(762, ry + 100, "Gather Person And Object", size=8, fill=LAT[2])
    c.text(762, ry + 111, "Tokens By Pair Index", size=8, fill=LAT[2])
    c.flow(820, yb + 56, 848, yb + 56, "")
    c.tensor(850, yb + 36, 110, 40, "Pair Tokens", "T × K × 256", col=MUTED)
    c.flow(960, yb + 56, 988, yb + 56, "")
    c.box(990, yb + 26, 160, 60, "Temporal Edge Attention", "Same Pair Across Frames", kind="learn",
          sub2="Frame PE; 1 Layer, 4 Heads", tsize=10.5, ssize=8.5)
    c.flow(1150, yb + 56, 1168, yb + 56, "")
    c.tensor(1170, yb + 36, 112, 40, "Relation Tokens", "→ Unit 4", col=UNIT[3][0])
    c.text(990, yb + 102, "Within-Frame Self-Attention, Then Along The Pair's Timeline", size=8, fill=DIM)
    bottom = yb + 150
    c.end_unit(bottom, crop="u3")
    return bottom


def unit4(c: Canvas, y0: float, s: Seams, meta: dict, legend) -> float:
    ts, tracked = keyframe_labels(meta)
    y = c.begin_unit(y0, 4, role="Object Classes, Predicate Distributions And The Training Losses")
    ry = y + 8
    c.unit_port(30, ry, 150, 3, "Enriched Tokens", "From Unit 3, T × N × 256", h=40)
    c.flow(180, ry + 20, 208, ry + 20, "")
    c.box(210, ry + 2, 150, 36, "Node Head", "MLP 256 → 256 → 37", kind="learn", tsize=10.5, ssize=8)
    c.flow(360, ry + 20, 388, ry + 20, "")
    c.tensor(390, ry, 120, 40, "Node Logits", "T × N × 37", col=MUTED)
    c.flow(510, ry + 20, 538, ry + 20, "")
    c.tensor(540, ry, 190, 40, "Object Classes", "GT In PredCls, argmax In SGDet", col=MUTED)
    c.flow(730, ry + 20, 790, ry + 20, "", col=UNIT[2][0])
    c.text(796, ry + 18, "Back To Unit 3: CLIP Text Pathway", size=9, fill=UNIT[2][0])
    c.text(796, ry + 31, "CE Node Loss Only In SGDet (GT Classes Are Inputs In PredCls)", size=8, fill=DIM)
    c.unit_port(30, ry + 64, 150, 3, "Relation Tokens", "From Unit 3, T × K × 256", h=40)
    c.flow(180, ry + 84, 208, ry + 84, "")
    c.box(210, ry + 58, 170, 52, "Predicate Heads × 3", "MLP 256 → 128 → C", kind="learn", tsize=10.5, ssize=8.5)
    c.flow(380, ry + 84, 408, ry + 84, "")
    c.tensor(410, ry + 56, 190, 56, "Predicate Distributions", "Attention 3 · Spatial 6 · Contacting 17", col=RED)
    c.text(410, ry + 126, "Softmax · Sigmoid · Sigmoid", size=8, fill=DIM)
    lx, lw = 660, 450
    c.loss_node(lx, ry + 50, lw, "sg", "BCE / CE On Visible Pairs; Logits + τ·log π_c, τ = 0.5",
                note="Truly Unseen Pairs Receive No Edge Supervision (λ_vlm = 0)")
    c.loss_node(lx, ry + 106, lw, "sim", "Same Loss On Artificially Masked Pairs (p_mask = 0.3), Clean GT",
                note="Simulated Occlusion Teaches Recovery Instead Of Noisy VLM Labels")
    c.tensor(1140, ry + 78, 140, 48, "GT Predicates", "Visible Pairs Only", col=GREEN)
    c.flow(600, ry + 78, lx - 2, ry + 72, "", col=RED)
    c.flow(600, ry + 92, lx - 2, ry + 128, "", col=RED)
    c.flow(1140, ry + 94, lx + lw + 2, ry + 72, "", col=GREEN)
    c.flow(1140, ry + 110, lx + lw + 2, ry + 128, "", col=GREEN)
    c.rich(lx, ry + 176, [("Total:  ℒ = ℒ", {"fill": TXT, "serif": True}), ("sg", {"fill": RED, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("sim", {"fill": RED, "sub": True}),
                          (" + 0.05 ℒ", {"fill": TXT, "serif": True}), ("rec", {"fill": ORANGE, "sub": True}),
                          ("  (Unit 2)", {"fill": ORANGE}),
                          ("      Per-Class Log-Prior Offsets Are Train-Time Only", {"fill": DIM})], size=11)
    py = ry + 196
    bottom = c.strip_fit(30, py, 120, [
        ("loss_pairs_1", f"Which Loss Each Pair Receives, t = {ts[1]} (Training Pass)", 3.4),
        ("preds_0", f"Predicates, t = {ts[0]}", 3.15),
        ("preds_1", f"Predicates, t = {ts[1]}; {tracked.title()} Unseen", 3.15),
    ])
    c.end_unit(bottom + 6)
    legend_y = bottom + 36
    legend(legend_y)
    c.crop_mark("u4", y0 - 10, legend_y + 12)
    return legend_y + 22
