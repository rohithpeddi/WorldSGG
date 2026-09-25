"""WorldWise++ method figure, drawn as processing units with real panels:

    Unit 1      Observed Objects Processing Unit     frozen token grids + memory, geometry scaffold,
                                                     encoders, scaffold tokenizer, world slots + free queries
    Units 1 + 2 shared region                        entity decoder x 4 (temporal / spatial self-attention,
                                                     cross-attention to the memory): it refines the visible
                                                     slots and completes the [MASK] slots in the same pass
    Unit 2      Unobserved Objects Processing Unit   visibility embedding, completed slots, reconstruction
    Unit 3      Relationship Processing Unit         pair readout from the decoder's spatial attention,
                                                     pair encoder, temporal edge attention
    Unit 4      Decoders                             node head, predicate heads, detection heads,
                                                     box / corner refinement, losses

Usage::

    python scripts/paper_figures/dark/fig_worldwise_pp.py \
        --images assets/figures/architecture/intermediates/00T1E/worldwise_pp --video 00T1E
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUE, DIM, FRAME_B, GREEN, LAT, MUTED, NEW_F, NEW_S, ORANGE, PURPLE_F, PURPLE_S, RED,  # noqa: E402
                    TEAL, TXT, UNIT, VIOLET, Canvas, GEOMETRY_IMAGES, image_map)
from ww_units import keyframe_labels  # noqa: E402

IMAGES = GEOMETRY_IMAGES + [
    "frame_0", "frame_1", "frame_2", "visibility", "tokens_in", "tokens_out", "appearance_in",
    "grid_dino_0", "grid_dino_1", "grid_dino_2", "grid_pi3_0", "grid_pi3_1", "grid_pi3_2",
    "memory_0", "memory_1", "memory_2", "gate_0", "gate_1", "gate_2",
    "cross_attn_0", "cross_attn_1", "cross_attn_2", "temporal_attn", "spatial_attn",
    "det_0", "det_1", "det_2", "det_match_0", "det_match_1", "det_match_2", "bev_0", "bev_1", "bev_2",
    "preds_0", "preds_1", "preds_2", "train_mask", "recon_sim", "loss_pairs_0", "loss_pairs_1"]

OVERVIEW = [
    ["Frozen DINOv3-L + π³ Grids", "Token-Grid Fusion → Memory", "ROI Tokens → Gated Fusion",
     "Four Geometry / Motion Encoders", "Scaffold Tokenizer ([MASK])", "Q = 30 Free Queries"],
    ["+ Visibility Embedding", "  → Completed Slots", "Reconstruction vs EMA", "  Target (Training Only)"],
    ["Pair Readout From Decoder", "  Spatial Attention", "Pair Encoder (Text, Geometry)",
     "Temporal Edge Attention"],
    ["Node Head", "Predicate Heads × 3", "Detection Heads", "Box / Corner Refinement",
     "Scene-Graph, Detection And", "  Slot Losses"],
]


def unit1(c: Canvas, y0: float) -> float:
    y = c.begin_unit(y0, 1, role="Frozen Token Grids, Geometry Scaffold And The Set Of World Slots")
    fy = y + 18
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border=FRAME_B)
    c.text(130, fy + 126, "T Frames (672 × 378)", size=10.5, fill=MUTED, anchor="middle")
    # row A: the ROI path for the world slots (WorldWise+'s dinov3tok seam)
    c.tensor(264, fy + 6, 130, 40, "Object Boxes", "Same Boxes As WorldWise", col=MUTED)
    c.flow(394, fy + 26, 430, fy + 26, "")
    c.box(432, fy + 4, 150, 44, "ROI-Align (DINOv3)", "L16 · L20 · L24n; 7 × 7 Mean", kind="frozen", frozen=True,
          tsize=10, ssize=8.5)
    c.flow(582, fy + 26, 618, fy + 26, "")
    c.tensor(620, fy + 6, 150, 40, "Tier-1 Tokens", "T × N × 3072", col=VIOLET)
    c.flow(770, fy + 26, 806, fy + 26, "")
    c.box(808, fy + 4, 160, 44, "Gated Fusion Projector", "Linear → ReLU → LN (Single Stream)", kind="learn",
          tsize=10, ssize=8)
    c.flow(968, fy + 26, 1004, fy + 26, "", col=ORANGE)
    c.tensor(1006, fy + 6, 150, 40, "Appearance Tokens", "T × N × 256", col=ORANGE)
    # row B: DINOv3 grids
    c.flow(236, fy + 60, 262, fy + 92, "", col=VIOLET)
    c.box(264, fy + 66, 130, 52, "DINOv3-L", "ViT-L/16, Frozen", kind="frozen", frozen=True, tsize=11.5, ssize=9)
    c.flow(394, fy + 92, 430, fy + 92, "", col=VIOLET)
    c.tensor(432, fy + 70, 150, 44, "Patch Token Grids", "L16 · L20 · L24n; 24 × 42 × 1024", col=VIOLET)
    c.flow(507, fy + 68, 507, fy + 50, "", col=VIOLET)          # the patch tokens are what ROI-Align pools
    c.flow(582, fy + 92, 618, fy + 92, "", col=VIOLET)
    c.box(620, fy + 70, 150, 44, "Resample To π³ Lattice", "L24n; PCA 1024 → 256, Frozen", kind="frozen", frozen=True,
          tsize=10, ssize=8.5)
    c.flow(770, fy + 92, 806, fy + 118, "", col=VIOLET)
    # row C: pi3 grids
    c.flow(236, fy + 90, 262, fy + 158, "", col=TEAL)
    c.box(264, fy + 132, 130, 52, "π³", "Feed-Forward Geometry, Frozen", kind="frozen", frozen=True, tsize=11.5, ssize=8.5)
    c.flow(394, fy + 158, 430, fy + 158, "", col=TEAL)
    c.tensor(432, fy + 136, 150, 44, "G14 Grid", "27 × 48 × 1024, Global Layer", col=TEAL)
    c.flow(582, fy + 158, 618, fy + 158, "", col=TEAL)
    c.box(620, fy + 136, 150, 44, "PCA 1024 → 256", "Fit On Train Tokens, Frozen", kind="frozen", frozen=True, tsize=10,
          ssize=8.5)
    c.flow(770, fy + 158, 806, fy + 132, "", col=TEAL)
    c.box(808, fy + 96, 160, 58, "Token-Grid Fusion", "LN → Linear Per Stream;", kind="new",
          sub2="Per-Cell 2-Way Gate + 2-D Sine PE", tsize=11.5, ssize=8.5)
    c.flow(968, fy + 125, 1004, fy + 125, "")
    c.tensor(1006, fy + 105, 150, 40, "Memory Mₜ", "T × Hp·Wp × 256", col=ORANGE)
    c.text(1006, fy + 160, "→ Entity Decoder Keys / Values", size=8.5, fill=ORANGE)
    # row D: geometry scaffold -> encoders -> scaffold tokenizer -> slots + free queries
    gy = fy + 214
    c.arrow(130, fy + 134, 262, gy + 52, curve=f"M130 {fy + 134} V{gy + 52} H262")
    c.box(264, gy + 22, 140, 60, "Geometry Scaffold", "π³ Camera Poses; GT OBBs", kind="frozen", frozen=True,
          sub2="(Detector OBBs In SGDet)", tsize=11, ssize=8.5)
    TX, TW, TH = 440, 128, 34
    c.tensor(TX, gy, TW, TH, "OBB Corners", "T × N × 8 × 3")
    c.tensor(TX, gy + 44, TW, TH, "Camera Poses", "T × 4 × 4")
    c.tensor(TX, gy + 88, TW, TH, "Visibility Mask", "T × N → Units 2, 4")
    c.flow(404, gy + 40, TX - 2, gy + 17, "")
    c.flow(404, gy + 52, TX - 2, gy + 61, "")
    c.flow(404, gy + 66, TX - 2, gy + 103, "")
    ex, ew, eh = 620, 196, 30
    enc = [("Structural Encoder", "All Corners Of A Frame", gy - 10),
           ("Spatial Encoder", "Camera-Relative Features", gy + 26),
           ("Ego-Motion Encoder", "Relative Poses Over Time", gy + 62),
           ("Motion Encoder", "Velocity / Acceleration (World + Camera)", gy + 98)]
    for t, s, yy in enc:
        c.box(ex, yy, ew, eh, t, s, kind="learn", tsize=10, ssize=8)
    c.flow(568, gy + 14, 618, gy + 5, "")
    c.flow(568, gy + 17, 618, gy + 38, "")
    c.flow(568, gy + 58, 618, gy + 44, "")
    c.flow(568, gy + 61, 618, gy + 77, "")
    c.flow(568, gy + 20, 618, gy + 108, "")
    c.flow(568, gy + 64, 618, gy + 116, "")
    c.box(880, gy + 10, 150, 100, "Scaffold Tokenizer", "Concatenate + Fuse;", kind="learn", sub2="[MASK] If Unseen",
          tsize=11.5, ssize=9)
    for _, _, yy in enc:
        c.flow(816, yy + 15, 878, gy + 60, "", width=1.2)
    c.arrow(1156, fy + 26, 990, gy + 8, curve=f"M1156 {fy + 26} H1230 V{gy - 12} H990 V{gy + 8}")
    c.text(1236, fy + 80, "Appearance", size=8, fill=ORANGE)
    c.arrow(568, gy + 105, 955, gy + 112, curve=f"M568 {gy + 105} V{gy + 146} H955 V{gy + 112}")
    c.text(700, gy + 142, "Visibility: Which Cells Get [MASK]", size=8, fill=MUTED)
    c.flow(1030, gy + 40, 1068, gy + 30, "")
    c.tensor(1070, gy + 8, 170, 40, "N World Slots sₜ,ₙ", "T × N × 256; Visible / [MASK]", col=BLUE)
    c.box(1070, gy + 60, 170, 40, "Q Free Queries qₜ,q", "Learnable, Q = 30", kind="new", tsize=10, ssize=8)
    c.text(1070, gy + 118, "Training: 30 % Of Visible Cells Also Masked", size=8, fill=DIM)
    c.text(1070, gy + 130, "Slot Position: Linear(Structural Token)", size=8, fill=DIM)
    c.flow(1250, gy + 60, 1250, gy + 150, "", col=UNIT[0][0])
    c.text(1244, gy + 146, "To Entity Decoder", size=8.5, fill=UNIT[0][0], anchor="end")
    py = gy + 164
    bottom = c.strip_fit(30, py, 140, [
        ("grid_dino_1", "DINOv3 L24n Grid", 0.57), ("grid_pi3_1", "π³ G14 Grid", 0.57),
        ("gate_1", "Fusion Gate", 0.57), ("memory_1", "Memory Mₜ", 0.57),
        ("appearance_in", "Tier-1 Tokens Per Visible Slot (PCA)", 2.76), ("obb_1", "OBB Corners", 0.57),
        ("visibility", "World Slots: Visible / [MASK]", 2.76),
    ])
    c.end_unit(bottom + 6, crop="u1")
    return bottom + 6


def unit12(c: Canvas, y0: float, meta: dict) -> float:
    ts, tracked = keyframe_labels(meta)
    y = c.begin_unit(y0, (1, 2), role="")
    c.text(1282, y0 + 29, "Entity Decoder: One Pass Over Visible And [MASK] Slots", size=11, fill=MUTED, anchor="end")
    ry = y + 12
    c.unit_port(30, ry, 170, 1, "N World Slots sₜ,ₙ", "From Unit 1: Visible + [MASK]", h=40)
    c.unit_port(30, ry + 54, 170, 1, "Q Free Queries qₜ,q", "From Unit 1, Q = 30", h=40)
    c.unit_port(30, ry + 108, 170, 1, "Memory Mₜ", "From Unit 1, Fused Token Grid", h=40)
    c.text(30, ry + 166, "Slot Position: Linear(Structural Token)", size=8, fill=DIM)
    c.text(30, ry + 178, "Frame Position: Learned Embedding", size=8, fill=DIM)
    dx, dy, dw, dh = 260, ry - 8, 330, 176
    c.a(f'<rect x="{dx}" y="{dy}" width="{dw}" height="{dh}" rx="10" fill="{NEW_F}" stroke="{NEW_S}" stroke-width="1.8"/>')
    c.text(dx + dw / 2, dy + 18, "Entity Decoder × 4 Layers", size=12.5, fill=TXT, anchor="middle", weight="700")
    rows = [("(a) Temporal Self-Attention", "Slot n Across Its Frames", "Permanence"),
            ("(b) Spatial Self-Attention", "Slots + Free Queries Of Frame t", "Weights Kept"),
            ("(c) Cross-Attention", "Queries → Memory Mₜ (Keys / Values)", "Grounding"),
            ("(d) Feed-Forward", "256 → 1024 → 256", "")]
    for i, (t, s, tag) in enumerate(rows):
        yy = dy + 30 + i * 35
        c.a(f'<rect x="{dx + 10}" y="{yy}" width="{dw - 20}" height="29" rx="5" fill="{PURPLE_F}" stroke="{PURPLE_S}" '
            f'stroke-width="1"/>')
        c.text(dx + 18, yy + 12, t, size=9.5, fill=TXT, weight="600")
        c.text(dx + 18, yy + 24, s, size=8, fill=MUTED)
        if tag:
            c.text(dx + dw - 16, yy + 18, tag, size=8, fill=ORANGE, anchor="end")
    c.flow(200, ry + 20, dx - 2, dy + 44, "", col=BLUE)
    c.flow(200, ry + 74, dx - 2, dy + 80, "", col=ORANGE)
    c.flow(200, ry + 128, dx - 2, dy + 116, "", col=ORANGE)
    c.text(dx, dy + dh + 16, "Slots → (a)(b)(c) · Free Queries → (b)(c) · Memory → (c) As Keys / Values", size=8.5,
           fill=MUTED)
    c.text(dx, dy + dh + 28, "Replaces The Associative Retriever And The Inter-Object Transformer", size=8.5, fill=DIM)
    OX, OW = 640, 230
    c.tensor(OX, ry, OW, 40, "Decoded Slots", "T × N × 256 → Units 2, 3, 4", col=BLUE)
    c.tensor(OX, ry + 54, OW, 40, "Decoded Free Queries", "T × Q × 256 → Unit 4 (Detection)", col=ORANGE)
    c.tensor(OX, ry + 108, OW, 40, "Spatial Attention Weights", "From (b): 2 Dirs × 4 Layers × 8 Heads → Unit 3",
             col=LAT[2])
    c.flow(dx + dw, dy + 44, OX - 2, ry + 20, "", col=BLUE)
    c.flow(dx + dw, dy + 80, OX - 2, ry + 74, "", col=ORANGE)
    c.flow(dx + dw, dy + 64, OX - 2, ry + 124, "", col=LAT[2])
    # the three attentions at the right
    x = 884
    for key, cap_, hmax in (("temporal_attn", f"(a) Temporal Attn, {tracked.title()} Slot", 124),
                            ("spatial_attn", f"(b) Spatial Attn, t = {ts[1]}", 124)):
        w, h = c.fit(key, h=hmax, default=(150, hmax), max_w=190)
        c.image_slot(x, ry - 4, w, h, key, caption=cap_, caption_size=9)
        x += w + 12
    w, h = c.fit("cross_attn_1", h=124, default=(71, 124), max_w=90)
    c.image_slot(min(x, 1290 - w), ry - 4, w, h, "cross_attn_1", caption=f"(c) Cross-Attn, t = {ts[1]}", caption_size=9)
    bottom = ry + 214
    c.end_unit(bottom, crop="u12")
    return bottom


def unit2(c: Canvas, y0: float, meta: dict) -> float:
    y = c.begin_unit(y0, 2, role="Marks Which Slots Were Recovered And Trains Their Recovery")
    ry = y + 8
    c.tensor(30, ry + 18, 170, 40, "Decoded Slots", "From Units 1 + 2 (Entity Decoder)", col=BLUE)
    c.unit_port(30, ry + 70, 170, 1, "Visibility Mask", "From Unit 1, T × N")
    c.flow(200, ry + 38, 238, ry + 38, "")
    c.box(240, ry + 23, 170, 30, "+ Visibility Embedding", kind="learn", tsize=10)
    c.arrow(200, ry + 89, 325, ry + 55, curve=f"M200 {ry + 89} H325 V{ry + 55}")
    c.text(333, ry + 76, "Observed / Recovered Flag", size=8, fill=MUTED)
    c.text(333, ry + 87, "(Artificial Masks Count As Recovered)", size=8, fill=DIM)
    c.flow(410, ry + 38, 448, ry + 38, "")
    c.tensor(450, ry + 18, 150, 40, "Completed Slots", "T × N × 256", col=LAT[0])
    c.flow(600, ry + 38, 680, ry + 38, "To Units 3, 4", col=UNIT[2][0], lsize=9)
    c.text(686, ry + 42, "(Pair Encoder, Node Head)", size=8.5, fill=UNIT[2][0])
    yb = ry + 128
    c.text(30, yb - 8, "Training Only: Reconstruct The Hidden Appearance Of The Artificially Masked Cells",
           size=9.5, fill=ORANGE, weight="600")
    c.flow(525, ry + 58, 525, yb - 2, "Completed", lsize=8, loff=(8, 18), anchor="start")
    c.box(440, yb, 170, 36, "Reconstruction Projector", "Linear 256 → 256", kind="learn", tsize=10.5, ssize=8.5)
    c.unit_port(30, yb + 50, 170, 1, "Tier-1 Tokens", "From Unit 1, Before Masking", h=40)
    c.box(240, yb + 46, 250, 48, "EMA Target Projector", "EMA (m = 0.996) Of The Gated Fusion Projector",
          kind="frozen", frozen=True, tsize=10.5, ssize=8)
    c.flow(200, yb + 70, 238, yb + 70, "")
    lx, lw = 900, 380
    c.loss_node(lx, yb + 22, lw, "rec", "MSE( Projector(Completed Slot), EMA Target )", col=ORANGE,
                note="Artificially Masked Cells; λ = 0.5 × 0.1 = 0.05")
    c.flow(610, yb + 18, lx - 2, yb + 36, "Prediction", col=ORANGE, lsize=8, loff=(0, -6))
    c.elbow(490, yb + 70, lx - 2, yb + 56, col=ORANGE, xm=860)
    c.text(700, yb + 64, "Target", size=8, fill=ORANGE, anchor="middle")
    bottom = c.strip_fit(30, yb + 112, 110, [
        ("tokens_in", "Scaffold Slot Tokens (PCA); [MASK] Outlined", 2.76),
        ("tokens_out", "Decoded Slots (Same PCA Basis)", 2.76),
        ("train_mask", "Training Pass: Artificial Masks (p = 0.3)", 2.77),
        ("recon_sim", "Reconstruction vs EMA Target (Cosine)", 2.67),
    ])
    c.end_unit(bottom + 6, crop="u2")
    return bottom + 6


def unit3(c: Canvas, y0: float, meta: dict) -> float:
    ts, _ = keyframe_labels(meta)
    y = c.begin_unit(y0, 3, role="Relates Every Person-Object Pair From The Decoder's Attention, Then Across Time")
    ry = y + 8
    c.tensor(30, ry, 180, 40, "Spatial Attention Weights", "From Units 1 + 2, (b) Of Each Layer", col=LAT[2])
    c.tensor(30, ry + 52, 180, 40, "Decoded Slots", "From Units 1 + 2", col=BLUE)
    c.box(250, ry + 2, 210, 88, "Pair Readout", "Gather The p → o And o → p Weights:", kind="new",
          sub2="64 ⊕ qₚ ⊙ qₒ (256) = 320-d Per Pair", tsize=11.5, ssize=8.5)
    c.flow(210, ry + 20, 248, ry + 30, "", col=LAT[2])
    c.flow(210, ry + 72, 248, ry + 62, "", col=BLUE)
    c.flow(460, ry + 46, 498, ry + 46, "")
    c.box(500, ry + 22, 190, 48, "Readout Projector", "Linear 320 → 64, ReLU, LN", kind="learn", tsize=10.5, ssize=8.5)
    c.text(250, ry + 104, "Takes The Place Of The Union-Box Feature", size=8, fill=DIM)
    c.text(250, ry + 115, "No Inter-Object Transformer: (b) Already Relates Objects", size=8, fill=DIM)
    w, h = c.fit("spatial_attn", h=112, default=(140, 112), max_w=160)
    c.image_slot(1290 - w, ry - 4, w, h, "spatial_attn", caption=f"(b) Spatial Attn, t = {ts[1]}", caption_size=9)
    yb = ry + 134
    c.unit_port(30, yb, 180, 2, "Completed Slots", "From Unit 2, T × N × 256", h=40)
    c.unit_port(30, yb + 52, 180, 1, "Pair Geometry", "From Unit 1 OBB Corners, 8-D")
    c.box(250, yb + 52, 170, 38, "Geometry Projector", "Linear 8 → 64", kind="learn", tsize=10, ssize=8)
    c.flow(210, yb + 71, 248, yb + 71, "")
    c.unit_port(30, yb + 102, 180, 4, "Object Classes", "From Unit 4; GT In PredCls")
    c.box(250, yb + 102, 170, 38, "CLIP Text Embeddings", "37 × 512", kind="frozen", frozen=True, tsize=10, ssize=8)
    c.flow(210, yb + 121, 248, yb + 121, "")
    c.box(450, yb + 102, 150, 38, "Text Projector", "Linear 512 → 128", kind="learn", tsize=10, ssize=8)
    c.flow(420, yb + 121, 448, yb + 121, "")
    c.box(650, yb + 10, 230, 110, "Pair Encoder", "Person ⊕ Object ⊕ Readout ⊕ Textₚ ⊕ Textₒ", kind="learn",
          sub2="⊕ Geo: 896 → 256; 2 Self-Attn Layers", tsize=11.5, ssize=8.5)
    c.flow(210, yb + 20, 648, yb + 30, "Gather Person And Object Slots By Pair Index", col=LAT[0], lsize=8,
           loff=(0, -6), lpos=0.45)
    c.flow(420, yb + 71, 648, yb + 66, "")
    c.flow(600, yb + 121, 648, yb + 100, "")
    c.arrow(595, ry + 70, 700, yb + 8, curve=f"M595 {ry + 70} V{yb - 14} H700 V{yb + 8}")
    c.text(606, yb - 18, "Pair Readout Feature, T × K × 64", size=8, fill=MUTED)
    c.flow(880, yb + 65, 908, yb + 65, "")
    c.tensor(910, yb + 45, 104, 40, "Pair Tokens", "T × K × 256", col=MUTED)
    c.flow(1014, yb + 65, 1030, yb + 65, "")
    c.box(1032, yb + 35, 146, 60, "Temporal Edge Attention", "Same Pair Across Frames", kind="learn",
          sub2="Frame PE; 1 Layer", tsize=10, ssize=8.5)
    c.flow(1178, yb + 65, 1190, yb + 65, "")
    c.tensor(1192, yb + 45, 92, 40, "Relation Tokens", "→ Unit 4", col=UNIT[3][0])
    bottom = yb + 152
    c.end_unit(bottom, crop="u3")
    return bottom


def unit4(c: Canvas, y0: float, meta: dict) -> float:
    ts, tracked = keyframe_labels(meta)
    y = c.begin_unit(y0, 4, role="Object Classes, Predicates, Joint Detection And The Training Losses")
    ry = y + 8
    # node head
    c.unit_port(30, ry, 170, 2, "Completed Slots", "From Unit 2, T × N × 256", h=40)
    c.flow(200, ry + 20, 228, ry + 20, "")
    c.box(230, ry + 2, 150, 36, "Node Head", "MLP 256 → 256 → 37", kind="learn", tsize=10.5, ssize=8)
    c.flow(380, ry + 20, 408, ry + 20, "")
    c.tensor(410, ry, 190, 40, "Object Classes", "GT In PredCls, argmax In SGDet", col=MUTED)
    c.flow(600, ry + 20, 640, ry + 20, "", col=UNIT[2][0])
    c.text(646, ry + 18, "Back To Unit 3: CLIP Text Pathway", size=9, fill=UNIT[2][0])
    c.text(646, ry + 31, "CE Node Loss Only In SGDet", size=8, fill=DIM)
    # predicates
    c.unit_port(30, ry + 60, 170, 3, "Relation Tokens", "From Unit 3, T × K × 256", h=40)
    c.flow(200, ry + 80, 228, ry + 80, "")
    c.box(230, ry + 58, 150, 44, "Predicate Heads × 3", "MLP 256 → 128 → C", kind="learn", tsize=10.5, ssize=8)
    c.flow(380, ry + 80, 408, ry + 80, "")
    c.tensor(410, ry + 56, 190, 48, "Predicate Distributions", "Attention 3 · Spatial 6 · Contacting 17", col=RED)
    # joint detection
    c.tensor(30, ry + 122, 170, 40, "Decoded Free Queries", "From Units 1 + 2", col=ORANGE)
    c.flow(200, ry + 142, 228, ry + 142, "", col=ORANGE)
    c.box(230, ry + 118, 150, 48, "Detection Heads", "Class C+1 · 2-D Box", kind="new",
          sub2="3-D OBB (Dims, Yaw, Depth)", tsize=10.5, ssize=8)
    c.flow(380, ry + 142, 408, ry + 142, "")
    c.tensor(410, ry + 122, 190, 40, "Free-Query Detections", "T × Q Boxes + Classes", col=ORANGE)
    c.tensor(30, ry + 182, 170, 40, "Decoded Slots", "From Units 1 + 2", col=BLUE)
    c.flow(200, ry + 202, 228, ry + 202, "", col=BLUE)
    c.box(230, ry + 182, 150, 40, "Box / Corner Refinement", "Residual On The Input Box", kind="new", tsize=10,
          ssize=8)
    c.flow(380, ry + 202, 408, ry + 202, "")
    c.tensor(410, ry + 182, 190, 40, "Refined 2-D Box, 3-D Corners", "Per Slot", col=BLUE)
    # losses
    lx, lw = 660, 450
    c.loss_node(lx, ry + 44, lw, "sg", "BCE / CE On Visible + Masked Pairs (Clean GT); Logits + τ·log π_c",
                note="τ = 0.5; Truly Unseen Pairs Get No Edge Supervision (λ_vlm = 0)")
    c.loss_node(lx, ry + 112, lw, "det", "Hungarian Query ↔ GT: CE (No-Object × 0.1) + 5·L1 Box + L1 Corners",
                col=ORANGE, note="+ One-To-Many Auxiliary Matches (IoU > 0.5); λ_det = 1")
    c.loss_node(lx, ry + 178, lw, "slot", "L1( Refined Box − GT Box ) + L1( Refined Corners − GT Corners )",
                col=ORANGE, note="Valid And Visible Slots; λ_slot = 1")
    c.tensor(1140, ry + 50, 140, 40, "GT Predicates", "Visible Pairs", col=GREEN)
    c.tensor(1140, ry + 150, 140, 40, "GT Boxes", "2-D Boxes + OBB Corners", col=GREEN)
    c.flow(600, ry + 80, lx - 2, ry + 66, "", col=RED)
    c.flow(600, ry + 142, lx - 2, ry + 134, "", col=ORANGE)
    c.flow(600, ry + 202, lx - 2, ry + 200, "", col=ORANGE)
    c.flow(1140, ry + 70, lx + lw + 2, ry + 66, "", col=GREEN)
    c.flow(1140, ry + 164, lx + lw + 2, ry + 134, "", col=GREEN)
    c.flow(1140, ry + 176, lx + lw + 2, ry + 200, "", col=GREEN)
    c.rich(lx, ry + 250, [("Total:  ℒ = ℒ", {"fill": TXT, "serif": True}), ("sg", {"fill": RED, "sub": True}),
                          (" + 0.05 ℒ", {"fill": TXT, "serif": True}), ("rec", {"fill": ORANGE, "sub": True}),
                          ("  (Unit 2)", {"fill": ORANGE}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("det", {"fill": ORANGE, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("slot", {"fill": ORANGE, "sub": True})], size=11)
    c.text(lx, ry + 268, "Free-Query Detections Are Reported Separately; They Do Not Replace The Scene-Graph Boxes",
           size=8.5, fill=DIM)
    bottom = c.strip_fit(30, ry + 288, 130, [
        ("det_match_1", "Det. Match", 0.57), ("det_1", "Refined Boxes", 0.57),
        ("bev_1", "Refined 3-D Boxes (BEV)", 1.13),
        ("loss_pairs_1", f"Which Loss Each Pair Receives, t = {ts[1]}", 3.4),
        ("preds_1", f"Predicates, t = {ts[1]}; {tracked.title()} Unseen", 3.15),
    ])
    c.end_unit(bottom + 6)
    legend_y = bottom + 36
    c.legend(legend_y, [("frozen", "Frozen"), ("learn", "Learnable"), ("new", "Introduced By WorldWise++")],
             extra_swatches=[(VIOLET, "DINOv3 Semantics"), (TEAL, "π³ Geometry"), (BLUE, "World Slot"),
                             (ORANGE, "Free Query / Memory"), (LAT[0], "Completed Slot"), ("mask", "[MASK]"),
                             (RED, "Loss")])
    c.crop_mark("u4", y0 - 10, legend_y + 12)
    return legend_y + 22


def build(images, video: str, meta: dict) -> Canvas:
    c = Canvas(1720, 3600, images)
    c.band_title(46, "WorldWise++ · Image-Grounded Entity Decoder With Joint Detection")
    ov_end = c.unit_overview(80, OVERVIEW, overlap=("Entity Decoder × 4", "Shared By Units 1 + 2"), h=186)
    c.crop_mark("overview", 14, ov_end)
    tops, y = [], ov_end + 10
    for draw in (unit1, lambda cc, yy: unit12(cc, yy, meta), lambda cc, yy: unit2(cc, yy, meta),
                 lambda cc, yy: unit3(cc, yy, meta)):
        tops.append(y)
        y = draw(c, y) + 18
    tops.append(y)
    end = unit4(c, y, meta)
    bottoms = [t - 18 for t in tops[1:]] + [end - 36]
    cards = [
        (1, "See The Image",
         ["Frozen DINOv3 and π³ token grids are fused into a",
          "per-frame memory; every object gets a world slot",
          "from the scaffold tokenizer ([MASK] where unseen)",
          "and 30 learnable free queries join them."]),
        ((1, 2), "One Core, Three Attentions",
         ["Temporal attention keeps each slot's identity,",
          "spatial attention relates slots and free queries of",
          "a frame, and cross-attention reads the image memory.",
          "Visible and [MASK] slots pass through it together."]),
        (2, "Mark And Reconstruct",
         ["A visibility embedding flags recovered slots. At",
          "train time 30 % of visible cells are masked and",
          "their completed slots are reconstructed against an",
          "EMA copy of the gated fusion projector (0.05)."]),
        (3, "Relations From Attention",
         ["The decoder's own spatial-attention weights between",
          "a person and an object (64 values) and their slot",
          "product replace the union-box feature; the pair",
          "encoder and temporal edge attention follow."]),
        (4, "Detect To Reason",
         ["Free queries are trained with Hungarian matching and",
          "slots refine their own boxes. Removing this objective",
          "costs 8 mR on the occluded bucket: learning to",
          "localize is what teaches permanence."]),
    ]
    for k, (n, title, lines) in enumerate(cards):
        c.unit_card(tops[k], bottoms[k], n, title, lines)
    c.set_height(int(end + 10))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dump_intermediates.py output dir for the worldwise_pp cell")
    ap.add_argument("--video", default="00T1E")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3]
                                         / "assets/figures/architecture/paper_figures/dark/worldwise_pp.svg"))
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="white background (light) or the dark hero palette; read by common.py at import")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every text: Title Case, UPPER CASE or as written; read by common.py at import")
    args = ap.parse_args()
    meta = {}
    if args.images and (Path(args.images) / "meta.json").exists():
        meta = json.loads((Path(args.images) / "meta.json").read_text(encoding="utf-8"))
    c = build(image_map(args.images, IMAGES), args.video, meta)
    p = c.write(Path(args.out))
    print(f"wrote {p} ({c.w} x {c.h})" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))


if __name__ == "__main__":
    main()
