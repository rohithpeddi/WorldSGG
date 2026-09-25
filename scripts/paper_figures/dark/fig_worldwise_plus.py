"""WorldWise+ method figure, drawn as four processing units with real panels:

    Unit 1  Observed Objects Processing Unit     frozen DINOv3-L / π³ latents, ROI-Align, gated fusion,
                                                 geometry scaffold, encoders, scaffold tokenizer
    Unit 2  Unobserved Objects Processing Unit   associative retriever, visibility embedding, reconstruction
    Unit 3  Relationship Processing Unit         inter-object transformer, pair encoder, temporal edge attention
    Unit 4  Decoders                             node head, predicate heads, losses

Usage::

    python scripts/paper_figures/dark/fig_worldwise_plus.py \
        --images assets/figures/architecture/intermediates/00T1E/worldwise_plus \
        --grid-images assets/figures/architecture/intermediates/00T1E/worldwise_pp --video 00T1E

``--grid-images`` (the worldwise_pp dump) supplies only the frozen token-grid panels;
WorldWise+ itself ROI-pools those same frozen grids and never loads them whole.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DIM, FRAME_B, LAT, MUTED, ORANGE, RED, TEAL, UNIT, VIOLET,  # noqa: E402
                    Canvas, GEOMETRY_IMAGES, image_map)
from ww_units import Seams, unit2, unit3, unit4  # noqa: E402

IMAGES = GEOMETRY_IMAGES + [
    "frame_0", "frame_1", "frame_2", "visibility", "tokens_in", "tokens_out", "tokens_enriched", "retriever_attn",
    "appearance_in", "preds_0", "preds_1", "preds_2", "train_mask", "recon_sim", "loss_pairs_0", "loss_pairs_1"]
GRID_IMAGES = ["grid_dino_0", "grid_dino_1", "grid_dino_2", "grid_pi3_0", "grid_pi3_1", "grid_pi3_2"]

SEAMS = Seams(
    appearance="Tier-1 Tokens", appearance_sub="From Unit 1, Before Masking",
    ema_sub="EMA (m = 0.996) Of The Gated Fusion Projector",
    union="Union Tokens", union_sub="From Unit 1 Gated Fusion, 64-d",
    union_projector=None, union_col=ORANGE)

OVERVIEW = [
    ["Frozen DINOv3-L + π³ Latents", "ROI-Align → Tier-1 Tokens", "Gated Fusion Projector × 2",
     "Structural, Spatial, Ego-Motion", "  And Motion Encoders", "Scaffold Tokenizer ([MASK])"],
    ["Associative Retriever", "+ Visibility Embedding", "  → Completed Tokens", "Reconstruction vs EMA",
     "  Target (Training Only)"],
    ["Inter-Object Transformer", "  With 3-D PE", "Pair Encoder (Union, Text,", "  Pair Geometry)",
     "Temporal Edge Attention"],
    ["Node Head", "Predicate Heads × 3", "Logit Adjustment (τ = 0.5)", "Scene-Graph Losses"],
]


def unit1(c: Canvas, y0: float) -> float:
    y = c.begin_unit(y0, 1, role="Frozen Foundation-Model Latents Replace The Detector's Decoded Vector")
    fy = y + 18
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border=FRAME_B)
    c.text(130, fy + 126, "T Frames (672 × 378)", size=10.5, fill=MUTED, anchor="middle")
    # row A: frozen latents -> ROI-Align -> gated fusion
    c.flow(236, fy + 40, 262, fy + 30, "", col=VIOLET)
    c.flow(236, fy + 70, 262, fy + 100, "", col=TEAL)
    c.box(264, fy + 4, 140, 52, "DINOv3-L", "ViT-L/16, Frozen", kind="frozen", frozen=True, tsize=11.5, ssize=9)
    c.box(264, fy + 74, 140, 52, "π³", "Feed-Forward Geometry, Frozen", kind="frozen", frozen=True, tsize=11.5, ssize=8.5)
    c.flow(404, fy + 30, 440, fy + 30, "", col=VIOLET)
    c.flow(404, fy + 100, 440, fy + 100, "", col=TEAL)
    c.tensor(442, fy + 8, 160, 44, "Patch Token Grid", "L16 · L20 · L24n; 24 × 42 × 1024", col=VIOLET)
    c.tensor(442, fy + 78, 160, 44, "Decoder Latent Grid", "F4 · F14 · G14; 27 × 48 × 1024", col=TEAL)
    c.tensor(442, fy + 140, 160, 36, "Object And Union Boxes", "Same Boxes As WorldWise", col=MUTED)
    c.flow(602, fy + 30, 640, fy + 66, "", col=VIOLET)
    c.flow(602, fy + 100, 640, fy + 82, "", col=TEAL)
    c.flow(602, fy + 158, 640, fy + 98, "", col=MUTED)
    c.box(642, fy + 42, 130, 76, "ROI-Align", "7 × 7, Mean-Pooled", kind="new", sub2="Per Object And Per Union Box",
          tsize=11.5, ssize=8.5)
    c.flow(772, fy + 80, 808, fy + 80, "")
    c.tensor(810, fy + 44, 160, 72, "Tier-1 Tokens", "3072-d Per Stream", col=VIOLET)
    c.text(890, fy + 108, "Object: T × N · Union: T × K", size=8.5, fill=MUTED, anchor="middle")
    c.flow(970, fy + 80, 1006, fy + 80, "")
    c.box(1008, fy + 30, 200, 60, "Gated Fusion Projector × 2", "Per-Stream Linear; Per-Dimension", kind="new",
          sub2="Softmax Gate; ReLU → LN; One Per Seam", tsize=11.5, ssize=8.5)
    c.text(1108, fy + 20, "Single Stream (Headline dinov3tok) = Linear → ReLU → LN", size=8.5, fill=MUTED,
           anchor="middle")
    c.tensor(1008, fy + 124, 96, 40, "Appearance Tokens", "T × N × 256", col=ORANGE)
    c.tensor(1112, fy + 124, 110, 40, "Union Tokens", "T × K × 64 → Unit 3", col=ORANGE)
    c.flow(1056, fy + 90, 1056, fy + 122, "", col=ORANGE)
    c.flow(1167, fy + 90, 1167, fy + 122, "", col=ORANGE)
    c.text(642, fy + 136, "Replaces The Detector's Decoded 1024-d ROI Vector", size=8.5, fill=DIM)
    # row B: geometry scaffold -> encoders -> scaffold tokenizer
    gy = fy + 204
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
           ("Spatial Encoder", "Camera-Relative; → Unit 2 Q / K Bias", gy + 26),
           ("Ego-Motion Encoder", "Relative Poses Over Time", gy + 62),
           ("Motion Encoder", "Velocity / Acceleration (World + Camera)", gy + 98)]
    for t, s, yy in enc:
        c.box(ex, yy, ew, eh, t, s, kind="learn", tsize=10, ssize=8)
    c.flow(568, gy + 14, 618, gy + 5, "")          # corners -> structural
    c.flow(568, gy + 17, 618, gy + 38, "")         # corners -> spatial
    c.flow(568, gy + 58, 618, gy + 44, "")         # poses -> spatial
    c.flow(568, gy + 61, 618, gy + 77, "")         # poses -> ego-motion
    c.flow(568, gy + 20, 618, gy + 108, "")        # corners (centres) -> motion
    c.flow(568, gy + 64, 618, gy + 116, "")        # poses (rotation) -> motion
    c.box(900, gy + 10, 150, 100, "Scaffold Tokenizer", "Concatenate + Fuse;", kind="learn", sub2="[MASK] If Unseen",
          tsize=11.5, ssize=9)
    for _, _, yy in enc:
        c.flow(816, yy + 15, 898, gy + 60, "", width=1.2)
    c.flow(1040, fy + 164, 1000, gy + 8, "Appearance", col=ORANGE, lsize=8, loff=(8, 0), anchor="start")
    c.arrow(568, gy + 105, 975, gy + 112, curve=f"M568 {gy + 105} V{gy + 146} H975 V{gy + 112}")
    c.text(700, gy + 142, "Visibility: Which Cells Get [MASK]", size=8, fill=MUTED)
    c.text(985, gy + 128, "Training: 30 % Of Visible Cells Also Masked", size=8, fill=DIM)
    c.tensor(1090, gy + 40, 150, 40, "Object Tokens sₜ,ₙ", "T × N × 256", col=ORANGE)
    c.flow(1050, gy + 60, 1088, gy + 60, "")
    c.flow(1165, gy + 80, 1165, gy + 108, "To Unit 2", col=UNIT[1][0], loff=(8, 4), anchor="start")
    py = gy + 156
    bottom = c.strip_fit(30, py, 140, [
        ("grid_dino_1", "DINOv3 L24n Grid", 0.57), ("grid_pi3_1", "π³ G14 Grid", 0.57),
        ("appearance_in", "Tier-1 Tokens Per Visible Slot (PCA)", 2.76), ("obb_1", "OBB Corners", 0.57),
        ("camera_path", "Camera Poses → Ego-Motion", 1.62), ("visibility", "Object Tokens: Visible / [MASK]", 2.76),
    ])
    c.end_unit(bottom + 6, crop="u1")
    return bottom + 6


def legend(c: Canvas):
    def draw(y):
        c.legend(y, [("frozen", "Frozen"), ("learn", "Learnable"), ("new", "Introduced By WorldWise+")],
                 extra_swatches=[(VIOLET, "DINOv3 Semantics"), (TEAL, "π³ Geometry"), (LAT[0], "Recovered World Token"),
                                 (LAT[2], "Enriched Token"), ("mask", "[MASK]"), (RED, "Loss")])
    return draw


def build(images, video: str, meta: dict) -> Canvas:
    c = Canvas(1720, 3200, images)
    c.band_title(46, "WorldWise+ · Foundation-Model Latents At The Seam")
    ov_end = c.unit_overview(80, OVERVIEW)
    c.crop_mark("overview", 14, ov_end)
    tops = []
    y = ov_end + 10
    for draw in (unit1, lambda cc, yy: unit2(cc, yy, SEAMS, meta), lambda cc, yy: unit3(cc, yy, SEAMS, meta)):
        tops.append(y)
        y = draw(c, y) + 18
    tops.append(y)
    end = unit4(c, y, SEAMS, meta, legend(c))
    bottoms = [tops[1] - 18, tops[2] - 18, tops[3] - 18, end - 36]
    cards = [
        ("Swap The Seam",
         ["All appearance enters WorldWise through two",
          "projectors. WorldWise+ feeds them frozen DINOv3 /",
          "π³ latents pooled on the very same object and",
          "union boxes, instead of the detector's decoded",
          "1024-d vector; a per-dimension gate fuses streams."]),
        ("Mask And Retrieve, Unchanged",
         ["Each [MASK] token attends to the same object's",
          "naturally visible frames with camera-aware biases.",
          "30 % of visible cells are masked at train time and",
          "reconstructed against an EMA copy of the gated",
          "fusion projector (weight 0.05)."]),
        ("Reason In 3-D, Unchanged",
         ["Completed tokens attend to one another with a 3-D",
          "positional encoding. Each person-object pair joins",
          "its gated union tokens, the CLIP text of both",
          "classes and 8-D pair geometry, then attends along",
          "its own timeline."]),
        ("Hold Everything Else Fixed",
         ["Heads and losses are identical to WorldWise, so the",
          "+4.5 R / +5.1 mR in PredCls is attributable to the",
          "representation alone: latents beat outputs. Truly",
          "unseen pairs get no edge loss (λ_vlm = 0)."]),
    ]
    for k, (title, lines) in enumerate(cards):
        c.unit_card(tops[k], bottoms[k], k + 1, title, lines)
    c.set_height(int(end + 10))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dump_intermediates.py output dir for the worldwise_plus cell")
    ap.add_argument("--grid-images", default=None, help="dump_intermediates.py output dir for worldwise_pp (token grids)")
    ap.add_argument("--video", default="00T1E")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3]
                                         / "assets/figures/architecture/paper_figures/dark/worldwise_plus.svg"))
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="white background (light) or the dark hero palette; read by common.py at import")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every text: Title Case, UPPER CASE or as written; read by common.py at import")
    args = ap.parse_args()
    imgs = image_map(args.images, IMAGES)
    imgs.update(image_map(args.grid_images, GRID_IMAGES))
    meta = {}
    if args.images and (Path(args.images) / "meta.json").exists():
        meta = json.loads((Path(args.images) / "meta.json").read_text(encoding="utf-8"))
    c = build(imgs, args.video, meta)
    p = c.write(Path(args.out))
    print(f"wrote {p} ({c.w} x {c.h})" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))


if __name__ == "__main__":
    main()
