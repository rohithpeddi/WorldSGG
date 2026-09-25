"""WorldWise method figure, drawn as four processing units with real panels:

    Unit 1  Observed Objects Processing Unit     detector, geometry scaffold, encoders, scaffold tokenizer
    Unit 2  Unobserved Objects Processing Unit   associative retriever, visibility embedding, reconstruction
    Unit 3  Relationship Processing Unit         inter-object transformer, pair encoder, temporal edge attention
    Unit 4  Decoders                             node head, predicate heads, losses

An overview strip under the title summarises the four units; each unit writes a
crop marker (``crop_stages.py``) so the paper can place one unit per page.

Usage::

    python scripts/paper_figures/dark/fig_worldwise.py \
        --images assets/figures/architecture/intermediates/00T1E/worldwise --video 00T1E
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUE, DIM, FRAME_B, LAT, MUTED, RED, TXT, UNIT,  # noqa: E402
                    Canvas, GEOMETRY_IMAGES, image_map)
from ww_units import Seams, unit2, unit3, unit4  # noqa: E402

IMAGES = GEOMETRY_IMAGES + [
    "frame_0", "frame_1", "frame_2", "visibility", "tokens_in", "tokens_out", "tokens_enriched", "retriever_attn",
    "appearance_in", "preds_0", "preds_1", "preds_2", "train_mask", "recon_sim", "loss_pairs_0", "loss_pairs_1"]

SEAMS = Seams(
    appearance="ROI Features", appearance_sub="From Unit 1, Before Masking",
    ema_sub="EMA (m = 0.996) Of The Visual Projector",
    union="Union ROI Features", union_sub="From Unit 1 Detector, 1024-d",
    union_projector="Linear 1024 → 64, ReLU, LN", union_col=BLUE)

OVERVIEW = [
    ["Frozen Mono-3D Detector", "π³ Poses + OBB Corners", "Structural, Spatial, Ego-Motion", "  And Motion Encoders",
     "Visual Projector", "Scaffold Tokenizer ([MASK])"],
    ["Associative Retriever", "+ Visibility Embedding", "  → Completed Tokens", "Reconstruction vs EMA",
     "  Target (Training Only)"],
    ["Inter-Object Transformer", "  With 3-D PE", "Pair Encoder (Union, Text,", "  Pair Geometry)",
     "Temporal Edge Attention"],
    ["Node Head", "Predicate Heads × 3", "Logit Adjustment (τ = 0.5)", "Scene-Graph Losses"],
]


def unit1(c: Canvas, y0: float) -> float:
    y = c.begin_unit(y0, 1, role="A Token For Every Object At Every Frame; Appearance Only Where It Is Visible")
    fy = y + 18
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border=FRAME_B)
    c.text(130, fy + 126, "T Frames", size=10.5, fill=MUTED, anchor="middle")
    c.flow(236, fy + 40, 262, fy + 26, "Video", lsize=8.5, loff=(-8, -6))
    c.flow(236, fy + 76, 262, fy + 108, "")
    c.box(264, fy - 4, 136, 60, "Frozen Mono-3D Detector", "DINOv3 → FPN → RPN → ROI Head", kind="frozen", frozen=True,
          tsize=11, ssize=8.5)
    c.box(264, fy + 80, 136, 60, "Geometry Scaffold", "π³ Camera Poses; GT OBBs", kind="frozen", frozen=True,
          sub2="(Detector OBBs In SGDet)", tsize=11, ssize=8.5)
    TX, TW, TH = 440, 128, 34
    c.tensor(TX, fy - 8, TW, TH, "ROI Features", "T × N × 1024", col=BLUE)
    c.tensor(TX, fy + 32, TW, TH, "Union ROI Features", "T × K × 1024 → Unit 3", col=BLUE)
    c.tensor(TX, fy + 76, TW, TH, "OBB Corners", "T × N × 8 × 3")
    c.tensor(TX, fy + 116, TW, TH, "Camera Poses", "T × 4 × 4")
    c.tensor(TX, fy + 156, TW, TH, "Visibility Mask", "T × N → Units 2, 4")
    c.flow(400, fy + 18, TX - 2, fy + 9, "", col=BLUE)
    c.flow(400, fy + 36, TX - 2, fy + 49, "", col=BLUE)
    c.flow(400, fy + 100, TX - 2, fy + 93, "")
    c.flow(400, fy + 112, TX - 2, fy + 133, "")
    c.flow(400, fy + 126, TX - 2, fy + 170, "")
    ex, ew, eh = 620, 196, 32
    enc = [("Structural Encoder", "All Corners Of A Frame", fy - 12),
           ("Spatial Encoder", "Camera-Relative; → Unit 2 Q / K Bias", fy + 26),
           ("Ego-Motion Encoder", "Relative Poses Over Time", fy + 64),
           ("Motion Encoder", "Velocity / Acceleration (World + Camera)", fy + 102),
           ("Visual Projector", "Linear 1024 → 256, ReLU, LN", fy + 140)]
    for t, s, yy in enc:
        c.box(ex, yy, ew, eh, t, s, kind="learn", tsize=10, ssize=8)
    c.flow(568, fy + 90, 618, fy + 4, "")          # corners -> structural
    c.flow(568, fy + 92, 618, fy + 40, "")         # corners -> spatial
    c.flow(568, fy + 130, 618, fy + 46, "")        # poses -> spatial
    c.flow(568, fy + 132, 618, fy + 80, "")        # poses -> ego-motion
    c.flow(568, fy + 96, 618, fy + 114, "")        # corners (centres) -> motion
    c.flow(568, fy + 136, 618, fy + 122, "")       # poses (rotation) -> motion: camera-frame velocity
    c.flow(568, fy + 8, 618, fy + 152, "", col=BLUE)   # roi -> visual projector
    c.box(900, fy + 26, 150, 100, "Scaffold Tokenizer", "Concatenate + Fuse;", kind="learn", sub2="[MASK] If Unseen",
          tsize=11.5, ssize=9)
    for _, _, yy in enc:
        c.flow(816, yy + 16, 898, fy + 76, "", width=1.2)
    c.arrow(568, fy + 176, 975, fy + 128, curve=f"M568 {fy + 176} V{fy + 194} H975 V{fy + 128}")
    c.text(700, fy + 190, "Visibility: Which Cells Get [MASK]", size=8, fill=MUTED)
    c.text(985, fy + 150, "Training: 30 % Of Visible", size=8, fill=DIM)
    c.text(985, fy + 161, "Cells Also Masked", size=8, fill=DIM)
    c.tensor(1090, fy + 56, 150, 40, "Object Tokens sₜ,ₙ", "T × N × 256", col=BLUE)
    c.flow(1050, fy + 76, 1088, fy + 76, "")
    c.flow(1165, fy + 96, 1165, fy + 128, "To Unit 2", col=UNIT[1][0], loff=(8, 4), anchor="start")
    c.text(1090, fy + 146, "Visible: Projected Appearance + Geometry", size=8.5, fill=MUTED)
    c.text(1090, fy + 158, "Unseen: Learnable [MASK] + Geometry", size=8.5, fill=MUTED)
    py = fy + 214
    bottom = c.strip_fit(30, py, 150, [
        ("frame_1", "2-D Boxes", 0.57), ("obb_1", "OBB Corners", 0.57),
        ("scene3d_1", "Canonical Scene: Points, OBBs, Camera", 0.96), ("camera_path", "Camera Poses → Ego-Motion", 1.62),
        ("motion", "Object Centres → Motion Encoder", 1.42), ("visibility", "Object Tokens: Visible / [MASK]", 2.76),
    ])
    c.end_unit(bottom + 6, crop="u1")
    return bottom + 6


def legend(c: Canvas):
    def draw(y):
        c.legend(y, [("frozen", "Frozen"), ("learn", "Learnable")],
                 extra_swatches=[(BLUE, "Decoded Detector Token"), (LAT[0], "Recovered World Token"),
                                 (LAT[2], "Enriched Token"), ("mask", "[MASK]"), ("hatch", "Artificial Mask (p = 0.3)"),
                                 (RED, "Loss")])
    return draw


def build(images, video: str, meta: dict) -> Canvas:
    c = Canvas(1720, 3200, images)
    c.band_title(46, "WorldWise · Masked World Auto-Encoder")
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
        ("Every Object, Every Frame",
         ["The frozen detector gives appearance only where an",
          "object is visible; the π³ scaffold gives OBB corners",
          "and camera poses at every frame. Five encoders and",
          "the scaffold tokenizer fuse them; an unseen cell",
          "gets a learnable [MASK] instead of appearance."]),
        ("Mask And Retrieve",
         ["Each [MASK] token attends to the same object's",
          "naturally visible frames, with view-aware query and",
          "key biases from the camera. At train time 30 % of",
          "visible cells are masked too and reconstructed",
          "against an EMA target (weight 0.05)."]),
        ("Reason In 3-D",
         ["Completed tokens attend to one another with a 3-D",
          "positional encoding. Each person-object pair joins",
          "its union appearance, the CLIP text of both classes",
          "and 8-D pair geometry, then attends along its own",
          "timeline with temporal edge attention."]),
        ("Read Out And Supervise",
         ["A node head and three predicate heads read the",
          "tokens. Visible and artificially masked pairs get",
          "clean labels with a τ = 0.5 logit adjustment; truly",
          "unseen pairs get no edge loss (λ_vlm = 0), which",
          "lifted occluded recall by 9.5."]),
    ]
    for k, (title, lines) in enumerate(cards):
        c.unit_card(tops[k], bottoms[k], k + 1, title, lines)
    c.set_height(int(end + 10))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dump_intermediates.py output dir for the worldwise cell")
    ap.add_argument("--video", default="00T1E")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3]
                                         / "assets/figures/architecture/paper_figures/dark/worldwise.svg"))
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
