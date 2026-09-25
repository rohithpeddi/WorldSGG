"""WorldWise method figure (dark hero style), three stages with real panels:

    Stage 1  Perception And Geometry Scaffold
    Stage 2  Masked World Auto-Encoding
    Stage 3  Prediction Heads And Losses

Usage::

    python scripts/paper_figures/dark/fig_worldwise.py \
        --images outputs/intermediates/12XD3/worldwise --video 12XD3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUE, DIM, GREEN, LAT, MASKC, MUTED, ORANGE, PURPLE_S, RED, TXT,  # noqa: E402
                    Canvas, GEOMETRY_IMAGES, image_map)

IMAGES = GEOMETRY_IMAGES + [
    "frame_0", "frame_1", "frame_2", "visibility", "tokens_in", "tokens_out", "tokens_enriched", "retriever_attn",
    "appearance_in", "preds_0", "preds_1", "preds_2", "train_mask", "recon_sim", "loss_pairs_0", "loss_pairs_1"]


def build(images, video: str) -> Canvas:
    c = Canvas(1720, 1400, images)
    c.band_title(46, "WorldWise · Masked World Auto-Encoder")

    # ======================= Stage 1: perception and geometry scaffold =======================
    y = c.stage(88, "Stage 1 · Perception And Geometry Scaffold")
    fy = y + 20
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border="#3a4048")
    c.text(130, fy + 126, "T Frames", size=10.5, fill=MUTED, anchor="middle")
    c.flow(236, fy + 55, 262, fy + 55, "Video", lsize=8.5, loff=(0, -5))
    c.box(264, fy + 22, 136, 68, "Frozen Mono-3D Detector", "DINOv3 → FPN → RPN → ROI Head", kind="frozen", frozen=True,
          tsize=11, ssize=8.5)
    # detector outputs as tensors
    c.tensor(440, fy - 4, 128, 36, "ROI Features", "T × N × 1024 (Decoded)", col=BLUE)
    c.tensor(440, fy + 42, 128, 36, "OBB Corners", "T × N × 8 × 3", col=MUTED)
    c.tensor(440, fy + 88, 128, 36, "Camera Poses", "T × 4 × 4", col=MUTED)
    c.flow(400, fy + 44, 438, fy + 14, "", col=BLUE)
    c.flow(400, fy + 56, 438, fy + 60, "")
    c.flow(400, fy + 68, 438, fy + 106, "")
    # encoders
    ex, ew, eh = 620, 196, 32
    enc = [("Structural Encoder", "All Corners Of A Frame", fy - 12), ("Spatial Encoder", "Camera-Relative Corners", fy + 26),
           ("Ego-Motion Encoder", "Relative Poses Over Time", fy + 64), ("Motion Encoder", "Velocity / Acceleration", fy + 102),
           ("Visual Projector", "Linear → ReLU → LN", fy + 140)]
    for t, s, yy in enc:
        c.box(ex, yy, ew, eh, t, s, kind="learn", tsize=10, ssize=8)
    c.flow(568, fy + 58, 618, fy + 4, "")          # corners -> structural
    c.flow(568, fy + 60, 618, fy + 42, "")         # corners -> spatial
    c.flow(568, fy + 104, 618, fy + 46, "")        # poses -> spatial
    c.flow(568, fy + 106, 618, fy + 80, "")        # poses -> ego-motion
    c.flow(568, fy + 62, 618, fy + 118, "")        # corners -> motion
    c.flow(568, fy + 14, 618, fy + 156, "", col=BLUE)   # roi -> visual projector
    # scaffold tokenizer
    c.box(900, fy + 26, 150, 100, "Scaffold Tokenizer", "Concatenate + Fuse;", kind="learn", sub2="[MASK] If Unseen",
          tsize=11.5, ssize=9)
    for _, _, yy in enc:
        c.flow(816, yy + 16, 898, fy + 76, "", width=1.2)
    c.tensor(1090, fy + 56, 150, 40, "Object Tokens sₜ,ₙ", "T × N × 256", col=BLUE)
    c.flow(1050, fy + 76, 1088, fy + 76, "")
    c.flow(1165, fy + 96, 1165, fy + 128, "To Stage 2", col=BLUE, loff=(8, 4), anchor="start")
    c.text(1090, fy + 150, "Visible: Projected Appearance + Geometry", size=8.5, fill=MUTED)
    c.text(1090, fy + 162, "Unseen: Learnable [MASK] + Geometry", size=8.5, fill=MUTED)
    # pictures
    py = fy + 190
    c.strip(30, py, 150, [
        ("frame_1", "2-D Boxes", 85), ("obb_1", "OBB Corners", 85),
        ("scene3d_1", "Canonical Scene: Points, OBBs, Camera", 170), ("camera_path", "Camera Poses → Ego-Motion Encoder", 175),
        ("motion", "Object Centres → Motion Encoder", 200), ("visibility", "Object Tokens: Visible / [MASK]", 380),
    ], gap=14, caption_size=9, max_w=390)

    # ======================= Stage 2: masked world auto-encoding =======================
    y = c.stage(py + 196, "Stage 2 · Masked World Auto-Encoding")
    ry = y + 30
    c.tensor(30, ry + 25, 150, 40, "Object Tokens sₜ,ₙ", "From Stage 1", col=BLUE)
    c.flow(180, ry + 45, 226, ry + 45, "")
    c.box(228, ry, 190, 90, "Associative Retriever", "Per-Object Cross-Attention", kind="learn",
          sub2="Masked Frames ← Own Visible Frames", tsize=11.5, ssize=9)
    c.text(323, ry + 104, "View-Aware Q / K Bias From Camera Features", size=8.5, fill=MUTED, anchor="middle")
    c.text(323, ry + 116, "Keys / Values: Visible Frames Only (n_cross_attn_layers = 2)", size=8.5, fill=MUTED, anchor="middle")
    c.flow(418, ry + 45, 454, ry + 45, "")
    c.tensor(456, ry + 25, 140, 40, "Completed Tokens", "T × N × 256", col=LAT[0])
    c.flow(596, ry + 45, 632, ry + 45, "")
    c.box(634, ry + 30, 160, 30, "+ Visibility Embedding", kind="learn", tsize=10.5)
    c.text(714, ry + 74, "Observed / Recovered Flag", size=8.5, fill=MUTED, anchor="middle")
    c.flow(794, ry + 45, 830, ry + 45, "")
    c.box(832, ry, 190, 90, "Inter-Object Transformer", "Self-Attention Over The", kind="learn",
          sub2="Objects Of Each Frame (3 Layers)", tsize=11.5, ssize=9)
    c.flow(1022, ry + 45, 1058, ry + 45, "")
    c.tensor(1060, ry + 25, 150, 40, "Enriched Tokens", "T × N × 256", col=LAT[2])
    c.flow(1135, ry + 65, 1135, ry + 96, "To Stage 3", col=LAT[2], loff=(8, 4), anchor="start")
    c.flow(526, ry + 65, 526, ry + 96, "To Stage 3 (Reconstruction)", col=LAT[0], loff=(8, 4), anchor="start")
    py = ry + 130
    c.strip(30, py, 135, [
        ("tokens_in", "Scaffold Tokens (PCA); [MASK] Outlined", 340), ("retriever_attn", "Retriever Attention, Tracked Slot", 150),
        ("tokens_out", "After Retrieval (Same PCA Basis)", 340), ("tokens_enriched", "After Inter-Object Transformer", 340),
    ], gap=14, caption_size=9, max_w=350)

    # ======================= Stage 3: heads and losses =======================
    y = c.stage(py + 180, "Stage 3 · Prediction Heads And Losses")
    hy = y + 24
    c.tensor(30, hy + 20, 130, 40, "Enriched Tokens", "From Stage 2", col=LAT[2])
    c.tensor(30, hy + 72, 130, 40, "Union Features", "T × K × 1024 (Detector)", col=BLUE)
    c.tensor(30, hy + 124, 130, 40, "Pair Geometry", "8-D Relative 3-D", col=MUTED)
    c.tensor(30, hy + 176, 130, 40, "Completed Tokens", "From Stage 2", col=LAT[0])
    # node head
    c.flow(160, hy + 32, 196, hy + 20, "")
    c.box(198, hy + 4, 120, 32, "Node Head", kind="learn", tsize=11)
    c.flow(318, hy + 20, 352, hy + 20, "")
    c.tensor(354, hy + 4, 130, 32, "Object Classes", None, col=MUTED)
    c.text(354, hy + 48, "GT Override In PredCls", size=8, fill=DIM)
    # relation head
    c.flow(160, hy + 44, 196, hy + 78, "")
    c.flow(160, hy + 92, 196, hy + 92, "")
    c.flow(160, hy + 144, 196, hy + 106, "")
    c.box(198, hy + 58, 160, 72, "Relationship Predictor", "Pair Tokens ⊕ Union ⊕ Text", kind="learn",
          sub2="⊕ Pair Geometry; 2 Self-Attn Layers", tsize=11, ssize=8.5)
    c.flow(358, hy + 94, 394, hy + 94, "")
    c.box(396, hy + 74, 150, 40, "Temporal Edge Attention", "Same Pair Across Frames", kind="learn", tsize=10.5, ssize=8.5)
    c.flow(546, hy + 94, 582, hy + 94, "")
    c.tensor(584, hy + 66, 170, 56, "Predicate Distributions", "Attention 3 · Spatial 6 · Contacting 17", col=RED)
    # reconstruction
    c.flow(160, hy + 196, 196, hy + 196, "")
    c.box(198, hy + 180, 160, 32, "Reconstruction Projector", kind="learn", tsize=10.5)
    c.box(396, hy + 180, 150, 32, "EMA Target Projector", kind="frozen", frozen=True, tsize=10.5)
    c.text(396, hy + 226, "EMA Copy Of The Visual Projector", size=8, fill=DIM)
    c.text(396, hy + 238, "Applied To The ROI Features", size=8, fill=DIM)
    # losses
    lx = 790
    c.loss_node(lx, hy + 4, 470, "sg", "BCE / CE On Visible Pairs; Logits + τ·log πⲜ, τ = 0.5",
                note="Truly Unseen Pairs Receive No Edge Supervision (λ_vlm = 0)")
    c.loss_node(lx, hy + 60, 470, "sim", "Same Loss On Artificially Masked Pairs (p_mask = 0.3), Clean GT",
                note="Simulated Occlusion Teaches Recovery Instead Of Noisy VLM Labels")
    c.loss_node(lx, hy + 116, 470, "rec", "MSE( Reconstruction Projector(Completed), EMA Target )", col=ORANGE,
                note="Masked Cells; λ_rec = 0.5")
    c.tensor(lx + 300, hy + 176, 170, 40, "Ground-Truth Predicates", "Visible Pairs Only", col=GREEN)
    c.flow(754, hy + 84, lx - 2, hy + 24, "", col=RED)
    c.flow(754, hy + 94, lx - 2, hy + 80, "", col=RED)
    c.flow(lx + 385, hy + 176, lx + 385, hy + 52, "", col=GREEN)
    c.flow(358, hy + 196, lx - 2, hy + 140, "Prediction", col=ORANGE, lsize=8, loff=(0, -6), lpos=0.75)
    c.flow(546, hy + 196, lx - 2, hy + 150, "Target", col=ORANGE, lsize=8, loff=(0, 10), lpos=0.8)
    c.rich(lx, hy + 240, [("Total:  ℒ = ℒ", {"fill": TXT, "serif": True}), ("sg", {"fill": RED, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("sim", {"fill": RED, "sub": True}),
                          (" + 0.5 ℒ", {"fill": TXT, "serif": True}), ("rec", {"fill": ORANGE, "sub": True}),
                          ("      Per-Class Log-Prior Offsets Are Train-Time Only", {"fill": DIM})], size=11)
    py = hy + 262
    c.strip(30, py, 130, [
        ("train_mask", "Training-Mode Pass: Artificial Masks", 300), ("recon_sim", "Reconstruction vs EMA Target (Cosine)", 300),
        ("loss_pairs_0", "Which Loss Each Pair Receives", 300), ("preds_1", "Inference: Predicates At The Unseen Frame", 300),
    ], gap=14, caption_size=9, max_w=300)

    # ======================= cards and legend =======================
    CX, CW, CH = 1320, 380, 400
    c.card(CX, 88, CW, CH, 1, "Mask & Retrieve",
           ["Occlusion is treated as masking. Every object gets a",
            "token at every frame; unseen ones carry [MASK] and",
            "are recovered by attention over their own visible",
            "frames, with view-aware biases from the camera."])
    c.card(CX, 508, CW, CH, 2, "Simulate, Don't Supervise",
           ["30 % of visible tokens are masked at train time and",
            "supervised with clean labels plus an EMA reconstruction",
            "target. Noisy VLM pseudo-labels on unseen pairs are",
            "dropped (λ_vlm = 0): it lifted occluded recall by 9.5."])
    c.card(CX, 928, CW, CH, 3, "Reason In 3-D",
           ["Ego-motion, camera-frame and motion encoders fuse",
            "into every token; relation tokens carry explicit pair",
            "geometry; a tail-aware logit adjustment (τ = 0.5) sets",
            "the recall / mean-recall operating point."])
    c.legend(1372, [("frozen", "Frozen"), ("learn", "Learnable")],
             extra_swatches=[(BLUE, "Decoded FRCNN Token"), (LAT[0], "Recovered World Token"), ("mask", "[MASK]"),
                             ("hatch", "Artificial Mask (p = 0.3)"), (ORANGE, "Tracked (Occluded) Object"), (RED, "Loss")])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dump_intermediates.py output dir for the worldwise cell")
    ap.add_argument("--video", default="12XD3")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "outputs/paper_figures/dark/worldwise.svg"))
    args = ap.parse_args()
    c = build(image_map(args.images, IMAGES), args.video)
    p = c.write(Path(args.out))
    print(f"wrote {p}" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))


if __name__ == "__main__":
    main()
