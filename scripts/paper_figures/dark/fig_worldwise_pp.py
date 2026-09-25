"""WorldWise++ method figure (dark hero style), self-contained, three stages:

    Stage 1  Frozen Token Grids And Memory
    Stage 2  Geometry Scaffold And Entity Decoder
    Stage 3  Heads, Joint Detection And Losses

Usage::

    python scripts/paper_figures/dark/fig_worldwise_pp.py \
        --images outputs/intermediates/12XD3/worldwise_pp --video 12XD3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUE, DIM, GREEN, LAT, MUTED, NEW_F, NEW_S, ORANGE, PURPLE_S, RED, TEAL, TXT, VIOLET,  # noqa: E402
                    Canvas, GEOMETRY_IMAGES, image_map)

IMAGES = GEOMETRY_IMAGES + [
    "frame_0", "frame_1", "frame_2", "visibility", "tokens_in", "tokens_out", "appearance_in",
    "grid_dino_0", "grid_dino_1", "grid_dino_2", "grid_pi3_0", "grid_pi3_1", "grid_pi3_2",
    "memory_0", "memory_1", "memory_2", "gate_0", "gate_1", "gate_2",
    "cross_attn_0", "cross_attn_1", "cross_attn_2", "temporal_attn", "spatial_attn",
    "det_0", "det_1", "det_2", "det_match_0", "det_match_1", "det_match_2", "bev_0", "bev_1", "bev_2",
    "preds_0", "preds_1", "preds_2", "train_mask", "recon_sim", "loss_pairs_0", "loss_pairs_1"]


def build(images, video: str) -> Canvas:
    c = Canvas(1720, 1560, images)
    c.band_title(46, "WorldWise++ · Image-Grounded Entity Decoder With Joint Detection")

    # ======================= Stage 1: frozen token grids and memory =======================
    y = c.stage(88, "Stage 1 · Frozen Token Grids And Memory")
    fy = y + 20
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border="#3a4048")
    c.text(130, fy + 126, "T Frames (672 × 378)", size=10.5, fill=MUTED, anchor="middle")
    c.flow(236, fy + 40, 262, fy + 30, "", col=VIOLET)
    c.flow(236, fy + 70, 262, fy + 100, "", col=TEAL)
    c.box(264, fy + 4, 130, 52, "DINOv3-L", "ViT-L/16, Frozen", kind="frozen", frozen=True, tsize=11.5, ssize=9)
    c.box(264, fy + 74, 130, 52, "π³", "Frozen", kind="frozen", frozen=True, tsize=11.5, ssize=9)
    c.flow(394, fy + 30, 430, fy + 30, "", col=VIOLET)
    c.flow(394, fy + 100, 430, fy + 100, "", col=TEAL)
    c.tensor(432, fy + 8, 150, 44, "L24n Grid", "24 × 42 × 1024, Final Layer", col=VIOLET)
    c.tensor(432, fy + 78, 150, 44, "G14 Grid", "27 × 48 × 1024, Global Layer", col=TEAL)
    c.flow(582, fy + 30, 618, fy + 30, "", col=VIOLET)
    c.flow(582, fy + 100, 618, fy + 100, "", col=TEAL)
    c.box(620, fy + 8, 150, 44, "Resample To π³ Lattice", "+ PCA 1024 → 256, Frozen", kind="frozen", frozen=True, tsize=10, ssize=8.5)
    c.box(620, fy + 78, 150, 44, "PCA 1024 → 256", "Fit On Train Tokens, Frozen", kind="frozen", frozen=True, tsize=10, ssize=8.5)
    c.flow(770, fy + 30, 806, fy + 56, "", col=VIOLET)
    c.flow(770, fy + 100, 806, fy + 74, "", col=TEAL)
    c.box(808, fy + 36, 160, 58, "Token-Grid Fusion", "LN → Linear Per Stream;", kind="new",
          sub2="Per-Cell 2-Way Gate + 2-D Sine PE", tsize=11.5, ssize=8.5)
    c.flow(968, fy + 65, 1004, fy + 65, "")
    c.tensor(1006, fy + 45, 150, 40, "Memory Mₜ", "T × Hp·Wp × 256", col=ORANGE)
    c.text(1006, fy + 100, "→ Stage 2 (Cross-Attention Keys / Values)", size=8.5, fill=ORANGE)
    # ROI path for the slots
    c.tensor(432, fy + 140, 150, 36, "Object Boxes", "Same Boxes As WorldWise", col=MUTED)
    c.flow(507, fy + 52, 507, fy + 138, "", col=VIOLET, width=1.2)
    c.flow(582, fy + 158, 618, fy + 158, "", col=VIOLET)
    c.box(620, fy + 140, 150, 36, "ROI-Align (DINOv3)", "L16 · L20 · L24n", kind="frozen", frozen=True, tsize=10, ssize=8.5)
    c.flow(770, fy + 158, 806, fy + 158, "")
    c.tensor(808, fy + 140, 160, 36, "Tier-1 Tokens", "T × N × 3072", col=VIOLET)
    c.flow(968, fy + 158, 1004, fy + 158, "")
    c.box(1006, fy + 140, 150, 36, "Gated Fusion Projector", "Linear → ReLU → LN", kind="learn", tsize=10, ssize=8.5)
    c.tensor(1006, fy + 186, 150, 36, "Appearance Tokens", "T × N × 256 → Stage 2", col=ORANGE)
    c.flow(1081, fy + 176, 1081, fy + 184, "", col=ORANGE)
    py = fy + 236
    c.strip(30, py, 150, [
        ("grid_dino_1", "DINOv3 L24n", 85), ("grid_pi3_1", "π³ G14", 85),
        ("gate_1", "Gate (DINOv3)", 85), ("memory_1", "Memory Mₜ", 85),
        ("appearance_in", "Tier-1 Tokens Per Visible Slot (PCA)", 380),
    ], gap=14, caption_size=9, max_w=390)

    # ======================= Stage 2: geometry scaffold and entity decoder =======================
    y = c.stage(py + 196, "Stage 2 · Geometry Scaffold And Entity Decoder")
    ry = y + 20
    c.tensor(30, ry, 130, 36, "OBB Corners", "T × N × 8 × 3", col=MUTED)
    c.tensor(30, ry + 46, 130, 36, "Camera Poses", "T × 4 × 4", col=MUTED)
    c.tensor(30, ry + 92, 130, 36, "Appearance Tokens", "From Stage 1", col=ORANGE)
    ex, ew, eh = 200, 140, 24
    enc = [("Structural Encoder", ry - 6), ("Spatial Encoder", ry + 24), ("Ego-Motion Encoder", ry + 54),
           ("Motion Encoder", ry + 84)]
    for t, yy in enc:
        c.box(ex, yy, ew, eh, t, kind="learn", tsize=10)
    c.flow(160, ry + 16, 198, ry + 6, "")
    c.flow(160, ry + 18, 198, ry + 36, "")
    c.flow(160, ry + 62, 198, ry + 40, "")
    c.flow(160, ry + 64, 198, ry + 66, "")
    c.flow(160, ry + 20, 198, ry + 96, "")
    c.box(376, ry + 12, 130, 92, "Scaffold Tokenizer", "Fuse Geometry + Appearance;", kind="learn",
          sub2="[MASK] If Unseen", tsize=11, ssize=8.5)
    for _, yy in enc:
        c.flow(340, yy + 12, 374, ry + 58, "", width=1.2)
    c.flow(160, ry + 110, 374, ry + 90, "", col=ORANGE, width=1.2)
    c.flow(506, ry + 40, 542, ry + 18, "")
    c.tensor(544, ry - 2, 130, 40, "N World Slots sₜ,ₙ", "T × N × 256", col=BLUE)
    c.tensor(544, ry + 46, 130, 40, "Q Free Queries qₜ,q", "Learnable, Q = 30", col=ORANGE)
    c.tensor(544, ry + 94, 130, 40, "Memory Mₜ", "From Stage 1", col=ORANGE)
    c.text(544, ry + 148, "Slot Position: Linear(Structural Token)", size=8, fill=DIM)
    c.text(544, ry + 160, "Frame Position: Learned Embedding", size=8, fill=DIM)
    dx, dy, dw, dh = 720, ry - 8, 230, 168
    c.a(f'<rect x="{dx}" y="{dy}" width="{dw}" height="{dh}" rx="10" fill="{NEW_F}" stroke="{NEW_S}" stroke-width="1.8"/>')
    c.text(dx + dw / 2, dy + 18, "Entity Decoder × 4", size=12.5, fill=TXT, anchor="middle", weight="700")
    rows = [("(a) Temporal Self-Attention", "Slot n Across Its Frames", "Permanence"),
            ("(b) Spatial Self-Attention", "Slots + Free Queries Of Frame t", "Weights Kept"),
            ("(c) Cross-Attention", "Queries → Memory Mₜ", "Grounding"),
            ("(d) Feed-Forward", "256 → 1024 → 256", "")]
    for i, (t, s, tag) in enumerate(rows):
        yy = dy + 30 + i * 34
        c.a(f'<rect x="{dx + 10}" y="{yy}" width="{dw - 20}" height="28" rx="5" fill="#2a2140" stroke="{PURPLE_S}" stroke-width="1"/>')
        c.text(dx + 18, yy + 12, t, size=9.5, fill=TXT, weight="600")
        c.text(dx + 18, yy + 23, s, size=8, fill=MUTED)
        if tag:
            c.text(dx + dw - 16, yy + 18, tag, size=8, fill=ORANGE, anchor="end")
    c.flow(674, ry + 18, dx - 2, dy + 44, "", col=BLUE)
    c.flow(674, ry + 66, dx - 2, dy + 78, "", col=ORANGE)
    c.flow(674, ry + 114, dx - 2, dy + 112, "", col=ORANGE)
    c.text(720, ry + 184, "Slots → (a)(b)(c) · Free Queries → (b)(c) · Memory → (c) As Keys / Values",
           size=8.5, fill=MUTED)
    c.tensor(990, ry - 2, 200, 40, "Decoded Slots", "T × N × 256 → Stage 3", col=BLUE)
    c.tensor(990, ry + 46, 200, 40, "Decoded Free Queries", "T × Q × 256 → Stage 3", col=ORANGE)
    c.tensor(990, ry + 94, 200, 40, "Spatial Attention Weights", "T × 4 Layers × 8 Heads → Stage 3", col=LAT[2])
    c.flow(dx + dw, dy + 44, 988, ry + 18, "", col=BLUE)
    c.flow(dx + dw, dy + 78, 988, ry + 66, "", col=ORANGE)
    c.flow(dx + dw, dy + 64, 988, ry + 114, "From (b)", col=LAT[2], lsize=7.5, loff=(0, 10), lpos=0.7)
    c.text(720, ry + 172, "Replaces The Associative Retriever And The Inter-Object Transformer", size=8.5, fill=DIM)
    py = ry + 204
    c.strip(30, py, 135, [
        ("obb_1", "OBB Corners", 76), ("visibility", "N World Slots: Visible / [MASK]", 340),
        ("temporal_attn", "(a) Temporal Attention, Tracked Slot", 160), ("spatial_attn", "(b) Spatial Attention, Unseen Frame", 160),
        ("cross_attn_1", "(c) Cross-Attn", 76), ("tokens_out", "Decoded Slots (PCA)", 340),
    ], gap=14, caption_size=9, max_w=345)

    # ======================= Stage 3: heads, detection, losses =======================
    y = c.stage(py + 180, "Stage 3 · Heads, Joint Detection And Losses")
    hy = y + 24
    # detection branch
    c.tensor(30, hy, 140, 36, "Decoded Free Queries", "From Stage 2", col=ORANGE)
    c.flow(170, hy + 18, 206, hy + 18, "", col=ORANGE)
    c.box(208, hy - 6, 170, 48, "Detection Heads", "Class C+1 · 2-D Box (cxcywh)", kind="new", sub2="3-D OBB (Dims, Yaw, Depth)",
          tsize=10.5, ssize=8)
    c.flow(378, hy + 18, 414, hy + 18, "")
    c.tensor(416, hy, 150, 36, "Free-Query Detections", "T × Q Boxes + Classes", col=ORANGE)
    c.tensor(416, hy + 44, 150, 32, "Ground-Truth Boxes", "Visible Slots", col=GREEN)
    # slot branch
    c.tensor(30, hy + 90, 140, 36, "Decoded Slots", "From Stage 2", col=BLUE)
    c.flow(170, hy + 108, 206, hy + 108, "", col=BLUE)
    c.box(208, hy + 94, 130, 28, "+ Visibility Embedding", kind="learn", tsize=9)
    c.flow(338, hy + 108, 374, hy + 94, "")
    c.box(376, hy + 82, 100, 26, "Node Head", kind="learn", tsize=10)
    c.flow(338, hy + 108, 374, hy + 124, "")
    c.box(376, hy + 112, 140, 26, "Reconstruction Projector", kind="learn", tsize=9)
    c.flow(170, hy + 112, 206, hy + 150, "", col=BLUE)
    c.box(208, hy + 136, 170, 30, "Box / Corner Refinement Head", kind="new", tsize=9.5)
    c.flow(378, hy + 151, 414, hy + 151, "")
    c.tensor(416, hy + 136, 150, 32, "Refined 2-D Box, 3-D Corners", "Residual On The Input", col=BLUE)
    # relation branch
    c.tensor(30, hy + 184, 140, 36, "Spatial Attention Weights", "⊕ qₚ ⊙ qₒ Per Pair", col=LAT[2])
    c.flow(170, hy + 202, 206, hy + 202, "", col=LAT[2])
    c.box(208, hy + 180, 130, 44, "Pair Readout", "Linear → ReLU → LN", kind="new", tsize=10, ssize=8)
    c.text(208, hy + 236, "Replaces Union-Box Features", size=8, fill=DIM)
    c.flow(338, hy + 202, 374, hy + 202, "")
    c.box(376, hy + 180, 140, 44, "Relationship Predictor", "⊕ Text ⊕ Pair Geometry", kind="learn", tsize=10, ssize=8)
    c.flow(516, hy + 202, 552, hy + 202, "")
    c.box(554, hy + 186, 120, 32, "Temporal Edge Attn", kind="learn", tsize=9.5)
    c.tensor(554, hy + 226, 200, 32, "Predicate Distributions", "Attention 3 · Spatial 6 · Contacting 17", col=RED)
    c.flow(614, hy + 218, 614, hy + 224, "")
    # losses
    lx = 790
    c.loss_node(lx, hy - 6, 470, "det", "Hungarian Match Query ↔ GT: CE (No-Object × 0.1) + 5·L1 Box + L1 Corners",
                col=ORANGE, note="+ One-To-Many Auxiliary Matches (IoU > 0.5); λ_det = 1")
    c.loss_node(lx, hy + 50, 470, "slot", "L1( Refined Box − GT Box ) + L1( Refined Corners − GT Corners )",
                col=ORANGE, note="Valid And Visible Slots; λ_slot = 1")
    c.loss_node(lx, hy + 106, 470, "rec", "MSE( Reconstruction Projector(Decoded Slot), EMA Target )", col=ORANGE,
                note="Masked Cells; λ_rec = 0.5")
    c.loss_node(lx, hy + 162, 470, "sg", "BCE / CE On Visible Pairs, τ = 0.5 Logit Adj.; + Simulated-Unseen On Masked Pairs",
                note="Truly Unseen Pairs: No Edge Supervision (λ_vlm = 0)")
    c.flow(566, hy + 18, lx - 2, hy + 12, "Predictions", col=ORANGE, lsize=8, loff=(0, -6))
    c.flow(566, hy + 60, lx - 2, hy + 22, "Targets", col=GREEN, lsize=8, loff=(0, 10), lpos=0.6)
    c.flow(566, hy + 151, lx - 2, hy + 72, "", col=BLUE)
    c.flow(516, hy + 125, lx - 2, hy + 128, "", col=ORANGE)
    c.flow(754, hy + 242, lx - 2, hy + 186, "", col=RED)
    c.rich(lx, hy + 234, [("Total:  ℒ = ℒ", {"fill": TXT, "serif": True}), ("WorldWise", {"fill": RED, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("det", {"fill": ORANGE, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("slot", {"fill": ORANGE, "sub": True}),
                          ("      Scene-Graph Terms Byte-Identical To WorldWise", {"fill": DIM})], size=11)
    c.text(lx, hy + 254, "Free-Query Detections Are Not Substituted Into The Scene-Graph Protocol; They Are Reported Separately",
           size=8.5, fill=DIM)
    py = hy + 272
    c.strip(30, py, 125, [
        ("det_match_1", "Det. Match", 70), ("det_1", "Refined", 70), ("bev_1", "Refined 3-D Boxes (BEV)", 138),
        ("train_mask", "Artificial Masks", 250), ("loss_pairs_0", "Which Loss Each Pair Receives", 250),
        ("preds_1", "Inference: Unseen Frame", 250),
    ], gap=14, caption_size=9, max_w=260)

    CX, CW, CH = 1320, 380, 450
    c.card(CX, 88, CW, CH, 1, "See The Image",
           ["The decoder cross-attends to the fused DINOv3 / π³",
            "token grid of every frame, so a masked slot is",
            "recovered from its own visible frames and from the",
            "current image, not from pooled ROI vectors alone."])
    c.card(CX, 558, CW, CH, 2, "One Core, Three Attentions",
           ["Each layer runs temporal attention (permanence),",
            "spatial attention over slots and free queries (whose",
            "weights become the relation features) and image",
            "cross-attention; retriever and transformer are gone."])
    c.card(CX, 1028, CW, CH, 3, "Detect To Reason",
           ["Free DETR queries are trained with Hungarian matching",
            "and the slots refine their own boxes. Removing this",
            "objective costs 8 mR on the occluded bucket: learning",
            "to localize is what teaches permanence."])
    c.legend(1532, [("frozen", "Frozen"), ("learn", "Learnable"), ("new", "Introduced By This Variant")],
             extra_swatches=[(VIOLET, "DINOv3 Semantics"), (TEAL, "π³ Geometry"), (BLUE, "World Slot"),
                             (ORANGE, "Free Query / Memory"), ("mask", "[MASK]"), (RED, "Loss")])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dump_intermediates.py output dir for the worldwise_pp cell")
    ap.add_argument("--video", default="12XD3")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "outputs/paper_figures/dark/worldwise_pp.svg"))
    args = ap.parse_args()
    c = build(image_map(args.images, IMAGES), args.video)
    p = c.write(Path(args.out))
    print(f"wrote {p}" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))


if __name__ == "__main__":
    main()
