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
from common import (BLUE, DIM, FRAME_B, GREEN, LAT, MUTED, NEW_F, NEW_S, ORANGE, PURPLE_F, PURPLE_S, RED, TEAL, TXT, VIOLET,  # noqa: E402
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
        c.image_slot(30 + i * 68, fy, w, h, key, border=FRAME_B)
    c.text(130, fy + 126, "T Frames (672 × 378)", size=10.5, fill=MUTED, anchor="middle")
    # row A: the ROI path for the world slots (WorldWise+'s dinov3tok seam, unchanged)
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
    c.tensor(1006, fy + 6, 150, 40, "Appearance Tokens", "T × N × 256 → Stage 2", col=ORANGE)
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
    c.box(620, fy + 136, 150, 44, "PCA 1024 → 256", "Fit On Train Tokens, Frozen", kind="frozen", frozen=True, tsize=10, ssize=8.5)
    c.flow(770, fy + 158, 806, fy + 132, "", col=TEAL)
    # fusion of the two grids into the decoder memory
    c.box(808, fy + 96, 160, 58, "Token-Grid Fusion", "LN → Linear Per Stream;", kind="new",
          sub2="Per-Cell 2-Way Gate + 2-D Sine PE", tsize=11.5, ssize=8.5)
    c.flow(968, fy + 125, 1004, fy + 125, "")
    c.tensor(1006, fy + 105, 150, 40, "Memory Mₜ", "T × Hp·Wp × 256", col=ORANGE)
    c.text(1006, fy + 160, "→ Stage 2 (Cross-Attention Keys / Values)", size=8.5, fill=ORANGE)
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
    c.flow(160, ry + 16, 198, ry + 6, "")          # corners -> structural
    c.flow(160, ry + 18, 198, ry + 36, "")         # corners -> spatial
    c.flow(160, ry + 62, 198, ry + 40, "")         # poses -> spatial
    c.flow(160, ry + 64, 198, ry + 66, "")         # poses -> ego-motion
    c.flow(160, ry + 20, 198, ry + 96, "")         # corners (centres) -> motion
    c.flow(160, ry + 70, 198, ry + 102, "")        # poses (rotation) -> motion: camera-frame velocity
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
        c.a(f'<rect x="{dx + 10}" y="{yy}" width="{dw - 20}" height="28" rx="5" fill="{PURPLE_F}" stroke="{PURPLE_S}" stroke-width="1"/>')
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
    c.tensor(990, ry + 94, 200, 40, "Spatial Attention Weights", "2 Dirs × 4 Layers × 8 Heads → Stage 3", col=LAT[2])
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
    c.box(208, hy - 6, 170, 48, "Detection Heads", "Class C+1 · 2-D Box (Centre, Size)", kind="new",
          sub2="3-D OBB (Dims, Yaw, Depth, Offset)", tsize=10.5, ssize=8)
    c.flow(378, hy + 18, 414, hy + 18, "")
    c.tensor(416, hy, 150, 36, "Free-Query Detections", "T × Q Boxes + Classes", col=ORANGE)
    c.tensor(416, hy + 44, 150, 32, "Ground-Truth Boxes", "Visible Slots", col=GREEN)
    # slot branch
    c.tensor(30, hy + 90, 140, 36, "Decoded Slots", "From Stage 2", col=BLUE)
    c.flow(170, hy + 104, 206, hy + 99, "", col=BLUE)
    c.box(208, hy + 84, 170, 30, "Box / Corner Refinement Head", kind="new", tsize=9.5)
    c.flow(378, hy + 99, 414, hy + 99, "")
    c.tensor(416, hy + 84, 150, 32, "Refined 2-D Box, 3-D Corners", "Residual On The Input", col=BLUE)
    c.flow(170, hy + 112, 206, hy + 138, "", col=BLUE)
    c.box(208, hy + 124, 130, 28, "+ Visibility Embedding", kind="learn", tsize=9)
    c.flow(338, hy + 138, 374, hy + 138, "")
    c.tensor(376, hy + 120, 120, 36, "Completed Slots", "T × N × 256", col=LAT[0])
    c.flow(496, hy + 132, 528, hy + 124, "")
    c.box(530, hy + 112, 90, 24, "Node Head", kind="learn", tsize=10)
    c.text(626, hy + 128, "GT Override In PredCls", size=8, fill=DIM)
    c.flow(528, hy + 130, 506, hy + 178, "Text", lsize=7.5, loff=(-4, 10), anchor="end")   # class -> CLIP text
    c.flow(496, hy + 144, 528, hy + 154, "")
    c.box(530, hy + 142, 150, 24, "Reconstruction Projector", kind="learn", tsize=9)
    c.flow(436, hy + 156, 436, hy + 178, "Pair Tokens", col=LAT[0], lsize=7.5, loff=(6, 4), anchor="start")
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
    lx, lw = 790, 372
    c.loss_node(lx, hy - 6, lw, "det", "Hungarian Query ↔ GT: CE (No-Object × 0.1) + 5·L1 Box + L1 Corners",
                col=ORANGE, note="+ One-To-Many Auxiliary Matches (IoU > 0.5); λ_det = 1")
    c.loss_node(lx, hy + 50, lw, "slot", "L1( Refined Box − GT Box ) + L1( Refined Corners − GT Corners )",
                col=ORANGE, note="Valid And Visible Slots; λ_slot = 1")
    c.loss_node(lx, hy + 106, lw, "rec", "MSE( Reconstruction Projector(Completed Slot), EMA Target )", col=ORANGE,
                note="Target: EMA Gated Projector(Tier-1 Tokens); Masked Cells; λ_rec = 0.05")
    c.loss_node(lx, hy + 162, lw, "sg", "BCE / CE On Visible + Masked Pairs (Clean GT); Logits + τ·log π_c, τ = 0.5",
                note="As WorldWise; Truly Unseen Pairs Get No Edge Supervision (λ_vlm = 0)")
    c.tensor(lx + lw + 22, hy + 166, 106, 40, "GT Predicates", "Visible Pairs", col=GREEN)
    c.flow(566, hy + 18, lx - 2, hy + 12, "Predictions", col=ORANGE, lsize=8, loff=(0, -6))
    c.flow(566, hy + 58, lx - 2, hy + 22, "Targets", col=GREEN, lsize=8, loff=(0, 10), lpos=0.6)
    c.flow(566, hy + 66, lx - 2, hy + 66, "", col=GREEN)                    # gt boxes -> slot loss
    c.flow(566, hy + 100, lx - 2, hy + 82, "", col=BLUE)                    # refined boxes -> slot loss
    c.flow(680, hy + 154, lx - 2, hy + 128, "", col=ORANGE)                 # reconstruction -> rec loss
    c.flow(754, hy + 242, lx - 2, hy + 186, "", col=RED)
    c.flow(lx + lw + 22, hy + 186, lx + lw + 2, hy + 186, "", col=GREEN)
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
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="white background (light) or the dark hero palette; read by common.py at import")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every text: Title Case, UPPER CASE or as written; read by common.py at import")
    args = ap.parse_args()
    c = build(image_map(args.images, IMAGES), args.video)
    p = c.write(Path(args.out))
    print(f"wrote {p}" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))


if __name__ == "__main__":
    main()
