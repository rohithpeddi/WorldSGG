"""WorldWise+ method figure (dark hero style), self-contained, three stages:

    Stage 1  Frozen Foundation-Model Latents
    Stage 2  Geometry Scaffold And Masked World Auto-Encoding
    Stage 3  Prediction Heads And Losses

Usage::

    python scripts/paper_figures/dark/fig_worldwise_plus.py \
        --images outputs/intermediates/12XD3/worldwise_plus \
        --grid-images outputs/intermediates/12XD3/worldwise_pp --video 12XD3

``--grid-images`` (the worldwise_pp dump) supplies only the frozen token-grid panels;
WorldWise+ itself ROI-pools those same frozen grids and never loads them whole.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUE, DIM, FRAME_B, GREEN, LAT, MUTED, ORANGE, RED, TEAL, TXT, VIOLET,  # noqa: E402
                    Canvas, GEOMETRY_IMAGES, image_map)

IMAGES = GEOMETRY_IMAGES + [
    "frame_0", "frame_1", "frame_2", "visibility", "tokens_in", "tokens_out", "tokens_enriched", "retriever_attn",
    "appearance_in", "preds_0", "preds_1", "preds_2", "train_mask", "recon_sim", "loss_pairs_0", "loss_pairs_1"]
GRID_IMAGES = ["grid_dino_0", "grid_dino_1", "grid_dino_2", "grid_pi3_0", "grid_pi3_1", "grid_pi3_2"]


def build(images, video: str) -> Canvas:
    c = Canvas(1720, 1460, images)
    c.band_title(46, "WorldWise+ · Foundation-Model Latents At The Seam")

    # ======================= Stage 1: frozen foundation-model latents =======================
    y = c.stage(88, "Stage 1 · Frozen Foundation-Model Latents")
    fy = y + 20
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border=FRAME_B)
    c.text(130, fy + 126, "T Frames (672 × 378)", size=10.5, fill=MUTED, anchor="middle")
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
    # one projector per seam (visual_projector and union_proj), same form
    c.box(1008, fy + 30, 200, 60, "Gated Fusion Projector × 2", "Per-Stream Linear; Per-Dimension", kind="new",
          sub2="Softmax Gate; ReLU → LN; One Per Seam", tsize=11.5, ssize=8.5)
    c.rich(1108, fy + 12, [("g", {"fill": ORANGE, "serif": True, "italic": True}), (" = softmax", {"fill": ORANGE, "verbatim": True}),
                            ("k", {"fill": ORANGE, "sub": True, "italic": True}), (" W[h₁;h₂]   h = LN(ReLU(Σ", {"fill": ORANGE, "verbatim": True}),
                            ("k", {"fill": ORANGE, "sub": True, "italic": True}), (" g", {"fill": ORANGE, "verbatim": True}),
                            ("k", {"fill": ORANGE, "sub": True, "italic": True}), (" ⊙ h", {"fill": ORANGE, "verbatim": True}),
                            ("k", {"fill": ORANGE, "sub": True, "italic": True}), ("))", {"fill": ORANGE, "verbatim": True})], size=9, anchor="middle")
    c.text(1108, fy + 24, "Single Stream (Headline dinov3tok) = Linear → ReLU → LN", size=8.5, fill=MUTED, anchor="middle")
    c.tensor(1008, fy + 124, 96, 40, "Appearance Tokens", "T × N × 256", col=ORANGE)
    c.tensor(1112, fy + 124, 96, 40, "Union Tokens", "T × K × 64", col=ORANGE)
    c.flow(1056, fy + 90, 1056, fy + 122, "", col=ORANGE)
    c.flow(1160, fy + 90, 1160, fy + 122, "", col=ORANGE)
    c.text(1008, fy + 180, "→ Stage 2 (Scaffold Tokenizer)", size=8.5, fill=ORANGE)
    c.text(1112, fy + 192, "→ Stage 3 (Relationship Predictor)", size=8.5, fill=ORANGE)
    c.text(642, fy + 136, "Replaces The Detector's Decoded 1024-d ROI Vector;", size=8.5, fill=DIM)
    c.text(642, fy + 148, "Everything Downstream Is Unchanged", size=8.5, fill=DIM)
    py = fy + 216
    c.strip(30, py, 150, [
        ("grid_dino_1", "DINOv3 L24n", 85), ("grid_pi3_1", "π³ G14", 85),
        ("appearance_in", "Tier-1 Tokens Per Visible Slot (PCA)", 380), ("frame_1", "2-D Boxes", 85),
    ], gap=14, caption_size=9, max_w=390)

    # ======================= Stage 2: geometry scaffold and MWAE =======================
    y = c.stage(py + 196, "Stage 2 · Geometry Scaffold And Masked World Auto-Encoding")
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
    c.flow(506, ry + 58, 542, ry + 58, "")
    c.tensor(544, ry + 38, 120, 40, "Object Tokens", "T × N × 256", col=BLUE)
    c.tensor(544, ry + 92, 130, 36, "Camera Features", "Spatial Encoder → Q / K Bias", col=MUTED)
    c.flow(664, ry + 58, 700, ry + 58, "")
    c.flow(674, ry + 110, 700, ry + 92, "")        # per-object camera features -> view-aware bias
    c.box(702, ry + 14, 160, 88, "Associative Retriever", "Per-Object Cross-Attention", kind="learn",
          sub2="Masked ← Own Visible Frames", tsize=11, ssize=8.5)
    c.flow(862, ry + 58, 898, ry + 58, "Retrieved", lsize=7.5, loff=(0, -5))
    c.box(900, ry + 43, 120, 30, "+ Visibility Embedding", kind="learn", tsize=9)
    c.flow(1020, ry + 58, 1056, ry + 58, "")
    c.box(1058, ry + 14, 150, 88, "Inter-Object Transformer", "Self-Attention Over", kind="learn",
          sub2="Objects Per Frame", tsize=10.5, ssize=8.5)
    c.tensor(1058, ry + 116, 150, 36, "Enriched Tokens", "→ Stage 3", col=LAT[2])
    c.flow(1133, ry + 102, 1133, ry + 114, "")
    # the tokens after the visibility embedding feed both the transformer and the reconstruction head
    c.flow(960, ry + 73, 960, ry + 94, "")
    c.tensor(890, ry + 96, 140, 36, "Completed Tokens", "→ Stage 3 (Reconstruction)", col=LAT[0])
    c.text(376, ry + 118, "Byte-Identical To WorldWise", size=8.5, fill=DIM)
    py = ry + 166
    c.strip(30, py, 135, [
        ("obb_1", "OBB Corners", 76), ("camera_path", "Camera Poses", 160), ("visibility", "Object Tokens: Visible / [MASK]", 340),
        ("tokens_out", "After Retrieval (PCA)", 340), ("retriever_attn", "Retriever Attention", 150),
    ], gap=14, caption_size=9, max_w=345)

    # ======================= Stage 3: heads and losses =======================
    y = c.stage(py + 180, "Stage 3 · Prediction Heads And Losses")
    hy = y + 24
    c.tensor(30, hy + 20, 130, 40, "Enriched Tokens", "From Stage 2", col=LAT[2])
    c.tensor(30, hy + 72, 130, 40, "Union Tokens", "From Stage 1 (Frozen Latents)", col=ORANGE)
    c.tensor(30, hy + 124, 130, 40, "Pair Geometry", "8-D From OBB Corners", col=MUTED)
    c.tensor(30, hy + 176, 130, 40, "Completed Tokens", "From Stage 2", col=LAT[0])
    c.tensor(30, hy + 228, 130, 36, "Tier-1 Tokens", "From Stage 1 (Object Boxes)", col=VIOLET)
    c.flow(160, hy + 32, 196, hy + 20, "")
    c.box(198, hy + 4, 120, 32, "Node Head", kind="learn", tsize=11)
    c.flow(318, hy + 20, 352, hy + 20, "")
    c.tensor(354, hy + 4, 130, 32, "Object Classes", None, col=MUTED)
    c.text(404, hy + 48, "GT Override In PredCls; CE Loss In SGDet", size=8, fill=DIM)
    c.flow(372, hy + 36, 348, hy + 56, "Text", lsize=7.5, loff=(12, 4), anchor="start")   # class -> CLIP text
    c.flow(160, hy + 44, 196, hy + 78, "")
    c.flow(160, hy + 92, 196, hy + 92, "", col=ORANGE)
    c.flow(160, hy + 144, 196, hy + 106, "")
    c.box(198, hy + 58, 160, 72, "Relationship Predictor", "Pair Tokens ⊕ Union ⊕ Text", kind="learn",
          sub2="⊕ Pair Geometry; 2 Self-Attn Layers", tsize=11, ssize=8.5)
    c.flow(358, hy + 94, 394, hy + 94, "")
    c.box(396, hy + 74, 150, 40, "Temporal Edge Attention", "Same Pair Across Frames", kind="learn", tsize=10.5, ssize=8.5)
    c.flow(546, hy + 94, 582, hy + 94, "")
    c.tensor(584, hy + 66, 170, 56, "Predicate Distributions", "Attention 3 · Spatial 6 · Contacting 17", col=RED)
    c.flow(160, hy + 196, 196, hy + 196, "")
    c.box(198, hy + 180, 160, 32, "Reconstruction Projector", kind="learn", tsize=10.5)
    c.box(396, hy + 176, 150, 40, "EMA Target Projector", "EMA Copy Of The Gated Projector", kind="frozen", frozen=True,
          tsize=10.5, ssize=8)
    c.arrow(160, hy + 246, 471, hy + 218, col=VIOLET, curve=f"M160 {hy + 246} H471 V{hy + 218}")   # tier-1 -> ema target
    lx, lw = 790, 372
    c.loss_node(lx, hy + 4, lw, "sg", "BCE / CE On Visible Pairs; Logits + τ·log π_c, τ = 0.5",
                note="Truly Unseen Pairs Receive No Edge Supervision (λ_vlm = 0)")
    c.loss_node(lx, hy + 60, lw, "sim", "Same Loss On Artificially Masked Pairs (p_mask = 0.3), Clean GT",
                note="Simulated Occlusion Teaches Recovery Instead Of Noisy VLM Labels")
    c.loss_node(lx, hy + 116, lw, "rec", "MSE( Reconstruction Projector(Completed), EMA Target )", col=ORANGE,
                note="Artificially Masked Cells; λ_rec = 0.5 × 0.1 (Dominance) = 0.05")
    c.tensor(lx + lw + 22, hy + 34, 106, 44, "GT Predicates", "Visible Pairs Only", col=GREEN)
    c.flow(754, hy + 84, lx - 2, hy + 24, "", col=RED)
    c.flow(754, hy + 94, lx - 2, hy + 80, "", col=RED)
    c.flow(lx + lw + 22, hy + 50, lx + lw + 2, hy + 36, "", col=GREEN)
    c.flow(lx + lw + 22, hy + 62, lx + lw + 2, hy + 76, "", col=GREEN)
    c.flow(358, hy + 196, lx - 2, hy + 140, "Prediction", col=ORANGE, lsize=8, loff=(0, -6), lpos=0.75)
    c.flow(546, hy + 196, lx - 2, hy + 150, "Target", col=ORANGE, lsize=8, loff=(0, 10), lpos=0.8)
    c.rich(lx, hy + 240, [("Total:  ℒ = ℒ", {"fill": TXT, "serif": True}), ("sg", {"fill": RED, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("sim", {"fill": RED, "sub": True}),
                          (" + 0.05 ℒ", {"fill": TXT, "serif": True}), ("rec", {"fill": ORANGE, "sub": True}),
                          ("      Identical Recipe To WorldWise: A Clean A/B On The Representation", {"fill": DIM})], size=11)
    py = hy + 274
    c.strip(30, py, 130, [
        ("train_mask", "Training-Mode Pass: Artificial Masks", 300), ("recon_sim", "Reconstruction vs EMA Target (Cosine)", 300),
        ("loss_pairs_0", "Which Loss Each Pair Receives", 300), ("preds_1", "Inference: Predicates At The Unseen Frame", 300),
    ], gap=14, caption_size=9, max_w=300)

    CX, CW, CH = 1320, 380, 420
    c.card(CX, 88, CW, CH, 1, "Swap The Seam",
           ["All appearance enters WorldWise through two projectors.",
            "WorldWise+ feeds them frozen DINOv3 / π³ latents",
            "pooled on the very same object and union boxes,",
            "instead of the detector's decoded 1024-d vector."])
    c.card(CX, 528, CW, CH, 2, "Gate, Don't Average",
           ["With both streams a per-dimension softmax gate lets",
            "the model pick geometry or semantics per feature.",
            "The gate earns its parameters in SGDet, not PredCls:",
            "there the single DINOv3 stream is the headline cell."])
    c.card(CX, 968, CW, CH, 3, "Hold Everything Else Fixed",
           ["Scaffold, masking, EMA target, retriever, relation",
            "head and the loss are byte-identical to WorldWise,",
            "so the +4.5 R / +5.1 mR in PredCls is attributable",
            "to the representation alone: latents beat outputs."])
    c.legend(1432, [("frozen", "Frozen"), ("learn", "Learnable"), ("new", "Introduced By This Variant")],
             extra_swatches=[(VIOLET, "DINOv3 Semantics"), (TEAL, "π³ Geometry"), (LAT[0], "Recovered World Token"),
                             ("mask", "[MASK]"), (ORANGE, "Tracked (Occluded) Object"), (RED, "Loss")])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dump_intermediates.py output dir for the worldwise_plus cell")
    ap.add_argument("--grid-images", default=None, help="dump_intermediates.py output dir for worldwise_pp (token grids)")
    ap.add_argument("--video", default="12XD3")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "outputs/paper_figures/dark/worldwise_plus.svg"))
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="white background (light) or the dark hero palette; read by common.py at import")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every text: Title Case, UPPER CASE or as written; read by common.py at import")
    args = ap.parse_args()
    imgs = image_map(args.images, IMAGES)
    imgs.update(image_map(args.grid_images, GRID_IMAGES))
    c = build(imgs, args.video)
    p = c.write(Path(args.out))
    print(f"wrote {p}" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))


if __name__ == "__main__":
    main()
