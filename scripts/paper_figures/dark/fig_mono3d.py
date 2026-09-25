"""Monocular 3-D detector figure (hero style), self-contained, three stages:

    Stage 1  Frozen Backbone And Feature Pyramid
    Stage 2  Region Proposals And 2-D Detection
    Stage 3  Factorized 3-D Head And Disentangled Loss

Drawn from ``lib/detector/monocular3d/models/dino_mono_3d.py`` and
``losses/ovmono3d_loss.py`` (see ``setup/MONOCULAR_3D_DETECTOR.md``): a frozen
DINOv3-L ViT, a learnable ViTDet-style SimpleFeaturePyramid (p2..p6, 256
channels), torchvision Faster R-CNN (RPN, MultiScaleRoIAlign 7 × 7, two-FC box
head, class + box predictors) and the factorized 3-D head over [ROI ⊕ 2-D box
⊕ intrinsics] whose outputs are back-projected to 8 corners.  The 3-D head is
the orange module.  Losses: Faster R-CNN cls / box / rpn-obj / rpn-box plus the
OVMono3D disentangled Chamfer loss with per-box uncertainty weighting and the
three-phase weight ramp.

Usage::

    python scripts/paper_figures/dark/fig_mono3d.py \
        --images outputs/intermediates/12XD3/mono3d --video 12XD3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLUE, DIM, FRAME_B, GREEN, MUTED, NEW_F, NEW_S, ORANGE, PURPLE_F, PURPLE_S, RED, TXT, VIOLET,  # noqa: E402
                    Canvas, image_map)
from baseline_common import CW, CX, X0, card, note_panel, set_height, strip_fit, vflow  # noqa: E402

IMAGES = [f"{k}_{i}" for k in ("frame", "fpn", "proposals", "det2d", "det3d", "bev", "head") for i in range(3)]


def rows_box(c: Canvas, x, y, w, title, rows, kind_fill, kind_stroke, row_h=26, tsize=12):
    """A container with a title and one purple row per line, as the entity
    decoder in the WorldWise++ figure."""
    h = 30 + len(rows) * (row_h + 6)
    c.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{kind_fill}" stroke="{kind_stroke}" stroke-width="1.8"/>')
    c.text(x + w / 2, y + 18, title, size=tsize, fill=TXT, anchor="middle", weight="700")
    for i, (a, b, tag) in enumerate(rows):
        yy = y + 30 + i * (row_h + 6)
        c.a(f'<rect x="{x + 10}" y="{yy}" width="{w - 20}" height="{row_h}" rx="5" fill="{PURPLE_F}" stroke="{PURPLE_S}" stroke-width="1"/>')
        c.text(x + 18, yy + 11, a, size=9, fill=TXT, weight="600")
        if b:
            c.text(x + 18, yy + 21, b, size=7.5, fill=MUTED)
        if tag:
            c.text(x + w - 16, yy + 16, tag, size=8, fill=ORANGE, anchor="end")
    return h


def build(images, video: str) -> Canvas:
    c = Canvas(1720, 1700, images)
    c.band_title(46, "Monocular 3-D Detector · Frozen DINOv3-L, Trained Pyramid And Faster R-CNN, Factorized 3-D Head")

    # ======================= Stage 1: frozen backbone and feature pyramid =======================
    y = c.stage(88, "Stage 1 · Frozen Backbone And Feature Pyramid")
    fy = y + 20
    for i, key in enumerate(["frame_0", "frame_1", "frame_2"]):
        w, h = c.fit(key, w=62, default=(62, 44), max_h=110)
        c.image_slot(30 + i * 68, fy, w, h, key, border=FRAME_B)
    c.text(130, fy + 126, "Frames, One At A Time", size=10.5, fill=MUTED, anchor="middle")
    c.flow(236, fy + 55, 262, fy + 55, "Image", lsize=8.5, loff=(0, -5))
    c.box(264, fy + 22, 140, 68, "DINOv3-L", "ViT-L/16, Frozen; Patch Tokens", kind="frozen", frozen=True,
          sub2="Only (CLS, Registers Dropped)", tsize=11.5, ssize=8)
    c.flow(404, fy + 55, 438, fy + 55, "", col=VIOLET)
    c.tensor(440, fy + 35, 150, 40, "Patch Grid", "H/16 × W/16 × 1024", col=VIOLET)
    c.flow(590, fy + 55, 618, fy + 55, "", col=VIOLET)
    ph = rows_box(c, 620, fy - 10, 250, "Simple Feature Pyramid (ViTDet)", [
        ("× 4  ConvT 2×2 → GELU → ConvT 2×2", "→ p2, Stride 4", ""),
        ("× 2  ConvT 2×2", "→ p3, Stride 8", ""),
        ("× 1  Identity", "→ p4, Stride 16", ""),
        ("× ½  MaxPool 2×2", "→ p5, Stride 32", ""),
        ("p6 = MaxPool(p5, Stride 2)", "→ p6, Stride 64", ""),
    ], PURPLE_F, PURPLE_S, row_h=26, tsize=11.5)
    c.text(620, fy + ph + 4, "Each Level: 1×1 Conv → BN → ReLU → 3×3 Conv → BN, 256 Channels; All Learnable", size=8, fill=DIM)
    c.flow(870, fy + 55, 898, fy + 55, "")
    c.tensor(900, fy + 35, 170, 40, "Pyramid p2 … p6", "256 Ch, Strides 4 · 8 · 16 · 32 · 64", col=MUTED)
    c.flow(985, fy + 75, 985, fy + 106, "To Stage 2", loff=(8, 4), anchor="start")
    note_panel(c, 1090, fy - 10, 200, [
        "resize: <= 255k px, mult. of 16",
        "ImageNet normalisation (dataset)",
        "intrinsics scaled with the image",
        "no RCNN resize / normalise",
        "backbone runs under autocast",
    ], "Input Convention")
    c.text(1090, fy + 104, "Only The Pyramid, The RPN And The Heads", size=8.5, fill=MUTED)
    c.text(1090, fy + 116, "Train; The ViT Never Updates.", size=8.5, fill=MUTED)
    py = fy + ph + 24
    y1 = strip_fit(c, X0, py, 135, [
        ("frame_1", "Input, t = 19", 0.5625), ("fpn_0", "Pyramid Channel Norms p2 … p6, t = 1", 2.4),
        ("fpn_1", "t = 19", 2.4), ("fpn_2", "t = 23", 2.4),
    ])

    # ======================= Stage 2: proposals and 2-D detection =======================
    y = c.stage(y1 + 26, "Stage 2 · Region Proposals And 2-D Detection")
    ry = y + 24
    c.tensor(30, ry, 140, 40, "Pyramid p2 … p6", "From Stage 1", col=MUTED)
    c.flow(170, ry + 20, 198, ry + 20, "")
    c.box(200, ry - 8, 200, 56, "Region Proposal Network", "3 Anchors Per Cell, One Size Per Level:", kind="learn",
          sub2="32 · 64 · 128 · 256 · 512; Ratios 0.5 / 1 / 2", tsize=10.5, ssize=7.5)
    c.flow(400, ry + 20, 428, ry + 20, "")
    c.tensor(430, ry - 2, 190, 44, "Proposals", "Train: 512 Sampled / Image, 25 % Positive", col=MUTED)
    c.text(430, ry + 54, "Positive = IoU ≥ 0.5 With A GT Box; Inference: Top-Scored After NMS", size=7.5, fill=DIM)
    c.flow(620, ry + 20, 648, ry + 20, "")
    c.box(650, ry - 8, 170, 56, "MultiScaleRoIAlign", "7 × 7, Sampling Ratio 2", kind="tool",
          sub2="Level Chosen By Box Scale", tsize=10.5, ssize=8)
    c.flow(820, ry + 20, 848, ry + 20, "")
    c.tensor(850, ry, 150, 40, "ROI Grid", "N × 256 × 7 × 7", col=MUTED)
    # second row
    r2 = ry + 96
    vflow(c, 925, ry + 40, 275, r2 - 2, ry + 68, "Flatten (12544-d)", lsize=8, loff=(0, -5), lx=760)
    c.box(200, r2, 150, 48, "Box Head", "FC 12544 → 1024 → 1024, ReLU", kind="learn", tsize=10.5, ssize=8)
    c.flow(350, r2 + 24, 378, r2 + 24, "")
    c.tensor(380, r2 + 2, 190, 44, "Shared ROI Features", "N × 1024 → Stage 3 And The SGG Methods", col=BLUE)
    c.flow(570, r2 + 24, 598, r2 + 24, "", col=BLUE)
    c.box(600, r2 - 4, 200, 56, "Box Predictor", "Class: Linear → 37 (Incl. Background)", kind="learn",
          sub2="Box: Linear → 4 × 37 Deltas", tsize=10.5, ssize=8)
    c.flow(800, r2 + 24, 828, r2 + 24, "")
    c.tensor(830, r2 + 2, 170, 44, "Class Logits, Box Deltas", "N × 37, N × 4 × 37; NMS → Detections", col=MUTED)
    # 2-D losses on the right
    lx, lw = 1030, 250
    c.loss_node(lx, ry - 8, lw, "obj", "RPN: BCE(Objectness, Anchor Label)")
    c.loss_node(lx, ry + 36, lw, "rpn", "RPN: Smooth-L1(Anchor Deltas), Positives")
    c.loss_node(lx, ry + 80, lw, "cls", "CE(Class Logits, Matched GT Class)")
    c.loss_node(lx, ry + 124, lw, "box", "Smooth-L1(Box Deltas), Positive ROIs")
    bx = 1285
    c.arrow(300, ry - 10, bx, ry + 53, col=RED, head=False, curve=f"M300 {ry - 10} V{ry - 24} H{bx} V{ry + 53}")
    c.arrow(bx, ry + 9, lx + lw + 1, ry + 9, col=RED)
    c.arrow(bx, ry + 53, lx + lw + 1, ry + 53, col=RED)
    c.text(730, ry - 28, "Objectness Logits, Anchor Deltas", size=8, fill=RED, anchor="middle")
    c.flow(1000, r2 + 20, lx - 2, ry + 100, "", col=RED)
    c.flow(1000, r2 + 28, lx - 2, ry + 144, "", col=RED)
    c.text(1000, r2 + 62, "Predictions", size=8, fill=RED)
    c.text(lx, ry + 176, "Standard torchvision Faster R-CNN Terms; Anchor /", size=8, fill=DIM)
    c.text(lx, ry + 187, "ROI Matching Against The 2-D GT Boxes Of The Frame", size=8, fill=DIM)
    py = r2 + 72
    y2 = strip_fit(c, X0, py, 125, [
        ("proposals_1", "Proposals, t = 19", 0.5625), ("det2d_0", "Detections, t = 1", 0.5625),
        ("det2d_1", "t = 19", 0.5625), ("det2d_2", "t = 23", 0.5625),
    ])

    # ======================= Stage 3: factorized 3-D head and disentangled loss =======================
    y = c.stage(y2 + 26, "Stage 3 · Factorized 3-D Head And Disentangled Loss")
    hy = y + 24
    c.tensor(30, hy, 150, 40, "Shared ROI Features", "N × 1024, Stage 2", col=BLUE)
    c.tensor(30, hy + 50, 150, 40, "2-D Box / 1000", "(x₁, y₁, x₂, y₂) Of The ROI", col=MUTED)
    c.tensor(30, hy + 100, 150, 40, "Intrinsics / 1000", "f_x, f_y, c_x, c_y Of The Frame", col=MUTED)
    c.flow(180, hy + 20, 208, hy + 62, "", col=BLUE)
    c.flow(180, hy + 70, 208, hy + 70, "")
    c.flow(180, hy + 120, 208, hy + 78, "")
    c.box(210, hy + 44, 150, 52, "Context FC", "Concat 1032 → 512, ReLU", kind="new", tsize=11, ssize=8.5)
    c.flow(360, hy + 70, 388, hy + 70, "")
    hh = rows_box(c, 390, hy - 6, 250, "Five Linear Heads", [
        ("Dims (l, w, h) = softplus(·) + 10⁻⁴", "Init ≈ 0.7 m", ""),
        ("Yaw (sin θ, cos θ), L2-Normalised", "Init = Identity", ""),
        ("Depth z = softplus(·) + 10⁻⁴", "Init ≈ 1.3 m", ""),
        ("Centre Offset (Δu, Δv) In Pixels", "Init = 0", ""),
        ("Uncertainty μ", "Init = 0", "To The Loss"),
    ], NEW_F, NEW_S, row_h=24, tsize=11.5)
    c.flow(640, hy + 70, 668, hy + 70, "")
    c.box(670, hy + 30, 200, 80, "Pinhole Back-Projection", "u = cᵤ + Δu, v = cᵥ + Δv", kind="tool",
          sub2="X = (u − c_x) / f_x · z, Y = (v − c_y) / f_y · z", tsize=10.5, ssize=8)
    c.text(670, hy + 122, "Corners = R_θ · (± l/2, ± w/2, ± h/2) + (X, Y, z); Lifted By The", size=8, fill=DIM)
    c.text(670, hy + 133, "Camera Pose → The World-Frame OBBs Used In SGDet", size=8, fill=DIM)
    c.flow(870, hy + 70, 898, hy + 70, "")
    c.tensor(900, hy + 44, 160, 52, "8 Corners", "N × 8 × 3, Camera Frame", col=ORANGE)
    c.tensor(1090, hy + 44, 200, 52, "GT 3-D Corners", "Matched Positives, ≤ 64 Per Batch", col=GREEN)
    # losses
    ly = hy + hh + 20
    lx, lw = 670, 620
    c.loss_node(lx, ly, lw, "3D", "Σ Chamfer_SL1( Mixed Box, GT ) / Diag(GT) Over xy · z · Dims · Yaw  +  Chamfer_SL1( Pred, GT ) / Diag(GT)",
                col=ORANGE, note="Mixed Box = Predicted Attribute, GT Values Of The Rest; Attributes From Corners By xy-PCA; Diag ≥ 0.5 m; Clamp 100")
    c.loss_node(lx, ly + 56, lw, "unc", "√2 · exp(−μ) · ℒ_3D + μ, Averaged Over The Matched Boxes", col=ORANGE,
                note="μ Is STE-Clamped To [−5, 10]: A Box The Head Cannot Resolve Is Down-Weighted While μ Is Penalised")
    c.flow(980, hy + 96, 980, ly - 2, "Predicted Corners", col=ORANGE, lsize=8, loff=(8, 4), anchor="start")
    c.flow(1190, hy + 96, 1190, ly - 2, "Targets", col=GREEN, lsize=8, loff=(8, 4), anchor="start")
    c.arrow(515, hy - 6 + hh, lx - 2, ly + 79, col=ORANGE, width=1.2, curve=f"M515 {hy - 6 + hh} V{ly + 79} H{lx - 2}")
    c.text(522, ly + 74, "Uncertainty μ", size=8, fill=ORANGE)
    c.rich(lx, ly + 126, [("Total:  ℒ = ℒ", {"fill": TXT, "serif": True}), ("cls", {"fill": RED, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("box", {"fill": RED, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("obj", {"fill": RED, "sub": True}),
                          (" + ℒ", {"fill": TXT, "serif": True}), ("rpn", {"fill": RED, "sub": True}),
                          (" + w", {"fill": TXT, "serif": True}), ("3D", {"fill": ORANGE, "sub": True}),
                          ("(e) · ℒ", {"fill": TXT, "serif": True}), ("unc", {"fill": ORANGE, "sub": True}),
                          ("      Weights w_cls = w_box = w_obj = w_rpn = 1; The 3-D Loss Runs In Float32", {"fill": DIM})], size=11)
    note_panel(c, 30, ly - 6, 330, [
        "e <  R      : w = 0        (2-D only)",
        "R <= e < 2R : w = (e-R)/R  (linear)",
        "e >= 2R     : w = 1        (full)",
        "R = weight_3d_ramp_epochs = 5",
        "V1 configs: ramp off, 3-D from epoch 1",
    ], "Three-Phase 3-D Weight Ramp w_3D(e)")
    c.text(lx, ly + 146, "The Same Shared 1024-d ROI Vector Is Written Out As The Detector ROI Feature Read By Every Scene-Graph Method",
           size=8.5, fill=DIM)
    py = ly + 166
    y3 = strip_fit(c, X0, py, 125, [
        ("head_1", "3-D Head Outputs, t = 19", 1.5), ("det3d_0", "3-D Boxes, t = 1", 0.5625),
        ("det3d_1", "t = 19", 0.5625), ("det3d_2", "t = 23", 0.5625), ("bev_1", "Bird's-Eye View, t = 19", 1.3),
    ])

    # ======================= cards and legend =======================
    legend_y = y3 + 30
    c.legend(legend_y, [("frozen", "Frozen"), ("learn", "Learnable"), ("new", "Factorized 3-D Head"),
                        ("tool", "Parameter-Free Program")],
             extra_swatches=[(VIOLET, "DINOv3 Patch Tokens"), (BLUE, "Shared ROI Features"), (ORANGE, "3-D Prediction"),
                             (GREEN, "Ground Truth"), (RED, "Loss")])
    top, bottom, gap = 88, legend_y - 30, 20
    ch = (bottom - top - 2 * gap) / 3
    cards = [
        ("Frozen Backbone, Trained Adapter",
         ["The DINOv3-L ViT never updates; a ViTDet-style simple",
          "feature pyramid (transposed convolutions, identity,",
          "max-pool) turns its single 1024-d patch grid into five",
          "256-channel levels for a standard torchvision Faster",
          "R-CNN whose RPN and heads are trained from scratch."]),
        ("Factorized 3-D Head",
         ["One FC over the shared 1024-d ROI vector, normalised",
          "2-D box and intrinsics predicts dims, yaw, depth, a",
          "centre offset and an uncertainty; pinhole back-projection",
          "gives 8 camera-frame corners. On 12XD3 it places dish,",
          "food and sandwich at 0.8–1.0 m and 3–11 cm; the picture",
          "and the window are missed."]),
        ("Disentangled, Uncertainty-Weighted Loss",
         ["Each attribute is scored by rebuilding a box from its",
          "prediction and the GT of the rest (smooth-L1 Chamfer,",
          "diagonal-normalised), plus a holistic term; √2·exp(−μ)",
          "weights each box and μ is penalised (μ = −1.1 to −3.6 on",
          "12XD3). A three-phase ramp brings the 3-D term in late."]),
    ]
    for i, (title, lines) in enumerate(cards):
        card(c, CX, top + i * (ch + gap), CW, ch, i + 1, title, lines, tsize=16.5)
    set_height(c, int(legend_y + 22))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="dump directory for the mono3d panels (outputs/intermediates/<video>/mono3d)")
    ap.add_argument("--video", default="12XD3")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "outputs/paper_figures/dark/mono3d.svg"))
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="white background (light) or the dark hero palette; read by common.py at import")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every text: Title Case, UPPER CASE or as written; read by common.py at import")
    args = ap.parse_args()
    c = build(image_map(args.images, IMAGES), args.video)
    p = c.write(Path(args.out))
    print(f"wrote {p} ({c.w} x {c.h})" + (f"  (placeholders for: {sorted(set(c.missing))})" if c.missing else ""))


if __name__ == "__main__":
    main()
