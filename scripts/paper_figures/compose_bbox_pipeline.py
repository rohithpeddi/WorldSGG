"""Compose the 3D bounding-box pipeline figure (supplementary Fig. sup:fig:bbox_pipeline) from the
real intermediates rendered by panels_bbox.py (and two 00T1E panels of the scene-pipeline figure).

    SCENE_VIDEO=00T1E python scripts/paper_figures/compose_bbox_pipeline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_common as SC  # noqa: E402

REPO = SC.REPO
_arch_bbp = REPO / "assets" / "figures" / "architecture" / "bbox_pipeline" / f"panels_{SC.VIDEO}"
BBP = _arch_bbp if _arch_bbp.exists() else (REPO / "outputs" / "bbox_pipeline" / f"panels_{SC.VIDEO}")
SCP = SC.PANEL_DIR                         # scene-pipeline panels of the same video
W, H = 15.0, 9.6
INK = "#1f2933"; MUTED = "#6b7280"; HEAD_BG = "#e5e7eb"
BAND = {"i": "#eef1f5", "ii": "#fbeee4", "iii": "#e8f0fb", "iv": "#e6f4ee", "v": "#f3eefb", "vi": "#f6f0e2"}
ARROW = dict(arrowstyle="-|>,head_length=5,head_width=3", color="#374151", lw=1.6, shrinkA=0, shrinkB=0, mutation_scale=1)
HEAD = 0.72
fig = None


def fx(x): return x / W
def fy(y): return y / H


def band(x, y, w, h, color, title, sub=None):
    ax = fig.add_axes([fx(x), fy(y), fx(w), fy(h)]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", fc=color, ec="#cbd5e1", lw=0.8,
                                transform=ax.transAxes, clip_on=False))
    hb = fig.add_axes([fx(x + 0.08), fy(y + h - 0.38), fx(w - 0.16), fy(0.32)]); hb.set_axis_off()
    hb.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.08", fc=HEAD_BG, ec="none",
                                transform=hb.transAxes, clip_on=False))
    hb.text(0.5, 0.5, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    if sub:
        fig.text(fx(x + w / 2), fy(y + h - 0.5), sub, ha="center", va="top", fontsize=8, color=MUTED, style="italic")


def panel(path, x, ytop, w, h=None, caption=None, cap_size=7.5):
    im = Image.open(path).convert("RGBA")
    box = im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()   # trim empty 3-D axes margins
    if box:
        pad = 8
        im = im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.size[0], box[2] + pad), min(im.size[1], box[3] + pad)))
    iw, ih = im.size; r = iw / ih
    if h is None:
        hh, ww = w / r, w
    else:
        hh, ww = (w / r, w) if r > w / h else (h, h * r)
    x0 = x + (w - ww) / 2
    y0 = ytop - (h if h else hh) + ((h - hh) / 2 if h else 0)
    ax = fig.add_axes([fx(x0), fy(y0), fx(ww), fy(hh)]); ax.set_axis_off()
    ax.imshow(im, interpolation="lanczos")
    used = h if h else hh
    if caption:
        fig.text(fx(x + w / 2), fy(ytop - used - 0.04), caption, ha="center", va="top", fontsize=cap_size, color=INK)
        used += 0.2
    return used


def arrow(x0, y0, x1, y1, **kw):
    fig.patches.append(FancyArrowPatch((fx(x0), fy(y0)), (fx(x1), fy(y1)), transform=fig.transFigure, **{**ARROW, **kw}))


def chip(x, y, text, fc, ec, size=7.2, bold=False):
    fig.text(fx(x), fy(y), text, ha="left", va="center", fontsize=size, color=INK, fontweight="bold" if bold else "normal",
             bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=0.7))


def build():
    global fig
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    L = SC.LABEL_COL
    # ------------------------------------------------------------ row 1
    r1y, r1h = 5.05, 4.45
    top = r1y + r1h - HEAD
    band(0.15, r1y, 5.2, r1h, BAND["i"], "(i) Active Objects & Detection", "GT labels + LLM motion reasoning; Grounding DINO fused with GT")
    fig.text(fx(0.32), fy(top - 0.02), "AG labels", fontsize=7.2, color=MUTED, va="center")
    cx = 1.12
    for lab in ("person", "laptop", "shoe", "bed", "doorway"):
        chip(cx, top - 0.02, lab, "white", L[lab]); cx += 0.2 + 0.075 * len(lab) + 0.12
    fig.text(fx(0.32), fy(top - 0.34), "dynamic", fontsize=7.2, color=MUTED, va="center")
    cx = 1.12
    for lab in ("person", "laptop", "shoe"):
        chip(cx, top - 0.34, lab, L[lab] + "40", L[lab]); cx += 0.2 + 0.075 * len(lab) + 0.12
    fig.text(fx(3.25), fy(top - 0.34), "static", fontsize=7.2, color=MUTED, va="center")
    cx = 3.72
    for lab in ("bed", "doorway"):
        chip(cx, top - 0.34, lab, L[lab] + "40", L[lab]); cx += 0.2 + 0.075 * len(lab) + 0.12
    fig.text(fx(0.32), fy(top - 0.6), "Llama-3.1-8B lists objects involved in movement; a fixed list of furniture stays static",
             fontsize=6.3, color=MUTED, va="center", style="italic")
    panel(BBP / "b1_detection.png", 0.35, top - 0.78, 4.8, h=top - 0.78 - (r1y + 0.08))

    band(5.55, r1y, 4.6, r1h, BAND["ii"], "(ii) Instance Segmentation (SAM2)", "box prompts per frame + propagation from GT seeds")
    panel(BBP / "b2_sam2.png", 5.7, top - 0.05, 4.3, caption="per-object mask = image-mode mask ∪ video-mode mask", cap_size=7)

    band(10.35, r1y, 4.5, r1h, BAND["iii"], "(iii) Floor Alignment", "PromptHMR body ↔ π³ points: 7-DoF similarity")
    panel(BBP / "b0_floor_body.png", 10.5, top - 0.02, 4.2, h=r1h - HEAD - 0.35,
          caption="floor plane + metric scale shared by every object", cap_size=7)

    # ------------------------------------------------------------ row 2
    r2y, r2h = 0.15, 4.55
    t2 = r2y + r2h - HEAD
    band(0.15, r2y, 5.2, r2h, BAND["iv"], "(iv) Object Points & Multiscale Erosion", "box ∩ mask → π³ points (conf ≥ P5), eroded with k = 0..10 px")
    hb = panel(BBP / "b3_erosion.png", 0.35, t2 - 0.02, 4.8)
    panel(BBP / "b3_volumes.png", 0.35, t2 - hb - 0.08, 4.8, h=t2 - hb - 0.08 - (r2y + 0.08))

    band(5.55, r2y, 4.6, r2h, BAND["v"], "(v) Oriented Box Estimation", "AABB, PCA OBB and floor-parallel OBB from the chosen points")
    panel(BBP / "b4_obb.png", 5.65, t2 - 0.02, 4.4, h=t2 - 0.02 - (r2y + 0.08))

    band(10.35, r2y, 4.5, r2h, BAND["vi"], "(vi) Temporal Completion → World Boxes", "fill gaps across annotated frames; map to the floor frame")
    ht = panel(BBP / "b5_timeline.png", 10.45, t2 - 0.02, 4.3)
    panel(BBP / "b6_final_boxes.png", 10.45, t2 - ht - 0.1, 4.3, h=t2 - ht - 0.1 - (r2y + 0.3),
          caption="final boxes over time (floor frame)", cap_size=7)

    # ------------------------------------------------------------ arrows
    # row 1: (i) -> (ii) -> (iii); row 2: (iv) -> (v) -> (vi), each band edge to band edge
    mid1 = r1y + r1h / 2 - 0.3
    arrow(5.36, mid1, 5.54, mid1); arrow(10.16, mid1, 10.34, mid1)
    mid2 = r2y + r2h / 2 - 0.2
    arrow(5.36, mid2, 5.54, mid2); arrow(10.16, mid2, 10.34, mid2)
    # wrap-around (iii) -> (iv): down out of (iii), left along the gap between the rows, down into (iv)
    gy = (r1y + r2y + r2h) / 2
    seg = dict(arrowstyle="-", color=ARROW["color"], lw=ARROW["lw"], shrinkA=0, shrinkB=0)
    arrow(12.6, r1y, 12.6, gy, **seg)
    arrow(12.6, gy, 2.75, gy, **seg)
    arrow(2.75, gy, 2.75, r2y + r2h)
    fig.text(fx(7.7), fy(gy), "  masks + floor frame → 3D points  ", ha="center", va="center", fontsize=7, color=MUTED,
             style="italic", bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    return fig


def main():
    _arch_fig = REPO / "assets" / "figures" / "architecture" / "bbox_pipeline" / "figure"
    _def_out = (_arch_fig if _arch_fig.parent.exists() else (REPO / "outputs" / "bbox_pipeline" / "figure")) / f"WorldSGG3DBBoxPipeline_{SC.VIDEO}"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_def_out))
    ap.add_argument("--dpi", type=int, default=250)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    f = build()
    for ext in ("pdf", "png"):
        f.savefig(f"{out}.{ext}", dpi=a.dpi, facecolor="white")
        print("wrote", f"{out}.{ext}")


if __name__ == "__main__":
    main()
