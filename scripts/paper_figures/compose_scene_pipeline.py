"""Compose the 4D scene-pipeline figure (supplementary Fig. sup:fig:scene_pipeline) from
the empirical panels rendered for video 00T1E by panels_stage{1,2,3}.py.

Layout mirrors the existing schematic: four stage bands
  (i) frame sampling  ->  (ii) feed-forward pi3  ->  (iii) static-dynamic decomposition
  (iv) trimmed-ICP alignment                      ->  final 4D scene
but every picture is a real intermediate of 00T1E.  Missing panels render as grey
placeholders so the layout can be iterated before all renders exist.

    python scripts/paper_figures/compose_scene_pipeline.py [--out outputs/scene_pipeline/figure]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
PANELS = REPO / "outputs" / "scene_pipeline" / "panels"

W, H = 15.0, 10.6         # inches; \textwidth-scaled in LaTeX
INK = "#1f2933"; MUTED = "#6b7280"; HEAD_BG = "#e5e7eb"
BAND = {"i": "#eef1f5", "ii": "#e8f0fb", "iii": "#fbeee4", "iv": "#e6f4ee", "out": "#f6f0e2"}
ARROW = dict(arrowstyle="-|>,head_length=5,head_width=3", color="#374151", lw=1.6,
             shrinkA=0, shrinkB=0, mutation_scale=1)

fig = None

def fx(x): return x / W
def fy(y): return y / H

def band(x, y, w, h, color, title, sub=None):
    ax = fig.add_axes([fx(x), fy(y), fx(w), fy(h)]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=color, ec="#cbd5e1", lw=0.8, transform=ax.transAxes, clip_on=False))
    hb = fig.add_axes([fx(x + 0.08), fy(y + h - 0.38), fx(w - 0.16), fy(0.32)]); hb.set_axis_off()
    hb.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=HEAD_BG, ec="none", transform=hb.transAxes, clip_on=False))
    hb.text(0.5, 0.5, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    if sub:
        fig.text(fx(x + w / 2), fy(y + h - 0.5), sub, ha="center", va="top", fontsize=8, color=MUTED, style="italic")

def panel(name, x, ytop, w, h=None, caption=None, cap_size=7.5, frame=False):
    """Place PANELS/<name>.png with its top-left corner at (x, ytop) inches and width w.
    If h is None the height follows the image aspect; otherwise the image is fitted into (w, h).
    Returns the height actually used (so callers can stack panels)."""
    p = PANELS / f"{name}.png"
    if p.exists():
        im = Image.open(p); iw, ih = im.size; img = iw / ih
        if h is None:
            hh, ww = w / img, w
        else:
            hh, ww = (w / img, w) if img > w / h else (h, h * img)
        x0 = x + (w - ww) / 2; y0 = ytop - (h if h else hh) + ((h - hh) / 2 if h else 0)
        ax = fig.add_axes([fx(x0), fy(y0), fx(ww), fy(hh)]); ax.set_axis_off()
        ax.imshow(im, interpolation="lanczos")
        if frame:
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#cbd5e1")
        used = h if h else hh
    else:
        used = h if h else w * 0.6
        ax = fig.add_axes([fx(x), fy(ytop - used), fx(w), fy(used)]); ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04", fc="#d1d5db",
                                    ec="#9ca3af", lw=0.8, ls="--", transform=ax.transAxes))
        ax.text(0.5, 0.5, name, ha="center", va="center", fontsize=7, color=INK, wrap=True)
    if caption:
        fig.text(fx(x + w / 2), fy(ytop - used - 0.04), caption, ha="center", va="top", fontsize=cap_size, color=INK)
        used += 0.18
    return used

def label(x, y, s, size=8.5, bold=False, color=INK, ha="center", va="center", rot=0):
    fig.text(fx(x), fy(y), s, ha=ha, va=va, fontsize=size, color=color, fontweight="bold" if bold else "normal", rotation=rot)

def arrow(x0, y0, x1, y1, **kw):
    a = FancyArrowPatch((fx(x0), fy(y0)), (fx(x1), fy(y1)), transform=fig.transFigure, **{**ARROW, **kw})
    fig.patches.append(a)

def node(x, y, w, h, text, fc="#ffffff", ec="#9ca3af", size=8, bold=False):
    ax = fig.add_axes([fx(x), fy(y), fx(w), fy(h)]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15", fc=fc, ec=ec, lw=0.9,
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=size, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.15)


def build():
    global fig
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")

    # ───────────────────────── row 1 (top-anchored) ─────────────────────────
    r1y, r1h = 5.55, 4.9
    top = r1y + r1h - 0.5                  # first panel top, below the band header
    # Stage (i)
    band(0.15, r1y, 4.55, r1h, BAND["i"], "Stage (i): Adaptive Frame Sampling", "318 raw frames → 80 keyframes by visual overlap")
    y = top - 0.02
    y -= panel("s1_timeline", 0.30, y, 4.25) + 0.12
    hl = panel("s1_sift_matches", 0.30, y, 2.05, caption="SIFT + Lowe ratio test", cap_size=7)
    hr = panel("s1_homography_overlap", 2.48, y, 2.07, caption="RANSAC homography → overlap α", cap_size=7)
    # Stage (ii)
    band(4.95, r1y, 5.0, r1h, BAND["ii"], "Stage (ii): Feed-Forward π³ Inference", "one pass → per-view point maps, confidence, camera poses")
    y = top - 0.02
    hs = panel("s1_static_vs_dynamic", 5.08, y, 1.75, caption="Static / dynamic input frames", cap_size=7)
    node(6.98, top - 1.05, 0.95, 0.75, "π³
DINOv2-L enc.
+ transformer dec.", fc="#dbe7fb", ec="#4a7fd6", size=7)
    arrow(6.83, top - 0.68, 6.98, top - 0.68)
    arrow(7.93, top - 0.68, 8.08, top - 0.68)
    panel("s2_local_pointmaps", 8.10, y, 1.75, 1.45, caption="Per-view point maps", cap_size=7)
    y2 = top - 2.55
    panel("s2_confidence", 7.00, y2, 1.40, 1.15, caption="Confidence threshold", cap_size=7)
    panel("s2_world_cloud_initial", 8.45, y2, 1.40, 1.15, caption="World points + initial T_t", cap_size=7)
    # Stage (iii)
    band(10.15, r1y, 4.7, r1h, BAND["iii"], "Stage (iii): Static–Dynamic Decomposition", "mask-guided split of every frame's points")
    panel("s2_static_dynamic_split", 10.28, top - 0.02, 4.45, 2.3)
    panel("s2_dynamic_frames", 10.28, top - 2.5, 4.45, 1.5, caption="Per-frame dynamic clouds D_t (person) over the static background S", cap_size=7)

    # ───────────────────────── row 2 (y 0.15 .. 4.65) ─────────────────────────
    r2y, r2h = 0.15, 5.2
    band(0.15, r2y, 9.8, r2h, BAND["iv"], "Stage (iv): Per-Frame Alignment via Trimmed ICP", "source D_t → target S: correspondences → keep closest 80 % → weighted Kabsch, iterated")
    # ICP loop nodes
    ny = r2y + r2h - 1.25
    node(0.35, ny, 1.75, 0.62, "Find\ncorrespondences", size=7.5)
    node(2.35, ny, 1.75, 0.62, "Trimming\n(closest 80 %)", size=7.5)
    node(4.35, ny, 1.75, 0.62, "Weighted\nKabsch fit", size=7.5)
    arrow(2.10, ny + 0.31, 2.35, ny + 0.31); arrow(4.10, ny + 0.31, 4.35, ny + 0.31)
    arrow(5.22, ny, 5.22, ny - 0.18, lw=1.2); arrow(5.22, ny - 0.18, 1.22, ny - 0.18, lw=1.2, arrowstyle="-"); arrow(1.22, ny - 0.18, 1.22, ny, lw=1.2)
    label(3.22, ny - 0.30, "iterate until convergence", size=7, color=MUTED)
    panel("s3_icp_before_after", 0.30, ny - 0.45, 6.05, 3.2)
    panel("s3_floor_smpl", 6.50, r2y + r2h - 0.5, 3.30, 4.2, caption="Metric scale + floor from PromptHMR SMPL alignment", cap_size=7)
    # Output
    band(10.15, r2y, 4.7, r2h, BAND["out"], "Final Output: 4D Scene", "(S, {(F_t, T_t)}_{t=1..T}) → 3-D boxes + world scene graphs")
    panel("s1_masks",               10.28, r2y + 3.05, 1.60, 0.95, "Pre-computed masks", 7)
    node(12.05, r2y + 3.15, 1.05, 0.72, "Mask-aware\nscene merging", fc="#fdf3d0", ec="#f2b632", size=7.2)
    panel("s3_scene_4d",            10.28, r2y + 0.65, 4.45, 2.35, None)
    panel("s3_boxes_over_time",     13.20, r2y + 3.05, 1.55, 0.95, "3-D boxes over time", 7)
    label(12.50, r2y + 0.45, "4D scene of 00T1E: static background S, refined poses T_t, per-frame person geometry", size=7.5, color=MUTED)

    # ───────────────────────── inter-stage arrows ─────────────────────────
    mid = r1y + r1h / 2 - 0.3
    arrow(4.70, mid, 4.95, mid)                    # (i) -> (ii)
    arrow(9.95, mid, 10.15, mid)                   # (ii) -> (iii)
    arrow(12.5, r1y - 0.02, 12.5, r2y + r2h + 0.02)   # (iii) -> final
    arrow(3.5, r1y - 0.02, 3.5, r2y + r2h + 0.02)     # (i)/(ii) -> (iv)
    arrow(9.95, r2y + r2h / 2, 10.15, r2y + r2h / 2)  # (iv) -> final
    fig.text(fx(7.4), fy((r1y + r2y + r2h) / 2), "S  and  {D_t, T_t}", ha="center", va="center", fontsize=8, color=MUTED, style="italic")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs" / "scene_pipeline" / "figure" / "WorldSGG4DScenePipeline_00T1E"))
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=250)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    f = build()
    for ext in a.formats:
        f.savefig(f"{out}.{ext}", dpi=a.dpi, facecolor="white")
        print("wrote", f"{out}.{ext}")


if __name__ == "__main__":
    main()
