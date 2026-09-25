"""Compose the 4D scene-pipeline figure (supplementary Fig. sup:fig:scene_pipeline) from the empirical
panels rendered by panels_stage{1,2,3}.py for the video chosen with SCENE_VIDEO.

Five stage bands and the output:
  (i) frame sampling -> (ii) feed-forward pi3 -> (iii) static-dynamic decomposition
  (iv) trimmed-ICP alignment -> (v) PromptHMR human + floor alignment -> final 4D scene
Every picture is a real intermediate.  The panels are placed as vector PDFs (vcompose.Canvas), so the
text stays sharp and the pictures keep their native resolution; missing panels render as placeholders.

    SCENE_VIDEO=0DJ6R python scripts/paper_figures/compose_scene_pipeline.py [--out ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_common as SC  # noqa: E402
from vcompose import Canvas  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PANELS = SC.PANEL_DIR

W, H = 15.0, 11.25          # inches; scaled to \textwidth in LaTeX
INK = "#1f2933"; MUTED = "#4b5563"; HEAD_BG = "#e5e7eb"
BAND = {"i": "#eef1f5", "ii": "#e8f0fb", "iii": "#fbeee4", "iv": "#e6f4ee", "v": "#f3eefb", "out": "#f6f0e2"}
ARROW = dict(arrowstyle="-|>,head_length=5,head_width=3", color="#374151", lw=1.6,
             shrinkA=0, shrinkB=0, mutation_scale=1)
PI3 = "π³"
HEAD = 0.72                 # header + subtitle height inside a band (inches)

cv: Canvas = None


def fx(x): return x / W
def fy(y): return y / H


def band(x, y, w, h, color, title, sub=None):
    f = cv.base
    ax = f.add_axes([fx(x), fy(y), fx(w), fy(h)]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=color, ec="#cbd5e1", lw=0.8, transform=ax.transAxes, clip_on=False))
    hb = f.add_axes([fx(x + 0.08), fy(y + h - 0.38), fx(w - 0.16), fy(0.32)]); hb.set_axis_off()
    hb.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=HEAD_BG, ec="none", transform=hb.transAxes, clip_on=False))
    hb.text(0.5, 0.5, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    if sub:
        f.text(fx(x + w / 2), fy(y + h - 0.5), sub, ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")


def panel(name, x, ytop, w, h=None, caption=None, cap_size=8, crop=None, autocrop=True):
    """Place panel <name> with its top-left at (x, ytop) inches and width w (fitted into (w, h) when h
    is given).  Returns the height used, including the caption."""
    x0, y0, ww, hh = cv.place(PANELS / name, x, ytop, w, h, crop=crop, autocrop=autocrop)
    used = h if h else hh
    if caption:
        label(x + w / 2, ytop - used - 0.05, caption, size=cap_size, va="top")
        used += 0.2
    return used


def label(x, y, s, size=8.5, bold=False, color=INK, ha="center", va="center", rot=0, italic=False, box=False):
    kw = dict(bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none")) if box else {}
    cv.top.text(fx(x), fy(y), s, ha=ha, va=va, fontsize=size, color=color, fontweight="bold" if bold else "normal",
                rotation=rot, style="italic" if italic else "normal", **kw)


def arrow(x0, y0, x1, y1, **kw):
    cv.top.patches.append(FancyArrowPatch((fx(x0), fy(y0)), (fx(x1), fy(y1)), transform=cv.top.transFigure,
                                          **{**ARROW, **kw}))


def line(*pts, **kw):
    """Poly-line through pts (inches) ending in an arrow head."""
    seg = dict(arrowstyle="-", color=ARROW["color"], lw=ARROW["lw"], shrinkA=0, shrinkB=0)
    for (a, b), (c, d) in zip(pts[:-2], pts[1:-1]):
        arrow(a, b, c, d, **seg)
    arrow(*pts[-2], *pts[-1], **kw)


def node(x, y, w, h, text, fc="#ffffff", ec="#9ca3af", size=8.5, bold=False):
    ax = cv.top.add_axes([fx(x), fy(y), fx(w), fy(h)]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15", fc=fc, ec=ec, lw=0.9,
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=size, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.15)


def build():
    global cv
    cv = Canvas(W, H)
    b = SC.bundle()

    # ───────────────────────── row 1 ─────────────────────────
    r1y, r1h = 6.2, 4.9
    top = r1y + r1h - HEAD
    # Stage (i)
    band(0.15, r1y, 4.55, r1h, BAND["i"], "Stage (i): Adaptive Frame Sampling",
         f"{int(b['n_raw_frames'])} Raw Frames → {len(b['sampled_idx'])} Keyframes by Visual Overlap")
    y = top
    y -= panel("s1_timeline", 0.42, y, 4.0) + 0.12
    panel("s1_sift_matches", 0.30, y, 2.0, caption="SIFT + Lowe Ratio Test", cap_size=7.5)
    panel("s1_homography_overlap", 2.50, y, 2.05, caption="RANSAC Homography → Overlap α", cap_size=7.5)
    # Stage (ii)
    band(4.95, r1y, 5.0, r1h, BAND["ii"], f"Stage (ii): Feed-Forward {PI3} Inference",
         "One Pass → Per-View Point Maps, Confidence, Camera Poses")
    y = top
    hs = panel("s1_static_vs_dynamic", 5.08, y, 1.72, caption="Static / Dynamic Input Frames", cap_size=7.5)
    node(6.95, top - 1.05, 0.95, 0.75, f"{PI3}\nDINOv2-L Enc.\n+ Transformer\nDec.", fc="#dbe7fb", ec="#4a7fd6", size=7)
    arrow(6.80, top - 0.68, 6.95, top - 0.68)
    arrow(7.90, top - 0.68, 8.05, top - 0.68)
    panel("s2_local_pointmaps", 8.05, y + 0.02, 1.85)
    y2 = top - hs - 0.15
    floor_y = r1y + 0.08                  # keep both 3-D panels inside the band
    panel("s2_confidence", 5.08, y2 - 0.1, 2.3, h=min(2.3, y2 - 0.1 - floor_y))
    wy = top - 1.2                        # below the pi3 node and the local point-map strip
    panel("s2_world_cloud_initial", 7.45, wy, 2.45, h=min(2.9, wy - floor_y))
    # Stage (iii)
    band(10.15, r1y, 4.7, r1h, BAND["iii"], "Stage (iii): Static–Dynamic Decomposition",
         "Static Pass on Inpainted Frames + Masked Dynamic Foreground")
    y = top
    y -= panel("s2_static_dynamic_split", 10.28, y, 4.45) + 0.2
    panel("s2_dynamic_frames", 10.28, y, 4.45,
          caption=r"Per-Frame Dynamic Foreground (Person) over the Static Background $\mathcal{S}$", cap_size=7.5)

    # ───────────────────────── row 2 ─────────────────────────
    r2y, r2h = 0.15, 5.6
    t2 = r2y + r2h - HEAD
    # Stage (iv): trimmed ICP
    band(0.15, r2y, 6.4, r2h, BAND["iv"], "Stage (iv): Per-Frame Alignment via Trimmed ICP",
         r"Source $\mathcal{D}_t^{\rm fg}$ → Target $\mathcal{S}$: Correspondences → Keep Closest 80% → Weighted Kabsch")
    ny = t2 - 1.1
    node(0.45, ny, 1.75, 0.62, "Find\nCorrespondences")
    node(2.45, ny, 1.75, 0.62, "Trimming\n(Closest 80%)")
    node(4.45, ny, 1.75, 0.62, "Weighted\nKabsch Fit")
    arrow(2.20, ny + 0.31, 2.45, ny + 0.31); arrow(4.20, ny + 0.31, 4.45, ny + 0.31)
    line((5.32, ny), (5.32, ny - 0.2), (1.32, ny - 0.2), (1.32, ny), lw=1.2)
    label(3.32, ny - 0.34, "Iterate Until Convergence", size=7.5, color=MUTED, italic=True)
    icp_top = ny - 0.6
    hi = panel("s3_icp_before_after", 0.30, icp_top - 0.15, 6.1)
    label(3.35, icp_top - 0.15 - hi - 0.3, r"Trimming Ratio $\rho = 0.8$  ·  Max. Iterations $100$  ·  Tolerance $10^{-5}$",
          size=7.5, color=MUTED)
    # Stage (v): PromptHMR human + floor alignment
    band(6.75, r2y, 3.2, r2h, BAND["v"], "Stage (v): PromptHMR Alignment",
         "SMPL-X Human + Floor Plane → Metric Scale")
    hp = panel("s3_floor_smpl_posed", 6.85, t2 - 0.05, 3.0, h=2.0, caption="One Frame: SMPL-X Mesh on the π³ Points", cap_size=7.5)
    panel("s3_smpl_over_time", 6.85, t2 - 0.05 - hp - 0.12, 3.0, h=t2 - 0.05 - hp - 0.12 - (r2y + 0.75),
          caption="Meshes over Time on the Fitted Floor", cap_size=7.5)
    label(8.35, r2y + 0.3, "Similarity $(s, R, t)$: Floor Frame ↔ " + PI3 + " World", size=7.5, color=MUTED)
    # Output
    band(10.15, r2y, 4.7, r2h, BAND["out"], "Final Output: 4D Scene",
         r"$(\mathcal{S}, \{(\mathcal{F}_t, T_t)\}_{t=1}^{T})$ → 3D Boxes + World Scene Graphs")
    hm = panel("s1_masks", 10.28, t2, 1.45, caption="Pre-Computed Masks", cap_size=7.5)
    node(10.4, t2 - hm - 0.8, 1.2, 0.62, "Mask-Aware\nScene Merging", fc="#fdf3d0", ec="#f2b632", size=7.5)
    arrow(11.0, t2 - hm - 0.02, 11.0, t2 - hm - 0.18, lw=1.2)
    arrow(11.6, t2 - hm - 0.49, 11.85, t2 - hm - 0.49)
    hero = panel("s3_scene_4d", 11.85, t2 + 0.02, 2.9)
    y = t2 - max(hero, hm + 0.85) - 0.12
    panel("s3_boxes_over_time", 10.28, y, 4.45, h=y - (r2y + 0.35),
          caption="Use Case: Corrected 3D Object Boxes over Time in the 4D Scene", cap_size=7.5)

    # ───────────────────────── inter-stage arrows ─────────────────────────
    mid = r1y + r1h / 2 - 0.3
    arrow(4.71, mid, 4.94, mid)                         # (i) -> (ii)
    arrow(9.96, mid, 10.14, mid)                        # (ii) -> (iii)
    mid2 = r2y + r2h / 2
    arrow(6.56, mid2, 6.74, mid2)                       # (iv) -> (v)
    arrow(9.96, mid2, 10.14, mid2)                      # (v) -> final
    # (iii) -> (iv): down out of (iii), left along the gap between the rows, down into (iv)
    gy = (r1y + r2y + r2h) / 2
    line((10.75, r1y), (10.75, gy), (3.35, gy), (3.35, r2y + r2h))
    label(7.05, gy, r"  $\mathcal{S}$  and  $\{\mathcal{D}_t^{\rm fg}, T_t\}$  ", size=8.5, color=MUTED, box=True)
    # (iii) -> final: the masks and S feed the merging
    arrow(13.2, r1y, 13.2, r2y + r2h)
    return cv


def main():
    _arch_fig = REPO / "assets" / "figures" / "architecture" / "scene_pipeline" / "figure"
    _def_out = (_arch_fig if _arch_fig.parent.exists() else (REPO / "outputs" / "scene_pipeline" / "figure")) / f"WorldSGG4DScenePipeline_{SC.VIDEO}"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_def_out))
    ap.add_argument("--dpi", type=int, default=250, help="PNG preview resolution")
    a = ap.parse_args()
    build().save(Path(a.out), png_dpi=a.dpi)


if __name__ == "__main__":
    main()
