"""Small per-subsection figures for the supplementary (placed with wrapfig, text flowing around them).

Each figure is built from already-rendered panels of the real intermediates (scene pipeline for
0DJ6R, box pipeline for 00T1E), optionally cropped to a horizontal fraction of a panel, and laid out
in rows.  Output: assets/figures/architecture/wrap_figs/<name>.pdf (+ .png preview); panels stay vector.

    python scripts/paper_figures/compose_wrap_figs.py [name ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcompose import Canvas, resolve  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_arch = REPO / "assets" / "figures" / "architecture"
SCN = (_arch / "scene_pipeline" / "panels_0DJ6R") if (_arch / "scene_pipeline" / "panels_0DJ6R").exists() else (REPO / "outputs" / "scene_pipeline" / "panels_0DJ6R")
BOX = (_arch / "bbox_pipeline" / "panels_00T1E") if (_arch / "bbox_pipeline" / "panels_00T1E").exists() else (REPO / "outputs" / "bbox_pipeline" / "panels_00T1E")
SEM = (_arch / "semantic_pipeline") if (_arch / "semantic_pipeline").exists() else (REPO / "outputs" / "semantic_pipeline")
OUT = (_arch / "wrap_figs") if (_arch / "wrap_figs").exists() else (REPO / "outputs" / "wrap_figs")
OUT.mkdir(parents=True, exist_ok=True)


# name -> (width in inches, rows); a row is a list of (path, xfrac)
SPECS = {
    # ---------------- scene reconstruction (0DJ6R)
    "wrap_sampling": (3.0, [[(SCN / "s1_timeline.png",)]]),
    "wrap_pi3_inputs": (3.0, [[(SCN / "s1_static_vs_dynamic.png",)]]),
    "wrap_confidence": (3.0, [[(SCN / "s2_confidence.png",)]]),
    # RGB input of view 22 next to the 3-D result it produces
    "wrap_decomposition": (3.0, [[(SCN / "s1_rgb_static_k22.png",), (SCN / "s2_static_dynamic_split.png", (0.0, 0.5))],
                                 [(SCN / "s1_rgb_dyn_k22.png",), (SCN / "s2_static_dynamic_split.png", (0.5, 1.0))]]),
    "wrap_icp": (3.0, [[(SCN / "s1_rgb_dyn_k22.png",), (SCN / "s3_icp_before_after.png", (0.0, 0.355)),
                        (SCN / "s3_icp_before_after.png", (0.355, 0.68))]]),
    "wrap_merging": (3.0, [[(SCN / "s1_masks.png",)]]),
    # Stage (v) of the scene figure, for the floor-alignment section (regular figure, not wrapped);
    # RGB of the posed keyframe (22 -> file 000427.png, panels_floor_rgb.py) next to its 3-D view
    "prompthmr_0DJ6R": (6.0, [[(SCN / "s3_rgb_smpl_k22.png",), (SCN / "s3_floor_smpl_posed.png",),
                               (SCN / "s3_smpl_over_time.png",)]], 0.15),
    # ---------------- geometric annotation (00T1E)
    "wrap_detection": (3.0, [[(BOX / "b1_detection.png",)]]),
    "wrap_sam2": (3.0, [[(BOX / "b2_sam2.png",)]]),
    "wrap_floor": (3.0, [[(BOX / "b0_rgb_frame251.png",), (BOX / "b0_floor_body.png",)]], 0.08),   # RGB of frame 251
    "wrap_erosion": (3.0, [[(BOX / "b3_erosion.png",)]]),
    "wrap_obb": (3.0, [[(BOX / "b4_obb.png", (0, 1), (0.668, 1.0))]]),
    "wrap_temporal": (3.0, [[(BOX / "b5_timeline.png",)], [(BOX / "b6_final_boxes.png",)]]),
    "wrap_world_boxes": (3.0, [[(BOX / "b6_final_boxes.png",)]]),
    # ---------------- semantic annotation (00T1E)
    "wrap_sem_timeline": (3.0, [[(SEM / "panels_00T1E" / "m4_label_timeline.png",)]]),
    "wrap_sem_clip": (3.0, [[(SEM / "panels_00T1E" / "m1_clip_input.png",)]]),
    "wrap_sem_verification": (3.0, [[(SEM / "panels_00T1E" / "m3_candidates_verification.png",)]]),
    "wrap_sem_graph": (3.0, [[(SEM / "panels_00T1E" / "m2_event_graph.png",)]]),
    "sem_combined": (7.2, [[(SEM / "panels_00T1E" / "m1_clip_input.png",), (SEM / "panels_00T1E" / "m2_event_graph.png",),
                            (SEM / "panels_00T1E" / "m3_candidates_verification.png",)],
                           [(SEM / "panels_00T1E" / "m4_label_timeline.png",), (SEM / "panels_00T1E" / "m5_corrections.png",)]]),
}


def build(name, width, rows, gap=0.05):
    """Rows of panels, each row spanning the full width with a shared height; panels are placed as vector
    PDFs (vcompose), cropped to (xfrac, yfrac) and trimmed of transparent margins."""
    specs = []
    for row in rows:
        r = []
        for spec in row:
            path, xfrac, yfrac = (tuple(spec) + ((0.0, 1.0), (0.0, 1.0)))[:3]
            stem = Path(path).with_suffix("")
            _, win, asp = resolve(stem, crop=(xfrac[0], yfrac[0], xfrac[1], yfrac[1]), autocrop=True)
            r.append((stem, win, asp))
        # crops of one panel in the same row share a vertical window so they keep one scale
        for stem in {st for st, _, _ in r}:
            idx = [i for i, (st, _, _) in enumerate(r) if st == stem]
            if len(idx) > 1:
                t = min(r[i][1][1] for i in idx); b = max(r[i][1][3] for i in idx)
                for i in idx:
                    l, _, rr, _ = r[i][1]
                    _, win, asp = resolve(stem, crop=(l, t, rr, b))
                    r[i] = (stem, win, asp)
        specs.append(r)
    heights = [(width - gap * (len(r) - 1)) / sum(a for _, _, a in r) for r in specs]
    H = sum(heights) + gap * (len(rows) - 1)
    cv = Canvas(width, H)
    y = H
    for r, rh in zip(specs, heights):
        x = 0.0
        for stem, win, asp in r:
            cv.place(stem, x, y, rh * asp, rh, crop=win)
            x += rh * asp + gap
        y -= rh + gap
    cv.save(OUT / name, png_dpi=300)
    print(f"{name}: {width:.2f} x {H:.2f} in")


if __name__ == "__main__":
    names = sys.argv[1:] or list(SPECS)
    for n in names:
        w, rows, *gap = SPECS[n]
        build(n, w, rows, *gap)
