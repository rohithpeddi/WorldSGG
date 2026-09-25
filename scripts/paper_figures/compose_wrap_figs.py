"""Small per-subsection figures for the supplementary (placed with wrapfig, text flowing around them).

Each figure is built from already-rendered panels of the real intermediates (scene pipeline for
0DJ6R, box pipeline for 00T1E), optionally cropped to a horizontal fraction of a panel, and laid out
in rows.  Output: outputs/wrap_figs/<name>.pdf (+ .png preview).

    python scripts/paper_figures/compose_wrap_figs.py [name ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
_arch = REPO / "assets" / "figures" / "architecture"
SCN = (_arch / "scene_pipeline" / "panels_0DJ6R") if (_arch / "scene_pipeline" / "panels_0DJ6R").exists() else (REPO / "outputs" / "scene_pipeline" / "panels_0DJ6R")
BOX = (_arch / "bbox_pipeline" / "panels_00T1E") if (_arch / "bbox_pipeline" / "panels_00T1E").exists() else (REPO / "outputs" / "bbox_pipeline" / "panels_00T1E")
SEM = (_arch / "semantic_pipeline") if (_arch / "semantic_pipeline").exists() else (REPO / "outputs" / "semantic_pipeline")
OUT = (_arch / "wrap_figs") if (_arch / "wrap_figs").exists() else (REPO / "outputs" / "wrap_figs")
OUT.mkdir(parents=True, exist_ok=True)


def load(path, xfrac=(0.0, 1.0), yfrac=(0.0, 1.0), trim=True, pad=6):
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    im = im.crop((int(xfrac[0] * w), int(yfrac[0] * h), int(xfrac[1] * w), int(yfrac[1] * h)))
    if trim:
        box = im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
        if box:
            im = im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.size[0], box[2] + pad), min(im.size[1], box[3] + pad)))
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


# name -> (width in inches, rows); a row is a list of (path, xfrac)
SPECS = {
    # ---------------- scene reconstruction (0DJ6R)
    "wrap_sampling": (3.0, [[(SCN / "s1_timeline.png",)]]),
    "wrap_pi3_inputs": (3.0, [[(SCN / "s1_static_vs_dynamic.png",)]]),
    "wrap_confidence": (3.0, [[(SCN / "s2_confidence.png",)]]),
    "wrap_decomposition": (3.0, [[(SCN / "s2_static_dynamic_split.png",)]]),
    "wrap_icp": (3.0, [[(SCN / "s3_icp_before_after.png", (0.0, 0.37)), (SCN / "s3_icp_before_after.png", (0.37, 0.735))]]),
    "wrap_merging": (3.0, [[(SCN / "s1_masks.png",)]]),
    # ---------------- geometric annotation (00T1E)
    "wrap_detection": (3.0, [[(BOX / "b1_detection.png",)]]),
    "wrap_sam2": (3.0, [[(BOX / "b2_sam2.png",)]]),
    "wrap_floor": (3.0, [[(BOX / "b0_floor_body.png",)]]),
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
    imgs = [[load(*spec) for spec in row] for row in rows]
    # each row spans the full width; images in a row share the row height
    heights = []
    for row in imgs:
        rsum = sum(im.size[0] / im.size[1] for im in row)
        heights.append((width - gap * (len(row) - 1)) / rsum)
    H = sum(heights) + gap * (len(rows) - 1)
    fig = plt.figure(figsize=(width, H)); fig.patch.set_facecolor("white")
    y = H
    for row, rh in zip(imgs, heights):
        y -= rh
        x = 0.0
        for im in row:
            w = rh * im.size[0] / im.size[1]
            ax = fig.add_axes([x / width, y / H, w / width, rh / H]); ax.set_axis_off()
            ax.imshow(im, interpolation="lanczos")
            x += w + gap
        y -= gap
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"{name}: {width:.2f} x {H:.2f} in")


if __name__ == "__main__":
    names = sys.argv[1:] or list(SPECS)
    for n in names:
        w, rows = SPECS[n]
        build(n, w, rows)
