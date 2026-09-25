"""Stage (ii) feed-forward pi3 inference and Stage (iii) static/dynamic decomposition
panels of the 4D scene-pipeline figure (video 00T1E).

Writes ``outputs/scene_pipeline/panels/s2_*.png`` (transparent, 300 dpi).

Verified conventions
--------------------
* The video is **portrait** (raw frames 270x480, pi3 inputs 378x672).  ``pi3_image_hw``
  = (H, W) = (672, 378) and the stored images are upright, so ``pi3_pix_k`` unravels
  row-major with ``row = pix // 378``, ``col = pix % 378`` (row correlates +0.96 with the
  camera-frame y of ``pi3_local_k`` and -0.95 with final-frame z; the transposed unravel
  gives ~0 column correlation).  Flat indexing of an (H, W) mask therefore matches.
* SAM2 combined masks are binary {0, 255} at 480x270 (person + touching objects such as
  the laptop).  The person is the largest connected component in every frame.
* All 60k stored points per view already satisfy ``conf > pi3_conf_thr`` (the bundle was
  confidence-filtered at extraction), so the "after threshold" sub-panel has no dropped
  points to show.

Usage::

    python scripts/paper_figures/panels_stage2.py            # all panels
    python scripts/paper_figures/panels_stage2.py --only 4 5  # a subset
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D

from scene_common import (
    BUNDLE_DIR, COL, VIEW, bundle, to_final, pose_to_final, fig3d, set_equal, scatter,
    draw_frustum, save,
)

RNG = np.random.default_rng(0)
FAINT = "#b9bec7"          # faint static context cloud
TIME_CMAP = "plasma"       # sequential colormap for view index (time)


# ---- helpers ----------------------------------------------------------------
def kept_views() -> list[int]:
    return [int(k) for k in bundle()["pi3_view_ids"]]


def raw_frame_id(k: int) -> int:
    """1-indexed raw-frame file id shown by pi3 clip k."""
    return int(bundle()["sampled_idx"][k]) + 1


@lru_cache(maxsize=None)
def person_mask_flat(k: int) -> np.ndarray:
    """Boolean person mask of pi3 clip k, resized to the pi3 image and flattened row-major.

    The SAM2 mask belongs to raw frame sampled_idx[k] (visually identical to the pi3
    input sampled_idx[k]+1).  Person = largest connected component of the binary mask.
    """
    b = bundle()
    H, W = (int(v) for v in b["pi3_image_hw"])
    si = int(b["sampled_idx"][k])
    m = np.array(Image.open(BUNDLE_DIR / "masks" / f"{si:06d}.png")) > 0
    if m.ndim == 3:
        m = m[..., 0]
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros(H * W, bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    person = lab == (int(np.argmax(sizes)) + 1)
    person = np.array(Image.fromarray(person.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
    return person.ravel()


def union_split():
    """Union cloud in the final frame split into static / dynamic (person) points.

    Returns dict with P (final xyz), C (uint8), view, dyn (bool mask over the union).
    """
    b = bundle()
    P = to_final(b["pi3_all_points"])
    C = b["pi3_all_colors"]
    view = b["pi3_all_view"].astype(int)
    pix = b["pi3_all_pix"]
    dyn = np.zeros(len(P), bool)
    for v in np.unique(view):
        sel = view == v
        dyn[sel] = person_mask_flat(int(v))[pix[sel]]
    return dict(P=P, C=C, view=view, dyn=dyn)


def desaturate(C: np.ndarray, keep=0.35, tint=None, tint_w=0.0) -> np.ndarray:
    """uint8 RGB -> float RGB with most of the chroma removed (optionally tinted)."""
    c = np.asarray(C, np.float64) / 255.0
    luma = (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2])[:, None]
    out = keep * c + (1 - keep) * luma
    if tint is not None:
        t = np.array(mcolors.to_rgb(tint))[None]
        out = (1 - tint_w) * out + tint_w * t
    return np.clip(out, 0, 1)


def sub(idx_n: int, n: int) -> np.ndarray:
    if idx_n <= n:
        return np.arange(idx_n)
    return np.sort(RNG.choice(idx_n, n, replace=False))


def ax3d(fig, pos, view=VIEW):
    ax = fig.add_subplot(*pos, projection="3d")
    ax.view_init(**view)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    return ax


def fit(ax, P: np.ndarray, zoom=1.5, pad=0.05):
    """set_equal + zoom the 3-D box so the cloud fills the (axis-less) axes."""
    set_equal(ax, P, pad=pad)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
    except TypeError:
        pass


def cam_display(L: np.ndarray) -> np.ndarray:
    """Camera frame (x right, y down, z forward) -> display frame (x right, depth, up)."""
    L = np.asarray(L, np.float64)
    return np.stack([L[:, 0], L[:, 2], -L[:, 1]], 1)


CAM_VIEW = dict(elev=14, azim=-78)   # look along +depth (= camera +z) with slight elevation


# ---- panels -----------------------------------------------------------------
def p1_input_frames():
    ks = kept_views()
    fig, axes = plt.subplots(2, 4, figsize=(5.6, 5.4))
    fig.patch.set_alpha(0)
    for ax, k in zip(axes.ravel(), ks):
        im = np.array(Image.open(BUNDLE_DIR / f"pi3_view_{k:02d}.jpg"))
        if im.shape[0] < im.shape[1]:          # stored landscape -> rotate to upright portrait
            im = np.rot90(im, -1)
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(f"clip {k}  ·  frame {raw_frame_id(k):06d}", fontsize=7.5, pad=2)
    fig.subplots_adjust(wspace=0.04, hspace=0.08, left=0.01, right=0.99, top=0.96, bottom=0.01)
    save(fig, "s2_input_frames")


def p2_local_pointmaps():
    b = bundle()
    ks = [0, 22, 43, 65]
    fig = plt.figure(figsize=(6.4, 2.4))
    fig.patch.set_alpha(0)
    allD = np.concatenate([cam_display(b[f"pi3_local_{k}"]) for k in ks])
    for i, k in enumerate(ks):
        ax = ax3d(fig, (1, 4, i + 1), CAM_VIEW)
        D = cam_display(b[f"pi3_local_{k}"])
        idx = sub(len(D), 40000)
        scatter(ax, D[idx], b[f"pi3_colors_{k}"][idx], s=0.3)
        fit(ax, allD, zoom=1.8)
        ax.set_title(f"view {k}", fontsize=8, y=-0.1)
    fig.suptitle("Per-view local point maps (camera frame)", fontsize=9, y=0.995)
    fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=0.95, bottom=0.07)
    save(fig, "s2_local_pointmaps")


def p3_confidence():
    b = bundle()
    k = 43
    thr = float(b["pi3_conf_thr"])
    D = cam_display(b[f"pi3_local_{k}"])
    conf = b[f"pi3_conf_{k}"].astype(np.float32)
    keep = conf > thr
    idx = sub(len(D), 45000)
    fig = plt.figure(figsize=(5.2, 2.6))
    fig.patch.set_alpha(0)
    ax = ax3d(fig, (1, 2, 1), CAM_VIEW)
    sc = ax.scatter(D[idx, 0], D[idx, 1], D[idx, 2], c=conf[idx], cmap="viridis", s=0.35,
                    linewidths=0, depthshade=False, vmin=thr, vmax=float(np.percentile(conf, 99)))
    fit(ax, D, zoom=1.7)
    ax.set_title("pi3 confidence", fontsize=8.5, y=-0.1)
    cb = fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.0, fraction=0.03, aspect=25)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    ax = ax3d(fig, (1, 2, 2), CAM_VIEW)
    drop = idx[~keep[idx]]
    if len(drop):
        scatter(ax, D[drop], color="#c9ccd1", s=0.3, alpha=0.35)
    kp = idx[keep[idx]]
    scatter(ax, D[kp], b[f"pi3_colors_{k}"][kp], s=0.35)
    fit(ax, D, zoom=1.7)
    ax.set_title(f"kept: conf > τ = {thr:.3f}", fontsize=8.5, y=-0.1)
    fig.suptitle("Confidence thresholding", fontsize=9, y=0.995)
    fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=0.95, bottom=0.07)
    save(fig, "s2_confidence")


def p4_world_cloud_initial():
    b = bundle()
    P = to_final(b["pi3_all_points"])
    C = b["pi3_all_colors"]
    idx = sub(len(P), 200000)
    fig, ax = fig3d(size=(4.2, 3.0))
    scatter(ax, P[idx], C[idx], s=0.25, alpha=0.85)
    cams = []
    for k in kept_views():
        T = pose_to_final(b["pi3_poses"][k])
        draw_frustum(ax, T, COL["cam_init"], scale=0.3, lw=0.8)
        cams.append(T[:3, 3])
    fit(ax, np.concatenate([P[idx], np.array(cams)]), zoom=1.35)
    ax.set_title("Dynamic world point clouds + initial poses", fontsize=9, y=0.95)
    save(fig, "s2_world_cloud_initial")


def p5_static_dynamic_split():
    U = union_split()
    P, C, view, dyn = U["P"], U["C"], U["view"], U["dyn"]
    st = np.flatnonzero(~dyn)
    dy = np.flatnonzero(dyn)
    st_idx = st[sub(len(st), 150000)]
    dy_idx = dy[sub(len(dy), 80000)]
    print(f"union {len(P)}  static {len(st)}  dynamic {len(dy)} ({100 * len(dy) / len(P):.1f}%)")
    box = np.concatenate([P[st_idx], P[dy_idx]])
    fig = plt.figure(figsize=(7.0, 2.9))
    fig.patch.set_alpha(0)
    # static
    ax = ax3d(fig, (1, 2, 1))
    scatter(ax, P[st_idx], desaturate(C[st_idx], keep=0.45, tint=COL["static"], tint_w=0.18), s=0.25, alpha=0.85)
    fit(ax, box, zoom=1.35)
    ax.set_title("Static background S", fontsize=9, y=0.93)
    # dynamic, coloured by view index (time)
    ax = ax3d(fig, (1, 2, 2))
    ctx = st[sub(len(st), 60000)]
    scatter(ax, P[ctx], color=FAINT, s=0.15, alpha=0.18)
    nv = int(view.max()) + 1
    sc = ax.scatter(P[dy_idx, 0], P[dy_idx, 1], P[dy_idx, 2], c=view[dy_idx], cmap=TIME_CMAP,
                    vmin=0, vmax=nv - 1, s=0.4, linewidths=0, depthshade=False, alpha=0.95)
    fit(ax, box, zoom=1.35)
    ax.set_title("Per-frame dynamic $D_t$", fontsize=9, y=0.93)
    cb = fig.colorbar(sc, ax=ax, shrink=0.35, pad=0.0, fraction=0.025, aspect=22, ticks=[0, nv - 1])
    cb.ax.set_yticklabels(["t = 0", f"t = {nv - 1}"], fontsize=6)
    cb.ax.tick_params(length=2)
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
    save(fig, "s2_static_dynamic_split")


def p6_static_vggt():
    b = bundle()
    V = b["vggt_points"]
    C = b["vggt_colors"] if "vggt_colors" in b.files else None
    D = cam_display(V)    # VGGT frame is camera-like (z forward, y down): show it like a photo
    fig, ax = fig3d(size=(3.4, 2.9), view=dict(elev=10, azim=-82))
    if C is not None:
        scatter(ax, D, C, s=0.2, alpha=0.7)
    else:
        scatter(ax, D, color=COL["static"], s=0.2, alpha=0.7)
    fit(ax, D, zoom=1.5)
    ax.set_title("Static scene background (VGGT)", fontsize=9, y=0.95)
    save(fig, "s2_static_vggt")


def p7_dynamic_frames():
    b = bundle()
    U = union_split()
    P, view, dyn = U["P"], U["view"], U["dyn"]
    st = np.flatnonzero(~dyn)
    ctx = st[sub(len(st), 60000)]
    ks = [0, 43, 65]
    per = {}
    box = [P[ctx]]
    for k in ks:
        Pk = to_final(b[f"pi3_points_{k}"])
        m = person_mask_flat(k)[b[f"pi3_pix_{k}"]]
        per[k] = (Pk[m], b[f"pi3_colors_{k}"][m], pose_to_final(b["pi3_poses"][k]))
        box.append(Pk[m]); box.append(per[k][2][:3, 3][None])
        print(f"view {k}: {m.sum()} person points")
    box = np.concatenate(box)
    fig = plt.figure(figsize=(7.2, 2.6))
    fig.patch.set_alpha(0)
    for i, k in enumerate(ks):
        ax = ax3d(fig, (1, 3, i + 1))
        scatter(ax, P[ctx], color=FAINT, s=0.15, alpha=0.18)
        Pm, Cm, T = per[k]
        scatter(ax, Pm, color=COL["dynamic"], s=0.5, alpha=0.95)
        draw_frustum(ax, T, COL["cam_init"], scale=0.3, lw=0.9)
        fit(ax, box, zoom=1.35)
        ax.set_title(f"$D_t$, view {k} (frame {raw_frame_id(k):06d})", fontsize=8.5, y=0.93)
    fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
    save(fig, "s2_dynamic_frames")


PANELS = {1: p1_input_frames, 2: p2_local_pointmaps, 3: p3_confidence, 4: p4_world_cloud_initial,
          5: p5_static_dynamic_split, 6: p6_static_vggt, 7: p7_dynamic_frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    a = ap.parse_args()
    for i, fn in PANELS.items():
        if a.only is None or i in a.only:
            fn()


if __name__ == "__main__":
    main()
