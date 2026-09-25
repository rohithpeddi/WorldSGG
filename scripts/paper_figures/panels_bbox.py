"""Panels for the 3D bounding-box pipeline figure (supplementary Fig. sup:fig:bbox_pipeline),
rendered from the real intermediates bundled by extract_bbox_pipeline.py (video 00T1E by default).

    SCENE_VIDEO=00T1E python scripts/paper_figures/panels_bbox.py [det sam2 erosion volumes obb]

Conventions
-----------
* Raw frames and SAM2 masks are 270x480 (W x H); pi3 predictions are 378x672, i.e. x1.4.
* ``gt_bbox_xyxy`` in the box pickles is already at pi3 resolution.
* The pipeline's masked extraction = (2D box) AND (SAM2 image mask OR video mask), eroded with an
  elliptical kernel of size k, then pi3 confidence >= max(1e-3, 5th percentile).  Re-running this on
  frame 000087 reproduces the stored candidate point counts to within ~2% for person / shoe / laptop,
  so the masks drawn here are the ones the pipeline used (bed, a static object, uses another mask path).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_common as SC  # noqa: E402
from scene_common import COL, LABEL_COL, save  # noqa: E402
SC.enable_title_case()      # all figure text in Title Case (supplementary style)

REPO = SC.REPO
_arch_bb = REPO / "assets" / "figures" / "architecture" / "bbox_pipeline"
BB = (_arch_bb / SC.VIDEO) if (_arch_bb / SC.VIDEO).exists() else (REPO / "outputs" / "bbox_pipeline" / SC.VIDEO)
SC.PANEL_DIR = (_arch_bb / f"panels_{SC.VIDEO}") if _arch_bb.exists() else (REPO / "outputs" / "bbox_pipeline" / f"panels_{SC.VIDEO}")
SC.PANEL_DIR.mkdir(parents=True, exist_ok=True)
PI3_SCALE = 1.4
TINY, SMALL = 6.5, 7.5
KSHOW = (0, 5, 10)


def frame(s):
    return np.asarray(Image.open(BB / "frames" / f"{s}.png").convert("RGB"))


def sam2(s, label, mode):
    p = BB / "sam2" / f"{s}__{label}__{mode}.png"
    return np.asarray(Image.open(p)) > 0 if p.exists() else None


def boxes3d(run="bbox_annotations_3d_obb"):
    return json.load(open(BB / "boxes_3d.json"))[run]["frames"]


def objects(s, run="bbox_annotations_3d_obb"):
    return boxes3d(run)[f"{s}.png"]["objects"]


def rgba(c, a):
    r, g, b = matplotlib.colors.to_rgb(c)
    return (r, g, b, a)


def overlay(img, masks, alpha=0.55):
    out = img.astype(float) / 255.0
    for m, c in masks:
        if m is None:
            continue
        rgb = np.array(matplotlib.colors.to_rgb(c))
        out[m] = (1 - alpha) * out[m] + alpha * rgb
    return out


def outline(ax, m, c, lw=0.6):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for q in cnts:
        if cv2.contourArea(q) < 12:
            continue
        q = q.reshape(-1, 2)
        ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color=c, lw=lw)


def _fig(w, h):
    f = plt.figure(figsize=(w, h)); f.patch.set_alpha(0); return f


def _img_ax(fig, rect, img):
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    h, w = a.shape[:2]
    ax = fig.add_axes(rect)   # sharpened 2x copy, drawn in the original pixel coordinates of the overlays
    ax.imshow(SC.crisp(a[..., :3]), interpolation="lanczos", extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_edgecolor(COL["muted"])
    return ax


# ---------------------------------------------------------------- detection + GT fusion
def panel_detection(s="000163"):
    det = json.load(open(BB / "detections.json"))[f"{s}.png"]
    gt = json.load(open(BB / "ag_gt.json"))[s]
    img = frame(s)
    fig = _fig(3.3, 2.75)
    ax_l = _img_ax(fig, [0.0, 0.1, 0.49, 0.82], img)
    ax_r = _img_ax(fig, [0.51, 0.1, 0.49, 0.82], img)
    # left: Action Genome ground truth (x, y, w, h)
    for o in gt["objects"] or []:
        if not o.get("visible", True) or o.get("bbox") is None:
            continue
        x, y, w, h = o["bbox"]
        ax_l.add_patch(Rectangle((x, y), w, h, fill=False, ec=LABEL_COL[o["class"]], lw=1.1))
        ax_l.text(x + 2, y - 3, o["class"], color=LABEL_COL[o["class"]], fontsize=5.5, fontweight="bold")
    pb = np.array(gt["person"]["bbox"]).reshape(-1, 4) if gt["person"] else []
    for x1, y1, x2, y2 in pb:
        ax_l.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=LABEL_COL["person"], lw=1.1))
        ax_l.text(x1 + 2, y1 - 3, "person", color=LABEL_COL["person"], fontsize=5.5, fontweight="bold")
    ax_l.set_title("Action Genome GT", fontsize=SMALL, pad=2)
    # right: fused detections; GT carries pseudo-score 1.001, detector boxes keep their score
    n_det = n_gt = 0
    placed = []
    for (x1, y1, x2, y2), sc, lab in zip(det["boxes"], det["scores"], det["labels"]):
        is_gt = sc > 1.0
        n_gt += is_gt; n_det += not is_gt
        c = LABEL_COL[lab]
        ax_r.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=1.1, ls="-" if is_gt else (0, (3, 1.5))))
        below = (not is_gt) and any(abs(y1 - q[1]) < 14 and abs(x1 - q[0]) < 60 for q in placed)
        ty, va = (y2 + 2, "top") if below else (y1 - 3, "baseline")
        ax_r.text(x1 + 2, ty, f"{lab} {'GT' if is_gt else f'{sc:.2f}'}", color=c, fontsize=5.5, fontweight="bold", va=va)
        placed.append((x1, y1))
    ax_r.set_title("Fused: GT (1.001) + Grounding DINO", fontsize=SMALL, pad=2)
    fig.text(0.5, 0.02, f"frame {int(s)}: {n_gt} GT boxes kept, {n_det} detector boxes added after class-wise NMS",
             ha="center", fontsize=TINY - 0.5, color=COL["muted"])
    save(fig, "b1_detection")


# ---------------------------------------------------------------- SAM2 image / video / union
def panel_sam2(s="000087", labels=("person", "laptop", "shoe")):
    img = frame(s)
    fig = _fig(3.4, 2.35)
    titles = ("image mode", "video mode", "union (final)")
    for j, mode in enumerate(("image", "video", "union")):
        masks = []
        for lab in labels:
            mi, mv = sam2(s, lab, "image"), sam2(s, lab, "video")
            m = mi if mode == "image" else mv if mode == "video" else (
                (mi if mi is not None else False) | (mv if mv is not None else False))
            masks.append((None if m is None or m is False else np.asarray(m, bool), LABEL_COL[lab]))
        ax = _img_ax(fig, [j * 0.335, 0.1, 0.325, 0.82], overlay(img, masks))
        for m, c in masks:
            if m is not None:
                outline(ax, m, c)
        ax.set_title(titles[j], fontsize=SMALL, pad=2)
    h = [Line2D([], [], color=LABEL_COL[l], lw=3, label=l) for l in labels]
    fig.legend(handles=h, loc="lower center", ncol=len(labels), fontsize=TINY, frameon=False, bbox_to_anchor=(0.5, -0.03),
               handlelength=1.2, columnspacing=1.0)
    save(fig, "b2_sam2")


# ---------------------------------------------------------------- masked extraction + multiscale erosion
def object_mask(s, o):
    Z = np.load(BB / f"pi3_{s}.npz")
    H, W = Z["conf"].shape
    x1, y1, x2, y2 = o["gt_bbox_xyxy"]
    box = np.zeros((H, W), bool)
    box[int(max(0, y1)):int(np.ceil(y2)), int(max(0, x1)):int(np.ceil(x2))] = True
    mm = None
    for t in ("image", "video"):
        a = sam2(s, o["label"], t)
        if a is not None:
            a = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
            mm = a if mm is None else (mm | a)
    return (box & mm if mm is not None else box), Z


def eroded(m, k):
    if k == 0:
        return m
    return cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def panel_erosion(s="000087", label="person"):
    o = next(o for o in objects(s) if o["label"] == label)
    m, Z = object_mask(s, o)
    img = Z["image"]; C = Z["conf"]
    thr = max(1e-3, float(np.percentile(C, 5)))
    stored = {c["kernel_size"]: c for c in o["candidates"]}
    x1, y1, x2, y2 = [int(round(v)) for v in o["gt_bbox_xyxy"]]
    pad = 12
    ys, xs = slice(max(0, y1 - pad), min(img.shape[0], y2 + pad)), slice(max(0, x1 - pad), min(img.shape[1], x2 + pad))
    fig = _fig(3.4, 1.55)
    n = len(KSHOW)
    for j, k in enumerate(KSHOW):
        mk = eroded(m, k) & (C >= thr)
        ring = m & ~eroded(m, k)
        crop = img[ys, xs].astype(float) / 255.0
        ov = crop.copy()
        c = np.array(matplotlib.colors.to_rgb(LABEL_COL[label]))
        r = np.array(matplotlib.colors.to_rgb(COL["cam_init"]))
        ov[mk[ys, xs]] = 0.45 * ov[mk[ys, xs]] + 0.55 * c
        ov[ring[ys, xs]] = 0.3 * ov[ring[ys, xs]] + 0.7 * r
        ax = _img_ax(fig, [j / n + 0.005, 0.2, 1 / n - 0.01, 0.68], ov)
        ax.set_title(f"k = {k} px", fontsize=SMALL, pad=2)
        ax.text(0.5, -0.04, f"{stored[k]['num_points']:,} pts", transform=ax.transAxes, ha="center", va="top",
                fontsize=TINY, color=COL["text"])
    fig.text(0.5, 0.005, f"{label}, frame {int(s)}: box ∩ SAM2 mask, elliptical erosion (red = stripped boundary)",
             ha="center", va="bottom", fontsize=TINY - 0.5, color=COL["muted"])
    save(fig, "b3_erosion")


def panel_volumes(s="000087"):
    """Candidate volume vs erosion size for every object of the frame (obb run), normalised by k = 0."""
    fig = _fig(1.9, 1.55)
    ax = fig.add_axes([0.24, 0.24, 0.72, 0.62]); ax.patch.set_alpha(0)
    for o in objects(s):
        c = sorted(o["candidates"], key=lambda q: q["kernel_size"])
        k = np.array([q["kernel_size"] for q in c]); v = np.array([q["volume"] for q in c])
        ax.plot(k, v / v[0], color=LABEL_COL[o["label"]], lw=1.0, marker="o", ms=1.8, label=o["label"])
        i = int(np.argmin(v))
        ax.plot(k[i], v[i] / v[0], marker="*", ms=6, color=LABEL_COL[o["label"]], mec="k", mew=0.3)
    ax.set_xlabel("erosion kernel k (px)", fontsize=TINY, labelpad=1)
    ax.set_ylabel("volume / volume(k=0)", fontsize=TINY, labelpad=1)
    ax.tick_params(labelsize=TINY - 0.5, length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=TINY - 1, frameon=False, handlelength=1.2, loc="lower left", labelspacing=0.2)
    ax.set_title("min-volume selection (★)", fontsize=SMALL, pad=2)
    save(fig, "b3_volumes")


# ---------------------------------------------------------------- 3D: points + box variants
def final_tf():
    F = json.load(open(BB / "final_4d.json"))
    A = np.array(F["world_to_final"]["A_world_to_final"]); o = np.array(F["world_to_final"]["origin_world"])
    return lambda P: (A @ (np.asarray(P, float) - o).T).T


def box_edges(c):
    """Edges of a cuboid given its 8 corners in any order: the sign pattern of the corners in the
    cuboid's own axes gives a 3-bit code; edges join codes that differ in exactly one bit."""
    c = np.asarray(c, float); m = c.mean(0)
    _, _, Vt = np.linalg.svd(c - m)
    code = ((c - m) @ Vt.T > 0).astype(int) @ np.array([1, 2, 4])
    return [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(int(code[i]) ^ int(code[j])).count("1") == 1]


def draw_cuboid(ax, c, color, lw=1.0, ls="-", alpha=1.0, z=3):
    c = np.asarray(c, float)
    for i, j in box_edges(c):
        ax.plot(*zip(c[i], c[j]), color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)


def panel_obb(s="000087", zoom_label="laptop"):
    tf = final_tf()
    Z = np.load(BB / f"pi3_{s}.npz"); P = Z["points"]; C = Z["conf"]; img = Z["image"]
    thr = max(1e-3, float(np.percentile(C, 5)))
    rng = np.random.default_rng(0)
    obs = objects(s)
    fig = plt.figure(figsize=(4.4, 3.6)); fig.patch.set_alpha(0)
    axes = [fig.add_axes([0.0, 0.34, 1.0, 0.66], projection="3d"), fig.add_axes([0.0, 0.0, 0.62, 0.38], projection="3d")]
    for ax in axes:
        ax.view_init(elev=24, azim=-60); ax.set_axis_off(); ax.patch.set_alpha(0); ax.computed_zorder = False
    ok = np.isfinite(P).all(-1) & (C >= thr)
    ctx = np.flatnonzero(ok.ravel()); ctx = rng.choice(ctx, min(40000, len(ctx)), replace=False)
    Pc = tf(P.reshape(-1, 3)[ctx]); Cc = img.reshape(-1, 3)[ctx] / 255.0
    per = {}
    for o in obs:
        if o["label"] == "bed":
            continue            # static object: its mask comes from the static pipeline, not bundled here
        m, _ = object_mask(s, o)
        m = eroded(m, 10) & ok
        per[o["label"]] = (tf(P[m]), o)
    ax = axes[0]
    ax.scatter(Pc[:, 0], Pc[:, 1], Pc[:, 2], c=0.55 * Cc + 0.45 * 0.82, s=0.15, linewidths=0, alpha=0.6,
               depthshade=False, zorder=1)
    for lab, (Q, o) in per.items():
        q = Q[rng.choice(len(Q), min(4000, len(Q)), replace=False)]
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], color=LABEL_COL[lab], s=0.5, linewidths=0, alpha=0.9, depthshade=False, zorder=2)
        draw_cuboid(ax, tf(o["obb_floor_parallel"]["corners_world"]), LABEL_COL[lab], lw=1.1)
    bed = next((o for o in obs if o["label"] == "bed"), None)
    if bed is not None:
        draw_cuboid(ax, tf(bed["obb_floor_parallel"]["corners_world"]), LABEL_COL["bed"], lw=1.0)
    lo, hi = np.percentile(Pc, 2, 0), np.percentile(Pc, 98, 0)
    ext = [lo, hi] + ([tf(bed["obb_floor_parallel"]["corners_world"])] if bed is not None else [])
    SC.set_equal(ax, np.vstack(ext), pct=(0, 100), zoom=1.12)
    ax.text2D(0.5, 0.97, "floor-parallel OBBs (final), frame 87", transform=ax.transAxes, fontsize=SMALL, ha="center", va="top")
    labs = list(per) + (["bed"] if bed is not None else [])
    h = [Line2D([], [], color=LABEL_COL[l], lw=1.5, label=l) for l in labs]
    ax.legend(handles=h, loc="upper left", fontsize=TINY - 0.5, frameon=False, handlelength=1.2, labelspacing=0.2,
              bbox_to_anchor=(0.02, 0.92))
    Q, o = per[zoom_label]
    ax = axes[1]
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], color=LABEL_COL[zoom_label], s=0.8, linewidths=0, alpha=0.6, depthshade=False, zorder=1)
    variants = [("aabb_floor_aligned", "floor-aligned AABB", ":", COL["muted"]),
                ("obb_arbitrary", "PCA OBB", "--", COL["cam_init"]),
                ("obb_floor_parallel", "floor-parallel OBB", "-", "k")]
    corners = []
    for key, name, ls, col in variants:
        c = tf(o[key]["corners_world"]); corners.append(c)
        draw_cuboid(ax, c, col, lw=1.0, ls=ls)
    SC.set_equal(ax, np.vstack(corners + [Q]), pct=(0, 100), zoom=1.2)
    h = [Line2D([], [], color=col, lw=1.0, ls=ls, label=name) for _, name, ls, col in variants]
    ax.legend(handles=h, loc="center left", fontsize=TINY, frameon=False, handlelength=2.0, labelspacing=0.3,
              bbox_to_anchor=(1.0, 0.5))
    ax.text2D(0.02, 0.98, f"{zoom_label}: box variants", transform=ax.transAxes, fontsize=SMALL, va="top")
    save(fig, "b4_obb")


# ---------------------------------------------------------------- temporal completion
def panel_timeline():
    from matplotlib.patches import Patch
    F = json.load(open(BB / "final_4d.json"))["frames"]
    det = boxes3d()
    frames = sorted(F)
    order = ["person", "laptop", "shoe", "bed", "doorway"]
    fig = plt.figure(figsize=(4.2, 1.45)); fig.patch.set_alpha(0)
    ax = fig.add_axes([0.13, 0.32, 0.85, 0.62]); ax.patch.set_alpha(0)
    for r, lab in enumerate(order):
        for c, f in enumerate(frames):
            o = next((x for x in F[f]["objects"] if x["label"] == lab), None)
            if o is None:
                continue
            detected = any(x["label"] == lab for x in det.get(f, {"objects": []})["objects"])
            m = o.get("world4d_fill_method")
            base = LABEL_COL[lab]
            hatch = None
            if detected:
                fc = rgba(base, 1.0)
            elif m in ("interpolation", "static_copy"):
                fc = rgba(base, 0.35)
            else:
                fc, hatch = (1, 1, 1, 0), "/////"
            ax.add_patch(Rectangle((c, r), 0.92, 0.84, fc=fc, ec=base if hatch else "none", lw=0.4, hatch=hatch))
    ax.set_xlim(-0.2, len(frames)); ax.set_ylim(len(order) - 0.05, -0.1)
    ax.set_yticks(np.arange(len(order)) + 0.42); ax.set_yticklabels(order, fontsize=TINY)
    ticks = list(range(0, len(frames), 7)) + [len(frames) - 1]
    ax.set_xticks(np.array(ticks) + 0.46); ax.set_xticklabels([str(int(frames[i][:6])) for i in ticks], fontsize=TINY - 0.5)
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("annotated frame", fontsize=TINY, labelpad=1)
    g = "#6b7280"
    h = [Patch(fc=rgba(g, 1.0), ec=g, lw=0.4, label="detected"),
         Patch(fc=rgba(g, 0.35), ec=g, lw=0.4, label="interpolated (dynamic) / copied (static)"),
         Patch(fc=(1, 1, 1, 0), ec=g, lw=0.4, hatch="/////", label="held from nearest")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=TINY - 0.8, frameon=False, bbox_to_anchor=(0.55, -0.04),
               handlelength=1.4, columnspacing=1.0)
    save(fig, "b5_timeline")


# ================================================================ clearer point-cloud rendering
import matplotlib.patheffects as pe
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HALO = [pe.Stroke(linewidth=3.2, foreground="white", alpha=0.9), pe.Normal()]
VIEW_F = dict(elev=24, azim=-60)


def clean_cloud(P, C, voxel=0.006, k=12, nstd=2.0, pct=(0.5, 99.5)):
    """Voxel-downsample (uniform density, colour = voxel mean), drop statistical outliers
    (mean k-NN distance > mean + nstd * std) and far-away stragglers outside a percentile box."""
    P = np.asarray(P, float); C = np.asarray(C, float)
    if C.max() > 1.0:
        C = C / 255.0
    ok = np.isfinite(P).all(1)
    P, C = P[ok], C[ok]
    lo, hi = np.percentile(P, pct[0], 0), np.percentile(P, pct[1], 0)
    m = np.all((P >= lo) & (P <= hi), 1); P, C = P[m], C[m]
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Pv = np.zeros((len(cnt), 3)); Cv = np.zeros((len(cnt), 3))
    np.add.at(Pv, inv, P); np.add.at(Cv, inv, C)
    Pv /= cnt[:, None]; Cv /= cnt[:, None]
    d, _ = cKDTree(Pv).query(Pv, k=k + 1)
    md = d[:, 1:].mean(1)
    keep = md < md.mean() + nstd * md.std()
    return Pv[keep], Cv[keep]


def boost(C, sat=1.15, gain=1.0):
    """Mild saturation / brightness boost so textures read at print size."""
    C = np.asarray(C, float)
    g = C.mean(1, keepdims=True)
    return np.clip((g + sat * (C - g)) * gain, 0, 1)


def cloud_scatter(ax, P, C, s=0.35, alpha=1.0, z=1):
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=boost(C), s=s, linewidths=0, alpha=alpha, depthshade=False, zorder=z)


def halo_cuboid(ax, c, color, lw=1.4, ls="-", z=4):
    c = np.asarray(c, float)
    for i, j in box_edges(c):
        ln, = ax.plot(*zip(c[i], c[j]), color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
        ln.set_path_effects([pe.Stroke(linewidth=lw + 1.1, foreground="white", alpha=0.8), pe.Normal()])


def floor_patch(ax, lo, hi, z0=0.0, step=0.4):
    """Light checkerboard on the floor plane z = z0 of the final frame, clipped to the view box."""
    xs = np.arange(np.floor(lo[0] / step) * step, hi[0] + 1e-9, step)
    ys = np.arange(np.floor(lo[1] / step) * step, hi[1] + 1e-9, step)
    polys, cols = [], []
    for i, x in enumerate(xs[:-1]):
        for j, y in enumerate(ys[:-1]):
            x0, x1 = max(x, lo[0]), min(xs[i + 1], hi[0]); y0, y1 = max(y, lo[1]), min(ys[j + 1], hi[1])
            if x1 <= x0 or y1 <= y0:
                continue
            polys.append([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)])
            cols.append("#e3e7ee" if (i + j) % 2 == 0 else "#cfd5df")
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors="none", alpha=0.9, zorder=0)
    ax.add_collection3d(pc)


_STATIC = None
def static_scene():
    """Static part of the pi3 union cloud of the video (actor removed with the SAM2 masks), in the
    final frame, cleaned.  Cached."""
    global _STATIC
    if _STATIC is None:
        import panels_stage2 as P2
        U = P2.union_split()
        st = ~U["dyn"]
        _STATIC = clean_cloud(U["P"][st], U["C"][st])
    return _STATIC


def set_box(ax, lo, hi, zoom=1.3):
    ext = np.maximum(hi - lo, 1e-6); c = (lo + hi) / 2
    ax.set_xlim(c[0] - ext[0] / 2, c[0] + ext[0] / 2); ax.set_ylim(c[1] - ext[1] / 2, c[1] + ext[1] / 2)
    ax.set_zlim(c[2] - ext[2] / 2, c[2] + ext[2] / 2)
    ax.set_box_aspect(tuple(ext / ext.max()), zoom=zoom)


def ax3(fig, rect):
    ax = fig.add_axes(rect, projection="3d"); ax.view_init(**VIEW_F); ax.set_axis_off(); ax.patch.set_alpha(0)
    ax.computed_zorder = False
    return ax


def in_view(P, lo, hi):
    return np.all((P >= lo) & (P <= hi), 1)


# ---------------------------------------------------------------- (iii) floor + body, final frame
def panel_floor_body(k=43):
    b = SC.bundle()
    P, C = static_scene()
    Z = np.load(SC.BUNDLE_DIR / "smpl_posed.npz")
    V = np.asarray(Z["verts"][0, k], float); F = Z["faces_0"]
    M = Z["glb_to_floorsim"]; V = (M[:3, :3] @ V.T).T + M[:3, 3]
    Vf = SC.floorsim_to_final(V)
    lo = np.percentile(P, 1, 0); hi = np.percentile(P, 99, 0); lo[2] = 0.0
    fig = plt.figure(figsize=(3.4, 2.9)); fig.patch.set_alpha(0)
    ax = ax3(fig, [0, 0, 1, 1])
    hi[2] = np.percentile(P[:, 2], 97)
    set_box(ax, lo, hi, zoom=1.55)
    floor_patch(ax, lo, hi)
    m = in_view(P, lo, hi)
    cloud_scatter(ax, P[m], C[m])
    tri = Vf[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    l = np.array([0.4, -0.5, 0.8]); l /= np.linalg.norm(l)
    inten = 0.45 + 0.55 * np.abs(nrm @ l)
    base = np.array(matplotlib.colors.to_rgb(LABEL_COL["person"]))
    fc = np.clip(base[None] * inten[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors="none", zorder=3))
    fig.text(0.5, 0.02, f"PromptHMR body (frame {int(b['sampled_idx'][k])}) on the fitted floor, "
             f"$s$ = {float(b['global_floor_sim_s']):.2f}", ha="center", fontsize=TINY, color=COL["muted"])
    save(fig, "b0_floor_body")


# ---------------------------------------------------------------- (v) boxes of one frame, clean scene
def panel_obb_clear(s="000087", zoom_label="laptop"):
    tf = final_tf()
    Z = np.load(BB / f"pi3_{s}.npz"); Pw = Z["points"]; Cf = Z["conf"]; img = Z["image"]
    thr = max(1e-3, float(np.percentile(Cf, 5)))
    ok = np.isfinite(Pw).all(-1) & (Cf >= thr)
    obs = objects(s)
    # scene context = this frame's own points minus the object masks, cleaned
    objmask = np.zeros_like(ok)
    per = {}
    for o in obs:
        if o["label"] == "bed":
            continue
        m, _ = object_mask(s, o)
        objmask |= m
        per[o["label"]] = (clean_cloud(tf(Pw[eroded(m, 10) & ok]), img[eroded(m, 10) & ok], voxel=0.006, nstd=2.5), o)
    ctx = ok & ~objmask
    Pc, Cc = clean_cloud(tf(Pw[ctx]), img[ctx], voxel=0.005)
    bed = next((o for o in obs if o["label"] == "bed"), None)
    fig = plt.figure(figsize=(4.4, 3.6)); fig.patch.set_alpha(0)
    ax = ax3(fig, [0.0, 0.34, 1.0, 0.66])
    corners = [tf(o["obb_floor_parallel"]["corners_world"]) for _, o in per.values()]
    if bed is not None:
        corners.append(tf(bed["obb_floor_parallel"]["corners_world"]))
    lo = np.minimum(np.percentile(Pc, 1, 0), np.min([c.min(0) for c in corners], 0))
    hi = np.maximum(np.percentile(Pc, 99, 0), np.max([c.max(0) for c in corners], 0))
    lo[2] = min(lo[2], 0.0)
    set_box(ax, lo, hi, zoom=1.25)
    floor_patch(ax, lo, hi)
    m = in_view(Pc, lo, hi)
    cloud_scatter(ax, Pc[m], Cc[m], s=0.3)
    for lab, ((Q, _), o) in per.items():
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], color=LABEL_COL[lab], s=0.6, linewidths=0, depthshade=False, zorder=2)
        halo_cuboid(ax, tf(o["obb_floor_parallel"]["corners_world"]), LABEL_COL[lab], lw=1.0)
    if bed is not None:
        halo_cuboid(ax, tf(bed["obb_floor_parallel"]["corners_world"]), LABEL_COL["bed"], lw=0.9)
    ax.text2D(0.5, 0.99, f"floor-parallel OBBs (final), frame {int(s)}", transform=ax.transAxes, fontsize=SMALL,
              ha="center", va="top")
    labs = list(per) + (["bed"] if bed is not None else [])
    h = [Line2D([], [], color=LABEL_COL[l], lw=1.5, label=l) for l in labs]
    ax.legend(handles=h, loc="upper left", fontsize=TINY - 0.5, frameon=False, handlelength=1.2, labelspacing=0.2,
              bbox_to_anchor=(0.0, 0.93))
    # the three candidates for one object
    (Q, _), o = per[zoom_label]
    ax = ax3(fig, [0.0, 0.0, 0.62, 0.38])
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], color=LABEL_COL[zoom_label], s=1.6, linewidths=0, alpha=0.85, depthshade=False, zorder=1)
    variants = [("aabb_floor_aligned", "floor-aligned AABB", ":", COL["muted"]),
                ("obb_arbitrary", "PCA OBB", "--", COL["cam_init"]),
                ("obb_floor_parallel", "floor-parallel OBB", "-", "k")]
    cs = []
    for key, name, ls, col in variants:
        c = tf(o[key]["corners_world"]); cs.append(c)
        halo_cuboid(ax, c, col, lw=1.1, ls=ls)
    allc = np.vstack(cs + [Q]); set_box(ax, allc.min(0), allc.max(0), zoom=1.2)
    h = [Line2D([], [], color=col, lw=1.1, ls=ls, label=name) for _, name, ls, col in variants]
    ax.legend(handles=h, loc="center left", fontsize=TINY, frameon=False, handlelength=2.0, labelspacing=0.3,
              bbox_to_anchor=(1.0, 0.5))
    ax.text2D(0.02, 0.98, f"{zoom_label}: box variants", transform=ax.transAxes, fontsize=SMALL, va="top")
    save(fig, "b4_obb")


# ---------------------------------------------------------------- (vi) final boxes over time, clean scene
def panel_final_boxes(frames=("000010.png", "000240.png")):
    F = json.load(open(BB / "final_4d.json"))["frames"]
    P, C = static_scene()
    allb = [np.array(o["corners_final"]) for f in frames for o in F[f]["objects"]]
    lo = np.min([c.min(0) for c in allb], 0) - 0.1; hi = np.max([c.max(0) for c in allb], 0) + 0.1
    lo[2] = 0.0; hi[2] = max(hi[2], np.percentile(P[:, 2], 97))
    m = in_view(P, lo, hi); P, C = P[m], C[m]
    fig = plt.figure(figsize=(4.6, 2.2)); fig.patch.set_alpha(0)
    n = len(frames)
    labels = []
    for i, f in enumerate(frames):
        ax = ax3(fig, [i / n, 0.06, 1 / n, 0.94])
        set_box(ax, lo, hi, zoom=1.42)
        floor_patch(ax, lo, hi)
        cloud_scatter(ax, P, C, s=0.3)
        for o in F[f]["objects"]:
            c = np.array(o["corners_final"])
            halo_cuboid(ax, c, LABEL_COL[o["label"]], lw=1.2 if o["label"] == "person" else 0.9)
            if o["label"] not in labels:
                labels.append(o["label"])
        ax.text2D(0.5, 0.86, f"t = {int(f[:6])}", transform=ax.transAxes, ha="center", va="bottom", fontsize=SMALL)
    h = [Line2D([], [], color=LABEL_COL[l], lw=1.6, label=l) for l in labels]
    fig.legend(handles=h, loc="lower center", ncol=len(labels), fontsize=TINY, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.2, columnspacing=1.0)
    save(fig, "b6_final_boxes")


if __name__ == "__main__":
    which = sys.argv[1:] or ["det", "sam2", "erosion", "volumes", "obb", "timeline", "floor", "final"]
    fns = {"det": panel_detection, "sam2": panel_sam2, "erosion": panel_erosion, "volumes": panel_volumes, "obb": panel_obb_clear, "timeline": panel_timeline,
           "floor": panel_floor_body, "final": panel_final_boxes}
    for w in which:
        fns[w]()
