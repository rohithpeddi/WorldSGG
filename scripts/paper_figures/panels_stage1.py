"""Stage (i) panels for the 4D scene-pipeline figure (video 00T1E): adaptive frame
sampling and 2-D preprocessing.

Panels written to ``PANEL_DIR`` (all names start with ``s1_``):

* ``s1_sift_matches``       SIFT inlier matches between the reference frame and a
                            discarded candidate.
* ``s1_homography_overlap`` warped candidate quadrilateral + intersection polygon on
                            the reference frame, for a kept and a discarded candidate.
* ``s1_timeline``           318 raw frames -> 80 kept frames, with thumbnails.
* ``s1_selected_frames``    filmstrip of 5 kept frames with their raw indices.
* ``s1_static_vs_dynamic``  original vs person-removed ("static") frames.
* ``s1_masks``              SAM2 combined masks blended over the same frames.

File-name conventions (verified by pixel matching against frames_raw / masks):

* ``frames_raw/{i:06d}.png``      raw frame i (1-indexed, only i <= 40 exported).
* ``frames_sampled/{s:06d}.jpg``  kept frame with raw index s = sampled_idx[k]; it is
                                  pixel-identical (up to JPEG noise) to ``frames_raw/{s:06d}.png``.
* ``frames_static/{k+1:06d}.png`` static (inpainted) version of sampled position k.
* ``masks/{s:06d}.png``           binary SAM2 mask (0/255) of kept frame s.

Run:  python scripts/paper_figures/panels_stage1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_common import bundle, BUNDLE_DIR, PANEL_DIR, save, COL, DPI  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from PIL import Image  # noqa: E402

# ---- sampler hyper-parameters (supplementary Alg. 1) ---------------------------
LOWE_RATIO = 0.75
RANSAC_REPROJ = 4.0
RANSAC_ITERS = 2000
RANSAC_CONF = 0.995

SMALL = 7          # pt, captions
TINY = 6           # pt, tick / index labels
FRAME_LW = 0.5     # frame border line width

RAW_DIR = BUNDLE_DIR / "frames_raw"
SAMPLED_DIR = BUNDLE_DIR / "frames_sampled"
STATIC_DIR = BUNDLE_DIR / "frames_static"
MASK_DIR = BUNDLE_DIR / "masks"

SAMPLED_IDX = [int(x) for x in bundle()["sampled_idx"]]
N_RAW = int(bundle()["n_raw_frames"])          # 318

# muted palette for the match lines
MATCH_PALETTE = ["#4a7fd6", "#e07a2f", "#2aa876", "#d64a4a", "#b16be3", "#d62f8a", "#f2b632", "#5aa9c9"]


# ---- image loading ---------------------------------------------------------------
def raw_frame(i: int) -> np.ndarray:
    """RGB uint8 image of raw frame i (1-indexed). Falls back to frames_sampled for
    kept frames beyond the 40 exported raw frames (identical content, JPEG)."""
    p = RAW_DIR / f"{i:06d}.png"
    if not p.exists():
        p = SAMPLED_DIR / f"{i:06d}.jpg"
    return np.asarray(Image.open(p).convert("RGB"))


def static_frame(s: int) -> np.ndarray:
    k = SAMPLED_IDX.index(s)
    return np.asarray(Image.open(STATIC_DIR / f"{k + 1:06d}.png").convert("RGB"))


def mask_of(s: int) -> np.ndarray:
    return np.asarray(Image.open(MASK_DIR / f"{s:06d}.png").convert("L")) > 0


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# ---- geometry ----------------------------------------------------------------------
def _signed_area(poly) -> float:
    x = np.array([p[0] for p in poly], float); y = np.array([p[1] for p in poly], float)
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _clip_sutherland_hodgman(subject, clip):
    """Intersection of a convex clip polygon with an arbitrary subject polygon."""
    clip = list(clip)
    if _signed_area(clip) > 0:      # make the clip polygon clockwise in image coords
        clip = clip[::-1]

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p, q, a, b):
        d1 = (q[0] - p[0], q[1] - p[1]); d2 = (b[0] - a[0], b[1] - a[1])
        den = d1[0] * d2[1] - d1[1] * d2[0]
        t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
        return (p[0] + t * d1[0], p[1] + t * d1[1])

    out = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    out.append(intersect(s, e, a, b))
                out.append(e)
            elif inside(s, a, b):
                out.append(intersect(s, e, a, b))
            s = e
    return out


def intersection_polygon(quad: np.ndarray, w: int, h: int):
    """Intersection of the warped quad with the reference image rectangle.
    Returns (polygon Nx2, area)."""
    rect = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    try:
        from shapely.geometry import Polygon
        inter = Polygon([tuple(p) for p in quad]).buffer(0).intersection(Polygon(rect))
        if inter.is_empty:
            return np.zeros((0, 2)), 0.0
        if inter.geom_type != "Polygon":     # multi-polygon: take the largest piece
            inter = max(inter.geoms, key=lambda g: g.area)
        return np.asarray(inter.exterior.coords)[:-1], float(inter.area)
    except ImportError:
        poly = [tuple(p) for p in quad]
        if _signed_area(poly) > 0:
            poly = poly[::-1]
        ip = _clip_sutherland_hodgman(poly, rect)
        return np.asarray(ip), (abs(_signed_area(ip)) if len(ip) >= 3 else 0.0)


def sift_overlap(ref: np.ndarray, cand: np.ndarray):
    """Alg. 1 step for one candidate. Returns a dict with the matched keypoints,
    inlier mask, homography, warped quad, intersection polygon and overlap alpha."""
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(gray(ref), None)
    k2, d2 = sift.detectAndCompute(gray(cand), None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(d2, d1, k=2) if m.distance < LOWE_RATIO * n.distance]
    src = np.float32([k2[m.queryIdx].pt for m in good])      # candidate
    dst = np.float32([k1[m.trainIdx].pt for m in good])      # reference
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ,
                                maxIters=RANSAC_ITERS, confidence=RANSAC_CONF)
    inl = inl.ravel().astype(bool)
    h, w = ref.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    poly, area = intersection_polygon(quad, w, h)
    return dict(src=src, dst=dst, inliers=inl, n_good=len(good), n_inl=int(inl.sum()),
                H=H, quad=quad, poly=poly, alpha=area / float(w * h))


# ---- plotting helpers -------------------------------------------------------------
def _fig(w, h):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_alpha(0)
    return fig


def _img_axes(fig, rect, img, border=True):
    ax = fig.add_axes(rect)
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_alpha(0)
    for sp in ax.spines.values():
        sp.set_visible(border)
        sp.set_linewidth(FRAME_LW); sp.set_edgecolor(COL["muted"])
    return ax


# ---- panel 1: SIFT matches ---------------------------------------------------------
def panel_sift_matches(ref_idx=1, cand_idx=5, max_lines=60, seed=0):
    ref, cand = raw_frame(ref_idx), raw_frame(cand_idx)
    r = sift_overlap(ref, cand)
    h, w = ref.shape[:2]
    gap = int(0.08 * w)

    fw = 3.4
    fh = fw * h / (2 * w + gap) + 0.28
    fig = _fig(fw, fh)
    ax = fig.add_axes([0, 0.16 / fh, 1, 1 - 0.28 / fh])
    ax.imshow(ref, interpolation="lanczos", extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    ax.imshow(cand, interpolation="lanczos", extent=(w + gap - 0.5, 2 * w + gap - 0.5, h - 0.5, -0.5))
    ax.set_axis_off()

    rng = np.random.default_rng(seed)
    inl_ids = np.flatnonzero(r["inliers"])
    show = rng.choice(inl_ids, size=min(max_lines, len(inl_ids)), replace=False)
    for j, i in enumerate(show):
        c = MATCH_PALETTE[j % len(MATCH_PALETTE)]
        x1, y1 = r["dst"][i]; x2, y2 = r["src"][i]
        ax.plot([x1, x2 + w + gap], [y1, y2], color=c, lw=0.35, alpha=0.85, solid_capstyle="round")
        ax.plot([x1], [y1], "o", ms=1.3, mfc=c, mec="none")
        ax.plot([x2 + w + gap], [y2], "o", ms=1.3, mfc=c, mec="none")
    ax.set_xlim(-0.5, 2 * w + gap - 0.5); ax.set_ylim(h - 0.5, -0.5)

    fig.text(0.25, 1 - 0.09 / fh, f"reference (frame {ref_idx})", ha="center", va="top",
             fontsize=SMALL, color=COL["text"])
    fig.text(0.75, 1 - 0.09 / fh, f"candidate (frame {cand_idx})", ha="center", va="top",
             fontsize=SMALL, color=COL["text"])
    fig.text(0.5, 0.02 / fh,
             f"SIFT + Lowe ratio {LOWE_RATIO}: {r['n_good']} matches, "
             f"{r['n_inl']} RANSAC inliers ({len(show)} shown)",
             ha="center", va="bottom", fontsize=TINY, color=COL["muted"])
    save(fig, "s1_sift_matches")
    return r


# ---- panel 2: homography overlap --------------------------------------------------
def panel_homography_overlap(ref_idx=1, kept_idx=98, disc_idx=5):
    ref = raw_frame(ref_idx)
    h, w = ref.shape[:2]
    cases = [(kept_idx, "kept"), (disc_idx, "discarded")]
    results = {ci: sift_overlap(ref, raw_frame(ci)) for ci, _ in cases}

    # data bounding box of each sub-panel: reference rectangle U warped quad, small pad
    pad = 0.03 * w
    boxes = {}
    for ci, _ in cases:
        q = results[ci]["quad"]
        x0, y0 = min(0, q[:, 0].min()) - pad, min(0, q[:, 1].min()) - pad
        x1, y1 = max(w, q[:, 0].max()) + pad, max(h, q[:, 1].max()) + pad
        boxes[ci] = (x0, y0, x1, y1)

    # one common scale (inch per pixel) so both frames render at the same size
    fw = 3.4
    hgap_in, top_in, bot_in = 0.10, 0.13, 0.42
    total_w_px = sum(b[2] - b[0] for b in boxes.values())
    s = (fw - hgap_in - 0.04) / total_w_px
    heights_in = {ci: (b[3] - b[1]) * s for ci, b in boxes.items()}
    fh = max(heights_in.values()) + top_in + bot_in
    fig = _fig(fw, fh)

    left_in = 0.02
    for ci, tag in cases:
        r = results[ci]
        x0, y0, x1, y1 = boxes[ci]
        aw_in, ah_in = (x1 - x0) * s, (y1 - y0) * s
        rect = [left_in / fw, (fh - top_in - ah_in) / fh, aw_in / fw, ah_in / fh]   # top-aligned
        ax = fig.add_axes(rect)
        ax.imshow(ref, interpolation="lanczos", extent=(0, w, h, 0), alpha=0.95)
        ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, ec=COL["text"], lw=0.7))
        if len(r["poly"]) >= 3:
            ax.add_patch(MplPolygon(r["poly"], closed=True, fc=COL["static"], ec="none", alpha=0.35))
        ax.add_patch(MplPolygon(r["quad"], closed=True, fill=False, ec=COL["dynamic"], lw=1.0, ls="--"))
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_axis_off()
        cx = (left_in + aw_in / 2) / fw
        fig.text(cx, 1 - 0.02 / fh, f"candidate frame {ci}", ha="center", va="top",
                 fontsize=SMALL, color=COL["text"])
        fig.text(cx, 0.26 / fh, f"overlap α = {r['alpha']:.2f}", ha="center", va="bottom",
                 fontsize=SMALL, color=COL["text"], fontweight="bold")
        verdict = "α < τ  →  keep" if tag == "kept" else "α ≥ τ  →  discard"
        fig.text(cx, 0.14 / fh, verdict, ha="center", va="bottom", fontsize=TINY,
                 color=COL["dynamic"] if tag == "kept" else COL["muted"])
        left_in += aw_in + hgap_in
    fig.text(0.5, 0.01 / fh, f"reference: frame {ref_idx}    ■ intersection    - - warped candidate",
             ha="center", va="bottom", fontsize=TINY - 0.5, color=COL["muted"])
    save(fig, "s1_homography_overlap")
    return results


# ---- panel 3: timeline --------------------------------------------------------------
def panel_timeline(thumb_frames=(1, 86, 187, 243, 271, 318)):
    fw, fh = 3.6, 1.65
    fig = _fig(fw, fh)
    # timeline axis
    ax = fig.add_axes([0.03, 0.19, 0.94, 0.27])
    xs = np.arange(1, N_RAW + 1)
    ax.vlines(xs, 0, 0.45, color=COL["muted"], lw=0.25, alpha=0.7)
    ax.vlines(SAMPLED_IDX, 0, 1.0, color=COL["dynamic"], lw=0.7)
    ax.set_xlim(0, N_RAW + 1); ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xticks([1, 100, 200, 318])
    ax.tick_params(axis="x", labelsize=TINY, colors=COL["muted"], length=1.5, width=0.4, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_edgecolor(COL["muted"])
    ax.patch.set_alpha(0)

    fig.text(0.03, 0.005, f"{N_RAW} raw frames", ha="left", va="bottom", fontsize=SMALL, color=COL["muted"])
    fig.text(0.97, 0.005, f"→ {len(SAMPLED_IDX)} kept (adaptive)", ha="right", va="bottom",
             fontsize=SMALL, color=COL["dynamic"], fontweight="bold")

    # thumbnails, evenly spaced, with leader lines to their tick positions
    n = len(thumb_frames)
    th_h = 0.50            # figure fraction
    th_w = th_h * fh / fw * (270 / 480)
    slots = np.linspace(0.03 + th_w / 2, 0.97 - th_w / 2, n)
    for cx, s in zip(slots, thumb_frames):
        axi = fig.add_axes([cx - th_w / 2, 0.46, th_w, th_h])
        axi.imshow(raw_frame(s), interpolation="lanczos")
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_linewidth(FRAME_LW); sp.set_edgecolor(COL["dynamic"])
        # leader line in figure coordinates from thumbnail bottom to the tick top
        tx = ax.transData.transform((s, 1.0)); tx = fig.transFigure.inverted().transform(tx)
        fig.add_artist(plt.Line2D([cx, tx[0]], [0.46, tx[1]], color=COL["dynamic"], lw=0.4, alpha=0.8))
        axi.text(0.5, 1.02, f"{s}", transform=axi.transAxes, ha="center", va="bottom",
                 fontsize=TINY, color=COL["text"])
    save(fig, "s1_timeline")


# ---- panel 4: filmstrip -----------------------------------------------------------
def panel_selected_frames(n=5):
    ks = np.linspace(0, len(SAMPLED_IDX) - 1, n).round().astype(int)
    frames = [SAMPLED_IDX[k] for k in ks]
    fw = 3.5
    gap = 0.03
    tw = (0.98 - 0.02 - gap * (n - 1)) / n
    th_in = fw * tw * 480 / 270
    fh = th_in + 0.22
    fig = _fig(fw, fh)
    for j, (k, s) in enumerate(zip(ks, frames)):
        left = 0.02 + j * (tw + gap)
        ax = fig.add_axes([left, 0.2 / fh, tw, th_in / fh])
        ax.imshow(raw_frame(s), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(FRAME_LW); sp.set_edgecolor(COL["muted"])
        ax.text(0.5, -0.03, f"raw frame {s}", transform=ax.transAxes, ha="center", va="top",
                fontsize=TINY, color=COL["text"])
        ax.text(0.5, -0.14, f"kept #{k + 1}/{len(SAMPLED_IDX)}", transform=ax.transAxes, ha="center", va="top",
                fontsize=TINY - 0.5, color=COL["muted"])
    save(fig, "s1_selected_frames")
    return frames


# ---- panel 5: static vs dynamic --------------------------------------------------
def panel_static_vs_dynamic(frames=(38, 137, 243)):
    n = len(frames)
    fw = 3.4
    left0, gap = 0.13, 0.02
    tw = (0.99 - left0 - gap * (n - 1)) / n
    th_in = fw * tw * 480 / 270
    rgap_in = 0.06
    fh = 2 * th_in + rgap_in + 0.16
    fig = _fig(fw, fh)
    rows = [("Dynamic\n(original)", lambda s: raw_frame(s), COL["dynamic"]),
            ("Static\n(person removed)", lambda s: static_frame(s), COL["static"])]
    for i, (label, fn, col) in enumerate(rows):
        bottom = (0.02 + (1 - i) * (th_in + rgap_in)) / fh
        for j, s in enumerate(frames):
            left = left0 + j * (tw + gap)
            ax = fig.add_axes([left, bottom, tw, th_in / fh])
            ax.imshow(fn(s), interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(FRAME_LW); sp.set_edgecolor(col)
            if i == 0:
                ax.set_title(f"frame {s}", fontsize=TINY, color=COL["text"], pad=1.5)
        fig.text(left0 - 0.02, bottom + th_in / fh / 2, label, ha="right", va="center",
                 fontsize=SMALL, color=col, rotation=90, linespacing=1.1)
    save(fig, "s1_static_vs_dynamic")


# ---- panel 6: masks ---------------------------------------------------------------
def panel_masks(frames=(38, 137, 243), col="#e07a2f", alpha=0.55):
    n = len(frames)
    fw = 3.4
    gap = 0.02
    tw = (0.99 - 0.01 - gap * (n - 1)) / n
    th_in = fw * tw * 480 / 270
    fh = th_in + 0.40
    fig = _fig(fw, fh)
    rgb = np.array(matplotlib.colors.to_rgb(col))
    for j, s in enumerate(frames):
        img = raw_frame(s).astype(float) / 255.0
        m = mask_of(s)
        blend = img.copy()
        blend[m] = (1 - alpha) * img[m] + alpha * rgb
        left = 0.01 + j * (tw + gap)
        ax = fig.add_axes([left, 0.24 / fh, tw, th_in / fh])
        ax.imshow(blend, interpolation="lanczos")
        # thin contour around the mask
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 30:
                continue
            c = c.reshape(-1, 2)
            ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=col, lw=0.5)
        ax.set_xlim(-0.5, img.shape[1] - 0.5); ax.set_ylim(img.shape[0] - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(FRAME_LW); sp.set_edgecolor(COL["muted"])
        ax.set_title(f"frame {s}", fontsize=TINY, color=COL["text"], pad=1.5)
        ax.text(0.5, -0.03, f"mask area {100 * m.mean():.0f}%", transform=ax.transAxes, ha="center",
                va="top", fontsize=TINY - 0.5, color=COL["muted"])
    fig.text(0.5, 0.005, "SAM2 combined dynamic mask (person + carried objects)", ha="center",
             va="bottom", fontsize=TINY, color=COL["muted"])
    save(fig, "s1_masks")


if __name__ == "__main__":
    r1 = panel_sift_matches(ref_idx=1, cand_idx=5)
    print(f"sift_matches: ref 1 vs cand 5 -> {r1['n_good']} good, {r1['n_inl']} inliers, alpha={r1['alpha']:.3f}")
    res = panel_homography_overlap(ref_idx=1, kept_idx=98, disc_idx=5)
    for ci, r in res.items():
        print(f"overlap: ref 1 vs cand {ci}: alpha={r['alpha']:.3f} ({r['n_inl']} inliers)")
    panel_timeline()
    print("filmstrip frames:", panel_selected_frames())
    panel_static_vs_dynamic()
    panel_masks()
