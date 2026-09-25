"""Shared conventions for the 4D scene-pipeline figure panels (video 00T1E).

All panel scripts import from here so that every 3-D render uses the same
coordinate frame, viewpoint, colours, figure size and output location.

Coordinate frames in the bundle
--------------------------------
* ``world``  – the pi3 reconstruction frame (``pi3_points_k``, ``pi3_all_points``,
  ``pi3_poses``, ``corners_world`` in obb.json).
* ``final``  – the floor-aligned frame used by the annotation viewer: **z-up, floor at
  z = 0**.  ``p_final = A @ (p_world - origin)`` (``to_final``).  ``floor_final_*``,
  ``refined_poses``, ``d4_camera_poses`` and ``corners_final`` already live here.
* ``floorsim`` – the PromptHMR / floor-similarity frame: **y-up, floor at y = 0**,
  metric scale.  ``p_floorsim = s * R @ p_world + t`` (``to_floorsim``).  The SMPL
  meshes (``smpl_verts_i``) and the ``floor_verts`` grid (gv) live here.
* VGGT static cloud (``vggt_points``) is in its own frame and is only rendered alone.

Frame bookkeeping
-----------------
``sampled_idx[k]`` is the raw-frame index of the k-th adaptively sampled frame.
pi3 clip index k corresponds to file ``{sampled_idx[k]+1:06d}.png`` (a deliberate
off-by-one inherited from the annotation-time loader); the annotated frames are the
files ``{sampled_idx[k]:06d}.png`` listed in ``refined_stems``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# The video is chosen with the SCENE_VIDEO environment variable (default 00T1E, the first figure).
VIDEO = os.environ.get("SCENE_VIDEO", "00T1E")
_arch = REPO / "assets" / "figures" / "architecture" / "scene_pipeline"
if (_arch / VIDEO).exists():
    BUNDLE_DIR = _arch / VIDEO
    PANEL_DIR = _arch / ("panels" if VIDEO == "00T1E" else f"panels_{VIDEO}")
else:
    BUNDLE_DIR = REPO / "outputs" / "scene_pipeline" / VIDEO
    PANEL_DIR = REPO / "outputs" / "scene_pipeline" / ("panels" if VIDEO == "00T1E" else f"panels_{VIDEO}")

# Per-video frame choices (raw-frame indices for 2-D panels, annotated stems for box panels).
# Anything missing falls back to a spread over the video in scene_common.cfg().
_VIDEO_CFG = {
    "00T1E": dict(sift=(1, 5), homog=(1, 98, 5), thumbs=(1, 86, 187, 243, 271, 318), ticks=(1, 100, 200, 318),
                  dyn_frames=(38, 137, 243), time_frames=("000010.png", "000163.png", "000240.png"),
                  obb_frame="000010.png", obb_label="laptop", pf_frame="000010.png"),
    "0DJ6R": dict(sift=(1, 5), homog=(1, 75, 5), thumbs=(1, 124, 312, 410, 531, 993), ticks=(1, 250, 500, 750, 1011),
                  dyn_frames=(124, 385, 441), time_frames=("000210.png", "000413.png", "000742.png"),
                  obb_frame="000413.png", obb_label="phone", pf_frame="000413.png",
                  smpl_view=dict(elev=20, azim=-35), smpl_zoom=1.1),
}
PANEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- style ------------------------------------------------------------------
DPI = 300
VIEW = dict(elev=22, azim=-58)          # the one viewpoint for every final-frame 3-D panel
VIEW_FLOORSIM = dict(elev=18, azim=-62)  # for y-up floorsim renders (SMPL + floor)
COL = {
    "static":   "#4a7fd6",   # static background cloud
    "dynamic":  "#e07a2f",   # per-frame dynamic cloud / person
    "cam_init": "#d64a4a",   # initial pi3 camera poses
    "cam_ref":  "#2aa876",   # ICP-refined poses
    "floor":    "#9aa4b2",
    "smpl":     "#57c75a",
    "obb_prop": "#f2b632",   # proposal boxes
    "obb_final": "#d62f8a",  # corrected boxes
    "text":     "#1f2933",
    "muted":    "#6b7280",
}
class _LabelCol(dict):
    _extra = ["#4a7fd6", "#e07a2f", "#b16be3", "#d62f8a", "#5aa9c9", "#f2b632"]
    def __missing__(self, k):
        self[k] = self._extra[len(self) % len(self._extra)]
        return self[k]
LABEL_COL = _LabelCol({"person": "#57c75a", "bed": "#4a7fd6", "laptop": "#e07a2f", "doorway": "#b16be3",
                       "shoe": "#d62f8a", "phone": "#b16be3"})
FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 9, "axes.titlesize": 10, "savefig.dpi": DPI,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
RASTER_DPI = 400       # raster parts (point clouds, meshes, photos) of the panel PDFs


# ---- data -------------------------------------------------------------------
_B = None
def bundle():
    global _B
    if _B is None:
        _B = np.load(BUNDLE_DIR / "bundle.npz", allow_pickle=True)
    return _B

def obb():
    return json.load(open(BUNDLE_DIR / "obb.json"))

def bbox_meshes():
    return json.load(open(BUNDLE_DIR / "frame_bbox_meshes.json"))


def view_ids() -> list[int]:
    return [int(k) for k in bundle()["pi3_view_ids"]]

def cfg(key: str):
    """Per-video setting; view-derived defaults (for 00T1E these equal the original hand-picked values)."""
    c = _VIDEO_CFG.get(VIDEO, {})
    if key in c:
        return c[key]
    v = view_ids()
    default = {"views4": [v[0], v[2], v[4], v[6]], "view_mid": v[4], "views3": [v[0], v[4], v[6]],
               "smpl_k": v[4], "smpl_k_unposed": v[6], "smpl_view": VIEW_FLOORSIM, "smpl_zoom": 1.35}
    return default[key]


# ---- transforms -------------------------------------------------------------
def to_final(P: np.ndarray) -> np.ndarray:
    b = bundle()
    A, o = b["world_to_final_A"], b["world_to_final_origin"]
    return (A @ (np.asarray(P, np.float64) - o).T).T

def pose_to_final(T: np.ndarray) -> np.ndarray:
    """Map a world-frame camera-to-world 4x4 pose into the final frame."""
    b = bundle()
    A, o = b["world_to_final_A"], b["world_to_final_origin"]
    M = np.eye(4); M[:3, :3] = A; M[:3, 3] = -A @ o
    return M @ T

def to_floorsim(P: np.ndarray) -> np.ndarray:
    """world -> floorsim (metric, y-up, floor at y=0).  VERIFIED 2026-09-25: the stored
    similarity (s, R, t) maps floorsim -> world, so world -> floorsim is the inverse:
    p_fs = R^T (p_w - t) / s   (gives floor at y~0, ceiling ~2.2 m, matching the SMPL height)."""
    b = bundle()
    s, R, t = float(b["global_floor_sim_s"]), b["global_floor_sim_R"], b["global_floor_sim_t"]
    return (R.T @ ((np.asarray(P, np.float64) - t) / s).T).T

def floorsim_to_world(P: np.ndarray) -> np.ndarray:
    """floorsim -> world: p_w = s R p_fs + t (the stored direction)."""
    b = bundle()
    s, R, t = float(b["global_floor_sim_s"]), b["global_floor_sim_R"], b["global_floor_sim_t"]
    return (s * (R @ np.asarray(P, np.float64).T)).T + t

def floorsim_to_final(P: np.ndarray) -> np.ndarray:
    return to_final(floorsim_to_world(P))

def sampled_file(k: int) -> str:
    """Annotated-frame file name of sampled index k."""
    return f"{int(bundle()['sampled_idx'][k]):06d}.png"


# ---- plotting helpers -------------------------------------------------------
def fig3d(size=(3.2, 2.6), view=VIEW):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(**view)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return fig, ax

def set_equal(ax, P: np.ndarray, pad=0.04, zoom=1.35, pct=(1, 99)):
    """Equal *scale* axes fitted tightly to the points P (N,3).

    The axes box takes the data's own proportions (non-cubic), so an elongated scene
    fills the canvas instead of shrinking inside a cube; ``zoom`` enlarges further
    (matplotlib >= 3.7 ``set_box_aspect(..., zoom=)``)."""
    P = np.asarray(P, float)
    lo, hi = np.percentile(P, pct[0], axis=0), np.percentile(P, pct[1], axis=0)
    ext = np.maximum(hi - lo, 1e-6) * (1 + pad)
    c = (lo + hi) / 2
    ax.set_xlim(c[0] - ext[0] / 2, c[0] + ext[0] / 2)
    ax.set_ylim(c[1] - ext[1] / 2, c[1] + ext[1] / 2)
    ax.set_zlim(c[2] - ext[2] / 2, c[2] + ext[2] / 2)
    try:
        ax.set_box_aspect(tuple(ext / ext.max()), zoom=zoom)
    except TypeError:
        ax.set_box_aspect(tuple(ext / ext.max()))

def scatter(ax, P, C=None, s=0.35, alpha=0.9, color=None, **kw):
    if C is not None:
        C = np.asarray(C)
        if C.dtype != np.float64 and C.max() > 1.0:
            C = C / 255.0
        return ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, s=s, alpha=alpha, linewidths=0, depthshade=False, **kw)
    return ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=color, s=s, alpha=alpha, linewidths=0, depthshade=False, **kw)

def draw_frustum(ax, T: np.ndarray, color, scale=0.12, lw=0.8, alpha=0.9):
    """Camera-to-world 4x4 pose -> a small pyramid at the camera centre."""
    w, h, d = 0.5 * scale, 0.32 * scale, 0.9 * scale
    pts = np.array([[0, 0, 0], [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]])
    P = (T[:3, :3] @ pts.T).T + T[:3, 3]
    for i in range(1, 5):
        ax.plot(*zip(P[0], P[i]), color=color, lw=lw, alpha=alpha)
    loop = [1, 2, 3, 4, 1]
    ax.plot(P[loop, 0], P[loop, 1], P[loop, 2], color=color, lw=lw, alpha=alpha)

def draw_box(ax, corners: np.ndarray, color, lw=1.0, alpha=1.0, ls="-"):
    """8 corners in the (0..3 bottom, 4..7 top) or arbitrary order: draw the 12 edges of the hull."""
    corners = np.asarray(corners, float)
    # robust: connect each corner to its 3 nearest neighbours
    from itertools import combinations
    d = np.linalg.norm(corners[:, None] - corners[None], axis=-1)
    edges = set()
    for i in range(8):
        for j in np.argsort(d[i])[1:4]:
            edges.add(tuple(sorted((i, int(j)))))
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color=color, lw=lw, alpha=alpha, ls=ls)

def draw_floor(ax, verts, faces, color=COL["floor"], alpha=0.25, checker=True):
    """Floor grid mesh; with checker=True alternate faces are shaded like a checkerboard."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = np.asarray(verts, float); faces = np.asarray(faces, int)
    polys = verts[faces]
    if checker:
        cols = []
        for i, f in enumerate(faces):
            cent = verts[f].mean(0)
            cols.append("#c9d1dc" if (int(np.floor(cent[0] / 0.4)) + int(np.floor(cent[1] / 0.4))) % 2 == 0 else "#8f9aa8")
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors="none", alpha=alpha)
    else:
        pc = Poly3DCollection(polys, facecolors=color, edgecolors="none", alpha=alpha)
    ax.add_collection3d(pc)

def _enable_3d_rasterization():
    """mplot3d's collections override ``draw`` without the ``allow_rasterization`` wrapper, so
    ``set_rasterized`` is ignored and a 3-D cloud would be written as ~10^5 vector circles.  The
    projection happens before ``draw``, so wrapping it is safe."""
    from matplotlib.artist import allow_rasterization
    from mpl_toolkits.mplot3d import art3d
    for cls in (art3d.Path3DCollection, art3d.Poly3DCollection, art3d.Patch3DCollection, art3d.Line3DCollection):
        if "draw" in cls.__dict__ and not getattr(cls.draw, "_supports_rasterization", False):
            cls.draw = allow_rasterization(cls.draw)


_enable_3d_rasterization()


def rasterize_heavy(fig, min_items: int = 100):
    """Rasterise the artists that are expensive as vectors (scatter clouds, dense meshes); text, box
    edges, frusta and axes stay vector."""
    for ax in fig.axes:
        for c in ax.collections:
            n = max(len(c.get_offsets()) if hasattr(c, "get_offsets") else 0, len(c.get_paths()))
            if n >= min_items:
                c.set_rasterized(True)


def save_fig(fig, out_stem, transparent=True, png_dpi=DPI, pdf_dpi=RASTER_DPI, pad=0.02):
    """Write ``out_stem.png`` (preview / alpha measurements) and ``out_stem.pdf`` (vector text, raster
    clouds) with one shared bounding box, so crops measured on the PNG apply to the PDF."""
    out_stem = Path(out_stem)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
    fig.savefig(out_stem.with_suffix(".png"), dpi=png_dpi, transparent=transparent, bbox_inches=bb)
    rasterize_heavy(fig)
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=pdf_dpi, transparent=transparent, bbox_inches=bb)
    plt.close(fig)
    print("saved", out_stem.with_suffix(".png"), "+ .pdf")
    return out_stem.with_suffix(".png")


def save(fig, name: str, transparent=True):
    return save_fig(fig, PANEL_DIR / name, transparent=transparent)


# ---- Title Case ------------------------------------------------------------------
_SMALL_WORDS = {"a", "an", "the", "and", "or", "nor", "but", "of", "to", "in", "on", "at", "by", "for", "from",
                "with", "vs", "via", "per", "as", "into", "over", "if"}
_UNITS = {"px", "m", "cm", "mm", "deg", "s", "ms", "pts"}
_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


class Verbatim(str):
    """Text that title_case must not touch (quoted model output, identifiers)."""


def title_case(s):
    """Title Case for figure text.  Leaves alone: math/markup tokens ($ \\ _ { }), words that already contain
    capitals (GT, SAM2, Qwen2.5-VL), single-letter variables (k, t), units, number-led tokens (2nd) and
    non-ASCII words (π³).  Small words stay lower case except at the start of a line."""
    import re
    if not isinstance(s, str) or not s or isinstance(s, Verbatim):
        return s
    lines = []
    for line in s.split("\n"):
        out, first = [], True
        for tok in line.split(" "):
            m = re.match(r"^([^A-Za-z]*)([a-zA-Z][A-Za-z\-']*)(.*)$", tok)
            if tok and m and not any(c in tok for c in "$\\_{}"):
                pre, word, post = m.groups()
                keep = ((pre and pre[-1].isalnum()) or (len(word) == 1 and word != "a")
                        or word.lower() in _UNITS or word in _ROMAN
                        or (not first and word.lower() in _SMALL_WORDS))
                if not keep:      # capitalise the all-lower-case parts of hyphenated words (VLM-seeded -> VLM-Seeded)
                    tok = pre + "-".join(p[:1].upper() + p[1:] if p.islower() and (len(p) > 1 or len(word.split("-")) == 1) else p
                                         for p in word.split("-")) + post
            if tok:
                first = False
            out.append(tok)
        lines.append(" ".join(out))
    return "\n".join(lines)


def enable_title_case():
    """Route every matplotlib text of this process through title_case (panel titles, labels, legends, ticks)."""
    import matplotlib.text as mtext
    if getattr(mtext.Text.set_text, "_title_case", False):
        return
    orig = mtext.Text.set_text

    def set_text(self, s):
        if isinstance(s, Verbatim):          # matplotlib re-sets the text as a plain str while drawing,
            self._verbatim = True            # so remember the choice on the artist
        if isinstance(s, str) and not getattr(self, "_verbatim", False):
            s = title_case(s)
        return orig(self, s)
    set_text._title_case = True
    mtext.Text.set_text = set_text


# ---- colour / photo enhancement --------------------------------------------------
def vivid(C, sat=1.3, contrast=1.12, gamma=0.95):
    """Photo colours of a point cloud (uint8 or float) -> float RGB with more saturation and contrast,
    so the clouds do not read as washed out once scaled down in the paper."""
    c = np.asarray(C, np.float64)
    if c.max() > 1.0:
        c = c / 255.0
    luma = (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2])[:, None]
    c = luma + sat * (c - luma)
    c = 0.5 + contrast * (c - 0.5)
    return np.clip(c, 0, 1) ** gamma


def crisp(img, scale=2, amount=0.6, sigma=1.0, sat=1.1):
    """Low-resolution video frame -> Lanczos-upsampled, unsharp-masked, slightly more saturated uint8 RGB."""
    import cv2
    im = np.asarray(img)
    if scale != 1:
        im = cv2.resize(im, (im.shape[1] * scale, im.shape[0] * scale), interpolation=cv2.INTER_LANCZOS4)
    f = im.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigma * scale)
    f = f + amount * (f - blur)
    if sat != 1.0 and f.ndim == 3:
        l = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])[..., None]
        f = l + sat * (f - l)
    return np.clip(f, 0, 255).astype(np.uint8)
