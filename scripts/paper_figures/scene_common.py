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
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO / "outputs" / "scene_pipeline" / "00T1E"
PANEL_DIR = REPO / "outputs" / "scene_pipeline" / "panels"
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
LABEL_COL = {"person": "#57c75a", "bed": "#4a7fd6", "laptop": "#e07a2f", "doorway": "#b16be3", "shoe": "#d62f8a"}
FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 9, "axes.titlesize": 10, "savefig.dpi": DPI})


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
    b = bundle()
    s, R, t = float(b["global_floor_sim_s"]), b["global_floor_sim_R"], b["global_floor_sim_t"]
    return (s * (R @ np.asarray(P, np.float64).T)).T + t

def floorsim_to_world(P: np.ndarray) -> np.ndarray:
    b = bundle()
    s, R, t = float(b["global_floor_sim_s"]), b["global_floor_sim_R"], b["global_floor_sim_t"]
    return (R.T @ ((np.asarray(P, np.float64) - t) / s).T).T

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

def set_equal(ax, P: np.ndarray, pad=0.05):
    """Equal aspect box around the points P (N,3)."""
    lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    c = (lo + hi) / 2; r = (hi - lo).max() / 2 * (1 + pad)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

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

def save(fig, name: str, transparent=True):
    out = PANEL_DIR / f"{name}.png"
    fig.savefig(out, dpi=DPI, transparent=transparent, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", out)
    return out
