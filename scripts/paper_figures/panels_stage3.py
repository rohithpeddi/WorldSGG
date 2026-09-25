"""Stage (iv) camera alignment, the final 4D scene, the floor/human panel and the 3-D box
panels of the scene-pipeline figure (video 00T1E).  Writes ``s3_*.png`` into PANEL_DIR.

Run with the scene4cast interpreter::

    C:/Users/rohit/anaconda3/envs/scene4cast/python.exe scripts/paper_figures/panels_stage3.py [icp 4d smpl ms pf time]

Geometry notes (verified numerically):

* pi3 clip index ``k`` <-> annotated stem ``sampled_idx[k]`` (``pi3_frame_ids == sampled_idx[:77]``);
  stem 000310 has a refined pose but no pi3 pose (pi3 covers the first 77 of 80 sampled frames).
* The ICP refinement barely moves the (static) camera: mean centre shift 5e-3 scene units, max 3e-2;
  mean rotation change 0.9 deg (max 7.3 deg), hence the residual strip in ``s3_icp_before_after``.
* ``smpl_verts_i`` in bundle.npz are the *bind-pose* templates of the skinned world4d.glb (the
  extractor read ``geom.vertices`` and dropped the armature animation), so ``s3_floor_smpl`` shows
  the person's dynamic pi3 points instead.  ``s3_smpl_posed.npz`` (next to the bundle) holds the
  linear-blend-skinned meshes at keyframes kf in {0,10,41,65,76} (kf k <-> pi3 clip k) computed
  from the GLB animation in PromptHMR's metric world (y-up, floor y=0, static camera at
  (0, 1.29, 0)); with the corrected ``to_floorsim`` that world coincides with ``floorsim`` (pi3
  camera at y=1.335; kf-10 mesh ~0.25 m from the annotated person box).  ``s3_floor_smpl_posed``
  is the optional variant that draws that mesh.
* obb corners are ordered bottom loop 0-3, top loop 4-7 (``corners_final`` and every
  ``corners_world``), so boxes are drawn from a fixed edge list (``draw_box_ordered``); the
  nearest-neighbour heuristic in ``scene_common.draw_box`` picks face diagonals on flat boxes.
* floorsim panels are plotted in the cyclic permutation (z, x, y) -> (X, Y, Z): right-handed, z-up,
  so ``draw_floor``'s checker (keyed on the first two data columns) works on the y=0 floor.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy import ndimage

import scene_common as SC
from scene_common import (
    bundle, obb, to_final, pose_to_final, to_floorsim,
    fig3d, set_equal, scatter, draw_frustum, draw_floor, save,
    COL, LABEL_COL, VIEW, VIEW_FLOORSIM,
)

# The bundle was relocated from outputs/scene_pipeline to assets/figures/architecture/scene_pipeline
# while the panel scripts were being written; follow whichever location holds the data.
if not (SC.BUNDLE_DIR / "bundle.npz").exists():
    _alt = SC.REPO / "assets" / "figures" / "architecture" / "scene_pipeline"
    if (_alt / SC.VIDEO / "bundle.npz").exists():
        SC.BUNDLE_DIR = _alt / SC.VIDEO
        SC.PANEL_DIR = _alt / "panels"
        SC.PANEL_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_DIR = SC.BUNDLE_DIR

B = bundle()
O = obb()
FRAMES = O["frames"]                       # "000010.png" -> [objects]
STEMS = [str(s) for s in B["refined_stems"]]
FIDS = [int(f) for f in B["pi3_frame_ids"]]
RNG = np.random.default_rng(0)
GREY = np.array([0.72, 0.74, 0.78])
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
BOX_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


# ---------------------------------------------------------------- helpers
def stem_of(frame: str) -> str:
    return frame[:6]


def refined_pose(stem: str) -> np.ndarray:
    return B["refined_poses"][STEMS.index(stem)]


def init_pose_final(stem: str) -> np.ndarray | None:
    s = int(stem)
    return pose_to_final(B["pi3_poses"][FIDS.index(s)]) if s in FIDS else None


def objects(frame: str, label: str | None = None):
    return [ob for ob in FRAMES[frame] if label is None or ob["label"] == label]


def person_box(frame: str) -> np.ndarray | None:
    obs = objects(frame, "person")
    return np.array(obs[0]["corners_final"], float) if obs else None


def cloud_final(n: int | None = 150_000):
    P, C = B["pi3_all_points"], B["pi3_all_colors"]
    if n is not None and n < len(P):
        idx = RNG.choice(len(P), n, replace=False)
        P, C = P[idx], C[idx]
    return to_final(P), C.astype(np.float64) / 255.0


def dim(C: np.ndarray, w: float = 0.55) -> np.ndarray:
    """Blend true colours towards grey."""
    return (1 - w) * C + w * GREY


def draw_box_ordered(ax, corners, color, lw=1.0, alpha=1.0, ls="-", zorder=None):
    """Wireframe box with corners ordered bottom loop 0-3 / top loop 4-7."""
    c = np.asarray(corners, float)
    for i, j in BOX_EDGES:
        ax.plot(*zip(c[i], c[j]), color=color, lw=lw, alpha=alpha, ls=ls, zorder=Z_BOX if zorder is None else zorder)


def crop_floor(verts, faces, xlim, ylim):
    """Keep only floor faces whose centroid lies inside the axes box (mplot3d does not clip)."""
    verts = np.asarray(verts, float); faces = np.asarray(faces, int)
    c = verts[faces].mean(1)
    keep = (c[:, 0] >= xlim[0]) & (c[:, 0] <= xlim[1]) & (c[:, 1] >= ylim[0]) & (c[:, 1] <= ylim[1])
    return verts, faces[keep]


Z_FLOOR, Z_CLOUD, Z_BOX, Z_CAM = 0, 1, 2, 3


def manual_zorder(ax):
    """mplot3d normally re-sorts whole collections by their mean depth, which paints the floor over
    the cloud and box faces over everything; with explicit z-orders the floor stays underneath."""
    ax.computed_zorder = False


def lift_lines(ax):
    """Frustum lines drawn by the shared helper get the camera z-order."""
    for ln in ax.lines:
        if ln.get_zorder() < Z_BOX:
            ln.set_zorder(Z_CAM)


def final_floor(ax, alpha=0.3):
    v, f = crop_floor(B["floor_final_verts"], B["floor_final_faces"], ax.get_xlim(), ax.get_ylim())
    draw_floor(ax, v, f, alpha=alpha)
    ax.collections[-1].set_zorder(Z_FLOOR)


def box_ext(boxes, margin):
    allc = np.vstack(boxes)
    return allc.min(0) - margin, allc.max(0) + margin


def set_limits(ax, lo, hi, zoom=1.35):
    ext = np.maximum(hi - lo, 1e-6); c = (lo + hi) / 2
    ax.set_xlim(c[0] - ext[0] / 2, c[0] + ext[0] / 2)
    ax.set_ylim(c[1] - ext[1] / 2, c[1] + ext[1] / 2)
    ax.set_zlim(c[2] - ext[2] / 2, c[2] + ext[2] / 2)
    ax.set_box_aspect(tuple(ext / ext.max()), zoom=zoom)


def in_box(P, lo, hi):
    return np.all((P >= lo) & (P <= hi), axis=1)


# ---------------------------------------------------------------- panel 1
def panel_icp_before_after():
    P, C = cloud_final(150_000)
    init = {s: init_pose_final(s) for s in STEMS}
    init = {s: T for s, T in init.items() if T is not None}
    ref = {s: refined_pose(s) for s in STEMS}
    both = [s for s in STEMS if s in init]
    dC = np.array([np.linalg.norm(init[s][:3, 3] - ref[s][:3, 3]) for s in both])
    dR = np.array([np.degrees(np.arccos(np.clip((np.trace(init[s][:3, :3].T @ ref[s][:3, :3]) - 1) / 2, -1, 1)))
                   for s in both])
    print(f"[icp] {len(both)} matched cameras: |dC| mean {dC.mean():.4f} max {dC.max():.4f}; "
          f"dtheta mean {dR.mean():.2f} deg max {dR.max():.2f} deg")

    fig = plt.figure(figsize=(7.4, 2.5))
    fig.patch.set_alpha(0)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.0, left=0.0, right=0.98, top=0.9, bottom=0.08)
    cams = np.array([T[:3, 3] for T in ref.values()])
    ext = np.vstack([P, np.repeat(cams, 200, axis=0)])

    for col, (title, poses, colr) in enumerate([
        ("Initial poses $T_t$", init, COL["cam_init"]),
        ("Refined poses $T_t$ (trimmed ICP)", ref, COL["cam_ref"]),
    ]):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.view_init(**VIEW); ax.set_axis_off(); ax.patch.set_alpha(0)
        set_equal(ax, ext)
        if col == 1:
            final_floor(ax)
        scatter(ax, P, C, s=0.12, alpha=0.85)
        for T in poses.values():
            draw_frustum(ax, T, colr, scale=0.28, lw=0.8, alpha=0.6)
        ax.set_title(title, pad=-2)

    # residual strip: the refinement is sub-centimetre for a static camera, so show it explicitly
    ax = fig.add_subplot(gs[0, 2])
    ax.patch.set_alpha(0)
    t = np.arange(len(both))
    ax.bar(t, dC * 100, color=COL["cam_ref"], alpha=0.75, width=0.8)
    ax.set_ylabel(r"centre shift $\|\Delta c_t\|$ ($10^{-2}$ units)", fontsize=6, color=COL["cam_ref"])
    ax.tick_params(labelsize=5.5, length=2)
    ax2 = ax.twinx(); ax2.patch.set_alpha(0)
    ax2.plot(t, dR, color=COL["cam_init"], lw=0.9, marker="o", ms=1.6)
    ax2.set_ylabel(r"rotation change $\Delta\theta_t$ (deg)", fontsize=6, color=COL["cam_init"])
    ax2.tick_params(labelsize=5.5, length=2)
    ax.set_xlabel("annotated frame index", fontsize=6, labelpad=1)
    ax.set_xticks(np.unique(np.linspace(0, len(STEMS) - 1, 4).round().astype(int)))
    ax.set_title(r"$T_t^{\rm init}\!\to T_t^{\rm refined}$", pad=2)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    ax.text(0.04, 0.96, f"mean shift {dC.mean()*100:.2f}$\\times10^{{-2}}$\nmean rotation {dR.mean():.1f}°",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=COL["muted"])
    save(fig, "s3_icp_before_after")


def panel_icp_residual():
    """Stage (iv) from the real trimmed ICP (static_icp.py): one frame's dynamic cloud D_t coloured by
    its distance to the static background S before and after registration, and the per-frame
    MSE convergence of all frames."""
    from scipy.spatial import cKDTree
    Z = np.load(BUNDLE_DIR / "static_icp.npz")
    Ti, mse, T_dyn = Z["T_icp"], Z["mse"], B["pi3_poses"]
    k = int(SC.cfg("view_mid"))
    S = Z["static_points"]; S = S[RNG.choice(len(S), min(250_000, len(S)), replace=False)]
    Sf = to_final(S)
    tree = cKDTree(Sf)
    Dw = B[f"pi3_points_{k}"].astype(np.float64)
    before = to_final(Dw)
    after = to_final(Dw @ Ti[k][:3, :3].T + Ti[k][:3, 3])
    d0, _ = tree.query(before); d1, _ = tree.query(after)
    vmax = float(np.percentile(d0, 95))
    rot = np.degrees(np.arccos(np.clip((np.trace(Ti[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    shift = np.linalg.norm(Ti[:, :3, 3], axis=1)
    print(f"[icp] frame {k}: median NN dist {np.median(d0):.4f} -> {np.median(d1):.4f}; all frames |tau| mean "
          f"{shift.mean():.4f}, rot mean {rot.mean():.2f} deg, iters mean {Z['iters'].mean():.1f}")
    fig = plt.figure(figsize=(7.4, 2.5)); fig.patch.set_alpha(0)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.0, left=0.0, right=0.98, top=0.9, bottom=0.1)
    lo, hi = np.percentile(np.vstack([before, after]), 1, 0), np.percentile(np.vstack([before, after]), 99, 0)
    ctx = in_box(Sf, lo - 0.1, hi + 0.1)
    cams = {"Before ICP": (before, d0, pose_to_final(T_dyn[k]), COL["cam_init"]),
            "After trimmed ICP": (after, d1, pose_to_final(Ti[k] @ T_dyn[k]), COL["cam_ref"])}
    for col, (title, (P, d, T, cc)) in enumerate(cams.items()):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.view_init(**VIEW); ax.set_axis_off(); ax.patch.set_alpha(0); manual_zorder(ax)
        set_limits(ax, lo, hi, zoom=1.3)
        scatter(ax, Sf[ctx][::3], color="#c9ccd1", s=0.12, alpha=0.25, zorder=Z_FLOOR)
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=d, cmap="magma_r", vmin=0, vmax=vmax, s=0.25,
                        linewidths=0, depthshade=False, zorder=Z_CLOUD)
        draw_frustum(ax, T, cc, scale=0.25, lw=1.0); lift_lines(ax)
        ax.set_title(f"{title}: median dist. to S {np.median(d):.3f}", pad=-2, fontsize=7.5)
    cax = fig.add_axes([0.30, 0.07, 0.16, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=5, length=1.5); cb.outline.set_linewidth(0.3)
    cb.set_label("distance to S", fontsize=5.5, labelpad=1.5)
    ax = fig.add_subplot(gs[0, 2]); ax.patch.set_alpha(0)
    cmap = plt.get_cmap("plasma")
    for t in range(len(mse)):
        h = mse[t][np.isfinite(mse[t])]
        ax.plot(np.arange(1, len(h) + 1), h / h[0], color=cmap(t / max(1, len(mse) - 1)), lw=0.6, alpha=0.8)
    ax.set_xlabel("ICP iteration", fontsize=6, labelpad=1); ax.set_ylabel("trimmed MSE / initial", fontsize=6)
    ax.tick_params(labelsize=5.5, length=2)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.set_title(f"convergence, all {len(mse)} frames", pad=2, fontsize=7.5)
    ax.text(0.97, 0.97, f"mean translation {shift.mean():.3f}\nmean rotation {rot.mean():.1f}°", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.5, color=COL["muted"])
    save(fig, "s3_icp_before_after")


# ---------------------------------------------------------------- panel 2
def pick_frames_spread(n: int):
    """First and last annotated frame plus the ones that spread the person's position the most
    (the person lies on the bed for the first half of the video, so evenly spaced frames overlap)."""
    fr = [f for f in FRAMES if stem_of(f) in STEMS and person_box(f) is not None]
    cen = np.array([person_box(f).mean(0) for f in fr])
    chosen = [0, len(fr) - 1]
    while len(chosen) < n:
        d = np.min([np.linalg.norm(cen - cen[c], axis=1) for c in chosen], axis=0)
        chosen.append(int(np.argmax(d)))
    return [fr[i] for i in sorted(chosen)]


def panel_scene_4d():
    P, C = cloud_final(None)
    frames = pick_frames_spread(6)
    boxes = [person_box(f) for f in frames]
    cams = [refined_pose(stem_of(f)) for f in frames]
    fig, ax = fig3d(size=(4.0, 3.2))
    manual_zorder(ax)
    ext = np.vstack([P] + [np.repeat(bx, 400, axis=0) for bx in boxes])
    set_equal(ax, ext)
    final_floor(ax, alpha=0.18)
    scatter(ax, P, C, s=0.08, alpha=0.8, zorder=Z_CLOUD)
    n = len(frames)
    steps = [1.0, 0.65, 0.35]
    alphas = [steps[min(2, int(3 * i / n))] for i in range(n)]
    centres = np.array([[bx[:, 0].mean(), bx[:, 1].mean(), bx[:, 2].min()] for bx in boxes])
    ax.plot(centres[:, 0], centres[:, 1], centres[:, 2] + 0.01, color=COL["dynamic"], lw=0.9, ls=":", alpha=0.9, zorder=Z_BOX)
    for i, (f, bx, T) in enumerate(zip(frames, boxes, cams)):
        a = alphas[i]
        draw_box_ordered(ax, bx, COL["dynamic"], lw=1.6, alpha=a)
        draw_frustum(ax, T, COL["cam_ref"], scale=0.25, lw=0.8, alpha=max(a, 0.5))
    lift_lines(ax)
    t0, t1 = int(stem_of(frames[0])), int(stem_of(frames[-1]))
    handles = [Line2D([], [], color=COL["dynamic"], lw=1.6, alpha=a, label=l)
               for a, l in zip(steps, (f"t = {t0}", "\u2192", f"t = {t1}"))]
    handles.append(Line2D([], [], color=COL["cam_ref"], lw=0.8, label="$T^t$"))
    ax.legend(handles=handles, loc="lower left", fontsize=5.5, frameon=False, handlelength=1.2,
              labelspacing=0.2, borderaxespad=0.0, bbox_to_anchor=(0.0, 0.02), title="person $\\mathcal{F}^t$",
              title_fontsize=5.5)
    ax.set_title(r"4D scene $(\mathcal{S},\{(\mathcal{F}^t, T^t)\}_{t=1}^{T})$", pad=-2)
    save(fig, "s3_scene_4d")
    print("[4d] frames", [stem_of(f) for f in frames])


# ---------------------------------------------------------------- panel 3
def fs_plot(P: np.ndarray) -> np.ndarray:
    """floorsim (x, y-up, z) -> plot frame (z, x, y): cyclic, right-handed, z-up."""
    P = np.asarray(P, float)
    return P[:, [2, 0, 1]]


def person_points_floorsim(k: int, min_conf: bool = True):
    """Dynamic pi3 points of clip k inside the largest connected component of the person mask."""
    stem = int(B["sampled_idx"][k])
    H, W = int(B["pi3_image_hw"][0]), int(B["pi3_image_hw"][1])
    m = np.array(Image.open(BUNDLE_DIR / "masks" / f"{stem:06d}.png").resize((W, H), Image.NEAREST)) > 0
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    pix = B[f"pi3_pix_{k}"]
    sel = m[pix // W, pix % W]
    if min_conf:
        sel &= B[f"pi3_conf_{k}"].astype(np.float32) >= float(B["pi3_conf_thr"])
    print(f"[smpl] clip {k} (stem {stem}): mask cc {m.mean()*100:.1f}% of frame, {sel.sum()} person points")
    return to_floorsim(B[f"pi3_points_{k}"][sel])


def _floorsim_base(ax, P, C, extra, extra_col=None, extra_s=1.2, ext_pts=None):
    """Floor + cloud in the (z, x, y) plot frame.  Points passed as ``extra`` with ``extra_col`` are
    merged into the same scatter so that mplot3d depth-sorts them per point (a second scatter
    collection would be drawn wholly before or after the cloud and get hidden).  ``ext_pts`` only
    widen the axis limits (e.g. a mesh drawn afterwards)."""
    manual_zorder(ax)
    ext = np.vstack([fs_plot(P), fs_plot(extra), fs_plot(np.array([[0, -0.02, 0]]))]
                    + ([fs_plot(ext_pts)] if ext_pts is not None else []))
    set_equal(ax, ext)
    lo, hi = np.percentile(P, 1, axis=0) - 0.5, np.percentile(P, 99, axis=0) + 0.5
    v, f = crop_floor(fs_plot(B["floor_verts"]), B["floor_faces"], (lo[2], hi[2]), (lo[0], hi[0]))
    draw_floor(ax, v, f, alpha=0.45)
    ax.collections[-1].set_zorder(Z_FLOOR)
    if extra_col is None:
        scatter(ax, fs_plot(P), C, s=0.12, alpha=0.85, zorder=Z_CLOUD)
    else:
        Pm = np.vstack([fs_plot(P), fs_plot(extra)])
        Cm = np.vstack([C, np.tile(np.array(to_rgb(extra_col))[None], (len(extra), 1))])
        sm = np.r_[np.full(len(P), 0.12), np.full(len(extra), extra_s)]
        scatter(ax, Pm, Cm, s=sm, alpha=0.9, zorder=Z_CLOUD)


def shaded_mesh(ax, V, F, base, light=(0.6, 0.4, 1.0), alpha=1.0):
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    l = np.asarray(light, float); l /= np.linalg.norm(l)
    inten = 0.5 + 0.5 * np.clip(nrm @ l, 0, 1)
    fc = np.clip(np.array(to_rgb(base))[None] * inten[:, None], 0, 1)
    pc = Poly3DCollection(tri, facecolors=fc, edgecolors=np.clip(fc * 0.75, 0, 1), linewidths=0.05, alpha=alpha)
    ax.add_collection3d(pc)


def panel_floor_smpl(k: int = SC.cfg("smpl_k_unposed"), posed: bool = False):
    P = to_floorsim(B["pi3_all_points"])
    idx = RNG.choice(len(P), 150_000, replace=False)
    P, C = P[idx], B["pi3_all_colors"][idx] / 255.0
    stem = int(B["sampled_idx"][k])
    print(f"[smpl] floorsim cloud 1..99 pct {np.percentile(P, 1, axis=0).round(2)}..{np.percentile(P, 99, axis=0).round(2)}; "
          f"pi3 camera {to_floorsim(B['pi3_poses'][:, :3, 3]).mean(0).round(2)}; s={float(B['global_floor_sim_s']):.3f}")
    fig, ax = fig3d(size=(3.4, 2.8), view=VIEW_FLOORSIM)
    if posed:
        # smpl_posed.npz: LBS-posed GLB meshes, verts (2, 77, 11307, 3); mesh 0 is the tracked person,
        # mesh 1 a spurious second track.  glb_to_floorsim (4x4 similarity) maps the GLB frame to floorsim.
        Z = np.load(BUNDLE_DIR / "smpl_posed.npz")
        V = np.asarray(Z["verts"][0, k], np.float64); F = Z["faces_0"]
        if "glb_to_floorsim" in Z.files:
            M = Z["glb_to_floorsim"]
            V = (M[:3, :3] @ V.T).T + M[:3, 3]
        D = person_points_floorsim(k)
        _floorsim_base(ax, P, C, D, extra_col=COL["smpl"], ext_pts=V)
        mesh = Poly3DCollection(fs_plot(V)[F], facecolors=COL["smpl"], edgecolors=(*to_rgb(COL["smpl"]), 0.15),
                                linewidths=0.1, zorder=Z_BOX)
        ax.add_collection3d(mesh)
        print(f"[smpl] posed mesh 0 kf {k} (stem {stem}): feet y {V[:,1].min():.2f}, top {V[:,1].max():.2f}, "
              f"centroid {V.mean(0).round(2)}; pi3 person points centroid {D.mean(0).round(2)}, "
              f"|d| = {np.linalg.norm(D.mean(0) - V.mean(0)):.2f} m; mesh xz span {(V.max(0)-V.min(0))[[0,2]].round(2)}")
        ax.set_title("PromptHMR human + floor alignment", pad=-2)
        name = "s3_floor_smpl_posed"
    else:
        D = person_points_floorsim(k)
        _floorsim_base(ax, P, C, D, extra_col=COL["smpl"])
        ax.set_title("Floor + metric scale from PromptHMR alignment", pad=-2)
        name = "s3_floor_smpl"
    ax.text2D(0.02, 0.04, f"metric scale $s$ = {float(B['global_floor_sim_s']):.2f} from PromptHMR \u2194 \u03c0\u00b3 similarity",
              transform=ax.transAxes, fontsize=5.5, color=COL["muted"])
    save(fig, name)


# ---------------------------------------------------------------- panel 4
def panel_obb_multiscale(frame: str = SC.cfg("obb_frame"), label: str = SC.cfg("obb_label")):
    ob = objects(frame, label)[0]
    cands = sorted(ob["candidates"], key=lambda c: c["kernel_size"])
    ks = np.array([c["kernel_size"] for c in cands]); vols = np.array([c["volume"] for c in cands])
    chosen = int(np.argmin(vols))
    boxes = [to_final(np.array(c["corners_world"])) for c in cands]
    lo, hi = box_ext([boxes[chosen]], 0.25)
    P, C = cloud_final(None)
    m = in_box(P, lo, hi)
    P, C = P[m], C[m]
    fig, ax = fig3d(size=(3.4, 2.8))
    manual_zorder(ax)
    set_limits(ax, lo, hi)
    scatter(ax, P, C, s=1.5, alpha=0.5, zorder=Z_CLOUD)
    n = len(cands)
    for i, bx in enumerate(boxes):
        if i != chosen:
            draw_box_ordered(ax, bx, COL["obb_prop"], lw=1.0, alpha=0.12 + 0.8 * i / (n - 1))
    draw_box_ordered(ax, boxes[chosen], COL["obb_final"], lw=2.5, alpha=1.0)
    ax.text2D(0.02, 0.9, f"{label}: k = {ks[chosen]}, min volume", transform=ax.transAxes,
              color=COL["obb_final"], fontsize=6, ha="left", va="top")
    ax.set_title("Multi-scale erosion \u2192 min-volume OBB", pad=-2)
    ins = fig.add_axes([0.70, 0.07, 0.27, 0.2]); ins.patch.set_alpha(0)
    cols = [COL["obb_final"] if i == chosen else COL["obb_prop"] for i in range(n)]
    ins.bar(ks, vols * 1e3, color=cols, width=0.8)
    ins.set_xlabel("erosion kernel", fontsize=5.5, labelpad=1)
    ins.set_ylabel(r"volume ($10^{-3}$)", fontsize=5.5, labelpad=1)
    ins.set_xticks([0, 5, 10]); ins.tick_params(labelsize=5, length=1.5, pad=1)
    ins.set_ylim(vols.min() * 1e3 * 0.9, vols.max() * 1e3 * 1.05)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)
    save(fig, "s3_obb_multiscale")
    print(f"[obb] {label}@{frame}: chosen kernel {ks[chosen]} volume {vols[chosen]:.4f}; volumes {vols.round(4)}")


# ---------------------------------------------------------------- panel 5
def panel_obb_proposal_vs_final(frame: str = SC.cfg("pf_frame")):
    obs = objects(frame)
    P, C = cloud_final(150_000)
    fig, ax = fig3d(size=(3.4, 2.8))
    ext = np.vstack([P] + [np.repeat(np.array(ob["corners_final"]), 300, axis=0) for ob in obs])
    set_equal(ax, ext)
    final_floor(ax, alpha=0.2)
    scatter(ax, P, dim(C), s=0.12, alpha=0.35)
    for ob in obs:
        draw_box_ordered(ax, to_final(np.array(ob["corners_world"])), COL["obb_prop"], lw=0.8, ls="--", alpha=0.9)
        draw_box_ordered(ax, np.array(ob["corners_final"]), LABEL_COL[ob["label"]], lw=1.2)
    handles = [Line2D([], [], color=LABEL_COL[ob["label"]], lw=1.2, label=ob["label"]) for ob in obs]
    handles.append(Line2D([], [], color=COL["obb_prop"], lw=0.8, ls="--", label="proposal"))
    ax.legend(handles=handles, loc="upper right", fontsize=5.5, frameon=False, handlelength=1.6,
              borderaxespad=0.0, labelspacing=0.25, bbox_to_anchor=(1.0, 0.97))
    ax.set_title(f"Proposal vs corrected boxes (t={int(stem_of(frame))})", pad=-2)
    save(fig, "s3_obb_proposal_vs_final")


# ---------------------------------------------------------------- panel 6
def panel_boxes_over_time(frames=SC.cfg("time_frames")):
    """Early / mid / late; 000240 is the last annotated frame before the camera pans at the end of
    the video and the annotator re-places the boxes (from 000257 the 'bed' is a 0.5 m slab at the
    doorway), which would read as objects jumping rather than persisting."""
    allb = [np.array(ob["corners_final"]) for f in frames for ob in objects(f)]
    lo, hi = box_ext(allb, 0.25)
    P, C = cloud_final(None)
    m = in_box(P, lo, hi)
    P, C = P[m], C[m]
    fig = plt.figure(figsize=(7.4, 2.3)); fig.patch.set_alpha(0)
    gs = gridspec.GridSpec(1, 3, wspace=0.0, left=0.0, right=1.0, top=0.92, bottom=0.02)
    labels = []
    for i, f in enumerate(frames):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        ax.view_init(**VIEW); ax.set_axis_off(); ax.patch.set_alpha(0)
        manual_zorder(ax)
        set_limits(ax, lo, hi)
        final_floor(ax, alpha=0.18)
        scatter(ax, P, dim(C, 0.45), s=0.25, alpha=0.4, zorder=Z_CLOUD)
        for ob in objects(f):
            bx = np.array(ob["corners_final"])
            lw = 2.0 if ob["label"] == "person" else 1.0
            draw_box_ordered(ax, bx, LABEL_COL[ob["label"]], lw=lw)
            if ob["label"] not in labels:
                labels.append(ob["label"])
        draw_frustum(ax, refined_pose(stem_of(f)), COL["cam_ref"], scale=0.28, lw=1.0)
        lift_lines(ax)
        ax.set_title(f"t = {int(stem_of(f))}", pad=-2)
        if i == 0:
            handles = [Line2D([], [], color=LABEL_COL[l], lw=2.0 if l == "person" else 1.0, label=l) for l in labels]
            handles.append(Line2D([], [], color=COL["cam_ref"], lw=1.0, label="$T^t$"))
            ax.legend(handles=handles, loc="upper left", fontsize=5.5, frameon=False, handlelength=1.6,
                      labelspacing=0.2, borderaxespad=0.0, bbox_to_anchor=(0.0, 0.98))
    save(fig, "s3_boxes_over_time")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["icp", "4d", "smpl", "smpl_posed", "ms", "pf", "time"]
    fns = {"icp": panel_icp_residual if (BUNDLE_DIR / "static_icp.npz").exists() else panel_icp_before_after, "4d": panel_scene_4d, "smpl": panel_floor_smpl,
           "smpl_posed": lambda: panel_floor_smpl(k=SC.cfg("smpl_k"), posed=True),
           "ms": panel_obb_multiscale, "pf": panel_obb_proposal_vs_final, "time": panel_boxes_over_time}
    for w in which:
        fns[w]()
