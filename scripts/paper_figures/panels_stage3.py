"""Stage (iv) camera alignment, the final 4D scene, the floor/human panel and the 3-D box
panels of the scene-pipeline figure (video 00T1E).  Writes ``s3_*.png`` into PANEL_DIR.

Run with the scene4cast interpreter::

    C:/Users/rohit/anaconda3/envs/scene4cast/python.exe scripts/paper_figures/panels_stage3.py

Geometry notes (verified numerically, see the report in the session that wrote this file):

* pi3 clip index ``k`` <-> annotated stem ``sampled_idx[k]`` (``pi3_frame_ids == sampled_idx[:77]``);
  stem 000310 has a refined pose but no pi3 pose (pi3 covers the first 77 of 80 sampled frames).
* The ICP refinement barely moves the (static) camera: mean centre shift 5e-3 scene units, max 3e-2;
  mean rotation change 0.9 deg (max 7.3 deg), hence the residual strip in ``s3_icp_before_after``.
* ``smpl_verts_i`` in bundle.npz are the *bind-pose* templates of the skinned world4d.glb (the
  extractor read ``geom.vertices`` and dropped the armature animation).  ``s3_smpl_posed.npz``
  holds the linear-blend-skinned meshes at a few keyframes (kf k <-> pi3 clip k) computed from
  the GLB animation, in PromptHMR's metric world (static camera at (0, 1.29, 0), y-up, floor y=0).
  That world is not the bundle's ``floorsim`` frame; a floor-preserving similarity (scale, yaw,
  xz-shift) is fitted here from the PromptHMR pelvis trajectory to the corrected person boxes.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scene_common import (
    bundle, obb, to_final, pose_to_final, to_floorsim, BUNDLE_DIR,
    fig3d, set_equal, scatter, draw_frustum, draw_box, draw_floor, save,
    COL, LABEL_COL, VIEW, VIEW_FLOORSIM, DPI,
)

B = bundle()
O = obb()
FRAMES = O["frames"]                       # "000010.png" -> [objects]
STEMS = [str(s) for s in B["refined_stems"]]
FIDS = [int(f) for f in B["pi3_frame_ids"]]
RNG = np.random.default_rng(0)
GREY = np.array([0.72, 0.74, 0.78])


# ---------------------------------------------------------------- helpers
def stem_of(frame: str) -> str:
    return frame[:6]


def refined_pose(stem: str) -> np.ndarray:
    return B["refined_poses"][STEMS.index(stem)]


def init_pose_final(stem: str) -> np.ndarray | None:
    s = int(stem)
    return pose_to_final(B["pi3_poses"][FIDS.index(s)]) if s in FIDS else None


def objects(frame: str, label: str | None = None):
    obs = FRAMES[frame]
    return [ob for ob in obs if label is None or ob["label"] == label]


def cloud_final(n: int | None = 150_000):
    P, C = B["pi3_all_points"], B["pi3_all_colors"]
    if n is not None and n < len(P):
        idx = RNG.choice(len(P), n, replace=False)
        P, C = P[idx], C[idx]
    return to_final(P), C.astype(np.float64) / 255.0


def dim(C: np.ndarray, w: float = 0.55) -> np.ndarray:
    """Blend true colours towards grey."""
    return (1 - w) * C + w * GREY


def crop_floor(verts, faces, xlim, ylim, axes=(0, 1)):
    """Keep only floor faces whose centroid lies inside the axes box (mplot3d does not clip)."""
    verts = np.asarray(verts, float); faces = np.asarray(faces, int)
    c = verts[faces].mean(1)
    keep = (c[:, axes[0]] >= xlim[0]) & (c[:, axes[0]] <= xlim[1]) & (c[:, axes[1]] >= ylim[0]) & (c[:, axes[1]] <= ylim[1])
    return verts, faces[keep]


def final_floor(ax, alpha=0.3):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    v, f = crop_floor(B["floor_final_verts"], B["floor_final_faces"], xl, yl)
    draw_floor(ax, v, f, alpha=alpha)


def box_label_pos(corners):
    c = np.asarray(corners, float)
    return c[:, 0].mean(), c[:, 1].mean(), c[:, 2].max()


def label_text(ax, corners, text, color, dz=0.04, **kw):
    x, y, z = box_label_pos(corners)
    ax.text(x, y, z + dz, text, color=color, fontsize=5.5, ha="center", va="bottom", **kw)


# ---------------------------------------------------------------- panel 1
def panel_icp_before_after():
    P, C = cloud_final(150_000)
    init = {s: init_pose_final(s) for s in STEMS}
    init = {s: T for s, T in init.items() if T is not None}
    ref = {s: refined_pose(s) for s in STEMS}
    both = [s for s in STEMS if s in init]
    dC = np.array([np.linalg.norm(init[s][:3, 3] - ref[s][:3, 3]) for s in both])
    dR = []
    for s in both:
        R1, R2 = init[s][:3, :3], ref[s][:3, :3]
        dR.append(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))
    dR = np.array(dR)
    print(f"[icp] {len(both)} matched cameras: |dC| mean {dC.mean():.4f} max {dC.max():.4f}; "
          f"dtheta mean {dR.mean():.2f} deg max {dR.max():.2f} deg")

    fig = plt.figure(figsize=(7.4, 2.7))
    fig.patch.set_alpha(0)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.78], wspace=0.02, left=0.0, right=1.0, top=0.9, bottom=0.05)
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
            draw_frustum(ax, T, colr, scale=0.16, lw=0.7, alpha=0.55)
        ax.set_title(title, pad=-4)

    # residual strip: the refinement is sub-centimetre for a static camera, so show it explicitly
    ax = fig.add_subplot(gs[0, 2])
    ax.patch.set_alpha(0)
    t = np.arange(len(both))
    ax.bar(t, dC * 100, color=COL["cam_ref"], alpha=0.75, width=0.8, label=r"$\|\Delta c_t\|$ ($10^{-2}$ units)")
    ax.set_ylabel(r"camera-centre shift ($10^{-2}$ units)", fontsize=6.5, color=COL["cam_ref"])
    ax.tick_params(labelsize=6)
    ax2 = ax.twinx(); ax2.patch.set_alpha(0)
    ax2.plot(t, dR, color=COL["cam_init"], lw=1.0, marker="o", ms=1.8)
    ax2.set_ylabel(r"rotation change $\Delta\theta_t$ (deg)", fontsize=6.5, color=COL["cam_init"])
    ax2.tick_params(labelsize=6)
    ax.set_xlabel("annotated frame index", fontsize=6.5)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_title(r"$T_t^{\rm init}\!\to T_t^{\rm refined}$", pad=3)
    for a in (ax, ax2):
        for sp in ("top",):
            a.spines[sp].set_visible(False)
    ax.text(0.98, 0.95, f"mean shift {dC.mean()*100:.1f}e-2\nmean rot {dR.mean():.1f}°",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color=COL["muted"])
    save(fig, "s3_icp_before_after")


# ---------------------------------------------------------------- panel 2
def pick_frames(n: int):
    fr = [f for f in FRAMES if stem_of(f) in STEMS]
    idx = np.linspace(0, len(fr) - 1, n).round().astype(int)
    return [fr[i] for i in idx]


def panel_scene_4d():
    P, C = cloud_final(None)
    frames = pick_frames(6)
    boxes = [np.array(objects(f, "person")[0]["corners_final"]) for f in frames]
    cams = [refined_pose(stem_of(f)) for f in frames]
    fig, ax = fig3d(size=(4.0, 3.2))
    ext = np.vstack([P] + [np.repeat(bx, 400, axis=0) for bx in boxes])
    set_equal(ax, ext)
    final_floor(ax, alpha=0.3)
    scatter(ax, P, C, s=0.08, alpha=0.8)
    n = len(frames)
    alphas = np.linspace(1.0, 0.35, n)
    centres = np.array([[bx[:, 0].mean(), bx[:, 1].mean(), bx[:, 2].min()] for bx in boxes])
    ax.plot(centres[:, 0], centres[:, 1], centres[:, 2], color=COL["dynamic"], lw=0.8, ls=":", alpha=0.8)
    for i, (f, bx, T) in enumerate(zip(frames, boxes, cams)):
        a = alphas[i]
        draw_box(ax, bx, COL["dynamic"], lw=1.1, alpha=a)
        draw_frustum(ax, T, COL["cam_ref"], scale=0.18, lw=0.8, alpha=max(a, 0.5))
        label_text(ax, bx, f"t={int(stem_of(f))}", COL["dynamic"], alpha=a)
    ax.set_title(r"4D scene $(\mathcal{S},\{(\mathcal{F}^t, T^t)\}_{t=1}^{T})$", pad=-6)
    save(fig, "s3_scene_4d")
    print("[4d] frames", [stem_of(f) for f in frames])


# ---------------------------------------------------------------- panel 3
def fit_smpl_alignment(Z):
    """Floor-preserving similarity PromptHMR world -> floorsim from the pelvis trajectory."""
    traj = Z["root_traj_0"]
    A, og = B["world_to_final_A"], B["world_to_final_origin"]
    Ainv = np.linalg.inv(A)
    src, dst = [], []
    for f, obs in FRAMES.items():
        s = int(stem_of(f))
        if s not in FIDS:
            continue
        cf = np.array([ob for ob in obs if ob["label"] == "person"][0]["corners_final"])
        q = to_floorsim((Ainv @ cf.T).T + og)
        src.append(traj[FIDS.index(s)]); dst.append(q.mean(0))
    X = np.array(src)[:, [0, 2]]; Y = np.array(dst)[:, [0, 2]]
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc); D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R2 = (U @ D @ Vt).T; s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum(); t2 = my - s * R2 @ mx
    res = np.linalg.norm((s * (R2 @ X.T)).T + t2 - Y, axis=1)
    R = np.eye(3); R[np.ix_([0, 2], [0, 2])] = R2
    t = np.array([t2[0], 0.0, t2[1]])
    print(f"[smpl] PromptHMR->floorsim similarity: s={s:.3f} yaw={np.degrees(np.arctan2(R2[1,0],R2[0,0])):.1f} deg "
          f"t={t.round(3)} rms={np.sqrt((res**2).mean()):.3f} max={res.max():.3f} (n={len(X)})")
    cam = s * R @ np.array([0, 1.29, 0]) + t
    print(f"[smpl] PromptHMR camera -> {cam.round(3)} vs pi3 camera (floorsim) {to_floorsim(B['pi3_poses'][:, :3, 3]).mean(0).round(3)}")
    return lambda P: (s * (R @ np.asarray(P, float).T)).T + t


def floorsim_floor(ax, xlim, zlim, alpha=0.35):
    v, f = crop_floor(B["floor_verts"], B["floor_faces"], xlim, zlim, axes=(0, 2))
    cent = v[f].mean(1)
    cols = np.where(((np.floor(cent[:, 0] / 0.2) + np.floor(cent[:, 2] / 0.2)) % 2 == 0)[:, None],
                    np.array([[0.79, 0.82, 0.86]]), np.array([[0.56, 0.60, 0.66]]))
    pc = Poly3DCollection(v[f], facecolors=cols, edgecolors="none", alpha=alpha)
    ax.add_collection3d(pc)


def shaded_mesh(ax, V, F, base, light=(0.4, 1.0, 0.6), alpha=1.0):
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    l = np.asarray(light, float); l /= np.linalg.norm(l)
    inten = 0.5 + 0.5 * np.clip(nrm @ l, 0, 1)
    base = np.array(plt.matplotlib.colors.to_rgb(base))
    fc = np.clip(base[None] * inten[:, None], 0, 1)
    ec = np.clip(fc * 0.75, 0, 1)
    pc = Poly3DCollection(tri, facecolors=fc, edgecolors=ec, linewidths=0.05, alpha=alpha)
    ax.add_collection3d(pc)


def panel_floor_smpl(kf: int = 65):
    Z = np.load(BUNDLE_DIR / "s3_smpl_posed.npz")
    align = fit_smpl_alignment(Z)
    P = to_floorsim(B["pi3_all_points"])
    idx = RNG.choice(len(P), 150_000, replace=False)
    P, C = P[idx], B["pi3_all_colors"][idx] / 255.0
    V = align(Z[f"verts_0_{kf}"]); F = B["smpl_faces_0"]
    print(f"[smpl] kf {kf} (stem {int(B['sampled_idx'][kf])}): mesh y range {V[:,1].min():.3f}..{V[:,1].max():.3f}, "
          f"centre {V.mean(0).round(3)}; cloud y 1..99 pct {np.percentile(P[:,1],[1,99]).round(3)}")
    fig, ax = fig3d(size=(3.4, 2.8), view=VIEW_FLOORSIM)
    ax.view_init(**VIEW_FLOORSIM, vertical_axis="y")
    ext = np.vstack([P, np.repeat(V, 4, axis=0)])
    set_equal(ax, ext)
    floorsim_floor(ax, ax.get_xlim(), ax.get_zlim())
    scatter(ax, P, C, s=0.12, alpha=0.85)
    shaded_mesh(ax, V, F, COL["smpl"])
    ax.set_title("PromptHMR human + floor alignment", pad=-4)
    save(fig, "s3_floor_smpl")


# ---------------------------------------------------------------- panel 4
def panel_obb_multiscale(frame: str = "000010.png", label: str = "laptop"):
    ob = objects(frame, label)[0]
    cands = sorted(ob["candidates"], key=lambda c: c["kernel_size"])
    ks = np.array([c["kernel_size"] for c in cands]); vols = np.array([c["volume"] for c in cands])
    chosen = int(np.argmin(vols))
    boxes = [to_final(np.array(c["corners_world"])) for c in cands]
    allc = np.vstack(boxes)
    centre = allc.mean(0); r = max((allc.max(0) - allc.min(0)).max() * 1.6, 0.45)
    P, C = cloud_final(None)
    m = np.all(np.abs(P - centre) <= r * 1.05, axis=1)
    P, C = P[m], dim(C[m])
    fig, ax = fig3d(size=(3.4, 2.8))
    for f, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        f(c - r, c + r)
    ax.set_box_aspect((1, 1, 1))
    scatter(ax, P, C, s=0.5, alpha=0.35)
    n = len(cands)
    for i, bx in enumerate(boxes):
        a = 0.15 + 0.85 * i / (n - 1)
        if i == chosen:
            draw_box(ax, bx, COL["obb_final"], lw=1.8, alpha=1.0)
        else:
            draw_box(ax, bx, COL["obb_prop"], lw=0.7, alpha=a)
    label_text(ax, boxes[chosen], f"{label}: k={ks[chosen]}, min volume", COL["obb_final"], dz=0.02)
    ax.set_title("Multi-scale erosion \u2192 min-volume OBB", pad=-4)
    # inline bar chart
    ins = fig.add_axes([0.68, 0.10, 0.29, 0.24]); ins.patch.set_alpha(0)
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
def panel_obb_proposal_vs_final(frame: str = "000010.png"):
    obs = objects(frame)
    P, C = cloud_final(150_000)
    fig, ax = fig3d(size=(3.4, 2.8))
    ext = np.vstack([P] + [np.repeat(np.array(ob["corners_final"]), 300, axis=0) for ob in obs])
    set_equal(ax, ext)
    final_floor(ax, alpha=0.2)
    scatter(ax, P, dim(C), s=0.12, alpha=0.35)
    for ob in obs:
        draw_box(ax, to_final(np.array(ob["corners_world"])), COL["obb_prop"], lw=0.8, ls="--", alpha=0.9)
        draw_box(ax, np.array(ob["corners_final"]), LABEL_COL[ob["label"]], lw=1.2)
    handles = [Line2D([], [], color=LABEL_COL[ob["label"]], lw=1.2, label=ob["label"]) for ob in obs]
    handles.append(Line2D([], [], color=COL["obb_prop"], lw=0.8, ls="--", label="proposal"))
    ax.legend(handles=handles, loc="upper right", fontsize=5.5, frameon=False, handlelength=1.6,
              borderaxespad=0.0, labelspacing=0.3, bbox_to_anchor=(1.02, 0.98))
    ax.set_title(f"Proposal vs corrected boxes (t={int(stem_of(frame))})", pad=-4)
    save(fig, "s3_obb_proposal_vs_final")


# ---------------------------------------------------------------- panel 6
def panel_boxes_over_time(frames=("000010.png", "000163.png", "000297.png")):
    P, C = cloud_final(120_000)
    allb = [np.array(ob["corners_final"]) for f in frames for ob in objects(f)]
    ext = np.vstack([P] + [np.repeat(bx, 200, axis=0) for bx in allb])
    fig = plt.figure(figsize=(7.4, 2.5)); fig.patch.set_alpha(0)
    gs = gridspec.GridSpec(1, 3, wspace=0.0, left=0.0, right=1.0, top=0.92, bottom=0.02)
    for i, f in enumerate(frames):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        ax.view_init(**VIEW); ax.set_axis_off(); ax.patch.set_alpha(0)
        set_equal(ax, ext)
        final_floor(ax, alpha=0.2)
        scatter(ax, P, dim(C, 0.45), s=0.1, alpha=0.4)
        for ob in objects(f):
            bx = np.array(ob["corners_final"])
            draw_box(ax, bx, LABEL_COL[ob["label"]], lw=1.1)
            label_text(ax, bx, ob["label"], LABEL_COL[ob["label"]], dz=0.03)
        draw_frustum(ax, refined_pose(stem_of(f)), COL["cam_ref"], scale=0.2, lw=1.0)
        ax.set_title(f"t = {int(stem_of(f))}", pad=-4)
    save(fig, "s3_boxes_over_time")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["icp", "4d", "smpl", "ms", "pf", "time"]
    fns = {"icp": panel_icp_before_after, "4d": panel_scene_4d, "smpl": panel_floor_smpl,
           "ms": panel_obb_multiscale, "pf": panel_obb_proposal_vs_final, "time": panel_boxes_over_time}
    for w in which:
        fns[w]()
