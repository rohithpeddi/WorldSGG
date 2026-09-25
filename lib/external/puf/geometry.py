"""Pi3 lifting of WSGG object slots into 3D Gaussians (floor-aligned frame).

For every frame of the Track-1 test stream (the ``common_frames`` of
``dataloader.world_ag_dataset.WorldAG`` -- the same frames, in the same order,
as the dump records' ``frame_t``) and every slot with a 2D box, the slot's box
is mapped from the feature-PKL pixel space (``target_size``) to the Pi3 grid of
``pi3_dynamic/<vid>_10/predictions.npz`` and the Pi3 points inside it are
lifted into the canonical frame (``world_to_final``):

* **mean / covariance** from the central 50% x 50% of the box (the region PUF
  uses for its point cloud), after a foreground filter on camera distance
  (median +- 2.5 MAD).  PUF instead projects the box centre with a depth map and
  propagates the 2D box covariance through the pinhole Jacobian (its Eq. 1-7);
  Pi3 gives dense points but no intrinsics, so we use the empirical covariance
  of the lifted points, floored at ``(cov_floor * scale)^2``.
* **extent points**: up to ``max_pts`` foreground points from the whole box,
  accumulated per node to build the oriented floor-aligned output box.

Nothing here reads annotation *relations*; the annotation PKL is used only for
the frame list, the camera/world alignment, and (optionally, for the 3D side
statistic) the GT ``corners_final`` of each slot.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lib.mllm.data.geometry import apply_transform

UNOBSERVED_SOURCES = ("rag", "gdino", "correction")

DEFAULT_PATHS = {
    "ag_root": "/data/rohith/ag",
    "feat_root": "/data/rohith/ag/features/roi_features",
    "annot_test": "/data/rohith/ag/world4d_rel_annotations_worldbbox/test",
    "pi3_dynamic": "/data2/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    "split_file": "/data3/rohith/ag/splits/test_worldbbox_1511.txt",
}


def _to_short(label: str) -> str:
    from lib.mllm.data.worldbbox import to_short
    return to_short(label)


def load_wb_video(vid: str, paths: Dict[str, str]):
    from lib.mllm.data.worldbbox import WorldBBoxVideo
    wb = {
        "ag_root": paths["ag_root"],
        "frames_annotated": f"{paths['ag_root']}/frames_annotated",
        "frames": f"{paths['ag_root']}/frames",
        "pi3_dynamic": paths["pi3_dynamic"],
        "sampled_frames_idx": f"{paths['ag_root']}/sampled_frames_idx",
    }
    p = Path(paths["annot_test"]) / f"{vid}.mp4.pkl"
    if not p.exists():
        p = Path(paths["annot_test"]) / f"{vid}.pkl"
    return WorldBBoxVideo(vid, p, wb)


def common_frames(feat: dict, annot: dict) -> List[str]:
    """Identical to WorldAG._align_frames: sorted feature-frame names that the
    annotation PKL also has."""
    ann = {k.split("/")[-1] if "/" in k else k for k in annot.get("frames", {})}
    return sorted(set(feat.get("frames", {}).keys()) & ann)


def _box_points(P, C, box, conf_min, central: bool, stride_target: int = 48):
    H, W = P.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    if central:
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
        x1, x2, y1, y2 = cx - w / 4, cx + w / 4, cy - h / 4, cy + h / 4
    xa, xb = int(np.clip(np.floor(x1), 0, W - 1)), int(np.clip(np.ceil(x2), 0, W))
    ya, yb = int(np.clip(np.floor(y1), 0, H - 1)), int(np.clip(np.ceil(y2), 0, H))
    if xb <= xa:
        xb = min(W, xa + 1)
    if yb <= ya:
        yb = min(H, ya + 1)
    sx = max(1, (xb - xa) // stride_target)
    sy = max(1, (yb - ya) // stride_target)
    pts = P[ya:yb:sy, xa:xb:sx].reshape(-1, 3)
    cf = C[ya:yb:sy, xa:xb:sx].reshape(-1)
    fin = np.isfinite(pts).all(-1) & (np.abs(pts).sum(-1) > 0)
    m = fin & (cf > conf_min)
    if m.sum() < 8:
        m = fin
    return pts[m]


def _fg_filter(pts, cam_center):
    if len(pts) < 8:
        return pts
    d = np.linalg.norm(pts - cam_center[None], axis=1)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-6
    keep = np.abs(d - med) <= 2.5 * 1.4826 * mad + 1e-3 * med
    return pts[keep] if keep.sum() >= 5 else pts


def lift_box(P, C, cam_center, box, scale, conf_min=0.05, cov_floor=0.03, max_pts=96, rng=None):
    """-> (mean(3), cov(3,3), n_pts, extent_pts(M,3)) or None."""
    if box is None or not np.all(np.isfinite(box)) or (box[2] - box[0]) <= 0 or (box[3] - box[1]) <= 0:
        return None
    core = _fg_filter(_box_points(P, C, box, conf_min, central=True), cam_center)
    if len(core) < 5:
        return None
    mean = np.median(core, axis=0)
    cov = np.cov(core.T) if len(core) > 3 else np.zeros((3, 3))
    cov = cov + np.eye(3) * (cov_floor * scale) ** 2
    full = _fg_filter(_box_points(P, C, box, conf_min, central=False), cam_center)
    if len(full) > max_pts:
        rng = rng or np.random.RandomState(0)
        full = full[rng.choice(len(full), max_pts, replace=False)]
    return (mean.astype(np.float32), cov.astype(np.float32), int(len(core)),
            full.astype(np.float16))


def video_scale(v, P_final: np.ndarray, C: np.ndarray, cams: np.ndarray, ks: List[int]) -> float:
    """Median camera-to-point distance (canonical units) over the used Pi3 frames.
    Pi3 is up to scale, so every metric PUF constant is expressed in this unit."""
    ds = []
    for k in ks[:: max(1, len(ks) // 8)]:
        m = np.isfinite(P_final[k]).all(-1) & (C[k] > 0.05)
        if m.sum() < 100:
            m = np.isfinite(P_final[k]).all(-1)
        pts = P_final[k][m]
        if len(pts) > 4000:
            pts = pts[:: len(pts) // 4000]
        ds.append(np.median(np.linalg.norm(pts - cams[k][None], axis=1)))
    return float(np.median(ds)) if ds else 1.0


def build_video_geometry(vid: str, modes=("predcls", "sgdet"), paths: Optional[Dict[str, str]] = None,
                         feature_model: str = "resnet50", max_objects: int = 64,
                         conf_min: float = 0.05, cov_floor: float = 0.03) -> Dict[str, Any]:
    paths = {**DEFAULT_PATHS, **(paths or {})}
    v = load_wb_video(vid, paths)
    annot = v.annot
    T_w2f = v.T_world_to_final
    P = np.asarray(v.pi3["points"], dtype=np.float32)
    S, H3, W3 = P.shape[:3]
    C = np.asarray(v.pi3["conf"], dtype=np.float32)[..., 0]
    cams = (T_w2f[None] @ np.asarray(v.pi3["camera_poses"], dtype=np.float64))[:, :3, 3]
    rng = np.random.RandomState(0)
    out: Dict[str, Any] = {"video_id": vid, "pi3_hw": (H3, W3), "modes": {}}

    per_mode_frames = {}
    used_k = set()
    feats = {}
    for mode in modes:
        fp = Path(paths["feat_root"]) / mode / feature_model / "test" / f"{vid}.pkl"
        with open(fp, "rb") as f:
            feat = pickle.load(f)
        # keep only what we need (roi features are large)
        slim = {}
        for fn, ff in feat["frames"].items():
            slim[fn] = {k: ff.get(k) for k in ("labels", "label_ids", "sources", "bboxes_xyxy",
                                               "scores", "target_size")}
        feats[mode] = {"frames": slim}
        del feat
        frames = common_frames(feats[mode], annot)
        per_mode_frames[mode] = frames
        for fn in frames:
            k = v.pi3_index_for_frame(int(Path(fn).stem))
            if k is not None:
                used_k.add(k)
    P_final = {k: apply_transform(T_w2f, P[k]).astype(np.float32) for k in sorted(used_k)}
    del P
    scale = video_scale(v, P_final, C, cams, sorted(used_k))
    out["scale"] = scale

    ann_frames = {k.split("/")[-1]: fr for k, fr in annot["frames"].items()}
    for mode in modes:
        frames = per_mode_frames[mode]
        recs = []
        n_missing = 0
        for fn in frames:
            ff = feats[mode]["frames"][fn]
            k = v.pi3_index_for_frame(int(Path(fn).stem))
            labels = list(ff.get("labels") or [])
            N = min(len(labels), max_objects)
            label_ids = np.asarray(list(ff.get("label_ids") or [])[:N], dtype=np.int64)
            sources = list(ff.get("sources") or [])[:N]
            scores = ff.get("scores")
            scores = np.asarray(list(scores)[:N], dtype=np.float32) if scores is not None else np.ones(N, np.float32)
            boxes = np.asarray(ff.get("bboxes_xyxy") if ff.get("bboxes_xyxy") is not None else np.zeros((0, 4)),
                               dtype=np.float64)[:N]
            ts = ff.get("target_size")
            # GT corners per slot by label (predcls-style label match; side statistic only)
            af = ann_frames.get(fn, {})
            by_label = {}
            for o in af.get("object_info_list", []) or []:
                s = o.get("label") or _to_short(o.get("class", ""))
                by_label.setdefault(s, o)
            gt_c = np.full((N, 8, 3), np.nan, dtype=np.float32)
            for i in range(N):
                o = af.get("person_info", {}) if (label_ids[i] == 1) else by_label.get(_to_short(labels[i]))
                c = None if o is None else o.get("corners_final")
                if c is not None and np.asarray(c).shape == (8, 3):
                    gt_c[i] = np.asarray(c, dtype=np.float32)
            obs = {}
            cam = None
            if k is None:
                n_missing += 1
            else:
                cam = cams[k].astype(np.float32)
                if ts is not None:
                    tw, th = float(ts[0]), float(ts[1])
                else:
                    tw, th = float(W3), float(H3)
                s_xy = np.array([W3 / tw, H3 / th, W3 / tw, H3 / th])
                for i in range(N):
                    if i >= len(boxes):
                        break
                    g = lift_box(P_final[k], C[k], cams[k], boxes[i] * s_xy, scale,
                                 conf_min=conf_min, cov_floor=cov_floor, rng=rng)
                    if g is not None:
                        obs[i] = g
            recs.append({"frame": fn, "pi3_k": k, "cam": cam, "label_ids": label_ids,
                         "sources": sources, "scores": scores, "obs": obs, "gt_corners": gt_c.astype(np.float16)})
        out["modes"][mode] = {"frames": frames, "per_frame": recs, "n_frames_no_pi3": n_missing}
    return out
