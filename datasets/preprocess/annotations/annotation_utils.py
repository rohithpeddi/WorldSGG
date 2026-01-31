import contextlib
import gc
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
# import rerun as rr
import torch
from scipy.spatial.transform import Rotation as SciRot

sys.path.insert(0, os.path.dirname(__file__) + '/..')

# =====================================================================
# COMMON HELPERS
# =====================================================================
def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    stem = Path(video_id).stem
    if not stem:
        return None
    first_letter = stem[0]
    if first_letter.isdigit() and int(first_letter) < 5:
        return "04"
    elif first_letter.isdigit() and int(first_letter) >= 5:
        return "59"
    elif first_letter in "ABCD":
        return "AD"
    elif first_letter in "EFGH":
        return "EH"
    elif first_letter in "IJKL":
        return "IL"
    elif first_letter in "MNOP":
        return "MP"
    elif first_letter in "QRST":
        return "QT"
    elif first_letter in "UVWXYZ":
        return "UZ"
    return None


def _faces_u32(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces)
    if faces.dtype != np.uint32:
        faces = faces.astype(np.uint32)
    return faces


def _pinhole_from_fov(w: int, h: int, fov_y: float):
    fy = 0.5 * h / np.tan(0.5 * fov_y)
    fx = fy
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


def _is_empty_array(x):
    if x is None:
        return True
    if isinstance(x, (list, tuple)):
        return len(x) == 0
    try:
        return getattr(x, "numel", None) and x.numel() == 0
    except Exception:
        pass
    try:
        return hasattr(x, "size") and hasattr(x, "shape") and x.size == 0
    except Exception:
        pass
    return False


def _as_np(x, dtype=None):
    if x is None:
        return None
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def _load_pkl_if_exists(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _xywh_to_xyxy(b):  # [x,y,w,h] -> [x1,y1,x2,y2]
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]


def _resize_bbox_to(box_xyxy, original_size, target_size):
    orig_W, orig_H = original_size
    tgt_W, tgt_H = target_size
    x_scale = tgt_W / orig_W
    y_scale = tgt_H / orig_H
    return [
        box_xyxy[0] * x_scale,
        box_xyxy[1] * y_scale,
        box_xyxy[2] * x_scale,
        box_xyxy[3] * y_scale,
    ]


def _area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = _area_xyxy(a) + _area_xyxy(b) - inter
    return inter / max(ua, 1e-8)


def _union_boxes_xyxy(boxes: List[List[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def _mask_from_bbox(h: int, w: int, xyxy: List[float]) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def _resize_mask_to(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if mask.shape == (th, tw):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)


def _finite_and_nonzero(pts: np.ndarray) -> np.ndarray:
    good = np.isfinite(pts).all(axis=-1)
    nz = np.linalg.norm(pts, axis=-1) > 1e-12
    return good & nz


def transform_pts_R_offset(pts: np.ndarray, R: np.ndarray, offset: np.ndarray):
    return (R @ pts.T).T + offset[None, :]


def inv_transform_pts_R_offset(pts: np.ndarray, R: np.ndarray, offset: np.ndarray):
    # inverse of x' = R x + o  -> x = R^T (x' - o)
    return (R.T @ (pts - offset[None, :]).T).T


def _box_edges_from_corners(corners: np.ndarray) -> List[np.ndarray]:
    idx_pairs = [
        (0, 1), (0, 2), (0, 4),
        (7, 6), (7, 5), (7, 3),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (4, 5), (4, 6)
    ]
    return [np.vstack([corners[i], corners[j]]) for (i, j) in idx_pairs]


def _lift_2d_to_3d(frame_points_hw3: np.ndarray, u: float, v: float):
    H, W, _ = frame_points_hw3.shape
    ui = int(round(u))
    vi = int(round(v))
    if ui < 0 or ui >= W or vi < 0 or vi >= H:
        return None
    p3d = frame_points_hw3[vi, ui]
    if not np.isfinite(p3d).all() or np.abs(p3d).sum() < 1e-6:
        return None
    return p3d


def _build_frame_to_kps_map(results: dict, primary_person_id_1: Optional[str]):
    frame_to_kps = {}
    people_results = results.get("people", {})
    if not people_results:
        return frame_to_kps
    if primary_person_id_1 is None:
        primary_person_id_1 = list(people_results.keys())[0]
    pdata = people_results.get(primary_person_id_1, None)
    if pdata is None:
        return frame_to_kps
    frames = pdata.get("frames", None)
    kp_maps = pdata.get("keypoints_2d_map", None)
    if frames is None or kp_maps is None:
        return frame_to_kps
    for i, fidx in enumerate(frames):
        frame_to_kps[fidx] = kp_maps[i]
    return frame_to_kps


# ---------------------------------------------------------------------
# RERUN VIS HELPERS
# ---------------------------------------------------------------------

def _log_box_lines_rr(path: str, corners: np.ndarray,
                      rgba=(255, 255, 255, 255), radius=0.002):
    edges = _box_edges_from_corners(corners)
    for k, e in enumerate(edges):
        e = np.asarray(e, dtype=np.float32)
        rr.log(
            f"{path}/edge_{k}",
            rr.LineStrips3D(
                [e],
                radii=radius,
                colors=[rgba],
            ),
        )


# ------------------------------------------------------------------
# gdino <-> gt match
# ------------------------------------------------------------------
def _match_gdino_to_gt(
        gt_label: str,
        gt_xyxy: List[float],
        gd_boxes: List[List[float]],
        gd_labels: List[str],
        gd_scores: List[float],
        iou_thr: float = 0.3,
) -> List[float]:
    candidates = [
        (b, s) for b, l, s in zip(gd_boxes, gd_labels, gd_scores)
        if (l == gt_label)
    ]
    if not candidates:
        return gt_xyxy

    passing = [b for (b, s) in candidates if _iou_xyxy(b, gt_xyxy) >= iou_thr]
    if passing:
        return _union_boxes_xyxy(passing)

    best = max(candidates, key=lambda t: t[1])[0]
    return best


# ------------------------------------------------------------------
# HUMAN MESH ALIGNMENT RELATED FUNCTIONS
# ------------------------------------------------------------------
def _choose_primary_actor(results: dict, world4d: Dict[int, dict]) -> Tuple[Optional[str], Optional[int]]:
    people = results.get("people", {})
    if people:
        counts = {pid: len(pdata.get("frames", [])) for pid, pdata in people.items()}
        primary_person_id_1 = max(counts.items(), key=lambda x: x[1])[0]
        primary_track_id_0 = int(primary_person_id_1) - 1
        return primary_person_id_1, primary_track_id_0

    for _, frame_data in world4d.items():
        tids = frame_data.get("track_id", [])
        if tids:
            track_id_0 = int(tids[0]) if not isinstance(tids[0], torch.Tensor) else int(tids[0].item())
            person_id_1 = str(track_id_0 + 1)
            return person_id_1, track_id_0
    return None, None


def _find_actor_index_in_frame(frame_data: dict, primary_track_id_0: Optional[int]) -> Optional[int]:
    track_ids = frame_data.get("track_id", [])
    if len(track_ids) == 0:
        return None
    if primary_track_id_0 is None:
        return 0
    for idx, tid in enumerate(track_ids):
        tid_val = int(tid.item()) if isinstance(tid, torch.Tensor) else int(tid)
        if tid_val == int(primary_track_id_0):
            return idx
    return None


# ------------------------------------------------------------------
# similarity estimation
# ------------------------------------------------------------------
def _similarity_umeyama(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    var_src = (src_c ** 2).sum() / n
    s = (S * np.array([1, 1, np.linalg.det(U @ Vt)])).sum() / var_src
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def _robust_similarity_ransac(
        src: np.ndarray,
        dst: np.ndarray,
        *,
        max_iters: int = 800,
        inlier_thresh: float = 0.03,
        min_inliers: int = 4,
        scale_bounds: Tuple[float, float] = (0.4, 3.0),
):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape
    N = src.shape[0]
    if N < 3:
        return _similarity_umeyama(src, dst)

    best_num = -1
    best_model = None
    iters = min(max_iters, 100 + 30 * N)
    for _ in range(iters):
        idx = np.random.choice(N, 3, replace=False)
        s_cand, R_cand, t_cand = _similarity_umeyama(src[idx], dst[idx])
        s_cand = float(np.clip(s_cand, scale_bounds[0], scale_bounds[1]))
        src_tf = s_cand * (src @ R_cand.T) + t_cand
        err = np.linalg.norm(dst - src_tf, axis=1)
        inliers = err < inlier_thresh
        num_inl = int(inliers.sum())
        if num_inl > best_num and num_inl >= min_inliers:
            s_ref, R_ref, t_ref = _similarity_umeyama(src[inliers], dst[inliers])
            s_ref = float(np.clip(s_ref, scale_bounds[0], scale_bounds[1]))
            best_num = num_inl
            best_model = (s_ref, R_ref, t_ref)
    if best_model is None:
        return _similarity_umeyama(src, dst)
    return best_model


def _mad_based_mask(values: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """
    values: (N,) array
    returns: boolean mask where True = keep (not an outlier)
    """
    if values.size == 0:
        return np.ones_like(values, dtype=bool)
    median = np.median(values)
    abs_dev = np.abs(values - median)
    mad = np.median(abs_dev) + 1e-8  # to avoid div by zero
    modified_z = 0.6745 * abs_dev / mad
    return modified_z < thresh


def _average_sims_robust(per_frame_sims: Dict[int, Dict[str, Any]],
                         rot_thresh_deg: float = 15.0,
                         trans_thresh_scale: float = 3.5
                         ) -> Tuple[float, np.ndarray, np.ndarray]:
    if len(per_frame_sims) == 0:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    ws, scales, trans, quats, rotvecs = [], [], [], [], []

    for _, d in per_frame_sims.items():
        w = float(d.get("w", 1.0))
        s = float(d["s"])
        R = np.asarray(d["R"], dtype=np.float64)
        t = np.asarray(d["t"], dtype=np.float64)

        rot = SciRot.from_matrix(R)
        quat = rot.as_quat()
        rotvec = rot.as_rotvec()  # axis * angle

        ws.append(w)
        scales.append(s)
        trans.append(t)
        quats.append(quat)
        rotvecs.append(rotvec)

    ws = np.asarray(ws, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)  # (N, 3)
    quats = np.asarray(quats, dtype=np.float64)  # (N, 4)
    rotvecs = np.asarray(rotvecs, dtype=np.float64)  # (N, 3)

    # --- 1) scale outliers (MAD)
    scale_mask = _mad_based_mask(scales, thresh=3.5)

    # --- 2) translation outliers:
    # use L2 norm deviation from median
    trans_norms = np.linalg.norm(trans, axis=1)
    trans_mask = _mad_based_mask(trans_norms, thresh=trans_thresh_scale)

    # --- 3) rotation outliers:
    # compare each rotvec magnitude to median magnitude
    rot_mags = np.linalg.norm(rotvecs, axis=1)  # in radians
    rot_mags_deg = np.degrees(rot_mags)
    # simpler than MAD: hard threshold on degrees
    rot_mask = rot_mags_deg < rot_thresh_deg

    # combine masks
    keep_mask = scale_mask & trans_mask & rot_mask

    if not np.any(keep_mask):
        # fallback to naive average
        ws = ws / (ws.sum() + 1e-8)
        s_avg = float((ws * scales).sum())
        t_avg = (ws[:, None] * trans).sum(axis=0)
        q_avg = (ws[:, None] * quats).sum(axis=0)
        q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-8)
        R_avg = SciRot.from_quat(q_avg).as_matrix().astype(np.float32)
        return s_avg, R_avg, t_avg.astype(np.float32)

    # filter
    ws_f = ws[keep_mask]
    scales_f = scales[keep_mask]
    trans_f = trans[keep_mask]
    quats_f = quats[keep_mask]

    # reweight to sum to 1
    ws_f = ws_f / (ws_f.sum() + 1e-8)

    # scale
    s_avg = float((ws_f * scales_f).sum())

    # translation
    t_avg = (ws_f[:, None] * trans_f).sum(axis=0)

    # rotation: weighted quat average on inliers
    q_avg = (ws_f[:, None] * quats_f).sum(axis=0)
    q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-8)
    R_avg = SciRot.from_quat(q_avg).as_matrix().astype(np.float32)

    return s_avg, R_avg, t_avg.astype(np.float32)

def _safe_empty_cuda_cache():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
    except Exception:
        pass

@contextlib.contextmanager
def _torch_inference_ctx():
    # inference_mode() is stronger than no_grad() and reduces autograd overhead
    with torch.inference_mode():
        yield

@contextlib.contextmanager
def _npz_open(path):
    # Use mmap to avoid copying huge arrays outright; ensure file handle closes
    f = np.load(path, allow_pickle=True, mmap_mode='r')
    try:
        yield f
    finally:
        try:
            f.close()
        except Exception:
            pass

def _del_and_collect(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    _safe_empty_cuda_cache()
def _aabb(pts: np.ndarray):
    """
    Axis Aligned Bounding Box from points.
    Returns dict with min, max, center, lengths, corners.
    """
    if pts is None or pts.shape[0] == 0:
        return None
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    center = (min_pt + max_pt) / 2
    lengths = max_pt - min_pt
    
    # 8 corners
    c_list = []
    for ix in [min_pt[0], max_pt[0]]:
        for iy in [min_pt[1], max_pt[1]]:
            for iz in [min_pt[2], max_pt[2]]:
                c_list.append([ix, iy, iz])
    corners = np.array(c_list)
    
    return {
        "min": min_pt,
        "max": max_pt,
        "center": center,
        "lengths": lengths,
        "corners": corners
    }


def _pca_obb(pts: np.ndarray):
    """
    Oriented Bounding Box using PCA on points.
    Returns dict with center, axes, lengths, corners.
    """
    if pts is None or pts.shape[0] < 4:
        # Fallback to AABB if too few points, or return None?
        return _aabb(pts)
        
    mu = pts.mean(axis=0)
    data = pts - mu
    cov = np.cov(data, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort largest to smallest
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sort_idx]
    
    # Project
    proj = data @ eigvecs
    min_pt = proj.min(axis=0)
    max_pt = proj.max(axis=0)
    
    center_local = (min_pt + max_pt) / 2
    lengths = max_pt - min_pt
    
    # Back to world
    center_world = mu + eigvecs @ center_local
    
    # Corners in local
    c_local = []
    for ix in [min_pt[0], max_pt[0]]:
        for iy in [min_pt[1], max_pt[1]]:
            for iz in [min_pt[2], max_pt[2]]:
                c_local.append([ix, iy, iz])
    c_local = np.array(c_local)
    
    # Corners world: mu + c_local * eigvecs^T
    corners_world = mu + c_local @ eigvecs.T
    
    return {
        "center": center_world,
        "axes": eigvecs,
        "lengths": lengths,
        "corners": corners_world
    }
