"""
Geometry helpers for the MLLM tracks (canonical floor-aligned frame).

Conventions (same as ``world4d_rel_annotations_worldbbox`` / ``finalize_world4d_rel.py``):
  * the canonical ("final") frame is z-up with the floor levelled to z = 0;
  * ``corners_final = A_world_to_final @ (corners_world - origin_world)`` where
    ``corners_world`` live in the Pi3 world frame of ``predictions.npz``;
  * a floor-parallel OBB is parameterised as ``center (3), size (3) = (l, w, h),
    yaw`` (rotation about +z of the local x axis), and its 8 corners are emitted
    bottom face first (z_min) then top face (z_max) -- the ordering
    ``lib/detector/monocular3d/evaluation/evaluate_3d.py::compute_iou_3d_obb`` expects
    (it splits bottom/top by z, so any order within a face is fine).
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def make_world_to_final(A_world_to_final, origin_world) -> np.ndarray:
    """4x4 rigid transform T with T @ [p_world, 1] = [A (p_world - origin), 1]."""
    A = np.asarray(A_world_to_final, dtype=np.float64).reshape(3, 3)
    o = np.asarray(origin_world, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = A
    T[:3, 3] = -A @ o
    return T


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points of shape (..., 3)."""
    shp = pts.shape
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    q = p @ T[:3, :3].T + T[:3, 3]
    return q.reshape(shp)


def obb_to_corners(center: Sequence[float], size: Sequence[float], yaw: float) -> np.ndarray:
    """(8, 3) corners of a floor-parallel OBB; bottom 4 first, then top 4."""
    cx, cy, cz = [float(v) for v in center]
    l, w, h = [max(float(v), 1e-6) for v in size]
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]])
    xy = local @ R.T + np.array([cx, cy])
    bottom = np.c_[xy, np.full(4, cz - h / 2)]
    top = np.c_[xy, np.full(4, cz + h / 2)]
    return np.vstack([bottom, top]).astype(np.float64)


def corners_to_obb(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Inverse of :func:`obb_to_corners` for floor-parallel boxes.

    Works for any 8-corner box whose top/bottom faces are horizontal: the
    bottom face (lowest z) is fitted with a minimum-area rectangle in XY
    (exact when the box is a rectangle).  Returns ``(center, size, yaw)`` with
    ``yaw`` in ``(-pi/2, pi/2]`` and ``size = (l, w, h)``, ``l >= w``.
    """
    c = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    z_min, z_max = c[:, 2].min(), c[:, 2].max()
    z_mid = 0.5 * (z_min + z_max)
    bottom = c[c[:, 2] <= z_mid][:, :2]
    if bottom.shape[0] < 3:
        bottom = c[:4, :2]
    hull = _convex_hull(bottom)
    if len(hull) < 3:
        mn, mx = c.min(0), c.max(0)
        return (mn + mx) / 2, np.maximum(mx - mn, 1e-6), 0.0
    center, size, yaw = _min_rect_from_hull(hull, z_min, z_max)
    return center, size, float(yaw)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.asarray(pts, dtype=np.float64), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def scale_bbox(bbox_xyxy: Sequence[float], sx: float, sy: float) -> np.ndarray:
    b = np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)[:4]
    return np.array([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])


def obb_floor_parallel_from_points(pts_final: np.ndarray) -> np.ndarray:
    """Floor-parallel OBB corners (8, 3) enclosing points in the canonical frame
    (min-area rectangle in XY x [z_min, z_max]) -- the ``obb_floor_parallel``
    rule of ``corrected_world_bbox_generator.py`` in the z-up frame."""
    p = np.asarray(pts_final, dtype=np.float64).reshape(-1, 3)
    hull = _convex_hull(p[:, :2])
    if len(hull) < 3:
        mn, mx = p.min(0), p.max(0)
        return obb_to_corners((mn + mx) / 2, np.maximum(mx - mn, 1e-3), 0.0)
    center, size, yaw = _min_rect_from_hull(hull, p[:, 2].min(), p[:, 2].max())
    return obb_to_corners(center, size, yaw)


def _min_rect_from_hull(hull: np.ndarray, z_min: float, z_max: float):
    best = None
    n = len(hull)
    for i in range(n):
        e = hull[(i + 1) % n] - hull[i]
        ang = math.atan2(e[1], e[0])
        R = np.array([[math.cos(-ang), -math.sin(-ang)], [math.sin(-ang), math.cos(-ang)]])
        r = hull @ R.T
        mn, mx = r.min(0), r.max(0)
        area = float((mx[0] - mn[0]) * (mx[1] - mn[1]))
        if best is None or area < best[0] - 1e-12:
            best = (area, ang, mn, mx)
    area, ang, mn, mx = best
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    cxy = R @ ((mn + mx) / 2.0)
    l, w = float(mx[0] - mn[0]), float(mx[1] - mn[1])
    yaw = ang
    if w > l:
        l, w = w, l
        yaw += math.pi / 2
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return np.array([cxy[0], cxy[1], 0.5 * (z_min + z_max)]), np.array([l, w, z_max - z_min]), float(yaw)
