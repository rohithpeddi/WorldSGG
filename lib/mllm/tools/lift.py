"""
2D box -> 3D OBB lift from Pi3 points (B4, annotation-INDEPENDENT).

The rule of ``corrected_world_bbox_generator.py::_lift_2d_bbox_to_3d`` in the
canonical z-up frame: select the Pi3 points of the frame inside the (Pi-3-space)
box, at several erosion levels of the box mask, keep confident points, fit a
floor-parallel OBB (min-area rectangle in XY x [z_min, z_max]) and take the
smallest-volume candidate that still has enough points.  A robust percentile
trim per axis removes background points that leak through the box.

Cache: ``<caches.lifted3d>/<video>.json`` keyed by ``"<frame>.png|<label>|<x1,y1,x2,y2>"``
(box rounded to the pixel), values ``{"corners": [[..]*8], "n_points", "kernel",
"center", "size", "yaw"}`` or ``null`` when the box has too few points.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from lib.mllm.data.geometry import corners_to_obb, obb_floor_parallel_from_points
from lib.mllm.data.worldbbox import WorldBBoxVideo

logger = logging.getLogger(__name__)

KERNELS = (0, 5, 11)
MIN_POINTS = 25


def _erode(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    try:
        import cv2
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(mask.astype(np.uint8), ker, iterations=1).astype(bool)
    except Exception:
        from scipy.ndimage import binary_erosion
        return binary_erosion(mask, iterations=k // 2)


def lift_bbox(video: WorldBBoxVideo, k: int, bbox_pi3: Sequence[float], conf_min: float = 0.05,
              trim: float = 5.0, kernels=KERNELS, min_points: int = MIN_POINTS) -> Optional[Dict[str, Any]]:
    """Lift one Pi-3-space box on Pi3 frame ``k`` to OBB corners in the canonical frame."""
    pts, ok = video.points_final(k, conf_min=conf_min)
    H, W = ok.shape
    x1, y1, x2, y2 = [float(v) for v in bbox_pi3]
    x1, x2 = max(0, int(np.floor(x1))), min(W, int(np.ceil(x2)))
    y1, y2 = max(0, int(np.floor(y1))), min(H, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    box = np.zeros((H, W), bool)
    box[y1:y2, x1:x2] = True
    best = None
    for kz in kernels:
        sel = _erode(box, kz) & ok
        n = int(sel.sum())
        if n < min_points:
            continue
        p = pts[sel].reshape(-1, 3).astype(np.float64)
        if trim > 0 and len(p) > 4 * min_points:
            lo, hi = np.percentile(p, trim, 0), np.percentile(p, 100 - trim, 0)
            keep = np.all((p >= lo) & (p <= hi), 1)
            if keep.sum() >= min_points:
                p = p[keep]
        corners = obb_floor_parallel_from_points(p)
        center, size, yaw = corners_to_obb(corners)
        vol = float(np.prod(np.maximum(size, 1e-3)))
        cand = {"corners": corners.tolist(), "n_points": n, "kernel": int(kz),
                "center": center.tolist(), "size": size.tolist(), "yaw": float(yaw), "volume": vol}
        if best is None or vol < best["volume"]:
            best = cand
    return best


class LiftCache:
    def __init__(self, video: WorldBBoxVideo, cache_dir: str):
        self.video = video
        self.path = Path(cache_dir) / f"{video.video_id}.json"
        self._d: Dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._d = json.load(f)
        self._dirty = False

    @staticmethod
    def key(frame_file: str, label: str, bbox_pi3: Sequence[float]) -> str:
        b = ",".join(str(int(round(float(v)))) for v in bbox_pi3)
        return f"{Path(frame_file).name}|{label}|{b}"

    def get(self, frame_file: str, label: str, bbox_pi3: Sequence[float]) -> Optional[Dict[str, Any]]:
        kk = self.key(frame_file, label, bbox_pi3)
        if kk in self._d:
            return self._d[kk]
        fn = int(Path(frame_file).stem)
        k = self.video.pi3_index_for_frame(fn)
        res = None if k is None else lift_bbox(self.video, k, bbox_pi3)
        self._d[kk] = res
        self._dirty = True
        return res

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._d, f)
        os.replace(tmp, self.path)
        self._dirty = False
