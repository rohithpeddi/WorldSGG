"""
Bird's-eye-view renders of the Pi3 scene in the canonical floor frame (B4).

Annotation-INDEPENDENT part (cached once per video under ``<caches.bev>/``):
  * ``<video>_cloud.npz``: voxel-downsampled coloured point cloud of the whole
    clip in the canonical (z-up, floor z=0) frame -- ``xyz (N,3) float32``,
    ``rgb (N,3) uint8``, plus every Pi3 camera pose in that frame and the
    metric BEV window (``meta.json``: ``x0, y0, px_per_m, width, height``);
  * ``<video>_bev.png``: the base top-down render (colour = image RGB, brighter
    = higher), 0.5 m grid with metre labels, camera trajectory (grey dots).

Query-time part (never cached, depends on the object list): :func:`mark_bev`
stamps the target frame's camera (red arrow) and numbered object footprints
(OBB bottom faces) so the same base image serves predcls (GT boxes), sgdet
(lifted boxes) and the Track-B agent (its current graph state).

Pixel <-> metre: ``u = (x - x0) * px_per_m``, ``v = height - (y - y0) * px_per_m``
(y up on the image, x right), so the model can be told the metric extent.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lib.mllm.data.worldbbox import WorldBBoxVideo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------

def build_cloud(video: WorldBBoxVideo, conf_min: float = 0.05, stride: int = 2,
                voxel: float = 0.03, z_max: float = 3.0, max_frames: int = 40) -> Dict[str, np.ndarray]:
    """Accumulate every Pi3 frame (subsampled) into one voxelised cloud in the canonical frame."""
    S = video.pi3_num_frames
    idx = list(range(S)) if S <= max_frames else [int(round(i)) for i in np.linspace(0, S - 1, max_frames)]
    xs, cs = [], []
    for k in idx:
        pts, mask = video.points_final(k, conf_min=conf_min)
        img = video.image(k)
        pts = pts[::stride, ::stride]
        m = mask[::stride, ::stride]
        col = img[::stride, ::stride]
        # generous height band: a few videos have their floor fit above the
        # scene (all points at z<0), so do not clip at the nominal floor
        m &= (pts[..., 2] > -2.0) & (pts[..., 2] < z_max)
        xs.append(pts[m].reshape(-1, 3))
        cs.append(col[m].reshape(-1, 3))
    xyz = np.concatenate(xs, 0) if xs else np.zeros((0, 3), np.float32)
    rgb = np.concatenate(cs, 0) if cs else np.zeros((0, 3), np.uint8)
    if len(xyz) == 0:
        return {"xyz": xyz, "rgb": rgb}
    # voxel downsample: mean position/colour per voxel
    q = np.floor(xyz / voxel).astype(np.int64)
    key = (q[:, 0] - q[:, 0].min()) * 1_000_003 + (q[:, 1] - q[:, 1].min()) * 1_009 + (q[:, 2] - q[:, 2].min())
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    sx = np.zeros((len(uniq), 3), np.float64)
    sc = np.zeros((len(uniq), 3), np.float64)
    np.add.at(sx, inv, xyz)
    np.add.at(sc, inv, rgb.astype(np.float64))
    cnt = cnt[:, None].astype(np.float64)
    poses = np.stack([video.camera_pose_final_for_pi3(k) for k in range(S)], 0).astype(np.float32)
    return {"xyz": (sx / cnt).astype(np.float32), "rgb": (sc / cnt).clip(0, 255).astype(np.uint8),
            "camera_poses_final": poses, "pi3_frame_numbers": np.asarray(video.pi3_frame_numbers(), np.int64)}


def bev_window(xyz: np.ndarray, cams: Optional[np.ndarray], pad: float = 0.5, max_px: int = 768,
               min_px_per_m: float = 60.0) -> Dict[str, float]:
    if len(xyz):
        lo = np.percentile(xyz[:, :2], 1, axis=0)
        hi = np.percentile(xyz[:, :2], 99, axis=0)
    else:
        lo, hi = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
    if cams is not None and len(cams):
        c = cams[:, :2, 3]
        lo = np.minimum(lo, c.min(0))
        hi = np.maximum(hi, c.max(0))
    lo, hi = lo - pad, hi + pad
    ext = np.maximum(hi - lo, 1.0)
    ppm = min(max_px / ext.max(), 200.0)
    ppm = max(ppm, min_px_per_m)
    w, h = int(math.ceil(ext[0] * ppm)), int(math.ceil(ext[1] * ppm))
    return {"x0": float(lo[0]), "y0": float(lo[1]), "px_per_m": float(ppm), "width": w, "height": h}


def to_px(meta: Dict[str, float], x: float, y: float) -> Tuple[float, float]:
    return (x - meta["x0"]) * meta["px_per_m"], meta["height"] - (y - meta["y0"]) * meta["px_per_m"]


def to_m(meta: Dict[str, float], u: float, v: float) -> Tuple[float, float]:
    return u / meta["px_per_m"] + meta["x0"], (meta["height"] - v) / meta["px_per_m"] + meta["y0"]


def _font(size: int = 14):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_base(cloud: Dict[str, np.ndarray], meta: Dict[str, float]) -> Image.Image:
    """Top-down splat: each voxel paints one pixel (highest z wins), height-brightened."""
    W, H = meta["width"], meta["height"]
    canvas = np.full((H, W, 3), 245, np.uint8)
    xyz, rgb = cloud["xyz"], cloud["rgb"]
    if len(xyz):
        u = ((xyz[:, 0] - meta["x0"]) * meta["px_per_m"]).astype(int)
        v = (H - (xyz[:, 1] - meta["y0"]) * meta["px_per_m"]).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z, c = u[ok], v[ok], xyz[ok, 2], rgb[ok].astype(np.float32)
        order = np.argsort(z)                      # draw low first, high last (on top)
        u, v, z, c = u[order], v[order], z[order], c[order]
        shade = np.clip(0.55 + 0.45 * np.clip(z, 0, 2.0) / 2.0, 0, 1)[:, None]
        c = np.clip(c * shade + 25 * (1 - shade), 0, 255).astype(np.uint8)
        # splat each voxel as a square of about one voxel (3 cm) so the map is
        # a continuous surface instead of isolated dots
        r = max(1, int(round(0.03 * meta["px_per_m"])))
        for du in range(r):
            for dv in range(r):
                uu = np.clip(u + du - r // 2, 0, W - 1)
                vv = np.clip(v + dv - r // 2, 0, H - 1)
                canvas[vv, uu] = c
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    f = _font(12)
    # 0.5 m grid, labels every 1 m
    step = 0.5
    x = math.ceil(meta["x0"] / step) * step
    while x < meta["x0"] + W / meta["px_per_m"]:
        u, _ = to_px(meta, x, 0)
        major = abs(x - round(x)) < 1e-6
        d.line([(u, 0), (u, H)], fill=(120, 120, 120) if major else (200, 200, 200), width=1)
        if major:
            d.text((u + 2, H - 14), f"x={x:.0f}", fill=(60, 60, 60), font=f)
        x += step
    y = math.ceil(meta["y0"] / step) * step
    while y < meta["y0"] + H / meta["px_per_m"]:
        _, v = to_px(meta, 0, y)
        major = abs(y - round(y)) < 1e-6
        d.line([(0, v), (W, v)], fill=(120, 120, 120) if major else (200, 200, 200), width=1)
        if major:
            d.text((2, v - 13), f"y={y:.0f}", fill=(60, 60, 60), font=f)
        y += step
    cams = cloud.get("camera_poses_final")
    if cams is not None:
        for P in cams:
            u, v = to_px(meta, P[0, 3], P[1, 3])
            d.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(90, 90, 90))
    d.text((4, 4), f"BEV (top-down), grid 0.5 m, {meta['px_per_m']:.0f} px/m", fill=(0, 0, 0), font=f)
    return img


def mark_bev(base: Image.Image, meta: Dict[str, float], objects: Sequence[Dict[str, Any]],
             camera_pose: Optional[np.ndarray] = None, person_corners: Optional[np.ndarray] = None) -> Image.Image:
    """Stamp numbered OBB footprints (``{"id", "label", "corners"}``), the person (P) and
    the target camera (red arrow = viewing direction) on a copy of ``base``."""
    img = base.copy()
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(13)
    palette = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
               (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128),
               (170, 110, 40), (128, 0, 0), (0, 0, 128), (128, 128, 0), (255, 215, 180)]

    def footprint(corners, color, tag):
        c = np.asarray(corners, np.float64).reshape(8, 3)
        zmid = 0.5 * (c[:, 2].min() + c[:, 2].max())
        bot = c[c[:, 2] <= zmid][:, :2]
        if len(bot) < 3:
            bot = c[:4, :2]
        ctr = bot.mean(0)
        ang = np.arctan2(bot[:, 1] - ctr[1], bot[:, 0] - ctr[0])
        bot = bot[np.argsort(ang)]
        poly = [to_px(meta, x, y) for x, y in bot]
        d.polygon(poly, outline=color + (255,), fill=color + (60,))
        u, v = to_px(meta, ctr[0], ctr[1])
        tw = d.textlength(tag, font=f) if hasattr(d, "textlength") else 8 * len(tag)
        d.rectangle([u - tw / 2 - 3, v - 9, u + tw / 2 + 3, v + 9], fill=(255, 255, 255, 230), outline=color + (255,))
        d.text((u - tw / 2, v - 8), tag, fill=color + (255,), font=f)

    if person_corners is not None and np.any(person_corners):
        footprint(person_corners, (0, 0, 0), "P")
    for i, o in enumerate(objects):
        if o.get("corners") is None or not np.any(o["corners"]):
            continue
        footprint(o["corners"], palette[i % len(palette)], f"{o['id']}")
    if camera_pose is not None:
        P = np.asarray(camera_pose, np.float64)
        pos = P[:3, 3]
        fwd = P[:3, :3] @ np.array([0, 0, 1.0])   # OpenCV: +z forward
        fwd = fwd[:2]
        n = np.linalg.norm(fwd)
        if n > 1e-6:
            fwd = fwd / n
        u0, v0 = to_px(meta, pos[0], pos[1])
        u1, v1 = to_px(meta, pos[0] + 0.5 * fwd[0], pos[1] + 0.5 * fwd[1])
        d.line([(u0, v0), (u1, v1)], fill=(255, 0, 0, 255), width=3)
        d.ellipse([u0 - 5, v0 - 5, u0 + 5, v0 + 5], fill=(255, 0, 0, 255))
        d.text((u0 + 6, v0 + 2), "camera", fill=(255, 0, 0, 255), font=f)
    return img


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def bev_paths(video_id: str, cache_dir: str) -> Dict[str, Path]:
    root = Path(cache_dir)
    stem = Path(video_id).stem
    return {"cloud": root / f"{stem}_cloud.npz", "meta": root / f"{stem}_meta.json",
            "png": root / f"{stem}_bev.png"}


def get_bev(video: WorldBBoxVideo, cache_dir: str, rebuild: bool = False
            ) -> Tuple[Image.Image, Dict[str, float], Dict[str, np.ndarray]]:
    """Base render + window meta + cloud, building the cache on first use."""
    p = bev_paths(video.video_id, cache_dir)
    if not rebuild and all(x.exists() for x in p.values()):
        with open(p["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        z = np.load(p["cloud"])
        cloud = {k: z[k] for k in z.files}
        return Image.open(p["png"]).convert("RGB"), meta, cloud
    cloud = build_cloud(video)
    meta = bev_window(cloud["xyz"], cloud.get("camera_poses_final"))
    img = render_base(cloud, meta)
    p["png"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p["cloud"], **cloud)
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f)
    img.save(p["png"])
    return img, meta, cloud
