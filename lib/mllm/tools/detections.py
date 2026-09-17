"""
Per-frame GDino detections (annotation-INDEPENDENT cache, B4).

Source: ``/data/rohith/ag/detection/gdino_bboxes/<video>.mp4.pkl`` =
``{"<frame>.png": {"boxes": Tensor (n,4) xyxy ORIGINAL pixels, "scores": Tensor (n,),
"labels": [str]}}`` over the sampled frames (``sampled_frames_idx``).  Labels are
free text ("cup glass", "doork", "") and are normalised onto the AG vocabulary
here; empty / unknown labels are dropped.

Cache: ``<caches.detections>/<video>.json`` =
``{"video_id", "frames": {"<frame>.png": [{"label", "score", "bbox" (orig px),
"bbox_pi3" (Pi-3 px)}]}, "pi3_size", "orig_size"}``.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lib.mllm.data.worldbbox import NAME_TO_IDX, WorldBBoxVideo, to_short

logger = logging.getLogger(__name__)

_ALIASES = {
    "glass": "cup", "bottle": "cup", "cup": "cup", "camera": "phone", "phone": "phone",
    "doork": "doorknob", "doornob": "doorknob", "doorknob": "doorknob", "notebook": "paper",
    "paper": "paper", "closet": "closet", "cabinet": "closet", "sofa": "sofa", "couch": "sofa",
}


def normalise_gdino_label(label: str) -> Optional[str]:
    """Free-text GDino label -> AG short label, or None."""
    toks = [t for t in str(label).lower().replace("/", " ").split() if t]
    if not toks:
        return None
    for t in toks:
        if t in _ALIASES:
            return _ALIASES[t]
        s = to_short(t)
        if s in NAME_TO_IDX:
            return s
    return None


def _to_np(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


class _CPUUnpickler(pickle.Unpickler):
    """The GDino pickles hold CUDA tensors; map them to CPU so cache builders
    never touch (or need) a GPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import io
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_gdino_raw(video: WorldBBoxVideo, det_dir: str) -> Dict[str, Any]:
    p = Path(det_dir) / f"{video.video_id}.mp4.pkl"
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return _CPUUnpickler(f).load()


def build_detections(video: WorldBBoxVideo, det_dir: str, min_score: float = 0.25,
                     keep_person: bool = True) -> Dict[str, Any]:
    raw = load_gdino_raw(video, det_dir)
    sx, sy = video.bbox_scale
    out: Dict[str, Any] = {"video_id": video.video_id, "orig_size": list(video.orig_size),
                           "pi3_size": list(video.pi3_size), "frames": {}}
    for fk in sorted(raw.keys()):
        fd = raw[fk]
        boxes = _to_np(fd.get("boxes", np.zeros((0, 4)))).reshape(-1, 4)
        scores = _to_np(fd.get("scores", np.ones(len(boxes)))).reshape(-1)
        labels = list(fd.get("labels", [""] * len(boxes)))
        dets: List[Dict[str, Any]] = []
        for b, s, l in zip(boxes, scores, labels):
            lab = normalise_gdino_label(l)
            if lab is None or (lab == "person" and not keep_person) or float(s) < min_score:
                continue
            b = [float(v) for v in b]
            dets.append({"label": lab, "score": round(float(min(s, 1.0)), 4), "bbox": b,
                         "bbox_pi3": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy],
                         "raw_label": str(l)})
        out["frames"][Path(fk).name] = dets
    return out


def detections_cache_path(video_id: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"{Path(video_id).stem}.json"


def get_detections(video: WorldBBoxVideo, det_dir: str, cache_dir: str, rebuild: bool = False) -> Dict[str, Any]:
    p = detections_cache_path(video.video_id, cache_dir)
    if p.exists() and not rebuild:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    d = build_detections(video, det_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f)
    os.replace(tmp, p)
    return d


def video_level_objects(dets: Dict[str, Any], min_frames: int = 1, min_score: float = 0.3) -> List[str]:
    """Sorted labels detected (>= min_score) in at least ``min_frames`` frames, person excluded."""
    cnt: Dict[str, int] = {}
    for dl in dets.get("frames", {}).values():
        seen = {d["label"] for d in dl if d["score"] >= min_score and d["label"] != "person"}
        for l in seen:
            cnt[l] = cnt.get(l, 0) + 1
    return sorted(l for l, c in cnt.items() if c >= min_frames)
