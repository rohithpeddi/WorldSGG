"""
worldbbox data adapter (B1 of docs/EXECUTION_PLAN_WORLDBBOX.md §3)
===================================================================

One place that joins, per test video:

* the locked 1,511-video list (``/data3/rohith/ag/splits/test_worldbbox_1511.txt``);
* the ``world4d_rel_annotations_worldbbox/test/<video>.mp4.pkl`` annotation
  (per-frame person / objects with ``bbox_2d`` (original frame pixels, xyxy),
  ``corners_final`` (8,3) in the canonical z-up floor frame, ``visible``,
  ``source``, the three relationship lists; video-level ``camera_poses``
  (final frame, aligned with ``camera_frame_keys``), ``floor_final``,
  ``world_to_final``);
* the Pi3 dynamic scene ``pi3_dynamic/<video>_10/predictions.npz`` (raw Pi3
  world frame; keys ``points/local_points (S,H,W,3)``, ``conf (S,H,W,1)``,
  ``camera_poses (S,4,4)``, ``images (S,H,W,3) in [0,1]``);
* the frame index bookkeeping that links the two: Pi3 frame ``k`` is
  ``sampled_frames_idx[start + k]`` where ``start`` is recovered per video by
  matching ``T_world_to_final @ pi3_pose`` against the annotation poses.

Everything geometric is exposed in the canonical frame (``*_final``), so the
BEV renderer, the 2D->3D lifter, the Track-A/B prompts and the 3D evaluator all
speak one coordinate system.  Annotation-DEPENDENT fields (object lists,
relations, boxes) are separated from annotation-INDEPENDENT ones (Pi3 points,
frames, transforms) so the caches of B4 can be registered as independent.

Also provides :class:`WorldBBoxAgData`, a drop-in for the vendored
``core.ag_data.AgDataBBAnnotations`` so the baseline runners read their object
lists from the worldbbox PKLs (``inference.annotation_source: worldbbox``).
"""
from __future__ import annotations

import logging
import os
import pickle
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lib.mllm.data.geometry import (
    apply_transform, corners_to_obb, make_world_to_final, scale_bbox,
)

logger = logging.getLogger(__name__)

# Canonical vocab (dataloader/world_ag_dataset.py is the source of truth; the
# lists are duplicated here only so this module has no torch dependency).
OBJECT_CLASSES = [
    "__background__", "person", "bag", "bed", "blanket", "book", "box",
    "broom", "chair", "closet/cabinet", "clothes", "cup/glass/bottle",
    "dish", "door", "doorknob", "doorway", "floor", "food", "groceries",
    "laptop", "light", "medicine", "mirror", "paper/notebook",
    "phone/camera", "picture", "pillow", "refrigerator", "sandwich",
    "shelf", "shoe", "sofa/couch", "table", "television", "towel",
    "vacuum", "window",
]
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet", "cup/glass/bottle": "cup",
    "paper/notebook": "paper", "sofa/couch": "sofa", "phone/camera": "phone",
}
LABEL_DENORMALIZE_MAP = {v: k for k, v in LABEL_NORMALIZE_MAP.items()}
NAME_TO_IDX = {n: i for i, n in enumerate(OBJECT_CLASSES) if i > 0}
for _s, _f in LABEL_DENORMALIZE_MAP.items():
    NAME_TO_IDX[_s] = NAME_TO_IDX[_f]

ATTENTION_RELATIONSHIPS = ["looking_at", "not_looking_at", "unsure"]
SPATIAL_RELATIONSHIPS = ["above", "beneath", "in_front_of", "behind", "on_the_side_of", "in"]
CONTACTING_RELATIONSHIPS = [
    "carrying", "covered_by", "drinking_from", "eating", "have_it_on_the_back",
    "holding", "leaning_on", "lying_on", "not_contacting", "other_relationship",
    "sitting_on", "standing_on", "touching", "twisting", "wearing", "wiping",
    "writing_on",
]

UNOBSERVED_SOURCES = ("rag", "gdino", "correction")


def to_short(label: str) -> str:
    return LABEL_NORMALIZE_MAP.get(label, label)


def to_full(label: str) -> str:
    return LABEL_DENORMALIZE_MAP.get(label, label)


def png_size(path: str) -> Tuple[int, int]:
    """(w, h) from a PNG header."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) >= 24 and head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", head[16:24])
        return int(w), int(h)
    from PIL import Image
    with Image.open(path) as im:
        return im.size


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _wb_cfg(cfg: Optional[dict] = None) -> dict:
    if cfg is None:
        from lib.mllm.core.config_loader import load_config
        cfg = load_config()
    paths = cfg.get("paths", {})
    wb = dict(paths.get("worldbbox", {}))
    wb.setdefault("ag_root", paths.get("ag_root", "/data/rohith/ag"))
    wb.setdefault("frames_annotated", paths.get("ag_frames_annotated",
                                                os.path.join(wb["ag_root"], "frames_annotated")))
    wb.setdefault("frames", paths.get("ag_frames", os.path.join(wb["ag_root"], "frames")))
    wb.setdefault("pi3_dynamic", paths.get("dynamic_scenes"))
    wb.setdefault("sampled_frames_idx", os.path.join(wb["ag_root"], "sampled_frames_idx"))
    return wb


class _LazyNpz:
    """Per-key lazy reader for a ``.npz``; ``shape(key)`` parses only the .npy header."""

    def __init__(self, path):
        self.path = Path(path)
        self._cache: Dict[str, np.ndarray] = {}
        self._shapes: Dict[str, tuple] = {}

    @property
    def files(self) -> List[str]:
        import zipfile
        with zipfile.ZipFile(self.path) as zf:
            return [n[:-4] for n in zf.namelist() if n.endswith(".npy")]

    def shape(self, key: str) -> tuple:
        if key in self._cache:
            return self._cache[key].shape
        if key not in self._shapes:
            import zipfile
            with zipfile.ZipFile(self.path) as zf, zf.open(f"{key}.npy") as f:
                version = np.lib.format.read_magic(f)
                try:
                    shp, _, _ = np.lib.format._read_array_header(f, version)
                except AttributeError:          # numpy >= 2.3 keeps the helper in a private module
                    from numpy.lib import _format_impl
                    shp, _, _ = _format_impl._read_array_header(f, version)
            self._shapes[key] = tuple(int(x) for x in shp)
        return self._shapes[key]

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            with np.load(self.path) as z:
                self._cache[key] = z[key]
        return self._cache[key]

    def __contains__(self, key: str) -> bool:
        return key in self.files


# ---------------------------------------------------------------------------
# Per-video record
# ---------------------------------------------------------------------------

@dataclass
class ObjectAnnot:
    label: str                      # short form ("phone")
    cls: str                        # full AG name ("phone/camera")
    class_idx: int                  # index into OBJECT_CLASSES
    bbox_2d: Optional[np.ndarray]   # (4,) xyxy in ORIGINAL frame pixels, or None
    visible: bool
    source: str                     # gt | gdino | rag | correction
    attention: List[str]
    spatial: List[str]
    contacting: List[str]
    corners_final: Optional[np.ndarray]   # (8,3) canonical frame, or None
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def observed(self) -> bool:
        return self.visible and self.source not in UNOBSERVED_SOURCES


@dataclass
class FrameAnnot:
    key: str                        # "<video>.mp4/<frame>.png"
    file: str                       # "<frame>.png"
    frame_num: int
    person_bbox_2d: Optional[np.ndarray]
    person_corners_final: Optional[np.ndarray]
    objects: List[ObjectAnnot]
    camera_pose_final: Optional[np.ndarray]   # (4,4) cam->final, if in camera_frame_keys
    pi3_index: Optional[int]                  # index into predictions.npz, if mapped


class WorldBBoxVideo:
    """Lazy per-video join of annotation + Pi3 scene (see module doc)."""

    def __init__(self, video_id: str, annot_path: Path, wb: dict):
        self.video_id = video_id                 # stem, e.g. "Q38XP"
        self.vid_mp4 = f"{video_id}.mp4"
        self.annot_path = Path(annot_path)
        self._wb = wb
        self._annot: Optional[dict] = None
        self._pi3: Optional[dict] = None
        self._pi3_start: Optional[int] = None
        self._sfi: Optional[List[int]] = None
        self._orig_size: Optional[Tuple[int, int]] = None
        self._frames: Optional[List[FrameAnnot]] = None

    # ---- raw annotation -------------------------------------------------
    @property
    def annot(self) -> dict:
        if self._annot is None:
            with open(self.annot_path, "rb") as f:
                self._annot = pickle.load(f)
        return self._annot

    @property
    def T_world_to_final(self) -> np.ndarray:
        w2f = self.annot["world_to_final"]
        return make_world_to_final(w2f["A_world_to_final"], w2f["origin_world"])

    @property
    def floor_final(self) -> Optional[Dict[str, np.ndarray]]:
        fl = self.annot.get("floor_final")
        if not fl:
            return None
        return {k: np.asarray(v) for k, v in fl.items() if v is not None}

    @property
    def leveled(self) -> dict:
        return self.annot.get("leveled") or {}

    @property
    def camera_frame_keys(self) -> List[str]:
        return list(self.annot.get("camera_frame_keys") or [])

    @property
    def camera_poses_final(self) -> Optional[np.ndarray]:
        cp = self.annot.get("camera_poses")
        return None if cp is None else np.asarray(cp, dtype=np.float64)

    # ---- frame bookkeeping ---------------------------------------------
    @property
    def orig_size(self) -> Tuple[int, int]:
        """(w, h) of the original frames (from the first annotated PNG)."""
        if self._orig_size is None:
            for sub in (self._wb["frames_annotated"], self._wb["frames"]):
                d = Path(sub) / self.vid_mp4
                if d.is_dir():
                    pngs = sorted(p for p in os.listdir(d) if p.endswith(".png"))
                    if pngs:
                        self._orig_size = png_size(str(d / pngs[0]))
                        break
            if self._orig_size is None:
                raise FileNotFoundError(f"[{self.video_id}] no frames under {self._wb['frames_annotated']}")
        return self._orig_size

    @property
    def sampled_frames_idx(self) -> List[int]:
        if self._sfi is None:
            p = Path(self._wb["sampled_frames_idx"]) / f"{self.video_id}.npy"
            self._sfi = [int(x) for x in np.load(p).tolist()]
        return self._sfi

    def frame_path(self, frame_file: str) -> Path:
        for sub in (self._wb["frames_annotated"], self._wb["frames"]):
            p = Path(sub) / self.vid_mp4 / frame_file
            if p.exists():
                return p
        raise FileNotFoundError(f"{self.vid_mp4}/{frame_file}")

    # ---- Pi3 scene (annotation-INDEPENDENT) -----------------------------
    @property
    def pi3_path(self) -> Path:
        return Path(self._wb["pi3_dynamic"]) / f"{self.video_id}_10" / "predictions.npz"

    @property
    def pi3(self) -> "_LazyNpz":
        """Raw Pi3 outputs, loaded per key on first access (an .npz cannot be
        memory-mapped; ``points``/``images`` are ~100 MB each per video)."""
        if self._pi3 is None:
            self._pi3 = _LazyNpz(self.pi3_path)
        return self._pi3

    @property
    def pi3_num_frames(self) -> int:
        return int(self.pi3.shape("camera_poses")[0])

    @property
    def pi3_size(self) -> Tuple[int, int]:
        """(W, H) of the Pi3 grids = the Pi-3 feature space of WorldAG boxes."""
        _, H, W = self.pi3.shape("conf")[:3]
        return int(W), int(H)

    @property
    def bbox_scale(self) -> Tuple[float, float]:
        """(sx, sy): original frame pixels -> Pi-3 space (same rule as WorldAG._bbox_scale)."""
        ow, oh = self.orig_size
        tw, th = self.pi3_size
        return tw / ow, th / oh

    @property
    def pi3_start(self) -> int:
        """Offset such that Pi3 frame k shows video frame sampled_frames_idx[start + k].

        Recovered by matching ``T_world_to_final @ pi3_pose`` to the annotation
        camera poses (final frame); falls back to the ag_pi3.py rule
        (window = [first annotated .. last annotated] in sampled order)."""
        if self._pi3_start is not None:
            return self._pi3_start
        sfi = self.sampled_frames_idx
        S = self.pi3_num_frames
        T = self.T_world_to_final
        cp = self.camera_poses_final
        start = None
        if cp is not None and len(self.camera_frame_keys) > 0:
            pf = T[None] @ np.asarray(self.pi3["camera_poses"], dtype=np.float64)
            votes: Dict[int, int] = {}
            for i, key in enumerate(self.camera_frame_keys):
                fn = int(Path(key).stem)
                if fn not in sfi:
                    continue
                d = np.abs(pf - cp[i][None]).reshape(S, -1).max(1)
                j = int(d.argmin())
                if d[j] < 1e-3:
                    st = sfi.index(fn) - j
                    votes[st] = votes.get(st, 0) + 1
            if votes:
                start = max(votes.items(), key=lambda kv: kv[1])[0]
                if len(votes) > 1:
                    logger.warning(f"[{self.video_id}] inconsistent Pi3 offsets {votes}; using {start}")
        if start is None:
            ann = sorted(int(Path(k).stem) for k in self.frame_keys())
            first = [f for f in ann if f in sfi]
            start = sfi.index(first[0]) if first else 0
            logger.warning(f"[{self.video_id}] Pi3 offset from annotated window: start={start}")
        if start < 0 or start + S > len(sfi):
            logger.warning(f"[{self.video_id}] Pi3 window [{start},{start + S}) exceeds "
                           f"sampled_frames_idx ({len(sfi)})")
        self._pi3_start = int(start)
        return self._pi3_start

    def pi3_index_for_frame(self, frame_num: int) -> Optional[int]:
        sfi = self.sampled_frames_idx
        if frame_num not in sfi:
            return None
        k = sfi.index(frame_num) - self.pi3_start
        return k if 0 <= k < self.pi3_num_frames else None

    def pi3_frame_numbers(self) -> List[int]:
        """Video frame number shown by each Pi3 frame."""
        sfi = self.sampled_frames_idx
        st = self.pi3_start
        return [sfi[st + k] if 0 <= st + k < len(sfi) else -1 for k in range(self.pi3_num_frames)]

    def points_final(self, k: int, conf_min: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Pi3 points of frame k in the canonical frame: ((H,W,3) float32, (H,W) bool mask)."""
        pts = np.asarray(self.pi3["points"][k], dtype=np.float32)
        conf = np.asarray(self.pi3["conf"][k])[..., 0]
        mask = np.isfinite(pts).all(-1) & (conf > conf_min) & (np.abs(pts).sum(-1) > 0)
        return apply_transform(self.T_world_to_final, pts).astype(np.float32), mask

    def camera_pose_final_for_pi3(self, k: int) -> np.ndarray:
        return self.T_world_to_final @ np.asarray(self.pi3["camera_poses"][k], dtype=np.float64)

    def image(self, k: int) -> np.ndarray:
        """Pi3 input image k as uint8 (H, W, 3)."""
        im = np.asarray(self.pi3["images"][k])
        return (im * 255.0).clip(0, 255).astype(np.uint8)

    # ---- annotation (annotation-DEPENDENT) ------------------------------
    def frame_keys(self) -> List[str]:
        return sorted(self.annot.get("frames", {}).keys(), key=lambda k: int(Path(k).stem))

    @property
    def frames(self) -> List[FrameAnnot]:
        if self._frames is not None:
            return self._frames
        cp = self.camera_poses_final
        cam_idx = {k: i for i, k in enumerate(self.camera_frame_keys)}
        out: List[FrameAnnot] = []
        for key in self.frame_keys():
            fr = self.annot["frames"][key]
            pi = fr.get("person_info", {}) or {}
            objs: List[ObjectAnnot] = []
            for o in fr.get("object_info_list", []) or []:
                label = o.get("label") or to_short(o.get("class", ""))
                cls = o.get("class") or to_full(label)
                bb = o.get("bbox_2d", o.get("bbox"))
                bb = None if bb is None else np.asarray(bb, dtype=np.float64).reshape(-1)[:4]
                c = o.get("corners_final")
                c = None if c is None else np.asarray(c, dtype=np.float64).reshape(8, 3)
                objs.append(ObjectAnnot(
                    label=label, cls=cls, class_idx=NAME_TO_IDX.get(label, NAME_TO_IDX.get(cls, 0)),
                    bbox_2d=bb, visible=bool(o.get("visible", True)), source=o.get("source", "gt"),
                    attention=list(o.get("attention_relationship", []) or []),
                    spatial=list(o.get("spatial_relationship", []) or []),
                    contacting=list(o.get("contacting_relationship", []) or []),
                    corners_final=c, raw=o))
            pbb = pi.get("bbox_2d", pi.get("person_bbox"))
            pbb = None if pbb is None else np.asarray(pbb, dtype=np.float64).reshape(-1)[:4]
            pc = pi.get("corners_final")
            pc = None if pc is None else np.asarray(pc, dtype=np.float64).reshape(8, 3)
            fn = int(Path(key).stem)
            out.append(FrameAnnot(
                key=key, file=Path(key).name, frame_num=fn,
                person_bbox_2d=pbb, person_corners_final=pc, objects=objs,
                camera_pose_final=(cp[cam_idx[key]] if (cp is not None and key in cam_idx) else None),
                pi3_index=self.pi3_index_for_frame(fn) if self.pi3_path.exists() else None,
            ))
        self._frames = out
        return out

    def frame(self, frame_file: str) -> Optional[FrameAnnot]:
        for f in self.frames:
            if f.file == frame_file or f.key == frame_file:
                return f
        return None

    def video_objects(self) -> List[str]:
        """Sorted short labels of every object annotated anywhere in the video."""
        return sorted({o.label for f in self.frames for o in f.objects})

    # ---- legacy processor view -----------------------------------------
    def to_processor_video_data(self) -> Dict[str, Any]:
        """The ``frames_final`` structure the vendored runners consume
        (``get_final_data_lite``): ``{"bbox_frames": {frame_file: {"objects": [...]}}}``."""
        bbox_frames: Dict[str, Any] = {}
        for f in self.frames:
            bbox_frames[f.file] = {
                "person_bbox": None if f.person_bbox_2d is None else f.person_bbox_2d.tolist(),
                "objects": [{
                    "label": o.label, "class": o.cls,
                    "bbox": None if o.bbox_2d is None else o.bbox_2d.tolist(),
                    "visible": o.visible, "source": o.source,
                } for o in f.objects],
            }
        return {"video_id": self.vid_mp4, "bbox_frames": bbox_frames,
                "frame_stems": [f.file for f in self.frames]}

    # ---- geometry conveniences -----------------------------------------
    def obb_params(self, corners_final: np.ndarray):
        """(center, size, yaw) of a floor-parallel OBB in the canonical frame."""
        return corners_to_obb(corners_final)

    def bbox_to_pi3(self, bbox_xyxy) -> np.ndarray:
        sx, sy = self.bbox_scale
        return scale_bbox(bbox_xyxy, sx, sy)


# ---------------------------------------------------------------------------
# Test-set index
# ---------------------------------------------------------------------------

class WorldBBoxTestSet:
    """The locked worldbbox test split: ``video_ids`` (1,511 stems) and per-video loaders."""

    def __init__(self, cfg: Optional[dict] = None, annotations_dir: Optional[str] = None,
                 split_file: Optional[str] = None):
        wb = _wb_cfg(cfg)
        self.wb = wb
        self.annotations_dir = Path(annotations_dir or wb["annotations_test"])
        self.split_file = Path(split_file or wb["split_file"])
        if self.split_file.exists():
            with open(self.split_file, "r", encoding="utf-8") as f:
                ids = [Path(l.strip()).stem for l in f if l.strip()]
        else:
            logger.warning(f"split file {self.split_file} missing; using every PKL in {self.annotations_dir}")
            ids = sorted(p.name.split(".")[0] for p in self.annotations_dir.glob("*.pkl"))
        self.video_ids: List[str] = sorted(set(ids))
        missing = [v for v in self.video_ids if not self._annot_path(v).exists()]
        if missing:
            logger.warning(f"{len(missing)} split videos have no annotation PKL (e.g. {missing[:3]})")
        self.annotation_version_name = wb.get("annotation_version_name", "test_worldbbox")
        self.manifest = wb.get("manifest", "/data3/rohith/ag/cache/manifest.json")

    def _annot_path(self, video_id: str) -> Path:
        stem = Path(video_id).stem
        p = self.annotations_dir / f"{stem}.mp4.pkl"
        return p if p.exists() else self.annotations_dir / f"{stem}.pkl"

    def __len__(self) -> int:
        return len(self.video_ids)

    def __contains__(self, video_id: str) -> bool:
        return Path(video_id).stem in set(self.video_ids)

    def load(self, video_id: str) -> WorldBBoxVideo:
        stem = Path(video_id).stem
        p = self._annot_path(stem)
        if not p.exists():
            raise FileNotFoundError(f"worldbbox annotation missing for {stem}: {p}")
        return WorldBBoxVideo(stem, p, self.wb)

    def annotation_version(self) -> Optional[str]:
        try:
            from tools.cache_manifest import current_annotation_version
            return current_annotation_version(self.annotation_version_name, manifest=self.manifest)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Drop-in for core.ag_data.AgDataBBAnnotations
# ---------------------------------------------------------------------------

class WorldBBoxAgData:
    """Same interface the vendored runners use (``get_final_data_lite(video_id)``),
    backed by the worldbbox annotation PKLs instead of
    ``world_annotations/bbox_annotations_3d_obb_final``."""

    def __init__(self, ag_root_directory: str = None, cfg: Optional[dict] = None):
        self.testset = WorldBBoxTestSet(cfg)
        self.ag_root_directory = Path(ag_root_directory or self.testset.wb["ag_root"])
        self._split = set(self.testset.video_ids)

    def get_final_data_lite(self, video_id: str) -> Dict[str, Any]:
        stem = Path(video_id).stem
        if stem not in self._split:
            raise FileNotFoundError(f"{stem} is not in the worldbbox test split")
        return self.testset.load(stem).to_processor_video_data()

    def get_video(self, video_id: str) -> WorldBBoxVideo:
        return self.testset.load(Path(video_id).stem)


def make_ag_data(ag_root_directory: str, cfg: Optional[dict] = None):
    """Pick the annotation source from ``inference.annotation_source``
    (``worldbbox`` (default) | ``legacy``)."""
    if cfg is None:
        from lib.mllm.core.config_loader import load_config
        cfg = load_config()
    src = (cfg.get("inference", {}) or {}).get("annotation_source", "worldbbox")
    if src == "legacy":
        from lib.mllm.core.ag_data import AgDataBBAnnotations
        logger.info("annotation source: legacy bbox_annotations_3d_obb_final")
        return AgDataBBAnnotations(ag_root_directory=ag_root_directory, cfg=cfg)
    logger.info("annotation source: worldbbox test PKLs")
    return WorldBBoxAgData(ag_root_directory, cfg)
