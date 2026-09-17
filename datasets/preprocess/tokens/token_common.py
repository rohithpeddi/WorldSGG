"""
Shared helpers for the frozen-backbone token caches (WorldFormer C1a / C1b).

Cache layout (all under ``/data3/rohith/ag/cache/tokens``):

    <stream>/<split>/<video>.npz      stream in {dinov3l, pi3}; split in {train, test}
    <stream>/<split>/done_shard<k>of<n>.txt
    <stream>/<split>/status_shard<k>of<n>.json

Video lists
-----------
* ``test``  -> the locked worldbbox list ``/data3/rohith/ag/splits/test_worldbbox_1511.txt``
* ``train`` -> videos that have BOTH a train feature PKL
  (``features/roi_features/predcls/dinov3l/train``) and a train annotation PKL
  (``world4d_rel_annotations/train``) — 7,516 videos.

Frames
------
Grids are cached for every frame in ``frames_annotated/<video>.mp4`` (the
annotation-independent superset of what any feature/annotation PKL uses).

Tier-1 (ROI-pooled) features are cached for the boxes of the EXISTING feature
PKLs (``features/roi_features/{predcls,sgdet}/dinov3l/<split>``): those boxes
come from raw AG GT + GDino fill (predcls) / the DINOv3 detector (sgdet), i.e.
they do not depend on the world4d annotation revision.  Union boxes for the
person-object pairs are pooled the same way.
"""
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torchvision.ops import roi_align

AG_ROOT = Path(os.environ.get("AG_ROOT", "/data/rohith/ag"))
DATA3_ROOT = Path(os.environ.get("AG_DATA3_ROOT", "/data3/rohith/ag"))
TOKEN_CACHE_ROOT = DATA3_ROOT / "cache" / "tokens"
TEST_SPLIT_FILE = DATA3_ROOT / "splits" / "test_worldbbox_1511.txt"

FEATURE_BACKBONE_DIR = "dinov3l"
MODES = ("predcls", "sgdet")

PIXEL_LIMIT = 255000
PI3_PATCH = 14


def compute_target_size(orig_w: int, orig_h: int,
                        pixel_limit: int = PIXEL_LIMIT, patch: int = PI3_PATCH) -> Tuple[int, int]:
    """Pi3-space resize (identical to pi3.utils.basic.load_images_as_tensor and
    extract_roi_features_base._compute_target_size)."""
    import math
    scale = math.sqrt(pixel_limit / (orig_w * orig_h)) if orig_w * orig_h > 0 else 1
    w_t, h_t = orig_w * scale, orig_h * scale
    k, m = round(w_t / patch), round(h_t / patch)
    while (k * patch) * (m * patch) > pixel_limit:
        if k / m > w_t / h_t:
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch, max(1, m) * patch


# ---------------------------------------------------------------------------
# Video / frame lists
# ---------------------------------------------------------------------------

def feature_split_dir(mode: str, split: str) -> Path:
    return AG_ROOT / "features" / "roi_features" / mode / FEATURE_BACKBONE_DIR / split


def list_videos(split: str) -> List[str]:
    if split == "test":
        vids = [l.strip() for l in TEST_SPLIT_FILE.read_text().splitlines() if l.strip()]
        return sorted(vids)
    if split == "train":
        feat = {p.stem for p in feature_split_dir("predcls", "train").glob("*.pkl")}
        annot = {p.name[:-8] for p in (AG_ROOT / "world4d_rel_annotations" / "train").glob("*.mp4.pkl")}
        return sorted(feat & annot)
    raise ValueError(split)


def shard_videos(videos: Sequence[str], shard: int, n_shards: int) -> List[str]:
    """Deterministic parity sharding over the sorted list (shard 0 = even indices)."""
    return list(videos[shard::n_shards])


def annotated_frames(video: str) -> List[str]:
    d = AG_ROOT / "frames_annotated" / f"{video}.mp4"
    return sorted(f for f in os.listdir(d) if f.endswith(".png"))


def frame_path(video: str, frame_file: str) -> Path:
    return AG_ROOT / "frames" / f"{video}.mp4" / frame_file


# ---------------------------------------------------------------------------
# Boxes from the existing feature PKLs (Tier-1 targets)
# ---------------------------------------------------------------------------

def _union_boxes(boxes: np.ndarray, labels: List[str], label_ids: List[int],
                 pair_indices: List[Tuple[int, int]]) -> np.ndarray:
    """Re-derive the union boxes in the exact order ``pair_indices`` was built
    (extract_roi_features_base._extract_roi_and_union_features): for every
    person index, for every non-person index."""
    person_idx = [i for i, l in enumerate(labels) if l == "person"]
    object_idx = [i for i, l in enumerate(labels) if l != "person"]
    out = []
    for p in person_idx:
        for o in object_idx:
            out.append([min(boxes[p, 0], boxes[o, 0]), min(boxes[p, 1], boxes[o, 1]),
                        max(boxes[p, 2], boxes[o, 2]), max(boxes[p, 3], boxes[o, 3])])
    out = np.asarray(out, dtype=np.float32).reshape(-1, 4)
    if len(out) != len(pair_indices):
        raise RuntimeError(f"union-box count {len(out)} != pair_indices {len(pair_indices)}")
    return out


def load_feature_boxes(mode: str, split: str, video: str) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """frame_file -> {"boxes": (N,4) f32 Pi3-space, "union_boxes": (P,4) f32, "target_size": (W,H)}"""
    p = feature_split_dir(mode, split) / f"{video}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        d = pickle.load(f)
    out = {}
    for fr, fd in d["frames"].items():
        boxes = np.asarray(fd["bboxes_xyxy"], dtype=np.float32).reshape(-1, 4)
        pairs = fd.get("pair_indices", [])
        if len(pairs) > 0 and "union_features" in fd:
            ub = _union_boxes(boxes, list(fd["labels"]), list(fd["label_ids"]), pairs)
        else:
            ub = np.zeros((0, 4), dtype=np.float32)
        out[fr] = {"boxes": boxes, "union_boxes": ub, "target_size": tuple(fd["target_size"])}
    return out


# ---------------------------------------------------------------------------
# ROI pooling over a token grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def roi_pool_grid(grid_chw: torch.Tensor, boxes_xyxy: np.ndarray, spatial_scale: float,
                  output_size: int = 7) -> torch.Tensor:
    """grid_chw: (C, Hg, Wg) on device; boxes in image pixels -> (N, C) mean of a
    7x7 roi_align (aligned=True, sampling_ratio=2)."""
    if len(boxes_xyxy) == 0:
        return torch.zeros((0, grid_chw.shape[0]), dtype=torch.float16, device=grid_chw.device)
    b = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=grid_chw.device)
    rois = torch.cat([torch.zeros((b.shape[0], 1), device=b.device), b], dim=1)
    pooled = roi_align(grid_chw[None].float(), rois, output_size=(output_size, output_size),
                       spatial_scale=spatial_scale, sampling_ratio=2, aligned=True)
    return pooled.mean(dim=(2, 3)).to(torch.float16)


def pool_tier1(grids_by_layer: Dict[str, torch.Tensor], frames: List[str],
               boxes_by_frame: Optional[Dict[str, Dict[str, np.ndarray]]], spatial_scale: float,
               prefix: str) -> Dict[str, np.ndarray]:
    """Pool every layer grid over the feature-PKL boxes of one mode.

    grids_by_layer: layer_name -> (T, C, Hg, Wg) tensor (frames order == ``frames``).
    Returns flat npz arrays: ``t1_<prefix>_frames``, ``t1_<prefix>_counts``,
    ``t1_<prefix>_boxes``, ``t1_<prefix>_<layer>`` and the ``t1u_`` union analogues.
    """
    out: Dict[str, np.ndarray] = {}
    if boxes_by_frame is None:
        return out
    idx = {f: i for i, f in enumerate(frames)}
    frs = [f for f in sorted(boxes_by_frame) if f in idx]
    counts = np.array([len(boxes_by_frame[f]["boxes"]) for f in frs], dtype=np.int32)
    ucounts = np.array([len(boxes_by_frame[f]["union_boxes"]) for f in frs], dtype=np.int32)
    out[f"t1_{prefix}_frames"] = np.array(frs)
    out[f"t1_{prefix}_counts"] = counts
    out[f"t1u_{prefix}_counts"] = ucounts
    out[f"t1_{prefix}_boxes"] = (np.concatenate([boxes_by_frame[f]["boxes"] for f in frs])
                                 if frs else np.zeros((0, 4), np.float32))
    out[f"t1u_{prefix}_boxes"] = (np.concatenate([boxes_by_frame[f]["union_boxes"] for f in frs])
                                  if frs else np.zeros((0, 4), np.float32))
    for lname, g in grids_by_layer.items():
        feats, ufeats = [], []
        for f in frs:
            gi = g[idx[f]]
            feats.append(roi_pool_grid(gi, boxes_by_frame[f]["boxes"], spatial_scale).cpu().numpy())
            ufeats.append(roi_pool_grid(gi, boxes_by_frame[f]["union_boxes"], spatial_scale).cpu().numpy())
        C = g.shape[1]
        out[f"t1_{prefix}_{lname}"] = np.concatenate(feats) if feats else np.zeros((0, C), np.float16)
        out[f"t1u_{prefix}_{lname}"] = np.concatenate(ufeats) if ufeats else np.zeros((0, C), np.float16)
    return out


# ---------------------------------------------------------------------------
# Atomic outputs, done lists, status
# ---------------------------------------------------------------------------

def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class DoneList:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if path.exists():
            self.done = {l.strip() for l in path.read_text().splitlines() if l.strip()}

    def __contains__(self, v: str) -> bool:
        return v in self.done

    def add(self, v: str) -> None:
        self.done.add(v)
        with open(self.path, "a") as f:
            f.write(v + "\n")


class Status:
    def __init__(self, path: Path, **static):
        self.path = path
        self.static = static
        self.t0 = time.time()

    def write(self, **kw) -> None:
        d = dict(self.static)
        d.update(kw)
        d["elapsed_s"] = round(time.time() - self.t0, 1)
        d["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, self.path)


def out_path(stream: str, split: str, video: str) -> Path:
    return TOKEN_CACHE_ROOT / stream / split / f"{video}.npz"


def video_is_cached(stream: str, split: str, video: str) -> bool:
    return out_path(stream, split, video).exists()


def add_common_args(ap):
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--videos", nargs="*", default=None, help="explicit video ids (validation)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_tier2", action="store_true", help="skip full grids (Tier-1 only)")
    return ap


def resolve_videos(args) -> List[str]:
    if args.videos:
        return list(args.videos)
    vids = shard_videos(list_videos(args.split), args.shard, args.n_shards)
    if args.limit:
        vids = vids[: args.limit]
    return vids
