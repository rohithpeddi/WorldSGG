#!/usr/bin/env python3
"""
WorldWise++ training cache: ONE frozen layer per stream, PCA-256, training frames only.

Source (Tier-2 grids, spinning disk, ~1.1 TB + 1.5 TB):
    /data3/rohith/ag/cache/tokens/dinov3l/<split>/<video>.npz   grid_L24n (T,Hd,Wd,1024) f16 on the /16-padded image
    /data3/rohith/ag/cache/tokens/pi3/<split>/<video>.npz       grid_G14  (T,Hp,Wp,1024) f16 on the Pi3 image

Output (NVMe, ``PP_GRID_ROOT`` = /code/rohith/ag/cache/pp_grids; symlinked from
/data3/rohith/ag/cache/pp_grids so that path always resolves):

    <root>/<split>/<video>.npz          uncompressed np.savez
        frames       (T,) <U16     keys of the predcls dinov3l feature PKL ``frames`` dict, sorted
                                   (= what dataloader/world_ag_dataset.py::WorldAG iterates)
        dino         (T,Hp,Wp,256) f16   DINOv3 grid_L24n cropped to the unpadded region and bilinearly
                                   resampled onto the Pi3 grid exactly as
                                   c2_worldformer.model.TokenGridFusion.forward does, then PCA-256
        pi3          (T,Hp,Wp,256) f16   Pi3 grid_G14, PCA-256
        grid_hw      (2,) int32    (Hp, Wp)
        target_size  (2,) int32    (H, W) of the Pi3-space image  (NOTE: normalised to H,W; the token
                                   npz files and the feature PKLs store it as (W,H))
        missing      (M,) <U16     PKL frames without a cached grid row in either stream (their rows
                                   in ``dino``/``pi3`` are zero-filled; expected M == 0)
    <root>/pca_dinov3_L24n.npz, <root>/pca_pi3_G14.npz
        mean (1024,) f32, components (256,1024) f32 (rows = principal axes; apply as
        (x - mean) @ components.T), explained_variance_ratio (256,) f32 (of the FULL 1024-d
        variance), n_fit_tokens, fit_videos, cum_var_256, seed, source_layer
    <root>/status_<split>[_shardKofN].json     progress
    <root>/verify_<split>.json                 integrity report
    <root>/README.md                           written by ``register``

Usage (CPU only; GPU 0 is busy; 3-4 workers max, the HDD is the bottleneck):
    python datasets/preprocess/tokens/build_pp_grid_cache.py fit-pca [--n-videos 150 --n-tokens 500000]
    python datasets/preprocess/tokens/build_pp_grid_cache.py build  --split test  [--shard k --n-shards n] [--workers 4]
    python datasets/preprocess/tokens/build_pp_grid_cache.py verify --split test
    python datasets/preprocess/tokens/build_pp_grid_cache.py register --split test
"""
import os

# keep BLAS/torch from oversubscribing 64 cores across 4 worker processes (must precede numpy import;
# with the "spawn" start method the children re-execute this module top so they inherit it too)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import random  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.preprocess.tokens.token_common import (  # noqa: E402
    DATA3_ROOT, PI3_PATCH, TOKEN_CACHE_ROOT, Status, feature_split_dir, list_videos, shard_videos,
)

PP_ROOT = Path(os.environ.get("PP_GRID_ROOT", "/code/rohith/ag/cache/pp_grids"))
PP_LINK = DATA3_ROOT / "cache" / "pp_grids"          # always resolves to the actual root
DINO_LAYER = "grid_L24n"
PI3_LAYER = "grid_G14"
D_OUT = 256
FRAME_DT = "<U16"
PCA_FILES = {"dino": "pca_dinov3_L24n.npz", "pi3": "pca_pi3_G14.npz"}
MANIFEST_KEY = "pp_grids/{split}"
MANIFEST_INPUTS = [
    "cache/tokens/dinov3l/<split>/<video>.npz (grid_L24n)",
    "cache/tokens/pi3/<split>/<video>.npz (grid_G14)",
    "features/roi_features/predcls/dinov3l/<split>/<video>.pkl (frame list only)",
    "pca_dinov3_L24n.npz / pca_pi3_G14.npz (fit on 150 seeded train videos)",
]


# ---------------------------------------------------------------------------
# Paths / small helpers
# ---------------------------------------------------------------------------

def out_dir(split: str) -> Path:
    return PP_ROOT / split


def out_path(split: str, video: str) -> Path:
    return out_dir(split) / f"{video}.npz"


def src_path(stream: str, split: str, video: str) -> Path:
    return TOKEN_CACHE_ROOT / stream / split / f"{video}.npz"


def ensure_root() -> Path:
    PP_ROOT.mkdir(parents=True, exist_ok=True)
    if PP_LINK.resolve() != PP_ROOT.resolve():
        if PP_LINK.is_symlink() or PP_LINK.exists():
            if PP_LINK.is_symlink():
                PP_LINK.unlink()
            else:
                raise RuntimeError(f"{PP_LINK} exists and is not a symlink; refusing to replace it")
        PP_LINK.parent.mkdir(parents=True, exist_ok=True)
        PP_LINK.symlink_to(PP_ROOT, target_is_directory=True)
    return PP_ROOT


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def pkl_frames(split: str, video: str) -> List[str]:
    """Sorted keys of the predcls dinov3l feature PKL ``frames`` dict."""
    p = feature_split_dir("predcls", split) / f"{video}.pkl"
    with open(p, "rb") as f:
        d = _NumpyCompatUnpickler(f).load()
    return sorted(d["frames"].keys())


def hw_from_wh(target_size: np.ndarray, grid_hw: Tuple[int, int], patch: int, what: str) -> Tuple[int, int]:
    """token npz ``target_size`` is stored as [W,H]; return (H,W) and check it against the grid."""
    a, b = int(target_size[0]), int(target_size[1])
    Hg, Wg = int(grid_hw[0]), int(grid_hw[1])
    if (b, a) == (Hg * patch, Wg * patch) or (Hg == Wg and a == b):
        return b, a                               # stored (W,H)
    if (a, b) == (Hg * patch, Wg * patch):
        return a, b                               # stored (H,W) (not expected)
    raise RuntimeError(f"{what}: target_size {target_size.tolist()} inconsistent with grid_hw {grid_hw}")


def read_member_rows(path: Path, member: str, rows: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, int]:
    """Read an uncompressed ``.npy`` member of an npz with ONE large sequential read of the
    contiguous row block [min(rows), max(rows)] (whole member if rows is None).

    ``np.load(npz)[member]`` streams the member in 256 KB chunks through zipfile, which
    interleaves badly with the other readers of the spinning disk (measured ~2x slower than
    a single read).  Returns (block, first_row); block[i] == member[first_row + i]."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(member + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path}:{member} is compressed")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        if h[:4] != b"PK\x03\x04":
            raise ValueError(f"{path}:{member} bad local header")
        n_name, n_extra = struct.unpack("<HH", h[26:30])
        f.seek(info.header_offset + 30 + n_name + n_extra)
        ver = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        if fortran or len(shape) < 1:
            raise ValueError(f"{path}:{member} unexpected layout {shape} fortran={fortran}")
        arr_off = f.tell()
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        r0, r1 = (0, shape[0]) if rows is None else ((min(rows), max(rows) + 1) if len(rows) else (0, 0))
        buf = np.empty((r1 - r0,) + tuple(shape[1:]), dtype=dtype)
        if buf.nbytes:
            try:
                os.posix_fadvise(f.fileno(), arr_off + r0 * row_bytes, buf.nbytes, os.POSIX_FADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
            f.seek(arr_off + r0 * row_bytes)
            mv = memoryview(buf).cast("B")
            got = 0
            while got < len(mv):
                n = f.readinto(mv[got:])
                if not n:
                    raise IOError(f"{path}:{member} short read {got}/{len(mv)}")
                got += n
    return buf, r0


def load_stream(stream: str, split: str, video: str, layer: str, want: Optional[Sequence[str]] = None):
    """Read ONE grid member (+ metadata) of a token npz.  Returns dict with
    frames (list), grid (T',Hg,Wg,1024) f16 for the wanted frames (all if want is None),
    grid_hw (Hg,Wg), target_wh, padded_wh (dino only)."""
    path = src_path(stream, split, video)
    with np.load(path) as z:
        frames = [str(f) for f in z["frames"]]
        grid_hw = tuple(int(x) for x in z["grid_hw"])
        target_wh = z["target_size"].astype(np.int64)
        padded_wh = z["padded_size"].astype(np.int64) if "padded_size" in z.files else None
    if want is None:
        g, _ = read_member_rows(path, layer)
    else:
        idx = {f: i for i, f in enumerate(frames)}
        rows = [idx[f] for f in want if f in idx]
        frames = [f for f in want if f in idx]
        blk, r0 = read_member_rows(path, layer, rows)
        g = blk[[r - r0 for r in rows]] if rows else blk
    return {"frames": frames, "grid": g, "grid_hw": grid_hw, "target_wh": target_wh, "padded_wh": padded_wh}


@torch.no_grad()
def dino_to_pi3_grid(grid: np.ndarray, dino_padded_hw: Tuple[int, int], image_hw: Tuple[int, int],
                     pi3_hw: Tuple[int, int]) -> torch.Tensor:
    """(T,Hd,Wd,C) f16 on the padded image -> (T,Hp,Wp,C) f32; crop + bilinear exactly like
    TokenGridFusion.forward (which does this on the projected grid; projection is linear and
    bilinear weights sum to 1, so resample-then-PCA == PCA-then-resample)."""
    if grid.shape[0] == 0:
        return torch.zeros((0, pi3_hw[0], pi3_hw[1], grid.shape[-1]), dtype=torch.float32)
    a = torch.from_numpy(grid).float().permute(0, 3, 1, 2)            # (T,C,Hd,Wd)
    Hd, Wd = a.shape[-2:]
    fh = image_hw[0] / dino_padded_hw[0]
    fw = image_hw[1] / dino_padded_hw[1]
    a = a[:, :, : max(1, round(Hd * fh)), : max(1, round(Wd * fw))]
    a = F.interpolate(a, size=pi3_hw, mode="bilinear", align_corners=False)
    return a.permute(0, 2, 3, 1).contiguous()                          # (T,Hp,Wp,C)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class PCA:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=True) as z:
            self.mean = z["mean"].astype(np.float32)
            self.components = z["components"].astype(np.float32)   # (256,1024)
        self.ct = np.ascontiguousarray(self.components.T)            # (1024,256)

    def project(self, x: np.ndarray) -> np.ndarray:
        """(...,1024) -> (...,256) f16, matmul in float32."""
        shp = x.shape[:-1]
        y = (x.reshape(-1, x.shape[-1]).astype(np.float32, copy=False) - self.mean) @ self.ct
        return y.astype(np.float16).reshape(*shp, self.ct.shape[1])


def _fit_worker(job):
    """One video of the PCA fit: returns (video, dino_tokens (n,1024) f32, pi3_tokens (n,1024) f32)."""
    video, quota, seed = job
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    d = load_stream("dinov3l", "train", video, DINO_LAYER)
    p = load_stream("pi3", "train", video, PI3_LAYER)
    H, W = hw_from_wh(p["target_wh"], p["grid_hw"], PI3_PATCH, f"pi3/{video}")
    Hpad, Wpad = int(d["padded_wh"][1]), int(d["padded_wh"][0])
    dg = dino_to_pi3_grid(d["grid"], (Hpad, Wpad), (H, W), p["grid_hw"]).numpy().reshape(-1, 1024)
    pg = p["grid"].reshape(-1, 1024)
    di = np.sort(rng.choice(dg.shape[0], size=min(quota, dg.shape[0]), replace=False))
    pi = np.sort(rng.choice(pg.shape[0], size=min(quota, pg.shape[0]), replace=False))
    return video, dg[di].astype(np.float32), pg[pi].astype(np.float32), len(d["frames"]), len(p["frames"])


def _fit_one(X: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    X = X.astype(np.float64)
    mean = X.mean(0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.maximum(evals[order], 0.0), evecs[:, order]
    evr = evals / max(evals.sum(), 1e-12)
    return {"mean": mean.astype(np.float32), "components": np.ascontiguousarray(evecs[:, :k].T).astype(np.float32),
            "explained_variance_ratio": evr[:k].astype(np.float32), "explained_variance_ratio_full": evr.astype(np.float32),
            "cum_var_256": np.float32(evr[:k].sum())}


def cmd_fit_pca(args):
    root = ensure_root()
    vids = list_videos("train")
    fit_videos = sorted(random.Random(args.seed).sample(vids, args.n_videos))
    quota = int(np.ceil(args.n_tokens / len(fit_videos)))
    print(f"fit-pca: {len(fit_videos)} train videos (seed {args.seed}), ~{quota} tokens/video/stream, "
          f"{args.workers} workers", flush=True)
    t0 = time.time()
    Xd, Xp, nfd, nfp = [], [], 0, 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        jobs = [(v, quota, args.seed * 100003 + i) for i, v in enumerate(fit_videos)]
        for i, (v, dt, pt, nd, npf) in enumerate(pool.imap_unordered(_fit_worker, jobs)):
            Xd.append(dt); Xp.append(pt); nfd += nd; nfp += npf
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"  {i + 1}/{len(jobs)} videos, {el:.0f}s, {(i + 1) / el * 60:.1f} videos/min", flush=True)
    Xd, Xp = np.concatenate(Xd), np.concatenate(Xp)
    print(f"tokens: dino {Xd.shape} pi3 {Xp.shape}; source frames dino={nfd} pi3={nfp}; "
          f"read {time.time() - t0:.0f}s", flush=True)
    for name, X, layer in (("dino", Xd, DINO_LAYER), ("pi3", Xp, PI3_LAYER)):
        t1 = time.time()
        r = _fit_one(X, D_OUT)
        evr = r["explained_variance_ratio_full"]
        cum = np.cumsum(evr)
        print(f"[{name}/{layer}] n={X.shape[0]} cum var @64={cum[63]:.4f} @128={cum[127]:.4f} "
              f"@256={cum[255]:.4f} @512={cum[511]:.4f}  ({time.time() - t1:.1f}s)", flush=True)
        np.savez(root / PCA_FILES[name], n_fit_tokens=np.int64(X.shape[0]), fit_videos=np.array(fit_videos),
                 seed=np.int64(args.seed), source_layer=np.array(layer),
                 source_stream=np.array("dinov3l" if name == "dino" else "pi3"),
                 dino_resampled_to_pi3_grid=np.bool_(name == "dino"), **r)
    print(f"fit-pca done in {time.time() - t0:.0f}s -> {root}", flush=True)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

_PCA: Dict[str, PCA] = {}


def _build_init(root: str):
    torch.set_num_threads(4)
    for k, f in PCA_FILES.items():
        _PCA[k] = PCA(Path(root) / f)


def build_video(split: str, video: str) -> Dict[str, object]:
    frames = pkl_frames(split, video)
    p = load_stream("pi3", split, video, PI3_LAYER, want=frames)
    d = load_stream("dinov3l", split, video, DINO_LAYER, want=frames)
    Hp, Wp = p["grid_hw"]
    H, W = hw_from_wh(p["target_wh"], (Hp, Wp), PI3_PATCH, f"pi3/{video}")
    if tuple(d["target_wh"].tolist()) != tuple(p["target_wh"].tolist()):
        raise RuntimeError(f"target_size mismatch dino {d['target_wh'].tolist()} vs pi3 {p['target_wh'].tolist()}")
    Hpad, Wpad = int(d["padded_wh"][1]), int(d["padded_wh"][0])

    T = len(frames)
    dino = np.zeros((T, Hp, Wp, D_OUT), np.float16)
    pi3 = np.zeros((T, Hp, Wp, D_OUT), np.float16)
    fidx = {f: i for i, f in enumerate(frames)}
    if d["frames"]:
        g = dino_to_pi3_grid(d["grid"], (Hpad, Wpad), (H, W), (Hp, Wp)).numpy()
        dino[[fidx[f] for f in d["frames"]]] = _PCA["dino"].project(g)
    if p["frames"]:
        pi3[[fidx[f] for f in p["frames"]]] = _PCA["pi3"].project(p["grid"])
    missing = sorted((set(frames) - set(d["frames"])) | (set(frames) - set(p["frames"])))

    op = out_path(split, video)
    op.parent.mkdir(parents=True, exist_ok=True)
    tmp = op.with_name(f".{video}.{os.getpid()}.tmp.npz")
    np.savez(tmp, frames=np.array(frames, dtype=FRAME_DT), dino=dino, pi3=pi3,
             grid_hw=np.array([Hp, Wp], np.int32), target_size=np.array([H, W], np.int32),
             missing=np.array(missing, dtype=FRAME_DT))
    os.replace(tmp, op)
    return {"video": video, "frames": T, "missing": len(missing), "bytes": op.stat().st_size}


def _build_worker(job):
    split, video = job
    try:
        r = build_video(split, video)
        r["ok"] = True
        return r
    except Exception as e:  # noqa: BLE001
        return {"video": video, "ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}


def cmd_build(args):
    root = ensure_root()
    for f in PCA_FILES.values():
        if not (root / f).exists():
            sys.exit(f"missing {root / f}: run fit-pca first")
    if getattr(args, "videos", None):
        videos = list(args.videos)                      # explicit ids, e.g. a video outside the split
    else:
        videos = shard_videos(list_videos(args.split), args.shard, args.n_shards)
    if args.limit:
        videos = videos[: args.limit]
    tag = "" if args.n_shards == 1 else f"_shard{args.shard}of{args.n_shards}"
    status = Status(root / f"status_{args.split}{tag}.json", split=args.split, shard=args.shard,
                    n_shards=args.n_shards, n_videos=len(videos), pid=os.getpid(), workers=args.workers)
    todo = [v for v in videos if args.overwrite or not out_path(args.split, v).exists()]
    n_skip = len(videos) - len(todo)
    if out_dir(args.split).exists():
        for stale in out_dir(args.split).glob(".*.tmp.npz"):   # leftovers of a killed run
            stale.unlink()
    print(f"build {args.split}{tag}: {len(videos)} videos, {n_skip} already cached, {len(todo)} to do, "
          f"{args.workers} workers -> {out_dir(args.split)}", flush=True)
    n_ok = n_err = n_frames = n_missing = 0
    nbytes = 0
    t0 = time.time()
    errors = []

    def _status(state):
        el = time.time() - t0
        done_n = n_ok + n_err
        rate = done_n / el * 60 if el > 0 else 0.0
        status.write(processed=n_skip + done_n, ok=n_ok, skipped=n_skip, errors=n_err, frames=n_frames,
                     missing_frames=n_missing, bytes=nbytes, videos_per_min=round(rate, 2),
                     eta_min=round((len(todo) - done_n) / rate, 1) if rate > 0 else None,
                     last_video=last[0], state=state, error_list=errors[-20:])

    last = [None]
    _status("running")
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_build_init, initargs=(str(root),)) as pool:
        for i, r in enumerate(pool.imap_unordered(_build_worker, [(args.split, v) for v in todo])):
            last[0] = r["video"]
            if r["ok"]:
                n_ok += 1; n_frames += r["frames"]; n_missing += r["missing"]; nbytes += r["bytes"]
            else:
                n_err += 1
                errors.append({"video": r["video"], "error": r["error"]})
                print(f"[ERR] {r['video']}: {r['error']}\n{r['tb']}", flush=True)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"[{args.split}{tag}] {i + 1}/{len(todo)} ok={n_ok} err={n_err} frames={n_frames} "
                      f"missing={n_missing} {nbytes / 1e9:.1f}GB {(i + 1) / el * 60:.1f} videos/min", flush=True)
                _status("running")
    _status("finished")
    print(f"[{args.split}{tag}] finished ok={n_ok} skip={n_skip} err={n_err} frames={n_frames} "
          f"missing={n_missing} {nbytes / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def _verify_worker(job):
    split, video = job
    op = out_path(split, video)
    r = {"video": video}
    try:
        if not op.exists():
            r["error"] = "missing npz"
            return r
        frames = pkl_frames(split, video)
        with np.load(op) as z:
            zf = [str(f) for f in z["frames"]]
            Hp, Wp = (int(x) for x in z["grid_hw"])
            H, W = (int(x) for x in z["target_size"])
            dino, pi3, missing = z["dino"], z["pi3"], [str(f) for f in z["missing"]]
        T = len(frames)
        probs = []
        if zf != frames:
            probs.append(f"frames != PKL frames ({len(zf)} vs {T})")
        if dino.shape != (T, Hp, Wp, D_OUT) or dino.dtype != np.float16:
            probs.append(f"dino shape/dtype {dino.shape} {dino.dtype}")
        if pi3.shape != (T, Hp, Wp, D_OUT) or pi3.dtype != np.float16:
            probs.append(f"pi3 shape/dtype {pi3.shape} {pi3.dtype}")
        if (H, W) != (Hp * PI3_PATCH, Wp * PI3_PATCH):
            probs.append(f"target_size {(H, W)} != grid_hw*{PI3_PATCH} {(Hp * PI3_PATCH, Wp * PI3_PATCH)}")
        if not (np.isfinite(dino).all() and np.isfinite(pi3).all()):
            probs.append("non-finite values")
        if not set(missing) <= set(frames):
            probs.append("missing not subset of frames")
        present = [i for i, f in enumerate(frames) if f not in set(missing)]
        if present and (not dino[present].any(axis=(1, 2, 3)).all()
                        or not pi3[present].any(axis=(1, 2, 3)).all()):
            probs.append("all-zero row for a non-missing frame")
        r.update(frames=T, missing=len(missing), grid_hw=[Hp, Wp], bytes=op.stat().st_size)
        if probs:
            r["error"] = "; ".join(probs)
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"
    return r


def cmd_verify(args):
    root = ensure_root()
    videos = list_videos(args.split)
    print(f"verify {args.split}: {len(videos)} videos, {args.workers} workers", flush=True)
    t0 = time.time()
    bad, n_frames, n_missing, nbytes, grids = [], 0, 0, 0, {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_verify_worker, [(args.split, v) for v in videos], chunksize=8)):
            if "error" in r:
                bad.append(r)
            else:
                n_frames += r["frames"]; n_missing += r["missing"]; nbytes += r["bytes"]
                grids[str(r["grid_hw"])] = grids.get(str(r["grid_hw"]), 0) + 1
            if (i + 1) % 200 == 0 or i + 1 == len(videos):
                print(f"  {i + 1}/{len(videos)} bad={len(bad)} frames={n_frames} missing={n_missing} "
                      f"{time.time() - t0:.0f}s", flush=True)
    rep = {"split": args.split, "n_videos": len(videos), "n_bad": len(bad), "frames": n_frames,
           "missing_frames": n_missing, "bytes": nbytes, "gb": round(nbytes / 1e9, 2),
           "grid_hw_hist": grids, "bad": bad[:200], "checked": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pca_files": {k: (root / f).exists() for k, f in PCA_FILES.items()}}
    with open(root / f"verify_{args.split}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "bad"}, indent=2))
    for b in bad[:20]:
        print("  BAD", b["video"], b["error"])
    print("VERIFY", "OK" if not bad else f"FAILED ({len(bad)} bad)", flush=True)
    sys.exit(0 if not bad else 1)


# ---------------------------------------------------------------------------
# register (manifest + README)
# ---------------------------------------------------------------------------

def _dir_size(p: Path) -> Tuple[int, int]:
    n, s = 0, 0
    for f in p.glob("*.npz"):
        n += 1; s += f.stat().st_size
    return n, s


def _readme(root: Path) -> str:
    lines = ["# WorldWise++ PCA-256 grid cache (`pp_grids`)", "",
             f"Actual root: `{root}` (symlink `{PP_LINK}` -> root). Built by",
             "`datasets/preprocess/tokens/build_pp_grid_cache.py`; CPU only.", "",
             "## Per-video `<split>/<video>.npz` (uncompressed `np.savez`)", "",
             "| member | shape / dtype | meaning |", "|---|---|---|",
             "| `frames` | `(T,) <U16` | sorted keys of the predcls dinov3l feature PKL `frames` dict (what `WorldAG` iterates) |",
             "| `dino` | `(T,Hp,Wp,256) float16` | DINOv3-L `grid_L24n` cropped to the unpadded region + bilinear (`align_corners=False`) onto the Pi3 grid exactly as `TokenGridFusion.forward`, then PCA-256 |",
             "| `pi3` | `(T,Hp,Wp,256) float16` | Pi3 `grid_G14`, PCA-256 |",
             "| `grid_hw` | `(2,) int32` | `(Hp, Wp)` — varies with aspect ratio, never hard-code |",
             "| `target_size` | `(2,) int32` | `(H, W)` of the Pi3-space image (= `grid_hw * 14`; NOTE the token npz / feature PKLs store `(W,H)`) |",
             "| `missing` | `(M,) <U16` | PKL frames with no cached grid row in either stream; their rows are zero-filled (expected `M == 0`) |",
             "", "PCA files `pca_dinov3_L24n.npz`, `pca_pi3_G14.npz`: `mean (1024,) f32`, `components (256,1024) f32`",
             "(rows = principal axes; apply as `(x - mean) @ components.T`), `explained_variance_ratio (256,) f32`",
             "(fraction of the full 1024-d variance per axis), `explained_variance_ratio_full (1024,)`, `cum_var_256`,",
             "`n_fit_tokens`, `fit_videos` (150 seeded-random TRAIN videos, seed 0), `source_layer`. Fit on the",
             "resampled-onto-Pi3-grid DINOv3 tokens (i.e. exactly the distribution stored) and raw Pi3 tokens,",
             "~500k tokens per stream, covariance eigendecomposition in float64.", "",
             "## Loading", "", "```python", "import numpy as np",
             "z = np.load('/data3/rohith/ag/cache/pp_grids/train/001YG.npz')",
             "frames, dino, pi3 = z['frames'], z['dino'], z['pi3']     # (T,), (T,Hp,Wp,256) f16, (T,Hp,Wp,256) f16",
             "Hp, Wp = z['grid_hw']; H, W = z['target_size']; missing = z['missing']",
             "```", "", "## PCA variance retained", ""]
    for k, f in PCA_FILES.items():
        p = root / f
        if p.exists():
            with np.load(p, allow_pickle=True) as z:
                full = np.cumsum(z["explained_variance_ratio_full"])
                lines.append(f"- `{f}` ({str(z['source_stream'])}/{str(z['source_layer'])}, n_fit_tokens={int(z['n_fit_tokens'])}): "
                             f"cum. variance @64 = {full[63]:.4f}, @128 = {full[127]:.4f}, **@256 = {full[255]:.4f}**, @512 = {full[511]:.4f}")
    lines += ["", "## Realised sizes / build time", ""]
    for split in ("train", "test"):
        d = out_dir(split)
        if not d.exists():
            continue
        n, s = _dir_size(d)
        st = root / f"status_{split}.json"
        extra = ""
        if st.exists():
            j = json.load(open(st))
            extra = (f"; build: {j.get('ok')} ok / {j.get('errors')} err, {j.get('frames')} frames, "
                     f"{j.get('elapsed_s', 0) / 60:.0f} min at {j.get('videos_per_min')} videos/min, state={j.get('state')}")
        vr = root / f"verify_{split}.json"
        if vr.exists():
            v = json.load(open(vr))
            extra += f"; verify: {v['n_bad']} bad of {v['n_videos']}, {v['frames']} frames, missing={v['missing_frames']}, grid_hw={v['grid_hw_hist']}"
        lines.append(f"- `{split}`: {n} videos, {s / 1e9:.1f} GB{extra}")
    lines += ["", f"Manifest key `{MANIFEST_KEY.format(split='<split>')}` (annotation-independent) in "
              f"`{DATA3_ROOT / 'cache' / 'manifest.json'}`.", ""]
    return "\n".join(lines)


def cmd_register(args):
    from tools.cache_manifest import register
    root = ensure_root()
    videos = list_videos(args.split)
    missing = [v for v in videos if not out_path(args.split, v).exists()]
    n, s = _dir_size(out_dir(args.split))
    print(f"pp_grids/{args.split}: {len(videos) - len(missing)}/{len(videos)} videos cached, {s / 1e9:.1f} GB, "
          f"missing {len(missing)}")
    if missing and not args.force:
        print("  e.g.", missing[:10]); sys.exit(1)
    (root / "README.md").write_text(_readme(root))
    register(MANIFEST_KEY.format(split=args.split), str(out_dir(args.split)), inputs=MANIFEST_INPUTS,
             annotation_version=None,
             note=f"{len(videos) - len(missing)} videos, {s / 1e9:.1f} GB; missing={len(missing)}; "
                  f"one layer per stream (dinov3l L24n resampled to Pi3 grid, pi3 G14), PCA-256 f16; root={root}")
    print("registered; README written")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit-pca")
    f.add_argument("--n-videos", type=int, default=150)
    f.add_argument("--n-tokens", type=int, default=500_000, help="per stream")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--workers", type=int, default=4)
    b = sub.add_parser("build")
    b.add_argument("--split", required=True, choices=["train", "test"])
    b.add_argument("--shard", type=int, default=0)
    b.add_argument("--n-shards", type=int, default=1)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--videos", nargs="*", default=None, help="explicit video ids instead of the split list")
    b.add_argument("--overwrite", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--split", required=True, choices=["train", "test"])
    v.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("register")
    r.add_argument("--split", required=True, choices=["train", "test"])
    r.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"fit-pca": cmd_fit_pca, "build": cmd_build, "verify": cmd_verify, "register": cmd_register}[args.cmd](args)


if __name__ == "__main__":
    main()
