#!/usr/bin/env python3
"""
C1c — derive ROI-feature PKLs from the token caches (CPU, minutes per split).

Takes the EXISTING feature PKL of a video (object list, Pi3-space boxes, labels,
sources, pair_indices, sgdet boxes_3d / assigned_labels / scores — everything but
the appearance vectors) and swaps ``roi_features`` / ``union_features`` for the
Tier-1 token features cached by ``dinov3_tokens.py`` / ``pi3_tokens.py``:

    dinov3_tok : concat of DINOv3 layers  (default L16 | L20 | L24n)  -> 3072-d
    pi3_tok    : concat of Pi3 layers     (default F4  | F14 | G14)   -> 3072-d
    fused      : [dinov3_tok | pi3_tok]                                -> 6144-d

Output (exact ``extract_roi_features_{predcls,sgdet}.py`` schema + provenance):
    /data3/rohith/ag/cache/roi_derived/<mode>/<stream>/{train,test_worldbbox}/<video>.pkl
and symlinks so ``WorldAG(feature_model=<stream>)`` finds them:
    /data/rohith/ag/features/roi_features/<mode>/<stream>/train          -> .../train
    /data/rohith/ag/features/roi_features/<mode>/<stream>/test           -> .../test_worldbbox
    /data/rohith/ag/features/roi_features/<mode>/<stream>/test_worldbbox -> .../test_worldbbox

Every frame's boxes are asserted equal to the boxes the tokens were pooled on.
Frames whose tokens are missing (e.g. not in the Pi3 clip) are dropped and counted.

Usage:
    python datasets/preprocess/tokens/derive_roi_features.py --split test --mode predcls --stream fused
    python datasets/preprocess/tokens/derive_roi_features.py --split train --mode sgdet --stream all --workers 16
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.preprocess.tokens.token_common import (  # noqa: E402
    AG_ROOT, DATA3_ROOT, TOKEN_CACHE_ROOT, _NumpyCompatUnpickler, feature_split_dir, list_videos,
)

DERIVED_ROOT = DATA3_ROOT / "cache" / "roi_derived"
STREAMS = ("dinov3_tok", "pi3_tok", "fused")
DEFAULT_LAYERS = {"dinov3l": ["L16", "L20", "L24n"], "pi3": ["F4", "F14", "G14"]}
STREAM_SOURCES = {"dinov3_tok": ["dinov3l"], "pi3_tok": ["pi3"], "fused": ["dinov3l", "pi3"]}


def split_out_name(split: str) -> str:
    return "test_worldbbox" if split == "test" else "train"


def _load_pkl(p: Path):
    with open(p, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def _tier1_index(z, mode: str):
    """frame -> (start, count, ustart, ucount) into the flat Tier-1 arrays."""
    key = f"t1_{mode}_frames"
    if key not in z.files:
        return {}
    frames = z[key].tolist()
    counts = z[f"t1_{mode}_counts"]
    ucounts = z[f"t1u_{mode}_counts"]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if len(counts) else np.zeros(0, int)
    ustarts = np.concatenate([[0], np.cumsum(ucounts)[:-1]]) if len(ucounts) else np.zeros(0, int)
    return {f: (int(starts[i]), int(counts[i]), int(ustarts[i]), int(ucounts[i]))
            for i, f in enumerate(frames)}


def derive_video(video: str, split: str, mode: str, stream: str, layers: Dict[str, List[str]],
                 overwrite: bool = False) -> Dict[str, int]:
    out = DERIVED_ROOT / mode / stream / split_out_name(split) / f"{video}.pkl"
    if out.exists() and not overwrite:
        return {"skipped": 1}
    src = feature_split_dir(mode, split) / f"{video}.pkl"
    if not src.exists():
        return {"no_feature_pkl": 1}
    d = _load_pkl(src)
    sources = STREAM_SOURCES[stream]
    zs, idxs = {}, {}
    for s in sources:
        p = TOKEN_CACHE_ROOT / s / split / f"{video}.npz"
        if not p.exists():
            return {"no_tokens": 1}
        zs[s] = np.load(p)
        idxs[s] = _tier1_index(zs[s], mode)
    # Tier-1 arrays are read once per layer (npz members are lazily decompressed)
    t1 = {s: {L: zs[s][f"t1_{mode}_{L}"] for L in layers[s]} for s in sources}
    t1u = {s: {L: zs[s][f"t1u_{mode}_{L}"] for L in layers[s]} for s in sources}
    t1b = {s: zs[s][f"t1_{mode}_boxes"] for s in sources}

    frames_out, dropped = {}, 0
    for fr in sorted(d["frames"]):
        fd = d["frames"][fr]
        n = int(np.asarray(fd["bboxes_xyxy"]).reshape(-1, 4).shape[0])
        if any(fr not in idxs[s] for s in sources):
            dropped += 1
            continue
        feats, ufeats = [], []
        ok = True
        for s in sources:
            st, cnt, ust, ucnt = idxs[s][fr]
            if cnt != n:
                ok = False
                break
            if not np.allclose(t1b[s][st:st + cnt], np.asarray(fd["bboxes_xyxy"], np.float32).reshape(-1, 4),
                               atol=1e-3):
                ok = False
                break
            feats.append(np.concatenate([t1[s][L][st:st + cnt] for L in layers[s]], axis=1))
            npairs = len(fd.get("pair_indices", []))
            if "union_features" in fd and npairs > 0:
                if ucnt != npairs:
                    ok = False
                    break
                ufeats.append(np.concatenate([t1u[s][L][ust:ust + ucnt] for L in layers[s]], axis=1))
        if not ok:
            dropped += 1
            continue
        new = dict(fd)
        new["roi_features"] = np.concatenate(feats, axis=1).astype(np.float16)
        if ufeats:
            new["union_features"] = np.concatenate(ufeats, axis=1).astype(np.float16)
        elif "union_features" in new:
            del new["union_features"]
        frames_out[fr] = new
    if not frames_out:
        return {"empty": 1, "dropped_frames": dropped}

    D = next(iter(frames_out.values()))["roi_features"].shape[1]
    res = {k: v for k, v in d.items() if k != "frames"}
    res.update({
        "model": stream, "feature_dim": int(D), "frames": frames_out,
        "token_streams": [(s, layers[s], 1024 * len(layers[s])) for s in sources],
        "derived_from": str(src),
        "token_caches": {s: str(TOKEN_CACHE_ROOT / s / split / f"{video}.npz") for s in sources},
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    return {"ok": 1, "frames": len(frames_out), "dropped_frames": dropped}


def make_symlinks(mode: str, stream: str) -> None:
    base = AG_ROOT / "features" / "roi_features" / mode / stream
    base.mkdir(parents=True, exist_ok=True)
    for name, target in (("train", DERIVED_ROOT / mode / stream / "train"),
                         ("test", DERIVED_ROOT / mode / stream / "test_worldbbox"),
                         ("test_worldbbox", DERIVED_ROOT / mode / stream / "test_worldbbox")):
        target.mkdir(parents=True, exist_ok=True)
        link = base / name
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == str(target):
                continue
            if link.is_symlink():
                link.unlink()
            else:
                raise RuntimeError(f"{link} exists and is not a symlink")
        os.symlink(str(target), str(link))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--mode", required=True, choices=["predcls", "sgdet", "all"])
    ap.add_argument("--stream", required=True, choices=list(STREAMS) + ["all"])
    ap.add_argument("--dinov3_layers", default=",".join(DEFAULT_LAYERS["dinov3l"]))
    ap.add_argument("--pi3_layers", default=",".join(DEFAULT_LAYERS["pi3"]))
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_symlinks", action="store_true")
    ap.add_argument("--register", action="store_true", help="register finished splits in the cache manifest")
    args = ap.parse_args()

    layers = {"dinov3l": args.dinov3_layers.split(","), "pi3": args.pi3_layers.split(",")}
    modes = ["predcls", "sgdet"] if args.mode == "all" else [args.mode]
    streams = list(STREAMS) if args.stream == "all" else [args.stream]
    videos = args.videos or list_videos(args.split)
    if args.limit:
        videos = videos[: args.limit]

    for mode in modes:
        for stream in streams:
            t0 = time.time()
            agg: Dict[str, int] = {}
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(derive_video, v, args.split, mode, stream, layers, args.overwrite)
                        for v in videos]
                for i, fu in enumerate(as_completed(futs)):
                    for k, val in fu.result().items():
                        agg[k] = agg.get(k, 0) + val
                    if (i + 1) % 500 == 0:
                        print(f"[{mode}/{stream}/{args.split}] {i + 1}/{len(videos)} {agg}", flush=True)
            print(f"[{mode}/{stream}/{args.split}] DONE {len(videos)} videos in {time.time() - t0:.0f}s: {agg}",
                  flush=True)
            if not args.no_symlinks:
                make_symlinks(mode, stream)
            if args.register:
                from tools.cache_manifest import current_annotation_version, register
                out_dir = DERIVED_ROOT / mode / stream / split_out_name(args.split)
                n_out = len(list(out_dir.glob("*.pkl")))
                register(
                    f"roi_derived/{mode}/{stream}/{split_out_name(args.split)}", str(out_dir),
                    inputs=[f"tokens/{s}/{args.split}" for s in STREAM_SOURCES[stream]]
                    + [str(feature_split_dir(mode, args.split))],
                    annotation_version=(current_annotation_version("test_worldbbox")
                                        if args.split == "test" else None),
                    note=json.dumps({"layers": {s: layers[s] for s in STREAM_SOURCES[stream]},
                                     "n_videos": n_out, "stats": agg}),
                )


if __name__ == "__main__":
    main()
