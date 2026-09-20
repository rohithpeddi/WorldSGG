#!/usr/bin/env python3
"""
C1a — DINOv3-L patch-token cache (frozen ``facebook/dinov3-vitl16-pretrain-lvd1689m``).

Preprocessing reproduces the ROI-feature pipeline exactly
(``extract_roi_features_base._load_and_preprocess_frame`` + ``_NoOpRCNNTransform``):
cv2 BGR->RGB, INTER_LINEAR resize to the Pi3-space size (pixel_limit 255000,
multiples of 14 -> 672x378 for AG), ImageNet normalisation, zero-pad
bottom/right to a multiple of the DINOv3 patch (16) -> 672x384 -> 42x24 = 1008
patch tokens, fp16 autocast.

Per-video npz (``/data3/rohith/ag/cache/tokens/dinov3l/<split>/<video>.npz``):
    frames            (T,) str      all frames of frames_annotated/<video>.mp4
    target_size       (2,) [W,H]    Pi3-space size (boxes live here)
    grid_hw           (2,) [Hg,Wg]  token grid on the padded image
    patch             16
    grid_L16, grid_L24n   (T,Hg,Wg,1024) fp16   Tier-2 grids
                      L16 = hidden_states[16] (pre-norm block output);
                      L24n = last_hidden_state (post final LayerNorm) — what the
                      detector FPN consumed.
    t1_<mode>_*, t1u_<mode>_*   Tier-1 ROI-pooled (7x7 roi_align mean) for
                      layers L16, L20, L24n over the existing feature-PKL boxes
                      (see token_common.pool_tier1).

Usage:
    CUDA_VISIBLE_DEVICES=1 python datasets/preprocess/tokens/dinov3_tokens.py \
        --split test --shard 0 --n_shards 2
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.preprocess.tokens.token_common import (  # noqa: E402
    MODES, TOKEN_CACHE_ROOT, DoneList, Status, add_common_args, annotated_frames,
    atomic_savez, compute_target_size, frame_path, load_feature_boxes, out_path,
    pool_tier1, resolve_videos,
)

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
STREAM = "dinov3l"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
GRID_LAYERS = ("L16", "L24n")
T1_LAYERS = ("L16", "L20", "L24n")


def load_frame(video: str, frame_file: str):
    img = cv2.imread(str(frame_path(video, frame_file)))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    tw, th = compute_target_size(w, h)
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    t = (t - MEAN) / STD
    return t, (tw, th)


@torch.no_grad()
def run_backbone(model, batch: torch.Tensor, patch: int):
    """batch (B,3,Hp,Wp) padded -> dict layer -> (B, C, Hg, Wg) fp16."""
    B, _, Hp, Wp = batch.shape
    Hg, Wg = Hp // patch, Wp // patch
    n = Hg * Wg
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(pixel_values=batch, output_hidden_states=True)
    hs = out.hidden_states  # len 25: [emb, L1..L24]
    grids = {
        "L16": hs[16][:, -n:, :],
        "L20": hs[20][:, -n:, :],
        "L24n": out.last_hidden_state[:, -n:, :],
    }
    return {k: v.to(torch.float16).permute(0, 2, 1).reshape(B, -1, Hg, Wg).contiguous()
            for k, v in grids.items()}, (Hg, Wg)


def process_video(model, patch: int, video: str, split: str, device, batch_size: int,
                  tier2: bool = True, decode_pool: ThreadPoolExecutor = None,
                  writer: ThreadPoolExecutor = None):
    frames = annotated_frames(video)
    tensors, keep = [], []
    tsize = None
    loaded = decode_pool.map(lambda f: load_frame(video, f), frames) if decode_pool else map(
        lambda f: load_frame(video, f), frames)
    for f, r in zip(frames, loaded):
        if r is None:
            continue
        tensors.append(r[0])
        keep.append(f)
        tsize = r[1]
    if not tensors:
        raise RuntimeError("no frames loaded")
    frames = keep
    tw, th = tsize
    Hp = ((th + patch - 1) // patch) * patch
    Wp = ((tw + patch - 1) // patch) * patch

    grids = {k: [] for k in T1_LAYERS}
    grid_hw = None
    for i in range(0, len(tensors), batch_size):
        chunk = torch.stack(tensors[i:i + batch_size]).to(device)
        chunk = F.pad(chunk, (0, Wp - tw, 0, Hp - th))  # zero pad right/bottom (RCNN batch_images)
        g, grid_hw = run_backbone(model, chunk, patch)
        for k in T1_LAYERS:
            grids[k].append(g[k])
    grids = {k: torch.cat(v) for k, v in grids.items()}  # (T, C, Hg, Wg) on device

    arrays = {
        "frames": np.array(frames),
        "target_size": np.array([tw, th], dtype=np.int32),
        "padded_size": np.array([Wp, Hp], dtype=np.int32),
        "grid_hw": np.array(grid_hw, dtype=np.int32),
        "patch": np.array(patch, dtype=np.int32),
        "model_id": np.array(MODEL_ID),
        "grid_layers": np.array(GRID_LAYERS if tier2 else []),
        "t1_layers": np.array(T1_LAYERS),
    }
    if tier2:
        for k in GRID_LAYERS:
            arrays[f"grid_{k}"] = grids[k].permute(0, 2, 3, 1).cpu().numpy()  # (T,Hg,Wg,C)
    for mode in MODES:
        boxes = load_feature_boxes(mode, split, video)
        arrays.update(pool_tier1(grids, frames, boxes, 1.0 / patch, mode))
    if writer is not None:  # overlap the (I/O-bound) npz write with the next video
        return len(frames), writer.submit(atomic_savez, out_path(STREAM, split, video), **arrays)
    atomic_savez(out_path(STREAM, split, video), **arrays)
    return len(frames), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--decode_threads", type=int, default=6)
    args = ap.parse_args()

    from transformers import AutoModel
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    patch = int(model.config.patch_size)
    assert patch == 16, patch

    videos = resolve_videos(args)
    tag = f"shard{args.shard}of{args.n_shards}"
    base = TOKEN_CACHE_ROOT / STREAM / args.split
    done = DoneList(base / f"done_{tag}.txt")
    status = Status(base / f"status_{tag}.json", stream=STREAM, split=args.split, shard=tag,
                    n_videos=len(videos), pid=os.getpid())
    n_ok = n_skip = n_err = n_frames = 0
    t0 = time.time()
    decode_pool = ThreadPoolExecutor(max_workers=args.decode_threads)
    writer = ThreadPoolExecutor(max_workers=1)
    pending = []  # (video, future) of in-flight npz writes; marked done once written

    def _drain(max_pending: int):
        nonlocal n_ok, n_err
        while len(pending) > max_pending:
            pv, fut = pending.pop(0)
            try:
                fut.result()
                done.add(pv)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"[ERR write] {pv}: {e}", flush=True)

    for i, v in enumerate(videos):
        if not args.overwrite and (v in done or out_path(STREAM, args.split, v).exists()):
            n_skip += 1
            if v not in done:
                done.add(v)
            continue
        try:
            nf, fut = process_video(model, patch, v, args.split, device, args.batch_size,
                                    tier2=not args.no_tier2, decode_pool=decode_pool, writer=writer)
            n_frames += nf
            pending.append((v, fut))
            _drain(2)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[ERR] {v}: {e}", flush=True)
            traceback.print_exc()
        if (i + 1) % 10 == 0 or i + 1 == len(videos):
            el = time.time() - t0
            print(f"[{tag}] {i + 1}/{len(videos)} ok={n_ok} skip={n_skip} err={n_err} "
                  f"frames={n_frames} {n_frames / max(el, 1e-6):.1f} fr/s", flush=True)
            status.write(processed=i + 1, ok=n_ok, skipped=n_skip, errors=n_err, frames=n_frames,
                         frames_per_s=round(n_frames / max(el, 1e-6), 2), last_video=v,
                         state="running")
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    _drain(0)
    writer.shutdown()
    decode_pool.shutdown()
    status.write(processed=len(videos), ok=n_ok, skipped=n_skip, errors=n_err, frames=n_frames,
                 state="finished")
    print(f"[{tag}] finished ok={n_ok} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
