#!/usr/bin/env python3
"""
C1b — Pi3 decoder-latent cache (frozen ``yyfz233/Pi3``, ``pi3`` env).

The clip fed to Pi3 reproduces ``3DBBoxAnnotationTool/backend/auto/reconstruction/pi3/ag_pi3.py``
exactly, so the latents align with the existing geometry in
``/data2/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic/<video>_10/predictions.npz``:

* sampled ids ``s = sampled_frames_idx/<video>.npy``; clip = ``s[s.index(first_annotated) : s.index(last_annotated)+1]``
* the vendored loader indexes the SORTED file list of ``frames/<video>.mp4`` by the
  sampled id itself (``filenames[id]``), i.e. it fed ``{id+1:06d}.png`` — an
  off-by-one that we reproduce (verified: MAE 0.0 against the stored ``images``).
  ``clip_ids`` (nominal) and ``clip_files`` (actually fed) are both stored.
* PIL LANCZOS resize to the Pi3 size (672x378), ToTensor, bf16 autocast, the whole
  clip in ONE forward (global attention spans the clip).

Decoder layers (arXiv 2511.22686, Sec. B.1, 0-indexed per attention type; the
decoder alternates frame (even block) / global (odd block) attention):
    frame layer j  -> decoder block 2j        F4, F12..F16 -> blocks 8, 24, 26, 28, 30, 32
    global layer j -> decoder block 2j+1      G13..G15     -> blocks 27, 29, 31

Per-video npz (``/data3/rohith/ag/cache/tokens/pi3/<split>/<video>.npz``):
    clip_ids (N,) clip_files (N,) str   frames (T,) annotated frames   frame_clip_idx (T,)
    target_size [W,H]  grid_hw [27,48]  patch 14  layer_blocks (dict as json str)
    grid_F14, grid_G14   (T,27,48,1024) fp16   Tier-2 (annotated frames only; 5 register tokens stripped)
    camera_poses (N,4,4) f32 from this run; pose_maxabs_diff_vs_existing scalar
    t1_<mode>_*, t1u_<mode>_*   Tier-1 pooled for F4,F12,F13,F14,F15,F16,G13,G14,G15

Usage:
    CUDA_VISIBLE_DEVICES=2 ~/anaconda3/envs/pi3/bin/python datasets/preprocess/tokens/pi3_tokens.py \
        --split test --shard 1 --n_shards 2
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.preprocess.tokens.token_common import (  # noqa: E402
    AG_ROOT, MODES, TOKEN_CACHE_ROOT, DoneList, Status, add_common_args, annotated_frames,
    atomic_savez, compute_target_size, load_feature_boxes, out_path, pool_tier1, resolve_videos,
)

STREAM = "pi3"
MODEL_ID = "yyfz233/Pi3"
PATCH = 14
N_REG = 5
FRAME_LAYERS = (4, 12, 13, 14, 15, 16)
GLOBAL_LAYERS = (13, 14, 15)
LAYER_BLOCKS = {**{f"F{j}": 2 * j for j in FRAME_LAYERS}, **{f"G{j}": 2 * j + 1 for j in GLOBAL_LAYERS}}
GRID_LAYERS = ("F14", "G14")
EXISTING_DIR = Path("/data2/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic")


def build_clip(video: str):
    s = np.load(AG_ROOT / "sampled_frames_idx" / f"{video}.npy").tolist()
    ann = annotated_frames(video)
    ann_ids = [int(f[:-4]) for f in ann]
    i0, i1 = s.index(ann_ids[0]), s.index(ann_ids[-1])
    clip_ids = s[i0:i1 + 1]
    files = sorted(f for f in os.listdir(AG_ROOT / "frames" / f"{video}.mp4")
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    clip_files, kept_ids = [], []
    for cid in clip_ids:  # vendored loader: filenames[i] for i in sample_idx (interval 1)
        if 0 <= cid < len(files):
            clip_files.append(files[cid])
            kept_ids.append(cid)
    pos = {cid: k for k, cid in enumerate(kept_ids)}
    frame_clip_idx = [pos.get(a, -1) for a in ann_ids]
    return ann, np.array(kept_ids), clip_files, np.array(frame_clip_idx)


def load_clip(video: str, clip_files):
    imgs = [Image.open(AG_ROOT / "frames" / f"{video}.mp4" / f).convert("RGB") for f in clip_files]
    W0, H0 = imgs[0].size
    tw, th = compute_target_size(W0, H0)
    ts = []
    for im in imgs:
        a = np.asarray(im.resize((tw, th), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        ts.append(torch.from_numpy(a).permute(2, 0, 1))
    return torch.stack(ts), (tw, th)


class Hooks:
    def __init__(self, model, n_reg: int):
        self.cap = {}
        self.sel = None  # tensor of clip indices to keep
        self.n_reg = n_reg
        self.handles = []
        for name, b in LAYER_BLOCKS.items():
            self.handles.append(model.decoder[b].register_forward_hook(self._make(name)))

    def _make(self, name):
        def hook(_m, _inp, out):
            h = out if torch.is_tensor(out) else out[0]
            C = h.shape[-1]
            h = h.reshape(-1, self.hw, C)[:, self.n_reg:, :]  # (N, hw_patch, C)
            self.cap[name] = h[self.sel].to(torch.float16)
        return hook

    def set(self, hw_with_reg: int, sel: torch.Tensor):
        self.hw = hw_with_reg
        self.sel = sel
        self.cap = {}


@torch.no_grad()
def process_video(model, hooks: Hooks, video: str, split: str, device, tier2: bool = True):
    frames, clip_ids, clip_files, frame_clip_idx = build_clip(video)
    imgs, (tw, th) = load_clip(video, clip_files)
    N = imgs.shape[0]
    Hg, Wg = th // PATCH, tw // PATCH
    valid = frame_clip_idx >= 0
    sel = torch.as_tensor(frame_clip_idx[valid], dtype=torch.long, device=device)
    hooks.set(Hg * Wg + N_REG, sel)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(imgs[None].to(device))
    poses = pred["camera_poses"].float()[0].cpu().numpy()
    del pred

    grids = {k: v.permute(0, 2, 1).reshape(-1, v.shape[-1], Hg, Wg).contiguous()
             for k, v in hooks.cap.items()}  # (T, C, Hg, Wg)
    hooks.cap = {}
    kept_frames = [f for f, ok in zip(frames, valid) if ok]

    diff = -1.0
    ex = EXISTING_DIR / f"{video}_10" / "predictions.npz"
    if ex.exists():
        try:
            with np.load(ex) as z:
                cp = z["camera_poses"]
            if cp.shape == poses.shape:
                diff = float(np.abs(cp - poses).max())
            else:
                diff = -2.0  # frame-count mismatch
        except Exception:  # noqa: BLE001
            diff = -3.0

    arrays = {
        "clip_ids": clip_ids.astype(np.int32), "clip_files": np.array(clip_files),
        "frames": np.array(kept_frames), "frame_clip_idx": frame_clip_idx[valid].astype(np.int32),
        "frames_missing_from_clip": np.array([f for f, ok in zip(frames, valid) if not ok]),
        "target_size": np.array([tw, th], np.int32), "grid_hw": np.array([Hg, Wg], np.int32),
        "patch": np.array(PATCH, np.int32), "model_id": np.array(MODEL_ID),
        "layer_blocks": np.array(json.dumps(LAYER_BLOCKS)),
        "grid_layers": np.array(GRID_LAYERS if tier2 else []),
        "t1_layers": np.array(list(LAYER_BLOCKS)),
        "camera_poses": poses.astype(np.float32),
        "pose_maxabs_diff_vs_existing": np.array(diff, np.float32),
    }
    if tier2:
        for k in GRID_LAYERS:
            arrays[f"grid_{k}"] = grids[k].permute(0, 2, 3, 1).cpu().numpy()
    for mode in MODES:
        boxes = load_feature_boxes(mode, split, video)
        arrays.update(pool_tier1(grids, kept_frames, boxes, 1.0 / PATCH, mode))
    atomic_savez(out_path(STREAM, split, video), **arrays)
    return N, len(kept_frames), diff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()

    from pi3.models.pi3 import Pi3
    device = torch.device("cuda")
    model = Pi3.from_pretrained(MODEL_ID).to(device).eval()
    hooks = Hooks(model, N_REG)

    videos = resolve_videos(args)
    tag = f"shard{args.shard}of{args.n_shards}"
    base = TOKEN_CACHE_ROOT / STREAM / args.split
    done = DoneList(base / f"done_{tag}.txt")
    status = Status(base / f"status_{tag}.json", stream=STREAM, split=args.split, shard=tag,
                    n_videos=len(videos), pid=os.getpid())
    n_ok = n_skip = n_err = n_clip = n_frames = 0
    max_diff = 0.0
    t0 = time.time()
    for i, v in enumerate(videos):
        if not args.overwrite and (v in done or out_path(STREAM, args.split, v).exists()):
            n_skip += 1
            if v not in done:
                done.add(v)
            continue
        try:
            N, T, diff = process_video(model, hooks, v, args.split, device, tier2=not args.no_tier2)
            n_clip += N
            n_frames += T
            if diff > max_diff:
                max_diff = diff
            done.add(v)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[ERR] {v}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i + 1 == len(videos):
            el = time.time() - t0
            print(f"[{tag}] {i + 1}/{len(videos)} ok={n_ok} skip={n_skip} err={n_err} "
                  f"clipframes={n_clip} annframes={n_frames} {n_clip / max(el, 1e-6):.1f} clipfr/s "
                  f"max_pose_diff={max_diff:.4f}", flush=True)
            status.write(processed=i + 1, ok=n_ok, skipped=n_skip, errors=n_err, clip_frames=n_clip,
                         frames=n_frames, clip_frames_per_s=round(n_clip / max(el, 1e-6), 2),
                         max_pose_diff=max_diff, last_video=v, state="running")
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
    status.write(processed=len(videos), ok=n_ok, skipped=n_skip, errors=n_err, clip_frames=n_clip,
                 frames=n_frames, max_pose_diff=max_diff, state="finished")
    print(f"[{tag}] finished ok={n_ok} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
