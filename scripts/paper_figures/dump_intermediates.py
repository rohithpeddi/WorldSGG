"""Dump the intermediate tensors of one video's inference for the three WorldWise
variants and render them as figure-ready panels.

Every panel is *empirical*: it is read out of a single forward pass of the trained
checkpoint on the real test item.  Nothing is drawn that the model did not compute.

For each method (``worldwise``, ``worldwise_plus``, ``worldwise_pp``) the script

* loads the predcls test dataset and the best checkpoint exactly as
  ``tools/dump_predictions.py`` does,
* hooks the modules of interest (scaffold tokenizer, associative retriever /
  entity decoder, grid fusion, the attention modules whose weights the model
  itself discards) so that their outputs are captured without changing the
  computation,
* writes ``<out>/<method>/tensors.npz`` with every captured array, a
  ``preds.json`` with the predicted and ground-truth predicates of the key
  frames, and one PNG per panel (see ``PANELS`` in the module docstring of each
  ``render_*`` function).

Usage (on CS93371, from the repository root)::

    ~/anaconda3/envs/scene4cast/bin/python scripts/paper_figures/dump_intermediates.py \
        --video HI75B --out-dir /data3/rohith/ag/runs/intermediates

Three key frames are chosen automatically: the first frame in which the
*tracked* object is visible, one in which it is unseen, and the last frame in
which it is visible again (or the last frame).  The tracked object is the slot
with the most unseen frames among those seen at least three times, i.e. the
object whose permanence the model has to reason about.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402
from PIL import Image                                              # noqa: E402

from wsgg_base import load_wsgg_config                             # noqa: E402
from tools.reeval_test import (                                    # noqa: E402
    build_model, forward_all_frames, load_checkpoint, make_test_dataset,
)

CELLS = {
    "worldwise": "configs/methods/predcls/worldwise_predcls_dinov3l.yaml",
    "worldwise_plus": "configs/methods/predcls/worldwise_plus_dinov3tok_predcls.yaml",
    "worldwise_pp": "configs/methods/predcls/worldwise_pp_dinov3_predcls.yaml",
}
FRAMES_ROOT = "/data/rohith/ag/frames"

# Dark palette shared with the SVG figure generators (scripts/paper_figures/dark).
BG = "#0f1114"
TXT = "#e6e8ec"
MUTED = "#9aa3b2"
BLUE = "#5aa2f2"
VIOLET = "#a78bfa"
TEAL = "#2dd4bf"
ORANGE = "#ff8a3d"
RED = "#f87171"
GREEN = "#4ade80"
MASKC = "#64748b"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def pca_rgb(X: np.ndarray, basis: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None):
    """Project rows of X (M, D) onto 3 principal components and map to [0, 1] RGB.
    Returns (rgb (M, 3), basis) so several panels can share one basis."""
    from sklearn.decomposition import PCA
    if basis is None:
        p = PCA(n_components=3, random_state=0).fit(X)
        Z = p.transform(X)
        lo, hi = np.percentile(Z, 2, axis=0), np.percentile(Z, 98, axis=0)
        basis = (p, lo, hi)
    p, lo, hi = basis
    Z = p.transform(X)
    rgb = np.clip((Z - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return rgb, basis


def dark_fig(w: float, h: float):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig


def save(fig, path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def style_axis(ax, spine=MUTED):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(spine)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TXT)


def upsample_map(m: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear upsample a (Hp, Wp) map to (H, W)."""
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------
class Capture:
    """Collects intermediate tensors by module hooks and light monkey-patches."""

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self._handles = []

    def add(self, key: str, value):
        self.store.setdefault(key, []).append(value.detach().cpu() if isinstance(value, torch.Tensor) else value)

    def hook_output(self, module: torch.nn.Module, key: str, pick=None):
        def h(_m, _i, out):
            self.add(key, pick(out) if pick else out)
        self._handles.append(module.register_forward_hook(h))

    def hook_input(self, module: torch.nn.Module, key: str, idx: int = 0):
        def h(_m, inp):
            self.add(key, inp[idx])
        self._handles.append(module.register_forward_pre_hook(h))

    def patch_mha_weights(self, mha: torch.nn.MultiheadAttention, key: str, average: bool = True):
        """Force ``need_weights=True`` on an attention module and record the weights.
        The returned output tensor is unchanged, so the forward pass is identical."""
        orig = mha.forward

        def fwd(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = average
            out, w = orig(*args, **kwargs)
            self.add(key, w)
            return out, w
        mha.forward = fwd  # type: ignore[assignment]
        self._handles.append(("patch", mha, orig))

    def close(self):
        for h in self._handles:
            if isinstance(h, tuple):
                h[1].forward = h[2]
            else:
                h.remove()
        self._handles = []

    def get(self, key: str, i: int = 0):
        return self.store[key][i]

    def has(self, key: str) -> bool:
        return key in self.store


def install_hooks(model, method: str) -> Capture:
    cap = Capture()
    tok = model.scaffold_tokenizer
    cap.hook_output(tok, "scaffold", pick=lambda o: o[0])                # hybrid tokens (T, N, d)
    cap.hook_output(tok, "is_masked", pick=lambda o: o[1])
    cap.hook_input(tok.visual_projector, "visual_in")                    # raw appearance vector (T, N, D)
    cap.hook_output(tok.visual_projector, "visual_proj")                 # projected appearance (T, N, d)
    if method in ("worldwise", "worldwise_plus"):
        cap.hook_output(model.retriever, "completed")
        for i, layer in enumerate(model.retriever.cross_attn_layers):
            cap.patch_mha_weights(layer, f"retr_attn_{i}")
        cap.hook_output(model.inter_object_encoder, "enriched")
    else:
        gf = model.grid_fusion
        cap.hook_output(gf.gate, "grid_gate_logits")                    # (T, HW, 2)
        cap.hook_output(gf.norm, "memory")                               # fused memory before PE (T, HW, d)
        for i, layer in enumerate(model.decoder):
            cap.hook_output(layer, f"dec_out_{i}", pick=lambda o: o[0])
            cap.hook_output(layer, f"spatial_attn_{i}", pick=lambda o: o[1])   # (T, heads, Q+N, Q+N)
            cap.patch_mha_weights(layer.cross_attn, f"cross_attn_{i}")           # (T, Q+N, HW)
            cap.patch_mha_weights(layer.temporal_attn, f"temporal_attn_{i}")     # (N, T, T)
    return cap


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
def load_item(ds, video: str):
    keep = [v for v in ds.video_list if v.replace(".mp4", "") == video]
    if not keep:
        raise SystemExit(f"{video} not in the test split of this cell")
    idx = ds.video_list.index(keep[0])
    return ds[idx], keep[0]


def target_size(ds, video_key: str, frame: str) -> Tuple[int, int]:
    """(W, H) of the feature-PKL coordinate space for this frame."""
    feat = ds._load_feature_pkl(video_key)
    ff = feat["frames"][frame]
    ts = ff.get("target_size", None)
    if ts is None:
        from dataloader.world_ag_dataset import _compute_target_size
        H, W = raw_frame(video_key.replace(".mp4", ""), frame).shape[:2]
        tw, th = _compute_target_size(W, H)
        return int(tw), int(th)
    return int(ts[0]), int(ts[1])


def raw_frame(video: str, frame: str) -> np.ndarray:
    p = Path(FRAMES_ROOT) / f"{video}.mp4" / frame
    if not p.exists():
        p = Path(FRAMES_ROOT) / f"{video}.mp4" / (Path(frame).stem + ".png")
    return np.asarray(Image.open(p).convert("RGB"))


def pick_keyframes(vis: np.ndarray, valid: np.ndarray, labels: Sequence[str]) -> Tuple[int, List[int]]:
    """Tracked slot + three key frames (visible / unseen / visible-again-or-last)."""
    T, N = vis.shape
    best, best_score = None, -1
    for n in range(N):
        if not valid[:, n].any() or labels[n] == "person":
            continue
        n_vis = int(vis[:, n].sum())
        n_unseen = int((valid[:, n] & ~vis[:, n]).sum())
        if n_vis >= 3 and n_unseen > best_score:
            best, best_score = n, n_unseen
    if best is None:
        best = int(np.argmax(valid.sum(0)))
    v = vis[:, best]
    first_vis = int(np.argmax(v)) if v.any() else 0
    unseen = np.where(valid[:, best] & ~v)[0]
    mid = int(unseen[len(unseen) // 2]) if len(unseen) else T // 2
    later = np.where(v & (np.arange(T) > mid))[0]
    last = int(later[-1]) if len(later) else T - 1
    keys = sorted(set([first_vis, mid, last]))
    while len(keys) < 3:
        keys.append(min(T - 1, keys[-1] + 1))
        keys = sorted(set(keys))
    return best, keys[:3]


# ---------------------------------------------------------------------------
# panels shared by every method
# ---------------------------------------------------------------------------
def render_frames(out: Path, video: str, frames: List[str], keys: List[int], boxes: np.ndarray,
                  vis: np.ndarray, valid: np.ndarray, labels: List[str], ts: Tuple[int, int],
                  tracked: int):
    """frame_<k>.png: the raw frame with the input 2-D boxes of the visible objects
    (solid) and a chip naming each unseen object (no box exists for it)."""
    for k, t in enumerate(keys):
        img = raw_frame(video, frames[t])
        H, W = img.shape[:2]
        sx, sy = W / ts[0], H / ts[1]
        fig = dark_fig(4.0, 4.0 * H / W)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        unseen = []
        for n in range(boxes.shape[1]):
            if not valid[t, n]:
                continue
            col = ORANGE if n == tracked else (BLUE if labels[n] != "person" else TXT)
            if vis[t, n] and boxes[t, n].any():
                x0, y0, x1, y1 = boxes[t, n] * np.array([sx, sy, sx, sy])
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2.2, ec=col))
                ax.text(x0 + 3, y0 + 3, labels[n], color="black", fontsize=9, va="top", ha="left",
                        bbox=dict(fc=col, ec="none", pad=1.5))
            else:
                unseen.append((labels[n], col))
        for i, (lab, col) in enumerate(unseen):
            ax.text(6, 8 + i * 22, f"{lab}: unseen (no 2-D box)", color=col, fontsize=9, va="top", ha="left",
                    bbox=dict(fc="#000000aa", ec=col, lw=1.2, pad=2, ls="--"))
        ax.text(6, H - 6, f"t = {t + 1} / {len(frames)}   ({frames[t]})", color=TXT, fontsize=8, va="bottom",
                bbox=dict(fc="#000000aa", ec="none", pad=2))
        save(fig, out / f"frame_{k}.png", dpi=150)


def render_visibility(out: Path, vis: np.ndarray, valid: np.ndarray, labels: List[str], keys: List[int],
                      tracked: int):
    """visibility.png: objects x frames; filled = visible, dashed hollow = unseen.
    This is the real token grid the scaffold tokenizer builds."""
    T, N = vis.shape
    rows = [n for n in range(N) if valid[:, n].any()]
    fig = dark_fig(max(4.0, 0.16 * T + 1.2), 0.26 * len(rows) + 0.7)
    ax = fig.add_axes([0.22, 0.18, 0.76, 0.78])
    style_axis(ax, spine=BG)
    for r, n in enumerate(rows):
        for t in range(T):
            if not valid[t, n]:
                continue
            y = len(rows) - 1 - r
            if vis[t, n]:
                col = ORANGE if n == tracked else BLUE
                ax.add_patch(Rectangle((t + 0.08, y + 0.08), 0.84, 0.84, fc=col, ec="none"))
            else:
                ax.add_patch(Rectangle((t + 0.12, y + 0.12), 0.76, 0.76, fill=False, ec=MASKC, ls=(0, (2, 1.5)), lw=1))
    for t in keys:
        ax.add_patch(Rectangle((t, -0.35), 1, len(rows) + 0.35, fill=False, ec=TXT, lw=1.0, ls=":"))
    ax.set_xlim(0, T); ax.set_ylim(-0.4, len(rows))
    ax.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([labels[n] for n in rows], color=TXT, fontsize=8)
    ax.set_xticks([0.5] + [t + 0.5 for t in keys] + [T - 0.5])
    ax.set_xticklabels(["1"] + [str(t + 1) for t in keys] + [str(T)], fontsize=7)
    ax.set_xlabel("frame", fontsize=8)
    save(fig, out / "visibility.png")


def render_token_grid(out: Path, name: str, tokens: np.ndarray, valid: np.ndarray, vis: np.ndarray,
                      labels: List[str], basis=None, title: str = "", mark_masked: bool = True):
    """<name>.png: objects x frames grid coloured by a 3-component PCA of the token at
    that cell.  Masked cells are outlined when ``mark_masked``."""
    T, N, D = tokens.shape
    rows = [n for n in range(N) if valid[:, n].any()]
    X = tokens[valid]                                         # (M, D)
    rgb, basis = pca_rgb(X, basis)
    img = np.zeros((len(rows), T, 3)) + 0.06
    k = 0
    idx = {(t, n): i for i, (t, n) in enumerate(zip(*np.where(valid)))}
    for r, n in enumerate(rows):
        for t in range(T):
            if valid[t, n]:
                img[r, t] = rgb[idx[(t, n)]]
    fig = dark_fig(max(4.0, 0.16 * T + 1.2), 0.26 * len(rows) + 0.7)
    ax = fig.add_axes([0.22, 0.18, 0.76, 0.78])
    style_axis(ax, spine=BG)
    ax.imshow(img, interpolation="nearest", aspect="auto", extent=(0, T, 0, len(rows)))
    if mark_masked:
        for r, n in enumerate(rows):
            for t in range(T):
                if valid[t, n] and not vis[t, n]:
                    ax.add_patch(Rectangle((t + 0.1, len(rows) - 1 - r + 0.1), 0.8, 0.8, fill=False,
                                           ec=TXT, ls=(0, (2, 1.5)), lw=0.9))
    ax.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([labels[n] for n in rows], color=TXT, fontsize=8)
    ax.set_xticks([0.5, T - 0.5]); ax.set_xticklabels(["1", str(T)], fontsize=7)
    ax.set_xlim(0, T); ax.set_ylim(0, len(rows))
    if title:
        ax.set_title(title, fontsize=8, color=TXT, loc="left")
    save(fig, out / f"{name}.png")
    return basis


def render_temporal_attention(out: Path, name: str, A: np.ndarray, vis_row: np.ndarray, valid_row: np.ndarray,
                              label: str, keys: List[int], title: str):
    """<name>.png: for one slot, the (query frame x key frame) attention. Visible
    key frames are ticked on the top axis; the tracked frames on the left."""
    T = A.shape[0]
    fig = dark_fig(3.6, 3.4)
    ax = fig.add_axes([0.14, 0.12, 0.72, 0.74])
    style_axis(ax)
    im = ax.imshow(A, cmap="magma", interpolation="nearest", aspect="equal", vmin=0)
    for t in range(T):
        if valid_row[t] and vis_row[t]:
            ax.add_patch(Rectangle((t - 0.5, -1.4), 1, 0.7, fc=ORANGE, ec="none", clip_on=False))
        if valid_row[t] and not vis_row[t]:
            ax.add_patch(Rectangle((-1.4, t - 0.5), 0.7, 1, fc=MASKC, ec="none", clip_on=False))
    ax.set_xlabel("key frame (visible frames ticked in orange)", fontsize=7)
    ax.set_ylabel("query frame (unseen frames ticked in grey)", fontsize=7)
    ax.set_xticks([0, T - 1]); ax.set_xticklabels(["1", str(T)])
    ax.set_yticks([0, T - 1]); ax.set_yticklabels(["1", str(T)])
    ax.set_title(f"{title}: {label}", fontsize=8, loc="left", pad=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=6)
    save(fig, out / f"{name}.png")


def predicate_names(ds):
    return list(ds.attention_relationships), list(ds.spatial_relationships), list(ds.contacting_relationships)


def collect_preds(ds, item, pred, keys: List[int], labels: List[str]) -> Dict[str, Any]:
    att_n, spa_n, con_n = predicate_names(ds)
    att = to_np(pred["attention_distribution"]); spa = to_np(pred["spatial_distribution"]); con = to_np(pred["contacting_distribution"])
    pidx = to_np(item["person_idx"]).astype(int); oidx = to_np(item["object_idx"]).astype(int)
    pv = to_np(item["pair_valid"]).astype(bool)
    g_att = to_np(item["gt_attention"]).astype(int); g_spa = to_np(item["gt_spatial"]); g_con = to_np(item["gt_contacting"])
    vis = to_np(item["visibility_mask"]).astype(bool)
    out = {}
    for t in keys:
        rows = []
        for k in range(pidx.shape[1]):
            if not pv[t, k]:
                continue
            o = oidx[t, k]
            rows.append({
                "object": labels[o], "visible": bool(vis[t, o]),
                "pred_attention": att_n[int(att[t, k].argmax())],
                "gt_attention": att_n[g_att[t, k]] if 0 <= g_att[t, k] < len(att_n) else None,
                "pred_spatial": [spa_n[i] for i in np.argsort(-spa[t, k])[:2]],
                "pred_spatial_p": [float(spa[t, k, i]) for i in np.argsort(-spa[t, k])[:2]],
                "gt_spatial": [spa_n[i] for i in np.where(g_spa[t, k] > 0.5)[0]],
                "pred_contacting": [con_n[i] for i in np.argsort(-con[t, k])[:2]],
                "pred_contacting_p": [float(con[t, k, i]) for i in np.argsort(-con[t, k])[:2]],
                "gt_contacting": [con_n[i] for i in np.where(g_con[t, k] > 0.5)[0]],
            })
        out[str(t)] = rows
    return out


def render_preds(out: Path, preds: Dict[str, Any], keys: List[int], tracked_label: str):
    """preds_<k>.png: the predicted person-object predicates at each key frame, with
    the ground truth beside them; a hit is green, a miss red, an unseen object is
    marked so the reader can see permanence at work."""
    for k, t in enumerate(keys):
        rows = preds[str(t)]
        fig = dark_fig(5.2, 0.36 * max(len(rows), 1) + 0.5)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_facecolor(BG)
        y = 0.94
        ax.text(0.02, y, f"t = {t + 1}", color=TXT, fontsize=8, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.40, y, "attention", color=MUTED, fontsize=7, va="top", transform=ax.transAxes)
        ax.text(0.61, y, "spatial", color=MUTED, fontsize=7, va="top", transform=ax.transAxes)
        ax.text(0.81, y, "contacting", color=MUTED, fontsize=7, va="top", transform=ax.transAxes)
        dy = 0.86 / max(len(rows), 1)
        for i, r in enumerate(rows):
            yy = y - 0.09 - i * dy
            col = ORANGE if r["object"] == tracked_label else TXT
            tag = "" if r["visible"] else "  (unseen)"
            ax.text(0.02, yy, f"person → {r['object']}{tag}", color=col, fontsize=7.5, va="top", transform=ax.transAxes)
            a_ok = r["pred_attention"] == r["gt_attention"]
            ax.text(0.40, yy, r["pred_attention"], color=GREEN if a_ok else RED, fontsize=7.5, va="top", transform=ax.transAxes)
            s = r["pred_spatial"][0]; s_ok = s in r["gt_spatial"]
            ax.text(0.61, yy, s, color=GREEN if s_ok else RED, fontsize=7.5, va="top", transform=ax.transAxes)
            c = r["pred_contacting"][0]; c_ok = c in r["gt_contacting"]
            ax.text(0.81, yy, c, color=GREEN if c_ok else RED, fontsize=7.5, va="top", transform=ax.transAxes)
        save(fig, out / f"preds_{k}.png")


# ---------------------------------------------------------------------------
# grid panels (WorldWise+ / WorldWise++)
# ---------------------------------------------------------------------------
def render_grid_pca(out: Path, name: str, grid: np.ndarray, keys: List[int], title: str,
                    boxes: Optional[np.ndarray] = None, image_hw: Optional[Tuple[int, int]] = None,
                    vis: Optional[np.ndarray] = None, valid: Optional[np.ndarray] = None,
                    labels: Optional[List[str]] = None, tracked: int = -1):
    """<name>_<k>.png: PCA-RGB of a (T, Hp, Wp, D) token grid at each key frame, with
    the object boxes drawn in grid coordinates when given (they are in the
    (H, W) pixel space of the same image)."""
    T, Hp, Wp, D = grid.shape
    X = grid.reshape(-1, D).astype(np.float32)
    sub = X[np.random.RandomState(0).choice(len(X), min(len(X), 60000), replace=False)]
    _, basis = pca_rgb(sub)
    for k, t in enumerate(keys):
        rgb, _ = pca_rgb(grid[t].reshape(-1, D).astype(np.float32), basis)
        img = rgb.reshape(Hp, Wp, 3)
        fig = dark_fig(3.6, 3.6 * Hp / Wp)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img, interpolation="nearest")
        if boxes is not None and image_hw is not None:
            H, W = image_hw
            sx, sy = Wp / W, Hp / H
            for n in range(boxes.shape[1]):
                if valid is not None and (not valid[t, n] or not vis[t, n]) or not boxes[t, n].any():
                    continue
                x0, y0, x1, y1 = boxes[t, n] * np.array([sx, sy, sx, sy]) - 0.5
                col = ORANGE if n == tracked else TXT
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.6, ec=col))
                if labels is not None:
                    ax.text(x0 + 0.3, y0 + 0.3, labels[n], color="black", fontsize=7, va="top",
                            bbox=dict(fc=col, ec="none", pad=1))
        ax.text(0.02, 0.98, f"{title}  t = {t + 1}", color=TXT, fontsize=8, va="top", transform=ax.transAxes,
                bbox=dict(fc="#000000aa", ec="none", pad=2))
        save(fig, out / f"{name}_{k}.png", dpi=150)


def render_heatmap_on_frame(out: Path, name: str, maps: np.ndarray, keys: List[int], video: str,
                            frames: List[str], title: str, cmap: str = "inferno", alpha: float = 0.62,
                            box: Optional[np.ndarray] = None, box_hw: Optional[Tuple[int, int]] = None,
                            box_label: str = "", box_vis: Optional[np.ndarray] = None):
    """<name>_<k>.png: a (T, Hp, Wp) map upsampled onto the raw frame."""
    for k, t in enumerate(keys):
        img = raw_frame(video, frames[t])
        H, W = img.shape[:2]
        m = upsample_map(maps[t], (H, W))
        m = (m - m.min()) / max(m.max() - m.min(), 1e-8)
        fig = dark_fig(4.0, 4.0 * H / W)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img)
        ax.imshow(m, cmap=cmap, alpha=alpha, interpolation="bilinear")
        if box is not None and box_hw is not None and box[t].any() and (box_vis is None or box_vis[t]):
            sx, sy = W / box_hw[1], H / box_hw[0]
            x0, y0, x1, y1 = box[t] * np.array([sx, sy, sx, sy])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2, ec=ORANGE, ls="--"))
            ax.text(x0 + 3, y0 + 3, box_label, color="black", fontsize=8, va="top", bbox=dict(fc=ORANGE, ec="none", pad=1.5))
        elif box_vis is not None and not box_vis[t]:
            ax.text(6, H - 30, f"{box_label}: unseen at this frame (no 2-D box)", color=ORANGE, fontsize=8, va="bottom",
                    bbox=dict(fc="#000000aa", ec=ORANGE, lw=1.2, pad=2, ls="--"))
        ax.text(0.02, 0.98, f"{title}  t = {t + 1}", color=TXT, fontsize=8, va="top", transform=ax.transAxes,
                bbox=dict(fc="#000000aa", ec="none", pad=2))
        save(fig, out / f"{name}_{k}.png", dpi=150)


def render_spatial_attention(out: Path, A: np.ndarray, valid_t: np.ndarray, labels: List[str], Q: int, t: int,
                             tracked: int):
    """spatial_attn.png: the decoder's spatial self-attention at one key frame
    (averaged over layers and heads), slots labelled, free queries collapsed to
    their mean row / column so the panel stays legible."""
    slots = [n for n in range(len(valid_t)) if valid_t[n]]
    idx = [Q + n for n in slots]
    S = A[np.ix_(idx, idx)]
    fq = A[:Q][:, idx].mean(0, keepdims=True)         # free queries -> slots
    sf = A[idx][:, :Q].mean(1, keepdims=True)         # slots -> free queries
    M = np.block([[S, sf], [fq, np.array([[A[:Q, :Q].mean()]])]])
    names = [labels[n] for n in slots] + ["free q (mean)"]
    fig = dark_fig(3.6, 3.3)
    ax = fig.add_axes([0.3, 0.05, 0.62, 0.72])
    style_axis(ax)
    im = ax.imshow(M, cmap="magma", interpolation="nearest", vmin=0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for lab, n in zip(ax.get_yticklabels(), slots + [-1]):
        lab.set_color(ORANGE if n == tracked else TXT)
    for lab, n in zip(ax.get_xticklabels(), slots + [-1]):
        lab.set_color(ORANGE if n == tracked else TXT)
    ax.set_title(f"spatial self-attention (query row → key column), t = {t + 1}", fontsize=7.5, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(colors=MUTED, labelsize=6)
    save(fig, out / "spatial_attn.png")


def render_detections(out: Path, video: str, frames: List[str], keys: List[int], det: Dict[str, Any],
                      image_hw: Tuple[int, int], in_boxes: np.ndarray, valid: np.ndarray, vis: np.ndarray,
                      labels: List[str], classes: List[str], tracked: int, thr: float = 0.3):
    """det_<k>.png: free-query detections (orange, score-thresholded) and the refined
    slot boxes (blue) against the detector's input boxes (dotted) at key frames."""
    H, W = image_hw
    logits = to_np(det["logits"]); xyxy = to_np(det["boxes_xyxy"]); slot = to_np(det["slot_boxes"])
    prob = torch.softmax(torch.from_numpy(logits), -1).numpy()
    for k, t in enumerate(keys):
        img = raw_frame(video, frames[t])
        Hi, Wi = img.shape[:2]
        sx, sy = Wi / W, Hi / H
        fig = dark_fig(4.0, 4.0 * Hi / Wi)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img)
        # free queries
        p = prob[t]; cls = p[:, :-1].argmax(-1); sc = p[np.arange(len(p)), cls]
        for q in np.argsort(-sc)[:8]:
            if sc[q] < thr:
                continue
            x0, y0, x1, y1 = xyxy[t, q] * np.array([sx, sy, sx, sy])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.8, ec=ORANGE))
            ax.text(x0 + 2, y1 - 2, f"{classes[cls[q]]} {sc[q]:.2f}", color="black", fontsize=7, va="bottom",
                    bbox=dict(fc=ORANGE, ec="none", pad=1))
        # slots: input (dotted) and refined (solid)
        for n in range(in_boxes.shape[1]):
            if not valid[t, n] or not vis[t, n] or not in_boxes[t, n].any():
                continue
            b0 = in_boxes[t, n] * np.array([sx, sy, sx, sy])
            b1 = slot[t, n] * np.array([W, H, W, H]) * np.array([sx, sy, sx, sy])
            ax.add_patch(Rectangle((b0[0], b0[1]), b0[2] - b0[0], b0[3] - b0[1], fill=False, lw=1.2, ec=MUTED, ls=":"))
            ax.add_patch(Rectangle((b1[0], b1[1]), b1[2] - b1[0], b1[3] - b1[1], fill=False, lw=1.8, ec=BLUE))
            ax.text(b1[0] + 2, b1[1] + 2, labels[n], color="black", fontsize=7, va="top", bbox=dict(fc=BLUE, ec="none", pad=1))
        ax.text(0.02, 0.98, f"free-query detections (orange) · refined slots (blue) · input (dotted)  t = {t + 1}",
                color=TXT, fontsize=6.5, va="top", transform=ax.transAxes, bbox=dict(fc="#000000aa", ec="none", pad=2))
        save(fig, out / f"det_{k}.png", dpi=150)


def convex_hull(P: np.ndarray) -> np.ndarray:
    """Closed 2-D convex hull (monotone chain) of the points P (M, 2)."""
    pts = sorted(set(map(tuple, np.round(P, 6))))
    if len(pts) <= 2:
        return np.array(pts + pts[:1])
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = lower[:-1] + upper[:-1]
    return np.array(h + h[:1])


def render_bev(out: Path, keys: List[int], corners_in: np.ndarray, corners_ref: np.ndarray, valid: np.ndarray,
               vis: np.ndarray, labels: List[str], tracked: int):
    """bev_<k>.png: bird's-eye view of the input 3-D boxes (dotted) and the refined
    slot boxes (solid) in the world frame; the tracked object in orange even when unseen."""
    for k, t in enumerate(keys):
        fig = dark_fig(3.4, 3.4)
        ax = fig.add_axes([0.1, 0.1, 0.86, 0.82]); style_axis(ax)
        allpts = []
        for n in range(corners_in.shape[1]):
            if not valid[t, n]:
                continue
            col = ORANGE if n == tracked else (BLUE if vis[t, n] else MASKC)
            for C, ls, lw in ((corners_in[t, n], ":", 1.0), (corners_ref[t, n], "-", 1.6)):
                xy = C[:, :2]
                hull = convex_hull(xy)
                ax.plot(hull[:, 0], hull[:, 1], ls=ls, lw=lw, color=col)
                allpts.append(xy)
            c = corners_ref[t, n].mean(0)
            ax.text(c[0], c[1], labels[n] + ("" if vis[t, n] else " (unseen)"), color=col, fontsize=6.5, ha="center")
        if allpts:
            P = np.concatenate(allpts); lo, hi = P.min(0) - 0.3, P.max(0) + 0.3
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(f"bird's-eye view, world frame, t = {t + 1}  (input dotted, refined solid)", fontsize=7, loc="left")
        save(fig, out / f"bev_{k}.png")


# ---------------------------------------------------------------------------
# geometry scaffold inputs: OBB corners, camera poses, the π³ scene
# ---------------------------------------------------------------------------
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box_edges(C: np.ndarray):
    """Edge index pairs of the (8, 3) corners C.  The annotation convention is
    bottom four then top four; if opposite edges disagree in length (another
    ordering) fall back to the three nearest neighbours of every corner."""
    L = np.array([np.linalg.norm(C[i] - C[j]) for i, j in BOX_EDGES])
    groups = [L[[0, 2, 4, 6]], L[[1, 3, 5, 7]], L[[8, 9, 10, 11]]]
    if all(g.max() - g.min() < 0.05 * max(g.mean(), 1e-6) for g in groups):
        return BOX_EDGES
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    E = set()
    for i in range(8):
        for j in np.argsort(D[i])[1:4]:
            E.add((min(i, int(j)), max(i, int(j))))
    return sorted(E)


class LegacyScene:
    """π³ scene access for a video outside the WorldBBox split, whose legacy
    annotation carries neither camera poses nor a canonical transform: the
    canonical frame is then the π³ world frame itself, and the clip index of an
    annotated frame is recovered from the sampled-frame list (the clip starts at
    the first annotated frame, as the reconstruction script defines it)."""

    def __init__(self, video: str, frames: List[str], annot_dir: str = "world4d_rel_annotations"):
        import glob
        import pickle
        paths = sorted(glob.glob(f"/data2/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic/{video}_*/predictions.npz"))
        if not paths:
            raise FileNotFoundError(f"no π³ reconstruction for {video}")
        z = np.load(paths[0])
        self.points, self.local_points = z["points"], z["local_points"]
        self.conf, self.images, self.poses = z["conf"], z["images"], z["camera_poses"]
        self.pi3 = {"local_points": self.local_points}
        self.s = np.load(f"/data/rohith/ag/sampled_frames_idx/{video}.npy").tolist()
        ids = sorted(int(Path(f).stem) for f in frames)
        self.start = self.s.index(ids[0])
        H, W = self.points.shape[1:3]
        self.pi3_size = (W, H)
        # The legacy annotation stores corners in a canonical frame whose transform
        # was not saved.  Recover the rigid world->canonical transform the way the
        # boxes were built: the π³ points inside each 2-D box give a world centroid,
        # the stored corners give the canonical one, Kabsch aligns the pairs.
        self.T = np.eye(4)
        ap = Path("/data/rohith/ag") / annot_dir / "test" / f"{video}.mp4.pkl"
        if ap.exists():
            with open(ap, "rb") as fh:
                d = pickle.load(fh)
            if d.get("world_to_final") is not None:
                from lib.mllm.data.geometry import make_world_to_final
                w2f = d["world_to_final"]
                self.T = make_world_to_final(w2f["A_world_to_final"], w2f["origin_world"])
            else:
                self.T = self._recover_transform(d, video)

    def _recover_transform(self, d: dict, video: str) -> np.ndarray:
        keys = sorted(d["frames"].keys())
        first = keys[0].split("/")[-1]
        im = Image.open(Path(FRAMES_ROOT) / f"{video}.mp4" / first)
        W0, H0 = im.size
        H, W = self.points.shape[1:3]
        sx, sy = W / W0, H / H0
        A, B = [], []
        for key in keys:
            k = self.pi3_index_for_frame(int(key.split("/")[-1].split(".")[0]))
            if k is None:
                continue
            fr = d["frames"][key]
            for o in list(fr.get("object_info_list", [])) + [fr.get("person_info", {})]:
                if o.get("bbox_2d") is None or o.get("corners_final") is None:
                    continue
                bb = np.asarray(o["bbox_2d"], float).reshape(-1)[:4] * [sx, sy, sx, sy]
                x0, y0, x1, y1 = [int(round(v)) for v in bb]
                x0, x1, y0, y1 = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                pts = np.asarray(self.points[k, y0:y1, x0:x1]).reshape(-1, 3)
                cf = np.asarray(self.conf[k, y0:y1, x0:x1])[..., 0].reshape(-1)
                pts = pts[np.isfinite(pts).all(1) & (cf > np.percentile(cf, 30))]
                if len(pts) < 50:
                    continue
                A.append(np.median(pts, 0)); B.append(np.asarray(o["corners_final"], float).mean(0))
        A, B = np.array(A), np.array(B)
        if len(A) < 4:
            print("  [geometry] too few box correspondences to recover the canonical frame; using the π³ world frame")
            return np.eye(4)

        def kabsch(a, b):
            ca, cb = a.mean(0), b.mean(0)
            U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
            D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ D @ U.T
            return R, cb - R @ ca
        R, t = kabsch(A, B)
        res = np.linalg.norm(A @ R.T + t - B, axis=1)
        inl = res <= np.percentile(res, 60)
        R, t = kabsch(A[inl], B[inl])
        res = np.linalg.norm(A @ R.T + t - B, axis=1)
        print(f"  [geometry] canonical frame recovered from {len(A)} box centroids: median residual {np.median(res):.3f} m")
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return T

    def pi3_index_for_frame(self, fn: int):
        try:
            k = self.s.index(int(fn)) - self.start
        except ValueError:
            return None
        return k if 0 <= k < self.points.shape[0] else None

    def points_final(self, k: int, conf_min: float = 0.0):
        pts = np.asarray(self.points[k], dtype=np.float32)
        conf = np.asarray(self.conf[k])[..., 0]
        mask = np.isfinite(pts).all(-1) & (conf > conf_min) & (np.abs(pts).sum(-1) > 0)
        return (pts @ self.T[:3, :3].T + self.T[:3, 3]).astype(np.float32), mask

    def image(self, k: int) -> np.ndarray:
        return (np.asarray(self.images[k]) * 255.0).clip(0, 255).astype(np.uint8)

    def camera_pose_final_for_pi3(self, k: int) -> np.ndarray:
        return self.T @ np.asarray(self.poses[k], dtype=np.float64)


_LEGACY_CACHE: Dict[str, "LegacyScene"] = {}


def legacy_scene(video: str, frames: List[str], annot_dir: str) -> "LegacyScene":
    if video not in _LEGACY_CACHE:
        _LEGACY_CACHE[video] = LegacyScene(video, frames, annot_dir)
    return _LEGACY_CACHE[video]


def load_wb_video(video: str, frames: Optional[List[str]] = None, annot_dir: str = "world4d_rel_annotations"):
    from lib.mllm.core.config_loader import load_config
    from lib.mllm.data.worldbbox import WorldBBoxTestSet
    cfg = load_config(str(REPO / "configs/mllm/server.yaml"))
    ts = WorldBBoxTestSet(cfg)
    if video in ts:
        return ts.load(video)
    print(f"  [geometry] {video} is not in the WorldBBox split: reading the π³ scene directly")
    return legacy_scene(video, frames or [], annot_dir)


def recover_camera_poses(video: str, frames: List[str], annot_dir: str) -> Optional[np.ndarray]:
    """(T, 4, 4) camera->canonical poses for the annotated frames of a legacy
    video: the π³ camera poses carried into the recovered canonical frame."""
    try:
        sc = legacy_scene(video, frames, annot_dir)
    except Exception as e:                                   # noqa: BLE001
        print(f"  [poses] π³ reconstruction unavailable: {e!r}")
        return None
    out = []
    for f in frames:
        k = sc.pi3_index_for_frame(int(Path(f).stem))
        if k is None:
            print(f"  [poses] frame {f} not in the π³ clip; poses not recovered")
            return None
        out.append(sc.camera_pose_final_for_pi3(k))
    return np.stack(out).astype(np.float32)


def fit_focal(local_pts: np.ndarray) -> Tuple[float, float, float]:
    """Pinhole focal length (px) that maps π³'s own camera-frame points back onto
    their pixel grid, principal point at the image centre: least squares over
    u - cx = f x / z and v - cy = f y / z.  π³ ships no intrinsics."""
    H, W = local_pts.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    v, u = np.mgrid[0:H, 0:W]
    x, y, z = local_pts[..., 0], local_pts[..., 1], local_pts[..., 2]
    ok = np.isfinite(z) & (z > 1e-3)
    zs = np.where(ok, z, 1.0)
    a = np.concatenate([(x / zs)[ok], (y / zs)[ok]])
    b = np.concatenate([(u + 0.5 - cx)[ok], (v + 0.5 - cy)[ok]])
    f = float((a * b).sum() / max((a * a).sum(), 1e-9))
    return f, cx, cy


def project(C: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    z = np.clip(C[..., 2], 1e-3, None)
    return np.stack([f * C[..., 0] / z + cx, f * C[..., 1] / z + cy], -1)


def to_final(C: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(..., 3) camera-frame points -> canonical frame with the 4x4 camera->final pose P."""
    return C @ P[:3, :3].T + P[:3, 3]


def style_3d(ax):
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.06, 0.066, 0.078, 1.0))
        axis.line.set_color(RULE_3D)
        axis.set_tick_params(colors=MUTED, labelsize=6)
    ax.grid(False)


RULE_3D = "#3a3f47"


def to_camera(C: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(..., 3) canonical-frame points -> camera frame, P being the 4x4 camera->canonical pose."""
    return (C - P[:3, 3]) @ P[:3, :3]


def render_obb(out: Path, video: str, frames: List[str], keys: List[int], corners: np.ndarray, poses: np.ndarray,
               valid: np.ndarray, vis: np.ndarray, labels: List[str], tracked: int, f: float, cx: float, cy: float,
               pi3_wh: Tuple[int, int]):
    """obb_<k>.png: the input 3-D OBB corners (canonical frame) of every valid slot,
    moved into the frame's camera with the annotation pose and projected onto the raw
    frame (12 edges).  Visible objects solid, unseen ones dashed: an unseen slot still
    carries corners (the scaffold's geometry) although it has no 2-D box."""
    for k, t in enumerate(keys):
        img = raw_frame(video, frames[t])
        Hr, Wr = img.shape[:2]
        sx, sy = Wr / pi3_wh[0], Hr / pi3_wh[1]
        fig = dark_fig(4.0, 4.0 * Hr / Wr)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img)
        for n in range(corners.shape[1]):
            if not valid[t, n] or not corners[t, n].any():
                continue
            C = to_camera(corners[t, n], poses[t])
            if (C[:, 2] <= 0.05).any():
                continue
            P = project(C, f, cx, cy) * np.array([sx, sy])
            seen = bool(vis[t, n])
            col = ORANGE if n == tracked else (TXT if labels[n] == "person" else BLUE)
            if not seen and n != tracked:
                col = MASKC
            for i, j in box_edges(C):
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=col, lw=1.6 if seen else 1.3,
                        ls="-" if seen else (0, (3, 2)), alpha=0.95)
            top = P[np.argmin(P[:, 1])]
            ax.text(float(np.clip(top[0], 4, Wr - 60)), float(np.clip(top[1] - 4, 12, Hr - 4)),
                    labels[n] + ("" if seen else " (unseen)"), color="black", fontsize=8, va="bottom",
                    bbox=dict(fc=col, ec="none", pad=1.2))
        ax.set_xlim(0, Wr); ax.set_ylim(Hr, 0)
        ax.text(0.02, 0.985, f"input OBB corners, projected (f = {f:.0f} px)   t = {t + 1}", color=TXT, fontsize=7.5,
                va="top", transform=ax.transAxes, bbox=dict(fc="#000000aa", ec="none", pad=2))
        save(fig, out / f"obb_{k}.png", dpi=150)


def render_scene3d(out: Path, vid, frames: List[str], keys: List[int], corners: np.ndarray, poses: np.ndarray,
                   valid: np.ndarray, vis: np.ndarray, labels: List[str], tracked: int, f: float, cx: float, cy: float,
                   stride: int = 3):
    """scene3d_<k>.png: the canonical (floor) frame at a key frame: π³ points of that
    frame coloured by the image, every slot's OBB (unseen dashed), the camera path
    up to t and the current camera frustum."""
    for k, t in enumerate(keys):
        fn = int(Path(frames[t]).stem)
        kk = vid.pi3_index_for_frame(fn)
        fig = dark_fig(4.4, 3.8)
        ax = fig.add_subplot(111, projection="3d")
        style_3d(ax)
        allp = []
        if kk is not None:
            pts, mask = vid.points_final(kk)
            img = vid.image(kk)
            pts, mask, img = pts[::stride, ::stride], mask[::stride, ::stride], img[::stride, ::stride]
            p = pts[mask]; c = img[mask] / 255.0
            if len(p) > 0:
                lo, hi = np.percentile(p, 2, 0), np.percentile(p, 98, 0)
                keep = np.all((p >= lo) & (p <= hi), 1)
                p, c = p[keep], c[keep]
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=1.0, alpha=0.6, linewidths=0, depthshade=False)
                allp.append(p)
        P = poses[t]
        for n in range(corners.shape[1]):
            if not valid[t, n] or not corners[t, n].any():
                continue
            C = corners[t, n]                                  # already in the canonical frame
            seen = bool(vis[t, n])
            col = ORANGE if n == tracked else (TXT if labels[n] == "person" else BLUE)
            if not seen and n != tracked:
                col = MASKC
            for i, j in box_edges(C):
                ax.plot([C[i, 0], C[j, 0]], [C[i, 1], C[j, 1]], [C[i, 2], C[j, 2]], color=col, lw=1.4,
                        ls="-" if seen else (0, (3, 2)))
            cc = C.mean(0)
            ax.text(cc[0], cc[1], C[:, 2].max() + 0.05, labels[n] + ("" if seen else " (unseen)"), color=col, fontsize=6.5)
            allp.append(C)
        cam = poses[:, :3, 3]
        ax.plot(cam[:t + 1, 0], cam[:t + 1, 1], cam[:t + 1, 2], color=GREEN, lw=1.4)
        ax.scatter(cam[t:t + 1, 0], cam[t:t + 1, 1], cam[t:t + 1, 2], color=GREEN, s=18)
        d = 0.35
        W, H = vid.pi3_size
        rays = np.array([[(u - cx) / f, (v - cy) / f, 1.0] for u, v in ((0, 0), (W, 0), (W, H), (0, H))]) * d
        fr = to_final(rays, P); o = P[:3, 3]
        for i in range(4):
            ax.plot([o[0], fr[i, 0]], [o[1], fr[i, 1]], [o[2], fr[i, 2]], color=GREEN, lw=0.9)
            j = (i + 1) % 4
            ax.plot([fr[i, 0], fr[j, 0]], [fr[i, 1], fr[j, 1]], [fr[i, 2], fr[j, 2]], color=GREEN, lw=0.9)
        allp.append(cam[:t + 1]); allp.append(fr)
        A = np.concatenate(allp)
        lo, hi = A.min(0) - 0.05, A.max(0) + 0.05
        rng = np.maximum(hi - lo, 0.2)
        ax.set_xlim(lo[0], lo[0] + rng[0]); ax.set_ylim(lo[1], lo[1] + rng[1]); ax.set_zlim(lo[2], lo[2] + rng[2])
        ax.set_box_aspect(tuple(rng / rng.max()))           # metric aspect, tight to the scene
        fwd = P[:3, 2]                                        # look along the camera's viewing direction, from behind and above
        ax.view_init(elev=24, azim=float(np.degrees(np.arctan2(-fwd[1], -fwd[0]))))
        ax.set_position([0.0, 0.0, 1.0, 0.94])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"canonical frame, t = {t + 1}: π³ points · OBB corners · camera path", fontsize=7.5,
                     color=TXT, loc="left")
        save(fig, out / f"scene3d_{k}.png", dpi=160)


def relative_motion(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-step |translation| (m) and rotation angle (deg) of P_t^-1 P_{t+1}: the
    quantities the ego-motion encoder is fed (as 6-D rotation + translation)."""
    T = poses.shape[0]
    dt, dr = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        rel = np.linalg.inv(poses[t - 1]) @ poses[t]
        dt[t] = np.linalg.norm(rel[:3, 3])
        dr[t] = np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3]) - 1) / 2, -1, 1)))
    return dt, dr


def render_camera_path(out: Path, poses: np.ndarray, keys: List[int]):
    """camera_path.png: bird's-eye camera trajectory in the canonical frame with the
    viewing direction at every frame, plus the per-step relative motion."""
    T = poses.shape[0]
    cam = poses[:, :3, 3]
    fwd = poses[:, :3, 2]
    span = float(max(np.ptp(cam[:, 0]), np.ptp(cam[:, 1]), 0.02))
    arrow = 0.25 * span
    fig = dark_fig(4.2, 3.6)
    ax = fig.add_axes([0.14, 0.42, 0.82, 0.5]); style_axis(ax)
    ax.plot(cam[:, 0], cam[:, 1], color=GREEN, lw=1.4)
    for t in range(T):
        ax.annotate("", xy=(cam[t, 0] + arrow * fwd[t, 0], cam[t, 1] + arrow * fwd[t, 1]), xytext=(cam[t, 0], cam[t, 1]),
                    arrowprops=dict(arrowstyle="->", color=GREEN if t in keys else MUTED, lw=1.2 if t in keys else 0.7))
    for t in keys:
        ax.scatter(cam[t, 0], cam[t, 1], color=ORANGE, s=22, zorder=5)
        ax.annotate(f"t = {t + 1}", (cam[t, 0], cam[t, 1]), xytext=(4, 4), textcoords="offset points", color=ORANGE, fontsize=7)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7)
    ax.margins(0.3)
    ax.set_title(f"camera poses, bird's-eye (arrow = viewing direction; path spans {span * 100:.0f} cm)", fontsize=7, loc="left")
    dt, dr = relative_motion(poses)
    ax2 = fig.add_axes([0.1, 0.08, 0.86, 0.22]); style_axis(ax2)
    ax2.bar(np.arange(1, T + 1), dt, color=GREEN, width=0.8)
    ax2.set_ylabel("|Δt| m", fontsize=7, color=GREEN)
    ax3 = ax2.twinx(); ax3.plot(np.arange(1, T + 1), dr, color=ORANGE, lw=1.2); ax3.set_ylabel("ΔR °", fontsize=7, color=ORANGE)
    ax3.tick_params(colors=MUTED, labelsize=6)
    for s in ax3.spines.values():
        s.set_color(MUTED)
    ax2.set_xlabel("frame  (per-step relative pose → ego-motion encoder)", fontsize=7)
    ax2.set_xlim(0.5, T + 0.5)
    save(fig, out / "camera_path.png")


def render_motion(out: Path, corners: np.ndarray, poses: np.ndarray, valid: np.ndarray, vis: np.ndarray,
                  labels: List[str], keys: List[int], tracked: int):
    """motion.png: bird's-eye trajectories of the object centres in the canonical
    frame (unseen stretches dashed): the motion encoder's velocity / acceleration input."""
    T, N = valid.shape
    fig = dark_fig(4.2, 3.6)
    ax = fig.add_axes([0.1, 0.1, 0.86, 0.82]); style_axis(ax)
    palette = [BLUE, VIOLET, TEAL, "#f9a8d4", "#facc15", "#fb7185", "#a3e635", "#38bdf8"]
    ci = 0
    for n in range(N):
        if not valid[:, n].any():
            continue
        col = ORANGE if n == tracked else (TXT if labels[n] == "person" else palette[ci % len(palette)])
        if labels[n] != "person" and n != tracked:
            ci += 1
        ctr = np.full((T, 2), np.nan)
        for t in range(T):
            if valid[t, n] and corners[t, n].any():
                ctr[t] = corners[t, n].mean(0)[:2]                 # canonical frame already
        for t in range(1, T):
            if np.isfinite(ctr[t]).all() and np.isfinite(ctr[t - 1]).all():
                ax.plot(ctr[t - 1:t + 1, 0], ctr[t - 1:t + 1, 1], color=col, lw=1.4,
                        ls="-" if (vis[t, n] and vis[t - 1, n]) else (0, (2, 2)))
        ok = np.isfinite(ctr).all(1)
        if ok.any():
            t0 = int(np.where(ok)[0][0]); t1 = int(np.where(ok)[0][-1])
            ax.scatter(ctr[t0, 0], ctr[t0, 1], color=col, s=14, marker="o")
            ax.scatter(ctr[t1, 0], ctr[t1, 1], color=col, s=26, marker="s")
            ax.annotate(labels[n], (ctr[t1, 0], ctr[t1, 1]), xytext=(4, 3 + 9 * (ci % 3)), textcoords="offset points",
                        color=col, fontsize=7)
        for t in keys:
            if np.isfinite(ctr[t]).all():
                ax.scatter(ctr[t, 0], ctr[t, 1], facecolors="none", edgecolors=TXT, s=40, lw=0.8)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7)
    ax.set_title("object centres over time, bird's-eye (○ start, □ end, key frames ringed; unseen dashed)",
                 fontsize=7, loc="left")
    save(fig, out / "motion.png")


def render_geometry(out: Path, video: str, frames: List[str], keys: List[int], corners: np.ndarray,
                    poses: Optional[np.ndarray], valid: np.ndarray, vis: np.ndarray, labels: List[str], tracked: int,
                    meta: Dict[str, Any]):
    vid = load_wb_video(video, frames, meta.get("annot_dir") or "world4d_rel_annotations")
    W, H = vid.pi3_size
    kk = vid.pi3_index_for_frame(int(Path(frames[keys[0]]).stem))
    if kk is not None:
        f, cx, cy = fit_focal(np.asarray(vid.pi3["local_points"][kk], dtype=np.float32))
    else:
        f, cx, cy = 1.1 * max(W, H), W / 2.0, H / 2.0
    meta["focal_px_fitted"] = f
    meta["pi3_size_wh"] = [int(W), int(H)]
    if poses is None:
        print("  [geometry] no camera poses in the item: geometry panels skipped")
        return
    render_obb(out, video, frames, keys, corners, poses, valid, vis, labels, tracked, f, cx, cy, (W, H))
    render_scene3d(out, vid, frames, keys, corners, poses, valid, vis, labels, tracked, f, cx, cy)
    render_camera_path(out, poses, keys)
    render_motion(out, corners, poses, valid, vis, labels, keys, tracked)


# ---------------------------------------------------------------------------
# the loss, pictorially: a training-mode pass with artificial masking, the
# reconstruction target, which loss each pair receives, and (WorldWise++) the
# Hungarian matching of free queries to ground-truth boxes
# ---------------------------------------------------------------------------
def masked_pass(model, conf, b, seed: int = 0):
    """One forward pass as at training time (artificial masking of visible tokens
    with the configured p_mask_visible) but with every dropout switched off, so the
    only difference from the inference pass is the masking.  No parameter changes."""
    import torch.nn as nn
    torch.manual_seed(seed)
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.MultiheadAttention)):
            m.eval()
    tok = model.scaffold_tokenizer
    ema_update = getattr(tok, "_update_ema_target", None)
    if ema_update is not None:                   # the training forward nudges the EMA target; keep the checkpoint's
        tok._update_ema_target = lambda: None
    kw = dict(visual_features_seq=b["visual_features"], corners_seq=b["corners"],
              valid_mask_seq=b["valid_mask"], visibility_mask_seq=b["visibility_mask"],
              person_idx_seq=b["person_idx"], object_idx_seq=b["object_idx"],
              pair_valid=b["pair_valid"], camera_pose_seq=b.get("camera_poses"),
              union_features_seq=b.get("union_features"),
              node_labels_seq=b.get("object_classes") if conf.mode == "predcls" else None,
              p_mask_visible=float(getattr(conf, "p_mask_visible", 0.3)))
    if conf.method_name == "worldwise_pp":
        kw.update(grid_dino_seq=b["grid_dino"], grid_pi3_seq=b["grid_pi3"], image_hw=b["image_hw"],
                  bboxes_2d_seq=b.get("bboxes_2d"))
    with torch.no_grad():
        out = model(**kw)
    if ema_update is not None:
        tok._update_ema_target = ema_update
    model.eval()
    return out


def render_train_mask(out: Path, vis: np.ndarray, valid: np.ndarray, art: np.ndarray, labels: List[str], keys: List[int],
                      tracked: int):
    """train_mask.png: the token grid of the training-mode pass: visible (filled),
    unseen (dashed) and artificially masked visible cells (hatched orange): these
    last ones are the simulated-unseen cells that receive clean supervision."""
    T, N = vis.shape
    rows = [n for n in range(N) if valid[:, n].any()]
    fig = dark_fig(max(4.0, 0.16 * T + 1.2), 0.26 * len(rows) + 0.7)
    ax = fig.add_axes([0.22, 0.18, 0.76, 0.78]); style_axis(ax, spine=BG)
    for r, n in enumerate(rows):
        y = len(rows) - 1 - r
        for t in range(T):
            if not valid[t, n]:
                continue
            if vis[t, n] and art[t, n]:
                ax.add_patch(Rectangle((t + 0.08, y + 0.08), 0.84, 0.84, fc="none", ec=ORANGE, hatch="////", lw=1.0))
            elif vis[t, n]:
                ax.add_patch(Rectangle((t + 0.08, y + 0.08), 0.84, 0.84, fc=ORANGE if n == tracked else BLUE, ec="none", alpha=0.9))
            else:
                ax.add_patch(Rectangle((t + 0.12, y + 0.12), 0.76, 0.76, fill=False, ec=MASKC, ls=(0, (2, 1.5)), lw=1))
    for t in keys:
        ax.add_patch(Rectangle((t, -0.35), 1, len(rows) + 0.35, fill=False, ec=TXT, lw=1.0, ls=":"))
    ax.set_xlim(0, T); ax.set_ylim(-0.4, len(rows))
    ax.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([labels[n] for n in rows], color=TXT, fontsize=8)
    ax.set_xticks([0.5, T - 0.5]); ax.set_xticklabels(["1", str(T)], fontsize=7)
    ax.set_title(f"training-mode pass: hatched = artificially masked visible token (p = 0.3), "
                 f"{int((art & vis & valid).sum())} of {int((vis & valid).sum())}", fontsize=7.5, loc="left")
    save(fig, out / "train_mask.png")


def render_recon_sim(out: Path, pred: np.ndarray, target: np.ndarray, vis: np.ndarray, valid: np.ndarray, art: np.ndarray,
                     labels: List[str]):
    """recon_sim.png: cosine similarity between the reconstruction head's output and
    the EMA target for every visible cell; the artificially masked cells (outlined)
    are the ones the reconstruction loss is about."""
    T, N, _ = pred.shape
    rows = [n for n in range(N) if valid[:, n].any()]
    num = (pred * target).sum(-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1) + 1e-8
    sim = num / den
    img = np.full((len(rows), T), np.nan)
    for r, n in enumerate(rows):
        for t in range(T):
            if valid[t, n] and vis[t, n]:
                img[r, t] = sim[t, n]
    fig = dark_fig(max(4.0, 0.16 * T + 1.2), 0.26 * len(rows) + 0.7)
    ax = fig.add_axes([0.22, 0.18, 0.70, 0.78]); style_axis(ax, spine=BG)
    cm = plt.get_cmap("viridis").copy(); cm.set_bad(BG)
    im = ax.imshow(np.ma.masked_invalid(img), cmap=cm, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                   extent=(0, T, 0, len(rows)))
    for r, n in enumerate(rows):
        for t in range(T):
            if valid[t, n] and vis[t, n] and art[t, n]:
                ax.add_patch(Rectangle((t + 0.06, len(rows) - 1 - r + 0.06), 0.88, 0.88, fill=False, ec=ORANGE, lw=1.2))
    ax.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([labels[n] for n in rows], color=TXT, fontsize=8)
    ax.set_xticks([0.5, T - 0.5]); ax.set_xticklabels(["1", str(T)], fontsize=7)
    ax.set_xlim(0, T); ax.set_ylim(0, len(rows))
    m_art = float(np.nanmean(sim[art & vis & valid])) if (art & vis & valid).any() else float("nan")
    m_vis = float(np.nanmean(sim[vis & valid & ~art])) if (vis & valid & ~art).any() else float("nan")
    ax.set_title(f"reconstruction vs EMA target, cosine: masked cells {m_art:.2f} · unmasked {m_vis:.2f}", fontsize=7.5, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(colors=MUTED, labelsize=6)
    save(fig, out / "recon_sim.png")


def render_loss_pairs(out: Path, ds, item, pred_m, art: np.ndarray, keys: List[int], labels: List[str], tracked: int):
    """loss_pairs_<k>.png: every person-object pair of a key frame with the loss it
    receives in the training-mode pass: visible -> scene-graph loss on the ground
    truth (with logit adjustment); artificially masked -> simulated-unseen loss on
    the same clean ground truth; truly unseen -> no direct edge supervision."""
    att_n, spa_n, con_n = predicate_names(ds)
    con = to_np(pred_m["contacting_distribution"]); spa = to_np(pred_m["spatial_distribution"])
    pidx = to_np(item["person_idx"]).astype(int); oidx = to_np(item["object_idx"]).astype(int)
    pv = to_np(item["pair_valid"]).astype(bool); vis = to_np(item["visibility_mask"]).astype(bool)
    g_con = to_np(item["gt_contacting"]); g_spa = to_np(item["gt_spatial"])
    for k, t in enumerate(keys):
        rows = [kk for kk in range(pidx.shape[1]) if pv[t, kk]]
        fig = dark_fig(6.8, 0.36 * max(len(rows), 1) + 0.6)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        y = 0.93
        ax.text(0.02, y, f"t = {t + 1}", color=TXT, fontsize=8, fontweight="bold", va="top", transform=ax.transAxes)
        for x, h in ((0.22, "token state"), (0.42, "loss applied"), (0.72, "pred (contacting)"), (0.87, "ground truth")):
            ax.text(x, y, h, color=MUTED, fontsize=7, va="top", transform=ax.transAxes)
        dy = 0.84 / max(len(rows), 1)
        for i, kk in enumerate(rows):
            o = oidx[t, kk]; yy = y - 0.1 - i * dy
            if vis[t, o] and art[t, o]:
                state, loss, col = "artificially masked", "ℒ simulated-unseen, clean GT", ORANGE
            elif vis[t, o]:
                state, loss, col = "visible", "ℒ sg  (τ = 0.5 logit adj.)", GREEN
            else:
                state, loss, col = "unseen", "none  (λ_vlm = 0)", MASKC
            ax.text(0.02, yy, f"person → {labels[o]}", color=ORANGE if o == tracked else TXT, fontsize=7.5, va="top", transform=ax.transAxes)
            ax.text(0.22, yy, state, color=col, fontsize=7.5, va="top", transform=ax.transAxes)
            ax.text(0.42, yy, loss, color=col, fontsize=7.5, va="top", transform=ax.transAxes)
            c = con_n[int(con[t, kk].argmax())]; gts = [con_n[j] for j in np.where(g_con[t, kk] > 0.5)[0]]
            ax.text(0.72, yy, c, color=GREEN if c in gts else RED, fontsize=7.5, va="top", transform=ax.transAxes)
            ax.text(0.87, yy, ", ".join(gts) if gts else "–", color=MUTED, fontsize=7, va="top", transform=ax.transAxes)
        save(fig, out / f"loss_pairs_{k}.png")


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (M, 4) and (K, 4) xyxy boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def render_det_match(out: Path, video: str, frames: List[str], keys: List[int], det: Dict[str, Any], item,
                     image_hw: Tuple[int, int], valid: np.ndarray, vis: np.ndarray, labels: List[str], classes: List[str],
                     o2m_iou: float = 0.5):
    """det_match_<k>.png: the detection loss's assignment at a key frame: ground-truth
    boxes (green), the free query Hungarian-matched to each (orange, line to its GT),
    the one-to-many auxiliary matches (IoU > 0.5, dotted) and the remaining confident
    queries (grey).  Cost = -p(class) + 5 L1(box); the corner term is omitted here."""
    from scipy.optimize import linear_sum_assignment
    H, W = image_hw
    logits = to_np(det["logits"]); xyxy = to_np(det["boxes_xyxy"])
    prob = torch.softmax(torch.from_numpy(logits), -1).numpy()
    gt = to_np(item["gt_bboxes_2d"]); cls = to_np(item["object_classes"]).astype(int)
    scale = np.array([W, H, W, H], dtype=np.float32)
    for k, t in enumerate(keys):
        img = raw_frame(video, frames[t]); Hi, Wi = img.shape[:2]
        s = np.array([Wi / W, Hi / H, Wi / W, Hi / H])
        g_idx = [n for n in range(gt.shape[1]) if valid[t, n] and vis[t, n] and gt[t, n].any()]
        G = gt[t, g_idx]; Q = xyxy[t]
        fig = dark_fig(4.0, 4.0 * Hi / Wi)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img)
        matched_q = set()
        if len(G):
            cost = np.zeros((len(Q), len(G)))
            for j, n in enumerate(g_idx):
                cost[:, j] = -prob[t, :, cls[t, n]] + 5.0 * np.abs(Q / scale - G[j] / scale).sum(-1)
            qi, gj = linear_sum_assignment(cost)
            iou = box_iou(Q, G)
            for j, n in enumerate(g_idx):
                b = G[j] * s
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=2.0, ec=GREEN))
                ax.text(b[0] + 2, b[1] + 2, f"GT {labels[n]}", color="black", fontsize=7, va="top", bbox=dict(fc=GREEN, ec="none", pad=1))
            for q, j in zip(qi, gj):
                matched_q.add(int(q)); b = Q[q] * s; g = G[j] * s
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.8, ec=ORANGE))
                ax.plot([(b[0] + b[2]) / 2, (g[0] + g[2]) / 2], [(b[1] + b[3]) / 2, (g[1] + g[3]) / 2], color=ORANGE, lw=1.0)
                p = prob[t, q, cls[t, g_idx[j]]]
                ax.text(b[2] - 2, b[3] - 2, f"q{q}  p={p:.2f}  IoU {iou[q, j]:.2f}", color="black", fontsize=6.5, va="bottom", ha="right",
                        bbox=dict(fc=ORANGE, ec="none", pad=1))
            for q in range(len(Q)):
                if q in matched_q:
                    continue
                j = int(iou[q].argmax())
                if iou[q, j] > o2m_iou:
                    matched_q.add(q); b = Q[q] * s
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.2, ec=ORANGE, ls=":"))
        p_obj = prob[t, :, :-1].max(-1)
        for q in np.argsort(-p_obj)[:6]:
            if q in matched_q or p_obj[q] < 0.3:
                continue
            b = Q[q] * s
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=0.9, ec=MUTED, alpha=0.7))
        ax.text(0.02, 0.985, f"detection loss assignment  t = {t + 1}\nGT (green) ← matched query (orange)\n"
                             f"one-to-many (dotted) · other confident (grey)", color=TXT, fontsize=6.5, va="top",
                transform=ax.transAxes, bbox=dict(fc="#000000aa", ec="none", pad=2), linespacing=1.3)
        ax.set_xlim(0, Wi); ax.set_ylim(Hi, 0)
        save(fig, out / f"det_match_{k}.png", dpi=150)


# ---------------------------------------------------------------------------
# per-method driver
# ---------------------------------------------------------------------------
def run_method(method: str, video: str, out_root: Path, device, ckpt: str, annot_dir: Optional[str] = None):
    out = out_root / method
    out.mkdir(parents=True, exist_ok=True)
    conf = load_wsgg_config(str(REPO / CELLS[method]))
    if annot_dir:
        conf.test_annot_dir = annot_dir                       # e.g. the legacy world4d_rel_annotations split
    ds = make_test_dataset(conf)
    item, video_key = load_item(ds, video)
    if item.get("camera_poses") is None:
        poses = recover_camera_poses(video, list(item["frame_names"]), annot_dir or "world4d_rel_annotations")
        if poses is not None:
            item["camera_poses"] = torch.from_numpy(poses)
            print(f"  [poses] annotation has no camera poses; {len(poses)} recovered from the π³ reconstruction")
    model = build_model(conf, ds, device)
    if ckpt in ("best", "best_model"):
        ckpt = os.path.join(conf.save_path, conf.experiment_name, "best_model.pth")
    ckpt_path = load_checkpoint(model, conf, ckpt, device)
    model.eval()
    cap = install_hooks(model, method)
    b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
    with torch.no_grad():
        pred = forward_all_frames(model, conf, b)
    cap.close()

    frames = list(item["frame_names"])
    T = int(item["T"])
    valid = to_np(item["valid_mask"]).astype(bool)
    vis = to_np(item["visibility_mask"]).astype(bool)
    classes = list(ds.object_classes)
    cls_idx = to_np(item["object_classes"]).astype(int)
    labels = []
    for n in range(valid.shape[1]):
        ts_ = np.where(valid[:, n])[0]
        labels.append(classes[cls_idx[ts_[0], n]] if len(ts_) else f"slot{n}")
    tracked, keys = pick_keyframes(vis, valid, labels)
    boxes = to_np(item["bboxes_2d"])
    ts = target_size(ds, video_key, frames[keys[0]])
    meta = {"video": video, "method": method, "checkpoint": ckpt_path, "T": T, "frames": frames,
            "keyframes": keys, "tracked_slot": tracked, "tracked_label": labels[tracked], "labels": labels,
            "target_size_wh": ts, "n_valid_slots": int(valid.any(0).sum()),
            "annot_dir": annot_dir or getattr(conf, "test_annot_dir", None)}
    print(f"[{method}] T={T} slots={meta['n_valid_slots']} tracked={labels[tracked]} keys={[k + 1 for k in keys]}")

    arrays: Dict[str, np.ndarray] = {
        "valid": valid, "vis": vis, "bboxes_2d": boxes, "corners": to_np(item["corners"]),
        "object_classes": cls_idx,
        "visual_in": to_np(cap.get("visual_in")), "visual_proj": to_np(cap.get("visual_proj")),
        "scaffold": to_np(cap.get("scaffold")), "is_masked": to_np(cap.get("is_masked")),
        "attention_distribution": to_np(pred["attention_distribution"]),
        "spatial_distribution": to_np(pred["spatial_distribution"]),
        "contacting_distribution": to_np(pred["contacting_distribution"]),
        "person_idx": to_np(item["person_idx"]), "object_idx": to_np(item["object_idx"]),
        "pair_valid": to_np(item["pair_valid"]),
    }

    # ---- shared panels ----
    render_frames(out, video, frames, keys, boxes, vis, valid, labels, ts, tracked)
    render_visibility(out, vis, valid, labels, keys, tracked)
    cam = item.get("camera_poses")
    try:
        render_geometry(out, video, frames, keys, to_np(item["corners"]), None if cam is None else to_np(cam),
                        valid, vis, labels, tracked, meta)
    except Exception as e:                                   # geometry panels are optional extras
        import traceback
        traceback.print_exc()
        print(f"[{method}] geometry panels skipped: {e!r}")
    basis = render_token_grid(out, "tokens_in", arrays["scaffold"], valid, vis, labels, None,
                              title="scaffold tokens (visible appearance / [MASK] + geometry)")
    preds = collect_preds(ds, item, pred, keys, labels)
    render_preds(out, preds, keys, labels[tracked])
    arrays["gt_bboxes_2d"] = to_np(item["gt_bboxes_2d"])
    # appearance input as a strip: WorldWise reads the decoded 1024-d vector, the
    # other two the 3072-d frozen-token vector; same panel, different content.
    render_token_grid(out, "appearance_in", arrays["visual_in"], valid & vis, vis, labels, None,
                      title=f"appearance input ({arrays['visual_in'].shape[-1]}-d), visible slots only", mark_masked=False)

    # ---- the loss, pictorially: one training-mode pass with artificial masking ----
    pred_m = masked_pass(model, conf, b)
    art = to_np(pred_m["artificially_masked"]).astype(bool)
    arrays.update({"train_artificially_masked": art,
                   "train_recon_pred": to_np(pred_m["reconstruction_predictions"]),
                   "train_recon_target": to_np(pred_m["reconstruction_targets"]),
                   "train_contacting_distribution": to_np(pred_m["contacting_distribution"])})
    render_train_mask(out, vis, valid, art, labels, keys, tracked)
    render_recon_sim(out, arrays["train_recon_pred"], arrays["train_recon_target"], vis, valid, art, labels)
    render_loss_pairs(out, ds, item, pred_m, art, keys, labels, tracked)

    if method in ("worldwise", "worldwise_plus"):
        completed = to_np(cap.get("completed")); enriched = to_np(cap.get("enriched"))
        arrays["completed"] = completed; arrays["enriched"] = enriched
        render_token_grid(out, "tokens_out", completed, valid, vis, labels, basis,
                          title="after associative retrieval (same PCA basis)")
        render_token_grid(out, "tokens_enriched", enriched, valid, vis, labels, None,
                          title="after inter-object transformer", mark_masked=False)
        L = len([k for k in cap.store if k.startswith("retr_attn_")])
        A = np.stack([to_np(cap.get(f"retr_attn_{i}")) for i in range(L)])         # (L, N, T, T)
        arrays["retriever_attn"] = A
        render_temporal_attention(out, "retriever_attn", A.mean(0)[tracked], vis[:, tracked], valid[:, tracked],
                                  labels[tracked], keys, "associative retriever, mean over layers")
        gs = model.gate_summary() if hasattr(model, "gate_summary") else None
        meta["gate_summary"] = gs
    else:
        Q = model.n_free_queries
        Hp, Wp = [int(v) for v in item["grid_hw"]]
        H, W = [int(v) for v in item["image_hw"]]
        meta.update({"grid_hw": [Hp, Wp], "image_hw": [H, W], "n_free_queries": Q})
        gd = to_np(item["grid_dino"]); gp = to_np(item["grid_pi3"])
        gate = torch.softmax(torch.from_numpy(to_np(cap.get("grid_gate_logits"))), -1).numpy()[..., 0].reshape(T, Hp, Wp)
        mem = to_np(cap.get("memory")).reshape(T, Hp, Wp, -1)
        arrays.update({"grid_gate_dino": gate, "memory": mem[keys], "grid_dino_keys": gd[keys], "grid_pi3_keys": gp[keys]})
        render_grid_pca(out, "grid_dino", gd, keys, "DINOv3 L24n grid (PCA)", boxes, (H, W), vis, valid, labels, tracked)
        render_grid_pca(out, "grid_pi3", gp, keys, "π³ G14 grid (PCA)", boxes, (H, W), vis, valid, labels, tracked)
        render_grid_pca(out, "memory", mem, keys, "fused memory Mₜ (PCA)")
        render_heatmap_on_frame(out, "gate", gate, keys, video, frames, "gate: DINOv3 weight per cell", cmap="coolwarm", alpha=0.55)
        L = model.n_decoder_layers
        cross = np.stack([to_np(cap.get(f"cross_attn_{i}")) for i in range(L)])      # (L, T, Q+N, HW)
        temporal = np.stack([to_np(cap.get(f"temporal_attn_{i}")) for i in range(L)])  # (L, N, T, T)
        spatial = np.stack([to_np(cap.get(f"spatial_attn_{i}")) for i in range(L)])    # (L, T, heads, Q+N, Q+N)
        arrays["cross_attn_tracked"] = cross[:, :, Q + tracked].reshape(L, T, Hp, Wp)
        arrays["temporal_attn_tracked"] = temporal[:, tracked]
        arrays["spatial_attn_keys"] = spatial[:, keys].mean(2)
        cm = cross[:, :, Q + tracked].mean(0).reshape(T, Hp, Wp)
        render_heatmap_on_frame(out, "cross_attn", cm, keys, video, frames,
                                f"cross-attention of slot '{labels[tracked]}' to the grid, mean over layers",
                                box=boxes[:, tracked], box_hw=(H, W), box_label=labels[tracked], box_vis=vis[:, tracked])
        render_temporal_attention(out, "temporal_attn", temporal.mean(0)[tracked], vis[:, tracked], valid[:, tracked],
                                  labels[tracked], keys, "decoder temporal self-attention, mean over layers")
        render_spatial_attention(out, spatial.mean(0)[keys[1]].mean(0), valid[keys[1]], labels, Q, keys[1], tracked)
        dec = to_np(cap.get(f"dec_out_{L - 1}"))[:, Q:]
        arrays["decoder_slots"] = dec
        render_token_grid(out, "tokens_out", dec, valid, vis, labels, basis, title="after the entity decoder (same PCA basis)")
        det = pred["det"]
        arrays.update({"det_logits": to_np(det["logits"]), "det_boxes_xyxy": to_np(det["boxes_xyxy"]),
                       "slot_boxes": to_np(det["slot_boxes"]), "slot_corners": to_np(det["slot_corners"])})
        render_detections(out, video, frames, keys, det, (H, W), boxes, valid, vis, labels, classes, tracked)
        render_det_match(out, video, frames, keys, det, item, (H, W), valid, vis, labels, classes)
        # input corners and the slot refinement both live in the canonical frame (predcls)
        c_in = to_np(item["corners"]); c_ref = to_np(det["slot_corners"])
        render_bev(out, keys, c_in, c_ref, valid, vis, labels, tracked)
        meta["gate_summary"] = model.gate_summary()

    np.savez_compressed(out / "tensors.npz", **arrays)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(out / "preds.json", "w") as f:
        json.dump(preds, f, indent=1)
    print(f"[{method}] wrote {sorted(p.name for p in out.iterdir())}")
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="HI75B")
    ap.add_argument("--out-dir", default="/data3/rohith/ag/runs/intermediates")
    ap.add_argument("--methods", nargs="*", default=list(CELLS))
    ap.add_argument("--checkpoint", default="best", help="'best' or checkpoint_<N>")
    ap.add_argument("--annot-dir", default=None,
                    help="override the test annotation folder (e.g. world4d_rel_annotations for a legacy-split video)")
    args, _ = ap.parse_known_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir) / args.video
    for m in args.methods:
        run_method(m, args.video, out_root, device, args.checkpoint, args.annot_dir)


if __name__ == "__main__":
    main()
