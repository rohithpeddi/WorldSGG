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

The five adapted baselines (``w_sttran``, ``w_sttran_pp``, ``w_dsgdetr``,
``w_dsgdetr_pp``, ``w_usg``; resnet50 predcls) and the monocular 3-D detector
(``mono3d``) are dumped by the same script; see ``run_baseline`` / ``run_mono3d``
and ``scripts/paper_figures/BASELINE_PANELS.md`` for the panel contract.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
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
WW_METHODS = list(CELLS)
# The five adapted training baselines (resnet50 predcls, best checkpoints resolved
# from the configs exactly like the WorldWise cells).  ``mono3d`` is the monocular
# 3-D detector itself (see ``MONO3D``); it is not a WSGG cell.
BASELINE_CELLS = {
    "w_sttran": "configs/methods/predcls/w_sttran_predcls_resnet50.yaml",
    "w_sttran_pp": "configs/methods/predcls/w_sttran_pp_predcls_resnet50.yaml",
    "w_dsgdetr": "configs/methods/predcls/w_dsgdetr_predcls_resnet50.yaml",
    "w_dsgdetr_pp": "configs/methods/predcls/w_dsgdetr_pp_predcls_resnet50.yaml",
    "w_usg": "configs/methods/predcls/w_usg_predcls_resnet50.yaml",
}
CELLS.update(BASELINE_CELLS)
SPATIAL_TIERS = ("w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp")     # tiers with the ObjectSpatialEncoder
MOTION_TIERS = ("w_dsgdetr_pp",)                                  # tiers with the ObjectMotionEncoder
MONO3D = dict(
    checkpoint="/home/rxp190007/CODE/Scene4Cast/lib/detector/monocular3d/v1_dinov3l_separate/checkpoint_30/checkpoint_state.pth",
    model="v3l", head_3d_mode="separate", num_classes=37, pixel_limit=255000, patch_size=14,
    annotations="/data/rohith/ag/world_annotations/monocular3d_bbox_annotations",
    score_thr=0.3, n_proposals=50, image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225),
)
FRAMES_ROOT = "/data/rohith/ag/frames"

# Palettes shared with the SVG figure generators (scripts/paper_figures/dark).
# ``apply_theme`` rebinds the module-level colour names; every renderer looks
# them up at call time, so the theme can be chosen from the command line.
PALETTES = {
    "dark": dict(BG="#0f1114", TXT="#e6e8ec", MUTED="#9aa3b2", BLUE="#5aa2f2", VIOLET="#a78bfa", TEAL="#2dd4bf",
                 ORANGE="#ff8a3d", RED="#f87171", GREEN="#4ade80", MASKC="#64748b", YELLOW="#facc15",
                 PINK="#f9a8d4", ROSE="#fb7185", LIME="#a3e635", SKY="#38bdf8",
                 OVERLAY="#000000aa", RULE_3D="#3a3f47", PANE=(0.06, 0.066, 0.078, 1.0)),
    "light": dict(BG="#ffffff", TXT="#151a22", MUTED="#4b5563", BLUE="#2c7be5", VIOLET="#7c4dff", TEAL="#0f9f93",
                  ORANGE="#ef6c1a", RED="#dc2626", GREEN="#16a34a", MASKC="#6b7280", YELLOW="#ca8a04",
                  PINK="#db2777", ROSE="#e11d48", LIME="#65a30d", SKY="#0284c7",
                  OVERLAY="#ffffffcc", RULE_3D="#c5cad3", PANE=(0.965, 0.97, 0.978, 1.0)),
}
THEME = "light"
CAPS = "title"          # "title" | "upper" | "none"
globals().update(PALETTES[THEME])


def apply_theme(name: str):
    """Rebind the palette names to the ``name`` theme (dark / light)."""
    global THEME
    THEME = name
    globals().update(PALETTES[name])


_WORD = re.compile(r"^[A-Za-z][A-Za-z\-'\u2019/]*$")
_LEAD = "([{\"'\u201c\u2018"
_TRAIL = ")]}\"'\u201d\u2019,.:;!?"
_UNITS = {"px", "cm", "mm", "m", "ms"}


def cap(s: str, mode: str = None) -> str:
    """Capitalise the first letter of every word (``title``) or upper-case the
    whole string (``upper``).  Words that are not plain words (formulas, symbols,
    subscripts, numbers, units, single-letter variables) are left as they are, so
    ``t = 12``, ``p(class)``, ``IoU`` and ``DINOv3-L`` survive."""
    mode = CAPS if mode is None else mode
    if not s or mode == "none":
        return s
    if mode == "upper":
        return s.upper()
    out = []
    for tok in re.split(r"(\s+)", s):
        if not tok or tok.isspace() or "$" in tok:
            out.append(tok)
            continue
        i = 0
        while i < len(tok) and tok[i] in _LEAD:
            i += 1
        core = tok[i:].rstrip(_TRAIL)
        if _WORD.match(core) and core not in _UNITS and (len(core) > 1 or (core in ("a", "i") and tok == core)):
            tok = tok[:i] + core[0].upper() + tok[i + 1:]
        out.append(tok)
    return "".join(out)


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


def capitalise_figure(fig):
    """Apply :func:`cap` to every text of ``fig``: titles, annotations, legends
    and the fixed tick labels (object names on the token grids)."""
    if CAPS == "none":
        return
    import matplotlib.text
    from matplotlib.ticker import FixedFormatter
    for t in fig.findobj(matplotlib.text.Text):
        try:
            t.set_text(cap(t.get_text()))
        except Exception:
            pass
    for ax in fig.axes:
        axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else [])
        for axis in axes:
            fmt = axis.get_major_formatter()
            if isinstance(fmt, FixedFormatter):
                fmt.seq = [cap(str(v)) for v in fmt.seq]


def save(fig, path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    capitalise_figure(fig)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def style_axis(ax, spine=None):
    spine = MUTED if spine is None else spine
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

    def hook_kwargs(self, module: torch.nn.Module, key: str, name: str, pos: Optional[int] = None):
        """Record the keyword argument ``name`` of a module call (the baselines call
        their tokenizer with keyword arguments only, so ``hook_input`` sees nothing)."""
        def h(_m, args, kwargs):
            v = kwargs.get(name, args[pos] if pos is not None and pos < len(args) else None)
            if v is not None:
                self.add(key, v)
        self._handles.append(module.register_forward_pre_hook(h, with_kwargs=True))

    def patch_method(self, obj, name: str, key: str):
        """Wrap the bound method ``obj.name`` so that its return value is recorded
        (used for the RPN's ``filter_proposals``, whose scores the detector drops)."""
        orig = getattr(obj, name)

        def fn(*args, **kwargs):
            out = orig(*args, **kwargs)
            self.add(key, out)
            return out
        setattr(obj, name, fn)
        self._handles.append(("attr", obj, name, orig))

    def close(self):
        for h in self._handles:
            if isinstance(h, tuple) and h[0] == "attr":
                setattr(h[1], h[2], h[3])
            elif isinstance(h, tuple):
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


KEYFRAMES: List[int] = []      # zero-based override from --keyframes (empty: automatic)


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
    if KEYFRAMES:
        return best, [min(T - 1, max(0, k)) for k in KEYFRAMES]
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
                    bbox=dict(fc=OVERLAY, ec=col, lw=1.2, pad=2, ls="--"))
        ax.text(6, H - 6, f"t = {t + 1} / {len(frames)}   ({frames[t]})", color=TXT, fontsize=8, va="bottom",
                bbox=dict(fc=OVERLAY, ec="none", pad=2))
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
                bbox=dict(fc=OVERLAY, ec="none", pad=2))
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
                    bbox=dict(fc=OVERLAY, ec=ORANGE, lw=1.2, pad=2, ls="--"))
        ax.text(0.02, 0.98, f"{title}  t = {t + 1}", color=TXT, fontsize=8, va="top", transform=ax.transAxes,
                bbox=dict(fc=OVERLAY, ec="none", pad=2))
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
                color=TXT, fontsize=6.5, va="top", transform=ax.transAxes, bbox=dict(fc=OVERLAY, ec="none", pad=2))
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
        axis.set_pane_color(PANE)
        axis.line.set_color(RULE_3D)
        axis.set_tick_params(colors=MUTED, labelsize=6)
    ax.grid(False)




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
                va="top", transform=ax.transAxes, bbox=dict(fc=OVERLAY, ec="none", pad=2))
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
    palette = [BLUE, VIOLET, TEAL, PINK, YELLOW, ROSE, LIME, SKY]
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
                    meta: Dict[str, Any], with_camera: bool = True, with_motion: bool = True):
    """obb_<k>, scene3d_<k>, camera_path and motion panels; the last two are only
    drawn for models that consume camera poses / object motion (``with_*``)."""
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
    if with_camera:
        render_camera_path(out, poses, keys)
    if with_motion:
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
                transform=ax.transAxes, bbox=dict(fc=OVERLAY, ec="none", pad=2), linespacing=1.3)
        ax.set_xlim(0, Wi); ax.set_ylim(Hi, 0)
        save(fig, out / f"det_match_{k}.png", dpi=150)


# ---------------------------------------------------------------------------
# adapted baselines: W-STTran / W-STTran++ / W-DSGDetr / W-DSGDetr++ / W-USG
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def slow_attention():
    """Disable PyTorch's fused transformer fast path for the duration of the
    block, so that every ``nn.MultiheadAttention`` inside an ``nn.TransformerEncoder``
    / ``nn.TransformerDecoder`` runs through its Python ``forward`` (where
    ``patch_mha_weights`` records the weights).  The numbers are the same, only
    the kernel differs; ``run_baseline`` checks this against a plain pass."""
    mha = getattr(torch.backends, "mha", None)
    if mha is not None and hasattr(mha, "set_fastpath_enabled"):
        prev = mha.get_fastpath_enabled()
        mha.set_fastpath_enabled(False)
        try:
            yield
        finally:
            mha.set_fastpath_enabled(prev)
    else:                                   # older torch: grad-enabled inputs also disable the fast path
        with torch.enable_grad():
            yield


def install_baseline_hooks(model, method: str) -> Capture:
    """Hooks for the five baselines.  Every module below is the real submodule
    of the loaded checkpoint; the attention weights are forced out of the
    ``nn.MultiheadAttention`` modules that the models otherwise discard."""
    cap = Capture()
    cap.hook_output(model.global_structural_encoder, "struct", pick=lambda o: o[0])       # (T, N, d_struct)
    cap.hook_kwargs(model.tokenizer, "buffer_in", "buffer_features", pos=1)                 # LKS buffer (T, N, 1024)
    cap.hook_kwargs(model.tokenizer, "staleness_in", "staleness", pos=4)                    # (T, N) long
    cap.hook_output(model.tokenizer, "tokens_in")                                            # (T, N, d_model)
    if hasattr(model, "object_spatial_encoder"):
        cap.hook_output(model.object_spatial_encoder, "spatial", pick=lambda o: o[1])      # (T, N, d_camera)
    if hasattr(model, "object_motion_encoder"):
        cap.hook_output(model.object_motion_encoder, "motion")                              # called twice: (T,N,d), (T-1,N,d)
        cap.hook_output(model.motion_fusion, "tokens_fused")
    if hasattr(model, "temporal_obj_encoder"):
        for i, layer in enumerate(model.temporal_obj_encoder.encoder.layers):
            cap.patch_mha_weights(layer.self_attn, f"temporal_obj_attn_{i}")               # (N, T, T)
        cap.hook_output(model.temporal_obj_encoder, "tokens_temporal")
    ctx = model.inter_object_encoder if hasattr(model, "inter_object_encoder") else model.object_context_encoder
    layers = ctx.transformer.layers if hasattr(ctx, "transformer") else ctx.encoder.layers
    for i, layer in enumerate(layers):
        cap.patch_mha_weights(layer.self_attn, f"inter_object_attn_{i}")                  # (T, N, N)
    cap.hook_output(ctx, "tokens_out")
    for i, layer in enumerate(model.rel_predictor.rel_transformer.layers):
        cap.patch_mha_weights(layer.self_attn, f"rel_attn_{i}")                            # (T, K, K)
    if hasattr(model, "temporal_edge_attn"):
        for i, layer in enumerate(model.temporal_edge_attn.encoder.layers):
            cap.patch_mha_weights(layer.self_attn, f"temporal_edge_attn_{i}")             # (pairs, T_max, T_max)
        cap.hook_input(model.temporal_edge_attn.temporal_pe, "tea_frame_idx")              # (pairs, T_max) frame ids
    if hasattr(model, "relation_decoder"):
        for i, layer in enumerate(model.relation_decoder.decoder.layers):
            cap.patch_mha_weights(layer.self_attn, f"usg_rel_self_attn_{i}")              # (T, K, K)
            cap.patch_mha_weights(layer.multihead_attn, f"usg_rel_cross_attn_{i}")        # (T, K, N)
    return cap


def lks_sources(vis: np.ndarray, valid: np.ndarray, sentinel: int = 1000):
    """NumPy replica of ``vectorized_lks_buffer``'s index arithmetic: for every
    cell the frame it copies from (``gather``), the staleness counter, the
    never-seen ("fog") mask and whether the copy came from the future.  The
    driver verifies it against the buffer the model actually consumed."""
    T, N = vis.shape
    over = vis & valid
    fid = np.repeat(np.arange(T)[:, None], N, 1)
    last_seen = np.maximum.accumulate(np.where(over, fid, -1), axis=0)
    past_st = fid - last_seen
    never_past = last_seen < 0
    next_seen = -np.maximum.accumulate(np.where(over, -fid, -(T + sentinel))[::-1], axis=0)[::-1]
    future_st = next_seen - fid
    never_future = next_seen > T - 1
    past_safe = np.where(never_past, sentinel, past_st)
    future_safe = np.where(never_future, sentinel, future_st)
    use_future = future_safe < past_safe
    gather = np.clip(np.where(use_future, next_seen, last_seen), 0, T - 1)
    staleness = np.maximum(np.where(use_future, future_safe, past_safe), 0)
    return gather, staleness, never_past & never_future, use_future


def render_lks_buffer(out: Path, vis: np.ndarray, valid: np.ndarray, labels: List[str], gather: np.ndarray,
                      never: np.ndarray, keys: List[int], tracked: int):
    """lks_buffer.png: the zero-order-hold memory on the slot x frame grid.
    Visible cells hold their own ROI features (filled); an unseen cell holds a
    copy of the nearest visible frame's features, coloured by the direction of
    the copy (teal = held from the past, violet = borrowed from the future) and
    printed with the source frame number; never-seen cells are fog (zero
    features, staleness sentinel 1000).  The tracked row also shows the copy arrows."""
    T, N = vis.shape
    rows = [n for n in range(N) if valid[:, n].any()]
    fig = dark_fig(max(4.8, 0.2 * T + 1.4), 0.3 * len(rows) + 1.05)
    ax = fig.add_axes([0.2, 0.15, 0.78, 0.7])
    style_axis(ax, spine=BG)
    for r, n in enumerate(rows):
        y = len(rows) - 1 - r
        for t in range(T):
            if not valid[t, n]:
                continue
            if vis[t, n]:
                ax.add_patch(Rectangle((t + 0.06, y + 0.06), 0.88, 0.88, fc=ORANGE if n == tracked else BLUE, ec="none"))
            elif never[t, n]:
                ax.add_patch(Rectangle((t + 0.06, y + 0.06), 0.88, 0.88, fc="none", ec=MASKC, hatch="xxxx", lw=0.5))
            else:
                src = int(gather[t, n])
                col = TEAL if src < t else VIOLET
                ax.add_patch(Rectangle((t + 0.06, y + 0.06), 0.88, 0.88, fc=col, ec="none", alpha=0.3))
                ax.add_patch(Rectangle((t + 0.06, y + 0.06), 0.88, 0.88, fill=False, ec=col, ls=(0, (2, 1.5)), lw=0.8))
                ax.text(t + 0.5, y + 0.5, str(src + 1), color=TXT, fontsize=5.5, ha="center", va="center")
    if tracked in rows:
        y = len(rows) - 1 - rows.index(tracked)
        for t in range(T):
            if valid[t, tracked] and not vis[t, tracked] and not never[t, tracked]:
                src = int(gather[t, tracked])
                rad = -min(0.45, 0.9 / max(abs(t - src), 1))          # arc height ≈ half a cell, whatever the span
                rad = rad if src < t else -rad
                ax.annotate("", xy=(t + 0.5, y + 0.96), xytext=(src + 0.5, y + 0.96),
                            arrowprops=dict(arrowstyle="->", color=TXT, lw=0.7, shrinkA=0, shrinkB=0,
                                            connectionstyle=f"arc3,rad={rad}"))
    for t in keys:
        ax.add_patch(Rectangle((t, -0.35), 1, len(rows) + 0.35, fill=False, ec=TXT, lw=1.0, ls=":"))
    ax.set_xlim(0, T); ax.set_ylim(-0.4, len(rows) + 0.5)
    ax.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([labels[n] for n in rows], color=TXT, fontsize=8)
    ax.set_xticks([0.5] + [t + 0.5 for t in keys] + [T - 0.5])
    ax.set_xticklabels(["1"] + [str(t + 1) for t in keys] + [str(T)], fontsize=7)
    ax.set_xlabel("frame", fontsize=8)
    n_copy = int((valid & ~vis & ~never).sum()); n_fog = int((valid & never).sum())
    ax.set_title(f"LKS buffer (zero-order hold): filled = own ROI features · number = frame copied from "
                 f"(teal: held from the past, violet: from the future) · hatched = fog, never seen\n"
                 f"{n_copy} copied cells, {n_fog} fog cells of {int(valid.sum())}", fontsize=7, loc="left")
    save(fig, out / "lks_buffer.png")


def render_staleness(out: Path, staleness: np.ndarray, valid: np.ndarray, labels: List[str], keys: List[int],
                     tracked: int, sentinel: int = 1000):
    """staleness.png: the per-slot staleness counter (frames to the nearest
    visible frame; the fixed sentinel 1000 for never-seen slots) over time on a
    log axis; log(1 + staleness) is the scalar the tokenizer receives."""
    T, N = staleness.shape
    rows = [n for n in range(N) if valid[:, n].any()]
    fig = dark_fig(4.6, 2.7)
    ax = fig.add_axes([0.14, 0.2, 0.83, 0.66]); style_axis(ax)
    palette = [BLUE, VIOLET, TEAL, PINK, YELLOW, ROSE, LIME, SKY]
    ci = 0
    for n in rows:
        s = staleness[:, n].astype(float) + 1.0
        s[~valid[:, n]] = np.nan
        if n == tracked:
            col, lw, z = ORANGE, 2.0, 5
        elif labels[n] == "person":
            col, lw, z = TXT, 1.0, 3
        else:
            col, lw, z = palette[ci % len(palette)], 1.1, 3
            ci += 1
        ax.plot(np.arange(1, T + 1), s, color=col, lw=lw, zorder=z, marker="o", ms=2.2, label=labels[n])
    ax.set_yscale("log")
    ax.axhline(sentinel + 1, color=MASKC, ls=":", lw=0.9)
    ax.text(T + 0.3, sentinel + 1, "never seen (sentinel 1000)", color=MASKC, fontsize=6, va="center", ha="right")
    for t in keys:
        ax.axvline(t + 1, color=TXT, ls=":", lw=0.8)
    ax.set_xlim(0.5, T + 0.5)
    ax.set_xlabel("frame", fontsize=7); ax.set_ylabel("staleness + 1  (log axis)", fontsize=7)
    leg = ax.legend(fontsize=6, ncol=min(len(rows), 4), loc="upper left", frameon=False, handlelength=1.2)
    for t_ in leg.get_texts():
        t_.set_color(TXT)
    ax.set_title("staleness counter per slot → log(1 + staleness) enters the LKS tokenizer", fontsize=7.5, loc="left")
    save(fig, out / "staleness.png")


def render_attention_matrix(out: Path, name: str, A: np.ndarray, row_names: List[str], col_names: List[str],
                            title: str, row_hl: Optional[List[bool]] = None, col_hl: Optional[List[bool]] = None,
                            xlabel: str = "key", ylabel: str = "query", mark: Optional[List[Tuple[int, int]]] = None):
    """<name>.png: an attention (or similarity) matrix with named rows / columns;
    highlighted names in orange (the tracked object); ``mark`` outlines cells."""
    R, C = A.shape
    w = 3.2 + 0.06 * C; h = 2.6 + 0.12 * R
    fig = dark_fig(w, h)
    ax = fig.add_axes([0.34 * 3.2 / w, 0.28 * 2.6 / h, 1 - 0.4 * 3.2 / w, 1 - 0.55 * 2.6 / h])
    style_axis(ax)
    vmax = float(np.nanmax(A)) if np.isfinite(A).any() else 1.0
    im = ax.imshow(A, cmap="magma", interpolation="nearest", vmin=min(0.0, float(np.nanmin(A))), vmax=max(vmax, 1e-6))
    ax.set_xticks(range(C)); ax.set_xticklabels(col_names, rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(R)); ax.set_yticklabels(row_names, fontsize=6.5)
    for lab, hl in zip(ax.get_xticklabels(), col_hl or [False] * C):
        lab.set_color(ORANGE if hl else TXT)
    for lab, hl in zip(ax.get_yticklabels(), row_hl or [False] * R):
        lab.set_color(ORANGE if hl else TXT)
    for (i, j) in (mark or []):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=GREEN, lw=1.6))
    ax.set_xlabel(xlabel, fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=7.5, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(colors=MUTED, labelsize=6)
    save(fig, out / f"{name}.png")


def pair_names(pidx: np.ndarray, oidx: np.ndarray, pv: np.ndarray, labels: List[str], t: int):
    ks = [k for k in range(pv.shape[1]) if pv[t, k]]
    return ks, [f"{labels[pidx[t, k]]} → {labels[oidx[t, k]]}" for k in ks]


def temporal_edge_matrix(A: np.ndarray, frame_idx: np.ndarray, pidx: np.ndarray, oidx: np.ndarray, pv: np.ndarray,
                         person: int, obj: int, T: int) -> Optional[np.ndarray]:
    """The (T, T) attention of one (person, object) pair inside the temporal edge
    attention's grouped batch ``A`` (pairs, T_max, T_max).  Groups are the sorted
    unique pair keys ``p * max_N + o`` (as the module builds them); the positions
    of a group map back to frames through the temporal-PE input ``frame_idx``,
    and padded key positions are the zero columns."""
    max_n = int(max(pidx.max(), oidx.max())) + 1
    keys_valid = (pidx * max_n + oidx)[pv]
    uniq = np.unique(keys_valid)
    key = person * max_n + obj
    if key not in uniq:
        return None
    g = int(np.where(uniq == key)[0][0])
    Ag = A[g]
    pos = np.where(Ag.sum(0) > 1e-6)[0]                      # valid key positions (padded ones are masked to 0)
    M = np.full((T, T), np.nan)
    for p in pos:
        for q in pos:
            M[int(frame_idx[g, p]), int(frame_idx[g, q])] = Ag[p, q]
    return M


def baseline_pair_losses(pred: Dict[str, torch.Tensor], b: Dict[str, Any], lambda_vlm: float, eps: float):
    """Per-pair terms of ``LKSLoss`` (the bucketed noisy-label loss every baseline
    trains with), reproduced term by term: visible pairs (both endpoints in view)
    get a clean CE (attention) + BCE (spatial, contacting) at weight 1; pairs with
    an unseen endpoint get the VLM pseudo-label with label smoothing eps (a KL
    against the smoothed target for the single-label head) at weight lambda_vlm.
    Every term is divided by the shared denominator N (all valid pairs), so the
    values are exactly the pair's contribution to the batch loss.  Returns
    (T, K) arrays and the bucket id (0 vis-vis, 1 vis-unseen, 2 unseen-unseen)."""
    import torch.nn.functional as F
    from lib.supervised.components import LabelSmoother
    pv = b["pair_valid"].bool()
    vis = b["visibility_mask"].bool()
    pidx, oidx = b["person_idx"].long(), b["object_idx"].long()
    T, K = pv.shape
    N = float(pv.sum().item())
    att_l, spa_l, con_l = pred["attention_logits"], pred["spatial_logits"], pred["contacting_logits"]
    g_att, g_spa, g_con = b["gt_attention"].long(), b["gt_spatial"].float(), b["gt_contacting"].float()
    sm = LabelSmoother(epsilon=eps) if eps > 0 else None
    p_vis = vis.gather(1, pidx.clamp(0, vis.shape[1] - 1)); o_vis = vis.gather(1, oidx.clamp(0, vis.shape[1] - 1))
    bucket = torch.full_like(pidx, 2)
    bucket[p_vis ^ o_vis] = 1
    bucket[p_vis & o_vis] = 0
    out = {k: np.zeros((T, K)) for k in ("att", "spa", "con")}
    out["bucket"] = to_np(bucket).astype(int); out["weight"] = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            if not pv[t, k]:
                continue
            a, s, c = att_l[t, k][None], spa_l[t, k][None], con_l[t, k][None]
            ga, gs, gc = g_att[t, k][None], g_spa[t, k][None], g_con[t, k][None]
            if bucket[t, k] == 0:
                w = 1.0
                la = F.cross_entropy(a, ga, reduction="sum")
            else:
                w = lambda_vlm
                if sm is not None:
                    gs, gc = sm.smooth_bce_target(gs), sm.smooth_bce_target(gc)
                    la = F.kl_div(F.log_softmax(a, -1).clamp(min=-100), sm.smooth_ce_target(ga, a.shape[-1]), reduction="sum")
                else:
                    la = F.cross_entropy(a, ga, reduction="sum")
            ls = F.binary_cross_entropy_with_logits(s, gs, reduction="sum")
            lc = F.binary_cross_entropy_with_logits(c, gc, reduction="sum")
            out["att"][t, k] = float(la) * w / N; out["spa"][t, k] = float(ls) * w / N; out["con"][t, k] = float(lc) * w / N
            out["weight"][t, k] = w
    out["N"] = N
    return out


def verify_pair_losses(terms: Dict[str, np.ndarray], pred, b, conf, method: str) -> Dict[str, float]:
    """Call the real loss class the trainer uses and compare its bucket totals
    with the sums of the reproduced per-pair terms."""
    from constants import Constants as const
    lam, eps = float(conf.lambda_vlm), float(conf.label_smoothing_vlm)
    if method == "w_usg":
        from lib.supervised.baselines.w_usg.loss import WUSGLoss
        fn = WUSGLoss(lambda_vlm=lam, label_smoothing=eps, mode=conf.mode, lambda_align=float(getattr(conf, "lambda_align", 0.1)))
    else:
        from lib.supervised.baselines.lks_buffer.loss import LKSLoss
        fn = LKSLoss(lambda_vlm=lam, label_smoothing=eps, mode=conf.mode)
    with torch.no_grad():
        L = fn(predictions=pred, gt_attention=b["gt_attention"], gt_spatial=b["gt_spatial"], gt_contacting=b["gt_contacting"],
               pair_valid=b["pair_valid"], visibility_mask=b["visibility_mask"], person_idx=b["person_idx"],
               object_idx=b["object_idx"], valid_mask=b.get("valid_mask"), gt_node_labels=b.get("object_classes"))
    rep = {"att": float(terms["att"].sum()), "spa": float(terms["spa"].sum()), "con": float(terms["con"].sum())}
    real = {"att": float(L[const.ATTENTION_RELATION_LOSS]), "spa": float(L[const.SPATIAL_RELATION_LOSS]),
            "con": float(L[const.CONTACTING_RELATION_LOSS])}
    res = {"loss_class": type(fn).__name__, "lambda_vlm": lam, "label_smoothing": eps,
           "real": real, "reproduced": rep, "total": float(L["total"]),
           "max_abs_diff": max(abs(rep[k] - real[k]) for k in rep),
           "buckets": {k: float(L[k]) for k in L if any(k.startswith(p) for p in ("vis_vis", "vis_unseen", "unseen_unseen"))}}
    if "alignment_loss" in L:
        res["alignment_loss"] = float(L["alignment_loss"])
    return res


def render_loss_pairs_baseline(out: Path, ds, item, pred, terms: Dict[str, np.ndarray], keys: List[int], labels: List[str],
                               tracked: int, lam: float, eps: float):
    """loss_pairs_<k>.png: every person-object pair of a key frame with the bucket
    the noisy-label loss puts it in, the weight, the three per-head terms it
    contributes (already /N), the predicted top contacting predicate and the label
    it is scored against (clean GT for visible pairs, VLM pseudo-label otherwise)."""
    att_n, spa_n, con_n = predicate_names(ds)
    con = to_np(pred["contacting_distribution"])
    pidx = to_np(item["person_idx"]).astype(int); oidx = to_np(item["object_idx"]).astype(int)
    pv = to_np(item["pair_valid"]).astype(bool)
    g_con = to_np(item["gt_contacting"])
    names = {0: "visible", 1: "unseen (one endpoint)", 2: "unseen (both)"}
    cols = {0: GREEN, 1: MASKC, 2: MASKC}
    for k, t in enumerate(keys):
        rows = [kk for kk in range(pidx.shape[1]) if pv[t, kk]]
        fig = dark_fig(7.6, 0.36 * max(len(rows), 1) + 0.9)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.text(0.02, 0.97, f"t = {t + 1}   training loss without artificial masking: visible → clean CE / BCE (w = 1); "
                            f"unseen → VLM pseudo-label, λ_vlm = {lam:g}, smoothing ε = {eps:g}",
                color=TXT, fontsize=7, va="top", transform=ax.transAxes)
        y = 0.97 - 0.9 / (len(rows) + 2.2)
        for x, h in ((0.02, "pair"), (0.2, "bucket"), (0.37, "w"), (0.43, "ℒ att"), (0.50, "ℒ spa"), (0.57, "ℒ con"),
                     (0.65, "pred (contacting)"), (0.82, "label")):
            ax.text(x, y, h, color=MUTED, fontsize=7, va="top", transform=ax.transAxes)
        dy = 0.86 / (len(rows) + 2.2)
        for i, kk in enumerate(rows):
            o = oidx[t, kk]; yy = y - (i + 1) * dy
            bkt = int(terms["bucket"][t, kk]); col = cols[bkt]
            ax.text(0.02, yy, f"person → {labels[o]}", color=ORANGE if o == tracked else TXT, fontsize=7.5, va="top", transform=ax.transAxes)
            ax.text(0.2, yy, names[bkt], color=col, fontsize=7.5, va="top", transform=ax.transAxes)
            ax.text(0.37, yy, f"{terms['weight'][t, kk]:g}", color=col, fontsize=7.5, va="top", transform=ax.transAxes)
            for x, key in ((0.43, "att"), (0.50, "spa"), (0.57, "con")):
                ax.text(x, yy, f"{terms[key][t, kk]:.3f}", color=TXT, fontsize=7.5, va="top", transform=ax.transAxes)
            c = con_n[int(con[t, kk].argmax())]; gts = [con_n[j] for j in np.where(g_con[t, kk] > 0.5)[0]]
            ax.text(0.65, yy, c, color=GREEN if c in gts else RED, fontsize=7.5, va="top", transform=ax.transAxes)
            tag = "GT: " if bkt == 0 else "VLM: "
            ax.text(0.82, yy, tag + (", ".join(gts) if gts else "–"), color=MUTED, fontsize=7, va="top", transform=ax.transAxes)
        save(fig, out / f"loss_pairs_{k}.png")


def render_align_logits(out: Path, logits_t: np.ndarray, temperature: float, valid_t: np.ndarray, cls_t: np.ndarray,
                        labels: List[str], classes: List[str], t: int, tracked: int):
    """align_logits.png (W-USG): cosine similarity between the projected object
    tokens and the frozen CLIP class embeddings at a key frame (the model's
    ``align_logits`` times the temperature), the GT class of every slot outlined."""
    slots = [n for n in range(len(valid_t)) if valid_t[n]]
    S = logits_t[slots] * temperature
    marks = [(i, int(cls_t[n])) for i, n in enumerate(slots)]
    render_attention_matrix(out, "align_logits", S, [labels[n] for n in slots], classes,
                            f"W-USG alignment: cos(object token, CLIP class), t = {t + 1} · GT outlined",
                            row_hl=[n == tracked for n in slots], xlabel="class", ylabel="object slot", mark=marks)


def run_baseline(method: str, video: str, out_root: Path, device, ckpt: str, annot_dir: Optional[str] = None):
    out = out_root / method
    out.mkdir(parents=True, exist_ok=True)
    conf = load_wsgg_config(str(REPO / CELLS[method]))
    if annot_dir:
        conf.test_annot_dir = annot_dir
    ds = make_test_dataset(conf)
    item, video_key = load_item(ds, video)
    if item.get("camera_poses") is None:
        poses = recover_camera_poses(video, list(item["frame_names"]), annot_dir or "world4d_rel_annotations")
        if poses is not None:
            item["camera_poses"] = torch.from_numpy(poses)
    model = build_model(conf, ds, device)
    if ckpt in ("best", "best_model"):
        ckpt = os.path.join(conf.save_path, conf.experiment_name, "best_model.pth")
    ckpt_path = load_checkpoint(model, conf, ckpt, device)
    model.eval()
    b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
    cap = install_baseline_hooks(model, method)
    with torch.no_grad(), slow_attention():
        pred = forward_all_frames(model, conf, b)
    cap.close()
    with torch.no_grad():                                     # plain pass: the hooks must not change the numbers
        plain = forward_all_frames(model, conf, b)
    hook_dev = max(float((pred[k] - plain[k]).abs().max()) for k in
                   ("attention_distribution", "spatial_distribution", "contacting_distribution"))
    missing = [k for k in ("struct", "buffer_in", "staleness_in", "tokens_in", "tokens_out", "inter_object_attn_0", "rel_attn_0")
               if not cap.has(k)]
    if missing:
        print(f"[{method}] WARNING: hooks captured nothing for {missing}")

    frames = list(item["frame_names"]); T = int(item["T"])
    valid = to_np(item["valid_mask"]).astype(bool); vis = to_np(item["visibility_mask"]).astype(bool)
    classes = list(ds.object_classes); cls_idx = to_np(item["object_classes"]).astype(int)
    labels = []
    for n in range(valid.shape[1]):
        ts_ = np.where(valid[:, n])[0]
        labels.append(classes[cls_idx[ts_[0], n]] if len(ts_) else f"slot{n}")
    tracked, keys = pick_keyframes(vis, valid, labels)
    boxes = to_np(item["bboxes_2d"]); ts = target_size(ds, video_key, frames[keys[0]])
    pidx = to_np(item["person_idx"]).astype(int); oidx = to_np(item["object_idx"]).astype(int)
    pv = to_np(item["pair_valid"]).astype(bool)
    meta: Dict[str, Any] = {"video": video, "method": method, "checkpoint": ckpt_path, "T": T, "frames": frames,
                            "keyframes": keys, "tracked_slot": tracked, "tracked_label": labels[tracked], "labels": labels,
                            "target_size_wh": ts, "n_valid_slots": int(valid.any(0).sum()),
                            "annot_dir": annot_dir or getattr(conf, "test_annot_dir", None),
                            "feature_model": getattr(conf, "feature_model", None),
                            "hooked_vs_plain_max_abs_diff": hook_dev, "hooks_missing": missing,
                            "training_pass": "no artificial masking: the inference pass with dropout off is the training forward",
                            "panels_skipped": {}}
    print(f"[{method}] T={T} slots={meta['n_valid_slots']} tracked={labels[tracked]} keys={[k + 1 for k in keys]} "
          f"hook deviation {hook_dev:.2e}")

    arrays: Dict[str, np.ndarray] = {
        "valid": valid, "vis": vis, "bboxes_2d": boxes, "corners": to_np(item["corners"]), "object_classes": cls_idx,
        "person_idx": pidx, "object_idx": oidx, "pair_valid": pv,
        "attention_distribution": to_np(pred["attention_distribution"]),
        "spatial_distribution": to_np(pred["spatial_distribution"]),
        "contacting_distribution": to_np(pred["contacting_distribution"]),
        "visual_in": to_np(item["visual_features"]), "gt_bboxes_2d": to_np(item["gt_bboxes_2d"]),
    }

    # ---- shared panels (frames, visibility, geometry, predictions) ----
    render_frames(out, video, frames, keys, boxes, vis, valid, labels, ts, tracked)
    render_visibility(out, vis, valid, labels, keys, tracked)
    cam = item.get("camera_poses")
    try:
        render_geometry(out, video, frames, keys, to_np(item["corners"]), None if cam is None else to_np(cam),
                        valid, vis, labels, tracked, meta, with_camera=method in SPATIAL_TIERS, with_motion=method in MOTION_TIERS)
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        meta["panels_skipped"]["geometry"] = repr(e)
    preds = collect_preds(ds, item, pred, keys, labels)
    render_preds(out, preds, keys, labels[tracked])

    # ---- the LKS memory: sources verified against the buffer the tokenizer received ----
    gather, stale_np, never, use_future = lks_sources(vis, valid)
    buf = to_np(cap.get("buffer_in")); stale = to_np(cap.get("staleness_in")).astype(int)
    raw = arrays["visual_in"]
    copy_ok = all(np.allclose(buf[t, n], raw[gather[t, n], n], atol=1e-5) for t in range(T) for n in range(valid.shape[1])
                  if valid[t, n] and not never[t, n])
    fog_ok = bool(np.all(buf[valid & never] == 0)) if (valid & never).any() else True
    meta["lks"] = {"staleness_matches": bool(np.array_equal(stale[valid], stale_np[valid])), "copies_match": bool(copy_ok),
                   "fog_zero": fog_ok, "n_copied": int((valid & ~vis & ~never).sum()), "n_fog": int((valid & never).sum()),
                   "tracked_sources": [None if not valid[t, tracked] else ("own" if vis[t, tracked] else ("fog" if never[t, tracked] else int(gather[t, tracked]) + 1)) for t in range(T)]}
    arrays.update({"lks_buffer": buf, "lks_staleness": stale, "lks_gather": gather, "lks_never": never})
    render_lks_buffer(out, vis, valid, labels, gather, never, keys, tracked)
    render_staleness(out, stale, valid, labels, keys, tracked)

    # ---- token grids ----
    render_token_grid(out, "appearance_in", raw, valid & vis, vis, labels, None,
                      title=f"appearance input ({raw.shape[-1]}-d resnet50 ROI features), visible slots only", mark_masked=False)
    struct = to_np(cap.get("struct")); arrays["struct_tokens"] = struct
    render_token_grid(out, "struct_tokens", struct, valid, vis, labels, None,
                      title="global structural encoder tokens (OBB corners → MLP)", mark_masked=False)
    tokens_in = to_np(cap.get("tokens_in")); arrays["tokens_in"] = tokens_in
    basis = render_token_grid(out, "tokens_in", tokens_in, valid, vis, labels, None,
                              title="LKS tokenizer output (geometry ⊕ buffered appearance ⊕ log staleness)")
    if cap.has("spatial"):
        sp = to_np(cap.get("spatial")); arrays["spatial_feats"] = sp
        render_token_grid(out, "spatial_feats", sp, valid, vis, labels, None,
                          title="object spatial encoder (camera-relative distance / azimuth → MLP)", mark_masked=False)
    if cap.has("motion"):
        m0, m1 = to_np(cap.get("motion", 0)), to_np(cap.get("motion", 1))
        mo = np.concatenate([m0[:1], m1], 0) if m1.shape[0] == T - 1 else m0
        arrays["motion_feats"] = mo
        render_token_grid(out, "motion_feats", mo, valid, vis, labels, None,
                          title="object motion encoder (world / camera-frame velocity → MLP; frame 1 = no-motion embedding)",
                          mark_masked=False)

    # ---- temporal object encoder (W-DSGDetr, W-DSGDetr++) ----
    Lt = len([k for k in cap.store if k.startswith("temporal_obj_attn_")])
    if Lt:
        A = np.stack([to_np(cap.get(f"temporal_obj_attn_{i}")) for i in range(Lt)])           # (L, N, T, T)
        arrays["temporal_obj_attn_tracked"] = A[:, tracked]
        render_temporal_attention(out, "temporal_obj_attn", A.mean(0)[tracked], vis[:, tracked], valid[:, tracked],
                                  labels[tracked], keys, "temporal object encoder self-attention, mean over layers")
        tt = to_np(cap.get("tokens_temporal")); arrays["tokens_temporal"] = tt
        render_token_grid(out, "tokens_temporal", tt, valid, vis, labels, basis,
                          title="after the temporal object encoder (same PCA basis)")

    # ---- inter-object transformer (W-USG: object context encoder) ----
    Li = len([k for k in cap.store if k.startswith("inter_object_attn_")])
    if Li:
        A = np.stack([to_np(cap.get(f"inter_object_attn_{i}")) for i in range(Li)]).mean(0)   # (T, N, N)
        arrays["inter_object_attn_keys"] = A[keys]
        what = "object context encoder (no 3-D PE)" if method == "w_usg" else "inter-object transformer"
        for k, t in enumerate(keys):
            slots = [n for n in range(valid.shape[1]) if valid[t, n]]
            names = [labels[n] + ("" if vis[t, n] else " (unseen)") for n in slots]
            render_attention_matrix(out, f"inter_object_attn_{k}", A[t][np.ix_(slots, slots)], names, names,
                                    f"{what} self-attention, mean over layers, t = {t + 1}",
                                    row_hl=[n == tracked for n in slots], col_hl=[n == tracked for n in slots])
    tok_out = to_np(cap.get("tokens_out")); arrays["tokens_out"] = tok_out
    render_token_grid(out, "tokens_out", tok_out, valid, vis, labels, basis,
                      title=("after the object context encoder" if method == "w_usg" else "after the inter-object transformer")
                      + " (same PCA basis)")

    # ---- relationship predictor pair self-attention ----
    Lr = len([k for k in cap.store if k.startswith("rel_attn_")])
    if Lr:
        A = np.stack([to_np(cap.get(f"rel_attn_{i}")) for i in range(Lr)]).mean(0)             # (T, K, K)
        arrays["rel_attn_keys"] = A[keys]
        for k, t in enumerate(keys):
            ks, names = pair_names(pidx, oidx, pv, labels, t)
            hl = [oidx[t, kk] == tracked for kk in ks]
            render_attention_matrix(out, f"rel_attn_{k}", A[t][np.ix_(ks, ks)], names, names,
                                    f"relationship predictor pair self-attention, mean over layers, t = {t + 1}",
                                    row_hl=hl, col_hl=hl, xlabel="key pair", ylabel="query pair")

    # ---- temporal edge attention of the tracked pair ----
    Le = len([k for k in cap.store if k.startswith("temporal_edge_attn_")])
    if Le:
        A = np.stack([to_np(cap.get(f"temporal_edge_attn_{i}")) for i in range(Le)]).mean(0)   # (pairs, T_max, T_max)
        fidx = to_np(cap.get("tea_frame_idx")).astype(int)
        persons = [n for n in range(valid.shape[1]) if labels[n] == "person"]
        M = None
        for p in persons:
            M = temporal_edge_matrix(A, fidx, pidx, oidx, pv, p, tracked, T)
            if M is not None:
                meta["tracked_pair"] = [p, tracked]
                break
        if M is None:
            meta["panels_skipped"]["temporal_edge_attn"] = "tracked pair not in the grouped batch"
        else:
            arrays["temporal_edge_attn_tracked"] = M
            pair_valid_row = np.array([bool(pv[t][(pidx[t] == p) & (oidx[t] == tracked)].any()) for t in range(T)])
            render_temporal_attention(out, "temporal_edge_attn", np.nan_to_num(M), vis[:, tracked], pair_valid_row,
                                      f"person → {labels[tracked]}", keys, "temporal edge attention, mean over layers")

    # ---- W-USG relation decoder and text alignment ----
    Lu = len([k for k in cap.store if k.startswith("usg_rel_self_attn_")])
    if Lu:
        t = keys[1]
        S = np.stack([to_np(cap.get(f"usg_rel_self_attn_{i}")) for i in range(Lu)]).mean(0)   # (T, K, K)
        X = np.stack([to_np(cap.get(f"usg_rel_cross_attn_{i}")) for i in range(Lu)]).mean(0)  # (T, K, N)
        arrays["usg_rel_self_attn_keys"] = S[keys]; arrays["usg_rel_cross_attn_keys"] = X[keys]
        ks, names = pair_names(pidx, oidx, pv, labels, t)
        slots = [n for n in range(valid.shape[1]) if valid[t, n]]
        hl = [oidx[t, kk] == tracked for kk in ks]
        render_attention_matrix(out, "usg_rel_self_attn_1", S[t][np.ix_(ks, ks)], names, names,
                                f"relation decoder self-attention over pair queries, mean over layers, t = {t + 1}",
                                row_hl=hl, col_hl=hl, xlabel="key pair", ylabel="query pair")
        render_attention_matrix(out, "usg_rel_cross_attn_1", X[t][np.ix_(ks, slots)], names,
                                [labels[n] + ("" if vis[t, n] else " (unseen)") for n in slots],
                                f"relation decoder cross-attention: pair query × object-token memory, t = {t + 1}",
                                row_hl=hl, col_hl=[n == tracked for n in slots], xlabel="object token (memory)", ylabel="pair query")
        if "align_logits" in pred:
            al = to_np(pred["align_logits"]); arrays["align_logits"] = al
            render_align_logits(out, al[t], float(model.align_temperature), valid[t], cls_idx[t], labels, classes, t, tracked)

    # ---- the loss, per pair: the baselines train on the very same forward (no masking) ----
    lam, eps = float(conf.lambda_vlm), float(conf.label_smoothing_vlm)
    terms = baseline_pair_losses(pred, b, lam, eps)
    meta["loss"] = verify_pair_losses(terms, pred, b, conf, method)
    meta["loss"]["per_pair_terms"] = "reproduced term by term from LKSLoss (CE / BCE / KL with LabelSmoother), divided by N; sums checked against the loss class"
    meta["loss"]["bucket_counts"] = {"vis_vis": int((terms["bucket"] == 0)[pv].sum()), "vis_unseen": int((terms["bucket"] == 1)[pv].sum()),
                                     "unseen_unseen": int((terms["bucket"] == 2)[pv].sum())}
    arrays.update({f"loss_{k}": v for k, v in terms.items() if isinstance(v, np.ndarray)})
    render_loss_pairs_baseline(out, ds, item, pred, terms, keys, labels, tracked, lam, eps)
    print(f"[{method}] loss check: real {meta['loss']['real']} vs reproduced {meta['loss']['reproduced']} "
          f"(max diff {meta['loss']['max_abs_diff']:.2e}); buckets {meta['loss']['bucket_counts']}")

    np.savez_compressed(out / "tensors.npz", **arrays)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(out / "preds.json", "w") as f:
        json.dump(preds, f, indent=1)
    print(f"[{method}] wrote {sorted(p.name for p in out.iterdir())}")
    del model
    torch.cuda.empty_cache()
    return frames, keys, tracked, labels, classes


# ---------------------------------------------------------------------------
# the monocular 3-D detector (DINOv3-L, separate 3-D head) on the key frames
# ---------------------------------------------------------------------------
def mono3d_target_size(orig_w: int, orig_h: int, pixel_limit: int, patch_size: int) -> Tuple[int, int]:
    """``ag_dataset_3d.ActionGenomeDataset3D._compute_target_size`` verbatim."""
    scale = math.sqrt(pixel_limit / (orig_w * orig_h)) if orig_w * orig_h > 0 else 1
    w_target, h_target = orig_w * scale, orig_h * scale
    k = round(w_target / patch_size); m = round(h_target / patch_size)
    while (k * patch_size) * (m * patch_size) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch_size, max(1, m) * patch_size


_MONO3D_LABEL_REMAP = {"closet": "closet/cabinet", "cup": "cup/glass/bottle", "paper": "paper/notebook",
                       "sofa": "sofa/couch", "phone": "phone/camera"}


def load_mono3d_annotation(video: str):
    """The per-video 3-D annotation the detector was trained on: camera-frame OBB
    corners per frame and the video's pinhole intrinsics."""
    from lib.detector.monocular3d.datasets.ag_dataset_3d import _load_pkl_compat
    p = Path(MONO3D["annotations"]) / f"{video}.pkl"
    if not p.exists():
        return None
    d = _load_pkl_compat(str(p))
    return {"intrinsics": d.get("intrinsics"), "frames": d.get("frames_final", {}).get("bbox_frames", {})}


def mono3d_head(pred_3d, feats: torch.Tensor, boxes: torch.Tensor, intr: torch.Tensor):
    """Re-run the 3-D prediction layers on captured inputs and keep the
    factorised parameters (the module only returns corners and mu)."""
    import torch.nn.functional as F
    from lib.detector.monocular3d.models.dino_mono_3d import _compute_3d_corners
    ref = pred_3d.input_reference_size
    x = F.relu(pred_3d.context_fc(torch.cat([feats, boxes / ref, intr / ref], 1)))
    dims = F.softplus(pred_3d.dim_pred(x)) + 1e-4
    rot_raw = pred_3d.rot_pred(x)
    rot = rot_raw / torch.sqrt((rot_raw ** 2).sum(1, keepdim=True) + 1e-6)
    depth = F.softplus(pred_3d.depth_pred(x)) + 1e-4
    off = pred_3d.center_offset_pred(x)
    mu = pred_3d.mu_pred(x).squeeze(-1)
    corners = _compute_3d_corners(dims, rot, depth, off, boxes, intr[:, :2], intr[:, 2:])
    return {"dims": dims, "yaw_deg": torch.rad2deg(torch.atan2(rot[:, 0], rot[:, 1])), "depth": depth[:, 0],
            "offset": off, "mu": mu, "corners": corners}


def class_colour(c: int) -> str:
    pal = [BLUE, VIOLET, TEAL, PINK, YELLOW, ROSE, LIME, SKY, GREEN, ORANGE]
    return TXT if c == 1 else pal[c % len(pal)]


def render_mono3d_frame(out: Path, k: int, img: np.ndarray, t: int, frame: str, wh: Tuple[int, int]):
    H, W = img.shape[:2]
    fig = dark_fig(3.6, 3.6 * H / W)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.text(6, H - 6, f"detector input {wh[0]}×{wh[1]}   t = {t + 1}  ({frame})", color=TXT, fontsize=7.5, va="bottom",
            bbox=dict(fc=OVERLAY, ec="none", pad=2))
    save(fig, out / f"frame_{k}.png", dpi=150)


def render_fpn(out: Path, k: int, fpn: Dict[str, torch.Tensor], t: int):
    """fpn_<k>.png: the SimpleFeaturePyramid levels p2..p6 as per-level channel-norm heatmaps."""
    names = list(fpn.keys())
    fig = dark_fig(1.5 * len(names) + 0.3, 3.4)
    for i, n in enumerate(names):
        f = fpn[n][0].float().numpy()
        m = np.linalg.norm(f, axis=0)
        m = (m - m.min()) / max(m.max() - m.min(), 1e-8)
        ax = fig.add_axes([0.02 + i * 0.97 / len(names), 0.06, 0.93 / len(names), 0.84])
        ax.imshow(m, cmap="inferno", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.set_title(f"{n}  {f.shape[1]}×{f.shape[2]}×{f.shape[0]}", fontsize=6.5, color=TXT)
    fig.text(0.02, 0.965, f"feature pyramid (channel L2 norm per cell), t = {t + 1}", color=TXT, fontsize=7.5)
    save(fig, out / f"fpn_{k}.png")


def render_boxes_on_image(out: Path, name: str, img: np.ndarray, boxes: np.ndarray, colours: List[str], texts: List[str],
                          title: str, lw: float = 1.4, alpha: float = 1.0):
    H, W = img.shape[:2]
    fig = dark_fig(3.6, 3.6 * H / W)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    for bx, col, tx in zip(boxes, colours, texts):
        x0, y0, x1, y1 = bx
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=lw, ec=col, alpha=alpha))
        if tx:
            ax.text(x0 + 2, y0 + 2, tx, color="black", fontsize=6.5, va="top", bbox=dict(fc=col, ec="none", pad=1))
    ax.text(0.02, 0.985, title, color=TXT, fontsize=7, va="top", transform=ax.transAxes, bbox=dict(fc=OVERLAY, ec="none", pad=2))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    save(fig, out / name, dpi=150)


def render_det3d(out: Path, k: int, img: np.ndarray, corners: np.ndarray, cls: np.ndarray, scores: np.ndarray,
                 classes: List[str], intr: Tuple[float, float, float, float], gt: List[Tuple[str, np.ndarray]], t: int):
    """det3d_<k>.png: predicted 8-corner boxes projected with the frame intrinsics
    (colour per class); the annotation's boxes dashed."""
    H, W = img.shape[:2]
    fx, fy, cx, cy = intr
    fig = dark_fig(3.6, 3.6 * H / W)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)

    def proj(C):
        z = np.clip(C[:, 2], 1e-3, None)
        return np.stack([fx * C[:, 0] / z + cx, fy * C[:, 1] / z + cy], -1)
    for lab, C in gt:
        P = proj(C)
        for i, j in box_edges(C):
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=MASKC, lw=1.0, ls=(0, (3, 2)))
    for C, c, s in zip(corners, cls, scores):
        P = proj(C); col = class_colour(int(c))
        for i, j in box_edges(C):
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=col, lw=1.4)
        top = P[np.argmin(P[:, 1])]
        ax.text(float(np.clip(top[0], 4, W - 60)), float(np.clip(top[1] - 3, 12, H - 4)), f"{classes[int(c)]} {s:.2f}",
                color="black", fontsize=6.5, va="bottom", bbox=dict(fc=col, ec="none", pad=1))
    ax.text(0.02, 0.985, f"predicted 3-D boxes projected (f = {fx:.0f} px)  ·  annotation dashed   t = {t + 1}",
            color=TXT, fontsize=7, va="top", transform=ax.transAxes, bbox=dict(fc=OVERLAY, ec="none", pad=2))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    save(fig, out / f"det3d_{k}.png", dpi=150)


def render_mono3d_bev(out: Path, k: int, corners: np.ndarray, cls: np.ndarray, scores: np.ndarray, classes: List[str],
                      gt: List[Tuple[str, np.ndarray]], t: int):
    """bev_<k>.png: top-down (x lateral, z depth) camera-frame view of the
    predicted boxes (solid, colour per class) against the annotation's (dashed)."""
    fig = dark_fig(3.4, 3.4)
    ax = fig.add_axes([0.12, 0.1, 0.84, 0.8]); style_axis(ax)
    pts = [np.zeros((1, 2))]
    for lab, C in gt:
        h = convex_hull(C[:, [0, 2]]); ax.plot(h[:, 0], h[:, 1], color=MASKC, lw=1.0, ls=(0, (3, 2)))
        c = C[:, [0, 2]].mean(0); ax.text(c[0], c[1], lab, color=MASKC, fontsize=6, ha="center", va="bottom")
        pts.append(C[:, [0, 2]])
    for C, c, s in zip(corners, cls, scores):
        col = class_colour(int(c)); h = convex_hull(C[:, [0, 2]])
        ax.plot(h[:, 0], h[:, 1], color=col, lw=1.5)
        cc = C[:, [0, 2]].mean(0); ax.text(cc[0], cc[1], f"{classes[int(c)]} {s:.2f}", color=col, fontsize=6, ha="center", va="top")
        pts.append(C[:, [0, 2]])
    ax.scatter([0], [0], marker="^", color=GREEN, s=30, zorder=5); ax.text(0.02, 0.0, "camera", color=GREEN, fontsize=6)
    P = np.concatenate(pts); lo, hi = P.min(0) - 0.15, P.max(0) + 0.15
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("x (m, lateral)", fontsize=7); ax.set_ylabel("z (m, depth)", fontsize=7)
    ax.set_title(f"bird's-eye view, camera frame, t = {t + 1}  (predicted solid, annotation dashed)", fontsize=7, loc="left")
    save(fig, out / f"bev_{k}.png")


def render_mono3d_head(out: Path, k: int, head: Dict[str, np.ndarray], cls: np.ndarray, scores: np.ndarray,
                       classes: List[str], t: int, max_rows: int = 8):
    """head_<k>.png: per detection the factorised 3-D head outputs: dims l, w, h
    and depth as bars, yaw and the uncertainty mu as text."""
    n = min(len(cls), max_rows)
    fig = dark_fig(5.6, 0.42 * max(n, 1) + 0.8)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.text(0.02, 0.96, f"3-D head outputs per detection, t = {t + 1}   (bars: l · w · h and depth in metres)",
            color=TXT, fontsize=7.5, va="top", transform=ax.transAxes)
    dmax = max(float(head["dims"][:n].max()), 1e-3) if n else 1.0
    zmax = max(float(head["depth"][:n].max()), 1e-3) if n else 1.0
    for x, h in ((0.02, "detection"), (0.30, "l, w, h (m)"), (0.60, "depth (m)"), (0.80, "yaw"), (0.90, "μ")):
        ax.text(x, 0.89, h, color=MUTED, fontsize=7, va="top", transform=ax.transAxes)
    dy = 0.74 / max(n, 1)
    for i in range(n):
        y = 0.78 - i * dy; col = class_colour(int(cls[i]))
        ax.text(0.02, y, f"{classes[int(cls[i])]}  {scores[i]:.2f}", color=col, fontsize=7, va="top", transform=ax.transAxes)
        for j, (v, c2) in enumerate(zip(head["dims"][i], (BLUE, TEAL, VIOLET))):
            ax.add_patch(Rectangle((0.30, y - 0.02 - j * dy * 0.28), 0.25 * float(v) / dmax, dy * 0.24, transform=ax.transAxes, fc=c2, ec="none"))
        ax.text(0.56, y, " ".join(f"{float(v):.2f}" for v in head["dims"][i]), color=TXT, fontsize=6, va="top", ha="right", transform=ax.transAxes)
        ax.add_patch(Rectangle((0.60, y - 0.02 - dy * 0.45), 0.16 * float(head["depth"][i]) / zmax, dy * 0.45, transform=ax.transAxes, fc=ORANGE, ec="none"))
        ax.text(0.77, y, f"{float(head['depth'][i]):.2f}", color=TXT, fontsize=6.5, va="top", ha="right", transform=ax.transAxes)
        ax.text(0.80, y, f"{float(head['yaw_deg'][i]):+.0f}°", color=TXT, fontsize=7, va="top", transform=ax.transAxes)
        ax.text(0.90, y, f"{float(head['mu'][i]):+.2f}", color=TXT, fontsize=7, va="top", transform=ax.transAxes)
    save(fig, out / f"head_{k}.png")


def run_mono3d(video: str, out_root: Path, device, frames: List[str], keys: List[int], classes: List[str]):
    """The monocular 3-D detector on the three key frames: hooks on the pyramid,
    the RPN's proposal filter and the 3-D head's inputs; every panel is read out
    of that forward pass."""
    import torchvision.transforms.functional as TF
    from lib.detector.monocular3d.models.dino_mono_3d import DinoV3Monocular3D
    out = out_root / "mono3d"
    out.mkdir(parents=True, exist_ok=True)
    model = DinoV3Monocular3D(num_classes=MONO3D["num_classes"], pretrained=True, model=MONO3D["model"],
                              head_3d_mode=MONO3D["head_3d_mode"]).to(device).eval()
    state = torch.load(MONO3D["checkpoint"], map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    epoch = state.get("epoch", "?")
    del state
    ann = load_mono3d_annotation(video)
    pred_3d = model.head_3d_separate.pred_3d
    thr = MONO3D["score_thr"]
    meta: Dict[str, Any] = {"video": video, "method": "mono3d", "checkpoint": MONO3D["checkpoint"], "epoch": epoch,
                            "backbone": MONO3D["model"], "head_3d_mode": MONO3D["head_3d_mode"], "keyframes": keys,
                            "frames": [frames[t] for t in keys], "score_threshold": thr, "n_proposals_drawn": MONO3D["n_proposals"],
                            "pixel_limit": MONO3D["pixel_limit"], "patch_size_for_resize": MONO3D["patch_size"],
                            "annotation_intrinsics": None if ann is None else ann["intrinsics"], "per_frame": {}, "classes": classes}
    dets_json: Dict[str, Any] = {}
    arrays: Dict[str, np.ndarray] = {}
    for k, t in enumerate(keys):
        raw = raw_frame(video, frames[t]); H0, W0 = raw.shape[:2]
        tw, th = mono3d_target_size(W0, H0, MONO3D["pixel_limit"], MONO3D["patch_size"])
        img = np.asarray(Image.fromarray(raw).resize((tw, th), Image.BILINEAR))
        x = TF.normalize(torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1), list(MONO3D["image_mean"]), list(MONO3D["image_std"]))
        cap = Capture()
        cap.hook_output(model.backbone, "fpn", pick=lambda o: {kk: v.detach().float().cpu() for kk, v in o.items()})
        cap.patch_method(model.rpn, "filter_proposals", "proposals")
        cap.hook_input(pred_3d, "feat", 0); cap.hook_input(pred_3d, "box", 1); cap.hook_input(pred_3d, "intr", 2)
        with torch.no_grad():
            det = model([x.to(device)])[0]
        cap.close()
        fpn = cap.get("fpn")
        pb, ps = cap.get("proposals")
        pb, ps = to_np(pb[0]), to_np(ps[0])
        order = np.argsort(-ps)[:MONO3D["n_proposals"]]
        boxes = to_np(det["boxes"]); labels_d = to_np(det["labels"]).astype(int); scores = to_np(det["scores"])
        keep = scores >= thr
        # intrinsics: (a) what this forward used (image-size default, as the ROI extraction does),
        # (b) the annotation's intrinsics scaled to the resized frame as ag_dataset_3d does at training time.
        cands = {"image_default": (float(max(th, tw)), float(max(th, tw)), tw / 2.0, th / 2.0)}
        if ann is not None and ann["intrinsics"] is not None:
            it = ann["intrinsics"]; sx, sy = tw / W0, th / H0
            cands["annotation_scaled_as_training"] = (float(it["fx"]) * sx, float(it["fy"]) * sy, float(it["cx"]) * sx, float(it["cy"]) * sy)
            cands["annotation_raw"] = (float(it["fx"]), float(it["fy"]), float(it["cx"]), float(it["cy"]))
        gt: List[Tuple[str, np.ndarray]] = []
        if ann is not None and frames[t] in ann["frames"]:
            for o in ann["frames"][frames[t]].get("objects", []):
                lab = _MONO3D_LABEL_REMAP.get(o.get("label"), o.get("label"))
                gt.append((lab, np.asarray(o["obb_corners_final"], dtype=np.float32).reshape(8, 3)))
        heads: Dict[str, Dict[str, np.ndarray]] = {}
        errs: Dict[str, Optional[float]] = {}
        if cap.has("feat") and keep.any():
            feat, bx, intr_used = cap.get("feat").to(device), cap.get("box").to(device), cap.get("intr").to(device)
            for name, (fx, fy, cx, cy) in cands.items():
                intr = torch.tensor([fx, fy, cx, cy], device=device).expand(len(bx), 4)
                with torch.no_grad():
                    h = mono3d_head(pred_3d, feat, bx, intr)
                heads[name] = {kk: to_np(v) for kk, v in h.items()}
                # centre error against the annotation boxes of the same class (nearest), over confident detections
                e = []
                for i in np.where(keep)[0]:
                    cands_gt = [C for lab, C in gt if lab == classes[labels_d[i]]]
                    if cands_gt:
                        e.append(min(np.linalg.norm(C.mean(0) - heads[name]["corners"][i].mean(0)) for C in cands_gt))
                errs[name] = float(np.mean(e)) if e else None
            used = to_np(intr_used)[0]
            same = bool(np.allclose(heads["image_default"]["corners"], to_np(det["boxes_3d"]), atol=1e-3))
        else:
            used, same = None, None
        chosen = "image_default"
        scored = {n: e for n, e in errs.items() if e is not None}
        if scored:
            chosen = min(scored, key=scored.get)
        head = heads.get(chosen)
        draw_intr = cands.get("annotation_raw", cands["image_default"])     # the pinhole that reprojects the annotation boxes
        meta["per_frame"][str(t)] = {"frame": frames[t], "orig_wh": [W0, H0], "resized_wh": [tw, th],
                                     "intrinsics_used_in_forward": None if used is None else [float(v) for v in used],
                                     "intrinsics_candidates": cands, "centre_error_m_vs_annotation": errs,
                                     "intrinsics_chosen_for_head": chosen, "intrinsics_for_projection": draw_intr,
                                     "rerun_matches_model_output": same, "n_detections": int(len(scores)),
                                     "n_above_threshold": int(keep.sum()), "n_proposals": int(len(ps)), "n_annotation_boxes": len(gt)}
        # ---- panels ----
        render_mono3d_frame(out, k, img, t, frames[t], (tw, th))
        render_fpn(out, k, fpn, t)
        render_boxes_on_image(out, f"proposals_{k}.png", img, pb[order], [ORANGE] * len(order), [""] * len(order),
                              f"top-{len(order)} RPN proposals by objectness (of {len(ps)})   t = {t + 1}", lw=0.8, alpha=0.75)
        idx = np.where(keep)[0]
        render_boxes_on_image(out, f"det2d_{k}.png", img, boxes[idx], [class_colour(int(labels_d[i])) for i in idx],
                              [f"{classes[labels_d[i]]} {scores[i]:.2f}" for i in idx],
                              f"2-D detections, score ≥ {thr:g}   t = {t + 1}")
        if head is not None:
            render_det3d(out, k, img, head["corners"][idx], labels_d[idx], scores[idx], classes, draw_intr, gt, t)
            render_mono3d_bev(out, k, head["corners"][idx], labels_d[idx], scores[idx], classes, gt, t)
            render_mono3d_head(out, k, {kk: v[idx] for kk, v in head.items()}, labels_d[idx], scores[idx], classes, t)
        dets_json[str(t)] = {
            "frame": frames[t],
            "detections": [{"class": classes[labels_d[i]], "label": int(labels_d[i]), "score": float(scores[i]),
                            "box_xyxy": boxes[i].tolist(),
                            **({} if head is None else {"dims_lwh": head["dims"][i].tolist(), "yaw_deg": float(head["yaw_deg"][i]),
                                                        "depth": float(head["depth"][i]), "centre_offset_px": head["offset"][i].tolist(),
                                                        "mu": float(head["mu"][i]), "corners_cam": head["corners"][i].tolist()})}
                           for i in idx],
            "annotation": [{"class": lab, "corners_cam": C.tolist()} for lab, C in gt],
        }
        arrays[f"proposals_{k}"] = pb[order]; arrays[f"proposal_scores_{k}"] = ps[order]
        arrays[f"det_boxes_{k}"] = boxes; arrays[f"det_labels_{k}"] = labels_d; arrays[f"det_scores_{k}"] = scores
        if head is not None:
            arrays[f"det_corners_{k}"] = head["corners"]
        for kk, v in fpn.items():
            arrays[f"fpn_{k}_{kk}_norm"] = np.linalg.norm(v[0].numpy(), axis=0)
        print(f"[mono3d] t={t + 1}: {len(ps)} proposals, {int(keep.sum())} detections ≥ {thr}, "
              f"intrinsics errors {errs} → head uses {chosen}")
    np.savez_compressed(out / "tensors.npz", **arrays)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(out / "dets.json", "w") as f:
        json.dump(dets_json, f, indent=1)
    print(f"[mono3d] wrote {sorted(p.name for p in out.iterdir())}")
    del model
    torch.cuda.empty_cache()


def baseline_keyframes(video: str, annot_dir: Optional[str]):
    """Key frames for the detector dump: those of the predcls item the baselines
    see (same rule, same slot), read from the W-STTran cell's dataset."""
    conf = load_wsgg_config(str(REPO / CELLS["w_sttran"]))
    if annot_dir:
        conf.test_annot_dir = annot_dir
    ds = make_test_dataset(conf)
    item, _ = load_item(ds, video)
    valid = to_np(item["valid_mask"]).astype(bool); vis = to_np(item["visibility_mask"]).astype(bool)
    classes = list(ds.object_classes); cls_idx = to_np(item["object_classes"]).astype(int)
    labels = [classes[cls_idx[np.where(valid[:, n])[0][0], n]] if valid[:, n].any() else f"slot{n}" for n in range(valid.shape[1])]
    tracked, keys = pick_keyframes(vis, valid, labels)
    return list(item["frame_names"]), keys, tracked, labels, classes


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
    ap.add_argument("--methods", nargs="*", default=WW_METHODS,
                    help=f"any of {WW_METHODS + list(BASELINE_CELLS) + ['mono3d']}")
    ap.add_argument("--checkpoint", default="best", help="'best' or checkpoint_<N>")
    ap.add_argument("--annot-dir", default=None,
                    help="override the test annotation folder (e.g. world4d_rel_annotations for a legacy-split video)")
    ap.add_argument("--theme", default="light", choices=list(PALETTES),
                    help="panel palette: white background (light) or the dark hero palette")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every panel text: Title Case, UPPER CASE or as written")
    ap.add_argument("--keyframes", nargs=3, type=int, default=None,
                    help="three one-based key frames to show instead of the automatic visible / unseen / last pick")
    args, _ = ap.parse_known_args()
    if args.keyframes:
        KEYFRAMES[:] = [k - 1 for k in args.keyframes]
    apply_theme(args.theme)
    global CAPS
    CAPS = args.caps
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir) / args.video
    key_info = None
    for m in args.methods:
        if m == "mono3d":
            if key_info is None:
                key_info = baseline_keyframes(args.video, args.annot_dir)
            frames, keys, tracked, labels, classes = key_info
            print(f"[mono3d] key frames {[k + 1 for k in keys]} (tracked slot '{labels[tracked]}')")
            run_mono3d(args.video, out_root, device, frames, keys, classes)
        elif m in BASELINE_CELLS:
            key_info = run_baseline(m, args.video, out_root, device, args.checkpoint, args.annot_dir)
        else:
            run_method(m, args.video, out_root, device, args.checkpoint, args.annot_dir)


if __name__ == "__main__":
    main()
