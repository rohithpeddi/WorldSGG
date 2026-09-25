"""Real panels for the five MLLM architecture figures from one video's run bundle.

    python scripts/paper_figures/dark/mllm_panels.py --bundle outputs/intermediates/00T1E/mllm_bundle \\
        --out outputs/intermediates/00T1E [--theme light|dark] [--frame 000170.png] [--object laptop]

Writes ``<out>/{zero_shot,caption_all,graph_rag,track_a,track_b}/<slot>.png`` (``_dark`` suffix on the
directories for the dark palette), one PNG per slot key of the matching ``fig_*.py`` at 4x the slot
size, plus ``<out>/mllm_picks.json``: which frame, object, segment and mode each panel shows and the
fixed rule that chose it (never the prediction's correctness).  The bundle comes from
``scripts/paper_figures/dump_mllm_intermediates.py`` (server).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

from common import PALETTES, THEME  # noqa: E402  (reads --theme / FIG_THEME)

P = PALETTES[THEME]
S = 4                                    # panel pixels per slot pixel
SANS_F, MONO_F, SERIF_F = "Segoe UI", "Consolas", "Georgia"
MARKS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
         (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128)]   # tools/marks.py
MARK_HEX = ["#%02x%02x%02x" % c for c in MARKS]
ENT_COLS = ["#e6194b", "#3cb44b", "#2f7fd8", "#f58231"] if THEME == "light" else \
           ["#ff6b8a", "#62d27a", "#5aa2f2", "#ffa45c"]
PERSON_COL = "#111111" if THEME == "light" else "#e6e8ec"
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(n) -> str:
    return str(n).translate(_SUB)


def short(label: str) -> str:
    return label.replace("_", " ")


# ------------------------------------------------------------------------------------------
# drawing helpers (coordinates in slot pixels, y down)
# ------------------------------------------------------------------------------------------
class Panel:
    def __init__(self, w, h, bg=None, card=False):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w * S / 72, h * S / 72), dpi=72)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self._lim()
        self.fig.patch.set_facecolor(bg or P["BG"])
        if card:
            self.rect(3, 3, w - 6, h - 6, fc=P["CARD"], r=3)

    def _lim(self):
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(self.h, 0)
        self.ax.set_aspect("auto")
        self.ax.axis("off")

    def text(self, x, y, s, size=8.4, col=None, font="sans", weight="normal", ha="left", va="baseline",
             style="normal", clip_w=None, z=5):
        fam = {"sans": SANS_F, "mono": MONO_F, "serif": SERIF_F}[font]
        if clip_w is not None:
            s = self.fit_text(s, clip_w, size, font, weight)
        return self.ax.text(x, y, s, fontsize=size * S, color=col or P["TXT"], family=fam, weight=weight, ha=ha,
                            va=va, style=style, zorder=z)

    def tw(self, s, size=8.4, font="sans", weight="normal", style="normal") -> float:
        """Rendered width of ``s`` in slot pixels."""
        fam = {"sans": SANS_F, "mono": MONO_F, "serif": SERIF_F}[font]
        t = self.ax.text(0, 0, s, fontsize=size * S, family=fam, weight=weight, style=style)
        bb = t.get_window_extent(renderer=self.fig.canvas.get_renderer())
        t.remove()
        return bb.width / S

    def fit_text(self, s, w, size=8.4, font="sans", weight="normal") -> str:
        """``s`` shortened with an ellipsis until it renders within ``w`` slot pixels."""
        if self.tw(s, size, font, weight) <= w:
            return s
        lo, hi = 0, len(s)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.tw(s[:mid].rstrip() + "…", size, font, weight) <= w:
                lo = mid
            else:
                hi = mid - 1
        return s[:lo].rstrip() + "…"

    def wrap(self, s, w, size=8.4, font="sans", weight="normal"):
        """Greedy word wrap by rendered width; over-long words are split."""
        lines, cur = [], ""
        for word in s.split(" "):
            cand = word if not cur else cur + " " + word
            if self.tw(cand, size, font, weight) <= w:
                cur = cand
                continue
            if cur:
                lines.append(cur)
            while self.tw(word, size, font, weight) > w and len(word) > 1:
                k = len(word)
                while k > 1 and self.tw(word[:k], size, font, weight) > w:
                    k -= 1
                lines.append(word[:k])
                word = word[k:]
            cur = word
        if cur:
            lines.append(cur)
        return lines

    def rect(self, x, y, w, h, fc="none", ec=None, lw=1.0, r=0.0, ls="-", alpha=1.0, z=1):
        style = f"round,pad=0,rounding_size={r}" if r else "square,pad=0"
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec or "none", lw=lw * S * 0.75,
                                         ls=ls, alpha=alpha, zorder=z))

    def line(self, xs, ys, col=None, lw=1.0, ls="-", alpha=1.0, z=2):
        self.ax.plot(xs, ys, color=col or P["LINE"], lw=lw * S * 0.75, ls=ls, alpha=alpha, zorder=z,
                     solid_capstyle="round")

    def poly(self, pts, fc="none", ec=None, lw=1.0, ls="-", alpha=1.0, z=3):
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec or "none", lw=lw * S * 0.75, ls=ls, alpha=alpha,
                                  zorder=z))

    def image(self, img, x, y, w, h, border=None, lw=1.0, align="center", z=2):
        """Fit ``img`` inside (x, y, w, h) keeping its aspect; returns the drawn rectangle."""
        a = np.asarray(img)
        ih, iw = a.shape[:2]
        s = min(w / iw, h / ih)
        dw, dh = iw * s, ih * s
        dx = x + (w - dw) / 2 if align == "center" else (x if align == "left" else x + w - dw)
        dy = y + (h - dh) / 2
        self.ax.imshow(a, extent=(dx, dx + dw, dy + dh, dy), interpolation="lanczos", zorder=z)
        self._lim()
        if border:
            self.rect(dx, dy, dw, dh, ec=border, lw=lw, z=z + 1)
        return dx, dy, dw, dh

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=72, facecolor=self.fig.get_facecolor())
        plt.close(self.fig)


def clip(s: str, w: float, size: float, mono: bool) -> str:
    n = max(4, int(w / (size * (0.55 if mono else 0.5))))
    return s if len(s) <= n else s[: n - 1] + "…"


def wrap(s: str, w: float, size: float, mono: bool):
    return textwrap.wrap(s, width=max(8, int(w / (size * (0.55 if mono else 0.49)))), break_long_words=True)


def card_head(pn: Panel, head: str, y=14.5, col=None, size=8.8):
    pn.text(9, y, head, size=size, col=col or P["ORANGE"], weight="semibold", clip_w=pn.w - 16)


def strip_fences(s: str) -> str:
    return (s or "").replace("```json", "").replace("```", "").strip()


def label_of(att):
    if isinstance(att, dict):
        return att.get("label")
    return att


def labels_of(lst):
    return [label_of(x) for x in (lst or [])]


# ------------------------------------------------------------------------------------------
# bundle + picks
# ------------------------------------------------------------------------------------------
class Bundle:
    def __init__(self, root: Path):
        self.root = root

        def js(name):
            p = root / name
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        self.video = js("video.json")
        self.graph = js("graph.json")
        # the three unlocalized runners share one pickle schema; their captures differ
        self.runs = {meth: {m: js(f"{meth}_{m}.json") for m in ("predcls", "sgdet")}
                     for meth in ("rag_all", "zero_shot", "caption_all")}
        self.caps = {meth: {m: js(f"capture_{meth}_{m}.json") or (js(f"capture_{m}.json") if meth == "rag_all" else None)
                            for m in ("predcls", "sgdet")} for meth in ("rag_all", "zero_shot", "caption_all")}
        self.rag, self.cap = self.runs["rag_all"], self.caps["rag_all"]
        self.ta = {m: js(f"track_a_{m}.json") for m in ("predcls", "sgdet")}
        self.tb = {m: js(f"track_b_{m}.json") for m in ("predcls", "sgdet")}
        self.meta = js("cache/meta.json")
        self.dets = js("cache/detections.json")
        self.frames = {f["file"]: f for f in self.video["frames"]}

    def tensors(self, mode, method="rag_all"):
        for p in (self.root / f"capture_{method}_{mode}.npz",) + \
                 ((self.root / f"capture_{mode}.npz",) if method == "rag_all" else ()):
            if p.exists():
                return np.load(p)
        return None

    def payload(self, mode, stem):
        d = self.root / "payload" / mode
        js = json.loads((d / f"{stem}_objects.json").read_text(encoding="utf-8"))
        imgs = [Image.open(d / f"{stem}_img{j}.png").convert("RGB") for j in range(js["n_images"])]
        return {"images": imgs, "objects": js["objects"], "person_corners": js["person_corners"],
                "retrieved": js.get("retrieved", ""), "prompt_a": (d / f"{stem}_prompt_a.txt").read_text(encoding="utf-8"),
                "prompt_b": (d / f"{stem}_prompt_b.txt").read_text(encoding="utf-8")}

    def key_frames(self):
        return sorted(int(p.stem) for p in (self.root / "frames_annotated").glob("*.png"))


def make_picks(b: Bundle, args) -> dict:
    frames = b.video["frames"]
    cap = b.cap["predcls"]
    prompt_obj = {p: o for o, p in cap["batch"]["prompts_by_object"].items()}
    kw = {prompt_obj.get(k["prompt"]): k for k in cap["keywords"]}
    ret = {prompt_obj.get(r["question"]): r for r in cap["retrieval"]}
    unseen = Counter(o["label"] for f in frames for o in f["objects"] if not o["visible"])
    objs = sorted({o["label"] for f in frames for o in f["objects"]})
    parsed = [o for o in objs if kw.get(o) and kw[o]["keywords"]]
    obj = args.object or max(parsed or objs, key=lambda o: (unseen[o], [-ord(ch) for ch in o]))
    rule_obj = "object with parsed P2 keywords and the most unseen frames"

    def mix(f):
        vis = [o["visible"] for o in f["objects"]]
        return sum(vis) * (len(vis) - sum(vis))
    if args.frame:
        f = args.frame
        rule_f = "given on the command line"
    else:
        cand = [fr for fr in frames if any(o["label"] == obj and not o["visible"] for o in fr["objects"])]
        cand = cand or frames
        best = max(mix(fr) for fr in cand)
        tied = [fr for fr in cand if mix(fr) == best]
        f = tied[len(tied) // 2]["file"]
        rule_f = (f"a frame where {obj} is unseen, maximising (#visible x #unseen objects); middle of the ties")
    # the top-1 node of the featured object's retrieval -> its segment
    nodes = {n["id"]: n for n in cap["batch"]["graph_nodes"]}
    top1 = ret[obj]["nodes"][0]
    seg_frame = nodes[top1]["source_annotated_frame"]
    clips = b.graph["clips"]
    seg = next(i for i, c in enumerate(clips) if c["clip_metadata"]["annotated_frame"] == seg_frame)
    kf = [c["clip_metadata"]["annotated_frame"] for c in clips]
    n = len(kf)
    trio = sorted({kf[round(0.12 * (n - 1))], kf[seg], kf[round(0.88 * (n - 1))]})
    while len(trio) < 3:
        extra = [k for k in kf if k not in trio]
        trio = sorted(trio + extra[len(extra) // 2: len(extra) // 2 + 1])
    # Track B frame for the critic: flagged, and the repair lowered the violation count; a moved box first
    fnum = int(Path(f).stem)

    def moved(v):
        pre, post = v["objects_pre"], v["objects"]
        flag = {x["label"] for x in v["violations_pre"]}
        return any((pre.get(l) or {}).get("corners") != (post.get(l) or {}).get("corners") for l in flag)
    crit = None
    tiers = [("box moved by the repair and fewer violations", lambda v: moved(v) and len(v["violations_post"]) < len(v["violations_pre"])),
             ("fewer violations after the repair", lambda v: len(v["violations_post"]) < len(v["violations_pre"])),
             ("flagged by the critic", lambda v: True)]
    for why, test in tiers:
        for mode in ("sgdet", "predcls"):
            tb = b.tb.get(mode)
            ok = [k for k, v in (tb or {}).get("frames", {}).items() if v["violations_pre"] and test(v)]
            if ok:
                k = min(ok, key=lambda k: (abs(int(Path(k).stem) - fnum), k))
                crit = {"mode": mode, "frame": k, "rule": f"Track B frame: {why}; SGDet before PredCls; nearest to f"}
                break
        if crit:
            break
    by_time = sorted(nodes.values(), key=lambda n: n["source_annotated_frame"])
    node_rank = {str(n["id"]): i + 1 for i, n in enumerate(by_time)}
    return {"video": b.video["video_id"], "node_rank": node_rank, "object": obj, "object_rule": rule_obj, "frame": f, "frame_rule": rule_f,
            "top1_node": top1, "segment": seg, "segment_rule": "segment of the top-1 retrieved node of the object",
            "key_frames": trio, "stage2_mode": "predcls", "critic": crit}


# ------------------------------------------------------------------------------------------
# Graph-RAG panels
# ------------------------------------------------------------------------------------------
def gr_frames(b, pk, out):
    for i, k in enumerate(pk["key_frames"]):
        shutil.copy2(b.root / "frames_annotated" / f"{k:06d}.png", out / f"frame_{i}.png")


def gr_segments(b, pk, out):
    w, h = 236, 128
    pn = Panel(w, h)
    clips = b.graph["clips"]
    th = np.load(b.root / "raw_thumbs.npz")
    raw_f, thumbs = list(th["frames"]), th["thumbs"]
    F0, F1 = min(raw_f), max(raw_f)
    x0, x1, ty, bh = 10, w - 10, 20, 9

    def X(fr):
        return x0 + (fr - F0) / max(1, F1 - F0) * (x1 - x0)
    pn.text(x0, 11, f"{F1 - F0 + 1} frames · {len(clips)} key frames → {len(clips)} segments", size=7.6,
            col=P["DIM"])
    for j, c in enumerate(clips):
        m = c["clip_metadata"]
        pn.rect(X(m["start_frame"]), ty, max(0.6, X(m["end_frame"]) - X(m["start_frame"])), bh,
                fc=P["GRIDLINE"] if j % 2 else P["BAR_OFF"], z=1)
        kx = X(m["annotated_frame"])
        pn.line([kx, kx], [ty - 1, ty + bh + 1], col=P["ORANGE"], lw=0.9, z=3)
    seg = clips[pk["segment"]]["clip_metadata"]
    sx0, sx1 = X(seg["start_frame"]), X(seg["end_frame"])
    pn.rect(sx0 - 1, ty - 3, sx1 - sx0 + 2, bh + 6, ec=P["ORANGE"], lw=1.3, z=4)
    fr = [f for f in range(seg["start_frame"], seg["end_frame"] + 1, 2) if f in raw_f]
    if seg["annotated_frame"] not in fr and seg["annotated_frame"] in raw_f:
        fr = sorted(fr + [seg["annotated_frame"]])
    if len(fr) > 6:
        keep = sorted(set(np.linspace(0, len(fr) - 1, 6).round().astype(int).tolist()))
        pick = [fr[i] for i in keep]
        if seg["annotated_frame"] not in pick:
            j = min(range(len(pick)), key=lambda i: abs(pick[i] - seg["annotated_frame"]))
            pick[j] = seg["annotated_frame"]
        fr = sorted(set(pick))
    tw_h = 62
    ar = thumbs.shape[2] / thumbs.shape[1]
    tw = tw_h * ar
    gap = 3
    total = len(fr) * tw + (len(fr) - 1) * gap
    rx = (w - total) / 2
    ry = 38
    pn.line([sx0, rx], [ty + bh + 3, ry - 2], col=P["DIM"], lw=0.6, z=1)
    pn.line([sx1, rx + total], [ty + bh + 3, ry - 2], col=P["DIM"], lw=0.6, z=1)
    for i, f in enumerate(fr):
        key = f == seg["annotated_frame"]
        pn.image(thumbs[raw_f.index(f)], rx + i * (tw + gap), ry, tw, tw_h,
                 border=P["ORANGE"] if key else P["FRAME_B"], lw=1.6 if key else 0.8)
    i = pk["segment"] + 1
    pn.text(w / 2, h - 7, f"S{sub(i)} · frames {seg['start_frame']}–{seg['end_frame']} · "
            f"orange = key frame k{sub(i)} = {seg['annotated_frame']}", size=7.8, ha="center")
    pn.save(out / "segments.png")


def gr_caption(b, pk, out):
    w, h = 196, 70
    i = pk["segment"]
    pn = Panel(w, h, card=True)
    card_head(pn, f"Caption c{sub(i + 1)}  (≤ 100 Tokens)")
    cap = b.graph["clips"][i]["caption"] or "(empty)"
    lines = pn.wrap(f"“{cap}”", w - 18, 8.2)
    for j, ln in enumerate(lines[:3]):
        if j == 2 and len(lines) > 3:
            ln = clip(ln + " …", w - 18, 8.2, False)
        pn.text(9, 29 + j * 12.5, ln, size=8.2)
    pn.save(out / "caption_card.png")


def _node_lines(node):
    ents = [e.split(",")[0].strip() for e in node.get("entities") or []]
    acts = []
    for a in node.get("actions") or []:
        p = [s.strip() for s in a.split(",", 1)]
        acts.append(f"{p[0]}: {p[1][0].lower() + p[1][1:]}" if len(p) == 2 and p[1] else a)
    scen = [s.strip() for s in node.get("scenes") or []]
    return ents, acts, scen


def gr_node(b, pk, out):
    w, h = 196, 100
    i = pk["segment"]
    clip_ = b.graph["clips"][i]
    node = clip_["nodes"][0] if clip_["nodes"] else {}
    ents, acts, scen = _node_lines(node)
    pn = Panel(w, h, card=True)
    card_head(pn, f"Node n{sub(i + 1)}  (JSON, ≤ 512 Tokens)")
    size, lh, x = 7.6, 11.2, 9
    rows = []
    ent_lines = pn.wrap(" · ".join(ents), w - 64, size, "mono") or ["—"]
    rows += [("entities", ent_lines[0])] + [("", ln) for ln in ent_lines[1:2]]
    for a in acts[:2]:
        rows.append(("actions" if a is acts[0] else "", a))
    rows.append(("scenes", " · ".join(scen) or "—"))
    y = 28
    for k, v in rows[:6]:
        pn.text(x, y, k, size=size, col=P["DIM"], font="mono")
        pn.text(x + 52, y, v, size=size, col=P["TXT"], font="mono", clip_w=w - 64)
        y += lh
    pn.save(out / "node_card.png")


def _entity_picks(b, pk):
    ent = b.cap["predcls"]["batch"]["entity_graph"]
    gt = sorted({o["label"] for f in b.video["frames"] for o in f["objects"]})
    picked = []
    for g in gt:
        keys = [k for k in ent if k != "person" and (g in k.split() or k.rstrip("s") == g or g.rstrip("s") == k
                                                    or k in (g, g + "s"))]
        if keys:
            k = max(keys, key=lambda k: len(ent[k]))
            if k not in picked:
                picked.append(k)
    rest = sorted((k for k in ent if k not in picked and k != "person"), key=lambda k: -len(ent[k]))
    picked += rest[: max(0, 3 - len(picked))]
    return picked[:4], ent


def gr_event_graph(b, pk, out):
    """The merged graph's entity index (name -> {nodes}): each shown name fans out to the
    segment nodes that mention it; nodes sharing a name are linked through it."""
    w, h = 252, 214
    pn = Panel(w, h)
    nodes = sorted(b.cap["predcls"]["batch"]["graph_nodes"], key=lambda n: n["source_annotated_frame"])
    order = [n["id"] for n in nodes]
    n = len(order)
    ny = h - 40
    xs = {nid: 12 + i * (w - 24) / max(1, n - 1) for i, nid in enumerate(order)}
    ents, index = _entity_picks(b, pk)
    names = ents + (["person"] if "person" in index else [])
    mean_x = {e: float(np.mean([xs[m] for m in index[e] if m in xs])) for e in names}
    names = sorted(names, key=lambda e: mean_x[e])
    k = len(names)
    anchor = {e: (26 + i * (w - 52) / max(1, k - 1), 34 if i % 2 == 0 else 58) for i, e in enumerate(names)}

    def col_of(e):
        return P["BAR_OFF"] if e == "person" else ENT_COLS[ents.index(e) % len(ENT_COLS)]
    for e in sorted(names, key=lambda e: e != "person"):
        ax_, ay = anchor[e]
        for m in index[e]:
            if m in xs:
                pn.line([ax_, xs[m]], [ay + 5, ny - 3.5], col=col_of(e), lw=0.55 if e == "person" else 0.75,
                        alpha=0.9, z=1 if e == "person" else 2)
    for e in names:
        ax_, ay = anchor[e]
        lab = f"{e} ({len(index[e])})"
        tw = pn.tw(lab, 7.4, weight="semibold")
        pn.rect(ax_ - tw / 2 - 4, ay - 9, tw + 8, 14, fc=P["BG"], ec=col_of(e), lw=1.0, r=3, z=4)
        pn.text(ax_, ay + 1.8, lab, size=7.4, col=P["MUTED"] if e == "person" else col_of(e), ha="center",
                weight="semibold", z=5)
    top = pk["top1_node"]
    for nid in order:
        is_top = nid == top
        pn.ax.add_patch(plt.Circle((xs[nid], ny), 3.8 if is_top else 2.9, fc=P["ORANGE"] if is_top else P["NODE_F"],
                                   ec=P["ORANGE"] if is_top else P["LINE"], lw=0.9 * S * 0.75, zorder=6))
    ti = order.index(top)
    for i, nid in enumerate(order):
        if (i % 7 == 0 or i == n - 1) and abs(i - ti) > 2:
            pn.text(xs[nid], ny + 13, f"n{sub(i + 1)}", size=7, col=P["DIM"], ha="center")
    pn.text(xs[top], ny + 13, f"n{sub(ti + 1)}", size=7.8, col=P["ORANGE"], ha="center", weight="semibold")
    pn.text(8, 12, f"entity index: {len(index)} names → nodes, {len(names)} shown", size=7.2, col=P["DIM"])
    pn.text(w / 2, h - 6, f"{n} nodes, one per segment, in time order", size=7.2, col=P["DIM"], ha="center")
    pn.save(out / "event_graph.png")


def gr_video_v(b, pk, out, method="rag_all"):
    w, h = 160, 58
    t = b.tensors("sgdet", method) or b.tensors("predcls", method)
    V = t["video_v"]
    pn = Panel(w, h)
    n = len(V)
    rows = 2
    cols = math.ceil(n / rows)
    ch = (h - 6 - 3) / rows
    cw = ch * V.shape[2] / V.shape[1]
    gw = cols * cw + (cols - 1) * 2
    x0 = (w - gw) / 2
    for i in range(n):
        r, c = divmod(i, cols)
        pn.image(V[i], x0 + c * (cw + 2), 3 + r * (ch + 3), cw, ch, border=P["FRAME_B"], lw=0.5)
    pn.save(out / "video_v.png")


def gr_objects(b, pk, out, method="rag_all"):
    w, h = 186, 96
    em = (b.runs[method]["sgdet"] or {}).get("estimation_meta") or {}
    pn = Panel(w, h)
    sc = {o: float(s.get("yes_prob", 0)) for o, s in (em.get("object_scores") or {}).items()}
    gt = set(em.get("annotated_objects") or [])
    rows = sorted(sc, key=lambda o: (-sc[o], o))
    pn.text(10, 12, "object", size=7.6, col=P["DIM"])
    pn.text(w - 10, 12, "P(Yes)", size=7.6, col=P["DIM"], ha="right")
    n = 6
    rh = 10.2
    for i, o in enumerate(rows[:n]):
        y = 18 + i * rh
        isgt = o in gt
        pn.text(10, y + 7.6, short(o), size=7.8, col=P["TXT"] if isgt else P["MUTED"],
                weight="semibold" if isgt else "normal", clip_w=58)
        if isgt:
            pn.text(66, y + 7.4, "GT", size=6.4, col=P["GREEN"], weight="bold")
        bx, bw = 80, (w - 80 - 30)
        pn.rect(bx, y + 2.5, bw, 5, fc=P["BAR_OFF"], r=1.5, z=1)
        pn.rect(bx, y + 2.5, bw * sc[o], 5, fc=P["GREEN"], r=1.5, alpha=0.85, z=2)
        pn.text(w - 10, y + 7.6, f"{sc[o]:.2f}", size=7.4, col=P["MUTED"], ha="right", font="mono")
    found = len(gt & set(sc))
    pn.text(10, h - 6, f"{len(sc)} kept of {len(em.get('raw_estimated') or [])} named · GT {found}/{len(gt)}"
            + (f" · +{len(rows) - n} more" if len(rows) > n else ""), size=7.2, col=P["DIM"])
    pn.save(out / "objects.png")


def gr_keywords(b, pk, out):
    w, h = 132, 62
    cap = b.cap["predcls"]
    prompt = cap["batch"]["prompts_by_object"][pk["object"]]
    rec = next(k for k in cap["keywords"] if k["prompt"] == prompt)
    kws = rec["keywords"]
    pn = Panel(w, h, card=True)
    card_head(pn, f"Keywords · {short(pk['object'])}", size=8.4)
    s = json.dumps(kws)
    lines = pn.wrap(s, w - 18, 7.8, "mono")
    for j, ln in enumerate(lines[:2]):
        pn.text(9, 28 + j * 11.5, ln, size=7.8, font="mono", col=P["TXT"])
    failed = [o for o, p in cap["batch"]["prompts_by_object"].items()
              if not next(k for k in cap["keywords"] if k["prompt"] == p)["keywords"]]
    note = "+ the question P₄(o) itself"
    if failed:
        note += f" · {', '.join(failed)}: unparsed"
    pn.text(9, h - 8, note, size=6.6, col=P["DIM"], clip_w=w - 14)
    pn.save(out / "keywords_card.png")


def gr_ranked(b, pk, out):
    w, h = 176, 96
    cap = b.cap["predcls"]
    prompt = cap["batch"]["prompts_by_object"][pk["object"]]
    r = next(x for x in cap["retrieval"] if x["question"] == prompt)
    sims = dict(zip(r["node_ids"], r["node_sims"]))
    kept = r["nodes"]
    order = sorted(sims, key=lambda n: -sims[n])
    pn = Panel(w, h)
    ax0, ay, aw, ah = 26, h - 18, w - 34, h - 40
    lo = min(0.45, math.floor(min(sims.values()) * 20) / 20)
    hi = math.ceil(max(sims.values()) * 20) / 20 + 0.03
    bw = aw / len(order)
    for i, nid in enumerate(order):
        v = sims[nid]
        bh = (v - lo) / (hi - lo) * ah
        col = P["ORANGE"] if nid == kept[0] else (P["PURPLE_S"] if nid in kept else P["BAR_OFF"])
        pn.rect(ax0 + i * bw + 0.4, ay - bh, max(0.5, bw - 0.8), bh, fc=col, z=2)
    ty = ay - (0.5 - lo) / (hi - lo) * ah
    pn.line([ax0 - 3, ax0 + aw], [ty, ty], col=P["RED"], lw=0.9, ls=(0, (3, 2)), z=3)
    pn.text(ax0 - 4, ty + 2.5, "0.5", size=7, col=P["RED"], ha="right")
    for v in (lo, math.floor(hi * 20) / 20):
        yy = ay - (v - lo) / (hi - lo) * ah
        pn.text(ax0 - 4, yy + 2.5, f"{v:.2f}".lstrip("0"), size=6.6, col=P["DIM"], ha="right")
    pn.line([ax0, ax0 + aw], [ay, ay], col=P["LINE"], lw=0.8)
    pn.text(ax0 + aw / 2, ay + 11, f"{len(order)} nodes by mean cosine · top {len(kept)} kept", size=7.2,
            col=P["DIM"], ha="center")
    pn.text(ax0, 11, f"top-1 = n{sub(pk['node_rank'][str(kept[0])])} · {sims[kept[0]]:.2f}", size=7.6,
            col=P["ORANGE"], weight="semibold")
    pn.save(out / "ranked_nodes.png")


def gr_context(b, pk, out):
    w, h = 204, 96
    cap = b.cap["predcls"]
    bt = cap["batch"]
    i = next(j for j, e in enumerate(bt["entries"]) if e[1] == pk["object"])
    ctx = bt["node_contexts"][i].strip()
    pn = Panel(w, h, card=True)
    size, lh = 7.4, 10.4
    head, _, body = ctx.partition("\n")
    pn.text(9, 15, head, size=size, font="mono", col=P["ORANGE"], clip_w=w - 16)
    lines = pn.wrap(body, w - 18, size, "mono")
    nmax = int((h - 24) // lh)
    for j, ln in enumerate(lines[:nmax]):
        if j == nmax - 1 and len(lines) > nmax:
            ln = clip(ln, w - 26, size, True) + " …"
        pn.text(9, 27 + j * lh, ln, size=size, font="mono", col=P["TXT"])
    pn.save(out / "context_card.png")


def gr_query(b, pk, out, method="rag_all"):
    w, h = 300, 104
    t = b.tensors("predcls", method)
    cap = b.caps[method]["predcls"]
    stems = cap["batch"]["q_stems"] if method == "rag_all" else cap["q_stems"]
    tgt = t["q_targets"][stems.index(pk["frame"])]
    ctxf = t["q_context"]
    pn = Panel(w, h)
    lab_h = 14
    th = h - lab_h - 6
    tw = th * tgt.shape[1] / tgt.shape[0]
    rows = 2
    cols = math.ceil(len(ctxf) / rows)
    ch = (th - 3) / rows
    cw = ch * ctxf.shape[2] / ctxf.shape[1]
    total = tw + 8 + cols * cw + (cols - 1) * 2
    x0 = (w - total) / 2
    pn.image(tgt, x0, 4, tw, th, border=P["ORANGE"], lw=1.8)
    for i in range(len(ctxf)):
        r, c = divmod(i, cols)
        pn.image(ctxf[i], x0 + tw + 8 + c * (cw + 2), 4 + r * (ch + 3), cw, ch, border=P["FRAME_B"], lw=0.5)
    fnum = int(Path(pk["frame"]).stem)
    H, W = tgt.shape[:2]
    pn.text(w / 2, h - 5, f"[ target f = {fnum} ; {len(ctxf)} key frames ] · {W}×{H} each", size=7.8,
            col=P["DIM"], ha="center")
    pn.save(out / "query_tensor.png")


def _pred_for(rag, frame, obj):
    fr = rag["frames"][frame]
    return next(p for p in fr["predictions"] if p["object"] == obj)


def gr_answer(b, pk, out, method="rag_all"):
    w, h = 210, 116
    p = _pred_for(b.runs[method]["predcls"], pk["frame"], pk["object"])
    fnum = int(Path(pk["frame"]).stem)
    pn = Panel(w, h, card=True)
    card_head(pn, f"Answer For (f = {fnum}, o = {short(pk['object'])})")
    raw = strip_fences(p["raw_response"])
    lines = []
    for ln in raw.splitlines():
        lines += pn.wrap(ln.strip() if ln.strip() in ("{", "}") else ln.rstrip(), w - 18, 7.8, "mono") or [""]
    y = 30
    for ln in lines[:6]:
        pn.text(9, y, ln, size=7.8, font="mono", col=P["TXT"])
        y += 11.6
    pn.text(9, h - 8, "parsed; every label scores 1.0 (no verification)", size=6.8, col=P["DIM"], clip_w=w - 14)
    pn.save(out / "answer_card.png")


ABBR = {"not_looking_at": "not looking", "looking_at": "looking", "unsure": "unsure",
        "not_contacting": "no contact", "other_relationship": "other", "in_front_of": "in front",
        "on_the_side_of": "side", "sitting_on": "sitting on", "lying_on": "lying on", "standing_on": "standing on",
        "leaning_on": "leaning on", "covered_by": "covered by", "have_it_on_the_back": "on back",
        "drinking_from": "drinking", "writing_on": "writing on"}


def ab(label):
    return ABBR.get(label, (label or "—").replace("_", " "))


def gr_scene_graph(b, pk, out, method="rag_all"):
    w, h = 304, 140
    fr = b.frames[pk["frame"]]
    rag = b.runs[method]["predcls"]["frames"][pk["frame"]]
    objs = fr["objects"]
    pn = Panel(w, h)
    px, py = 24, (h - 14) / 2
    n = len(objs)
    ys = [10 + (h - 30) * (i + 0.5) / n for i in range(n)]
    ox = w - 44
    bus = px + 21
    pn.line([px + 14, bus], [py, py], col=P["PURPLE_S"], lw=1.0, z=1)
    pn.line([bus, bus], [min(ys), max(ys)], col=P["PURPLE_S"], lw=1.0, z=1)
    for i, o in enumerate(objs):
        pred = next((p for p in rag["predictions"] if p["object"] == o["label"]), None)
        oy = ys[i]
        pn.line([bus, ox - 13], [oy, oy], col=P["PURPLE_S"], lw=1.0, z=1)
        mx = bus + 6
        gt_set = set(o["attention"]) | set(o["contacting"]) | set(o["spatial"])
        if pred:
            labs = [label_of(pred["attention"])] + labels_of(pred["contacting"]) + labels_of(pred["spatial"])
        else:
            labs = []
        xx = mx
        for j, lb in enumerate(labs):
            s = ab(lb) + (" ·" if j < len(labs) - 1 else "")
            pn.text(xx, oy - 3, s, size=7.6, col=P["GREEN"] if lb in gt_set else P["RED"], weight="semibold",
                    z=6)
            xx += pn.tw(s + " ", 7.6, weight="semibold")
        gts = " · ".join(ab(x) for x in o["attention"] + o["contacting"] + o["spatial"])
        pn.text(mx, oy + 9, f"GT: {gts}", size=6.8, col=P["DIM"], clip_w=ox - mx - 16, z=6)
        feat = o["label"] == pk["object"]
        ring = plt.Circle((ox, oy), 12.5, fc=P["NODE_F"], ec=P["ORANGE"] if feat else P["LINE"],
                          lw=(1.6 if feat else 1.0) * S * 0.75, ls="-" if o["visible"] else (0, (2, 1.5)), zorder=4)
        pn.ax.add_patch(ring)
        lab = short(o["label"])
        fs = min(6.9, 6.9 * 21 / max(1e-6, pn.tw(lab, 6.9)))
        pn.text(ox, oy + fs * 0.37, lab, size=fs, ha="center", z=6)
        if not o["visible"]:
            pn.text(ox + 15, oy + 2.6, "unseen", size=6.4, col=P["DIM"], style="italic", z=6)
    pn.ax.add_patch(plt.Circle((px, py), 14, fc=P["NODE_F"], ec=P["TXT"], lw=1.2 * S * 0.75, zorder=4))
    pn.text(px, py + 2.6, "person", size=7, ha="center", z=6)
    fnum = int(Path(pk["frame"]).stem)
    pn.text(w / 2, h - 5, f"frame {fnum} · green = in GT, red = not · class only, no 3-D boxes", size=7.2,
            col=P["DIM"], ha="center")
    pn.save(out / "scene_graph.png")


# ------------------------------------------------------------------------------------------
# Localized MLLM panels
# ------------------------------------------------------------------------------------------
def _cloud(b):
    z = np.load(b.root / "cache" / "cloud.npz")
    return z["xyz"], z["rgb"]


def _setup3d(pn, elev, azim, xyz, pad=0.1, box=None, zoom=1.0, rect=(0, 0, 1, 1)):
    ax = pn.fig.add_axes(list(rect), projection="3d")
    ax.set_facecolor(P["BG"])
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    lo, hi = (xyz.min(0), xyz.max(0)) if box is None else box
    ctr, rng = (lo + hi) / 2, (hi - lo) * (1 + pad)
    ax.set_xlim(ctr[0] - rng[0] / 2, ctr[0] + rng[0] / 2)
    ax.set_ylim(ctr[1] - rng[1] / 2, ctr[1] + rng[1] / 2)
    ax.set_zlim(ctr[2] - rng[2] / 2, ctr[2] + rng[2] / 2)
    ax.set_box_aspect(tuple(np.maximum(rng, 1e-3)), zoom=zoom)
    return ax


def _floor_grid(ax, x0, x1, y0, y1, step=0.5):
    for x in np.arange(math.ceil(x0 / step) * step, x1 + 1e-6, step):
        ax.plot([x, x], [y0, y1], [0, 0], color=P["GRIDLINE"], lw=0.5 * S * 0.75, zorder=0)
    for y in np.arange(math.ceil(y0 / step) * step, y1 + 1e-6, step):
        ax.plot([x0, x1], [y, y], [0, 0], color=P["GRIDLINE"], lw=0.5 * S * 0.75, zorder=0)


def _cams(b):
    C = np.array([np.asarray(p)[:3, 3] for p in b.video["camera_poses_pi3_final"] if p is not None])
    F = np.array([np.asarray(p)[:3, :3] @ np.array([0, 0, 1.0]) for p in b.video["camera_poses_pi3_final"]
                  if p is not None])
    return C, F


def _view_azim(F):
    f = F[:, :2].mean(0)
    return math.degrees(math.atan2(-f[1], -f[0]))


def lm_cloud(b, pk, out):
    w, h = 190, 112
    xyz, rgb = _cloud(b)
    C, F = _cams(b)
    pn = Panel(w, h)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xyz), min(len(xyz), 16000), replace=False)
    lo = np.minimum(xyz.min(0), C.min(0))
    hi = np.maximum(xyz.max(0), C.max(0))
    lo[2] = min(lo[2], 0)
    ax = _setup3d(pn, 24, _view_azim(F) + 35, xyz, pad=0.0, box=(lo, hi), zoom=1.75)
    _floor_grid(ax, lo[0], hi[0], lo[1], hi[1])
    ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], c=rgb[idx] / 255.0, s=2.4 * S, linewidths=0, depthshade=False)
    ax.plot(C[:, 0], C[:, 1], C[:, 2], color=P["ORANGE"], lw=1.8 * S * 0.75, zorder=10)
    ax.scatter(C[::4, 0], C[::4, 1], C[::4, 2], color=P["ORANGE"], s=7 * S, depthshade=False, zorder=11)
    ax.scatter(C[:1, 0], C[:1, 1], C[:1, 2], color=P["ORANGE"], s=22 * S, depthshade=False, zorder=12)
    ov = pn.fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, w)
    ov.set_ylim(h, 0)
    ov.axis("off")
    ov.patch.set_alpha(0)
    ov.text(6, h - 5, "floor z = 0 · camera path", fontsize=7.2 * S, color=P["DIM"], family=SANS_F)
    pn.save(out / "cloud.png")


def lm_bev_base(b, pk, out):
    w, h = 180, 124
    img = Image.open(b.root / "cache" / "bev.png").convert("RGB")
    m = b.meta
    pn = Panel(w, h)
    dx, dy, dw, dh = pn.image(img, 4, 4, w * 0.56, h - 8, border=P["FRAME_B"], lw=0.8, align="left")
    tx = dx + dw + 8
    rows = [("x₀", f"{m['x0']:.2f} m"), ("y₀", f"{m['y0']:.2f} m"), ("scale", f"{m['px_per_m']:.0f} px/m"),
            ("size", f"{m['width']}×{m['height']} px"), ("grid", "0.5 m")]
    for i, (k, v) in enumerate(rows):
        pn.text(tx, 16 + i * 21.5, k, size=7.2, col=P["DIM"])
        pn.text(tx, 26.5 + i * 21.5, v, size=7.8, col=P["TXT"], font="mono")
    pn.save(out / "bev_base.png")


def lm_proposals(b, pk, out, frame=None):
    w, h = 160, 96
    f = frame or pk["frame"]
    img = Image.open(b.root / "frames_annotated" / f).convert("RGB")
    dets = b.dets["frames"].get(f, [])
    pay = b.payload("sgdet", Path(f).stem)
    ids = {(o["label"], tuple(round(v, 1) for v in o["bbox"])): o["id"] for o in pay["objects"] if o.get("bbox")}
    pn = Panel(w, h)
    dx, dy, dw, dh = pn.image(img, 3, 3, w * 0.5, h - 6, border=P["FRAME_B"], lw=0.6, align="left")
    s = dw / img.width
    rows = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        x1, y1, x2, y2 = d["bbox"]
        if d["label"] == "person":
            col, tag = "#111111", "P"
        else:
            i = ids.get((d["label"], tuple(round(v, 1) for v in d["bbox"])))
            col = MARK_HEX[(i - 1) % len(MARK_HEX)] if i else P["DIM"]
            tag = str(i) if i else "·"
        pn.rect(dx + x1 * s, dy + y1 * s, (x2 - x1) * s, (y2 - y1) * s, ec=col, lw=1.2, z=4)
        rows.append((tag, d["label"], d["score"], col))
    tx = dx + dw + 7
    pn.text(tx, 12, "GDino ≥ 0.25", size=7.2, col=P["DIM"])
    for i, (tag, lab, sc, col) in enumerate(rows[:6]):
        y = 25 + i * 11
        pn.rect(tx, y - 7, 7, 7, fc=col, z=3)
        pn.text(tx + 10, y, f"{short(lab)}", size=7.4, col=P["TXT"], clip_w=w - tx - 30)
        pn.text(w - 5, y, f"{sc:.2f}", size=7, col=P["MUTED"], ha="right", font="mono")
    if len(rows) > 6:
        pn.text(tx, 25 + 6 * 11, f"+{len(rows) - 6} more", size=6.8, col=P["DIM"])
    pn.save(out / "proposals_2d.png")


def _edges(c):
    c = np.asarray(c).reshape(8, 3)
    zmid = (c[:, 2].min() + c[:, 2].max()) / 2
    bot = c[c[:, 2] <= zmid]
    top = c[c[:, 2] > zmid]
    if len(bot) != 4 or len(top) != 4:
        return []

    def ring(q):
        ctr = q[:, :2].mean(0)
        return q[np.argsort(np.arctan2(q[:, 1] - ctr[1], q[:, 0] - ctr[0]))]
    bot, top = ring(bot), ring(top)
    top = np.array([top[np.argmin(np.linalg.norm(top[:, :2] - p[:2], axis=1))] for p in bot])
    e = [(bot[i], bot[(i + 1) % 4]) for i in range(4)] + [(top[i], top[(i + 1) % 4]) for i in range(4)]
    e += [(bot[i], top[i]) for i in range(4)]
    return e


def lm_lifted(b, pk, out, frame=None):
    w, h = 180, 100
    f = frame or pk["frame"]
    pay = b.payload("sgdet", Path(f).stem)
    boxes = [(o["id"], o["label"], np.asarray(o["corners"])) for o in pay["objects"] if o.get("corners")]
    pc = np.asarray(pay["person_corners"]) if pay["person_corners"] else None
    xyz, rgb = _cloud(b)
    allc = np.concatenate([c for _, _, c in boxes] + ([pc] if pc is not None else []), 0) if boxes or pc is not None \
        else xyz
    lo, hi = allc.min(0) - 0.35, allc.max(0) + 0.35
    lo[2] = 0.0
    m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    pts, cols = xyz[m], rgb[m]
    rng = np.random.default_rng(1)
    if len(pts) > 9000:
        k = rng.choice(len(pts), 9000, replace=False)
        pts, cols = pts[k], cols[k]
    C, F = _cams(b)
    pn = Panel(w, h)
    ax = _setup3d(pn, 26, _view_azim(F) + 25, pts, pad=0.0, box=(lo, hi), zoom=1.6)
    _floor_grid(ax, lo[0], hi[0], lo[1], hi[1], step=0.25)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols / 255.0, s=1.8 * S, alpha=0.6, linewidths=0,
               depthshade=False)
    for i, lab, c in boxes:
        col = MARK_HEX[(i - 1) % len(MARK_HEX)]
        for a, bb in _edges(c):
            ax.plot(*zip(a, bb), color=col, lw=1.3 * S * 0.75, zorder=20)
    if pc is not None:
        for a, bb in _edges(pc):
            ax.plot(*zip(a, bb), color=PERSON_COL, lw=1.1 * S * 0.75, zorder=20)
    ov = pn.fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, w)
    ov.set_ylim(h, 0)
    ov.axis("off")
    ov.patch.set_alpha(0)
    labs = [f"{i} {short(lab)}" for i, lab, _ in boxes]
    x = 6
    for (i, lab, _), s in zip(boxes, labs):
        ov.add_patch(FancyBboxPatch((x, 5), 6, 6, boxstyle="square,pad=0", fc=MARK_HEX[(i - 1) % len(MARK_HEX)],
                                    ec="none"))
        ov.text(x + 8.5, 11, s, fontsize=7 * S, color=P["MUTED"], family=SANS_F)
        x += 16 + pn.tw(s, 7)
    ov.text(6, h - 5, f"frame {int(Path(f).stem)} · P = person", fontsize=7 * S, color=P["DIM"], family=SANS_F,
            bbox=dict(boxstyle="square,pad=0.15", fc=P["BG"], ec="none", alpha=0.85))
    pn.save(out / "lifted_obbs.png")


def _legend_ids(pn, objs, x, y, size=7.4, lh=11.5, maxn=6):
    for i, o in enumerate(objs[:maxn]):
        col = MARK_HEX[(o["id"] - 1) % len(MARK_HEX)]
        yy = y + i * lh
        pn.rect(x, yy - 7, 7, 7, fc=col, z=3)
        vis = o.get("visible", True)
        pn.text(x + 10, yy, f"{o['id']} {short(o['label'])}", size=size, col=P["TXT"])
        if not vis:
            pn.text(x + 10, yy + 8.5, "not visible", size=6.4, col=P["DIM"], style="italic")
            y += 7


def lm_payload(b, pk, out, mode=None, frame=None, retrieved=False):
    """Images 1-4 and the text of one Track A payload; with ``retrieved`` the text card
    shows where Track B's retrieved block sits (before "Task:")."""
    mode = mode or pk["stage2_mode"]
    stem = Path(frame or pk["frame"]).stem
    pay = b.payload(mode, stem)
    imgs = pay["images"]
    # image 1: marked target + id legend
    pn = Panel(170, 110)
    dx, dy, dw, dh = pn.image(imgs[0], 3, 3, 170 * 0.52, 104, border=P["ORANGE"], lw=1.4, align="left")
    pn.text(dx + dw + 7, 13, "ids in this prompt", size=7, col=P["DIM"])
    _legend_ids(pn, pay["objects"], dx + dw + 7, 26)
    pn.save(out / "marked_frame.png")
    for j, key in ((1, "context_0"), (2, "context_1")):
        pn = Panel(110, 70)
        pn.image(imgs[j], 3, 3, 104, 64, border=P["FRAME_B"], lw=0.6)
        pn.save(out / f"{key}.png")
    pn = Panel(170, 112)
    dx, dy, dw, dh = pn.image(imgs[-1], 3, 3, 170 * 0.6, 106, border=P["FRAME_B"], lw=0.6, align="left")
    tx = dx + dw + 7
    for i, (k, v) in enumerate((("P", "person"), ("1…N", "object ids"), ("→", "camera, red"), ("·", "camera path"))):
        pn.text(tx, 20 + i * 13, k, size=7.4, col=P["TXT"], weight="semibold")
        pn.text(tx + 20, 20 + i * 13, v, size=7, col=P["MUTED"])
    pn.save(out / "marked_bev.png")
    # prompt excerpt
    txt = (pay["prompt_b"] if retrieved else pay["prompt_a"]).splitlines()
    keep = []
    for ln in txt:
        if ln.startswith("- P person") or ln.startswith("- camera") or re.match(r"- \d+ ", ln):
            keep.append(ln)
    task = next((ln for ln in txt if ln.startswith("Task:")), "")
    scene = next((ln for ln in txt if ln.startswith("Scene at") or ln.startswith("Detector proposals")), "")
    w, h = 230, 112
    pn = Panel(w, h, card=True)
    pn.text(9, 14, scene, size=7, col=P["DIM"], font="mono", clip_w=w - 16)
    y = 25.5
    n_keep = 4 if retrieved else 7
    for ln in keep[:n_keep]:
        ln = ln.replace(" in the target frame", "").replace(" m, size=", " size=")
        col = P["ORANGE"] if "NOT visible" in ln else P["TXT"]
        pn.text(9, y, ln, size=6.9, font="mono", col=col, clip_w=w - 16)
        y += 10.2
    if retrieved:
        ri = next((i for i, ln in enumerate(txt) if ln.startswith("Retrieved context")), None)
        if ri is not None:
            pn.text(9, y, txt[ri], size=6.9, font="mono", col=P["ORANGE"], clip_w=w - 16)
            y += 10.2
            at = next((ln for ln in txt[ri + 1:] if "at the target" in ln), txt[ri + 1] if ri + 1 < len(txt) else "")
            pn.text(9, y, at, size=6.9, font="mono", col=P["ORANGE"], clip_w=w - 16)
            y += 10.2
            pn.text(9, y, "…", size=6.9, font="mono", col=P["ORANGE"])
            y += 10.2
    pn.text(9, min(y + 2, h - 8), task, size=6.9, font="mono", col=P["MUTED"], clip_w=w - 16)
    pn.save(out / "prompt_card.png")


def _objects_json(raw):
    txt = strip_fences(raw or "")
    if "</think>" in txt:
        txt = txt.rsplit("</think>", 1)[1]
    i = txt.find("{")
    try:
        return json.loads(txt[i:]) if i >= 0 else None
    except json.JSONDecodeError:
        items = []
        for m in re.finditer(r"\{[^{}]*\}", txt):
            try:
                items.append(json.loads(m.group()))
            except json.JSONDecodeError:
                pass
        return {"objects": items, "truncated": True}


def lm_answer(b, pk, out):
    w, h = 272, 116
    fr = b.ta[pk["stage2_mode"]]["frames"][pk["frame"]]
    d = _objects_json(fr["raw_response"]) or {"objects": []}
    items = d.get("objects") or []
    n_ids = len(fr.get("ids") or {})
    pn = Panel(w, h, card=True)
    card_head(pn, f"Answer, frame {int(Path(pk['frame']).stem)} ({pk['stage2_mode']})", size=8.4)
    size, lh = 7.2, 10.2
    y = 27
    lines = []
    for it in items[:n_ids]:
        lines += pn.wrap(json.dumps(it, separators=(", ", ": ")), w - 18, size, "mono")
    nmax = int((h - 30) // lh)
    for j, ln in enumerate(lines[:nmax]):
        pn.text(9, y + j * lh, ln, size=size, font="mono", col=P["TXT"])
    extra = len(items) - n_ids
    note = f"{n_ids} ids asked"
    if extra > 0:
        note += f" · {extra} more entries invented past the list (ignored)"
    if d.get("truncated"):
        note += " · cut at 1,024 tokens"
    pn.text(9, h - 7, note, size=6.6, col=P["DIM"], clip_w=w - 14)
    pn.save(out / "answer_card.png")


def lm_graph(b, pk, out):
    w, h = 272, 66
    mode = pk["stage2_mode"]
    fr = b.ta[mode]["frames"][pk["frame"]]
    gt = {o["label"]: o for o in b.frames[pk["frame"]]["objects"]}
    pay = b.payload(mode, Path(pk["frame"]).stem)
    objs = pay["objects"]
    pn = Panel(w, h)
    n = max(1, len(objs))
    cw = (w - 8) / n
    for k, o in enumerate(objs):
        x = 4 + k * cw
        col = MARK_HEX[(o["id"] - 1) % len(MARK_HEX)]
        pn.rect(x + 1, 3, cw - 2, 12, fc=col, r=2, alpha=0.9, z=2)
        tag = f"{o['id']} {short(o['label'])}" + ("" if o.get("visible", True) else " · unseen")
        pn.text(x + cw / 2, 12, tag, size=6.9, col="#ffffff", ha="center", weight="semibold", clip_w=cw - 4, z=6)
        pred = (fr["objects"] or {}).get(o["label"])
        g = gt.get(o["label"])
        gset = set(g["attention"] + g["contacting"] + g["spatial"]) if g else set()
        for r, head in enumerate(("attention", "contacting", "spatial")):
            labs = (pred or {}).get(head) or []
            s = ", ".join(ab(x_) for x_ in labs) if labs else "—"
            ok = labs and all(x_ in gset for x_ in labs)
            some = labs and any(x_ in gset for x_ in labs)
            c = P["GREEN"] if ok else (P["ORANGE"] if some else P["RED"])
            pn.text(x + cw / 2, 27 + r * 11, s, size=6.9, col=c if pred else P["DIM"], ha="center", clip_w=cw - 4)
    pn.text(w / 2, h - 3, "predicates on the frame's boxes · green = in GT, orange = partly, red = not", size=6.4,
            col=P["DIM"], ha="center")
    pn.save(out / "localized_graph.png")


def lm_retrieved(b, pk, out):
    w, h = 300, 72
    crit = pk["critic"] or {"mode": "predcls", "frame": pk["frame"]}
    fr = b.tb[crit["mode"]]["frames"][crit["frame"]]
    pn = Panel(w, h, card=True)
    card_head(pn, f"Retrieved Context · frame {int(Path(crit['frame']).stem)} · ≤ 900 Characters", size=8.2)
    lines = [ln for ln in (fr.get("retrieved") or "").splitlines() if ln.strip()]
    size, lh = 6.9, 10
    nmax = int((h - 22) // lh)
    if len(lines) > nmax:
        at = next((i for i, ln in enumerate(lines) if "at the target" in ln), len(lines) // 2)
        lo = max(0, min(at - nmax // 2, len(lines) - nmax))
        lines = lines[lo: lo + nmax]
    for j, ln in enumerate(lines):
        col = P["ORANGE"] if "at the target" in ln else P["TXT"]
        pn.text(9, 26 + j * lh, ln, size=size, font="mono", col=col, clip_w=w - 16)
    pn.save(out / "retrieved_card.png")


def lm_violations(b, pk, out):
    w, h = 236, 100
    crit = pk["critic"]
    pn = Panel(w, h)
    pn.rect(3, 3, w - 6, h - 6, fc=P["LOSS_F"], ec=P["RED"], lw=1.2, r=6, ls=(0, (5, 3)))
    if not crit:
        pn.text(12, 20, "No frame flagged by the critic", size=9, col=P["RED"], weight="bold")
        pn.save(out / "violations_card.png")
        return
    fr = b.tb[crit["mode"]]["frames"][crit["frame"]]
    v = fr["violations_pre"]
    pn.text(12, 18, f"Violations → Repair Prompt ({len(v)})", size=9, col=P["RED"], weight="bold")
    size, lh = 6.9, 9.8
    lines = []
    for x in v[:12]:
        lines += pn.wrap(x["msg"], w - 26, size, "mono")
    nmax = int((h - 34) // lh)
    for j, ln in enumerate(lines[:nmax]):
        if j == nmax - 1 and len(lines) > nmax:
            ln = clip(ln, w - 40, size, True) + " …"
        pn.text(12, 32 + j * lh, ln, size=size, font="mono", col=P["RED_TXT"])
    pn.save(out / "violations_card.png")


def _foot(corners):
    c = np.asarray(corners, np.float64).reshape(8, 3)
    zmid = 0.5 * (c[:, 2].min() + c[:, 2].max())
    bot = c[c[:, 2] <= zmid][:, :2]
    if len(bot) < 3:
        bot = c[:4, :2]
    ctr = bot.mean(0)
    return bot[np.argsort(np.arctan2(bot[:, 1] - ctr[1], bot[:, 0] - ctr[0]))]


def _change_text(lab, a, c):
    """One-line description of what the repair changed for one object."""
    parts = []
    if a and c and a.get("corners") and c.get("corners"):
        ca, cc = np.asarray(a["corners"]).mean(0), np.asarray(c["corners"]).mean(0)
        if abs(cc[2] - ca[2]) > 0.02:
            parts.append(f"centre z {ca[2]:.2f} → {cc[2]:.2f} m")
        dxy = float(np.linalg.norm(cc[:2] - ca[:2]))
        if dxy > 0.02:
            parts.append(f"moved {dxy:.2f} m")
    for head in ("attention", "contacting", "spatial"):
        pa, pc_ = (a or {}).get(head) or [], (c or {}).get(head) or []
        if pa != pc_:
            parts.append(f"{', '.join(pa) or '—'} → {', '.join(pc_) or '—'}")
    if a is None:
        parts.append("added")
    if c is None:
        parts.append("dropped")
    return f"{short(lab)}: " + "; ".join(parts)


def lm_before_after(b, pk, out):
    w, h = 360, 110
    crit = pk["critic"]
    pn = Panel(w, h)
    if not crit:
        pn.save(out / "critic_before_after.png")
        return
    mode, f = crit["mode"], crit["frame"]
    fr = b.tb[mode]["frames"][f]
    pay = b.payload(mode, Path(f).stem)
    idcol = {o["label"]: MARK_HEX[(o["id"] - 1) % len(MARK_HEX)] for o in pay["objects"]}
    flagged = {x["label"] for x in fr["violations_pre"]}
    pre, post = fr["objects_pre"], fr["objects"]
    changed = sorted(l for l in set(pre) | set(post) if (pre.get(l) or {}) != (post.get(l) or {}))
    focus = [l for l in changed if l in flagged] or changed or sorted(flagged)
    pc = np.asarray(pay["person_corners"]) if pay["person_corners"] else None
    boxes = [np.asarray(o["corners"]) for d in (pre, post) for l, o in d.items() if l in focus and o.get("corners")]
    if pc is not None:
        boxes.append(pc)
    allc = np.concatenate(boxes, 0)
    lo, hi = allc.min(0) - 0.3, allc.max(0) + 0.3
    lo[2] = min(lo[2], 0.0)
    xyz, rgb = _cloud(b)
    m = np.all((xyz[:, :2] >= lo[:2]) & (xyz[:, :2] <= hi[:2]), axis=1) & (xyz[:, 2] <= hi[2])
    pts, cols = xyz[m], rgb[m]
    if len(pts) > 6000:
        k = np.random.default_rng(2).choice(len(pts), 6000, replace=False)
        pts, cols = pts[k], cols[k]
    C, F = _cams(b)
    text_h = 26
    for side, (objs, before) in enumerate(((pre, True), (post, False))):
        ax = _setup3d(pn, 14, _view_azim(F) + 30, pts, pad=0.0, box=(lo, hi), zoom=1.45,
                      rect=(0.02 + side * 0.53, text_h / h, 0.43, 1 - text_h / h - 0.01))
        _floor_grid(ax, lo[0], hi[0], lo[1], hi[1], step=0.5)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols / 255.0, s=1.4 * S, alpha=0.35, linewidths=0,
                       depthshade=False)
        if pc is not None:
            for a_, b_ in _edges(pc):
                ax.plot(*zip(a_, b_), color=PERSON_COL, lw=1.0 * S * 0.75, zorder=20)
        for lab, o in objs.items():
            if not o.get("corners"):
                continue
            c = np.asarray(o["corners"])
            if np.any(c.mean(0)[:2] < lo[:2]) or np.any(c.mean(0)[:2] > hi[:2]):
                continue
            if lab in focus:
                col = P["RED"] if before and lab in flagged else (P["GREEN"] if not before else idcol.get(lab, P["ORANGE"]))
                lw, ls = 1.7, ((0, (3, 2)) if before and lab in flagged else "-")
            else:
                col, lw, ls = idcol.get(lab, P["DIM"]), 0.8, "-"
            for a_, b_ in _edges(c):
                ax.plot(*zip(a_, b_), color=col, lw=lw * S * 0.75, ls=ls, zorder=21)
    ov = pn.fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, w)
    ov.set_ylim(h, 0)
    ov.axis("off")
    ov.patch.set_alpha(0)
    ya = (h - text_h) / 2
    ov.annotate("", xy=(0.55 * w - 4, ya), xytext=(0.45 * w + 4, ya),
                arrowprops=dict(arrowstyle="-|>", color=P["ORANGE"], lw=1.6 * S * 0.75, mutation_scale=9 * S))
    kinds = Counter(x["kind"] for x in fr["violations_pre"])
    k_txt = ", ".join(f"{k} ×{n}" if n > 1 else k for k, n in kinds.items())
    post_n = len(fr["violations_post"])
    fnum = int(Path(f).stem)
    ov.text(0.235 * w, h - 15, f"before ({mode}, frame {fnum}): {k_txt}", fontsize=7.2 * S, color=P["RED"],
            family=SANS_F, ha="center")
    ov.text(0.765 * w, h - 15, "after one repair: " + (f"{post_n} flag left" if post_n == 1 else
                                                       f"{post_n} flags left" if post_n else "no flag left"),
            fontsize=7.2 * S, color=P["GREEN"] if not post_n else P["ORANGE"], family=SANS_F, ha="center")
    log = " · ".join(_change_text(l, pre.get(l), post.get(l)) for l in focus[:2])
    ov.text(w / 2, h - 4, pn.fit_text(log, w - 10, 6.8), fontsize=6.8 * S, color=P["MUTED"], family=SANS_F,
            ha="center")
    pn.save(out / "critic_before_after.png")


def _crit(pk):
    return pk["critic"] or {"mode": "predcls", "frame": pk["frame"]}


def _raw_card(pn, raw, y0, w, h, size=7.0, lh=10.0, foot=None):
    lines = []
    for ln in strip_fences(raw or "").splitlines():
        lines += pn.wrap(ln.rstrip(), w - 18, size, "mono") or [""]
    nmax = int((h - y0 - (12 if foot else 4)) // lh)
    for j, ln in enumerate(lines[:nmax]):
        if j == nmax - 1 and len(lines) > nmax:
            ln = clip(ln, w - 30, size, True) + " …"
        pn.text(9, y0 + j * lh, ln, size=size, font="mono", col=P["TXT"])
    if foot:
        pn.text(9, h - 7, foot, size=6.6, col=P["DIM"], clip_w=w - 14)


def tb_proposal(b, pk, out):
    """Track B call 1: the raw first proposal of the critic's frame (the −critic arm)."""
    w, h = 210, 96
    crit = _crit(pk)
    fr = b.tb[crit["mode"]]["frames"][crit["frame"]]
    pn = Panel(w, h, card=True)
    card_head(pn, f"Call 1 · frame {int(Path(crit['frame']).stem)} ({crit['mode']})", size=8.2)
    n = len(fr["objects_pre"])
    _raw_card(pn, fr["raw_response_pre"], 27, w, h,
              foot=f"{n} object{'s' if n != 1 else ''} parsed → objects_pre" + (" · parsed" if fr["parsed"] else ""))
    pn.save(out / "proposal_card.png")


def tb_repair(b, pk, out):
    """Track B call 2: the raw repaired answer of the critic's frame."""
    w, h = 210, 96
    crit = _crit(pk)
    fr = b.tb[crit["mode"]]["frames"][crit["frame"]]
    pn = Panel(w, h, card=True)
    card_head(pn, f"Call 2 · Repair · {fr['n_calls']} calls for this frame", size=8.2)
    if fr["n_calls"] < 2:
        pn.text(9, 34, "no repair call: the critic did not fire", size=7.4, col=P["DIM"])
    else:
        _raw_card(pn, fr["raw_response"], 27, w, h,
                  foot=f"{len(fr['objects'])} objects parsed → objects · {len(fr['violations_post'])} flag(s) left")
    pn.save(out / "repair_card.png")


def tb_graph(b, pk, out):
    """The emitted (post-repair) localized graph of the critic's frame: one column per object."""
    w, h = 272, 66
    crit = _crit(pk)
    fr = b.tb[crit["mode"]]["frames"][crit["frame"]]
    gt = {o["label"]: o for o in b.frames[crit["frame"]]["objects"]}
    ids = {v: int(k) for k, v in fr["ids"].items()}
    objs = list(fr["objects"].items())
    pn = Panel(w, h)
    n = max(1, len(objs))
    cw = (w - 8) / n
    for k, (lab, o) in enumerate(objs):
        x = 4 + k * cw
        i = ids.get(lab)
        col = MARK_HEX[(i - 1) % len(MARK_HEX)] if i else P["ORANGE"]
        pn.rect(x + 1, 3, cw - 2, 12, fc=col, r=2, alpha=0.9, z=2)
        tag = f"{i if i else 'new'} {short(lab)}"
        pn.text(x + cw / 2, 12, tag, size=6.9, col="#ffffff", ha="center", weight="semibold", clip_w=cw - 4, z=6)
        g = gt.get(lab)
        gset = set(g["attention"] + g["contacting"] + g["spatial"]) if g else set()
        for r, head in enumerate(("attention", "contacting", "spatial")):
            labs = o.get(head) or []
            s = ", ".join(ab(x_) for x_ in labs) if labs else "—"
            ok = labs and all(x_ in gset for x_ in labs)
            some = labs and any(x_ in gset for x_ in labs)
            c = (P["GREEN"] if ok else (P["ORANGE"] if some else P["RED"])) if g else P["MUTED"]
            pn.text(x + cw / 2, 27 + r * 11, s, size=6.9, col=c, ha="center", clip_w=cw - 4)
    pn.text(w / 2, h - 3, f"objects after repair · frame {int(Path(crit['frame']).stem)} ({crit['mode']}) · "
            "green = in GT, orange = partly, red = not, grey = no GT slot", size=6.2, col=P["DIM"], ha="center")
    pn.save(out / "repaired_graph.png")


# ------------------------------------------------------------------------------------------
# U-WSGG-Zero / U-WSGG-Sub panels (from the capture wrappers of the two runners)
# ------------------------------------------------------------------------------------------
def ul_context_frames(b, pk, out, method):
    """The shared annotated context C: the ≤ 15 key frames every Q(f) of the video carries."""
    w, h = 250, 92
    t = b.tensors("predcls", method)
    C = t["q_context"]
    pn = Panel(w, h)
    n = len(C)
    rows = 2
    cols = math.ceil(n / rows)
    lab_h = 12
    th = h - lab_h - 6
    ch = (th - 3) / rows
    cw = ch * C.shape[2] / C.shape[1]
    gw = cols * cw + (cols - 1) * 2
    x0 = (w - gw) / 2
    for i in range(n):
        r, c = divmod(i, cols)
        pn.image(C[i], x0 + c * (cw + 2), 3 + r * (ch + 3), cw, ch, border=P["FRAME_B"], lw=0.5)
    H, W = C.shape[1:3]
    pn.text(w / 2, h - 4, f"{n} of {len(b.video['frames'])} key frames (linspace) · {W}×{H} each", size=7.4,
            col=P["DIM"], ha="center")
    pn.save(out / "context_frames.png")


def _prompt_rec(b, pk, method, mode="predcls"):
    cap = b.caps[method][mode]
    return next(p for p in cap["prompts"] if p["frame"] == pk["frame"] and p["object"] == pk["object"])


def ul_prompt(b, pk, out, method):
    """The exact text of the P4 prompt for (f, o): zero_shot shows the question, caption_all the
    caption prefix in the order the runner sorted it, followed by the question's head."""
    rec = _prompt_rec(b, pk, method)
    text = rec["text"]
    obj, fnum = pk["object"], int(Path(pk["frame"]).stem)
    size, lh = 6.8, 9.6
    if method == "zero_shot":
        w, h = 262, 146
        pn = Panel(w, h, card=True)
        card_head(pn, f"Question P₄(o) · o = {short(obj)} · same text at every frame", size=8)
        para = text.split("\n\n")[0]
        lines = pn.wrap(para, w - 18, size, "mono")[:4]
        y = 27
        for ln in lines:
            pn.text(9, y, ln, size=size, font="mono", col=P["TXT"], clip_w=w - 16)
            y += lh
        for num, head, key in ((1, "ATTENTION · exactly one of: ", "Pick EXACTLY ONE label from: "),
                               (2, "CONTACTING · one or more of: ", "Pick ONE OR MORE labels from: "),
                               (3, "SPATIAL · one or more of: ", "Pick ONE OR MORE labels from: ")):
            m = re.search(rf"{num}\. .*?\n\s*{re.escape(key)}(.*)", text)
            labels = m.group(1).strip() if m else ""
            n_lab = len(labels.split(", "))
            s = f"{num}. {head}{labels}"
            s = pn.fit_text(s, w - 18 - (34 if n_lab > 6 else 0), size, "mono") + (f" ({n_lab})" if n_lab > 6 else "")
            pn.text(9, y, s, size=size, font="mono", col=P["MUTED"], clip_w=w - 16)
            y += lh
        tail = text[text.find("Respond ONLY"):].replace("\n", " ")
        tail = re.sub(r"\s+", " ", tail)
        for ln in pn.wrap(tail, w - 18, size, "mono")[:3]:
            pn.text(9, y, ln, size=size, font="mono", col=P["TXT"], clip_w=w - 16)
            y += lh
        pn.text(9, h - 7, f"{len(text)} characters · identical for every frame f · ≤ {rec['max_new_tokens']} new tokens",
                size=6.6, col=P["DIM"], clip_w=w - 14)
    else:
        w, h = 290, 150
        pn = Panel(w, h, card=True)
        prefix, _, question = text.partition("\n\n")
        cap_lines = [ln for ln in prefix.splitlines() if ln.startswith("[Frame")]
        card_head(pn, f"Prompt For (f = {fnum}, o = {short(obj)}) · Caption Prefix c(f), Nearest First", size=8)
        y = 27
        pn.text(9, y, prefix.splitlines()[0], size=size, font="mono", col=P["MUTED"], clip_w=w - 16)
        y += lh
        for ln in cap_lines[:4]:
            k = int(re.match(r"\[Frame (\d+)\]", ln).group(1))
            pn.text(9, y, ln, size=size, font="mono", col=P["ORANGE"] if k == fnum else P["TXT"], clip_w=w - 16)
            y += lh
        pn.text(9, y, f"… {len(cap_lines)} caption lines, {len(prefix)} characters, sorted by |kᵢ − {fnum}|",
                size=size, font="mono", col=P["DIM"], clip_w=w - 16)
        y += lh + 3
        pn.line([9, w - 9], [y - 6, y - 6], col=P["RULE"], lw=0.5)
        q1 = question.split("\n\n")[0]
        for ln in pn.wrap(q1, w - 18, size, "mono")[:3]:
            pn.text(9, y, ln, size=size, font="mono", col=P["TXT"], clip_w=w - 16)
            y += lh
        pn.text(9, y, "… the same three questions and JSON format as zero_shot", size=size, font="mono",
                col=P["MUTED"], clip_w=w - 16)
        pn.text(9, h - 7, f"{len(text)} characters in total · ≤ {rec['max_new_tokens']} new tokens", size=6.6,
                col=P["DIM"], clip_w=w - 14)
    pn.save(out / "prompt_card.png")


def ul_discovery(b, pk, out, method):
    """The P6 object-estimation prompt and raw answer of the SGDet run."""
    w, h = 262, 92
    e = (b.caps[method]["sgdet"] or {}).get("estimation")
    pn = Panel(w, h, card=True)
    card_head(pn, "P₆ Object Estimation · SGDet", size=8.2)
    if not e:
        pn.text(9, 34, "no SGDet capture", size=7.4, col=P["DIM"])
        pn.save(out / "discovery_card.png")
        return
    lines = e["prompt"].splitlines()
    size, lh = 6.6, 9.4
    caps = [ln for ln in lines if ln.startswith("[Frame")]
    ctx = next((ln for ln in lines if ln.startswith("(No captions")), None)
    show = [(lines[0], P["TXT"])]
    if ctx:
        show.append((ctx, P["ORANGE"]))
    else:
        show.append((caps[0], P["ORANGE"]))
        show.append((f"… {len(caps)} caption lines", P["ORANGE"]))
    cand = next((ln for ln in lines if ln.startswith('"')), "")
    show.append(("candidates: " + cand, P["MUTED"]))
    show.append(("→ " + strip_fences(e["raw_response"]).replace("\n", " "), P["TXT"]))
    y = 27
    for s, col in show[:5]:
        pn.text(9, y, s, size=size, font="mono", col=col, clip_w=w - 16)
        y += lh
    vs = e.get("video_shape") or []
    pn.text(9, h - 7, f"{len(e['parsed'])} names parsed · ≤ {e['max_new_tokens']} tokens · V = {vs[0] if vs else '?'} frames",
            size=6.6, col=P["DIM"], clip_w=w - 14)
    pn.save(out / "discovery_card.png")


def ul_transcript(b, pk, out, method="caption_all"):
    """Every Stage-1 caption in time order, as caption_all reads them from the pickle."""
    w, h = 252, 150
    caps = b.caps[method]["predcls"]["captions"]
    fnum = int(Path(pk["frame"]).stem)
    pn = Panel(w, h, card=True)
    card_head(pn, f"Caption Transcript · {len(caps)} Captions, Time Order", size=8.2)
    size, lh = 6.6, 9.4
    nmax = int((h - 36) // lh)
    for j, (k, t) in enumerate(caps[:nmax]):
        pn.text(9, 27 + j * lh, f"[Frame {k}] {t}", size=size, font="mono",
                col=P["ORANGE"] if k == fnum else P["TXT"], clip_w=w - 16)
    rest = len(caps) - nmax
    pn.text(9, h - 7, (f"… {rest} more · " if rest > 0 else "") + f"orange = the caption of f = {fnum}", size=6.6,
            col=P["DIM"], clip_w=w - 14)
    pn.save(out / "transcript.png")


def named(fn, name, **kw):
    """``fn`` bound to ``kw`` under a panel name (for --only and error messages)."""
    def run(b, pk, out):
        return fn(b, pk, out, **kw)
    run.__name__ = name
    return run


def _unloc(method):
    return [named(gr_frames, f"{method}_frames"), named(ul_context_frames, f"{method}_context", method=method),
            named(gr_query, f"{method}_query", method=method), named(gr_video_v, f"{method}_video_v", method=method),
            named(gr_objects, f"{method}_objects", method=method),
            named(ul_discovery, f"{method}_discovery", method=method),
            named(ul_prompt, f"{method}_prompt", method=method), named(gr_answer, f"{method}_answer", method=method),
            named(gr_scene_graph, f"{method}_scene_graph", method=method)]


GRAPH_RAG = [gr_frames, gr_segments, gr_caption, gr_node, gr_event_graph, gr_video_v, gr_objects, gr_keywords,
             gr_ranked, gr_context, gr_query, gr_answer, gr_scene_graph]
ZERO_SHOT = _unloc("zero_shot")
CAPTION_ALL = _unloc("caption_all") + [named(gr_segments, "caption_all_segments"),
                                       named(gr_caption, "caption_all_caption"),
                                       named(ul_transcript, "caption_all_transcript")]
TRACK_A = [gr_frames, lm_cloud, lm_bev_base, lm_proposals, lm_lifted, lm_payload, lm_answer, lm_graph]


def _track_b(pk):
    crit = _crit(pk)
    return [gr_frames, lm_cloud, lm_bev_base,
            named(lm_proposals, "tb_proposals", frame=crit["frame"]),
            named(lm_lifted, "tb_lifted", frame=crit["frame"]),
            named(lm_payload, "tb_payload", mode=crit["mode"], frame=crit["frame"], retrieved=True),
            lm_retrieved, tb_proposal, lm_violations, tb_repair, tb_graph, lm_before_after]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--object", default=None)
    ap.add_argument("--theme", default="light", choices=["light", "dark"], help="read by common.py at import")
    ap.add_argument("--only", default=None, help="comma-separated panel function names")
    ap.add_argument("--figures", default="zero_shot,caption_all,graph_rag,track_a,track_b",
                    help="comma-separated figure directories to write")
    args = ap.parse_args()
    b = Bundle(Path(args.bundle))
    pk = make_picks(b, args)
    out = Path(args.out)
    sfx = "" if THEME == "light" else "_dark"
    groups = {"zero_shot": ZERO_SHOT, "caption_all": CAPTION_ALL, "graph_rag": GRAPH_RAG, "track_a": TRACK_A,
              "track_b": _track_b(pk)}
    only = set(args.only.split(",")) if args.only else None
    for fig in args.figures.split(","):
        fns = groups[fig]
        if fig in ("zero_shot", "caption_all") and not (b.runs[fig]["predcls"] and b.caps[fig]["predcls"]):
            print(f"  {fig}: no run / capture in the bundle, skipped")
            continue
        d = out / f"{fig}{sfx}"
        d.mkdir(parents=True, exist_ok=True)
        for fn in fns:
            if only and fn.__name__ not in only:
                continue
            try:
                fn(b, pk, d)
            except Exception as e:  # noqa: BLE001 - one broken panel must not block the others
                print(f"  {fig}/{fn.__name__}: FAILED {e!r}")
    (out / "mllm_picks.json").write_text(json.dumps(pk, indent=2), encoding="utf-8")
    print(json.dumps(pk, indent=2))


if __name__ == "__main__":
    main()
