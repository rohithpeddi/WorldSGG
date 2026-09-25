#!/usr/bin/env python3
"""Publication panels for the semantic-annotation stage (supp. Sec. semantic annotation).

Reads the bundle written by ``extract_semantic_pipeline.py``
(``outputs/semantic_pipeline/<vid>/bundle.json`` + ``frames/``) and renders
compact, light-style, transparent 300-dpi PNGs sized for a half-text-width
wrap figure into ``outputs/semantic_pipeline/panels_<vid>/``:

  m1_clip_input               target frame (GT boxes) + the clip frames the VLM sees
  m2_event_graph              merged coarse event graph (clip nodes <-> entity keys)
  m3_candidates_verification  candidate labels per axis with verification p_yes
  m4_label_timeline           contacting labels over all annotated frames, VLM vs final
  m5_corrections              VLM seed -> human correction for unobserved objects

Usage::

    python panels_semantic.py --video 00T1E --target 000273
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

_repo = Path(__file__).resolve().parents[2]
_arch_sem = _repo / "assets" / "figures" / "architecture" / "semantic_pipeline"
ROOT = _arch_sem if _arch_sem.exists() else (_repo / "outputs" / "semantic_pipeline")
DPI = 300
FS, SM, TN = 7.5, 6.8, 6.2          # base / small / tiny font sizes (pt at final size)

OBJ_COL = {"person": "#57c75a", "bed": "#4a7fd6", "laptop": "#e07a2f",
           "doorway": "#b16be3", "shoe": "#d62f8a"}           # same as scene_common.LABEL_COL
AXIS_COL = {"attention": "#4a7fd6", "contacting": "#e07a2f", "spatial": "#2aa876"}
TEXT, MUTED, GRID = "#1f2933", "#6b7280", "#d7dbe0"
CONT_COL = {
    "not_contacting": "#e3e6ea", "leaning_on": "#9ecae1", "lying_on": "#4a7fd6",
    "sitting_on": "#1f4e9c", "touching": "#f2b632", "holding": "#e07a2f",
    "wearing": "#d62f8a", "other_relationship": "#8c8c8c", "unknown": "#ffffff",
}
MODEL_DISPLAY = {"qwen3vl": "Qwen2.5-VL-7B", "kimikvl": "Kimi-VL-A3B", "internvl": "InternVL2.5-8B"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
    "savefig.transparent": True, "savefig.dpi": DPI,
})


def pretty(lbl):
    return lbl.replace("_", " ")


def save(fig, out, name):
    p = out / f"{name}.png"
    fig.savefig(p, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# m1: clip input
# ---------------------------------------------------------------------------
def m1_clip_input(b, fdir, out, target):
    tc = b["target_clips"][target]
    cm = tc["clip_metadata"]
    others = [f for f in tc["clip_frames"] if f != target]
    fin = b["final_labels"][target]
    W, H = b["image_size"]

    fig = plt.figure(figsize=(3.1, 2.3))
    # left: target frame, big
    axT = fig.add_axes([0.0, 0.10, 0.36, 0.78])
    axT.imshow(Image.open(fdir / f"{target}.png"))
    x0, y0, x1, y1 = fin["person_bbox"][0]
    axT.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.1, ec=OBJ_COL["person"]))
    axT.text(x0 + 4, y1 - 6, "person", color="white", fontsize=TN, fontweight="bold", va="bottom",
             bbox=dict(fc=OBJ_COL["person"], ec="none", pad=0.6))
    unobs = []
    for o in fin["objects"]:
        if o["bbox"] is None:
            unobs.append(o["class"])
            continue
        x0, y0, x1, y1 = o["bbox"]
        c = OBJ_COL.get(o["class"], "#444")
        axT.add_patch(Rectangle((x0 + 2, y0 + 2), x1 - x0 - 4, y1 - y0 - 4, fill=False, lw=1.1, ec=c))
        axT.text(x0 + 6, y0 + 8, o["class"], color="white", fontsize=TN, fontweight="bold", va="top",
                 bbox=dict(fc=c, ec="none", pad=0.6))
    axT.set_xlim(0, W); axT.set_ylim(H, 0); axT.set_xticks([]); axT.set_yticks([])
    for s in axT.spines.values():
        s.set_edgecolor(TEXT); s.set_linewidth(1.2)
    axT.set_title(f"target frame {int(target)}", fontsize=FS, pad=2, fontweight="bold")
    axT.text(0.5, -0.02, "unobserved:\n" + ", ".join(sorted(unobs)), transform=axT.transAxes,
             ha="center", va="top", fontsize=TN, color=MUTED, linespacing=1.15)

    # right: the other clip frames, 3 x 2
    n = len(others)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    left, right, top, bot = 0.405, 1.0, 0.88, 0.10
    cw = (right - left) / ncol
    rh = (top - bot) / nrow
    for i, f in enumerate(others):
        r, c = divmod(i, ncol)
        ax = fig.add_axes([left + c * cw + 0.006, top - (r + 1) * rh + 0.035, cw - 0.012, rh - 0.05])
        ax.imshow(Image.open(fdir / f"{f}.png"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(GRID); s.set_linewidth(0.6)
        ax.text(0.5, -0.02, f"{int(f)}", transform=ax.transAxes, ha="center", va="top", fontsize=TN, color=TEXT)
    fig.text((left + right) / 2, 0.905, f"VLM clip: frames {cm['start_frame']}–{cm['end_frame']}, every 2nd",
             ha="center", va="bottom", fontsize=SM, color=TEXT)
    fig.text((left + right) / 2, 0.035, f"{len(tc['clip_frames'])} clip frames (incl. target),\ntotal_pixels = 128 000",
             ha="center", va="top", fontsize=TN, color=MUTED, linespacing=1.15)
    save(fig, out, "m1_clip_input")


# ---------------------------------------------------------------------------
# m2: event graph
# ---------------------------------------------------------------------------
def m2_event_graph(b, out, target, model="internvl", min_deg=7):
    g = b["graphs"][model]
    clips = g["clips"]
    mg = g["merged"]
    ek = mg["entity_keys"]
    tgt = int(target)
    node_frame = {}
    off = 0
    tnode = None
    for c in clips:
        for nd in c["nodes"]:
            node_frame[nd["id"] + off] = c["clip_metadata"]["annotated_frame"]
            if c["clip_metadata"]["annotated_frame"] == tgt:
                tnode = nd["id"] + off
        off += len(c["nodes"])
    tclip = [c for c in clips if c["clip_metadata"]["annotated_frame"] == tgt][0]
    tkeys = {k for k, v in ek.items() if tnode in v}
    keys = [k for k, v in ek.items() if len(v) >= min_deg or k in tkeys]
    keys.sort(key=lambda k: -len(ek[k]))
    objs = set(b["rag_all"]["qwen3vl"]["video_objects"])
    obj_key = {k: o for k in keys for o in objs if k.rstrip("s") == o}

    fig = plt.figure(figsize=(3.1, 2.45))
    ax = fig.add_axes([0.02, 0.33, 0.96, 0.59])
    nf = sorted(node_frame.values())
    fmin, fmax = nf[0], nf[-1]
    X = lambda f: (f - fmin) / (fmax - fmin)  # noqa: E731
    yN, yK = 1.0, 0.0
    kx = {k: (i + 0.5) / len(keys) for i, k in enumerate(keys)}
    # edges
    for k in keys:
        for n in ek[k]:
            is_t = n == tnode
            ax.plot([X(node_frame[n]), kx[k]], [yN, yK], lw=0.9 if is_t else 0.3,
                    color="#d64a4a" if is_t else "#9aa4b2", alpha=1.0 if is_t else 0.35,
                    zorder=3 if is_t else 1)
    # clip nodes
    for n, f in node_frame.items():
        is_t = n == tnode
        ax.scatter(X(f), yN, s=26 if is_t else 9, color="#d64a4a" if is_t else "#6b7280",
                   zorder=4, edgecolor="white", lw=0.4)
    ax.text(X(tgt), yN + 0.07, f"clip @ {tgt}", ha="center", va="bottom", fontsize=TN, color="#d64a4a",
            fontweight="bold")
    ax.text(-0.01, yN, "clip\nnodes", ha="right", va="center", fontsize=TN, color=MUTED)
    ax.text(-0.01, yK, "entity\nkeys", ha="right", va="center", fontsize=TN, color=MUTED)
    # entity keys
    for k in keys:
        deg = len(ek[k])
        c = OBJ_COL[obj_key[k]] if k in obj_key else ("#d64a4a" if k in tkeys else "#4b5563")
        ax.scatter(kx[k], yK, s=6 + 1.6 * deg, color=c, zorder=4, edgecolor="white", lw=0.4, marker="s")
        ax.text(kx[k], yK - 0.07, k, rotation=50, ha="right", va="top", fontsize=TN,
                color=c if (k in obj_key or k in tkeys) else TEXT,
                fontweight="bold" if k in obj_key else "normal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.18)
    ax.axis("off")
    fig.text(0.5, 0.995, f"{MODEL_DISPLAY[model]} event graph · {mg['n_nodes']} clip nodes · "
             f"{len(ek)} entity keys", ha="center", va="top", fontsize=SM - 0.3, fontweight="bold")
    fig.text(0.5, 0.935, f"frames {fmin}–{fmax}; keys with ≥{min_deg} clips shown "
             f"(size ∝ #clips)", ha="center", va="top", fontsize=TN, color=MUTED)
    nd = tclip["nodes"][0]
    act = nd["actions"][0].split(",", 1)[-1].strip() if nd.get("actions") else ""
    txt = f"“{tclip['subtitle']}”"
    fig.text(0.02, 0.06, f"subtitle @ {tgt}: {txt}\naction: {act[:62]}{'…' if len(act) > 62 else ''}",
             ha="left", va="bottom", fontsize=TN, color=TEXT, linespacing=1.3,
             bbox=dict(fc="#f5f6f8", ec=GRID, lw=0.5, boxstyle="round,pad=0.3"))
    save(fig, out, "m2_event_graph")


# ---------------------------------------------------------------------------
# m3: candidates + verification
# ---------------------------------------------------------------------------
def m3_candidates(b, out, target, objects, model="qwen3vl"):
    preds = {p["object"]: p for p in b["rag_all"][model]["frames"][f"{target}.png"]["predictions"]}
    fin = {o["class"]: o for o in b["final_labels"][target]["objects"]}
    rows = []  # (object, axis, label, p, in_final)
    for ob in objects:
        p = preds[ob]
        cands = [("attention", p["attention"])] + [("contacting", c) for c in p["contacting"]] + \
                [("spatial", s) for s in p["spatial"]]
        for ax_name, c in cands:
            rows.append((ob, ax_name, c["label"], c["yes_prob"], c["label"] in fin[ob][ax_name]))

    nrow = len(rows)
    fig = plt.figure(figsize=(3.1, 0.30 + 0.205 * nrow + 0.16 * len(objects)))
    ax = fig.add_axes([0.40, 0.15, 0.48, 0.77])
    y = 0
    ys, yl = [], []
    last = None
    for ob, axn, lbl, p, ok in rows:
        if ob != last:
            y += 0.55 if last is not None else 0.0
            obs = "observed" if fin[ob]["bbox"] is not None else "unobserved"
            t = ax.text(-0.83, y - 0.02, f"{ob}", transform=ax.transData, ha="left", va="bottom",
                        fontsize=SM, fontweight="bold", color=OBJ_COL.get(ob, TEXT))
            ax.annotate(f"  ({obs})", xycoords=t, xy=(1, 0), va="bottom", ha="left",
                        fontsize=TN, color=MUTED)
            y += 0.62
            last = ob
        ax.barh(y, 1.0, height=0.62, color="#f1f3f5", zorder=0)
        ax.barh(y, p, height=0.62, color=AXIS_COL[axn], zorder=1)
        ax.text(-0.02, y, f"{pretty(lbl)}", ha="right", va="center", fontsize=TN)
        ax.text(-0.80, y, {"attention": "att.", "contacting": "cont.", "spatial": "spat."}[axn],
                ha="left", va="center", fontsize=TN, color=AXIS_COL[axn], fontweight="bold")
        if p > 0.8:
            ax.text(p - 0.02, y, f"{p:.2f}", ha="right", va="center", fontsize=TN, color="white",
                    fontweight="bold")
        else:
            ax.text(p + 0.02, y, f"{p:.2f}", ha="left", va="center", fontsize=TN)
        ax.text(1.12, y, "✓" if ok else "✗", ha="center", va="center", fontsize=FS,
                color="#2aa876" if ok else "#d64a4a", fontweight="bold")
        ys.append(y)
        y += 1.0
    ax.set_ylim(y - 0.3, -0.2)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"], fontsize=TN)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_xlabel("verification $p_{\\mathrm{yes}}$", fontsize=TN, labelpad=1)
    ax.text(1.12, -0.2, "in final?", ha="center", va="bottom", fontsize=TN, color=MUTED)
    fig.text(0.02, 0.005, "✓/✗: label is / is not in the final set (GT if observed, human if not)",
             ha="left", va="bottom", fontsize=TN - 0.4, color=MUTED)
    fig.text(0.5, 0.995, f"{MODEL_DISPLAY[model]} RAG candidates, frame {int(target)}",
             ha="center", va="top", fontsize=SM, fontweight="bold")
    save(fig, out, "m3_candidates_verification")


# ---------------------------------------------------------------------------
# m4: contacting timeline
# ---------------------------------------------------------------------------
def m4_timeline(b, out, target, model="qwen3vl"):
    frames = b["annotated_frames"]
    objs = sorted(b["rag_all"][model]["video_objects"])
    fin = {f: {o["class"]: o for o in b["final_labels"][f]["objects"]} for f in frames}
    vlm = {f: {p["object"]: p for p in b["rag_all"][model]["frames"][f"{f}.png"]["predictions"]} for f in frames}
    n = len(frames)
    fig = plt.figure(figsize=(3.15, 2.25))
    ax = fig.add_axes([0.30, 0.30, 0.68, 0.60])
    rh = 0.42
    ytick, ylab = [], []
    used = set()
    for i, ob in enumerate(objs):
        base = i * 1.25
        for j, (src, yy) in enumerate([("VLM", base), ("final", base + rh + 0.04)]):
            for k, f in enumerate(frames):
                if src == "VLM":
                    labs = [c["label"] for c in vlm[f][ob]["contacting"]]
                    unobs = False
                else:
                    labs = fin[f][ob]["contacting"]
                    unobs = fin[f][ob]["bbox"] is None
                labs = labs or ["unknown"]
                w = 1.0 / len(labs)
                for q, l in enumerate(labs):
                    used.add(l)
                    ax.add_patch(Rectangle((k + q * w, yy), w, rh, fc=CONT_COL.get(l, "#999"),
                                           ec="none", zorder=1))
                    if l == "unknown":
                        ax.add_patch(Rectangle((k + 0.12, yy + 0.06), 0.76, rh - 0.12, fc="none",
                                               ec=MUTED, lw=0.4, zorder=2))
                ax.add_patch(Rectangle((k, yy), 1, rh, fc="none", ec="white", lw=0.4, zorder=2))
                if src == "final" and unobs:
                    ax.add_patch(Rectangle((k, yy), 1, rh, fc="none", ec="#1f2933", lw=0,
                                           hatch="/////", alpha=0.35, zorder=3))
            ytick.append(yy + rh / 2)
            ylab.append(src)
        ax.text(-0.15, base + rh + 0.02, ob, ha="right", va="center", fontsize=SM, fontweight="bold",
                color=OBJ_COL.get(ob, TEXT), transform=ax.get_yaxis_transform())
    kt = frames.index(target)
    ax.add_patch(Rectangle((kt, -0.1), 1, len(objs) * 1.25 - 0.25 + 0.2, fc="none", ec="#d64a4a",
                           lw=0.9, zorder=4))
    ax.set_xlim(0, n)
    ax.set_ylim(len(objs) * 1.25 - 0.2, -0.15)
    ax.set_yticks(ytick); ax.set_yticklabels(ylab, fontsize=TN, color=MUTED)
    ax.tick_params(axis="y", length=0, pad=1)
    xt = [0, 9, 13, 19, frames.index(target), 28]
    ax.set_xticks([k + 0.5 for k in xt]); ax.set_xticklabels([str(int(frames[k])) for k in xt], fontsize=TN)
    ax.tick_params(axis="x", length=2, pad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(f"annotated frame ({n} frames)", fontsize=TN, labelpad=1)
    fig.text(0.5, 0.995, f"contacting labels: {MODEL_DISPLAY[model]} RAG vs final",
             ha="center", va="top", fontsize=SM, fontweight="bold")
    # legend
    order = [l for l in CONT_COL if l in used]
    lx, ly = 0.02, 0.105
    fig_w = 0.0
    items = [(pretty(l), CONT_COL[l], None) for l in order] + [("unobserved", "white", "/////")]
    per_row = 4
    for i, (t, c, h) in enumerate(items):
        r, cidx = divmod(i, per_row)
        x = lx + cidx * 0.245
        yv = ly - r * 0.065
        fig.patches.append(Rectangle((x, yv), 0.028, 0.04, transform=fig.transFigure, fc=c,
                                     ec=MUTED if c in ("white", "#ffffff", "#e3e6ea") else "none",
                                     lw=0.4, hatch=h))
        fig.text(x + 0.035, yv + 0.02, t, fontsize=TN, va="center")
    save(fig, out, "m4_label_timeline")


# ---------------------------------------------------------------------------
# m5: human corrections
# ---------------------------------------------------------------------------
def m5_corrections(b, out, picks):
    corr = b["corrections"]
    tot = 0
    per_axis = Counter()
    any_edit = 0
    for f, lst in corr.items():
        for p in lst:
            if p["seed"] is None:
                continue
            tot += 1
            e = False
            for a in ("attention", "contacting", "spatial"):
                if sorted(p["seed"][a]) != sorted(p["human"][a]):
                    per_axis[a] += 1
                    e = True
            any_edit += e
    rows = []
    for f, ob, a in sorted(picks):
        p = [q for q in corr[f] if q["object"] == ob][0]
        rows.append((int(f), ob, a, ", ".join(pretty(x) for x in p["seed"][a]),
                     ", ".join(pretty(x) for x in p["human"][a])))

    fig = plt.figure(figsize=(3.1, 0.50 + 0.2 * len(rows) + 0.78))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    H = fig.get_figheight()
    dy = 0.2 / H
    y = 1 - 0.14 / H
    ax.text(0.5, 1 - 0.02 / H, "Human correction of unobserved-object labels", ha="center", va="top",
            fontsize=SM, fontweight="bold")
    y -= 0.18 / H
    cols = [0.01, 0.13, 0.28, 0.38, 0.70]
    for x, t in zip(cols, ["frame", "object", "axis", "VLM seed", "human"]):
        ax.text(x, y, t, fontsize=TN, color=MUTED, va="center")
    y -= 0.06 / H
    ax.plot([0.01, 0.99], [y, y], color=GRID, lw=0.6)
    for f, ob, a, s, h in rows:
        y -= dy
        ax.text(cols[0], y, str(f), fontsize=TN, va="center")
        ax.text(cols[1], y, ob, fontsize=TN, va="center", color=OBJ_COL.get(ob, TEXT), fontweight="bold")
        ax.text(cols[2], y, {"attention": "att.", "contacting": "cont.", "spatial": "spat."}[a],
                fontsize=TN, va="center", color=AXIS_COL[a], fontweight="bold")
        ax.text(cols[3], y, s, fontsize=TN, va="center", color="#9b2c2c")
        ax.text(cols[4] - 0.035, y, "→", fontsize=TN, va="center", color=MUTED)
        ax.text(cols[4], y, h, fontsize=TN, va="center", color="#1c6b4a", fontweight="bold")
    y -= 0.12 / H
    ax.plot([0.01, 0.99], [y, y], color=GRID, lw=0.6)
    # per-axis summary bars
    y -= 0.16 / H
    ax.text(0.01, y, f"{any_edit}/{tot} VLM-seeded predictions edited; edits per axis:", fontsize=TN,
            va="center")
    y -= 0.36 / H
    bw = 0.22
    for i, a in enumerate(("attention", "contacting", "spatial")):
        x = 0.01 + i * 0.33
        frac = per_axis[a] / tot
        ax.add_patch(Rectangle((x, y - 0.05 / H), bw, 0.1 / H, fc="#f1f3f5", ec="none"))
        ax.add_patch(Rectangle((x, y - 0.05 / H), bw * frac, 0.1 / H, fc=AXIS_COL[a], ec="none"))
        ax.text(x + bw + 0.01, y, f"{per_axis[a]}", fontsize=TN, va="center")
        ax.text(x, y + 0.09 / H, pretty(a), fontsize=TN, va="bottom", color=AXIS_COL[a])
    save(fig, out, "m5_corrections")
    return tot, any_edit, dict(per_axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="00T1E")
    ap.add_argument("--target", default="000273")
    ap.add_argument("--objects", default="doorway,bed,shoe")
    args = ap.parse_args()
    bdir = ROOT / args.video
    b = json.load(open(bdir / "bundle.json"))
    out = ROOT / f"panels_{args.video}"
    out.mkdir(parents=True, exist_ok=True)
    m1_clip_input(b, bdir / "frames", out, args.target)
    m2_event_graph(b, out, args.target)
    m3_candidates(b, out, args.target, args.objects.split(","))
    m4_timeline(b, out, args.target)
    if "corrections" in b:
        picks = [("000201", "shoe", "contacting"), ("000201", "shoe", "attention"),
                 ("000273", "shoe", "spatial"), ("000273", "bed", "spatial"),
                 ("000111", "laptop", "spatial"), ("000010", "doorway", "spatial")]
        print("m5 stats", m5_corrections(b, out, picks))


if __name__ == "__main__":
    main()
