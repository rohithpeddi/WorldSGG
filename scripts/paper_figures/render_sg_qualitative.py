"""Qualitative scene-graph figures for the supplementary, rendered in the paper's style from the data that
export_sg_json.py writes on the server (assets/figures/qualitative_sg/<video>/<video>_predcls_sg.json + frames/).

Every figure is drawn at its printed size (5.5 in = \\textwidth), so the type sizes are the final ones.  A
scene graph is a star around the person: one edge per annotated object, with the attention, spatial and
contacting predicates written on the edge.  Observed objects (OO) have solid grey nodes, unobserved objects
(UO) dashed orange ones.  In a method panel every predicted predicate is marked against the ground truth
(check = in the ground-truth set of that edge, cross = not), and k/n under a node counts the ground-truth
predicates of that edge the method recovers.  PredCls, with constraint: one predicate per head and edge.  On
the frames used here every prediction lies inside the top 20 of the frame, so a check equals a recall hit at
R@20; check_consistency() asserts this against the exported ``recalled`` sets before anything is drawn.

    python scripts/paper_figures/render_sg_qualitative.py [gt supervised zeroshot]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_common import crisp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "assets" / "figures" / "qualitative_sg"
OUT = DATA / "figures"
PAPER = Path(r"C:\Users\rohit\LaTeXProjects\WSGG_Paper\updated_submission\sup_images\qualitative")

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})

INK, MUTED, EDGE = "#1f2933", "#6b7280", "#c3c8d0"
OBS, OBS_FILL = "#8a8983", "#f2f2f0"        # observed objects (OO), as in the supplementary charts
UNOBS, UNOBS_FILL = "#d95926", "#fdeee6"    # unobserved objects (UO)
GOOD, BAD = "#2e7d4f", "#c0392b"
CHECK, CROSS = "\u2713", "\u2717"
DASH = (0, (2.2, 1.4))

TEXT_W = 5.5
FS_TITLE, FS_HEAD, FS_SUB, FS_NODE, FS_PRED, FS_LEG = 6.6, 6.2, 5.3, 5.6, 5.0, 5.6
LINE = 0.074                                  # inches between two predicate lines at FS_PRED

SHORT = {"looking_at": "looking", "not_looking_at": "not looking", "unsure": "unsure",
         "in_front_of": "in front", "on_the_side_of": "on side", "not_contacting": "no contact",
         "drinking_from": "drinking from", "have_it_on_the_back": "on back", "other_relationship": "other",
         "covered_by": "covered by", "leaning_on": "leaning on", "lying_on": "lying on",
         "sitting_on": "sitting on", "standing_on": "standing on", "writing_on": "writing on"}
ATT = {"looking_at", "not_looking_at", "unsure"}
SPA = {"above", "beneath", "in_front_of", "behind", "on_the_side_of", "in"}

NAMES = {"w_dsgdetr_pp": "W-DSGDetr++", "worldwise": "WorldWise", "worldwise_pp": "WorldWise++",
         "caption_all": "U-WSGG-Sub", "rag_all": "UWSGG-GraphRAG", "track_a": "Method A", "track_b": "Method B"}
# (main paper and Appendix I call the localized methods Method A / Method B; Appendix H calls them Track A / B)

_R = None


def text_w(s, size, bold=False):
    """Rendered width of s in inches."""
    global _R
    if _R is None:
        f = plt.figure(); _R = (f, f.canvas.get_renderer())
    fig, r = _R
    t = fig.text(0, 0, s, fontsize=size, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(renderer=r).width / fig.dpi
    t.remove()
    return w


def short(p):
    return SHORT.get(p, p.replace("_", " "))


def head_of(p):
    return 0 if p in ATT else (1 if p in SPA else 2)


# ------------------------------------------------------------------ data
def load_frame(video, frame_num):
    d = json.loads((DATA / video / f"{video}_predcls_sg.json").read_text(encoding="utf-8"))
    fr = next(f for f in d["frames"] if f["frame_num"] == frame_num)
    img = np.asarray(Image.open(DATA / video / "frames" / fr["file"]).convert("RGB"))
    objs = [o for o in fr["objects"] if o["observed"]] + [o for o in fr["objects"] if not o["observed"]]
    return fr, objs, img


def gt_preds(o):
    return list(o["attention"]) + list(o["spatial"]) + list(o["contacting"])


def check_consistency(fr, objs, method):
    """A predicted predicate is marked correct iff it is in the edge's GT set; assert this equals the metric's
    recalled set (R@20, with constraint) on this frame, so the marks never disagree with the tables."""
    m = fr["methods"][method]
    rec = {tuple(k) for k in m["recalled"]}
    for o in objs:
        for p in m["said"].get(o["label"], []):
            assert (p in gt_preds(o)) == ((o["label"], p) in rec), (fr["frame_num"], method, o["label"], p)


# ------------------------------------------------------------------ drawing
def node(ax, x, y, w, h, text, observed, weight="normal", person=False, size=FS_NODE):
    if person:
        ec, fc, ls = INK, "#e9edf1", "-"
    else:
        ec, fc, ls = (OBS, OBS_FILL, "-") if observed else (UNOBS, UNOBS_FILL, DASH)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                linewidth=0.7, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=3))
    if text:
        ax.text(x, y, text, fontsize=size, color=INK, ha="center", va="center", fontweight=weight, zorder=4)


def add_axes_in(fig, x, y, w, h):
    FH = fig.get_figheight()
    return fig.add_axes([x / TEXT_W, y / FH, w / TEXT_W, h / FH])


def graph_panel(fig, rect, objs, title, subtitle, lines_of, frac_of=None):
    """rect in inches (x, y, w, h).  lines_of(o) -> [(text, colour, style)] written on the edge to o."""
    x0, y0, W, H = rect
    ax = add_axes_in(fig, x0, y0, W, H)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(W / 2, H - 0.02, title, fontsize=FS_HEAD, fontweight="bold", color=INK, ha="center", va="top")
    if subtitle:
        ax.text(W / 2, H - 0.135, subtitle, fontsize=FS_SUB, color=MUTED, ha="center", va="top")
    top, bot = H - 0.36, 0.14
    n = len(objs)
    ys = np.linspace(top, bot, n) if n > 1 else np.array([(top + bot) / 2])
    py = float((top + bot) / 2)
    pw = text_w("person", FS_NODE, True) + 0.07
    ow = max(text_w(o["label"], FS_NODE) for o in objs) + 0.08
    nh = 0.14
    px, ox = 0.01 + pw / 2, W - 0.01 - ow / 2
    node(ax, px, py, pw, nh, "person", True, weight="bold", person=True)
    right = ox - ow / 2 - 0.03                    # labels end here, just before the object node
    for o, y in zip(objs, ys):
        ls = "-" if o["observed"] else DASH
        ax.plot([px + pw / 2, ox - ow / 2], [py, y], color=EDGE, lw=0.6, ls=ls, zorder=1)
        node(ax, ox, y, ow, nh, o["label"], o["observed"])
        if frac_of is not None:
            ax.text(ox, y - nh / 2 - 0.012, frac_of(o), fontsize=FS_SUB, color=MUTED, ha="center", va="top")
        rows = lines_of(o)
        # a label block next to a near-horizontal edge must not run into the person node
        avail = right - (px + pw / 2 + 0.03) if abs(y - py) < LINE * (len(rows) + 1) / 2 + nh / 2 else right - 0.02
        widest = max(text_w(t, FS_PRED) for t, _, _ in rows)
        fs = FS_PRED * min(1.0, avail / widest)
        step = LINE * fs / FS_PRED
        yy = y + step * (len(rows) - 1) / 2
        for i, (txt, col, style) in enumerate(rows):
            ax.text(right, yy - i * step, txt, fontsize=fs, color=col, ha="right", va="center", style=style,
                    zorder=5, path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    return ax


def gt_lines(o):
    return [(short(p), INK, "normal") for p in sorted(gt_preds(o), key=head_of)]


def method_lines(fr, method):
    said = fr["methods"][method]["said"]

    def f(o):
        ps = said.get(o["label"])
        if not ps:
            return [("no prediction", MUTED, "italic")]
        g = set(gt_preds(o))
        return [(f"{CHECK if p in g else CROSS} {short(p)}", GOOD if p in g else BAD, "normal")
                for p in sorted(ps, key=head_of)]
    return f


def method_frac(fr, method):
    said = fr["methods"][method]["said"]

    def f(o):
        g = set(gt_preds(o))
        return f"{len(g & set(said.get(o['label'], [])))}/{len(g)}"
    return f


def score(fr, objs, method):
    said = fr["methods"][method]["said"]
    rec = lambda os: (sum(len(set(gt_preds(o)) & set(said.get(o["label"], []))) for o in os),  # noqa: E731
                      sum(len(set(gt_preds(o))) for o in os))
    (h, t), (uh, ut) = rec(objs), rec([o for o in objs if not o["observed"]])
    return f"{h}/{t} Predicates \u00b7 UO {uh}/{ut}"


def rgb_panel(fig, rect, img, fr, objs, video):
    """Frame with the GT boxes of the person and the observed objects; unobserved objects are named under it."""
    x0, y0, W, H = rect
    boxes = [("person", fr.get("person_bbox_2d"), INK)] + [(o["label"], o["bbox_2d"], "#374151")
                                                          for o in objs if o["observed"] and o["bbox_2d"]]
    boxes = [(lab, b, col) for lab, b, col in boxes if b]
    if img.shape[1] > img.shape[0] and boxes:
        # landscape frame: keep the full height and crop the width to the annotated boxes (plus a margin),
        # so the frame is not printed postage-stamp small in a narrow column; every box stays in view
        h0, w0 = img.shape[:2]
        bx1 = min(b[0] for _, b, _ in boxes); bx2 = max(b[2] for _, b, _ in boxes)
        cw = min(w0, max((bx2 - bx1) * 1.15 + 16, 0.9 * h0))
        cx = min(max((bx1 + bx2) / 2, cw / 2), w0 - cw / 2)
        c0 = int(round(cx - cw / 2)); c1 = int(round(cx + cw / 2))
        img = img[:, c0:c1]
        boxes = [(lab, [b[0] - c0, b[1], b[2] - c0, b[3]], col) for lab, b, col in boxes]
    h_img, w_img = img.shape[:2]
    uo = [o["label"] for o in objs if not o["observed"]]
    # "Unobserved: a, b, c" wrapped to the column width
    lines, cur = [], "Unobserved:"
    for i, lab in enumerate(uo):
        piece = f" {lab}" + ("," if i < len(uo) - 1 else "")
        if text_w(cur + piece, FS_SUB, True) > W - 0.02 and cur != "Unobserved:":
            lines.append(cur); cur = piece.strip()
        else:
            cur += piece
    if uo:
        lines.append(cur)
    top_pad, strip = 0.15, 0.1 * len(lines) + 0.03
    iw = W; ih = iw * h_img / w_img
    if ih > H - top_pad - strip:
        ih = H - top_pad - strip; iw = ih * w_img / h_img
    ix, iy = x0 + (W - iw) / 2, y0 + H - top_pad - ih
    FH = fig.get_figheight()
    fig.text((x0 + W / 2) / TEXT_W, (y0 + H - 0.02) / FH, f"{video} \u00b7 Frame {fr['frame_num']}",
             fontsize=FS_HEAD, fontweight="bold", color=INK, ha="center", va="top")
    ax = add_axes_in(fig, ix, iy, iw, ih)
    ax.imshow(crisp(img), extent=(0, w_img, h_img, 0), interpolation="lanczos")
    ax.set_xlim(0, w_img); ax.set_ylim(h_img, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_edgecolor(MUTED)
    px_per_in = w_img / iw
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    placed = []
    for lab, b, col in boxes:
        x1, y1, x2, y2 = b
        r = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=0.8, ec=col, zorder=3)
        r.set_path_effects(halo); ax.add_patch(r)
        tw = (text_w(lab, FS_PRED) + 0.04) * px_per_in
        th = 0.085 * px_per_in
        tx, ty = min(max(x1 + 1, 1), w_img - tw - 1), max(y1 + 1, 1)
        for _ in range(12):               # stack the tag below an earlier tag it would cover
            hit = [q for q in placed if tx < q[0] + q[2] and q[0] < tx + tw and ty < q[1] + q[3] and q[1] < ty + th]
            if not hit:
                break
            ty = max(q[1] + q[3] for q in hit) + 1
        placed.append((tx, ty, tw, th))
        ax.text(tx + 0.02 * px_per_in, ty + th / 2, lab, fontsize=FS_PRED, color="white", ha="left", va="center",
                zorder=4, bbox=dict(boxstyle="square,pad=0.1", fc=col, ec="none", alpha=0.88))
    for i, line in enumerate(lines):
        fig.text((ix) / TEXT_W, (iy - 0.03 - 0.1 * i) / FH, line, fontsize=FS_SUB, color=UNOBS, ha="left",
                 va="top", fontweight="bold")


def legend(fig, y, methods=True):
    """Key across the figure bottom; items are measured and centred."""
    items = [("node", True, "Observed Object (OO)"), ("node", False, "Unobserved Object (UO)")]
    if methods:
        items += [("text", GOOD, f"{CHECK} In Ground Truth"), ("text", BAD, f"{CROSS} Not in Ground Truth"),
                  ("text", MUTED, "k/n: Ground-Truth Predicates Recovered")]
    sw, gap = 0.2, 0.14
    fs = FS_LEG
    while True:                                      # shrink the key until it fits the text width
        widths = [(sw + 0.05 if k == "node" else 0) + text_w(t, fs) for k, _, t in items]
        total = sum(widths) + gap * (len(items) - 1)
        if total <= TEXT_W - 0.08 or fs <= 4.6:
            break
        fs -= 0.1
    FH = fig.get_figheight()
    ax = fig.add_axes([0, y / FH, 1, 0.14 / FH]); ax.axis("off")
    ax.set_xlim(0, TEXT_W); ax.set_ylim(0, 0.14)
    x = (TEXT_W - total) / 2
    for (kind, arg, t), w in zip(items, widths):
        if kind == "node":
            node(ax, x + sw / 2, 0.07, sw, 0.09, "", arg)
            ax.text(x + sw + 0.05, 0.07, t, fontsize=fs, va="center", color=INK)
        else:
            ax.text(x, 0.07, t, fontsize=fs, va="center", color=arg)
        x += w + gap


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / f"{name}.pdf").write_bytes((OUT / f"{name}.pdf").read_bytes())
    print("wrote", OUT / f"{name}.pdf", "->", PAPER / f"{name}.pdf")


# ------------------------------------------------------------------ figures
def row_height(objs, lines_max=4):
    return 0.36 + 0.14 + max(1, len(objs) - 1) * max(0.36, LINE * lines_max + 0.1) + 0.06


def img_width(img, row_h, portrait_w, landscape_w):
    h, w = img.shape[:2]
    return portrait_w if h > w else landscape_w


def fig_gt(examples):
    """Annotated world scene graphs of two frames side by side, each frame | its ground-truth graph (dataset
    appendix).  Both graphs list the same objects in the same order, so a visibility change reads across."""
    loaded = [(v, *load_frame(v, f)) for v, f in examples]
    order = [o["label"] for o in loaded[0][2]]
    for i, (v, fr, objs, img) in enumerate(loaded):          # one object order for every panel
        rank = {l: k for k, l in enumerate(order)}
        loaded[i] = (v, fr, sorted(objs, key=lambda o: rank.get(o["label"], len(order))), img)
    rh = max(max(row_height(objs, max(len(gt_preds(o)) for o in objs)) for _, _, objs, _ in loaded), 1.6)
    leg = 0.2
    H = rh + leg
    fig = plt.figure(figsize=(TEXT_W, H)); fig.patch.set_facecolor("white")
    gap = 0.12
    colw = (TEXT_W - gap) / len(loaded)
    for i, (v, fr, objs, img) in enumerate(loaded):
        x = i * (colw + gap)
        h, w = img.shape[:2]
        iw = min(1.05, (rh - 0.3) * w / h) if h > w else 1.45
        rgb_panel(fig, (x, leg, iw, rh), img, fr, objs, v)
        graph_panel(fig, (x + iw + 0.04, leg, colw - iw - 0.04, rh), objs, "Ground Truth", "", gt_lines)
    legend(fig, 0.02, methods=False)
    save(fig, "sg_gt_examples")


def fig_methods(rows, methods, name):
    """One row per (video, frame): frame | ground truth | one scene graph per method."""
    loaded = [(v, *load_frame(v, f)) for v, f in rows]
    for v, fr, objs, _ in loaded:
        for m in methods:
            check_consistency(fr, objs, m)
    heights = [max(row_height(objs, max(len(gt_preds(o)) for o in objs)), 1.6) for _, _, objs, _ in loaded]
    gap_row, leg = 0.1, 0.2
    H = sum(heights) + gap_row * (len(rows) - 1) + leg
    fig = plt.figure(figsize=(TEXT_W, H)); fig.patch.set_facecolor("white")
    rgb_w, gap = 1.1, 0.05
    gw = (TEXT_W - rgb_w - gap * (len(methods) + 1)) / (len(methods) + 1)
    y = H
    for (v, fr, objs, img), rh in zip(loaded, heights):
        y -= rh
        rgb_panel(fig, (0, y, rgb_w, rh), img, fr, objs, v)
        x = rgb_w + gap
        graph_panel(fig, (x, y, gw, rh), objs, "Ground Truth", "", gt_lines)
        for m in methods:
            x += gw + gap
            graph_panel(fig, (x, y, gw, rh), objs, NAMES[m], score(fr, objs, m), method_lines(fr, m),
                        method_frac(fr, m))
        y -= gap_row
    legend(fig, 0.02)
    save(fig, name)


# example choices live here so the figure and its caption stay in sync
# Frames were vetted by eye: the objects labelled unobserved are out of view in the image.
GT_EXAMPLES = [("X37P1", 523), ("X37P1", 839)]          # towel unobserved, then in view and held
SUPERVISED_ROWS = [("XACI3", 280), ("X37P1", 839)]      # WorldWise++ ahead, then W-DSGDetr++ one predicate ahead
ZEROSHOT_ROWS = [("GFK4S", 41), ("XACI3", 280)]         # four UO objects without features; same frame as above

FIGS = {
    "gt": lambda: fig_gt(GT_EXAMPLES),
    "supervised": lambda: fig_methods(SUPERVISED_ROWS, ["w_dsgdetr_pp", "worldwise_pp"], "sg_supervised_predcls"),
    "zeroshot": lambda: fig_methods(ZEROSHOT_ROWS, ["rag_all", "track_b"], "sg_zeroshot_predcls"),
}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(FIGS):
        FIGS[n]()
