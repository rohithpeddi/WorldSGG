"""Primitives and schematic placeholders for the five MLLM architecture figures
(``fig_zero_shot.py``, ``fig_caption_all.py``, ``fig_graph_rag.py``, ``fig_track_a.py``,
``fig_track_b.py``), on top of ``common.Canvas``.

Every method is training-free, so the module kinds differ from the WorldWise
figures: every network is frozen (snowflake), deterministic code is a *program*
(teal), the component the method introduces is orange, and calls that exist in
the code but change no reported number are *ghosted*.  Prompt families carry the
P0-P6 badges of ``setup/UNLOCALIZED_GRAPH_RAG.md`` §7.

The three unlocalized methods share their visual input, their SGDet object
discovery and their parsing; ``visual_input_block``, ``object_set_block`` and
``parse_block`` draw those modules in full so that each figure is self-contained.

Every intermediate is an image slot (:meth:`Canvas.image_slot`).  Until the real
panels are dumped, a slot draws a schematic of what it will hold, built only from
the prompt templates and output schemas in the code, never from invented results.
A PNG named after the slot key replaces the schematic (``--images <dir>``).
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

from common import (BAR_OFF, BG, CARD, DIM, FILM_F, GHOST_S, GREEN, GRIDLINE, LAT, LINE, LOSS_F, MUTED, NODE_F,  # noqa: E501
                    ORANGE, PURPLE_S, RED, RED_TXT, RULE, SANS, SERIF, SPROCKET, THEME, TXT, Canvas)

MONO = "Consolas, 'Cascadia Mono', 'Courier New', monospace"
PAPER = "#e7e9ec"          # the real BEV renders are light (base colour 245)
GRID_L, GRID_D = "#cfd3d8", "#a9afb7"
FOOT = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4"]   # the palette of tools/marks.py

PROMPT = {                  # id: (fill, text, name) — ids as in setup/UNLOCALIZED_GRAPH_RAG.md §7
    0: (("#e5e7eb", "#111827") if THEME == "dark" else ("#374151", "#ffffff")) + ("Event-Graph Construction",),
    1: ("#3b82f6", "#ffffff", "Caption Generation"),
    2: ("#14b8a6", "#03201c", "Keyword Extraction"),
    3: ("#d946ef", "#ffffff", "Node Relevance Check"),
    4: ("#facc15", "#231c02", "Relationship Query"),
    5: ("#4ade80", "#052e16", "Yes / No Check"),
    6: ("#c2803a", "#ffffff", "Object Discovery"),
}
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(n) -> str:
    return str(n).translate(_SUB)


class MCanvas(Canvas):
    """Canvas plus the marks the MLLM figures need."""

    def badge(self, x, y, n: int, s: float = 18):
        f, t, _ = PROMPT[n]
        self.a(f'<rect x="{x:.1f}" y="{y:.1f}" width="{s}" height="{s}" rx="4" fill="{f}" '
               f'stroke="{BG}" stroke-width="1.5"/>')
        self.text(x + s / 2, y + s * 0.73, "P" + sub(n), size=s * 0.62, fill=t, anchor="middle", weight="700",
                  font=SERIF)

    def vlm(self, x, y, w, h, model: str, prompts: Sequence[int] = (), sub2=None, ghost=False,
            title="Frozen VLM", tsize=11.5, ssize=9.5):
        self.box(x, y, w, h, title, model, kind="ghost" if ghost else "frozen", frozen=not ghost, sub2=sub2,
                 tsize=tsize, ssize=ssize)
        for i, n in enumerate(prompts):
            self.badge(x + 8 + i * 22, y - 9, n)

    def tool(self, x, y, w, h, title, sub_=None, sub2=None, tsize=11, ssize=8.8, kind="tool"):
        self.box(x, y, w, h, title, sub_, kind=kind, sub2=sub2, tsize=tsize, ssize=ssize)

    def note(self, x, y, s, size=9, fill=DIM, anchor="start", style="normal"):
        self.text(x, y, s, size=size, fill=fill, anchor=anchor, style=style)

    def slot_title(self, x, y, s, col=MUTED, size=9.5):
        self.text(x, y - 6, s, size=size, fill=col, weight="600")

    def badge_legend(self, x, y, ids: Sequence[int]):
        for n in ids:
            self.badge(x, y - 13, n, s=16)
            name = PROMPT[n][2]
            self.text(x + 21, y, name, size=10, fill=MUTED)
            x += 21 + len(name) * 5.7 + 18
        return x

    def caption(self, x, y, lead: str, lines: Sequence[str], size=11.5, lh=17):
        self.rich(x, y, [(lead + " ", {"bold": True, "fill": TXT}), (lines[0], {"fill": MUTED})], size=size)
        for i, ln in enumerate(lines[1:], 1):
            self.text(x, y + i * lh, ln, size=size, fill=MUTED)

    def violation(self, x, y, w, h, title, lines: Sequence[str]):
        """A violation list: the critic's analogue of a loss node (red, dashed)."""
        self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{LOSS_F}" stroke="{RED}" '
               f'stroke-width="1.4" stroke-dasharray="5 3"/>')
        self.text(x + 10, y + 17, title, size=10.5, fill=RED, weight="700")
        for i, ln in enumerate(lines):
            self.text(x + 10, y + 33 + i * 13, ln, size=8.6, fill=RED_TXT, font=MONO)


# ---------------------------------------------------------------------------
# schematic placeholders: f(c, x, y, w, h), drawn when the slot's PNG is absent
# ---------------------------------------------------------------------------

def _clip(s: str, w: float, size: float, mono: bool) -> str:
    n = max(4, int((w - 16) / (size * (0.57 if mono else 0.53))))
    return s if len(s) <= n else s[: n - 1] + "…"


def ph_text(lines: Sequence[str], head: str = None, head_col=ORANGE, mono=True, size=8.4, cols=None):
    """A text card: the template or schema the slot will show as real text."""
    def draw(c, x, y, w, h):
        c.a(f'<rect x="{x + 3}" y="{y + 3}" width="{w - 6}" height="{h - 6}" rx="3" fill="{CARD}"/>')
        yy = y + 15
        if head:
            c.text(x + 9, yy, head, size=8.8, fill=head_col, weight="600")
            yy += 14
        for i, ln in enumerate(lines):
            if yy > y + h - 5:
                break
            col = (cols[i] if cols and i < len(cols) else MUTED)
            s = _clip(ln, w, size, mono).replace(" ", "\u00a0") if mono else _clip(ln, w, size, mono)
            c.text(x + 9, yy, s, size=size, fill=col, font=MONO if mono else SANS)
            yy += size * 1.5
    return draw


def ph_film(n=18, segs=((0, 5), (6, 11), (12, 17)), keys=(3, 9, 15)):
    """A film strip cut into key-frame segments (orange = annotated key frame)."""
    def draw(c, x, y, w, h):
        bx, bw = x + 8, w - 16
        by, bh = y + 12, min(48.0, h * 0.46)
        c.a(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="3" fill="{FILM_F}" stroke="{RULE}"/>')
        for i in range(int(bw // 8)):
            sx = bx + 3 + i * 8
            c.a(f'<rect x="{sx:.1f}" y="{by + 2}" width="4" height="3" fill="{SPROCKET}"/>')
            c.a(f'<rect x="{sx:.1f}" y="{by + bh - 5}" width="4" height="3" fill="{SPROCKET}"/>')
        fw = (bw - 6) / n
        for i in range(n):
            fx = bx + 3 + i * fw
            fill = ORANGE if i in keys else f"url(#photo{i % 3})"
            op = 1 if i in keys else 0.7
            c.a(f'<rect x="{fx + 1:.1f}" y="{by + 8}" width="{fw - 2:.1f}" height="{bh - 16}" rx="1.5" '
                f'fill="{fill}" opacity="{op}"/>')
        for j, (a, b) in enumerate(segs):
            x0, x1 = bx + 3 + a * fw + 1, bx + 3 + (b + 1) * fw - 1
            yb = by + bh + 6
            c.a(f'<path d="M{x0:.1f} {yb} v5 H{x1:.1f} v-5" fill="none" stroke="{ORANGE}" stroke-width="1.3"/>')
            c.text((x0 + x1) / 2, yb + 18, "S" + sub(j + 1), size=10.5, fill=TXT, anchor="middle", font=SERIF)
        c.text(x + w / 2, y + h - 7, "orange = annotated key frame kᵢ, the centre of Sᵢ", size=8.3, fill=DIM,
               anchor="middle")
    return draw


def ph_event_graph(n=5):
    """Event graph: one node per segment (entities / actions / scenes); nodes that
    mention the same entity name are linked through the entity index."""
    def draw(c, x, y, w, h):
        nw, nh = 48, 46
        pts = [(x + 14 + nw / 2 + i * (w - 28 - nw) / (n - 1), y + (h * 0.31 if i % 2 == 0 else h * 0.64))
               for i in range(n)]
        links = [(0, 1, "e₁"), (1, 2, "e₂"), (2, 3, "e₁"), (3, 4, "e₃"), (0, 2, "e₁"), (2, 4, "e₃")]
        for a, b, lab in links:
            (x1, y1), (x2, y2) = pts[a], pts[b]
            c.a(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{LINE}" stroke-width="1.2"/>')
            c.text((x1 + x2) / 2, (y1 + y2) / 2 - 3, lab, size=9, fill=ORANGE, anchor="middle", style="italic",
                   font=SERIF)
        for i, (cx, cy) in enumerate(pts):
            nx, ny = cx - nw / 2, cy - nh / 2
            c.a(f'<rect x="{nx:.1f}" y="{ny:.1f}" width="{nw}" height="{nh}" rx="6" fill="{NODE_F}" '
                f'stroke="{LAT[i % len(LAT)]}" stroke-width="1.4"/>')
            c.text(cx, ny + 13, "n" + sub(i + 1), size=10, fill=TXT, anchor="middle", weight="600")
            for r, col in enumerate((LAT[0], LAT[2], LAT[4])):
                c.a(f'<rect x="{nx + 6:.1f}" y="{ny + 19 + r * 8:.1f}" width="{nw - 12}" height="5" rx="1.5" '
                    f'fill="{col}" opacity="0.85"/>')
        ly = y + h - 9
        x0 = x + 10
        for lab, col in (("entities", LAT[0]), ("actions", LAT[2]), ("scenes", LAT[4])):
            c.a(f'<rect x="{x0}" y="{ly - 6}" width="9" height="5" rx="1.5" fill="{col}"/>')
            c.text(x0 + 13, ly, lab, size=8.3, fill=DIM)
            x0 += 13 + len(lab) * 4.8 + 12
        c.text(x + w - 8, y + 14, "eⱼ = shared entity name", size=8.3, fill=DIM, anchor="end")
    return draw


def ph_photo(i=0, tag=None):
    """A plain frame (context frames): a photo-toned rectangle."""
    def draw(c, x, y, w, h):
        c.a(f'<rect x="{x + 4}" y="{y + 4}" width="{w - 8}" height="{h - 8}" rx="2" fill="url(#photo{i % 3})"/>')
        if tag:
            c.text(x + w / 2, y + h - 9, tag, size=8.5, fill="#c8cdd6", anchor="middle")
    return draw


def ph_violations(lines: Sequence[str]):
    """The critic's output: violation sentences, as sent back in the repair prompt."""
    def draw(c, x, y, w, h):
        c.violation(x + 3, y + 3, w - 6, h - 6, "Violations → Repair Prompt (≤ 12)", lines)
    return draw


def ph_frames_strip(n=10, target=False, label=None, lowres=False):
    """A row of small frames; with ``target`` the first one is the target frame."""
    def draw(c, x, y, w, h):
        th = h - (20 if label else 8)
        if target:
            tw = min(th * 1.45, w * 0.28)
            c.a(f'<rect x="{x + 6}" y="{y + 5}" width="{tw:.1f}" height="{th - 2:.1f}" rx="2" fill="url(#photo0)" '
                f'stroke="{ORANGE}" stroke-width="2"/>')
            c.text(x + 6 + tw / 2, y + 5 + th / 2 + 4, "f", size=13, fill=TXT, anchor="middle", font=SERIF,
                   style="italic")
            gx = x + 12 + tw
        else:
            gx = x + 6
        gw = x + w - 6 - gx
        rows = 3 if target else (2 if n > 8 else 1)
        cols = math.ceil(n / rows)
        cw, ch = gw / cols - 3, (th - 2) / rows - 3
        k = 0
        for r in range(rows):
            for q in range(cols):
                if k >= n:
                    break
                op = 0.55 if lowres else 0.8
                c.a(f'<rect x="{gx + q * (cw + 3):.1f}" y="{y + 5 + r * (ch + 3):.1f}" width="{cw:.1f}" '
                    f'height="{ch:.1f}" rx="1.5" fill="url(#photo{(k + 1) % 3})" opacity="{op}"/>')
                k += 1
        if label:
            c.text(x + w / 2, y + h - 5, label, size=8.3, fill=DIM, anchor="middle")
    return draw


def ph_objects(n=4):
    """Discovered objects with their Yes/No object score (schematic bars)."""
    vals = (0.92, 0.81, 0.66, 0.42)

    def draw(c, x, y, w, h):
        c.text(x + 10, y + 15, "object", size=8.3, fill=DIM)
        c.text(x + w - 10, y + 15, "P(Yes)", size=8.3, fill=DIM, anchor="end")
        rh = (h - 26) / n
        for i in range(n):
            yy = y + 22 + i * rh
            c.a(f'<rect x="{x + 10}" y="{yy + 2:.1f}" width="30" height="{rh - 6:.1f}" rx="4" fill="{NODE_F}" '
                f'stroke="{LAT[i]}" stroke-width="1.2"/>')
            c.text(x + 25, yy + rh / 2 + 2, "o" + sub(i + 1), size=9, fill=TXT, anchor="middle")
            bw = (w - 64) * vals[i]
            c.a(f'<rect x="{x + 48}" y="{yy + rh / 2 - 3:.1f}" width="{bw:.1f}" height="6" rx="2" fill="{GREEN}" '
                f'opacity="0.8"/>')
    return draw


def ph_ranked(n=12):
    """Nodes ranked by mean cosine to the query set; bars above 0.5 are kept, the
    first is the top-1 node whose text becomes the context block (schematic)."""
    vals = (0.86, 0.8, 0.75, 0.71, 0.67, 0.63, 0.6, 0.57, 0.53, 0.51, 0.47, 0.43)

    def draw(c, x, y, w, h):
        ax, ay, aw, ah = x + 30, y + h - 24, w - 42, h - 46
        lo, hi = 0.3, 0.9
        bw = aw / n
        for i, v in enumerate(vals[:n]):
            bh = (v - lo) / (hi - lo) * ah
            col = ORANGE if i == 0 else (PURPLE_S if v > 0.5 else BAR_OFF)
            c.a(f'<rect x="{ax + i * bw + 1.5:.1f}" y="{ay - bh:.1f}" width="{bw - 3:.1f}" height="{bh:.1f}" '
                f'rx="1.5" fill="{col}"/>')
        ty = ay - (0.5 - lo) / (hi - lo) * ah
        c.a(f'<line x1="{ax - 4}" y1="{ty:.1f}" x2="{ax + aw}" y2="{ty:.1f}" stroke="{RED}" stroke-dasharray="3 3"/>')
        c.text(ax - 6, ty + 3, "0.5", size=8.3, fill=RED, anchor="end")
        c.a(f'<line x1="{ax}" y1="{ay}" x2="{ax + aw}" y2="{ay}" stroke="{LINE}"/>')
        c.text(ax + aw / 2, ay + 14, "event-graph nodes, ranked (top 20 kept)", size=8.3, fill=DIM, anchor="middle")
        c.text(ax + bw / 2, y + 14, "top-1", size=8.5, fill=ORANGE, anchor="start")
        c.text(ax - 6, y + 14, "cos", size=8.3, fill=DIM, anchor="end")
    return draw


def ph_scene_graph(n=3, labels=("att · con · spa",) * 3, boxes=False):
    """Person → object edges carrying the three predicate heads."""
    def draw(c, x, y, w, h):
        px, py = x + 46, y + h / 2
        ys = [y + h * (0.2 + 0.6 * i / max(1, n - 1)) for i in range(n)]
        ox = x + w - 42
        for i, oy in enumerate(ys):
            c.a(f'<line x1="{px + 16}" y1="{py:.1f}" x2="{ox - 16}" y2="{oy:.1f}" stroke="{PURPLE_S}" stroke-width="1.4"/>')
            c.text((px + ox) / 2, (py + oy) / 2 - 4, labels[i], size=8.5, fill=TXT, anchor="middle")
            c.a(f'<circle cx="{ox}" cy="{oy:.1f}" r="15" fill="{NODE_F}" stroke="{LAT[i]}" stroke-width="1.5"/>')
            c.text(ox, oy + 3.5, "o" + sub(i + 1), size=9.5, fill=TXT, anchor="middle")
        c.a(f'<circle cx="{px}" cy="{py:.1f}" r="17" fill="{NODE_F}" stroke="{TXT}" stroke-width="1.5"/>')
        c.text(px, py + 3.5, "person", size=8.5, fill=TXT, anchor="middle")
        c.text(x + w / 2, y + h - 6, "class only · no 3-D boxes", size=8.3, fill=DIM, anchor="middle")
    return draw


# ---------------------------- localized placeholders ----------------------------

def _lcg(seed: int):
    s = seed

    def nxt():
        nonlocal s
        s = (1103515245 * s + 12345) % (2 ** 31)
        return s / 2 ** 31
    return nxt


def ph_cloud():
    """Canonical frame: floor grid (z = 0), point cloud, camera path."""
    def draw(c, x, y, w, h):
        fx0, fy0 = x + 14, y + h - 16
        dx, dy = (w - 60), -(h * 0.36)                      # floor parallelogram, skewed back
        sk = 34
        for i in range(7):
            t = i / 6
            c.a(f'<line x1="{fx0 + t * dx:.1f}" y1="{fy0:.1f}" x2="{fx0 + t * dx + sk:.1f}" y2="{fy0 + dy:.1f}" '
                f'stroke="{GRIDLINE}" stroke-width="1"/>')
            c.a(f'<line x1="{fx0 + t * sk:.1f}" y1="{fy0 + t * dy:.1f}" x2="{fx0 + dx + t * sk:.1f}" '
                f'y2="{fy0 + t * dy:.1f}" stroke="{GRIDLINE}" stroke-width="1"/>')
        r = _lcg(7)
        for _ in range(230):
            u, v, z = r(), r(), r() ** 1.6
            px = fx0 + u * dx + v * sk
            py = fy0 + v * dy - z * h * 0.42
            g = int(120 + 100 * r())
            c.a(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.1" fill="rgb({g},{g - 8},{g - 20})" opacity="0.9"/>')
        for i in range(9):                                   # camera path
            t = i / 8
            cx_ = fx0 + dx * (0.25 + 0.5 * t)
            cy_ = fy0 + dy * 0.55 - h * 0.30 - 10 * math.sin(t * math.pi)
            c.a(f'<circle cx="{cx_:.1f}" cy="{cy_:.1f}" r="2.2" fill="{MUTED}"/>')
        c.a(f'<line x1="{x + w - 14}" y1="{y + h - 22}" x2="{x + w - 14}" y2="{y + 16}" stroke="{TXT}" '
            f'stroke-width="1.2" marker-end="url(#ah)"/>')
        c.text(x + w - 18, y + 22, "z", size=10, fill=TXT, anchor="end", font=SERIF, style="italic")
        c.text(fx0 + 4, fy0 - 4, "floor z = 0", size=8.3, fill=DIM)
    return draw


def _bev_base(c, x, y, w, h, seed=3):
    c.a(f'<rect x="{x + 4}" y="{y + 4}" width="{w - 8}" height="{h - 8}" rx="3" fill="{PAPER}"/>')
    step = 14
    k = 0
    xx = x + 4 + step
    while xx < x + w - 4:
        c.a(f'<line x1="{xx:.1f}" y1="{y + 4}" x2="{xx:.1f}" y2="{y + h - 4}" stroke="{GRID_D if k % 2 else GRID_L}" '
            f'stroke-width="0.8"/>')
        xx += step
        k += 1
    k = 0
    yy = y + 4 + step
    while yy < y + h - 4:
        c.a(f'<line x1="{x + 4}" y1="{yy:.1f}" x2="{x + w - 4}" y2="{yy:.1f}" stroke="{GRID_D if k % 2 else GRID_L}" '
            f'stroke-width="0.8"/>')
        yy += step
        k += 1
    r = _lcg(seed)
    for _ in range(5):                                         # furniture blobs
        bw, bh = 18 + 30 * r(), 12 + 22 * r()
        bx, by = x + 10 + (w - 20 - bw) * r(), y + 10 + (h - 20 - bh) * r()
        c.a(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="#8f8a84" opacity="0.45"/>')
    for i in range(10):                                        # camera path, grey dots
        t = i / 9
        c.a(f'<circle cx="{x + w * (0.2 + 0.55 * t):.1f}" cy="{y + h * (0.78 - 0.18 * math.sin(t * 3)):.1f}" r="1.8" '
            f'fill="#5a5a5a"/>')


def _foot(c, cx, cy, fw, fh, ang, col, tag, dashed=False, tag_col=None):
    a = math.radians(ang)
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        px, py = sx * fw / 2, sy * fh / 2
        pts.append((cx + px * math.cos(a) - py * math.sin(a), cy + px * math.sin(a) + py * math.cos(a)))
    d = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
    dash = ' stroke-dasharray="3 2"' if dashed else ""
    c.a(f'<polygon points="{d}" fill="{col}" fill-opacity="0.25" stroke="{col}" stroke-width="1.6"{dash}/>')
    tw = 7 + 6 * len(tag)
    c.a(f'<rect x="{cx - tw / 2:.1f}" y="{cy - 7:.1f}" width="{tw}" height="13" rx="2" fill="#ffffff" '
        f'stroke="{col}" stroke-width="1"/>')
    c.text(cx, cy + 3.5, tag, size=9, fill=tag_col or col, anchor="middle", weight="700")


def ph_bev(marks=True, predicted=False, flagged=False, repaired=False, caption=None):
    """The top-down map: base render; with ``marks`` the numbered footprints, the
    person P and the camera arrow at the target frame."""
    def draw(c, x, y, w, h):
        _bev_base(c, x, y, w, h)
        if marks:
            _foot(c, x + w * 0.30, y + h * 0.32, 30, 20, 12, FOOT[0], "1")
            _foot(c, x + w * 0.70, y + h * 0.30, 26, 16, -18, FOOT[1], "2")
            far = (x + w * 0.80, y + h * 0.66)
            near = (x + w * 0.56, y + h * 0.60)
            if flagged:
                _foot(c, far[0], far[1], 22, 16, 30, RED, "3", dashed=True)
                c.a(f'<circle cx="{far[0]:.1f}" cy="{far[1]:.1f}" r="17" fill="none" stroke="{RED}" '
                    f'stroke-width="1.5" stroke-dasharray="3 2"/>')
            elif repaired:
                _foot(c, near[0], near[1], 22, 16, 30, FOOT[2], "3")
            else:
                _foot(c, far[0], far[1], 22, 16, 30, FOOT[2], "3")
            _foot(c, x + w * 0.42, y + h * 0.60, 16, 14, 0, "#111111", "P", tag_col="#111111")
            cx_, cy_ = x + w * 0.30, y + h * 0.82
            c.a(f'<line x1="{cx_:.1f}" y1="{cy_:.1f}" x2="{cx_ + 18:.1f}" y2="{cy_ - 16:.1f}" stroke="#ff0000" '
                f'stroke-width="2.4"/>')
            c.a(f'<circle cx="{cx_:.1f}" cy="{cy_:.1f}" r="3.5" fill="#ff0000"/>')
            if h >= 90:
                c.text(cx_ + 6, cy_ + 12, "camera", size=8, fill="#ff0000")
        if predicted:
            c.a(f'<line x1="{x + w * 0.42:.1f}" y1="{y + h * 0.60:.1f}" x2="{x + w * 0.30:.1f}" y2="{y + h * 0.32:.1f}" '
                f'stroke="#7c3aed" stroke-width="1.3" stroke-dasharray="3 2"/>')
        if caption:
            c.text(x + w / 2, y + h - 8, caption, size=8.3, fill="#4b5563", anchor="middle")
    return draw


def ph_marked_frame(unseen=True, proposals=False):
    """Set-of-mark frame: numbered 2-D boxes, the person in black, unseen ids in a legend."""
    def draw(c, x, y, w, h):
        c.a(f'<rect x="{x + 4}" y="{y + 4}" width="{w - 8}" height="{h - 8}" rx="2" fill="url(#photo1)"/>')

        def bx(rx, ry, rw, rh, col, tag):
            X, Y, W, H = x + 4 + rx * (w - 8), y + 4 + ry * (h - 8), rw * (w - 8), rh * (h - 8)
            c.a(f'<rect x="{X:.1f}" y="{Y:.1f}" width="{W:.1f}" height="{H:.1f}" fill="none" stroke="{col}" '
                f'stroke-width="2"/>')
            tw = 6 + 5.6 * len(tag)
            c.a(f'<rect x="{X:.1f}" y="{Y - 12:.1f}" width="{tw:.1f}" height="12" fill="{col}"/>')
            c.text(X + 3, Y - 2.5, tag, size=8.5, fill="#ffffff", weight="600")
        bx(0.40, 0.32, 0.22, 0.62, "#000000", "P person")
        bx(0.08, 0.52, 0.26, 0.34, FOOT[0], "1")
        bx(0.68, 0.46, 0.20, 0.24, FOOT[1], "2")
        if proposals:
            bx(0.70, 0.16, 0.18, 0.20, FOOT[2], "3")
        if unseen and not proposals:
            c.a(f'<rect x="{x + 6}" y="{y + 6}" width="96" height="14" fill="#ffffff" opacity="0.8"/>')
            c.text(x + 9, y + 16.5, "3 (not visible here)", size=8.3, fill=FOOT[2])
    return draw


def ph_obb3d():
    """Lifted floor-parallel OBBs over the canonical floor grid."""
    def draw(c, x, y, w, h):
        fx0, fy0 = x + 12, y + h - 14
        dx, dy, sk = w - 56, -(h * 0.32), 30
        for i in range(6):
            t = i / 5
            c.a(f'<line x1="{fx0 + t * dx:.1f}" y1="{fy0}" x2="{fx0 + t * dx + sk:.1f}" y2="{fy0 + dy:.1f}" '
                f'stroke="{GRIDLINE}"/>')
            c.a(f'<line x1="{fx0 + t * sk:.1f}" y1="{fy0 + t * dy:.1f}" x2="{fx0 + dx + t * sk:.1f}" '
                f'y2="{fy0 + t * dy:.1f}" stroke="{GRIDLINE}"/>')

        def cube(u, v, bw, bd, bh, col):
            ox, oy = fx0 + u * dx + v * sk, fy0 + v * dy
            ex, ey = bd * sk, bd * dy
            base = [(ox, oy), (ox + bw, oy), (ox + bw + ex, oy + ey), (ox + ex, oy + ey)]
            top = [(px, py - bh) for px, py in base]
            e = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for i, j in e:
                for poly in (base, top):
                    c.a(f'<line x1="{poly[i][0]:.1f}" y1="{poly[i][1]:.1f}" x2="{poly[j][0]:.1f}" y2="{poly[j][1]:.1f}" '
                        f'stroke="{col}" stroke-width="1.4"/>')
                c.a(f'<line x1="{base[i][0]:.1f}" y1="{base[i][1]:.1f}" x2="{top[i][0]:.1f}" y2="{top[i][1]:.1f}" '
                    f'stroke="{col}" stroke-width="1.4"/>')
            r = _lcg(int(u * 100 + v * 10))
            for _ in range(24):
                a, b, z = r(), r(), r()
                c.a(f'<circle cx="{ox + a * bw + b * ex:.1f}" cy="{oy + b * ey - z * bh:.1f}" r="1" fill="{col}" '
                    f'opacity="0.7"/>')
        cube(0.08, 0.35, 44, 0.35, 30, FOOT[0])
        cube(0.52, 0.2, 26, 0.25, 16, FOOT[1])
        cube(0.66, 0.62, 18, 0.2, 40, FOOT[2])
        c.text(fx0 + 2, y + 14, "depth band · erosion {0, 5, 11} · min-area rectangle", size=8, fill=DIM)
    return draw


def ph_before_after():
    """The critic on one frame: a flagged box before repair, the repaired graph after."""
    def draw(c, x, y, w, h):
        hw = (w - 30) / 2
        ph_bev(flagged=True)(c, x, y, hw, h - 16)
        ph_bev(repaired=True)(c, x + hw + 30, y, hw, h - 16)
        c.a(f'<line x1="{x + hw + 6:.1f}" y1="{y + h / 2 - 8:.1f}" x2="{x + hw + 24:.1f}" y2="{y + h / 2 - 8:.1f}" '
            f'stroke="{ORANGE}" stroke-width="1.8" marker-end="url(#ahO)"/>')
        c.text(x + hw / 2, y + h - 3, "before: violation flagged", size=8.3, fill=RED, anchor="middle")
        c.text(x + hw + 30 + hw / 2, y + h - 3, "after one repair", size=8.3, fill=TXT, anchor="middle")
    return draw


# ------------------------- unlocalized per-object placeholders -------------------------

# the P4 relationship question (AG_RELATIONSHIP_QUERY_PROMPT), abridged line by line
P4_LINES = ['The first frame is the specific moment to analyze.',
            'The remaining frames provide surrounding context.',
            'The object "<o>" … may or may not be visible in',
            'the target frame. A person IS visible …',
            '1. ATTENTION: EXACTLY ONE of looking_at, …  (3)',
            '2. CONTACTING: ONE OR MORE of carrying, …  (17)',
            '3. SPATIAL: ONE OR MORE of above, …  (6)',
            'Respond ONLY in JSON: {"attention": "<label>",',
            ' "contacting": [...], "spatial": [...]}']


def ph_prompt_p4(head="Question P₄(o) · Text Depends Only On o", size=7.9):
    return ph_text(P4_LINES, head=head, size=size)


def ph_caption_prefix(sorted_by_distance=True, head=None):
    """The caption prefix ``caption_all`` puts before P4, one line per Stage-1 caption."""
    order = ["[Frame k_f] c_f      ← nearest first", "[Frame k_f±1] …", "[Frame k_f±2] …", "…"] if sorted_by_distance \
        else ["[Frame k₁] c₁", "[Frame k₂] c₂", "…", "[Frame k_F] c_F"]
    return ph_text(["The following captions describe what", "happens in this video:"] + order,
                   head=head or "Caption Context c(f) · Sorted By |kᵢ − f|", size=7.9)


def ph_transcript():
    return ph_text(["[Frame k₁] c₁", "[Frame k₂] c₂", "…", "[Frame k_F] c_F"],
                   head="Caption Transcript · F Captions, Time Order", size=7.9)


def ph_discovery(captions: bool):
    ctx = "[Frame kᵢ] cᵢ … (all captions)" if captions else "(No captions available for this video.)"
    return ph_text(["You are analyzing a video of a person …", ctx,
                    "candidate objects: \"bag\", \"bed\", … (36)",
                    "select ALL objects … the person interacts",
                    "with … Respond ONLY with a JSON list:",
                    '→ ["<name>", "<name>", …]'], head="P₆ Object Estimation", size=7.6)


def ph_context_frames(n=15):
    return ph_frames_strip(n=n, lowres=False, label="C · ≤ 15 key frames (linspace) · ≤ 512k px")


def ph_answer_p4(head="Answer For (f, o)"):
    return ph_text(['{"attention": "<label>",', ' "contacting": ["<label>", …],', ' "spatial": ["<label>", …]}', "",
                    "3 / 17 / 6 label sets"], head=head)


# ------------------------- shared blocks of the unlocalized figures -------------------------

def visual_input_block(c: MCanvas, y: float, x0: float = 30, frame_labels=("k₁", "k₂", "k₃")) -> dict:
    """Frames → shared annotated context C → [f ; C] = Q(f): the one video every P4 call
    receives (``_build_annotated_context`` + ``_prepend_target_frame``).  Returns the
    rectangle of the Q(f) slot so the caller can wire it into the query stage."""
    c.frames(x0, y, ["frame_0", "frame_1", "frame_2"], w=84, h=56, labels=frame_labels, max_total_h=196)
    c.note(x0 + 42, y + 214, "Annotated Key Frames", size=9.5, fill=MUTED, anchor="middle")
    c.note(x0 + 42, y + 227, "k₁ … k_F", size=9.5, fill=MUTED, anchor="middle")
    c.flow(x0 + 84, y + 88, x0 + 110, y + 88, "F Frames", lsize=8.3, loff=(0, -7))
    tx = x0 + 112
    c.tool(tx, y + 50, 150, 76, "Annotated Context", "≤ 15 Key Frames (Linspace)", "≤ 512k px In Total")
    c.flow(tx + 150, y + 88, tx + 176, y + 88, "")
    sx = tx + 178
    sw, sh = c.fit("context_frames", h=92, default=(250, 92), max_w=270)
    c.slot_title(sx, y + 42, "Shared Context C · Same For Every Frame f")
    c.image_slot(sx, y + 42, sw, sh, "context_frames", placeholder=ph_context_frames())
    c.flow(sx + sw, y + 88, sx + sw + 26, y + 88, "C", lsize=8.5, loff=(0, -6))
    px = sx + sw + 28
    c.tool(px, y + 50, 150, 76, "Prepend Target Frame", "f Resized To C's Frame", "Size · Bilinear")
    c.tensor(px, y + 154, 150, 36, "Target Frame f", "The Annotated Frame", col=ORANGE)
    c.flow(px + 75, y + 154, px + 75, y + 128, "", col=ORANGE)
    c.flow(px + 150, y + 88, px + 176, y + 88, "")
    qx = px + 178
    qw, qh = c.fit("query_tensor", h=104, default=(300, 104), max_w=330)
    c.slot_title(qx, y + 36, "Visual Input Q(f) = [f ; C] · One Video Per Prompt")
    c.image_slot(qx, y + 36, qw, qh, "query_tensor",
                 placeholder=ph_frames_strip(n=15, target=True, label="[ target frame f ; ≤ 15 key frames ]"))
    nx = qx + qw + 12
    c.wrap(nx, y + 52, ["16 frames as one video;", "the target is first and", "shares its temporal token",
                        "pair with context frame 1", "(Qwen-VL patches frames", "in pairs)"], size=8.3, fill=DIM,
           lh=12)
    return {"q": (qx, y + 36, qw, qh), "right": qx + qw}


def object_set_block(c: MCanvas, y: float, model: str, captions: bool, x0: float = 30) -> dict:
    """SGDet object discovery (P6 over V and the class list, exact vocabulary
    intersection, P5 Yes/No score per object) and the PredCls GT object set.
    Returns the rectangle of the objects slot."""
    c.note(x0, y + 6, "SGDet", size=10, fill=ORANGE)
    c.tensor(x0, y + 14, 110, 40, "All Raw Frames", "Stride 8 If > 120", col=MUTED)
    c.flow(x0 + 110, y + 34, x0 + 132, y + 34, "")
    c.tool(x0 + 134, y + 8, 120, 52, "Resize Video", "≤ 19 Frames · 128k px")
    c.flow(x0 + 254, y + 34, x0 + 276, y + 34, "")
    vx = x0 + 278
    vw, vh = c.fit("video_v", h=58, default=(160, 58), max_w=190)
    c.slot_title(vx, y + 5, "V · The Whole Video")
    c.image_slot(vx, y + 5, vw, vh, "video_v",
                 placeholder=ph_frames_strip(n=12, lowres=True, label="≤ 19 frames · 84×140"))
    c.flow(vx + vw, y + 34, vx + vw + 22, y + 34, "V", lsize=8.5, loff=(0, -6))
    ex = vx + vw + 24
    c.vlm(ex, y + 5, 130, 58, model, prompts=(6,),
          sub2="All Captions As Prefix" if captions else "“(No Captions Available)”")
    c.tensor(ex, y + 82, 130, 30, "Class List · 36 AG Names", None, col=MUTED)
    c.flow(ex + 65, y + 82, ex + 65, y + 65, "")
    c.flow(ex + 130, y + 34, ex + 152, y + 34, "")
    cx = ex + 154
    c.tensor(cx, y + 15, 100, 38, "Candidates", "JSON, Free Names", col=MUTED)
    c.flow(cx + 100, y + 34, cx + 122, y + 34, "")
    ix = cx + 124
    c.tool(ix, y + 11, 116, 46, "∩ AG Vocabulary", "Exact String Match")
    c.flow(ix + 116, y + 34, ix + 138, y + 34, "")
    yx = ix + 140
    c.vlm(yx, y + 5, 124, 58, model, prompts=(5,), sub2="“Is There A <o>?”")
    c.flow(yx + 124, y + 34, yx + 146, y + 34, "")
    ox = yx + 148
    ow, oh = c.fit("objects", h=96, default=(186, 96), max_w=210)
    c.slot_title(ox, y - 6, "Objects O + Object Score")
    c.image_slot(ox, y - 6, ow, oh, "objects", placeholder=ph_objects())
    # the P6 prompt / answer, under the discovery call
    dx, dw = ex + 140, yx - 10 - (ex + 140)
    c.image_slot(dx, y + 72, dw, 92, "discovery_card", placeholder=ph_discovery(captions))
    c.arrow(ex + 120, y + 63, ex + 150, y + 72, col=DIM, dashed=True, head=False)
    c.note(dx + dw / 2, y + 176, "P₆ Prompt And Raw Answer (SGDet) · ≤ 256 New Tokens", size=8.5, fill=DIM,
           anchor="middle")
    # PredCls: the GT object set
    c.tensor(yx, y + 82, 124, 30, "GT Video Objects", None, col=MUTED)
    c.note(yx, y + 124, "PredCls", size=10, fill=ORANGE)
    c.note(yx + 62, y + 76, "1 Token At T = 0", size=7.8, fill=DIM, anchor="middle")
    c.elbow(yx + 124, y + 97, ox - 2, y + 80, xm=yx + 136)
    return {"objects": (ox, y - 6, ow, oh), "right": ox + ow}


def perception_block(c: MCanvas, y: float) -> None:
    """Stage 1 of both localized figures: frozen Pi-3 and GDino, the canonical frame,
    the BEV map, the label normalisation and the 2-D → 3-D lift, ending in the object
    tables of the two modes (lib/mllm/tools/build_caches.py).  Occupies y .. y + 270."""
    c.frames(30, y, ["frame_0", "frame_1", "frame_2"], w=84, h=56, labels=("t₁", "t₂", "t₃"), max_total_h=196)
    c.note(72, y + 214, "Video Frames", size=10, fill=MUTED, anchor="middle")
    c.flow(114, y + 40, 140, y + 39, "")
    c.flow(114, y + 140, 140, y + 163, "")
    c.box(142, y + 8, 140, 62, "Frozen Pi-3", "Points · Confidence · Cameras", kind="frozen", frozen=True,
          tsize=11.5, ssize=8.8)
    c.flow(282, y + 39, 310, y + 39, "")
    c.tool(312, y + 2, 160, 74, "World → Canonical", "T_XY ∘ T_Δ ∘ T_auto", "z Up · Floor z = 0 · Metres")
    c.flow(472, y + 39, 500, y + 39, "")
    c.slot_title(502, y - 6, "Canonical Points + Camera Path")
    c.image_slot(502, y - 6, 190, 112, "cloud", placeholder=ph_cloud())
    c.flow(692, y + 39, 720, y + 39, "")
    c.tool(722, y + 2, 150, 74, "BEV Render", "≤ 40 Views · 3 cm Voxels", "Top-Down · 0.5 m Grid")
    c.flow(872, y + 39, 900, y + 39, "")
    c.slot_title(902, y - 12, "BEV Map + Metric Window")
    c.image_slot(902, y - 12, 180, 124, "bev_base", placeholder=ph_bev(marks=False))
    c.box(142, y + 132, 140, 62, "Frozen GDino", "2-D Proposals Per Frame", kind="frozen", frozen=True,
          tsize=11.5, ssize=8.8)
    c.flow(282, y + 163, 310, y + 170, "")
    c.slot_title(312, y + 126, "2-D Proposals")
    c.image_slot(312, y + 126, 160, 96, "proposals_2d", placeholder=ph_marked_frame(unseen=False, proposals=True))
    c.flow(472, y + 170, 500, y + 170, "")
    c.tool(502, y + 140, 150, 60, "Label Normalisation", "Alias Table → AG Names", "Score ≥ 0.25 · Person Kept")
    c.flow(652, y + 170, 680, y + 170, "")
    c.tool(682, y + 132, 190, 76, "Lift To 3-D", "Depth Band · Erosion {0, 5, 11}", "Min-Area Floor-Parallel OBB")
    c.arrow(630, y + 106, 630, y + 119, head=False)
    c.arrow(630, y + 119, 777, y + 119, head=False)
    c.arrow(777, y + 119, 777, y + 130)
    c.note(704, y + 114, "Points Of That Frame", size=8.5, fill=MUTED, anchor="middle")
    c.flow(872, y + 170, 900, y + 174, "")
    c.slot_title(902, y + 126, "Lifted OBBs")
    c.image_slot(902, y + 126, 180, 100, "lifted_obbs", placeholder=ph_obb3d())
    c.tensor(1110, y + 20, 180, 40, "Map + Window", "Base Image · x₀, y₀, px/m", col=MUTED)
    c.tensor(1110, y + 132, 180, 40, "SGDet Object Table", "Proposals + Lifted OBBs", col=MUTED)
    c.tensor(1110, y + 182, 180, 40, "PredCls Object Table", "GT Objects + GT OBBs", col=MUTED)
    c.flow(1082, y + 50, 1108, y + 40, "")
    c.flow(1082, y + 170, 1108, y + 152, "")
    c.note(1200, y + 238, "PredCls boxes come from the annotation", size=8.5, fill=DIM, anchor="middle")
    c.note(30, y + 254, "Annotation-independent caches, built once per video on CPU (lib/mllm/tools/build_caches.py); "
                        "every geometric quantity downstream, including the critic's checks, lives in this frame.",
           size=9, fill=DIM)


def payload_images(c: MCanvas, x: float, y: float, title_col=ORANGE) -> None:
    """The four images and the text of one Track A payload, inside a dashed
    'one prompt per annotated frame' box at (x, y); 440 × 296."""
    c.a(f'<rect x="{x}" y="{y}" width="440" height="296" rx="12" fill="none" stroke="{RULE}" stroke-dasharray="6 4"/>')
    c.note(x + 12, y + 18, "One Prompt Per Annotated Frame f", size=9.5, fill=MUTED)
    c.slot_title(x + 14, y + 42, "Image 1 · Target + Set-Of-Mark", col=title_col)
    c.image_slot(x + 14, y + 42, 170, 110, "marked_frame", placeholder=ph_marked_frame())
    c.slot_title(x + 198, y + 42, "Images 2–3 · Context")
    c.image_slot(x + 198, y + 42, 110, 70, "context_0", placeholder=ph_photo(0, "first key frame"))
    c.image_slot(x + 318, y + 42, 110, 70, "context_1", placeholder=ph_photo(2, "last key frame"))
    c.note(x + 313, y + 130, "first and last other annotated frames", size=8.3, fill=DIM, anchor="middle")
    c.slot_title(x + 14, y + 174, "Image 4 · Marked BEV")
    c.image_slot(x + 14, y + 174, 170, 112, "marked_bev", placeholder=ph_bev(marks=True))
    c.slot_title(x + 198, y + 174, "Text · Rules, Legend, Metric Table, Task")
    c.image_slot(x + 198, y + 174, 230, 112, "prompt_card",
                 placeholder=ph_text(["z up · floor z = 0 · box = {center, size, yaw}",
                                      "Image 1 target · 2-3 context · 4 map",
                                      "- P person: center=(x, y, z) m, size=(l, w, h) m",
                                      "- camera: position=(x, y, z) m, heading θ",
                                      "- 1 <label>: visible …; center=(…) m",
                                      "- 3 <label>: NOT visible …; center=(…) m",
                                      "Task: … answer with ONLY this JSON"], size=8))


def parse_block(c: MCanvas, y: float, x0: float = 30, verify_clip: str = "[f ; ±15 Frames, Every 2nd]",
                verify_prefix: str = "No Text Prefix") -> dict:
    """Stage 3 of the unlocalized figures: raw response → parse and validate → the
    ghosted Yes/No verification (with the clip it would read) → default scores →
    the class-only scene graph → the per-video pickle.  Returns the slot rectangle."""
    c.tensor(x0, y + 30, 130, 40, "Raw Response", "One Per (f, o)", col=MUTED)
    c.flow(x0 + 130, y + 50, x0 + 156, y + 50, "")
    px = x0 + 158
    c.tool(px, y + 16, 150, 68, "Parse And Validate", "JSON → Labels In 3 / 17 / 6 Sets", "Invalid Head → Unknown")
    c.flow(px + 150, y + 50, px + 176, y + 50, "Parsed", lsize=8.3, loff=(0, -7))
    gx = px + 178
    c.box(gx, y + 22, 176, 56, "Yes / No Verification", "P₅ Per Predicted Label", kind="ghost",
          sub2="Off In Every Reported Run", tsize=10.5, ssize=8.5)
    c.a('<g opacity="0.45">')
    c.badge(gx + 8, y + 13, 5, s=16)
    c.a('</g>')
    c.box(gx, y - 44, 176, 44, "Verification Clip", verify_clip, kind="ghost", sub2=verify_prefix, tsize=9.5,
          ssize=8)
    c.arrow(gx + 88, y + 0, gx + 88, y + 20, col=GHOST_S, dashed=True)
    c.flow(gx + 176, y + 50, gx + 202, y + 50, "")
    dx = gx + 204
    c.tool(dx, y + 22, 150, 56, "Default Scores", "Every Predicted Label 1.0", "Every Other Predicate 0")
    c.flow(dx + 150, y + 50, dx + 176, y + 50, "")
    sx = dx + 178
    sw, sh = c.fit("scene_graph", h=140, default=(304, 140), max_w=330)
    c.slot_title(sx, y - 4, "World Scene Graph At Frame f")
    c.image_slot(sx, y - 4, sw, sh, "scene_graph", placeholder=ph_scene_graph())
    c.flow(sx + sw, y + 50, sx + sw + 26, y + 50, "")
    kx = sx + sw + 28
    c.tensor(kx, y + 24, min(1290 - kx, 210), 52, "Pickle Per Video", "frames → predictions · Class Only", col=MUTED)
    c.note(kx, y + 92, "SGDet has no boxes: scored by", size=8.5, fill=DIM)
    c.note(kx, y + 105, "class only (loc3d at τ = 0)", size=8.5, fill=DIM)
    return {"graph": (sx, y - 4, sw, sh), "ghost_x": gx}
