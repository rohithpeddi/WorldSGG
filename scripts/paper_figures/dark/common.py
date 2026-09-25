"""Shared drawing primitives for the dark "stage band + numbered cards" method
figures (``fig_worldwise.py``, ``fig_worldwise_plus.py``, ``fig_worldwise_pp.py``).

The visual grammar follows the reference hero figure: a serif band title on a
horizontal rule, left-to-right data flow, three module kinds (frozen with a
snowflake, learnable in purple, the component this variant introduces in
orange), token grids drawn as coloured squares, script-L losses, and a right-hand
column of three numbered cards.

Every figure script builds a :class:`Canvas`, places elements with the helpers
below and writes a self-contained SVG.  Image *slots* (:meth:`Canvas.image_slot`)
take a PNG produced by ``scripts/paper_figures/dump_intermediates.py``; when the
file is absent a labelled placeholder is drawn instead, so the layout can be
reviewed before the inference panels exist.
"""
from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
BG = "#0f1114"
RULE = "#3a3f47"
TXT = "#e6e8ec"
MUTED = "#9aa3b2"
DIM = "#6b7280"
PURPLE_F, PURPLE_S = "#2b2344", "#7c6bd6"     # learnable, inherited
FROZ_F, FROZ_S = "#262b33", "#5b6470"         # frozen
NEW_F, NEW_S = "#3a2415", "#ff8a3d"           # introduced by this variant
GHOST_F, GHOST_S = "#171a1f", "#2f343c"       # unchanged, de-emphasised
BLUE = "#5aa2f2"      # decoded FRCNN tokens
VIOLET = "#a78bfa"    # DINOv3 semantics
TEAL = "#2dd4bf"      # pi3 geometry
LAT = ["#fdba74", "#fb923c", "#f9a8d4", "#e879f9", "#c084fc"]   # recovered world token
MASKC = "#64748b"
ORANGE = "#ff8a3d"
RED = "#f87171"
GREEN = "#4ade80"

SERIF = "Georgia, 'Times New Roman', serif"
SANS = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
KINDS = {"learn": (PURPLE_F, PURPLE_S), "frozen": (FROZ_F, FROZ_S),
         "new": (NEW_F, NEW_S), "ghost": (GHOST_F, GHOST_S)}


def esc(s: str) -> str:
    return html.escape(s, quote=False)


class Canvas:
    def __init__(self, width: int, height: int, images: Optional[Dict[str, Path]] = None):
        self.w, self.h = width, height
        self.parts: List[str] = []
        self.images = images or {}
        self.missing: List[str] = []
        self._defs()

    # -- output ---------------------------------------------------------------
    def _defs(self):
        self.parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">')
        self.parts.append('''<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#9aa3b2"/></marker>
  <marker id="ahO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#ff8a3d"/></marker>
  <marker id="ahR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#f87171"/></marker>
  <marker id="ahB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5aa2f2"/></marker>
  <marker id="ahV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#a78bfa"/></marker>
  <marker id="ahT" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#2dd4bf"/></marker>
  <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#ff8a3d" stroke-width="1.6"/></pattern>
  <linearGradient id="photo0" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6b5a4a"/><stop offset="1" stop-color="#2f3a4c"/></linearGradient>
  <linearGradient id="photo1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4a5a6b"/><stop offset="1" stop-color="#4c3a2f"/></linearGradient>
  <linearGradient id="photo2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5a6b4a"/><stop offset="1" stop-color="#3a2f4c"/></linearGradient>
</defs>''')
        self.parts.append(f'<rect width="{self.w}" height="{self.h}" fill="{BG}"/>')

    def a(self, s: str):
        self.parts.append(s)

    def write(self, path: Path):
        self.parts.append("</svg>")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(self.parts), encoding="utf-8")
        return path

    # -- text -----------------------------------------------------------------
    def text(self, x, y, s, size=12, fill=TXT, anchor="start", weight="normal", font=SANS, style="normal"):
        self.a(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{fill}" '
               f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}">{esc(s)}</text>')

    def rich(self, x, y, parts, size=12, anchor="start", font=SANS):
        t = [f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" text-anchor="{anchor}" fill="{TXT}">']
        for s, o in parts:
            attrs = []
            if o.get("fill"): attrs.append(f'fill="{o["fill"]}"')
            if o.get("sub"): attrs.append(f'baseline-shift="sub" font-size="{size * 0.72}"')
            if o.get("italic"): attrs.append('font-style="italic"')
            if o.get("bold"): attrs.append('font-weight="700"')
            if o.get("serif"): attrs.append(f'font-family="{SERIF}"')
            t.append(f'<tspan {" ".join(attrs)}>{esc(s)}</tspan>')
        t.append("</text>")
        self.a("".join(t))

    def loss(self, x, y, sub, col, size=15):
        self.rich(x, y, [("ℒ", {"fill": col, "serif": True, "italic": True}),
                         (sub, {"fill": col, "sub": True, "serif": True})], size=size)

    def wrap(self, x, y, lines: Sequence[str], size=11.5, fill=MUTED, lh=17):
        for i, ln in enumerate(lines):
            self.text(x, y + i * lh, ln, size=size, fill=fill)

    # -- shapes ---------------------------------------------------------------
    def snow(self, x, y, r=7, col="#dfe7f3"):
        p = []
        for k in range(3):
            ang = math.radians(k * 60)
            dx, dy = r * math.cos(ang), r * math.sin(ang)
            p.append(f'<line x1="{x - dx:.1f}" y1="{y - dy:.1f}" x2="{x + dx:.1f}" y2="{y + dy:.1f}"/>')
            for s in (-1, 1):
                for t in (0.5, 0.8):
                    px, py = x + s * dx * t, y + s * dy * t
                    for b in (-1, 1):
                        a2 = ang + b * math.radians(60)
                        p.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px + s * r * 0.3 * math.cos(a2):.1f}" '
                                 f'y2="{py + s * r * 0.3 * math.sin(a2):.1f}"/>')
        self.a(f'<g stroke="{col}" stroke-width="1.3" stroke-linecap="round">{"".join(p)}</g>')

    def box(self, x, y, w, h, title, sub=None, kind="learn", frozen=False, r=9, tsize=12.5, ssize=10.5, sub2=None):
        f, s = KINDS[kind]
        self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
        col = TXT if kind != "ghost" else DIM
        lines = [title] + ([sub] if sub else []) + ([sub2] if sub2 else [])
        n = len(lines)
        y0 = y + h / 2 - (n - 1) * 7.5 + 4.5
        for i, ln in enumerate(lines):
            if i == 0:
                self.text(x + w / 2, y0 + i * 15, ln, size=tsize, fill=col, anchor="middle", weight="600")
            else:
                self.text(x + w / 2, y0 + i * 15, ln, size=ssize, fill=(MUTED if kind != "ghost" else DIM), anchor="middle")
        if frozen:
            self.snow(x + w - 11, y + 11)

    def arrow(self, x1, y1, x2, y2, col=MUTED, dashed=False, width=1.6, head=True, curve=None):
        d = f"M{x1} {y1} L{x2} {y2}" if curve is None else curve
        dash = ' stroke-dasharray="4 4"' if dashed else ""
        mk = ""
        if head:
            mid = {ORANGE: "ahO", RED: "ahR", BLUE: "ahB", VIOLET: "ahV", TEAL: "ahT"}.get(col, "ah")
            mk = f' marker-end="url(#{mid})"'
        self.a(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}"{dash}{mk}/>')

    def elbow(self, x1, y1, x2, y2, col=MUTED, dashed=False, width=1.6, xm=None, head=True):
        xm = (x1 + x2) / 2 if xm is None else xm
        self.arrow(x1, y1, x2, y2, col, dashed, width, head, curve=f"M{x1} {y1} H{xm} V{y2} H{x2}")

    def grid(self, x, y, rows, cols, cell=14, gap=4, colors=None, kinds=None, cube=False):
        for r in range(rows):
            for c in range(cols):
                cx, cy = x + c * (cell + gap), y + r * (cell + gap)
                k = kinds[r][c] if kinds else "v"
                col = colors[r][c] if colors else BLUE
                if cube and k == "v":
                    self.a(f'<path d="M{cx + 3} {cy - 3} h{cell} l-3 3 h-{cell} z" fill="{col}" opacity="0.55"/>')
                    self.a(f'<path d="M{cx + cell} {cy} l3 -3 v{cell} l-3 3 z" fill="{col}" opacity="0.35"/>')
                if k == "v":
                    self.a(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="2" fill="{col}"/>')
                elif k == "m":
                    self.a(f'<rect x="{cx + 0.5}" y="{cy + 0.5}" width="{cell - 1}" height="{cell - 1}" rx="2" fill="none" '
                           f'stroke="{MASKC}" stroke-dasharray="2.5 2" stroke-width="1.2"/>')
                elif k == "a":
                    self.a(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="2" fill="url(#hatch)" '
                           f'stroke="{ORANGE}" stroke-width="1"/>')

    def band_title(self, y, label, x0=30, x1=1290, size=20):
        self.a(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{RULE}" stroke-width="1.5"/>')
        tw = len(label) * size * 0.53 + 40
        cx = (x0 + x1) / 2
        self.a(f'<rect x="{cx - tw / 2}" y="{y - 15}" width="{tw}" height="30" fill="{BG}"/>')
        self.text(cx, y + 6, label, size=size, fill=TXT, anchor="middle", weight="700", font=SERIF)

    def card(self, x, y, w, h, n, title, body_lines, size=11.5):
        self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#171a1f" stroke="#262b33" stroke-width="1.5"/>')
        cx, cy = x + 34, y + h / 2 - 8 * len(body_lines) + 14
        self.a(f'<circle cx="{cx}" cy="{cy}" r="13" fill="{ORANGE}"/>')
        self.text(cx, cy + 4.5, str(n), size=13, fill="#1a0f05", anchor="middle", weight="700")
        self.text(x + 62, cy + 5, title, size=19, fill=TXT, weight="700")
        self.wrap(x + 62, cy + 30, body_lines, size=size)

    def legend(self, y, items: Sequence[Tuple[str, str]], x0=30, extra_swatches: Sequence[Tuple[str, str]] = ()):
        """items: (kind, label) for the box kinds; extra_swatches: (colour|'mask'|'hatch', label)."""
        x = x0
        for kind, label in items:
            f, s = KINDS[kind]
            self.a(f'<rect x="{x}" y="{y - 11}" width="22" height="14" rx="3" fill="{f}" stroke="{s}" stroke-width="1.4"/>')
            if kind == "frozen":
                self.snow(x + 11, y - 4, r=4)
            self.text(x + 28, y, label, size=10.5, fill=MUTED)
            x += 28 + len(label) * 6.2 + 26
        for col, label in extra_swatches:
            if col == "mask":
                self.a(f'<rect x="{x + 0.5}" y="{y - 10.5}" width="12" height="12" rx="2" fill="none" stroke="{MASKC}" stroke-dasharray="2.5 2"/>')
            elif col == "hatch":
                self.a(f'<rect x="{x}" y="{y - 11}" width="13" height="13" rx="2" fill="url(#hatch)" stroke="{ORANGE}"/>')
            else:
                self.a(f'<rect x="{x}" y="{y - 11}" width="13" height="13" rx="2" fill="{col}"/>')
            self.text(x + 18, y, label, size=10.5, fill=MUTED)
            x += 18 + len(label) * 6.2 + 24

    # -- images ---------------------------------------------------------------
    def img_size(self, key: str) -> Optional[Tuple[int, int]]:
        """(width, height) in pixels of the PNG behind ``key``, or None."""
        p = self.images.get(key)
        if p is None or not Path(p).exists():
            return None
        try:
            from PIL import Image
            with Image.open(p) as im:
                return im.size
        except Exception:                                  # pragma: no cover - PIL missing
            return None

    def fit(self, key: str, w: Optional[float] = None, h: Optional[float] = None,
            default: Tuple[float, float] = (150, 100), max_w: Optional[float] = None,
            max_h: Optional[float] = None) -> Tuple[float, float]:
        """Slot size for ``key``: the missing side follows the image's aspect
        ratio; without an image the ``default`` fills in.  ``max_w`` / ``max_h``
        cap the derived side (the image is then letter-boxed by ``image_slot``)."""
        size = self.img_size(key)
        if w is None and h is None:
            w, h = default
        if size is None:
            return (w if w is not None else default[0], h if h is not None else default[1])
        iw, ih = size
        if w is None:
            w = h * iw / ih
            if max_w is not None and w > max_w:
                w = max_w
        elif h is None:
            h = w * ih / iw
            if max_h is not None and h > max_h:
                h = max_h
        return (w, h)

    def image_slot(self, x, y, w, h, key: str, caption: Optional[str] = None, border=RULE, placeholder=None,
                   caption_col=MUTED, caption_size=9.5):
        """Embed ``self.images[key]`` (a PNG) fitted inside (x, y, w, h), or draw a
        placeholder naming the missing panel.  Returns the drawn rectangle."""
        p = self.images.get(key)
        if p is not None and Path(p).exists():
            data = base64.b64encode(Path(p).read_bytes()).decode("ascii")
            self.a(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" '
                   f'xlink:href="data:image/png;base64,{data}"/>')
            self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="none" stroke="{border}" stroke-width="1"/>')
        else:
            self.missing.append(key)
            self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#171a1f" stroke="{border}" '
                   f'stroke-width="1" stroke-dasharray="4 3"/>')
            if placeholder:
                placeholder(self, x, y, w, h)
            else:
                self.text(x + w / 2, y + h / 2 + 4, key, size=9, fill=DIM, anchor="middle")
        if caption:
            self.text(x, y + h + 12, caption, size=caption_size, fill=caption_col)
        return (x, y, w, h)

    def frames(self, x, y, keys: Sequence[str], w=100, h=70, gapy=8, labels: Sequence[str] = ("t₁", "t₂", "t₃"),
               max_total_h: float = 380):
        """Three stacked frame thumbnails, real when the images exist.  The height
        follows the frame's aspect ratio at width ``w`` (portrait videos get tall
        thumbnails), capped so the stack fits in ``max_total_h``."""
        hs = []
        for key in keys:
            _, hh = self.fit(key, w=w, default=(w, h), max_h=(max_total_h - gapy * (len(keys) - 1)) / len(keys))
            hs.append(hh)
        yy = y
        for i, key in enumerate(keys):
            def ph(c, x0, y0, w0, h0, i=i):
                c.a(f'<rect x="{x0 + 4}" y="{y0 + 4}" width="{w0 - 8}" height="{h0 - 8}" rx="2" fill="url(#photo{i % 3})"/>')
                c.text(x0 + w0 / 2, y0 + h0 - 8, labels[i], size=9, fill="#c8cdd6", anchor="middle")
            self.image_slot(x, yy, w, hs[i], key, placeholder=ph, border="#3a4048")
            yy += hs[i] + gapy
        return yy - gapy

    def strip(self, x, y, h, items: Sequence[Tuple[str, str, float]], gap=16, caption_size=9.5, max_w=None):
        """A row of image slots of common height ``h``; each width follows the
        image's aspect (``default_w`` without an image).  ``items`` are
        (key, caption, default_w).  Returns the x after the last slot."""
        for key, caption, dw in items:
            w, _ = self.fit(key, h=h, default=(dw, h), max_w=max_w)
            self.image_slot(x, y, w, h, key, caption=caption, caption_size=caption_size)
            x += w + gap
        return x


    # -- stage-figure helpers -------------------------------------------------------
    def flow(self, x1, y1, x2, y2, label: str = "", col=MUTED, width=1.6, dashed=False, lpos: float = 0.5,
             loff: Tuple[float, float] = (0, -6), lsize: float = 9, lcol=None, elbow_x: Optional[float] = None,
             anchor: str = "middle"):
        """A labelled, headed arrow from (x1, y1) to (x2, y2): what flows is written
        next to it so every connection names its source and its payload."""
        if elbow_x is None:
            self.arrow(x1, y1, x2, y2, col=col, width=width, dashed=dashed)
            lx, ly = x1 + (x2 - x1) * lpos, y1 + (y2 - y1) * lpos
        else:
            self.elbow(x1, y1, x2, y2, col=col, width=width, dashed=dashed, xm=elbow_x)
            lx, ly = elbow_x, (y1 + y2) / 2
        if label:
            self.text(lx + loff[0], ly + loff[1], label, size=lsize, fill=lcol or col, anchor=anchor)

    def tensor(self, x, y, w, h, title: str, sub: Optional[str] = None, col=MUTED):
        """A data node: a rounded outline (no fill) naming an intermediate tensor."""
        self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{col}" stroke-width="1.2" '
               f'stroke-dasharray="1 0"/>')
        if sub:
            self.text(x + w / 2, y + h / 2 - 2, title, size=10.5, fill=TXT, anchor="middle", weight="600")
            self.text(x + w / 2, y + h / 2 + 11, sub, size=9, fill=col, anchor="middle")
        else:
            self.text(x + w / 2, y + h / 2 + 4, title, size=10.5, fill=TXT, anchor="middle", weight="600")

    def loss_node(self, x, y, w, name: str, formula: str, col=RED, note: Optional[str] = None):
        """A loss: script-L with subscript, its formula, and an optional note."""
        self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{46 if note else 34}" rx="8" fill="#1a1417" stroke="{col}" '
               f'stroke-width="1.4" stroke-dasharray="5 3"/>')
        self.loss(x + 10, y + 22, name, col, size=15)
        self.text(x + 44, y + 21, formula, size=9.5, fill=col)
        if note:
            self.text(x + 10, y + 38, note, size=8.5, fill=MUTED)

    def stage(self, y, title: str, x0=30, x1=1290):
        """A stage band title on a rule; returns the y where content starts."""
        self.band_title(y, title, x0=x0, x1=x1, size=17)
        return y + 24


GEOMETRY_IMAGES = ["obb_0", "obb_1", "obb_2", "scene3d_1", "camera_path", "motion"]


def geometry_row(c: "Canvas", y: float, video: str, h: float = 165) -> float:
    """The 'geometry scaffold inputs' row shared by the three figures: projected OBB
    corners at the key frames, the canonical-frame scene, the camera path and the
    object-centre trajectories.  Returns the y below the row."""
    c.band_title(y, f"Geometry scaffold inputs · {video}", size=15)
    py = y + 26
    c.strip(30, py, h, [
        ("obb_0", "1  t₁", 100),
        ("obb_1", "2  unseen frame", 100),
        ("obb_2", "3  last frame", 100),
        ("scene3d_1", "4  canonical frame: π³ points, OBBs, camera", 200),
        ("camera_path", "5  camera poses → ego-motion encoder", 190),
        ("motion", "6  object centres → motion encoder", 190),
    ], gap=16, caption_size=9, max_w=300)
    c.text(30, py + h + 36, "1-3  input OBB corners projected onto the frame with the focal length fitted to π³'s own points; "
                            "an unseen object (dashed) still has corners although it has no 2-D box → structural + spatial encoders · "
                            "4  the scene the boxes were fitted in · 5  per-step relative pose · 6  velocities and accelerations",
           size=9.5, fill=DIM)
    return py + h + 46


def image_map(directory: Optional[str], names: Sequence[str], extra: Optional[Dict[str, str]] = None) -> Dict[str, Path]:
    """{name: directory/name.png} for each name (plus ``extra`` explicit paths)."""
    out: Dict[str, Path] = {}
    if directory:
        d = Path(directory)
        for n in names:
            out[n] = d / f"{n}.png"
    for k, v in (extra or {}).items():
        out[k] = Path(v)
    return out
