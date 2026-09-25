"""Shared drawing primitives for the "stage band + numbered cards" method
figures (``fig_worldwise.py``, ``fig_worldwise_plus.py``, ``fig_worldwise_pp.py``).

Two palettes: ``light`` (white background, the default) and ``dark`` (the hero
palette the figures were designed in).  ``--theme dark|light`` on any figure
script, or ``FIG_THEME``, selects one; ``--caps title|upper|none`` (``FIG_CAPS``)
sets how every text is capitalised (Title Case by default).

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
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# palette: one dict per theme; the chosen one is bound to the module names below
# ---------------------------------------------------------------------------
PALETTES = {
    "dark": dict(
        BG="#0f1114", RULE="#3a3f47", TXT="#e6e8ec", MUTED="#9aa3b2", DIM="#6b7280",
        PURPLE_F="#2b2344", PURPLE_S="#7c6bd6",       # learnable, inherited
        FROZ_F="#262b33", FROZ_S="#5b6470",           # frozen
        NEW_F="#3a2415", NEW_S="#ff8a3d",             # introduced by this variant
        GHOST_F="#171a1f", GHOST_S="#2f343c",         # unchanged, de-emphasised
        TOOL_F="#0f2226", TOOL_S="#2dd4bf",           # deterministic program (MLLM figures)
        BLUE="#5aa2f2",       # decoded FRCNN tokens
        VIOLET="#a78bfa",     # DINOv3 semantics
        TEAL="#2dd4bf",       # pi3 geometry
        LAT=["#fdba74", "#fb923c", "#f9a8d4", "#e879f9", "#c084fc"],   # recovered world token
        MASKC="#64748b", ORANGE="#ff8a3d", RED="#f87171", GREEN="#4ade80",
        SNOW="#dfe7f3",                               # snowflake strokes
        CARD_F="#171a1f", CARD_S="#262b33",           # numbered cards
        LOSS_F="#1a1417",                             # loss-node fill
        PH_F="#171a1f", FRAME_B="#3a4048",            # placeholder fill, frame border
        NODE_F="#1c2027", LINE="#5b6470", GRIDLINE="#2f343c", FILM_F="#0b0d10", SPROCKET="#2a2f37",
        RED_TXT="#fca5a5", BAR_OFF="#3a3f47", CARD="#12151a",
        # the four processing units: (stroke, tint fill)
        UNIT=[("#60a5fa", "#111b2b"), ("#fbbf24", "#231c0d"), ("#a78bfa", "#1b1630"), ("#34d399", "#0e231c")],
    ),
    "light": dict(
        BG="#ffffff", RULE="#c5cad3", TXT="#151a22", MUTED="#4b5563", DIM="#8b93a1",
        PURPLE_F="#ece8fb", PURPLE_S="#6a58d4",
        FROZ_F="#eef1f5", FROZ_S="#7d8797",
        NEW_F="#fff0e3", NEW_S="#ef6c1a",
        GHOST_F="#f7f8fa", GHOST_S="#d3d8e0",
        TOOL_F="#e6f7f5", TOOL_S="#0f9f93",
        BLUE="#2c7be5",
        VIOLET="#7c4dff",
        TEAL="#0f9f93",
        LAT=["#f59e0b", "#ea580c", "#db2777", "#c026d3", "#7c3aed"],
        MASKC="#6b7280", ORANGE="#ef6c1a", RED="#dc2626", GREEN="#16a34a",
        SNOW="#3b82f6",
        CARD_F="#f5f6f8", CARD_S="#dde2e9",
        LOSS_F="#fff5f5",
        PH_F="#f3f4f6", FRAME_B="#c5cad3",
        NODE_F="#f3f4f6", LINE="#9aa3b2", GRIDLINE="#dfe3e8", FILM_F="#e9ecf0", SPROCKET="#ffffff",
        RED_TXT="#b91c1c", BAR_OFF="#d1d5db", CARD="#f4f5f7",
        UNIT=[("#2563eb", "#f4f8ff"), ("#b45309", "#fffaf0"), ("#7c3aed", "#f8f5ff"), ("#047857", "#f1fbf6")],
    ),
}


def _pick(flag: str, env: str, default: str, choices) -> str:
    """A setting from ``--flag value`` on the command line, else ``$ENV``, else the default."""
    v = os.environ.get(env, default)
    argv = sys.argv
    if flag in argv and argv.index(flag) + 1 < len(argv):
        v = argv[argv.index(flag) + 1]
    for a in argv:
        if a.startswith(flag + "="):
            v = a.split("=", 1)[1]
    if v not in choices:
        raise SystemExit(f"{flag}: expected one of {sorted(choices)}, got {v!r}")
    return v


THEME = _pick("--theme", "FIG_THEME", "light", PALETTES)
CAPS = _pick("--caps", "FIG_CAPS", "title", ("title", "upper", "none"))
globals().update(PALETTES[THEME])

SERIF = "Georgia, 'Times New Roman', serif"
SANS = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
KINDS = {"learn": (PURPLE_F, PURPLE_S), "frozen": (FROZ_F, FROZ_S),
         "new": (NEW_F, NEW_S), "ghost": (GHOST_F, GHOST_S), "tool": (TOOL_F, TOOL_S)}


def esc(s: str) -> str:
    return html.escape(s, quote=False)


# ---------------------------------------------------------------------------
# capitalisation: every text drawn through Canvas.text / Canvas.rich passes here
# ---------------------------------------------------------------------------
_WORD = re.compile(r"^[A-Za-z][A-Za-z\-'\u2019/]*$")
_LEAD = "([{\"'\u201c\u2018"
_TRAIL = ")]}\"'\u201d\u2019,.:;!?"
_UNITS = {"px", "cm", "mm", "m", "ms"}


def cap(s: str, mode: Optional[str] = None) -> str:
    """Capitalise the first letter of every word (``title``) or upper-case the
    whole string (``upper``).  Tokens that are not plain words (formulas,
    subscripts, numbers, units, single-letter variables) are left alone, so
    ``t = 12``, ``p(class)``, ``sₜ,ₙ``, ``IoU`` and ``DINOv3-L`` survive."""
    mode = CAPS if mode is None else mode
    if not s or mode == "none":
        return s
    if mode == "upper":
        return s.upper()
    out = []
    for tok in re.split(r"(\s+)", s):
        if not tok or tok.isspace():
            out.append(tok)
            continue
        i = 0
        while i < len(tok) and tok[i] in _LEAD:
            i += 1
        core = tok[i:].rstrip(_TRAIL)
        # A token with an upper-case letter after its first is an identifier or
        # an abbreviation (mR, IoU, PredCls, ReLU) and is left as written.
        if (_WORD.match(core) and core not in _UNITS and not any(ch.isupper() for ch in core[1:])
                and (len(core) > 1 or (core in ("a", "i") and tok == core))):
            # Title Case every part of a hyphenated / slashed compound
            # ("tail-aware" → "Tail-Aware", as the hand-written "Ego-Motion").
            new = re.sub(r"(^|[-/])([a-z])", lambda m: m.group(1) + m.group(2).upper(), core)
            tok = tok[:i] + new + tok[i + len(core):]
        out.append(tok)
    return "".join(out)


UNIT_NAMES = ["Observed Objects Processing Unit", "Unobserved Objects Processing Unit",
              "Relationship Processing Unit", "Decoders"]


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
        self.parts.append(f'''<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker>
  <marker id="ahO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{ORANGE}"/></marker>
  <marker id="ahR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{RED}"/></marker>
  <marker id="ahB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{BLUE}"/></marker>
  <marker id="ahV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{VIOLET}"/></marker>
  <marker id="ahT" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{TEAL}"/></marker>
  <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="{ORANGE}" stroke-width="1.6"/></pattern>
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
    def text(self, x, y, s, size=12, fill=None, anchor="start", weight="normal", font=SANS, style="normal",
             capitalise: bool = True):
        fill = TXT if fill is None else fill
        if capitalise and "mono" not in font.lower():
            s = cap(s)
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
            if not (o.get("sub") or o.get("verbatim")):
                s = cap(s)
            t.append(f'<tspan {" ".join(attrs)}>{esc(s)}</tspan>')
        t.append("</text>")
        self.a("".join(t))

    def loss(self, x, y, sub, col, size=15):
        self.rich(x, y, [("ℒ", {"fill": col, "serif": True, "italic": True}),
                         (sub, {"fill": col, "sub": True, "serif": True})], size=size)

    def wrap(self, x, y, lines: Sequence[str], size=11.5, fill=None, lh=17):
        fill = MUTED if fill is None else fill
        for i, ln in enumerate(lines):
            self.text(x, y + i * lh, ln, size=size, fill=fill)

    # -- shapes ---------------------------------------------------------------
    def snow(self, x, y, r=7, col=None):
        col = SNOW if col is None else col
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
        label = cap(label)
        self.a(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{RULE}" stroke-width="1.5"/>')
        tw = len(label) * size * 0.53 + 40
        cx = (x0 + x1) / 2
        self.a(f'<rect x="{cx - tw / 2}" y="{y - 15}" width="{tw}" height="30" fill="{BG}"/>')
        self.text(cx, y + 6, label, size=size, fill=TXT, anchor="middle", weight="700", font=SERIF)

    def card(self, x, y, w, h, n, title, body_lines, size=11.5):
        self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{CARD_F}" stroke="{CARD_S}" stroke-width="1.5"/>')
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

    def image_slot(self, x, y, w, h, key: str, caption: Optional[str] = None, border=None, placeholder=None,
                   caption_col=None, caption_size=9.5):
        """Embed ``self.images[key]`` (a PNG) fitted inside (x, y, w, h), or draw a
        placeholder naming the missing panel.  Returns the drawn rectangle."""
        border = RULE if border is None else border
        caption_col = MUTED if caption_col is None else caption_col
        p = self.images.get(key)
        if p is not None and Path(p).exists():
            data = base64.b64encode(Path(p).read_bytes()).decode("ascii")
            self.a(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" '
                   f'xlink:href="data:image/png;base64,{data}"/>')
            self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="none" stroke="{border}" stroke-width="1"/>')
        else:
            self.missing.append(key)
            self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{PH_F}" stroke="{border}" '
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
            self.image_slot(x, yy, w, hs[i], key, placeholder=ph, border=FRAME_B)
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
        self.a(f'<rect x="{x}" y="{y}" width="{w}" height="{46 if note else 34}" rx="8" fill="{LOSS_F}" stroke="{col}" '
               f'stroke-width="1.4" stroke-dasharray="5 3"/>')
        self.loss(x + 10, y + 22, name, col, size=15)
        self.text(x + 44, y + 21, formula, size=9.5, fill=col)
        if note:
            self.text(x + 10, y + 38, note, size=8.5, fill=MUTED)

    def stage(self, y, title: str, x0=30, x1=1290):
        """A stage band title on a rule; returns the y where content starts."""
        self.band_title(y, title, x0=x0, x1=x1, size=17)
        return y + 24

    def strip_fit(self, x, y, h, items: Sequence[Tuple[str, str, float]], gap=14, limit=1290, max_aspect=3.4,
                  caption_size=9) -> float:
        """A row of image slots of common height that always fits in [x, limit].
        ``items`` are (key, caption, default_aspect); a real PNG supplies its own
        aspect (capped at ``max_aspect``).  When the row would overflow, the height
        is reduced so that it fits.  Returns the y below the captions."""
        def widths(hh):
            ws = []
            for key, _, asp in items:
                size = self.img_size(key)
                a = min(size[0] / size[1], max_aspect) if size else asp
                ws.append(a * hh)
            return ws
        ws = widths(h)
        gaps = gap * (len(items) - 1)
        if sum(ws) + gaps > limit - x:
            h = (limit - x - gaps) / (sum(ws) / h)
            ws = widths(h)
        xx = x
        for (key, cap_, _), w in zip(items, ws):
            self.image_slot(xx, y, w, h, key, caption=cap_, caption_size=caption_size)
            xx += w + gap
        return y + h + 16

    def set_height(self, height: int) -> None:
        """Shrink a scratch canvas to the placed content: rewrite the root element and the background."""
        self.parts[0] = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                         f'viewBox="0 0 {self.w} {height}" width="{self.w}" height="{height}">')
        self.parts[2] = f'<rect width="{self.w}" height="{height}" fill="{BG}"/>'
        self.h = height

    # -- processing units ------------------------------------------------------------
    # Every training-based method is drawn as four processing units, top to bottom:
    #   1 Observed Objects Processing Unit    2 Unobserved Objects Processing Unit
    #   3 Relationship Processing Unit        4 Decoders
    # A unit is a tinted, rounded enclosure with a numbered header; ``n = (1, 2)``
    # marks a region shared by units 1 and 2 (the WorldWise++ entity decoder).  Each
    # unit also writes a ``<!-- crop name x0 y0 x1 y1 -->`` marker that
    # crop_stages.py turns into one page of the paper.
    def _unit_cols(self, n):
        ns = n if isinstance(n, tuple) else (n,)
        return [UNIT[k - 1] for k in ns]

    def begin_unit(self, y, n, role: str = "", x0=22, x1=1298) -> float:
        """Open unit ``n`` whose enclosure starts at ``y``; returns the y where content starts."""
        self._unit = dict(idx=len(self.parts), y=y, n=n, x0=x0, x1=x1)
        cols = self._unit_cols(n)
        ns = n if isinstance(n, tuple) else (n,)
        px, py = x0 + 16, y + 12
        pw = 58 if len(ns) == 1 else 92
        if len(ns) == 1:
            self.a(f'<rect x="{px}" y="{py}" width="{pw}" height="24" rx="12" fill="{cols[0][0]}"/>')
        else:
            half = pw / 2
            self.a(f'<rect x="{px}" y="{py}" width="{pw}" height="24" rx="12" fill="{cols[1][0]}"/>')
            self.a(f'<path d="M{px + 12} {py} H{px + half} V{py + 24} H{px + 12} A12 12 0 0 1 {px + 12} {py}" '
                   f'fill="{cols[0][0]}"/>')
        tag = f"Unit {ns[0]}" if len(ns) == 1 else f"Units {ns[0]} + {ns[1]}"
        self.text(px + pw / 2, py + 16.5, tag, size=11.5, fill=BG, anchor="middle", weight="700")
        if len(ns) == 1:
            name = UNIT_NAMES[ns[0] - 1]
        else:
            name = "Shared By The " + " And ".join(UNIT_NAMES[k - 1].replace(" Processing Unit", "") for k in ns) \
                   + " Processing Units"
        self.text(px + pw + 12, py + 18, name, size=18, fill=cols[0][0] if len(ns) == 1 else TXT, weight="700",
                  font=SERIF)
        if role:
            self.text(px + pw + 12 + len(cap(name)) * 18 * 0.6 + 20, py + 17, role, size=11, fill=MUTED)
        return y + 50

    def end_unit(self, y1, crop: Optional[str] = None, crop_y0: Optional[float] = None,
                 crop_y1: Optional[float] = None):
        """Close the open unit at ``y1``: its enclosure is inserted behind the content."""
        u = self._unit
        x0, x1, y0 = u["x0"], u["x1"], u["y"]
        cols = self._unit_cols(u["n"])
        if len(cols) == 1:
            enc = (f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="16" fill="{cols[0][1]}" '
                   f'stroke="{cols[0][0]}" stroke-width="1.8"/>')
        else:
            gid = f"ovl{len(self.parts)}"
            enc = (f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0">'
                   f'<stop offset="0" stop-color="{cols[0][1]}"/><stop offset="1" stop-color="{cols[1][1]}"/>'
                   f'</linearGradient></defs>'
                   f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="16" fill="url(#{gid})" '
                   f'stroke="{cols[0][0]}" stroke-width="1.8"/>'
                   f'<rect x="{x0 + 5}" y="{y0 + 5}" width="{x1 - x0 - 10}" height="{y1 - y0 - 10}" rx="12" '
                   f'fill="none" stroke="{cols[1][0]}" stroke-width="1.8" stroke-dasharray="7 4"/>')
        self.parts.insert(u["idx"], enc)
        if crop:
            self.crop_mark(crop, y0 - 10 if crop_y0 is None else crop_y0, y1 + 10 if crop_y1 is None else crop_y1)
        self._unit = None

    def crop_mark(self, name: str, y0: float, y1: float, x0: float = 12, x1: float = 1308):
        """Mark a page region for crop_stages.py."""
        self.a(f"<!-- crop {name} {x0:.0f} {y0:.0f} {x1:.0f} {y1:.0f} -->")

    def unit_port(self, x, y, w, n, title, sub=None, h=38):
        """An input tensor that arrives from unit ``n`` (outlined in that unit's colour)."""
        self.tensor(x, y, w, h, title, sub if sub is not None else f"From Unit {n}", col=UNIT[n - 1][0])

    def unit_card(self, y0, y1, n, title, body_lines, x=1320, w=380, size=11.5, tsize=18):
        """A card beside unit ``n`` spanning its height, numbered in the unit's colour."""
        cols = self._unit_cols(n)
        ns = n if isinstance(n, tuple) else (n,)
        h = y1 - y0
        self.a(f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" rx="14" fill="{CARD_F}" stroke="{CARD_S}" '
               f'stroke-width="1.5"/>')
        if len(cols) == 1:
            self.a(f'<rect x="{x}" y="{y0 + 14}" width="5" height="{h - 28}" rx="2.5" fill="{cols[0][0]}"/>')
        else:
            self.a(f'<rect x="{x}" y="{y0 + 14}" width="5" height="{h / 2 - 14}" rx="2.5" fill="{cols[0][0]}"/>')
            self.a(f'<rect x="{x}" y="{y0 + h / 2}" width="5" height="{h / 2 - 14}" rx="2.5" fill="{cols[1][0]}"/>')
        cx, cy = x + 34, y0 + h / 2 - 8 * len(body_lines) + 4
        if len(cols) == 1:
            self.a(f'<circle cx="{cx}" cy="{cy}" r="14" fill="{cols[0][0]}"/>')
        else:
            self.a(f'<path d="M{cx} {cy - 14} A14 14 0 0 0 {cx} {cy + 14} Z" fill="{cols[0][0]}"/>')
            self.a(f'<path d="M{cx} {cy - 14} A14 14 0 0 1 {cx} {cy + 14} Z" fill="{cols[1][0]}"/>')
        self.text(cx, cy + 4.5, "+".join(str(k) for k in ns), size=12 if len(ns) == 1 else 10, fill=BG,
                  anchor="middle", weight="700")
        self.text(x + 62, cy + 6, title, size=tsize, fill=TXT, weight="700")
        self.wrap(x + 62, cy + 30, body_lines, size=size)

    def unit_overview(self, y, modules: Sequence[Sequence[str]], left_key: str = "frame_1",
                      right_key: str = "preds_1", right_caption: str = "Output: Predicates At The Unseen Frame",
                      overlap: Optional[Tuple[str, str]] = None, h: float = 150, x0: float = 30,
                      x1: float = 1290) -> float:
        """The one-line architecture summary under the title: video -> the four units -> scene graph.
        ``modules[k]`` lists the modules of unit k+1 in this method; ``overlap`` = (title, subtitle)
        of a module drawn straddling units 1 and 2.  Returns the y below the strip."""
        wl, _ = self.fit(left_key, h=h - 22, default=(64, h - 22), max_w=110)
        self.image_slot(x0, y + 4, wl, h - 22, left_key, border=FRAME_B)
        self.text(x0 + wl / 2, y + h + 2, "Video", size=10, fill=MUTED, anchor="middle")
        wr, _ = self.fit(right_key, h=h - 22, default=(260, h - 22), max_w=260)
        xr = x1 - wr
        self.image_slot(xr, y + 4, wr, h - 22, right_key)
        self.text(xr + wr / 2, y + h + 2, right_caption, size=10, fill=MUTED, anchor="middle")
        aw = 34
        xa, xb = x0 + wl + aw, xr - aw
        n = 4
        bw = (xb - xa - (n - 1) * aw) / n
        yc = y + h / 2

        def block_arrow(xs):
            x2 = xs + aw - 5
            self.a(f'<path d="M{xs + 5} {yc - 7} H{x2 - 10} V{yc - 14} L{x2} {yc} L{x2 - 10} {yc + 14} '
                   f'V{yc + 7} H{xs + 5} Z" fill="{GHOST_F}" stroke="{LINE}" stroke-width="1.3"/>')
        block_arrow(x0 + wl)
        block_arrow(xb)
        names = [("Observed Objects", "Processing Unit"), ("Unobserved Objects", "Processing Unit"),
                 ("Relationship", "Processing Unit"), ("Decoders",)]
        for k in range(n):
            bx = xa + k * (bw + aw)
            col, fill = UNIT[k]
            self.a(f'<rect x="{bx}" y="{y}" width="{bw}" height="{h}" rx="12" fill="{fill}" stroke="{col}" '
                   f'stroke-width="2"/>')
            self.a(f'<circle cx="{bx + 20}" cy="{y + 21}" r="11" fill="{col}"/>')
            self.text(bx + 20, y + 25.5, str(k + 1), size=11.5, fill=BG, anchor="middle", weight="700")
            for i, ln in enumerate(names[k]):
                self.text(bx + 36, y + 26 + i * 16, ln, size=12.5, fill=col, weight="700", font=SERIF)
            yy = y + 26 + len(names[k]) * 16 + 8
            for ln in modules[k]:
                # a line starting with two spaces continues the bullet above it
                if ln.startswith("  "):
                    self.text(bx + 23, yy, ln.strip(), size=9.5, fill=TXT)
                else:
                    self.text(bx + 14, yy, "• " + ln, size=9.5, fill=TXT)
                yy += 13.5
            if k < n - 1:
                block_arrow(bx + bw)
        if overlap:
            cw = bw * 0.95
            cx = xa + bw + aw / 2 - cw / 2
            cy = y + h - 26
            gid = f"ovl{len(self.parts)}"
            self.a(f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0">'
                   f'<stop offset="0" stop-color="{UNIT[0][1]}"/><stop offset="1" stop-color="{UNIT[1][1]}"/>'
                   f'</linearGradient></defs>')
            self.a(f'<rect x="{cx}" y="{cy}" width="{cw}" height="42" rx="10" fill="url(#{gid})" '
                   f'stroke="{UNIT[0][0]}" stroke-width="1.8"/>')
            self.a(f'<rect x="{cx + 3}" y="{cy + 3}" width="{cw - 6}" height="36" rx="8" fill="none" '
                   f'stroke="{UNIT[1][0]}" stroke-width="1.5" stroke-dasharray="6 3"/>')
            self.text(cx + cw / 2, cy + 19, overlap[0], size=11, fill=TXT, anchor="middle", weight="700")
            self.text(cx + cw / 2, cy + 32, overlap[1], size=9, fill=MUTED, anchor="middle")
            return y + h + 36
        return y + h + 20


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
