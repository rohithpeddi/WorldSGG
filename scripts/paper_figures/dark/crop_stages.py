"""Cut a finished method figure into one SVG per page for the paper.

    python scripts/paper_figures/dark/crop_stages.py <figure.svg> [...] [--out-dir DIR]

A whole figure is 1720 px wide and holds its bands plus a card column; at text
width its labels are unreadable in print, so the paper places one band per page.

Processing-unit figures (the WorldWise family and the adapted baselines, drawn
with ``Canvas.begin_unit``) carry explicit markers ``<!-- crop NAME x0 y0 x1 y1 -->``:
``overview`` (title + the four-unit strip), ``u1`` .. ``u4`` and, for WorldWise++,
``u12`` (the entity decoder shared by units 1 and 2).  Each marker becomes
``<name>_<NAME>.svg``; the card column is left out, its text is in the paper, and
``u4`` keeps the legend.

Stage figures (the MLLM methods and the detector, drawn with ``Canvas.stage``)
have no markers; for those the script finds the four band rules (figure title,
Stage 1, Stage 2, Stage 3: a line from x = 30 to x = 1290, stroke-width 1.5) and
writes ``<name>_s1.svg`` .. ``<name>_s3.svg`` (x 20..1300; Stage 1 keeps the
title, Stage 3 the legend).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

RULE_RE = re.compile(r'<line x1="30" y1="([0-9.]+)" x2="1290" y2="[0-9.]+" stroke="#[0-9a-fA-F]{6}" stroke-width="1.5"/>')
SIZE_RE = re.compile(r'viewBox="0 0 (\d+) (\d+)" width="(\d+)" height="(\d+)"')
MARK_RE = re.compile(r"<!-- crop (\w+) (-?[0-9.]+) (-?[0-9.]+) (-?[0-9.]+) (-?[0-9.]+) -->")
X0, X1 = 20, 1300


def _write(s: str, root: str, x0, y0, x1, y1, out: Path) -> Path:
    w, h = x1 - x0, y1 - y0
    t = s.replace(root, f'viewBox="{x0:.0f} {y0:.0f} {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}"', 1)
    out.write_text(t, encoding="utf-8")
    return out


def crop(svg: Path, out_dir: Path) -> list[Path]:
    s = svg.read_text(encoding="utf-8")
    m = SIZE_RE.search(s)
    if not m:
        raise SystemExit(f"{svg}: no viewBox/width/height on the root element")
    W, H = int(m.group(1)), int(m.group(2))
    marks = MARK_RE.findall(s)
    if marks:
        return [_write(s, m.group(0), float(a), max(0.0, float(b)), float(cx), min(float(H), float(d)),
                       out_dir / f"{svg.stem}_{name}.svg") for name, a, b, cx, d in marks]
    ys = sorted({float(v) for v in RULE_RE.findall(s)})
    if len(ys) != 4:
        raise SystemExit(f"{svg}: no crop markers and expected 4 band rules (title + 3 stages), found {ys}")
    _, s1, s2, s3 = ys
    regions = [(20, s2 - 16), (s2 - 22, s3 - 16), (s3 - 22, H - 4)]
    return [_write(s, m.group(0), X0, y0, X1, y1, out_dir / f"{svg.stem}_s{k}.svg")
            for k, (y0, y1) in enumerate(regions, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svgs", nargs="+")
    ap.add_argument("--out-dir", default=None, help="default: next to each SVG")
    args = ap.parse_args()
    for f in args.svgs:
        svg = Path(f)
        outs = crop(svg, Path(args.out_dir) if args.out_dir else svg.parent)
        print(svg.name, "->", ", ".join(p.name for p in outs))


if __name__ == "__main__":
    main()
