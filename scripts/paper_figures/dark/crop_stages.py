"""Cut a finished method figure into one SVG per stage for the paper.

    python scripts/paper_figures/dark/crop_stages.py <figure.svg> [...] [--out-dir DIR]

A whole figure is 1720 px wide and holds three stage bands plus a card column;
at text width its labels are unreadable in print.  Every figure script draws
its stage titles with ``Canvas.stage`` -> ``band_title``: a rule from x = 30 to
x = 1290 (stroke-width 1.5) under a serif title.  This script finds those four
rules (figure title, Stage 1, Stage 2, Stage 3), and writes ``<name>_s1.svg``,
``<name>_s2.svg``, ``<name>_s3.svg`` whose ``viewBox`` selects the band
(x 20..1300; the cards column is left out, its text is in the paper).  Stage 1
keeps the figure title, Stage 3 keeps the legend and the in-figure caption.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

RULE_RE = re.compile(r'<line x1="30" y1="([0-9.]+)" x2="1290" y2="[0-9.]+" stroke="#[0-9a-fA-F]{6}" stroke-width="1.5"/>')
SIZE_RE = re.compile(r'viewBox="0 0 (\d+) (\d+)" width="(\d+)" height="(\d+)"')
X0, X1 = 20, 1300


def crop(svg: Path, out_dir: Path) -> list[Path]:
    s = svg.read_text(encoding="utf-8")
    m = SIZE_RE.search(s)
    if not m:
        raise SystemExit(f"{svg}: no viewBox/width/height on the root element")
    W, H = int(m.group(1)), int(m.group(2))
    ys = sorted({float(v) for v in RULE_RE.findall(s)})
    if len(ys) != 4:
        raise SystemExit(f"{svg}: expected 4 band rules (title + 3 stages), found {ys}")
    _, s1, s2, s3 = ys
    regions = [(20, s2 - 16), (s2 - 22, s3 - 16), (s3 - 22, H - 4)]
    out = []
    for k, (y0, y1) in enumerate(regions, 1):
        h = y1 - y0
        t = s.replace(m.group(0), f'viewBox="{X0} {y0:.0f} {X1 - X0} {h:.0f}" width="{X1 - X0}" height="{h:.0f}"', 1)
        p = out_dir / f"{svg.stem}_s{k}.svg"
        p.write_text(t, encoding="utf-8")
        out.append(p)
    return out


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
