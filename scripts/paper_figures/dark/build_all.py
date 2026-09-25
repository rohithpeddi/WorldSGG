"""Build the three dark method figures for every video under outputs/intermediates/.

    python scripts/paper_figures/dark/build_all.py [--root outputs/intermediates] [--out outputs/paper_figures/dark]

Each video gets its own sub-directory of ``--out``; the default 12XD3 figures are
also written at the top level of ``--out``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "outputs/intermediates"))
    ap.add_argument("--out", default=str(REPO / "outputs/paper_figures/dark"))
    ap.add_argument("--top-level", default="12XD3", help="video whose figures are also written at the top level")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    for vdir in sorted(p for p in root.iterdir() if p.is_dir()):
        v = vdir.name
        targets = [out / v] + ([out] if v == args.top_level else [])
        for tdir in targets:
            tdir.mkdir(parents=True, exist_ok=True)
            cmds = [
                [sys.executable, str(HERE / "fig_worldwise.py"), "--images", str(vdir / "worldwise"), "--video", v,
                 "--out", str(tdir / "worldwise.svg")],
                [sys.executable, str(HERE / "fig_worldwise_plus.py"), "--images", str(vdir / "worldwise_plus"),
                 "--grid-images", str(vdir / "worldwise_pp"), "--video", v, "--out", str(tdir / "worldwise_plus.svg")],
                [sys.executable, str(HERE / "fig_worldwise_pp.py"), "--images", str(vdir / "worldwise_pp"), "--video", v,
                 "--out", str(tdir / "worldwise_pp.svg")],
            ]
            for cmd in cmds:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
