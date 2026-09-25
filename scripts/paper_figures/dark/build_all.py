"""Build the method figures for every video under assets/figures/architecture/intermediates/.

    python scripts/paper_figures/dark/build_all.py [--root <intermediates>] [--out <paper_figures/dark>] [--videos 00T1E ...]
                                                   [--theme light|dark] [--caps title|upper|none] [--png] [--mllm]

Each video gets its own sub-directory of ``--out``; the ``--top-level`` video's (00T1E)
figures are also written at the top level of ``--out``.  ``--png`` rasterises every SVG with
``rasterize.ps1`` (headless Edge, Windows); ``--mllm`` also rebuilds the five
MLLM architecture figures (schematic slots) at the top level.  A video directory
holding ``zero_shot/``, ``caption_all/``, ``graph_rag/``, ``track_a/`` or
``track_b/`` panels (``mllm_panels.py``) also gets those figures; one with no
``worldwise*`` panels gets no WorldWise figures.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MLLM = ("zero_shot", "caption_all", "graph_rag", "track_a", "track_b")     # fig_<name>.py <-> <video>/<name>/
BASELINES = ("w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "w_usg")   # fig_<name>.py <-> <video>/<name>/


def main():
    _arch = REPO / "assets/figures/architecture"
    _def_root = (_arch / "intermediates") if (_arch / "intermediates").exists() else (REPO / "outputs/intermediates")
    _def_out = (_arch / "paper_figures/dark") if (_arch / "paper_figures/dark").exists() else (REPO / "outputs/paper_figures/dark")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_def_root))
    ap.add_argument("--out", default=str(_def_out))
    ap.add_argument("--top-level", default="00T1E", help="video whose figures are also written at the top level")
    ap.add_argument("--videos", nargs="*", default=None, help="only these video directories (default: all)")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"])
    ap.add_argument("--png", action="store_true", help="rasterise the SVGs with rasterize.ps1 (headless Edge)")
    ap.add_argument("--mllm", action="store_true", help="also rebuild the five schematic MLLM figures at the top level")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    style = ["--theme", args.theme, "--caps", args.caps]
    svgs = []
    sfx = "" if args.theme == "light" else "_dark"          # mllm_panels.py writes <figure>_dark/ for the dark palette
    for vdir in sorted(p for p in root.iterdir() if p.is_dir() and (args.videos is None or p.name in args.videos)):
        v = vdir.name
        targets = [out / v] + ([out] if v == args.top_level else [])
        worldwise = any((vdir / d).is_dir() for d in ("worldwise", "worldwise_plus", "worldwise_pp"))
        mllm = {name: next((vdir / d for d in (name + sfx, name) if (vdir / d).is_dir()), None) for name in MLLM}
        for tdir in targets:
            tdir.mkdir(parents=True, exist_ok=True)
            cmds = [
                [sys.executable, str(HERE / "fig_worldwise.py"), "--images", str(vdir / "worldwise"), "--video", v,
                 "--out", str(tdir / "worldwise.svg")],
                [sys.executable, str(HERE / "fig_worldwise_plus.py"), "--images", str(vdir / "worldwise_plus"),
                 "--grid-images", str(vdir / "worldwise_pp"), "--video", v, "--out", str(tdir / "worldwise_plus.svg")],
                [sys.executable, str(HERE / "fig_worldwise_pp.py"), "--images", str(vdir / "worldwise_pp"), "--video", v,
                 "--out", str(tdir / "worldwise_pp.svg")],
            ] if worldwise else []                          # MLLM-only videos (e.g. 00T1E) get no schematic WorldWise set
            for name in BASELINES:                          # adapted baselines, drawn as processing units
                if (vdir / name).is_dir():
                    cmds.append([sys.executable, str(HERE / f"fig_{name}.py"), "--images", str(vdir / name),
                                 "--video", v, "--out", str(tdir / f"{name}.svg")])
            for name, idir in mllm.items():                 # real MLLM panels from mllm_panels.py
                if idir is not None:
                    cmds.append([sys.executable, str(HERE / f"fig_{name}.py"), "--images", str(idir), "--video", v,
                                 "--out", str(tdir / f"{name}.svg")])
            for cmd in cmds:
                subprocess.run(cmd + style, check=True)
                svgs.append(cmd[cmd.index("--out") + 1])
    if args.mllm:
        for name in MLLM:
            target = out / f"{name}.svg"
            subprocess.run([sys.executable, str(HERE / f"fig_{name}.py"), "--out", str(target)] + style, check=True)
            svgs.append(str(target))
    if args.png and svgs:
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "rasterize.ps1")] + svgs,
                       check=True)


if __name__ == "__main__":
    main()
