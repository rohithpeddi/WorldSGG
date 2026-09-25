"""Export the architecture figures into the paper as vector PDFs.

    python scripts/paper_figures/dark/export_paper.py [--paper <updated_submission dir>] [--only name ...] [--png]

Every figure has one canonical source SVG (the video whose real intermediates
it carries) and one target name under ``<paper>/sup_images/architectures/``.
The SVGs are converted with ``svg2pdf.ps1`` (headless Edge print engine: text
stays text, the embedded panels stay raster).  ``--png`` also copies the
rasterised PNG next to the PDF for quick viewing.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_arch_out = REPO / "assets/figures/architecture/paper_figures/dark"
OUT = _arch_out if _arch_out.exists() else (REPO / "outputs/paper_figures/dark")
PAPER = Path(r"C:\Users\rohit\LaTeXProjects\WSGG_Paper\updated_submission")

# target name -> source SVG (relative to OUT)
FIGURES = {
    # training-based methods: adapted baselines and the detector, real panels of 12XD3
    "w_sttran": "12XD3/w_sttran.svg",
    "w_sttran_pp": "12XD3/w_sttran_pp.svg",
    "w_dsgdetr": "12XD3/w_dsgdetr.svg",
    "w_dsgdetr_pp": "12XD3/w_dsgdetr_pp.svg",
    "w_usg": "12XD3/w_usg.svg",
    "mono3d": "12XD3/mono3d.svg",
    "worldwise": "12XD3/worldwise.svg",
    "worldwise_plus": "12XD3/worldwise_plus.svg",
    "worldwise_pp": "12XD3/worldwise_pp.svg",
    # MLLM methods, real panels of 00T1E
    "zero_shot": "00T1E/zero_shot.svg",
    "caption_all": "00T1E/caption_all.svg",
    "graph_rag": "00T1E/graph_rag.svg",
    "track_a": "00T1E/track_a.svg",
    "track_b": "00T1E/track_b.svg",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(PAPER))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--stages", action="store_true",
                    help="also cut every figure into its three stage bands (crop_stages.py) and export "
                         "<name>_s1.pdf .. <name>_s3.pdf, the per-page figures the paper uses")
    ap.add_argument("--stages-only", action="store_true", help="export only the stage crops")
    args = ap.parse_args()
    target = Path(args.paper) / "sup_images/architectures"
    target.mkdir(parents=True, exist_ok=True)
    names = args.only or list(FIGURES)
    missing = [n for n in names if not (OUT / FIGURES[n]).exists()]
    if missing:
        print("missing source SVGs (skipped):", ", ".join(f"{n} <- {FIGURES[n]}" for n in missing))
    todo = [n for n in names if n not in missing]
    if not todo:
        return
    svgs = []
    if not args.stages_only:
        # svg2pdf names the PDF after the SVG stem, which already equals the target name
        svgs += [str(OUT / FIGURES[n]) for n in todo]
    if args.stages or args.stages_only:
        from crop_stages import crop
        stage_dir = OUT / "stages"
        stage_dir.mkdir(exist_ok=True)
        for n in todo:
            svgs += [str(p) for p in crop(OUT / FIGURES[n], stage_dir)]
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "svg2pdf.ps1"),
                    "-OutDir", str(target)] + svgs, check=True)
    for svg in svgs:
        pdf = target / (Path(svg).stem + ".pdf")
        print("ok" if pdf.exists() else "MISSING", pdf)
    if args.png:
        for n in todo:
            png = (OUT / FIGURES[n]).with_suffix(".png")
            if png.exists():
                shutil.copy(png, target / f"{n}.png")


if __name__ == "__main__":
    main()
