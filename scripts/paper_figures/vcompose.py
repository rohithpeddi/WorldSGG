"""Vector composition of the supplementary pipeline figures.

The compose scripts used to paste every panel PNG into one matplotlib figure with ``imshow``, which
re-sampled all pictures to the figure dpi and rasterised the text inside the panels (blurry when
zoomed, with light fringes around dark glyphs).  Here the panels are placed as the PDFs written by
``scene_common.save`` (vector text and lines; only point clouds / photos are raster, at their native
resolution), and the frame (bands, headers, arrows, labels) stays vector:

    base.pdf   bands and headers           (matplotlib, W x H inches)
    panels     each panel PDF, trimmed / scaled into its slot
    top.pdf    arrows, nodes and free text (matplotlib, transparent, drawn over the panels)

The three layers are stacked with pdflatex (TikZ), which embeds the PDFs without re-sampling.
A PNG preview is rendered from the final PDF with pdftoppm.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def media_box(pdf: Path) -> tuple[float, float]:
    """Page size (bp) of a single-page matplotlib PDF (its MediaBox is written uncompressed)."""
    m = re.search(rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s*\]", pdf.read_bytes())
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return x1 - x0, y1 - y0


def alpha_crop(png: Path, window=(0.0, 0.0, 1.0, 1.0), pad_frac: float = 0.006, thr: int = 8):
    """Fractional (left, top, right, bottom) bounds of the non-transparent content of a panel PNG inside
    the fractional window (left, top, right, bottom)."""
    im = Image.open(png).convert("RGBA")
    w, h = im.size
    l, t, r, b = window
    x0, y0, x1, y1 = int(l * w), int(t * h), int(round(r * w)), int(round(b * h))
    a = np.asarray(im.getchannel("A"))[y0:y1, x0:x1] > thr
    if not a.any():
        return window
    ys, xs = np.nonzero(a)
    p = pad_frac * max(w, h)
    return (max(l, (x0 + xs.min() - p) / w), max(t, (y0 + ys.min() - p) / h),
            min(r, (x0 + xs.max() + 1 + p) / w), min(b, (y0 + ys.max() + 1 + p) / h))


def resolve(stem, crop=None, autocrop=False):
    """(pdf path, (l, t, r, b) crop fractions, width/height aspect of the cropped panel)."""
    stem = Path(stem)
    pdf = stem.with_suffix(".pdf")
    pw, ph = media_box(pdf)
    win = crop if crop is not None else (0.0, 0.0, 1.0, 1.0)
    if autocrop:
        win = alpha_crop(stem.with_suffix(".png"), win)
    l, t, r, b = win
    return pdf, win, ((r - l) * pw) / ((b - t) * ph)


class Canvas:
    """A W x H inch figure with a base layer, placed panel PDFs and a top layer (y measured from the bottom)."""

    def __init__(self, W: float, H: float, bg="white"):
        self.W, self.H = W, H
        self.base = plt.figure(figsize=(W, H))
        self.base.patch.set_facecolor(bg)
        self.top = plt.figure(figsize=(W, H))
        self.top.patch.set_alpha(0)
        self.items: list[tuple] = []

    # ---- coordinates
    def fx(self, x): return x / self.W
    def fy(self, y): return y / self.H

    # ---- panels
    def place(self, stem, x, ytop, w, h=None, crop=None, autocrop=False, placeholder=True):
        """Place panel ``stem`` (path without extension; needs ``stem.pdf``) with its top-left corner at
        (x, ytop) inches and width w; with h it is fitted into (w, h) and centred.  ``crop`` is a
        fractional (left, top, right, bottom) window of the panel; ``autocrop`` trims transparent margins
        (measured on ``stem.png``).  Returns (x0, y0, ww, hh) of the placed picture."""
        stem = Path(stem)
        pdf = stem.with_suffix(".pdf")
        if not pdf.exists():
            if not placeholder:
                raise FileNotFoundError(pdf)
            hh = h if h else 0.6 * w
            ax = self.base.add_axes([self.fx(x), self.fy(ytop - hh), self.fx(w), self.fy(hh)]); ax.set_axis_off()
            ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04", fc="#d1d5db",
                                        ec="#9ca3af", lw=0.8, ls="--", transform=ax.transAxes))
            ax.text(0.5, 0.5, stem.name, ha="center", va="center", fontsize=7)
            return x, ytop - hh, w, hh
        pw, ph = media_box(pdf)
        _, (l, t, r, b), img = resolve(stem, crop, autocrop)
        if h is None:
            ww, hh = w, w / img
        else:
            ww, hh = (w, w / img) if img > w / h else (h * img, h)
        x0 = x + (w - ww) / 2
        y0 = ytop - (h if h else hh) + ((h - hh) / 2 if h else 0)
        trim = (l * pw, (1 - b) * ph, (1 - r) * pw, t * ph)   # left bottom right top (bp)
        self.items.append((pdf, x0, y0, ww, hh, trim))
        return x0, y0, ww, hh

    # ---- output
    def save(self, out: Path, png_dpi: int = 250, keep_tex: bool = False):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="vcompose_"))
        try:
            self.base.savefig(tmp / "base.pdf", facecolor=self.base.get_facecolor())
            self.top.savefig(tmp / "top.pdf", transparent=True)
            plt.close(self.base); plt.close(self.top)
            W, H = self.W, self.H
            lines = [r"\documentclass[border=0pt]{standalone}", r"\usepackage{graphicx}", r"\usepackage{tikz}",
                     r"\begin{document}", r"\begin{tikzpicture}[x=1in,y=1in]",
                     rf"\useasboundingbox (0,0) rectangle ({W:.4f},{H:.4f});",
                     rf"\node[anchor=south west,inner sep=0pt] at (0,0) {{\includegraphics[width={W:.4f}in,height={H:.4f}in]{{base.pdf}}}};"]
            for i, (pdf, x0, y0, ww, hh, trim) in enumerate(self.items):
                name = f"p{i:02d}.pdf"
                shutil.copyfile(pdf, tmp / name)
                tr = " ".join(f"{v:.3f}" for v in trim)
                lines.append(rf"\node[anchor=south west,inner sep=0pt] at ({x0:.4f},{y0:.4f}) "
                             rf"{{\includegraphics[trim={tr},clip,width={ww:.4f}in,height={hh:.4f}in]{{{name}}}}};")
            lines += [rf"\node[anchor=south west,inner sep=0pt] at (0,0) {{\includegraphics[width={W:.4f}in,height={H:.4f}in]{{top.pdf}}}};",
                      r"\end{tikzpicture}", r"\end{document}"]
            (tmp / "fig.tex").write_text("\n".join(lines), encoding="utf-8")
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"], cwd=tmp,
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("pdflatex failed:\n" + r.stdout[-3000:])
            shutil.copyfile(tmp / "fig.pdf", out.with_suffix(".pdf"))
            subprocess.run(["pdftoppm", "-r", str(png_dpi), "-png", "-singlefile", str(out.with_suffix(".pdf")),
                            str(out.with_suffix(""))], check=True)
            if keep_tex:
                shutil.copyfile(tmp / "fig.tex", out.with_suffix(".tex"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print("wrote", out.with_suffix(".pdf"), f"({out.with_suffix('.pdf').stat().st_size / 1e6:.1f} MB)")
