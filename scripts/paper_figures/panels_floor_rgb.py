"""RGB frames for the floor-alignment figures of the supplementary (Figs. sup:fig:wrap_floor and
sup:fig:prompthmr_0djr), drawn in the style of panels_stage1.panel_rgb_view so that each 3-D PromptHMR
panel sits next to the video frame it shows.

    python scripts/paper_figures/panels_floor_rgb.py

Frames: 0DJ6R keyframe 22 is the frame pi3 and PromptHMR saw (file 000427.png, i.e. sampled_idx[22] + 1;
see the frame bookkeeping in scene_common).  00T1E keyframe 43 is labelled "Frame 251" in b0_floor_body
(the annotated stem, sampled_idx[43]), so that frame is shown.  Exact raw frames were copied from the server
(/data/rohith/ag/frames/<video>.mp4/) into <bundle>/frames_pi3/; the bundle's frames_sampled/ is the fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_common as SC  # noqa: E402
from scene_common import COL, LABEL_COL, crisp, save_fig  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SC.enable_title_case()
ARCH = SC.REPO / "assets" / "figures" / "architecture"
SMALL = 8.5


def load_frame(video: str, idx: int) -> np.ndarray:
    root = ARCH / "scene_pipeline" / video
    for p in (root / "frames_pi3" / f"{idx:06d}.png", root / "frames_sampled" / f"{idx:06d}.jpg"):
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))
    raise FileNotFoundError(f"frame {idx} of {video}")


def rgb_panel(img: np.ndarray, title: str, edge: str, out_stem: Path, fw: float = 1.25):
    """One video frame (Lanczos x2 + unsharp for print, display only) with a coloured border and a title."""
    im = crisp(img)
    H, W = im.shape[:2]
    top = 0.22
    fig = plt.figure(figsize=(fw, fw * H / W + top)); fig.patch.set_alpha(0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1 - top / (fw * H / W + top)])
    ax.imshow(im, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(1.0)
    ax.set_title(title, fontsize=SMALL, color=COL["text"], pad=2)
    save_fig(fig, out_stem)


if __name__ == "__main__":
    # Fig. sup:fig:prompthmr_0djr (0DJ6R): the frame of the posed body, keyframe 22 -> file 000427.png
    rgb_panel(load_frame("0DJ6R", 427), "RGB Frame 427", COL["smpl"],
              ARCH / "scene_pipeline" / "panels_0DJ6R" / "s3_rgb_smpl_k22")
    # Fig. sup:fig:wrap_floor (00T1E): the frame of b0_floor_body, keyframe 43, labelled Frame 251
    rgb_panel(load_frame("00T1E", 251), "RGB Frame 251", LABEL_COL["person"],
              ARCH / "bbox_pipeline" / "panels_00T1E" / "b0_rgb_frame251")
