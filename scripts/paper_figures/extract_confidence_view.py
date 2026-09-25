"""Export one unfiltered pi3 view for the confidence panel of the scene-pipeline figure.

The main bundle keeps only points above a display quantile, so it cannot show which pixels the
pipeline itself drops.  This writes <out>/confview.npz for a single clip k with a uniform sample of
ALL pixels: local points, colour, sigmoid confidence and the depth-edge flag used by the pipeline
(relative depth range > 3 % in a 3x3 neighbourhood; such pixels get confidence 0).

    python scripts/paper_figures/extract_confidence_view.py --video 0DJ6R --k 22 \
        --out /data3/rohith/ag/runs/scene_pipeline/0DJ6R
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

AG2 = Path("/data2/rohith/ag")

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=60000)
ap.add_argument("--rtol", type=float, default=0.03)
ap.add_argument("--pass", dest="pass_", choices=["dynamic", "static"], default="dynamic",
                help="static = the pi3 pass over inpainted frames (thresholded at tau_static = 0.1 in the paper)")
a = ap.parse_args()

src = (Path("/data/rohith/ag/ag4D/static_scenes/pi3_static") if a.pass_ == "static"
       else AG2 / "ag4D" / "dynamic_scenes" / "pi3_dynamic") / f"{a.video}_10" / "predictions.npz"
z = np.load(src, allow_pickle=True)
L = z["local_points"][a.k]                      # (H, W, 3) camera frame
conf = z["conf"][a.k][..., 0].astype(np.float32)
img = (z["images"][a.k] * 255).clip(0, 255).astype(np.uint8)
d = L[..., 2]
rng_ = ndimage.maximum_filter(d, 3) - ndimage.minimum_filter(d, 3)
edge = rng_ / np.maximum(np.abs(d), 1e-6) > a.rtol
H, W = d.shape
ok = np.isfinite(L).all(-1).reshape(-1)
idx = np.nonzero(ok)[0]
s = np.sort(np.random.default_rng(0).choice(idx, min(a.n, len(idx)), replace=False))
np.savez_compressed(Path(a.out) / "confview.npz", k=a.k, pass_=a.pass_, local=L.reshape(-1, 3)[s].astype(np.float32),
                    colors=img.reshape(-1, 3)[s], conf=conf.reshape(-1)[s], edge=edge.reshape(-1)[s], pix=s.astype(np.int32),
                    hw=np.array([H, W]))
print(f"k={a.k} conf min {conf.min():.3f} max {conf.max():.3f}; frac conf<0.1 {np.mean(conf < 0.1):.3f}; "
      f"frac edge {edge.mean():.3f}; dropped (edge or conf<0.1) {np.mean(edge | (conf < 0.1)):.3f}")
