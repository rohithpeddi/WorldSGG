"""Dense per-clip pi3 world points for the main-paper dataset figure (panels a-c).

Writes <out>/dense_views.npz with, for every requested clip k: points_k (world), colors_k (uint8),
pix_k (flat pixel index into the pi3 image) for pixels that pass the pipeline's own filter
(finite, sigmoid confidence > 0.1, not a depth edge), uniformly subsampled to --n points.

    python scripts/paper_figures/extract_dense_views.py --video 0DJ6R --ks 5 11 17 \
        --out /data3/rohith/ag/runs/scene_pipeline/0DJ6R
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

AG2 = Path("/data2/rohith/ag")
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--ks", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=200000)
a = ap.parse_args()

z = np.load(AG2 / "ag4D" / "dynamic_scenes" / "pi3_dynamic" / f"{a.video}_10" / "predictions.npz", allow_pickle=True)
rng = np.random.default_rng(0)
out = {"ks": np.array(a.ks)}
for k in a.ks:
    P = z["points"][k].reshape(-1, 3)
    d = z["local_points"][k][..., 2]
    edge = ((ndimage.maximum_filter(d, 3) - ndimage.minimum_filter(d, 3)) / np.maximum(np.abs(d), 1e-6) > 0.03).reshape(-1)
    conf = z["conf"][k][..., 0].reshape(-1)
    ok = np.isfinite(P).all(1) & (conf > 0.1) & ~edge
    idx = np.nonzero(ok)[0]
    s = np.sort(rng.choice(idx, min(a.n, len(idx)), replace=False))
    out[f"points_{k}"] = P[s].astype(np.float32)
    out[f"colors_{k}"] = (z["images"][k].reshape(-1, 3)[s] * 255).clip(0, 255).astype(np.uint8)
    out[f"pix_{k}"] = s.astype(np.int32)
    print(f"k={k}: {ok.sum()} valid pixels, kept {len(s)}")
np.savez_compressed(Path(a.out) / "dense_views.npz", **out)
