"""Print, per candidate video, the per-object visibility pattern of the predcls
test item (v = visible, u = valid but unseen, . = absent), so the qualitative
intermediates can be drawn from a video whose permanence reasoning is actually
exercised.  Server-side helper for scripts/paper_figures/dump_intermediates.py."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from wsgg_base import load_wsgg_config                      # noqa: E402
from tools.reeval_test import make_test_dataset             # noqa: E402

videos = sys.argv[1:] or ["HI75B", "12XD3", "AQQQ5", "UG4M2", "R4SJJ", "GFK4S", "QI0EL", "IDXZK"]
conf = load_wsgg_config(str(REPO / "configs/methods/predcls/worldwise_plus_dinov3tok_predcls.yaml"))
ds = make_test_dataset(conf)
classes = list(ds.object_classes)
for v in videos:
    keys = [k for k in ds.video_list if k.replace(".mp4", "") == v]
    if not keys:
        print(v, "NOT IN SPLIT")
        continue
    it = ds[ds.video_list.index(keys[0])]
    valid = it["valid_mask"].numpy()
    vis = it["visibility_mask"].numpy()
    cls = it["object_classes"].numpy()
    T, N = valid.shape
    print(f"{v}: T={T} unseen_cells={int((valid & ~vis).sum())}")
    for n in range(N):
        if not valid[:, n].any():
            continue
        t0 = np.where(valid[:, n])[0][0]
        pat = "".join("v" if (valid[t, n] and vis[t, n]) else ("u" if valid[t, n] else ".") for t in range(T))
        print(f"    {classes[cls[t0, n]]:<16s} {pat}")
