"""Per-frame detail for candidate showcase videos: slot visibility pattern and, per method, which
frames are fully correct (every pair: top-1 attention == label, top-1 spatial and contacting in labels)."""
import pickle
import sys

import numpy as np

sys.path.insert(0, "/home/rxp190007/CODE/Scene4Cast/scripts/paper_figures")
from find_showcase import DUMPS  # noqa: E402

VIDS = sys.argv[1:]
CLS = None
try:
    sys.path.insert(0, "/home/rxp190007/CODE/Scene4Cast")
    from lib.supervised.config_constants import OBJECT_CLASSES as CLS  # type: ignore  # noqa
except Exception:
    pass


def ok(r, k):
    gs = set(np.flatnonzero(r["gt_spatial"][k] > 0.5).tolist())
    gc = set(np.flatnonzero(r["gt_contacting"][k] > 0.5).tolist())
    return (int(np.argmax(r["attention_distribution"][k])) == int(r["gt_attention"][k])
            and int(np.argmax(r["spatial_distribution"][k])) in gs
            and int(np.argmax(r["contacting_distribution"][k])) in gc)


data = {}
for m, p in DUMPS.items():
    recs = pickle.load(open(p, "rb"))["records"]
    data[m] = {v: sorted([r for r in recs if r["video_id"] == v], key=lambda r: r["frame_t"]) for v in VIDS}

for v in VIDS:
    recs = data["worldwise_pp"][v]
    T = len(recs)
    N = max(len(r["valid_mask"]) for r in recs)
    print("=" * 20, v, "T", T, "classes", recs[0]["object_classes"].tolist())
    for n in range(N):
        s = ""
        for r in recs:
            if n >= len(r["valid_mask"]) or not r["valid_mask"][n]:
                s += "."
            else:
                s += "V" if r["visibility_mask"][n] else "o"
        print(" slot", n, s)
    for m in DUMPS:
        s = ""
        for r in data[m][v]:
            ks = [k for k in range(len(r["pair_valid"])) if r["pair_valid"][k]]
            good = all(ok(r, k) for k in ks)
            un = any(not r["visibility_mask"][int(r["object_idx"][k])] for k in ks)
            s += ("P" if good else "x") if un else ("+" if good else "-")
        print(f" {m:15s}", s)
