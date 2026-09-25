#!/usr/bin/env python3
"""How a retrieved context shifts the predicted label distribution.

    python -m lib.mllm.tools.sgr3_pred_dist <run_dir> [<run_dir> ...]

Per run: mean number of labels emitted per (frame, object) for each predicate
category, and the share of the top labels per category. If a context acts as
a frequency prior, head labels gain share and mR falls while R rises.
"""
import collections
import json
import os
import pickle
import sys


def dist(run_dir):
    cnt = {c: collections.Counter() for c in ("attention", "contacting", "spatial")}
    n = 0
    for f in os.listdir(run_dir):
        if not f.endswith(".pkl"):
            continue
        d = pickle.load(open(os.path.join(run_dir, f), "rb"))
        for fr in d["frames"].values():
            for p in fr["predictions"]:
                n += 1
                for c in cnt:
                    for lab, s in (p.get(c) or {}).items():
                        if s and s > 0:
                            cnt[c][lab] += 1
    out = {"run": os.path.basename(run_dir.rstrip("/")), "pairs": n}
    for c, ct in cnt.items():
        tot = sum(ct.values())
        out[c] = {"labels_per_pair": round(tot / max(n, 1), 3),
                  "distinct_labels_used": len(ct),
                  "top": [(k, round(v / max(tot, 1), 3)) for k, v in ct.most_common(5)]}
    return out


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(dist(r)))
