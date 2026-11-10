"""INFRA gate (rebuttal): compare a dump made on another host with the saved CS93371 dump of the same checkpoint.

Records are matched by (video_id, frame_t) over the videos present in the new dump; every key must exist in both,
float arrays must agree to --tol (max abs diff), everything else exactly. Run on CS93371 (CPU only):

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python infra_gate_compare.py --new <dump.pkl> --ref <saved dump.pkl>
"""
import argparse
import pickle
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    with open(a.new, "rb") as f:
        new = pickle.load(f)
    with open(a.ref, "rb") as f:
        ref = pickle.load(f)
    print("new meta:", new.get("meta"))
    print("ref meta:", ref.get("meta"))
    nrec, rrec = new["records"], ref["records"]
    vids = sorted({r["video_id"] for r in nrec})
    rmap = {(r["video_id"], r.get("frame_t")): r for r in rrec if r["video_id"] in set(vids)}
    nmap = {(r["video_id"], r.get("frame_t")): r for r in nrec}
    ok = True
    if set(rmap) != set(nmap):
        ok = False
        print(f"FRAME SET DIFFERS: new {len(nmap)} ref {len(rmap)}; only new {sorted(set(nmap) - set(rmap))[:5]} "
              f"only ref {sorted(set(rmap) - set(nmap))[:5]}")
    worst = {}
    only_new, only_ref = set(), set()
    for key in sorted(set(nmap) & set(rmap)):
        n, r = nmap[key], rmap[key]
        only_new |= set(n) - set(r)
        only_ref |= set(r) - set(n)
        for k in set(n) & set(r):
            x, y = n[k], r[k]
            if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
                if x.shape != y.shape:
                    ok = False; worst[k] = float("inf"); continue
                d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
                worst[k] = max(worst.get(k, 0.0), d)
            else:
                same = np.array_equal(x, y) if isinstance(x, np.ndarray) else (x == y)
                if not same:
                    ok = False; worst[k] = float("inf")
                else:
                    worst.setdefault(k, 0.0)
    if only_new:
        print(f"  fields only in the new dump (newer dump tool; not compared): {sorted(only_new)}")
    if only_ref:
        ok = False
        print(f"  FIELDS MISSING from the new dump: {sorted(only_ref)}")
    for k in sorted(worst):
        print(f"  {k:28s} max|diff| = {worst[k]:.3g}")
    mx = max(worst.values()) if worst else float("inf")
    ok = ok and mx < a.tol
    print(f"videos {vids}  frames {len(set(nmap) & set(rmap))}  max|diff| {mx:.3g}  tol {a.tol:g}  ->",
          "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
