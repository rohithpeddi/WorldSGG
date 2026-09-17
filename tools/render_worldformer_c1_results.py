#!/usr/bin/env python3
"""
Render the WorldFormer C1 results table (markdown) from reeval JSONs + the bucketed
breakdown, for results/worldformer_c1_<date>.md.

    python tools/render_worldformer_c1_results.py \
        --score_dir /data3/rohith/ag/runs/worldformer/score \
        --ref_dir /data3/rohith/ag/runs/rescore/reeval \
        --constraint nc --k 20
"""
import argparse
import json
import os

CELLS = [("WorldWise v2e @ dinov3l (ref)", "worldwise_{mode}_dinov3l", "ref"),
         ("C1 dinov3_tok", "worldformer_c1_dinov3tok_{mode}", "score"),
         ("C1 pi3_tok", "worldformer_c1_pi3tok_{mode}", "score"),
         ("C1 fused", "worldformer_c1_fused_{mode}", "score")]


def _f(x):
    return "" if x is None else f"{100 * float(x):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_dir", default="/data3/rohith/ag/runs/worldformer/score")
    ap.add_argument("--ref_dir", default="/data3/rohith/ag/runs/rescore/reeval")
    ap.add_argument("--constraint", default="nc", choices=["nc", "wc"])
    ap.add_argument("--k", default="20")
    ap.add_argument("--frames", default="all", choices=["all", "last"])
    args = ap.parse_args()

    bucket_path = os.path.join(args.score_dir, "bucketed_breakdown_c1.json")
    buckets = {}
    if os.path.exists(bucket_path):
        for r in json.load(open(bucket_path)):
            buckets[r["meta"].get("experiment", r["meta"].get("method"))] = r["constraints"][args.constraint]

    k = args.k
    print(f"Constraint: {args.constraint}, frames: {args.frames}, K={k} (recall in %)\n")
    print("| mode | cell | R@10 | R@20 | R@50 | mR@10 | mR@20 | mR@50 | OO R@K | OU R@K | OU-nt R@K | OU-nt mR@K |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mode in ("predcls", "sgdet"):
        for label, stem, src in CELLS:
            exp = stem.format(mode=mode)
            p = os.path.join(args.ref_dir if src == "ref" else os.path.join(args.score_dir, "reeval"),
                             f"reeval_{exp}.json")
            row = [mode, label]
            if os.path.exists(p):
                s = json.load(open(p))["schemes"][args.frames][args.constraint]
                row += [_f(s["R"].get(kk)) for kk in ("10", "20", "50")]
                row += [_f(s["mR"].get(kk)) for kk in ("10", "20", "50")]
            else:
                row += [""] * 6
            b = buckets.get(exp)
            if b:
                bb = b["buckets"]
                nt = b.get("ou_nontrivial", {}).get("drop_trivial", {"R": {}, "mR": {}})
                row += [_f(bb.get("OO", {}).get("R", {}).get(int(k), bb.get("OO", {}).get("R", {}).get(k))),
                        _f(bb.get("OU", {}).get("R", {}).get(int(k), bb.get("OU", {}).get("R", {}).get(k))),
                        _f(nt["R"].get(int(k), nt["R"].get(k))), _f(nt["mR"].get(int(k), nt["mR"].get(k)))]
            else:
                row += [""] * 4
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
