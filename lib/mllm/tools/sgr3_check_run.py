#!/usr/bin/env python3
"""Sanity check of a finished rag_sgr3 / rag_all run before scoring.

    python -m lib.mllm.tools.sgr3_check_run <run_dir> [--video_list <txt>]

Runners swallow per-video exceptions and exit 0, so check: pkl count vs the
list, the fraction of (frame, object) predictions whose response parsed into at
least one label, and predicted vs GT (frame, object) pairs.
"""
import argparse
import json
import os
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--video_list", default="/data3/rohith/ag/splits/test_worldbbox_thinking150.txt")
    a = ap.parse_args()
    want = [Path(l.strip()).stem for l in open(a.video_list) if l.strip()]
    have = {Path(f).stem.replace(".mp4", "") for f in os.listdir(a.run_dir) if f.endswith(".pkl")}
    missing = [v for v in want if v not in have]
    n_pred = n_parsed = n_raw_none = gt_pairs = 0
    for v in want:
        p = os.path.join(a.run_dir, f"{v}.mp4.pkl")
        if not os.path.exists(p):
            p = os.path.join(a.run_dir, f"{v}.pkl")
        if not os.path.exists(p):
            continue
        d = pickle.load(open(p, "rb"))
        for fr in d["frames"].values():
            for pr in fr["predictions"]:
                n_pred += 1
                if pr.get("raw_response") is None:
                    n_raw_none += 1
                if pr.get("attention") or pr.get("contacting") or pr.get("spatial"):
                    n_parsed += 1
            gt_pairs += len(fr["objects"])
    out = {"run_dir": a.run_dir, "pkls": len(have & set(want)), "expected": len(want),
           "missing": missing[:20], "pred_pairs": n_pred, "expected_pairs(frames x video objects)": gt_pairs,
           "raw_none": n_raw_none, "parse_rate": round(n_parsed / max(n_pred, 1), 4)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
