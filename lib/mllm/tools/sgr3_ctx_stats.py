#!/usr/bin/env python3
"""Per-arm retrieved-context token statistics from the rag_sgr3 sidecars.

    python -m lib.mllm.tools.sgr3_ctx_stats /data3/rohith/ag/runs/mllm/rag_sgr3/ctx/qwen3vl_8b_R2k1 [...]

Statistics are per prompt (one (frame, object) query = one prompt), i.e.
weighted the way the contexts enter the model.
"""
import json
import os
import sys

import numpy as np


def stats(d):
    toks, raw, scenes, n_vid = [], [], [], 0
    per_obj_distinct = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        n_vid += 1
        e = json.load(open(os.path.join(d, f)))["entries"]
        toks += [x["ctx_tokens"] for x in e]
        raw += [x["r1_raw_tokens"] for x in e if "r1_raw_tokens" in x]
        scenes += [x["scenes_used"] for x in e if "scenes_used" in x]
        by_frame = {}
        for x in e:
            by_frame.setdefault(x["frame"], set()).add(x["context"])
        per_obj_distinct += [len(v) for v in by_frame.values()]
    t = np.array(toks)
    out = {"dir": d, "videos": n_vid, "prompts": int(len(t)),
           "ctx_tokens": {"mean": round(float(t.mean()), 1), "median": float(np.median(t)),
                          "p5": float(np.percentile(t, 5)), "p95": float(np.percentile(t, 95)),
                          "max": int(t.max()), "empty_frac": round(float((t == 0).mean()), 4)},
           "distinct_contexts_per_frame_mean": round(float(np.mean(per_obj_distinct)), 2)}
    if raw:
        r = np.array(raw)
        out["r1_raw_tokens"] = {"mean": round(float(r.mean()), 1), "median": float(np.median(r)),
                                "p95": float(np.percentile(r, 95)), "max": int(r.max()),
                                "empty_frac": round(float((r == 0).mean()), 4)}
    if scenes:
        out["scenes_used_mean"] = round(float(np.mean(scenes)), 2)
    return out


if __name__ == "__main__":
    print(json.dumps([stats(d) for d in sys.argv[1:]], indent=1))
