#!/usr/bin/env python3
"""Markdown table of the SGR3-style retrieval arms (scores + context budget).

    python -m lib.mllm.tools.sgr3_table [--scores /data3/rohith/ag/runs/mllm/rag_sgr3/scores]
"""
import argparse
import json
import os

from lib.mllm.tools.sgr3_ctx_stats import stats

ARMS = [  # (label, model dir, ctx dir or None)
    ("R0 none", "qwen3vl_8b_R0", "qwen3vl_8b_R0"),
    ("R1 BGE top-1 (existing, uncapped)", "qwen3vl_8b_150", "qwen3vl_8b_R1ctx"),
    ("R1@B BGE top-1, capped", "qwen3vl_8b_R1B", "qwen3vl_8b_R1B"),
    ("R2-1 SGR3-style, k=1", "qwen3vl_8b_R2k1", "qwen3vl_8b_R2k1"),
    ("R2-3 SGR3-style, k=3", "qwen3vl_8b_R2k3", "qwen3vl_8b_R2k3"),
    ("R2-5 SGR3-style, k=5", "qwen3vl_8b_R2k5", "qwen3vl_8b_R2k5"),
    ("R3 R1(B/2) + R2-1", "qwen3vl_8b_R3", "qwen3vl_8b_R3"),
    ("R2-rand (control: random train video)", "qwen3vl_8b_R2rand", "qwen3vl_8b_R2rand"),
]


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data3/rohith/ag/runs/mllm/rag_sgr3")
    a = ap.parse_args()
    rows = ["| arm | videos | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU nc mR@50 | "
            "OU-nt nc R@50 | OU-nt nc mR@50 | ctx tok mean | median | p95 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, md, cd in ARMS:
        p = os.path.join(a.root, "scores", f"{md}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        w = d.get("wsgg", {})
        c = os.path.join(a.root, "ctx", cd)
        cs = stats(c)["ctx_tokens"] if os.path.isdir(c) and os.listdir(c) else {}
        rows.append("| {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            label, d.get("n_videos"), d.get("n_split"),
            pct(g(w, "wc", "R", "20")), pct(g(w, "wc", "mR", "20")),
            pct(g(w, "nc", "R", "50")), pct(g(w, "nc", "mR", "50")),
            pct(g(w, "nc", "buckets", "buckets", "OO", "mR", "50")),
            pct(g(w, "nc", "buckets", "buckets", "OU", "mR", "50")),
            pct(g(w, "nc", "buckets", "ou_nontrivial", "drop_trivial", "R", "50")),
            pct(g(w, "nc", "buckets", "ou_nontrivial", "drop_trivial", "mR", "50")),
            cs.get("mean", "–"), cs.get("median", "–"), cs.get("p95", "–"), cs.get("max", "–")))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
