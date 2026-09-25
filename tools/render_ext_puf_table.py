"""Render the PUF-adaptation results table (markdown) from tools/ext_puf.py score outputs.

    python tools/render_ext_puf_table.py --score-dir /data3/rohith/ag/runs/ext/puf/full/score \
        [--ref "label|<reeval.json>|<bucketed.json>|<experiment>" ...] [--title ...]

Columns (all frames, %): wc R@20 / mR@20, wc R@50 / mR@50, nc R@50 / mR@50, and the
visibility buckets (no constraint, K=20, as in the reference tables): OO R / mR,
OU R / mR, OU-nt R / mR, plus the GT OU size as a slot-set check.
"""
import argparse
import glob
import json
import os


def _k(d, k):
    if d is None:
        return None
    return d.get(str(k), d.get(k))


def _f(x):
    return "n/a" if x is None else f"{100 * float(x):.1f}"


def load_reeval(p):
    r = json.load(open(p))
    return r["schemes"]["all"]


def load_bucket(p, exp=None):
    items = json.load(open(p))
    for it in items:
        if exp is None or it["meta"].get("experiment") == exp:
            return it
    raise KeyError(f"{exp} not in {p}")


def row(label, rv, bk):
    wc, nc = rv["wc"], rv["nc"]
    c = bk["constraints"]["nc"]
    b = c["buckets"]
    nt = c.get("ou_nontrivial", {}).get("drop_trivial", {})
    oo, ou = b.get("OO", {}), b.get("OU", {})
    cells = [label,
             _f(_k(wc["R"], 20)), _f(_k(wc["mR"], 20)), _f(_k(wc["R"], 50)), _f(_k(wc["mR"], 50)),
             _f(_k(nc["R"], 50)), _f(_k(nc["mR"], 50)),
             _f(_k(oo.get("R"), 20)), _f(_k(oo.get("mR"), 20)),
             _f(_k(ou.get("R"), 20)), _f(_k(ou.get("mR"), 20)),
             _f(_k(nt.get("R"), 20)), _f(_k(nt.get("mR"), 20)),
             f"{c['bucket_sizes'].get('OU', 0):,}"]
    return "| " + " | ".join(cells) + " |"


HDR = ("| method | wc R@20 | wc mR@20 | wc R@50 | wc mR@50 | nc R@50 | nc mR@50 | OO R@20 | OO mR@20 "
       "| OU R@20 | OU mR@20 | OU-nt R@20 | OU-nt mR@20 | GT OU |\n|" + "---|" * 14)

ORDER = ["ref_", "puf_frontend", "puf_lks_", "puf_lks_bi", "puf_fross", "puf_puf_", "puf_puf_prior_", "puf_puf_prior_vis"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    out = []
    if a.title:
        out.append(f"### {a.title}\n")
    for mode in ("predcls", "sgdet"):
        lines = []
        for spec in a.ref:
            label, rp, bp, exp = spec.split("|")
            if f"_{mode}" not in exp and mode not in label:
                continue
            lines.append(row(label, load_reeval(rp), load_bucket(bp, exp or None)))
        stems = sorted(os.path.basename(p)[len("reeval_"):-5]
                       for p in glob.glob(os.path.join(a.score_dir, "reeval", f"reeval_*_{mode}.json")))
        for s in stems:
            bp = os.path.join(a.score_dir, f"bucketed_breakdown_{s}.json")
            if not os.path.exists(bp):
                continue
            lines.append(row(s[: -len(mode) - 1], load_reeval(os.path.join(a.score_dir, "reeval", f"reeval_{s}.json")),
                             load_bucket(bp)))
        out.append(f"\n**{mode}**\n\n{HDR}\n" + "\n".join(lines) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
