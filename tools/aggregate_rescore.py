"""
Aggregate the worldbbox re-scoring outputs into one reference table.

Inputs (produced by scripts/remote/rescore_worldbbox.sh and
scripts/remote/score_worldwise_variants.sh), one or more roots:
  <root>/reeval/reeval_<stem>.json            (tools/reeval_test.py, --frames both)
  <root>/bucketed_breakdown*.json             (tools/bucketed_breakdown.py over the dumps)

Output: a markdown file with, per (mode, method, backbone/variant):
  all-frame  wc R@10/20/50, mR@10/20/50, nc R@20, mR@20,
  last-frame wc R@20 / mR@20 (for continuity with the training-time logs),
  and the visibility buckets (nc, K=20): OO R, OU R, OU-nontrivial R / mR.

Stems are accepted in both layouts:
  <method>_<mode>_<backbone>        e.g. worldwise_predcls_dinov3l        (reference ladder)
  <method>_<variant>_<mode>         e.g. worldwise_plus_dinov3tok_predcls (lineage variants;
                                    the historical worldformer_c1_* stems map to worldwise_plus)

    python tools/aggregate_rescore.py \
        --root /data3/rohith/ag/runs/rescore /data3/rohith/ag/runs/worldformer/score \
               /data3/rohith/ag/runs/worldwise_pp/score \
        --out analysis/worldbbox_lineage_<date>.md
"""
import argparse
import glob
import json
import os
import re

METHOD_ORDER = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "w_usg",
                "worldwise", "worldwise_plus", "worldwise_pp"]
BACKBONE_ORDER = ["resnet50", "dinov2b", "dinov2l", "dinov3l",
                  "dinov3tok", "pi3tok", "fused", "dinov3", "dinov3_nodet"]
METHOD_ALIASES = {"worldformer_c1": "worldwise_plus", "worldformer_c2": "worldwise_pp"}
VARIANT_METHODS = ("worldformer_c1", "worldformer_c2", "worldwise_plus", "worldwise_pp")


def _f(x, scale=100.0):
    return "n/a" if x is None else f"{scale * float(x):.1f}"


def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def parse_stem(stem):
    """'worldwise_predcls_dinov3l' -> ('worldwise', 'predcls', 'dinov3l');
    'worldwise_plus_dinov3tok_predcls' / 'worldformer_c1_dinov3tok_predcls'
        -> ('worldwise_plus', 'predcls', 'dinov3tok');
    'worldwise_pp_dinov3_nodet_sgdet' -> ('worldwise_pp', 'sgdet', 'dinov3_nodet')."""
    m = re.match(r"^(.*)_(predcls|sgdet)_(resnet50|dinov2b|dinov2l|dinov3l)$", stem)
    if m:
        return METHOD_ALIASES.get(m.group(1), m.group(1)), m.group(2), m.group(3)
    for meth in sorted(VARIANT_METHODS, key=len, reverse=True):
        m = re.match(rf"^{re.escape(meth)}_(.+)_(predcls|sgdet)$", stem)
        if m:
            return METHOD_ALIASES.get(meth, meth), m.group(2), m.group(1)
    return stem, "?", "?"


def load_reevals(roots):
    rows = {}
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "reeval", "reeval_*.json"))):
            stem = os.path.basename(p)[len("reeval_"):-len(".json")]
            with open(p, "r", encoding="utf-8") as f:
                rows[stem] = json.load(f)          # later roots override earlier ones
    return rows


def load_buckets(roots):
    out = {}
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "bucketed_breakdown*.json"))):
            with open(p, "r", encoding="utf-8") as f:
                items = json.load(f)
            for it in items:
                exp = _get(it, "meta", "experiment") or ""
                out[exp] = it
    return out


def bucket_cols(bres, k):
    if not bres:
        return ["n/a"] * 4
    c = _get(bres, "constraints", "nc") or {}
    b = c.get("buckets", {})
    nt = _get(c, "ou_nontrivial", "drop_trivial") or {}
    kk = str(k)
    return [
        _f(_get(b, "OO", "R", kk) if _get(b, "OO", "R", kk) is not None else _get(b, "OO", "R", k)),
        _f(_get(b, "OU", "R", kk) if _get(b, "OU", "R", kk) is not None else _get(b, "OU", "R", k)),
        _f(_get(nt, "R", kk) if _get(nt, "R", kk) is not None else _get(nt, "R", k)),
        _f(_get(nt, "mR", kk) if _get(nt, "mR", kk) is not None else _get(nt, "mR", k)),
    ]


def sk(d, k):
    """stats_block dicts may be keyed by int or str after JSON round-trip."""
    if d is None:
        return None
    return d.get(str(k), d.get(k))


def _order(s):
    method, _, backbone = parse_stem(s)
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else 99,
            BACKBONE_ORDER.index(backbone) if backbone in BACKBONE_ORDER else 99, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", default=["/data3/rohith/ag/runs/rescore"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    reevals = load_reevals(args.root)
    buckets = load_buckets(args.root)
    if not reevals:
        raise SystemExit(f"no reeval JSONs under {[os.path.join(r, 'reeval') for r in args.root]}")

    lines = []
    lines.append("# WorldBBox test set (1,511 videos) — reference table\n")
    lines.append("Best-epoch checkpoints re-scored on `world4d_rel_annotations_worldbbox/test`, "
                 "all frames (`tools/reeval_test.py --frames both`), visibility buckets from "
                 "`tools/bucketed_breakdown.py` (no-constraint). Values in %. "
                 "`worldwise_plus` = WorldWise+ (frozen-latent ROI tokens; backbone column = token stream), "
                 "`worldwise_pp` = WorldWise++ (entity decoder over the DINOv3/π³ grids; "
                 "`dinov3_nodet` = without the detection objective).\n")
    for mode in ("predcls", "sgdet"):
        lines.append(f"\n## {mode}\n")
        hdr = ("| method | backbone | wc R@10 | wc R@20 | wc R@50 | wc mR@10 | wc mR@20 | wc mR@50 "
               "| nc R@20 | nc mR@20 | last wc R@20 | last wc mR@20 "
               f"| OO R@{args.k} | OU R@{args.k} | OU-nt R@{args.k} | OU-nt mR@{args.k} |")
        lines.append(hdr)
        lines.append("|" + "---|" * (hdr.count("|") - 1))
        stems = sorted([s for s in reevals if parse_stem(s)[1] == mode], key=_order)
        for s in stems:
            r = reevals[s]
            method, _, backbone = parse_stem(s)
            al = _get(r, "schemes", "all") or {}
            la = _get(r, "schemes", "last") or {}
            wc, nc, lwc = al.get("wc", {}), al.get("nc", {}), la.get("wc", {})
            row = [method, backbone,
                   _f(sk(wc.get("R"), 10)), _f(sk(wc.get("R"), 20)), _f(sk(wc.get("R"), 50)),
                   _f(sk(wc.get("mR"), 10)), _f(sk(wc.get("mR"), 20)), _f(sk(wc.get("mR"), 50)),
                   _f(sk(nc.get("R"), 20)), _f(sk(nc.get("mR"), 20)),
                   _f(sk(lwc.get("R"), 20)), _f(sk(lwc.get("mR"), 20))]
            row += bucket_cols(buckets.get(r.get("experiment", ""), {}), args.k)
            lines.append("| " + " | ".join(row) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
