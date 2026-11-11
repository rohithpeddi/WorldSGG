"""Round 2 (full inventory): collect every score of the r2_* runs into one JSON + a markdown overview. CPU only.

Inputs (CS93371): train/scores/<exp>__<ckpt>.json (2-D rules, train_rescore.py) and, for SGDet,
train/score3d/<exp>__<ckpt>.json (3-D rules, rescore_3d.py); train/manifest_r2.json for run metadata.

Primary protocol (main paper): full annotated inventory (773,236 triplets; 147 phantom triplets dropped),
val-selected checkpoint (``best``), With Constraint ranked by confidence (``sorted``), No Constraint, per-frame
R@K / mR@K (mR over the 26 predicates). SGDet 2-D rule: a slot matches a visible object of its class at 2-D IoU >= 0.5,
else an unobserved object of its class; 3-D rule: a slot matches an object of its class at oriented 3-D IoU >= 0.10
(input boxes). Seeds: W-DSGDetr++ and WorldWise++ have seeds 0-2 (mean and sample std over seeds).

    python tools/rebuttal/inv_summarize.py [--root /data3/rohith/ag/runs/rebuttal/train] --out_json f --out_md f
"""
import argparse
import glob
import json
import os
import statistics

ROWS = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "w_usg", "worldwise", "worldwise_plus", "worldwise_pp"]
NAMES = {"w_sttran": "W-STTran", "w_sttran_pp": "W-STTran++", "w_dsgdetr": "W-DSGDetr", "w_dsgdetr_pp": "W-DSGDetr++",
         "w_usg": "W-USG", "worldwise": "WorldWise", "worldwise_plus": "WorldWise+", "worldwise_pp": "WorldWise++",
         "w_dsgdetr_pp_la": "W-DSGDetr++ + logit adj.", "worldwise_hold": "WorldWise hold-last-visible",
         "worldwise_masktok": "WorldWise mask token", "worldwise_pp_r50det": "WorldWise++ (ResNet-50 detector)"}
KS = ("10", "20", "50")
BUCKETS = ("All", "OO", "UO")


def exp_name(row, mode, seed=0):
    return f"r2_{row}_{mode}_s{seed}"


def cells(block, kind):
    """{bucket: {R@K, mR@K}} (fractions) of one scorer summary block (wc or nc)."""
    if block is None or kind not in block:
        return None
    out = {}
    for b in BUCKETS:
        d = block[kind].get(b)
        if d is None:
            continue
        out[b] = {f"{m}@{k}": d[m][k] for m in ("R", "mR") for k in KS if d[m].get(k) is not None}
    return out


def views(s2, s3, mode):
    """Every reported view of one (run, checkpoint): {view name: {wc|nc: cells}}."""
    v = {}
    if s2 is not None:
        if mode == "predcls":
            v["2d|full"] = {"wc": cells(s2.get("predcls|full|sorted"), "wc"),
                            "wc_construction": cells(s2.get("predcls|full|construction"), "wc"),
                            "nc": cells(s2.get("predcls|full|construction"), "nc")}
            v["2d|covered_incl_phantom"] = {"wc": cells(s2.get("predcls|covered|sorted"), "wc"),
                                            "nc": cells(s2.get("predcls|covered|construction"), "nc")}
        else:
            v["2d|full"] = {"wc": cells(s2.get("sgdet|full_rule|sorted"), "wc"),
                            "wc_construction": cells(s2.get("sgdet|full_rule|construction"), "wc"),
                            "nc": cells(s2.get("sgdet|full_rule|construction"), "nc")}
            if "sgdet|heldonly|sorted" in s2:
                v["2d|heldonly"] = {"wc": cells(s2.get("sgdet|heldonly|sorted"), "wc"),
                                    "nc": cells(s2.get("sgdet|heldonly|construction"), "nc")}
    if s3 is not None:
        for src in s3.get("box_sources", ["input"]):
            for r in ("iou3d@0.10", "iou3d@0.25", "iou3d@0.50", "centre@0.5", "centre@1.0"):
                blk = s3.get(f"{src}|full|{r}")
                if blk is not None:
                    v[f"3d|{src}|{r}"] = {"wc": cells(blk, "wc"), "nc": cells(blk, "nc")}
        v["_3d_gate"] = {"pass": s3["gate"]["pass"], "median": s3["gate"]["input"].get("median"),
                         "n": s3["gate"]["input"].get("n")}
        # consistency: the 3-D scorer's 2-D full column must equal train_rescore's full_rule column
        if s2 is not None and "2d|full_rule" in s3:
            diff = 0.0
            for kind, name in (("nc", "sgdet|full_rule|construction"), ("wc", "sgdet|full_rule|sorted")):
                for b in BUCKETS:
                    for m in ("R", "mR"):
                        for k in KS:
                            x = s3["2d|full_rule"][kind][b][m][k]
                            y = s2[name][kind][b][m][k]
                            if x is not None and y is not None:
                                diff = max(diff, abs(x - y))
            v["_consistency_2d_vs_3d_scorer"] = diff
    return v


def mean_std(vals):
    vals = [x for x in vals if x is not None]
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def aggregate(per_seed):
    """Mean/std over seeds, cell by cell, of view dicts."""
    out = {}
    first = per_seed[0]
    for view, kinds in first.items():
        if view.startswith("_"):
            continue
        out[view] = {}
        for kind, bk in kinds.items():
            if bk is None:
                continue
            out[view][kind] = {}
            for b, cs in bk.items():
                out[view][kind][b] = {c: mean_std([ps.get(view, {}).get(kind, {}).get(b, {}).get(c) for ps in per_seed
                                                   if ps.get(view, {}).get(kind)])
                                      for c in cs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data3/rohith/ag/runs/rebuttal/train")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    a = ap.parse_args()
    man = json.load(open(f"{a.root}/manifest_r2.json"))
    res = {"runs": {}, "seed_aggregates": {}, "missing": []}
    for exp, info in sorted(man.items()):
        for ck in ("best", "checkpoint_19"):
            p2, p3 = f"{a.root}/scores/{exp}__{ck}.json", f"{a.root}/score3d/{exp}__{ck}.json"
            s2 = json.load(open(p2)) if os.path.exists(p2) else None
            s3 = json.load(open(p3)) if os.path.exists(p3) else None
            if s2 is None or (info["mode"] == "sgdet" and s3 is None):
                res["missing"].append(f"{exp}__{ck}" + (" (2d)" if s2 is None else " (3d)"))
            if s2 is None and s3 is None:
                continue
            res["runs"][f"{exp}__{ck}"] = {"exp": exp, "ckpt": ck, **info,
                                           "epoch": ((s2 or {}).get("dump_meta") or {}).get("ckpt_epochs", {}),
                                           "views": views(s2, s3, info["mode"])}
    for row in ("w_dsgdetr_pp", "worldwise_pp"):
        for mode in ("predcls", "sgdet"):
            for ck in ("best", "checkpoint_19"):
                ps = [res["runs"].get(f"{exp_name(row, mode, s)}__{ck}") for s in (0, 1, 2)]
                ps = [p["views"] for p in ps if p is not None]
                if len(ps) >= 2:
                    res["seed_aggregates"][f"{row}_{mode}__{ck}"] = {"n_seeds": len(ps), "views": aggregate(ps)}
    # markdown overview: main-paper view (best, full, WC sorted + NC; SGDet 2-D and 3-D IoU >= 0.10)
    lines = ["# Round 2 (full inventory) results", "",
             f"Missing: {', '.join(res['missing']) or 'none'}", ""]
    for mode in ("predcls", "sgdet"):
        lines += [f"## {mode}, val-selected checkpoint, full inventory (R@20 / mR@20, %)", "",
                  "| run | WC All | NC All | NC OO | NC UO |" + (" 3D>=0.10 NC All | 3D>=0.10 NC UO | heldonly NC UO |"
                                                              if mode == "sgdet" else ""),
                  "|---|---|---|---|---|" + ("---|---|---|" if mode == "sgdet" else "")]
        for key, r in sorted(res["runs"].items()):
            if r["mode"] != mode or r["ckpt"] != "best":
                continue
            v = r["views"]

            def f(view, kind, b):
                c = (v.get(view) or {}).get(kind) or {}
                c = c.get(b)
                return "-" if not c else f"{100 * c['R@20']:.1f} / {100 * c['mR@20']:.1f}"
            row = f"| {r['exp']} | {f('2d|full', 'wc', 'All')} | {f('2d|full', 'nc', 'All')} | {f('2d|full', 'nc', 'OO')} | {f('2d|full', 'nc', 'UO')} |"
            if mode == "sgdet":
                row += f" {f('3d|input|iou3d@0.10', 'nc', 'All')} | {f('3d|input|iou3d@0.10', 'nc', 'UO')} | {f('2d|heldonly', 'nc', 'UO')} |"
            lines.append(row)
        lines.append("")
    gates = {k: (r["views"].get("_3d_gate"), r["views"].get("_consistency_2d_vs_3d_scorer"))
             for k, r in res["runs"].items() if r["mode"] == "sgdet"}
    lines += ["## SGDet gates (3-D frame gate, 2-D scorer vs 3-D scorer consistency)", ""]
    lines += [f"- {k}: gate {g[0]} consistency {g[1]}" for k, g in sorted(gates.items())]
    md = "\n".join(lines)
    print(md)
    if a.out_md:
        open(a.out_md, "w").write(md)
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
