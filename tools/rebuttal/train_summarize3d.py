"""TRAIN: summarise the 3-D IoU re-scores of the rebuttal-round SGDet runs (train/score3d/<exp>__<ckpt>.json).

Per run and checkpoint: the frame-convention gate (median 3-D IoU of 2-D-matched visible pairs), a consistency gate
against the run's 2-D score (train/scores/<exp>__<ckpt>.json: the 3-D scorer's "2d" column must reproduce
covered/full_rule NC and sorted With-Constraint R@K within 1e-3, since both dumps come from the same checkpoint),
and recall under each matching rule (2-D rule, IoU3D >= 0.10 / 0.25 / 0.50, centre <= 0.5 / 1.0 x GT diagonal),
for the input boxes and, for WorldWise++, its refined boxes. Writes markdown + a JSON of the table cells.

    python tools/rebuttal/train_summarize3d.py [--root /data3/rohith/ag/runs/rebuttal/train] [--out_md f] [--out_json f]
"""
import argparse
import glob
import json
import os

ORDER = ["reb_w_sttran_sgdet_s0", "reb_w_sttran_pp_sgdet_s0", "reb_w_dsgdetr_sgdet_s0", "reb_w_dsgdetr_pp_sgdet_s0",
         "reb_w_dsgdetr_pp_sgdet_s1", "reb_w_dsgdetr_pp_sgdet_s2", "reb_w_usg_sgdet_s0", "reb_worldwise_sgdet_s0",
         "reb_worldwise_plus_sgdet_s0", "reb_worldwise_pp_sgdet_s0", "reb_worldwise_pp_sgdet_s1",
         "reb_worldwise_pp_sgdet_s2", "reb_w_dsgdetr_pp_la_sgdet_s0", "reb_worldwise_hold_sgdet_s0",
         "reb_worldwise_masktok_sgdet_s0", "reb_worldwise_pp_r50det_sgdet_s0"]
NAMES = {"reb_w_sttran_sgdet_s0": "W-STTran", "reb_w_sttran_pp_sgdet_s0": "W-STTran++",
         "reb_w_dsgdetr_sgdet_s0": "W-DSGDetr", "reb_w_dsgdetr_pp_sgdet_s0": "W-DSGDetr++",
         "reb_w_dsgdetr_pp_sgdet_s1": "W-DSGDetr++ (seed 1)", "reb_w_dsgdetr_pp_sgdet_s2": "W-DSGDetr++ (seed 2)",
         "reb_w_usg_sgdet_s0": "W-USG", "reb_worldwise_sgdet_s0": "WorldWise", "reb_worldwise_plus_sgdet_s0":
         "WorldWise+", "reb_worldwise_pp_sgdet_s0": "WorldWise++", "reb_worldwise_pp_sgdet_s1": "WorldWise++ (seed 1)",
         "reb_worldwise_pp_sgdet_s2": "WorldWise++ (seed 2)", "reb_w_dsgdetr_pp_la_sgdet_s0":
         "W-DSGDetr++ + logit adj.", "reb_worldwise_hold_sgdet_s0": "WorldWise hold-last-visible",
         "reb_worldwise_masktok_sgdet_s0": "WorldWise mask token",
         "reb_worldwise_pp_r50det_sgdet_s0": "WorldWise++ (ResNet-50 detector)"}
RULES = ["2d", "iou3d@0.10", "iou3d@0.25", "iou3d@0.50", "centre@0.5", "centre@1.0"]
RULE_HDR = {"2d": "2-D rule", "iou3d@0.10": "IoU3D>=0.10", "iou3d@0.25": "IoU3D>=0.25", "iou3d@0.50": "IoU3D>=0.50",
            "centre@0.5": "centre<=0.5 diag", "centre@1.0": "centre<=1.0 diag"}
TOL = 1e-3


def cell(res, src, inv, rule):
    key = (f"2d|{'covered' if inv == 'covered' else 'full_rule'}" if rule == "2d" else f"{src}|{inv}|{rule}")
    return res.get(key)


def fmt(s, kind, k="20"):
    """'All/OO/UO' percentages of R or mR at K for NC or (sorted) WC."""
    if s is None:
        return "-"
    return "/".join(f"{100 * s[kind][b]['R'][k]:.1f}" for b in ("All", "OO", "UO"))


def fmt_m(s, kind, k="20"):
    if s is None:
        return "-"
    return "/".join(f"{100 * s[kind][b]['mR'][k]:.1f}" for b in ("All", "OO", "UO"))


def consistency(res3, res2):
    worst = 0.0
    pairs = [("2d|covered", "nc", "sgdet|covered|construction", "nc"), ("2d|covered", "wc", "sgdet|covered|sorted", "wc"),
             ("2d|full_rule", "nc", "sgdet|full_rule|construction", "nc"),
             ("2d|full_rule", "wc", "sgdet|full_rule|sorted", "wc")]
    for k3, t3, k2, t2 in pairs:
        for b in ("All", "OO", "UO"):
            for m in ("R", "mR"):
                for k in ("10", "20", "50"):
                    worst = max(worst, abs(res3[k3][t3][b][m][k] - res2[k2][t2][b][m][k]))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data3/rohith/ag/runs/rebuttal/train")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    lines, cells = [], {}
    found = {os.path.basename(f)[:-5] for f in glob.glob(f"{a.root}/score3d/*.json")}
    lines.append("# 3-D IoU re-score of the retrained SGDet runs\n")
    lines.append("Per-frame No-Constraint recall @20 (All/OO/UO, %). Matching: a predicted slot counts as a GT object "
                 "of its class under each rule; no class-only fallback in the 3-D rules. Caveats: coordinates are "
                 "reconstruction units, not metres; every unobserved test box is a frozen copy of the nearest seen box "
                 "(AUDIT DATA-10).\n")
    gate_rows = ["| run | ckpt | epoch | frame gate: median IoU3D (n) | consistency with 2-D score (max abs diff) |",
                 "|---|---|---|---|---|"]
    tabs = {}
    for ck in ("best", "checkpoint_19"):
        for inv in ("covered", "full"):
            for kind, f in (("NC R@20", fmt), ("NC mR@20", fmt_m)):
                tabs[(ck, inv, kind)] = ["| run | box | " + " | ".join(RULE_HDR[r] for r in RULES) + " |",
                                         "|---|---|" + "---|" * len(RULES)]
    for exp in ORDER:
        for ck in ("best", "checkpoint_19"):
            name = f"{exp}__{ck}"
            if name not in found:
                continue
            res3 = json.load(open(f"{a.root}/score3d/{name}.json"))
            p2 = f"{a.root}/scores/{name}.json"
            cons = consistency(res3, json.load(open(p2))) if os.path.exists(p2) else None
            g = res3["gate"]["input"]
            ep = (res3.get("meta") or {}).get("ckpt_epoch")
            gate_rows.append(f"| {NAMES[exp]} | {ck} | {ep} | {'PASS' if res3['gate']['pass'] else 'FAIL'} "
                             f"{g.get('median', 0):.3f} (n={g.get('n', 0)}) | "
                             + ("-" if cons is None else f"{'ok' if cons <= TOL else 'MISMATCH'} {cons:.1e}") + " |")
            cells[name] = {"gate": res3["gate"], "consistency_max_abs_diff": cons, "epoch": ep}
            for src in res3["box_sources"]:
                for inv in ("covered", "full"):
                    for kind, f in (("NC R@20", fmt), ("NC mR@20", fmt_m)):
                        row = [f(cell(res3, src, inv, r), "nc") if (r != "2d" or src == "input") else "same"
                               for r in RULES]
                        tabs[(ck, inv, kind)].append(f"| {NAMES[exp]} | {src} | " + " | ".join(row) + " |")
                        cells[name][f"{src}|{inv}|{kind}"] = row
    lines.append("## Gates\n")
    lines += gate_rows + [""]
    for (ck, inv, kind), rows in tabs.items():
        if len(rows) > 2:
            lines.append(f"## {kind}, {inv} inventory, {'val-selected' if ck == 'best' else 'final (epoch 20)'} "
                         f"checkpoint\n")
            lines += rows + [""]
    md = "\n".join(lines)
    print(md)
    if a.out_md:
        open(a.out_md, "w").write(md)
    if a.out_json:
        json.dump(cells, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
