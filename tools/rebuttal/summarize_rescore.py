"""Markdown tables from the rebuttal re-score JSONs (``rescore_rebuttal.py`` output pulled to --dir).

Prints: gates; C3 (With-Constraint construction order vs ranked by confidence); A1 (covered vs full inventory).
Values are percent, per-frame averages (R) and the paper's mR (sum over predicates / 26), as in Table 2.
"""
import argparse
import glob
import json
import os

ROWS = [("w_sttran", "W-STTran"), ("w_sttran_pp", "W-STTran++"), ("w_dsgdetr", "W-DSGDetr"),
        ("w_dsgdetr_pp", "W-DSGDetr++"), ("w_usg", "W-USG"), ("worldwise_dinov3l", "WorldWise"),
        ("worldwise_plus_dinov3tok", "WorldWise+"), ("worldwise_pp_dinov3", "WorldWise++")]
EXTRA = [("worldwise_resnet50", "WorldWise (ResNet-50)"), ("worldwise_dinov2b", "WorldWise (DINOv2-B)"),
         ("worldwise_dinov2l", "WorldWise (DINOv2-L)"), ("worldwise_plus_pi3tok", "WorldWise+ (pi3tok)"),
         ("worldwise_plus_fused", "WorldWise+ (fused)"), ("worldwise_pp_dinov3_nodet", "WorldWise++ (no det.)"),
         ("puf_lks", "PUF: LKS"), ("puf_lks_bi", "PUF: LKS bi"), ("puf_fross", "PUF: FROSS"), ("puf_puf", "PUF"),
         ("puf_puf_prior", "PUF + prior"), ("puf_puf_prior_noobs", "PUF + prior, no obs")]


def v(d, scorer, c, b, m, k):
    x = d[scorer][c][b][m][str(k)]
    return None if x is None else 100 * x


def f(x):
    return "–" if x is None else f"{x:.1f}"


def main(dirn):
    J = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(dirn, "*.json"))}
    rows = [(e, n) for e, n in ROWS + EXTRA if e in J]
    print("## Gates\n")
    print("| Experiment | covered/construction vs saved common-GT (max abs diff) | slots mapped to uncovered objects (full_rule) |")
    print("|---|---|---|")
    for e, n in rows:
        print(f"| {n} | {J[e]['gate_max_abs_diff_vs_common_gt']:.1e} | {J[e]['slot_stats'].get('slots_mapped_to_uncovered', 0):,} |")
    s = next(iter(J.values()))["gt_cache_stats"]
    print(f"\nFull ground truth: {s['full_triplets']} (zero-shot scorer: {s['expected']}); covered objects "
          f"{s['covered_objects']:,}, uncovered unobserved objects {s['uncovered_objects_unobserved']:,}, "
          f"covered relation mismatches {s.get('covered_rel_mismatch', 0)}, frames {s['frames']:,}.\n")

    for mode, cov in (("predcls", "predcls|covered"), ("sgdet", "sgdet|covered")):
        print(f"## C3 {mode}: With-Constraint, construction order -> ranked by confidence (All bucket)\n")
        print("| Method | R@10 | mR@10 | R@20 | mR@20 | UO R@20 | UO mR@20 |")
        print("|---|---|---|---|---|---|---|")
        for e, n in rows:
            d = J[e]
            cells = []
            for b, m, k in (("All", "R", 10), ("All", "mR", 10), ("All", "R", 20), ("All", "mR", 20),
                            ("UO", "R", 20), ("UO", "mR", 20)):
                a, s_ = v(d, f"{cov}|construction", "wc", b, m, k), v(d, f"{cov}|sorted", "wc", b, m, k)
                cells.append(f"{f(a)} → {f(s_)}")
            print(f"| {n} | " + " | ".join(cells) + " |")
        print()
    e = "worldwise_pp_dinov3"
    if e in J:
        d = J[e]
        print("Reconciliation with the audit probe (predicate-only key), WorldWise++ SGDet WC mR@10: construction "
              f"{f(v(d, 'sgdet|covered|construction', 'wc', 'All', 'mR', 10))}, predicate-only sort "
              f"{f(v(d, 'sgdet|covered|sorted_pred', 'wc', 'All', 'mR', 10))}, product sort "
              f"{f(v(d, 'sgdet|covered|sorted', 'wc', 'All', 'mR', 10))}.\n")

    for mode, variants in (("predcls", [("predcls|full", "full")]),
                           ("sgdet", [("sgdet|full_strict", "full, strict"), ("sgdet|full_rule", "full, rule")])):
        cov = f"{mode}|covered|construction"
        print(f"## A1 {mode}: covered GT vs full annotated inventory (No Constraint @20; WC R@20)\n")
        hdr = ["Method"]
        for b in ("All", "OO", "UO"):
            hdr += [f"{b} NC R@20", f"{b} NC mR@20"]
        hdr += ["All WC R@20"]
        print("| " + " | ".join(hdr) + " | GT |")
        print("|" + "---|" * (len(hdr) + 1))
        for e, n in rows:
            d = J[e]
            for sc, lab in [(cov, "covered")] + [(f"{p}|construction", l) for p, l in variants]:
                cells = []
                for b in ("All", "OO", "UO"):
                    cells += [f(v(d, sc, "nc", b, "R", 20)), f(v(d, sc, "nc", b, "mR", 20))]
                cells.append(f(v(d, sc, "wc", "All", "R", 20)))
                print(f"| {n} | " + " | ".join(cells) + f" | {lab} |")
        print()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\Users\rohit\PycharmProjects\WorldSGG\docs\rebuttal_rescore\supervised")
    main(ap.parse_args().dir)
