"""INFER: markdown tables from the rebuttal inference score JSONs (CPU, read-only).

    $PY tools/rebuttal/infer_tables.py --root /data3/rohith/ag/runs/rebuttal/infer [--md out.md]

Sections: I1+I2 SGDet with vs without GT visibility flags (covered and full GT), I1+I2 3-D matching, I4 ablations.
Values are percent with one decimal; per-frame R / mR (mR over 26 predicates); WC = With Constraint sorted by
confidence unless the table says construction order.
"""
import argparse
import json
import os

ROWS = ("w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "w_usg", "worldwise_dinov3l",
        "worldwise_plus_dinov3tok", "worldwise_pp_dinov3")
NAMES = {"w_sttran": "W-STTran", "w_sttran_pp": "W-STTran++", "w_dsgdetr": "W-DSGDetr", "w_dsgdetr_pp": "W-DSGDetr++",
         "w_usg": "W-USG", "worldwise_dinov3l": "WorldWise", "worldwise_plus_dinov3tok": "WorldWise+",
         "worldwise_pp_dinov3": "WorldWise++"}
ABL = ("notau", "nomask", "noema", "noego", "nomotion", "notempedge")
REB_RESCORE = "/data3/rohith/ag/runs/rebuttal/rescore"


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def cell(summ, c, b, m, k):
    try:
        return summ[c][b][m][str(k)]
    except (KeyError, TypeError):
        return None


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def block(rows, k, wc_key, nc_key):
    """rows: list of (label, json). Columns: All/OO/UO x WC R, WC mR, NC R, NC mR at K."""
    out = [f"| method | " + " | ".join(f"{b} WC R | {b} WC mR | {b} NC R | {b} NC mR" for b in ("All", "OO", "UO")) + " |",
           "|---|" + "---|" * 12]
    for label, d in rows:
        if d is None:
            out.append(f"| {label} | " + " | ".join(["(missing)"] + [""] * 11) + " |")
            continue
        cells = []
        for b in ("All", "OO", "UO"):
            w, n = d.get(wc_key), d.get(nc_key)
            cells += [pct(cell(w, "wc", b, "R", k)), pct(cell(w, "wc", b, "mR", k)),
                      pct(cell(n, "nc", b, "R", k)), pct(cell(n, "nc", b, "mR", k))]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data3/rohith/ag/runs/rebuttal/infer")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    S = os.path.join(a.root, "scores")
    L = []
    # ---------------- I1 + I2 ----------------
    L.append("## I1+I2: SGDet with GT visibility flags (published) vs without (WSGG_VISIBILITY_FROM_ANNOTATION=0)\n")
    L.append("best_model.pth of each row; 1,511 test videos, 48,834 frames; WC sorted by confidence.\n")
    gates = []
    for gt, wc_key, nc_key in (("covered", "sgdet|covered|sorted", "sgdet|covered|construction"),
                               ("full inventory (full_rule)", "sgdet|full_rule|sorted", "sgdet|full_rule|construction")):
        for k in (20, 10):
            rows = []
            for r in ROWS:
                for v, lab in (("visgt", "GT vis"), ("visoff", "no GT vis")):
                    rows.append((f"{NAMES[r]} ({lab})", load(f"{S}/{r}_sgdet_{v}.json")))
            L.append(f"\n### {gt} GT, @{k}\n")
            L.append(block(rows, k, wc_key, nc_key))
    for r in ROWS:
        d = load(f"{S}/{r}_sgdet_visgt.json")
        e = load(f"{S}/{r}_sgdet_visoff.json")
        gates.append(f"| {NAMES[r]} | {d.get('gate_max_abs_diff_vs_common_gt') if d else '-'} | "
                     f"{d['meta'].get('corner_fallback') if d else '-'} | "
                     f"{d['input_visibility'] if d else '-'} | {e['input_visibility'] if e else '-'} |")
    L.append("\nGate and input stats (gate = max |diff| of covered/construction cells vs common_gt/results/<row>.json):\n")
    L.append("| method | gate | corner fallback | input visibility (GT vis) | input visibility (no GT vis) |\n|---|---|---|---|---|")
    L += gates
    # ---------------- 3-D ----------------
    T = os.path.join(a.root, "score3d")
    L.append("\n## I1+I2: SGDet with 3-D matching (best_model.pth, GT visibility flags; covered GT unless noted)\n")
    L.append("Rules: 2d = common-GT rule (class + 2-D IoU>=0.5 visible, class only unobserved); iou3d@t = class + "
             "oriented 3-D IoU >= t for every object; centre@d = class + centre distance <= d x GT box diagonal. "
             "Coordinates are reconstruction units; every unobserved test box is a frozen copy of the nearest seen box "
             "(AUDIT DATA-10).\n")
    rules = ("2d", "iou3d@0.10", "iou3d@0.25", "iou3d@0.50", "centre@0.5", "centre@1.0")
    for inv in ("covered", "full"):
        for met in (("nc", "R"), ("nc", "mR"), ("wc", "R"), ("wc", "mR")):
            L.append(f"\n### {inv} GT, {met[0].upper()} {met[1]}@20 (All / OO / UO)\n")
            L.append("| method | box | " + " | ".join(rules) + " |\n|---|---|" + "---|" * len(rules))
            for r in ROWS:
                d = load(f"{T}/{r}_sgdet_visgt.json")
                if d is None:
                    L.append(f"| {NAMES[r]} | | (missing) |")
                    continue
                for src in d["box_sources"]:
                    cells = []
                    for rule in rules:
                        key = (f"2d|{'covered' if inv == 'covered' else 'full_rule'}" if rule == "2d"
                               else f"{src}|{inv}|{rule}")
                        s = d.get(key)
                        cells.append("/".join(pct(cell(s, met[0], b, met[1], 20)) for b in ("All", "OO", "UO")))
                    L.append(f"| {NAMES[r]} | {src} | " + " | ".join(cells) + " |")
    L.append("\n### 3-D gate and histograms (pairs matched by the 2-D rule; IoU3D vs the GT box)\n")
    L.append("| method | gate pass | visible-pair IoU3D median / mean (input) | input: visible med / >=0.25 | "
             "input: unobserved med / >=0.25 | refined: visible med / >=0.25 | refined: unobserved med / >=0.25 | "
             "fallback slots |\n|---|---|---|---|---|---|---|---|")
    for r in ROWS:
        d = load(f"{T}/{r}_sgdet_visgt.json")
        if d is None:
            continue
        g, h = d["gate"], d["hist"]

        def hm(k):
            x = h.get(k)
            return "-" if not x or not x.get("n") else f"{x['median']:.3f} / {100 * x['frac_ge_0.25']:.1f}% (n={x['n']})"
        L.append(f"| {NAMES[r]} | {g['pass']} | {g['input']['median']:.3f} / {g['input']['mean']:.3f} | "
                 f"{hm('input|visible')} | {hm('input|unobserved')} | {hm('refined|visible')} | "
                 f"{hm('refined|unobserved')} | {(d.get('corner_fallback') or {}).get('object_fallback_slots')} |")
    # ---------------- I4 ----------------
    L.append("\n## I4: WorldWise ablations on the reporting protocol (best_model.pth, 1,511 videos)\n")
    ref_p = load(f"{REB_RESCORE}/worldwise_dinov3l.json")
    ref_s = load(f"{S}/worldwise_dinov3l_sgdet_visgt.json")
    for mode, ref in (("predcls", ref_p), ("sgdet", ref_s)):
        for gt in ("covered", "full"):
            g2 = gt if mode == "predcls" else ("covered" if gt == "covered" else "full_rule")
            wc_key, nc_key = f"{mode}|{g2}|sorted", f"{mode}|{g2}|construction"
            rows = [("WorldWise (full model)", ref)] + [(f"- {x}", load(f"{S}/abl_{x}_{mode}.json")) for x in ABL]
            L.append(f"\n### {mode}, {gt} GT, @20\n")
            L.append(block(rows, 20, wc_key, nc_key))
    txt = "\n".join(L) + "\n"
    if a.md:
        with open(a.md, "w") as f:
            f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
