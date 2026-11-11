"""Collect V1 (round 2) and V2 (round 3) test scores into one compact JSON for the flow artifact. CPU, read-only.

Protocol of every cell: test split, full annotated inventory (773,236 triplets), val-selected checkpoint (``best``),
per-frame recall @20 in %. With Constraint = ranked by confidence; No Constraint = all predicates. 3-D = slot -> object
matching at oriented 3-D IoU >= 0.10 (rescore_3d.py; V1 input boxes, V2 its own boxes = "refined"). Seed rows of
W-DSGDetr++ and WorldWise++ (V1 and V2) are means over the seeds present (with the sample std and n). V2 has
one row per V1 row (same seeds, logit adjustment, completion variants, ResNet-50 detector); localisation = seed 0.

    python tools/rebuttal/v2/artifact_tables.py --out results_tables.json
"""
import argparse
import json
import os
import statistics

V1 = "/data3/rohith/ag/runs/rebuttal/train"
V2 = "/data3/rohith/ag/runs/rebuttal/train_v2"
ROWS_V1 = [("w_sttran", "W-STTran"), ("w_sttran_pp", "W-STTran++"), ("w_dsgdetr", "W-DSGDetr"),
           ("w_dsgdetr_pp", "W-DSGDetr++"), ("w_usg", "W-USG"), ("worldwise", "WorldWise"),
           ("worldwise_plus", "WorldWise+"), ("worldwise_pp", "WorldWise++"),
           ("w_dsgdetr_pp_la", "W-DSGDetr++ + logit adj."), ("worldwise_hold", "WorldWise hold-last-visible"),
           ("worldwise_masktok", "WorldWise mask token"), ("worldwise_pp_r50det", "WorldWise++ (ResNet-50 det.)")]
SEEDED = ("w_dsgdetr_pp", "worldwise_pp")
# V2 mirrors V1 row for row (Nov 11): (V1 row, experiment suffix, name). V2 experiment = r3_<V2 method><suffix>_<mode>_s<seed>.
V2_METHOD = {"w_sttran": "w_sttran_3d", "w_sttran_pp": "w_sttran_pp_3d", "w_dsgdetr": "w_dsgdetr_3d",
             "w_dsgdetr_pp": "w_dsgdetr_pp_3d", "w_usg": "w_usg_3d", "worldwise": "worldwise_3d",
             "worldwise_plus": "worldwise_plus_3d", "worldwise_pp": "worldwise_pp_3d"}
ROWS_V2 = [("w_sttran", "", "W-STTran-3D"), ("w_sttran_pp", "", "W-STTran++-3D"), ("w_dsgdetr", "", "W-DSGDetr-3D"),
           ("w_dsgdetr_pp", "", "W-DSGDetr++-3D"), ("w_usg", "", "W-USG-3D"), ("worldwise", "", "WorldWise-3D"),
           ("worldwise_plus", "", "WorldWise+-3D"), ("worldwise_pp", "", "WorldWise++-3D"),
           ("w_dsgdetr_pp", "_la", "W-DSGDetr++-3D + logit adj."), ("worldwise", "_hold", "WorldWise-3D hold-last-visible"),
           ("worldwise", "_masktok", "WorldWise-3D mask token"), ("worldwise_pp", "_r50det", "WorldWise++-3D (ResNet-50 det.)")]


# One 3-D threshold everywhere (user, Nov 11: "IoU3D >= 0.10 - fix this for all the runs"): relation recall under the
# 3-D rule and hidden-object localisation both report IoU3D >= 0.10 only (plus the centre error of localisation).
KEEP_LOC = ("iou>=0.10", "centre_err_median_diag")


def keep10(d):
    return None if d is None else {k: d[k] for k in KEEP_LOC if k in d}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def cell(blk, kind, bucket, metric="R", k="20"):
    try:
        v = blk[kind][bucket][metric][k]
    except (KeyError, TypeError):
        return None
    return None if v is None else round(100 * v, 2)


def metrics(s2, s3, mode, src3):
    if s2 is None:
        return None
    wc = s2.get(f"{mode}|full|sorted" if mode == "predcls" else "sgdet|full_rule|sorted")
    nc = s2.get(f"{mode}|full|construction" if mode == "predcls" else "sgdet|full_rule|construction")
    m = {"wc_R": cell(wc, "wc", "All"), "wc_mR": cell(wc, "wc", "All", "mR"),
         "nc_R": cell(nc, "nc", "All"), "nc_mR": cell(nc, "nc", "All", "mR"),
         "uo_R": cell(nc, "nc", "UO"), "uo_mR": cell(nc, "nc", "UO", "mR")}
    if s3 is not None:
        b3 = s3.get(f"{src3}|full|iou3d@0.10")
        m.update({"d3_R": cell(b3, "nc", "All"), "d3_mR": cell(b3, "nc", "All", "mR"), "d3_uo_R": cell(b3, "nc", "UO"),
                  "d3_uo_mR": cell(b3, "nc", "UO", "mR"), "gate3d": bool(s3.get("gate", {}).get("pass"))})
    return m


def agg(ms):
    ms = [m for m in ms if m]
    if not ms:
        return None
    out = {"n_seeds": len(ms)}
    for k in ms[0]:
        vals = [m.get(k) for m in ms if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)]
        if vals:
            out[k] = round(statistics.fmean(vals), 2)
            if len(vals) > 1:
                out[k + "_sd"] = round(statistics.stdev(vals), 2)
        elif k == "gate3d":
            out[k] = all(m.get(k) for m in ms)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"protocol": "test, full inventory (773,236 triplets), val-selected checkpoint, per-frame recall @20 (%); "
                       "WC ranked by confidence; 3D = IoU3D >= 0.10", "v1": {}, "v2": {}}
    for mode in ("predcls", "sgdet"):
        rows = []
        for key, name in ROWS_V1:
            seeds = (0, 1, 2) if key in SEEDED else (0,)
            ms = []
            for s in seeds:
                exp = f"r2_{key}_{mode}_s{s}"
                ms.append(metrics(load(f"{V1}/scores/{exp}__best.json"), load(f"{V1}/score3d/{exp}__best.json"),
                                  mode, "input"))
            if key.endswith("r50det") and mode == "predcls":
                continue
            rows.append({"key": key, "name": name, **(agg(ms) or {"pending": True})})
        res["v1"][mode] = rows
        rows2 = []
        for row, suf, name in ROWS_V2:
            if suf == "_r50det" and mode == "predcls":
                continue
            key = V2_METHOD[row] + suf
            seeds = (0, 1, 2) if (row in SEEDED and not suf) else (0,)
            ms, locs = [], []
            for s in seeds:
                exp = f"r3_{key}_{mode}_s{s}"
                ms.append(metrics(load(f"{V2}/scores/{exp}__best.json"), load(f"{V2}/score3d/{exp}__best.json"),
                                  mode, "refined"))
                locs.append(load(f"{V2}/loc/{exp}__best.json"))
            out = {"key": key, "name": name, "v1_counterpart": row + suf, **(agg(ms) or {"pending": True})}
            loc = locs[0]                                       # localisation: seed 0
            if loc:
                b = loc["buckets"].get("all", {})
                out["loc"] = {"n": b.get("n"), "v2": keep10(b.get("v2")), "copy": keep10(b.get("copy")),
                              "geo_cos": b.get("geo_cos_mean"),
                              "by_gap": {g: {"n": loc["buckets"][g]["n"],
                                             "v2_iou10": loc["buckets"][g]["v2"].get("iou>=0.10"),
                                             "copy_iou10": loc["buckets"][g]["copy"].get("iou>=0.10")}
                                         for g in ("1", "2-3", "4-10", ">10", "never visible") if g in loc["buckets"]}}
            rows2.append(out)
        res["v2"][mode] = rows2
    json.dump(res, open(a.out, "w"), indent=1)
    n1 = sum(1 for mode in res["v1"] for r in res["v1"][mode] if not r.get("pending"))
    n2 = sum(1 for mode in res["v2"] for r in res["v2"][mode] if not r.get("pending"))
    print(f"[artifact_tables] V1 rows with scores: {n1}; V2 rows with scores: {n2} -> {a.out}")


if __name__ == "__main__":
    main()
