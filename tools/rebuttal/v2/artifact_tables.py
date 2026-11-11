"""Collect V1 (round 2) and V2 (round 3) test scores into one compact JSON for the flow artifact. CPU, read-only.

Protocol of every cell: test split, full annotated inventory (773,236 triplets), val-selected checkpoint (``best``),
per-frame recall @20 in %. With Constraint = ranked by confidence; No Constraint = all predicates. 3-D = slot -> object
matching at oriented 3-D IoU >= 0.10 (rescore_3d.py; V1 input boxes, V2 its own boxes = "refined"). Seed rows of
W-DSGDetr++ and WorldWise++ are means over the seeds present (with the sample std and n).

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
ROWS_V2 = [("worldwise_3d", "WorldWise-3D", "worldwise"), ("worldwise_pp_3d", "WorldWise++-3D", "worldwise_pp")]


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
        for key, name, v1key in ROWS_V2:
            exp = f"r3_{key}_{mode}_s0"
            m = metrics(load(f"{V2}/scores/{exp}__best.json"), load(f"{V2}/score3d/{exp}__best.json"), mode, "refined")
            loc = load(f"{V2}/loc/{exp}__best.json")
            row = {"key": key, "name": name, "v1_counterpart": v1key, **(m or {"pending": True})}
            if loc:
                b = loc["buckets"].get("all", {})
                row["loc"] = {"n": b.get("n"), "v2": b.get("v2"), "copy": b.get("copy"), "geo_cos": b.get("geo_cos_mean"),
                              "by_gap": {g: {"n": loc["buckets"][g]["n"],
                                             "v2_iou10": loc["buckets"][g]["v2"].get("iou>=0.10"),
                                             "copy_iou10": loc["buckets"][g]["copy"].get("iou>=0.10")}
                                         for g in ("1", "2-3", "4-10", ">10", "never visible") if g in loc["buckets"]}}
            rows2.append(row)
        res["v2"][mode] = rows2
    json.dump(res, open(a.out, "w"), indent=1)
    n1 = sum(1 for mode in res["v1"] for r in res["v1"][mode] if not r.get("pending"))
    n2 = sum(1 for mode in res["v2"] for r in res["v2"][mode] if not r.get("pending"))
    print(f"[artifact_tables] V1 rows with scores: {n1}; V2 rows with scores: {n2} -> {a.out}")


if __name__ == "__main__":
    main()
