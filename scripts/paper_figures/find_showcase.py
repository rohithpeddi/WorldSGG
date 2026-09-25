"""Find test videos where WorldWise++ (PredCls) predicts every person-object pair correctly.

A pair-frame is correct when the attention argmax equals the label and the top-1 spatial and
contacting predicates are among the labelled ones (what the figures' predicate panels show; the
exact-set variant leaves no usable video).  Reads the all-frame PredCls dumps of
tools/dump_predictions.py on CS93371:

    python scripts/paper_figures/find_showcase.py rows.json        # ranking of perfect-WW++ videos
    python scripts/paper_figures/video_detail.py TM0BV HYOQB       # per-frame pattern per method
Per video we report: frames, slots, unseen pair-frames, WW++ accuracy overall / on unseen pairs,
and the same for the other methods, so a video can be picked where WW++ is perfect and the
others are not.
"""
import json
import pickle
import sys
from collections import defaultdict

import numpy as np

D = "/data3/rohith/ag/runs"
DUMPS = {
    "worldwise_pp": f"{D}/worldwise_pp/score/dumps/worldwise_pp_dinov3_predcls__all.pkl",
    "worldwise_plus": f"{D}/worldformer/score/dumps/worldformer_c1_dinov3tok_predcls__all.pkl",
    "worldwise": f"{D}/rescore/dumps/worldwise_predcls_dinov3l__all.pkl",
    "w_sttran": f"{D}/rescore/dumps/w_sttran_predcls_resnet50__all.pkl",
    "w_sttran_pp": f"{D}/rescore/dumps/w_sttran_pp_predcls_resnet50__all.pkl",
    "w_dsgdetr": f"{D}/rescore/dumps/w_dsgdetr_predcls_resnet50__all.pkl",
    "w_dsgdetr_pp": f"{D}/rescore/dumps/w_dsgdetr_pp_predcls_resnet50__all.pkl",
    "w_usg": f"{D}/rescore/dumps/w_usg_predcls_resnet50__all.pkl",
}


def topk_set(p, k):
    return set(np.argsort(-p)[:k].tolist())


def per_video(path):
    recs = pickle.load(open(path, "rb"))["records"]
    out = defaultdict(lambda: {"T": 0, "n": 0, "ok": 0, "nu": 0, "oku": 0, "slots": 0, "unseen_objs": set(),
                               "frames": {}})
    for r in recs:
        v = out[r["video_id"]]
        v["T"] += 1
        v["slots"] = max(v["slots"], int(r["valid_mask"].sum()))
        vis = r["visibility_mask"]
        fr_ok = True
        for k in range(len(r["pair_valid"])):
            if not r["pair_valid"][k]:
                continue
            o = int(r["object_idx"][k])
            unseen = not bool(vis[o])
            ga = int(r["gt_attention"][k])
            gs = set(np.flatnonzero(r["gt_spatial"][k] > 0.5).tolist())
            gc = set(np.flatnonzero(r["gt_contacting"][k] > 0.5).tolist())
            ok = (int(np.argmax(r["attention_distribution"][k])) == ga
                  and int(np.argmax(r["spatial_distribution"][k])) in gs
                  and int(np.argmax(r["contacting_distribution"][k])) in gc)
            v["n"] += 1
            v["ok"] += ok
            fr_ok &= ok
            if unseen:
                v["nu"] += 1
                v["oku"] += ok
                v["unseen_objs"].add(o)
        v["frames"][int(r["frame_t"])] = fr_ok
    return out


def main():
    res = {m: per_video(p) for m, p in DUMPS.items()}
    pp = res["worldwise_pp"]
    rows = []
    for vid, v in pp.items():
        if v["nu"] == 0 or v["n"] == 0:
            continue
        row = {"video": vid, "T": v["T"], "slots": v["slots"], "pairs": v["n"], "unseen_pairs": v["nu"],
               "n_unseen_objs": len(v["unseen_objs"]), "pp_acc": v["ok"] / v["n"], "pp_unseen_acc": v["oku"] / v["nu"],
               "pp_frames_perfect": sum(v["frames"].values())}
        for m, r in res.items():
            if m == "worldwise_pp":
                continue
            w = r.get(vid)
            row[m] = round(w["ok"] / w["n"], 3) if w and w["n"] else None
            row[m + "_u"] = round(w["oku"] / w["nu"], 3) if w and w["nu"] else None
        rows.append(row)
    json.dump(rows, open(sys.argv[1], "w"), indent=0)
    perfect = [r for r in rows if r["pp_acc"] == 1.0]
    print("videos with unseen pairs:", len(rows), " WW++ perfect on every pair-frame:", len(perfect))
    perfect.sort(key=lambda r: (-r["unseen_pairs"], r["T"]))
    for r in perfect[:40]:
        others = [r[m] for m in DUMPS if m != "worldwise_pp" and r.get(m) is not None]
        print(r["video"], "T", r["T"], "slots", r["slots"], "pairs", r["pairs"], "unseen", r["unseen_pairs"],
              "uobjs", r["n_unseen_objs"], "| WW", r["worldwise"], r["worldwise_u"], "WW+", r["worldwise_plus"],
              r["worldwise_plus_u"], "| best base", max(r[m] for m in ["w_sttran", "w_sttran_pp", "w_dsgdetr",
                                                                        "w_dsgdetr_pp", "w_usg"]),
              "min other", min(others))


if __name__ == "__main__":
    main()
