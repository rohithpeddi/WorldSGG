"""Carve a validation subset out of the AG4D training split, and describe the held-out test videos (CPU, read-only).

Nothing here trains or evaluates a model. It writes two video lists and a statistics file:

  <out>/val750.txt        750 training videos, the lowest sha1(video_id) among the eligible ones
  <out>/train_minus_val.txt  the remaining eligible training videos
  <out>/val_split_stats.json   val vs remaining train: frames, objects, triplets, unobserved share, predicate shares,
                               label sources of unobserved objects; plus the AG test videos outside the 1,511 split

Eligible training video: an annotation PKL in ``world4d_rel_annotations/train`` with >= 2 annotated frames, and a
feature PKL in every ``features/roi_features/<mode>/<feature>/train`` folder listed in FEATURES (so every method can
use the same list). The hash rule is fixed and independent of the labels, so the subset can be re-derived.

Usage (server):  ~/anaconda3/envs/scene4cast/bin/python make_val_split.py --out /data3/rohith/ag/runs/rebuttal/splits
"""
import argparse
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

AG = "/data/rohith/ag"
TRAIN_ANN = f"{AG}/world4d_rel_annotations/train"
FEATURES = [("predcls", "resnet50"), ("sgdet", "resnet50"), ("predcls", "dinov3l"), ("sgdet", "dinov3l")]
TEST_SETS = {"worldbbox_test": f"{AG}/world4d_rel_annotations_worldbbox/test",
             "old_test_1526": f"{AG}/world4d_rel_annotations/test",
             "old_test_bak_1735": f"{AG}/world4d_rel_annotations/test_bak"}
ATT = ["looking_at", "not_looking_at", "unsure"]
SPA = ["above", "beneath", "in_front_of", "behind", "on_the_side_of", "in"]
CON = ["carrying", "covered_by", "drinking_from", "eating", "have_it_on_the_back", "holding", "leaning_on",
       "lying_on", "not_contacting", "other_relationship", "sitting_on", "standing_on", "touching", "twisting",
       "wearing", "wiping", "writing_on"]
PRED = ATT + SPA + CON
N_VAL = 750


def vid_of(fname):
    return fname.replace(".pkl", "").replace(".mp4", "")


def video_stats(path):
    with open(path, "rb") as f:
        ann = pickle.load(f)
    fr = ann.get("frames", {}) or {}
    s = {"frames": len(fr), "objects": 0, "objects_unobserved": 0, "triplets": 0, "triplets_unobserved": 0,
         "pred": Counter(), "pred_unobserved": Counter(), "src_unobserved": Counter(), "src_visible": Counter()}
    for f_ in fr.values():
        for o in f_.get("object_info_list", []) or []:
            vis = bool(o.get("visible", True))
            s["objects"] += 1
            s["objects_unobserved"] += int(not vis)
            s["src_visible" if vis else "src_unobserved"][o.get("source", "gt")] += 1
            att = [a for a in (o.get("attention_relationship") or []) if a in ATT][:1] or ["looking_at"]
            rels = att + sorted({r for r in (o.get("spatial_relationship") or []) if r in SPA}) \
                + sorted({r for r in (o.get("contacting_relationship") or []) if r in CON})
            s["triplets"] += len(rels)
            s["pred"].update(rels)
            if not vis:
                s["triplets_unobserved"] += len(rels)
                s["pred_unobserved"].update(rels)
    return s


def collect(paths):
    with ThreadPoolExecutor(16) as ex:
        return dict(zip(paths, ex.map(video_stats, paths)))


def summarise(stats):
    agg = defaultdict(float)
    pred, pred_u, src_u = Counter(), Counter(), Counter()
    for s in stats:
        for k in ("frames", "objects", "objects_unobserved", "triplets", "triplets_unobserved"):
            agg[k] += s[k]
        pred.update(s["pred"]); pred_u.update(s["pred_unobserved"]); src_u.update(s["src_unobserved"])
    n = len(stats)
    tot = sum(pred.values())
    return {"videos": n, "frames_per_video": agg["frames"] / max(n, 1),
            "objects_per_frame": agg["objects"] / max(agg["frames"], 1),
            "unobserved_object_share": agg["objects_unobserved"] / max(agg["objects"], 1),
            "triplets": int(agg["triplets"]), "unobserved_triplet_share": agg["triplets_unobserved"] / max(agg["triplets"], 1),
            "predicate_share": {p: pred[p] / max(tot, 1) for p in PRED},
            "predicate_count": {p: int(pred[p]) for p in PRED},
            "unobserved_predicate_count": {p: int(pred_u[p]) for p in PRED},
            "unobserved_object_sources": dict(src_u)}


def main(out):
    os.makedirs(out, exist_ok=True)
    feat_sets = {}
    for mode, feat in FEATURES:
        d = f"{AG}/features/roi_features/{mode}/{feat}/train"
        feat_sets[f"{mode}/{feat}"] = {vid_of(x) for x in os.listdir(d) if x.endswith(".pkl")}
    ann_files = sorted(x for x in os.listdir(TRAIN_ANN) if x.endswith(".pkl"))
    stats = collect([f"{TRAIN_ANN}/{x}" for x in ann_files])
    by_vid = {vid_of(os.path.basename(p)): s for p, s in stats.items()}
    eligible, dropped = [], Counter()
    for v, s in sorted(by_vid.items()):
        miss = [k for k, fs in feat_sets.items() if v not in fs]
        if s["frames"] < 2:
            dropped["lt2_frames"] += 1
        elif miss:
            dropped["missing_features:" + ",".join(miss)] += 1
        else:
            eligible.append(v)
    ranked = sorted(eligible, key=lambda v: hashlib.sha1(v.encode()).hexdigest())
    val, rest = sorted(ranked[:N_VAL]), sorted(ranked[N_VAL:])
    with open(f"{out}/val750.txt", "w") as f:
        f.write("\n".join(val) + "\n")
    with open(f"{out}/train_minus_val.txt", "w") as f:
        f.write("\n".join(rest) + "\n")
    report = {"rule": f"eligible training videos sorted by sha1(video_id) hex digest; first {N_VAL} = validation",
              "train_annotation_pkls": len(ann_files), "eligible": len(eligible), "dropped": dict(dropped),
              "feature_dirs_checked": list(feat_sets), "val": summarise([by_vid[v] for v in val]),
              "train_minus_val": summarise([by_vid[v] for v in rest])}
    # held-out AG test videos: in the older test annotations but outside the 1,511-video test split
    test_ids = {k: {vid_of(x) for x in os.listdir(d) if x.endswith(".pkl")} for k, d in TEST_SETS.items()}
    test_feat = {}
    for mode, feat in FEATURES:
        d = f"{AG}/features/roi_features/{mode}/{feat}/test"
        test_feat[f"{mode}/{feat}"] = {vid_of(x) for x in os.listdir(d) if x.endswith(".pkl")} if os.path.isdir(d) else set()
    held = sorted((test_ids["old_test_bak_1735"] | test_ids["old_test_1526"]) - test_ids["worldbbox_test"])
    held_paths = []
    for v in held:
        for k in ("old_test_1526", "old_test_bak_1735"):
            if v in test_ids[k]:
                held_paths.append(f"{TEST_SETS[k]}/{v}.mp4.pkl")
                break
    hstats = collect(held_paths)
    report["held_out_test_videos"] = {
        "count": len(held), "in_old_test_1526": len(set(held) & test_ids["old_test_1526"]),
        "with_all_test_features": sum(1 for v in held if all(v in s for s in test_feat.values())),
        "test_feature_counts": {k: len(s) for k, s in test_feat.items()},
        "stats": summarise(list(hstats.values()))}
    report["worldbbox_test_reference"] = summarise(list(collect(
        [f"{TEST_SETS['worldbbox_test']}/{x}" for x in sorted(os.listdir(TEST_SETS['worldbbox_test']))]).values()))
    with open(f"{out}/held_out_test_videos.txt", "w") as f:
        f.write("\n".join(held) + "\n")
    with open(f"{out}/val_split_stats.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("train_annotation_pkls", "eligible", "dropped")}, indent=1))
    for k in ("val", "train_minus_val"):
        r = report[k]
        print(k, {x: (round(r[x], 4) if isinstance(r[x], float) else r[x]) for x in
                  ("videos", "frames_per_video", "objects_per_frame", "unobserved_object_share",
                   "unobserved_triplet_share", "unobserved_object_sources")})
        print("  min predicate count", min(r["predicate_count"].values()),
              "  min unobserved predicate count", min(r["unobserved_predicate_count"].values()))
    h = report["held_out_test_videos"]
    print("held_out", {k: h[k] for k in ("count", "in_old_test_1526", "with_all_test_features", "test_feature_counts")},
          h["stats"]["unobserved_object_sources"])
    print("worldbbox_test sources", report["worldbbox_test_reference"]["unobserved_object_sources"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data3/rohith/ag/runs/rebuttal/splits")
    main(ap.parse_args().out)
