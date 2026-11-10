"""Probe (CPU, read-only): annotation PKL layout and whether dump frames == sorted annotation frames per video."""
import os, pickle, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.expanduser("~/CODE/Scene4Cast"))
from dataloader.world_ag_dataset import OBJECT_CLASSES
A = "/data/rohith/ag/world4d_rel_annotations_worldbbox/test"
D = "/data3/rohith/ag/runs/rescore/dumps/w_sttran_predcls_resnet50__all.pkl"
recs = pickle.load(open(D, "rb"))["records"]
per_vid = Counter(r["video_id"] for r in recs)
print("person idx", OBJECT_CLASSES.index("person") if "person" in OBJECT_CLASSES else None, OBJECT_CLASSES[:4])
mismatch, n = 0, 0
for i, (vid, T) in enumerate(sorted(per_vid.items())):
    ann = pickle.load(open(f"{A}/{vid}.mp4.pkl", "rb"))
    fr = ann["frames"]
    if i == 0:
        print("top keys", list(ann.keys()))
        fk = sorted(fr)[0]; f = fr[fk]
        print("frame key", fk, "frame keys", list(f.keys()))
        print("person_info", {k: (v if not hasattr(v, "shape") else v.shape) for k, v in f["person_info"].items() if k not in ("corners_final",)})
        o = f["object_info_list"][0]
        print("object keys", {k: (v if not hasattr(v, "shape") else v.shape) for k, v in o.items() if k not in ("corners_final",)})
    if len(fr) != T:
        mismatch += 1
        if mismatch <= 5: print("count mismatch", vid, len(fr), T)
    n += 1
print("videos", n, "count mismatches", mismatch)
