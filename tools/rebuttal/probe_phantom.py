"""Probe: covered GT pairs whose object slot has no annotation object of the same short label (default labels)."""
import pickle, sys
from collections import Counter
sys.path.insert(0, "/data3/rohith/ag/runs/rebuttal/scripts")
import rescore_rebuttal as rr
ref = rr.cg.load(rr.cg.GT_REFERENCE)
keys = pickle.load(open(rr.GT_CACHE, "rb"))["keys"]
c, ex = Counter(), []
cur = None
for g, (vid, t) in zip(ref, keys):
    if vid != cur:
        ann = pickle.load(open(f"{rr.ANNOT}/{vid}.mp4.pkl", "rb")); cur = vid
        fr = sorted(ann["frames"], key=lambda k: k.split("/")[-1])
    af = ann["frames"][fr[t]]
    labels = {o.get("label", rr._to_short(o.get("class", ""))) for o in af["object_info_list"]}
    pv = g["pair_valid"].astype(bool); vis = g["visibility_mask"].astype(bool)
    for k in rr.np.where(pv)[0]:
        o = int(g["object_idx"][k]); cls = rr.OBJECT_CLASSES[int(g["object_classes"][o])]
        if rr._to_short(cls) not in labels:
            n = 1 + int((g["gt_spatial"][k] > .5).sum()) + int((g["gt_contacting"][k] > .5).sum())
            c["pairs"] += 1; c["triplets"] += n; c["uo" if not vis[o] else "oo"] += n; c[cls] += 1
            if len(ex) < 5: ex.append((vid, fr[t], cls, sorted(labels)))
print(dict(c)); print(ex)
