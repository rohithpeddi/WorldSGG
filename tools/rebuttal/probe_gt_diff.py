"""Probe: where the +147 UO triplets of the full GT come from (uncovered objects only)."""
import os, pickle, sys
from collections import Counter
sys.path.insert(0, "/data3/rohith/ag/runs/rebuttal/scripts")
import rescore_rebuttal as rr
c = Counter()
cache = pickle.load(open(rr.GT_CACHE, "rb"))
ref = rr.cg.load(rr.cg.GT_REFERENCE)
cur, ann = None, None
for g, (vid, t) in zip(ref, cache["keys"]):
    if vid != cur:
        ann = pickle.load(open(f"{rr.ANNOT}/{vid}.mp4.pkl", "rb")); cur = vid
        fr = sorted(ann["frames"], key=lambda k: k.split("/")[-1])
    af = ann["frames"][fr[t]]
    slots = {rr._to_short(rr.OBJECT_CLASSES[int(x)]) for x in g["object_classes"][g["valid_mask"].astype(bool)]}
    for o in af["object_info_list"]:
        if o.get("label") in slots: continue
        att = o.get("attention_relationship", []) or []
        if not att: c["empty_att"] += 1
        elif not any(a in rr._ATT_LABEL_TO_IDX for a in att): c["att_not_in_vocab"] += 1
        if len(att) > 1: c["multi_att"] += 1
        for key, m in (("spatial_relationship", rr._SPA_LABEL_TO_IDX), ("contacting_relationship", rr._CON_LABEL_TO_IDX)):
            L = o.get(key, []) or []
            c[key + "_dups"] += len(L) - len(set(L))
            c[key + "_oov"] += sum(1 for r in L if r not in m)
print(dict(c))
