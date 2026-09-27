"""Re-score training-based WSGG prediction dumps on one common ground truth, with recall per frame.

Why
---
The paper's SGDet columns were scored on each detector's own slots: ``evaluate_wsgg_video`` builds its ground
truth from the relationship labels the data loader attached to the detector slots, and in SGDet it compares the
detector boxes with themselves (the annotated boxes are read after ``gt_entry`` is built), so no 2D-IoU test is
applied, false detections contribute default labels, duplicates count twice, and missed objects leave the
denominator. The visibility buckets were in addition pooled over the corpus instead of averaged over frames.

What this script computes
-------------------------
Ground truth: the PredCls ground truth of every frame (identical for all methods and backbones: 481,790 triplets on
the 1,511-video test split), tagged OO / UO by the visibility of the object (UO = the paper's unobserved bucket, the
evaluator's ``OU``). It is the reference for BOTH settings.

* ``predcls``        slots are the ground-truth objects (identity mapping); reproduces the paper's PredCls All columns.
* ``sgdet_common``   every detector slot used by a prediction is mapped to a ground-truth object of the same class:
                     a visible object additionally needs 2D IoU >= 0.5 between the slot box and the annotated box;
                     an unobserved object is matched by class, which is the slot identity the models carry (one slot
                     per class; unobserved objects enter as supplied boxes). Ground-truth triplets whose objects get
                     no slot are misses; a ground-truth triplet counts once however many predictions hit it.
* ``sgdet_legacy``   the paper's current SGDet rule (ground truth = the SGDet record's own slot labels, identity
                     mapping); reproduces the reported SGDet All columns and serves as a check.

Ranking is the original code path (``evaluate_wsgg_video`` + ``evaluate_from_dict``): With Constraint keeps the
arg-max predicate of every (pair, family) row in construction order; No Constraint ranks all pair-predicate scores
and keeps the top 100. For every frame with ground truth, R@K = hits / ground-truth triplets, averaged over frames;
mR@K averages each predicate's per-frame recall over the frames where it occurs and then over predicates (the
paper's convention divides by all 26 predicates; ``mR_supported`` divides by the predicates with support).
Pooled (corpus-level) R/mR are reported as well, to cross-check against the existing bucket files.

Usage (on the server, from the repo root):
    python tools/rescore_common_gt.py --exp worldwise_pp_dinov3 --out /data3/rohith/ag/runs/common_gt/results
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from lib.supervised.evaluation_recall import argsort_desc  # noqa: E402

N_ATT, N_SPA, N_CON = 3, 6, 17
NUM_REL = N_ATT + N_SPA + N_CON
KS = (10, 20, 50)
BUCKETS = ("All", "OO", "UO")
R = "/data3/rohith/ag/runs"
DUMPS = {  # experiment -> (predcls dump, sgdet dump)
    **{m: (f"{R}/rescore/dumps/{m}_predcls_resnet50__all.pkl", f"{R}/rescore/dumps/{m}_sgdet_resnet50__all.pkl")
       for m in ("w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "w_usg")},
    **{f"worldwise_{b}": (f"{R}/rescore/dumps/worldwise_predcls_{b}__all.pkl", f"{R}/rescore/dumps/worldwise_sgdet_{b}__all.pkl")
       for b in ("resnet50", "dinov2b", "dinov2l", "dinov3l")},
    **{f"worldwise_plus_{b}": (f"{R}/worldformer/score/dumps/worldformer_c1_{b}_predcls__all.pkl",
                               f"{R}/worldformer/score/dumps/worldformer_c1_{b}_sgdet__all.pkl")
       for b in ("dinov3tok", "pi3tok", "fused")},
    **{f"worldwise_pp_{b}": (f"{R}/worldwise_pp/score/dumps/worldwise_pp_{b}_predcls__all.pkl",
                             f"{R}/worldwise_pp/score/dumps/worldwise_pp_{b}_sgdet__all.pkl")
       for b in ("dinov3", "dinov3_nodet")},
    # PUF-style fusion arms over the W-DSGDetr++ front end (full 1,511-video split)
    **{f"puf_{s}": (f"{R}/ext/puf/full/dumps/puf_{s}_predcls.pkl", f"{R}/ext/puf/full/dumps/puf_{s}_sgdet.pkl")
       for s in ("lks", "lks_bi", "fross", "puf", "puf_prior", "puf_prior_noobs")},
}
GT_REFERENCE = DUMPS["w_sttran"][0]   # PredCls ground truth is identical across dumps; checked per frame below


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)["records"]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ground_truth(rec):
    """GT relations (subject, object, predicate) with a bucket tag, in evaluate_wsgg_video's order."""
    pv = rec["pair_valid"].astype(bool)
    vis = rec["visibility_mask"].astype(bool)
    rels, tags = [], []
    for k in np.where(pv)[0]:
        p, o = int(rec["person_idx"][k]), int(rec["object_idx"][k])
        tag = ("OO" if vis[o] else "UO") if vis[p] else "X"      # X: person unobserved (not a paper bucket)
        rels.append((p, o, int(rec["gt_attention"][k])))
        tags.append(tag)
        for s in np.where(rec["gt_spatial"][k] > 0.5)[0]:
            rels.append((o, p, N_ATT + int(s)))
            tags.append(tag)
        for c in np.where(rec["gt_contacting"][k] > 0.5)[0]:
            rels.append((p, o, N_ATT + N_SPA + int(c)))
            tags.append(tag)
    return rels, tags


def ranked_rows(rec, constraint):
    """Predicted (slot_a, slot_b, predicate) rows, exactly as evaluate_wsgg_video + evaluate_from_dict order them."""
    pv = rec["pair_valid"].astype(bool)
    k_valid = int(pv.sum())
    if k_valid == 0:
        return np.zeros((0, 3), dtype=np.int64)
    pi, oi = rec["person_idx"][pv], rec["object_idx"][pv]
    rels_i = np.concatenate([np.stack([pi, oi], 1), np.stack([oi, pi], 1), np.stack([pi, oi], 1)], 0)
    att, spa, con = rec["attention_distribution"][pv], rec["spatial_distribution"][pv], rec["contacting_distribution"][pv]
    scores = np.concatenate([
        np.concatenate([att, np.zeros((k_valid, N_SPA + N_CON))], 1),
        np.concatenate([np.zeros((k_valid, N_ATT)), spa, np.zeros((k_valid, N_CON))], 1),
        np.concatenate([np.zeros((k_valid, N_ATT + N_SPA)), con], 1)], 0)
    if constraint == "no":
        obj_scores = np.asarray(rec.get("pred_scores", np.ones(len(rec["object_classes"]))), dtype=np.float32)
        overall = obj_scores[rels_i].prod(1)[:, None] * scores
        order = argsort_desc(overall)[:100]
        return np.column_stack((rels_i[order[:, 0]], order[:, 1]))
    return np.column_stack((rels_i, scores.argmax(1)))


def slot_map_common(srec, grec, stats):
    """Map the SGDet slots used by predictions to PredCls GT objects (class + IoU>=0.5 if visible; class if not)."""
    gv = grec["valid_mask"].astype(bool)
    gvis = grec["visibility_mask"].astype(bool)
    by_class = defaultdict(list)
    for j in np.where(gv)[0]:
        by_class[int(grec["object_classes"][j])].append(j)
    pv = srec["pair_valid"].astype(bool)
    used = set(srec["person_idx"][pv].tolist()) | set(srec["object_idx"][pv].tolist())
    labels = srec.get("pred_labels", srec["object_classes"])
    m = {}
    for s in used:
        cands = by_class.get(int(labels[s]), [])
        best, best_iou, hidden = -1, 0.0, -1
        for j in cands:
            if gvis[j]:
                v = iou(srec["bboxes_2d"][s], grec["bboxes_2d"][j])
                if v >= 0.5 and v > best_iou:
                    best, best_iou = j, v
            elif hidden < 0:
                hidden = j
        if best >= 0:
            m[s] = best
        elif hidden >= 0:
            m[s] = hidden
    # coverage diagnostics: which GT objects got at least one slot
    hit_objs = set(m.values())
    for j in np.where(gv)[0]:
        key = "visible" if gvis[j] else "unobserved"
        stats[f"gt_objects_{key}"] += 1
        stats[f"gt_objects_{key}_matched"] += int(j in hit_objs)
    return m


class Scorer:
    def __init__(self):
        self.frame_r = {c: {b: {k: [] for k in KS} for b in BUCKETS} for c in ("wc", "nc")}
        self.frame_pr = {c: {b: {k: [[] for _ in range(NUM_REL)] for k in KS} for b in BUCKETS} for c in ("wc", "nc")}
        self.tot = {b: np.zeros(NUM_REL, np.int64) for b in BUCKETS}
        self.hit = {c: {b: {k: np.zeros(NUM_REL, np.int64) for k in KS} for b in BUCKETS} for c in ("wc", "nc")}
        self.n_frames = 0

    def add(self, rels, tags, rows_by_c, m):
        """rels/tags: GT; rows_by_c: {'wc'|'nc': ranked rows}; m: slot -> GT object index (dict)."""
        self.n_frames += 1
        key_to_gt = defaultdict(list)
        for g, r in enumerate(rels):
            key_to_gt[r].append(g)
        members = {"All": list(range(len(rels))), "OO": [], "UO": []}
        for g, t in enumerate(tags):
            if t in members:
                members[t].append(g)
        for b in BUCKETS:
            for g in members[b]:
                self.tot[b][rels[g][2]] += 1
        for c, rows in rows_by_c.items():
            for k in KS:
                hit = set()
                for a, bb, p in rows[:k]:
                    ga, gb = m.get(int(a), -1), m.get(int(bb), -1)
                    if ga >= 0 and gb >= 0:
                        hit.update(key_to_gt.get((ga, gb, int(p)), ()))
                for b in BUCKETS:
                    idx = members[b]
                    if not idx:
                        continue
                    h = [g for g in idx if g in hit]
                    self.frame_r[c][b][k].append(len(h) / len(idx))
                    cnt = defaultdict(int)
                    hc = defaultdict(int)
                    for g in idx:
                        cnt[rels[g][2]] += 1
                    for g in h:
                        hc[rels[g][2]] += 1
                        self.hit[c][b][k][rels[g][2]] += 1
                    for p, n in cnt.items():
                        self.frame_pr[c][b][k][p].append(hc[p] / n)

    def summary(self):
        out = {}
        for c in ("wc", "nc"):
            out[c] = {}
            for b in BUCKETS:
                d = {"R": {}, "mR": {}, "mR_supported": {}, "pooled_R": {}, "pooled_mR": {}}
                for k in KS:
                    fr = self.frame_r[c][b][k]
                    d["R"][k] = float(np.mean(fr)) if fr else None
                    per = [float(np.mean(v)) if v else None for v in self.frame_pr[c][b][k]]
                    d["mR"][k] = float(sum(x for x in per if x is not None) / NUM_REL)
                    sup = [x for x in per if x is not None]
                    d["mR_supported"][k] = float(np.mean(sup)) if sup else None
                    tot, hit = self.tot[b], self.hit[c][b][k]
                    d["pooled_R"][k] = float(hit.sum() / tot.sum()) if tot.sum() else None
                    pm = [hit[p] / tot[p] for p in range(NUM_REL) if tot[p] > 0]
                    d["pooled_mR"][k] = float(np.mean(pm)) if pm else None
                d["n_frames"] = len(self.frame_r[c][b][KS[0]])
                d["n_gt"] = int(self.tot[b].sum())
                d["n_predicates_supported"] = int((self.tot[b] > 0).sum())
                out[c][b] = d
        return out


def run(exp, out_dir):
    t0 = time.time()
    pdump, sdump = DUMPS[exp]
    ref = load(GT_REFERENCE) if pdump != GT_REFERENCE else None
    prec = load(pdump)
    srec = load(sdump)
    assert len(prec) == len(srec), (len(prec), len(srec))
    scorers = {"predcls": Scorer(), "sgdet_common": Scorer(), "sgdet_legacy": Scorer()}
    stats = defaultdict(int)
    for i, (g, s) in enumerate(zip(prec, srec)):
        assert g["video_id"] == s["video_id"] and g.get("frame_t", i) == s.get("frame_t", i), (i, g["video_id"], s["video_id"])
        if ref is not None and i % 97 == 0:     # spot-check that this dump's PredCls GT is the common reference
            for key in ("pair_valid", "person_idx", "object_idx", "object_classes", "gt_attention", "visibility_mask"):
                assert np.array_equal(ref[i][key], g[key]), (exp, i, key)
        rels, tags = ground_truth(g)
        if rels:
            ident = {j: j for j in np.where(g["valid_mask"].astype(bool))[0].tolist()}
            scorers["predcls"].add(rels, tags, {"wc": ranked_rows(g, "with"), "nc": ranked_rows(g, "no")}, ident)
            m = slot_map_common(s, g, stats)
            scorers["sgdet_common"].add(rels, tags, {"wc": ranked_rows(s, "with"), "nc": ranked_rows(s, "no")}, m)
        lrels, ltags = ground_truth(s)
        if lrels and int(s["pair_valid"].sum()) > 0:
            ident_s = {j: j for j in np.where(s["valid_mask"].astype(bool))[0].tolist()}
            scorers["sgdet_legacy"].add(lrels, ltags, {"wc": ranked_rows(s, "with"), "nc": ranked_rows(s, "no")}, ident_s)
    result = {"experiment": exp, "predcls_dump": pdump, "sgdet_dump": sdump, "gt_reference": GT_REFERENCE,
              "n_records": len(prec), "slot_match": dict(stats),
              **{name: sc.summary() for name, sc in scorers.items()}, "seconds": round(time.time() - t0, 1)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{exp}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{exp}] done in {result['seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(DUMPS))
    ap.add_argument("--out", default=f"{R}/common_gt/results")
    args = ap.parse_args()
    run(args.exp, args.out)
