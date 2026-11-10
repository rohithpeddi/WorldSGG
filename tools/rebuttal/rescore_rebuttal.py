"""Rebuttal re-scores of the saved supervised prediction dumps (CPU only; no training, no inference).

Two questions, answered on the same dumps that produced Table 2 and App G (``tools/rescore_common_gt.py``):

C3  With-Constraint ranking.  ``rescore_common_gt.ranked_rows(rec, "with")`` keeps the arg-max predicate of every
    (pair, family) row in construction order (all attention rows, then spatial, then contacting) and R@K scores the
    first K. Here the same rows are also ranked by confidence before the cut:
      * ``sorted``      key = subject score x object score x predicate score, i.e. the sort that is commented out in
                        ``lib/supervised/evaluation_recall.py:487-490`` (object scores are 1 in PredCls);
      * ``sorted_pred`` key = predicate score only (the audit probe's key; SGDet covered only, for reconciliation).
    No Constraint is unchanged (it is already ranked), so sorted scorers report WC only.

A1  Full annotated inventory (wqgS Q1, SRrf Q1).  The covered ground truth has only the objects that own a PredCls
    slot (481,790 triplets, 67,347 UO). Here every object of ``object_info_list`` in every scored frame is ground
    truth (expected 773,236 triplets, 358,793 UO, the zero-shot scorer's GT). Objects without a slot get pseudo
    indices that no prediction maps to, so their triplets are misses. The covered GT also holds 147 triplets of
    GroundingDINO slots whose class is not annotated in that frame (all `doorknob`; the loader gives them a default
    `looking_at`); they are not annotated, so the full inventory drops them. SGDet comes in two variants:
      * ``full_strict`` the slot -> object map is the covered one (an uncovered object can never be hit);
      * ``full_rule``   the same matching rule applied to the full inventory: a slot that matches no visible object
                        of its class at 2D IoU >= 0.5 falls back to an unobserved object of that class, which may now
                        be an uncovered one (``slot_map_common``, L137-150, over the extended class lists).

Gates (must hold before any number is used):
  (1) ``predcls|covered|construction`` and ``sgdet|covered|construction`` reproduce the saved common-GT result JSON
      cell for cell (max |diff| reported);
  (2) every covered GT triplet of the dump is re-derived from the annotation (mismatch count reported, must be 0);
  (3) the full ground truth totals are reported next to the expected 773,236 / 414,443 / 358,793.

Frame alignment: dump records are (video, frame_t) over the loader's common frames = sorted frame files present in
both the feature and the annotation PKL. On the 1,511-video split every video has exactly as many dump records as
annotation frames (checked by ``probe_annot.py``), so frame_t indexes the sorted annotation frames.

Usage (server):
    PY=~/anaconda3/envs/scene4cast/bin/python
    $PY rescore_rebuttal.py --build-gt                 # once: full GT cache
    $PY rescore_rebuttal.py --exp worldwise_pp_dinov3  # per experiment (any key of rescore_common_gt.DUMPS)
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

REPO = os.environ.get("WSGG_REPO", os.path.expanduser("~/CODE/Scene4Cast"))
for p in (REPO, os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import rescore_common_gt as cg  # noqa: E402
from dataloader.world_ag_dataset import (  # noqa: E402
    OBJECT_CLASSES, _ATT_LABEL_TO_IDX, _CON_LABEL_TO_IDX, _SPA_LABEL_TO_IDX, _attention_label_to_idx, _to_short)

ANNOT = "/data/rohith/ag/world4d_rel_annotations_worldbbox/test"
OUT = "/data3/rohith/ag/runs/rebuttal/rescore"
GT_CACHE = f"{OUT}/full_gt.pkl"
PSEUDO = 100000          # pseudo object index offset for objects without a slot
PERSON = OBJECT_CLASSES.index("person")
SHORT_TO_ID = {}
for _i, _c in enumerate(OBJECT_CLASSES):
    SHORT_TO_ID.setdefault(_to_short(_c), _i)


def annot_rels(obj, p, o):
    """GT triplets of one annotation object, built exactly as the loader + ``cg.ground_truth`` build them."""
    out = [(p, o, _attention_label_to_idx(obj.get("attention_relationship", [])))]
    for s in sorted({_SPA_LABEL_TO_IDX[r] for r in obj.get("spatial_relationship", []) if r in _SPA_LABEL_TO_IDX}):
        out.append((o, p, cg.N_ATT + s))
    for c in sorted({_CON_LABEL_TO_IDX[r] for r in obj.get("contacting_relationship", []) if r in _CON_LABEL_TO_IDX}):
        out.append((p, o, cg.N_ATT + cg.N_SPA + c))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Full ground truth (built once from the common PredCls reference dump + the annotation PKLs)
# ----------------------------------------------------------------------------------------------------------------
def build_gt():
    t0 = time.time()
    ref = cg.load(cg.GT_REFERENCE)
    frames_of, ann_cache = {}, {}
    stats = defaultdict(int)
    extras = []          # per record: list of (rel, tag, class_id, pseudo_index)
    drops = []           # per record: indices of covered GT triplets that are not annotated (phantom slots)
    cur_vid, ann = None, None
    for i, g in enumerate(ref):
        vid = g["video_id"]
        if vid != cur_vid:
            with open(f"{ANNOT}/{vid}.mp4.pkl", "rb") as f:
                ann = pickle.load(f)
            frames_of[vid] = sorted(ann["frames"], key=lambda k: k.split("/")[-1])
            cur_vid = vid
        af = ann["frames"][frames_of[vid][int(g["frame_t"])]]
        valid = np.where(g["valid_mask"].astype(bool))[0]
        vis = g["visibility_mask"].astype(bool)
        pslots = [j for j in valid if int(g["object_classes"][j]) == PERSON]
        p = int(pslots[0]) if pslots else -1
        p_vis = bool(vis[p]) if p >= 0 else True
        slot_of = {}
        for j in valid:
            if int(g["object_classes"][j]) != PERSON:
                slot_of.setdefault(_to_short(OBJECT_CLASSES[int(g["object_classes"][j])]), int(j))
        # covered triplets of the dump, keyed by object slot, for gate (2)
        dump_rels, _ = cg.ground_truth(g)
        by_obj = defaultdict(set)
        for (a, b, r) in dump_rels:
            by_obj[b if a == p else a].add((a, b, r))
        # covered triplets of slots whose class has no annotation object in this frame (GroundingDINO fills such
        # as `doorknob`): the loader gives them a default `looking_at` label; they are not annotated, so the full
        # inventory drops them (the covered gate keeps them, as the saved common-GT results do)
        ann_labels = {o.get("label", _to_short(o.get("class", ""))) for o in af.get("object_info_list", [])}
        drop = [gi for gi, (a, b, r) in enumerate(dump_rels)
                if _to_short(OBJECT_CLASSES[int(g["object_classes"][b if a == p else a])]) not in ann_labels]
        stats["phantom_covered_triplets"] += len(drop)
        drops.append(drop)
        rec_extra, seen = [], set()
        for k, obj in enumerate(af.get("object_info_list", [])):
            short = obj.get("label", _to_short(obj.get("class", "")))
            if short in seen:
                stats["duplicate_class_objects"] += 1
                continue
            seen.add(short)
            visible = bool(obj.get("visible", True))
            if short in slot_of:
                o = slot_of[short]
                stats["covered_objects"] += 1
                if set(annot_rels(obj, p, o)) != by_obj.get(o, set()):
                    stats["covered_rel_mismatch"] += 1
                continue
            stats["uncovered_objects_" + ("visible" if visible else "unobserved")] += 1
            if not (obj.get("attention_relationship") or obj.get("spatial_relationship")
                    or obj.get("contacting_relationship")):
                stats["uncovered_objects_without_relations"] += 1
            tag = ("OO" if visible else "UO") if p_vis else "X"
            cid = SHORT_TO_ID.get(short, 0)
            o = PSEUDO + k
            for rel in annot_rels(obj, p, o):
                rec_extra.append((rel, tag, cid, o))
        extras.append(rec_extra)
        stats["frames"] += 1
    full = {"All": 0, "OO": 0, "UO": 0, "X": 0}
    for g, ex, dr in zip(ref, extras, drops):
        _, tags = cg.ground_truth(g)
        tags = [t for gi, t in enumerate(tags) if gi not in set(dr)]
        for t in tags + [e[1] for e in ex]:
            full[t] += 1
            if t != "X":
                full["All"] += 1
    stats = dict(stats)
    stats["full_triplets"] = full
    stats["expected"] = {"All": 773236, "OO": 414443, "UO": 358793}
    os.makedirs(OUT, exist_ok=True)
    with open(GT_CACHE, "wb") as f:
        pickle.dump({"extras": extras, "drops": drops, "stats": stats,
                     "keys": [(g["video_id"], int(g["frame_t"])) for g in ref]}, f)
    print(json.dumps(stats, indent=1))
    print(f"built in {time.time() - t0:.0f}s -> {GT_CACHE}")


# ----------------------------------------------------------------------------------------------------------------
# Ranking and slot maps
# ----------------------------------------------------------------------------------------------------------------
def wc_rows(rec, order):
    """With-Constraint rows: construction order (as in cg.ranked_rows) or ranked by confidence."""
    pv = rec["pair_valid"].astype(bool)
    kv = int(pv.sum())
    if kv == 0:
        return np.zeros((0, 3), dtype=np.int64)
    pi, oi = rec["person_idx"][pv], rec["object_idx"][pv]
    rels_i = np.concatenate([np.stack([pi, oi], 1), np.stack([oi, pi], 1), np.stack([pi, oi], 1)], 0)
    att, spa, con = rec["attention_distribution"][pv], rec["spatial_distribution"][pv], rec["contacting_distribution"][pv]
    sc = np.concatenate([
        np.concatenate([att, np.zeros((kv, cg.N_SPA + cg.N_CON))], 1),
        np.concatenate([np.zeros((kv, cg.N_ATT)), spa, np.zeros((kv, cg.N_CON))], 1),
        np.concatenate([np.zeros((kv, cg.N_ATT + cg.N_SPA)), con], 1)], 0)
    rows = np.column_stack((rels_i, sc.argmax(1)))
    if order == "construction":
        return rows
    key = sc.max(1)
    if order == "sorted":
        obj = np.asarray(rec.get("pred_scores", np.ones(len(rec["object_classes"]))), dtype=np.float64)
        key = obj[rels_i[:, 0]] * obj[rels_i[:, 1]] * key
    return rows[np.argsort(-key, kind="stable")]


def slot_map_full(srec, grec, extra):
    """slot_map_common over the full inventory: uncovered (unobserved) objects join their class's candidate list."""
    gv = grec["valid_mask"].astype(bool)
    gvis = grec["visibility_mask"].astype(bool)
    by_class = defaultdict(list)
    for j in np.where(gv)[0]:
        by_class[int(grec["object_classes"][j])].append(("slot", int(j)))
    for o, cid in sorted({(e[3], e[2]) for e in extra}):
        by_class[cid].append(("pseudo", o))
    pv = srec["pair_valid"].astype(bool)
    used = set(srec["person_idx"][pv].tolist()) | set(srec["object_idx"][pv].tolist())
    labels = srec.get("pred_labels", srec["object_classes"])
    m = {}
    for s in used:
        best, best_iou, hidden = -1, 0.0, -1
        for kind, j in by_class.get(int(labels[s]), []):
            if kind == "slot" and gvis[j]:
                v = cg.iou(srec["bboxes_2d"][s], grec["bboxes_2d"][j])
                if v >= 0.5 and v > best_iou:
                    best, best_iou = j, v
            elif hidden < 0:
                hidden = j
        if best >= 0:
            m[s] = best
        elif hidden >= 0:
            m[s] = hidden
    return m


def run(exp):
    t0 = time.time()
    with open(GT_CACHE, "rb") as f:
        cache = pickle.load(f)
    pdump, sdump = cg.DUMPS[exp]
    prec, srec = cg.load(pdump), cg.load(sdump)
    assert len(prec) == len(srec) == len(cache["keys"]), (len(prec), len(srec), len(cache["keys"]))
    names = ["predcls|covered|construction", "predcls|covered|sorted", "predcls|full|construction",
             "predcls|full|sorted", "sgdet|covered|construction", "sgdet|covered|sorted",
             "sgdet|covered|sorted_pred", "sgdet|full_strict|construction", "sgdet|full_strict|sorted",
             "sgdet|full_rule|construction", "sgdet|full_rule|sorted"]
    sc = {n: cg.Scorer() for n in names}
    st = defaultdict(int)
    for i, (g, s, ex, dr) in enumerate(zip(prec, srec, cache["extras"], cache["drops"])):
        assert (g["video_id"], int(g["frame_t"])) == cache["keys"][i] == (s["video_id"], int(s["frame_t"])), i
        rels, tags = cg.ground_truth(g)
        dr = set(dr)
        frels = [r for gi, r in enumerate(rels) if gi not in dr] + [e[0] for e in ex]
        ftags = [t for gi, t in enumerate(tags) if gi not in dr] + [e[1] for e in ex]
        ident = {j: j for j in np.where(g["valid_mask"].astype(bool))[0].tolist()}
        nc_p, nc_s = cg.ranked_rows(g, "no"), cg.ranked_rows(s, "no")
        p_con, p_srt = wc_rows(g, "construction"), wc_rows(g, "sorted")
        s_con, s_srt, s_pred = wc_rows(s, "construction"), wc_rows(s, "sorted"), wc_rows(s, "sorted_pred")
        m_cov = cg.slot_map_common(s, g, st) if rels or frels else {}
        if rels:
            sc["predcls|covered|construction"].add(rels, tags, {"wc": p_con, "nc": nc_p}, ident)
            sc["predcls|covered|sorted"].add(rels, tags, {"wc": p_srt}, ident)
            sc["sgdet|covered|construction"].add(rels, tags, {"wc": s_con, "nc": nc_s}, m_cov)
            sc["sgdet|covered|sorted"].add(rels, tags, {"wc": s_srt}, m_cov)
            sc["sgdet|covered|sorted_pred"].add(rels, tags, {"wc": s_pred}, m_cov)
        if frels:
            m_full = slot_map_full(s, g, ex)
            st["slots_mapped_to_uncovered"] += sum(1 for v in m_full.values() if v >= PSEUDO)
            sc["predcls|full|construction"].add(frels, ftags, {"wc": p_con, "nc": nc_p}, ident)
            sc["predcls|full|sorted"].add(frels, ftags, {"wc": p_srt}, ident)
            sc["sgdet|full_strict|construction"].add(frels, ftags, {"wc": s_con, "nc": nc_s}, m_cov)
            sc["sgdet|full_strict|sorted"].add(frels, ftags, {"wc": s_srt}, m_cov)
            sc["sgdet|full_rule|construction"].add(frels, ftags, {"wc": s_con, "nc": nc_s}, m_full)
            sc["sgdet|full_rule|sorted"].add(frels, ftags, {"wc": s_srt}, m_full)
    res = {"experiment": exp, "predcls_dump": pdump, "sgdet_dump": sdump, "gt_cache_stats": cache["stats"],
           "slot_stats": dict(st)}
    for n, s_ in sc.items():
        summ = s_.summary()
        if n.endswith("sorted") or n.endswith("sorted_pred"):
            summ.pop("nc")
        res[n] = summ
    # gate (1): covered/construction must reproduce the saved common-GT JSON
    saved = json.load(open(f"{cg.R}/common_gt/results/{exp}.json"))
    diff = 0.0
    for mine, theirs in (("predcls|covered|construction", "predcls"), ("sgdet|covered|construction", "sgdet_common")):
        for c in ("wc", "nc"):
            for b in cg.BUCKETS:
                for met in ("R", "mR", "pooled_R", "pooled_mR"):
                    for k in cg.KS:
                        a, bb = res[mine][c][b][met][k], saved[theirs][c][b][met][str(k)]
                        if a is None or bb is None:
                            assert a is None and bb is None, (mine, c, b, met, k, a, bb)
                            continue
                        diff = max(diff, abs(a - bb))
    res["gate_max_abs_diff_vs_common_gt"] = diff
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{exp}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{exp}] gate max|diff|={diff:.2e}  {res['seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-gt", action="store_true")
    ap.add_argument("--exp", choices=sorted(cg.DUMPS))
    a = ap.parse_args()
    if a.build_gt:
        build_gt()
    if a.exp:
        run(a.exp)
