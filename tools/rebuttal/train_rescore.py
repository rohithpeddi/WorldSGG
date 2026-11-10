"""TRAIN: score rebuttal-round dumps with the ``rescore_rebuttal`` rules (covered + full inventory, construction-
order and sorted With-Constraint), on ground truth rebuilt for the dumps' slot layout. CPU only.

The published full-GT cache (``rescore/full_gt.pkl``) is keyed to the published PredCls dump layout. With slot
alignment the slots of every frame are permuted, so the cache is rebuilt here from a PredCls ground-truth
reference with the new layout. ``rescore_rebuttal.py`` and ``rescore_common_gt.py`` are imported, never edited:
``build_gt`` runs with its two module globals (reference dump, cache path) pointed at our files, and scoring calls
the same ``cg.ground_truth`` / ``cg.ranked_rows`` / ``rr.wc_rows`` / ``cg.slot_map_common`` / ``rr.slot_map_full``
/ ``cg.Scorer`` in the same order as ``rescore_rebuttal.run``.

Commands
  gt-ref       GT-only PredCls records for the 1,511-video test split straight from the loader (no model):
               the GT fields of a dump (pair_valid, person/object_idx, object_classes, bboxes_2d, valid_mask,
               visibility_mask, gt_*). One per (feature model, slot alignment): SGDet boxes are matched against
               the PredCls boxes of the same feature box space.
  build-gt     full-GT cache from a reference (rescore_rebuttal.build_gt, parametrised).
  gate-ref     (a) a ``--slot_alignment none`` reference equals the published dump's GT fields exactly;
               (b) an aligned reference has, frame by frame, the same class-level GT triplet multiset (subject
                   class, object class, predicate, bucket) as the published reference, and its cache the same
                   full-inventory totals (773,236 / 414,443 / 358,793) and per-frame uncovered triplets.
  gate-scorer  scoring the published dumps with the published cache reproduces ``rescore/<exp>.json``.
  score        one dump (PredCls or SGDet) -> JSON with every rescore_rebuttal cell.
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WSGG_REPO", REPO)
for p in (REPO, os.path.join(REPO, "tools"), os.path.join(REPO, "tools", "rebuttal")):
    if p not in sys.path:
        sys.path.insert(0, p)
import rescore_rebuttal as rr  # noqa: E402
cg = rr.cg

GT_KEYS = ("gt_attention", "gt_spatial", "gt_contacting", "pair_valid", "person_idx", "object_idx",
           "object_classes", "bboxes_2d", "valid_mask", "visibility_mask")
LAYOUT_KEYS = ("pair_valid", "person_idx", "object_idx", "object_classes", "gt_attention", "visibility_mask")
PUBLISHED_CACHE = "/data3/rohith/ag/runs/rebuttal/rescore/full_gt.pkl"
PUBLISHED_SCORES = "/data3/rohith/ag/runs/rebuttal/rescore"


def load(path):
    return cg.load(path)


# ----------------------------------------------------------------------------------------------------------------
def cmd_gt_ref(a):
    from dataloader.world_ag_dataset import WorldAG
    ds = WorldAG(phase="test", data_path=a.data_path, mode="predcls", feature_model=a.feature_model,
                 include_invisible=True, max_objects=64, annot_dir_name="world4d_rel_annotations_worldbbox",
                 slot_alignment=a.slot_alignment, inventory=a.inventory)
    recs, t0 = [], time.time()
    for i in range(len(ds) if a.limit <= 0 else a.limit):
        it = ds[i]
        for t in range(int(it["T"])):
            r = {"video_id": it["video_id"], "frame_t": t, "is_last": t == it["T"] - 1}
            for k in GT_KEYS:
                src = {"bboxes_2d": "bboxes_2d"}.get(k, k)
                r[k] = it[src][t].numpy()
            recs.append(r)
        if i % 200 == 0:
            print(f"  {i}/{len(ds)} videos {time.time() - t0:.0f}s", flush=True)
    meta = {"kind": "gt_reference", "mode": "predcls", "feature_model": a.feature_model,
            "slot_alignment": a.slot_alignment, "inventory": a.inventory,
            "n_videos": len({r['video_id'] for r in recs}), "n_frames": len(recs)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"meta": meta, "records": recs}, f, protocol=4)
    print(json.dumps(meta))


def cmd_build_gt(a):
    rr.cg.GT_REFERENCE = a.ref
    rr.GT_CACHE = a.cache
    rr.OUT = os.path.dirname(a.cache)
    rr.build_gt()


# ----------------------------------------------------------------------------------------------------------------
def class_triplets(rec):
    rels, tags = cg.ground_truth(rec)
    oc = rec["object_classes"]
    return Counter((int(oc[s]), int(oc[o]), int(p), t) for (s, o, p), t in zip(rels, tags))


def cmd_gate_ref(a):
    ref = load(a.ref)
    pub = load(a.published)
    out = {"ref": a.ref, "published": a.published, "n_ref": len(ref), "n_pub": len(pub)}
    assert len(ref) == len(pub), (len(ref), len(pub))
    exact_bad = Counter()
    cls_bad = 0
    for i, (r, p) in enumerate(zip(ref, pub)):
        assert (r["video_id"], int(r["frame_t"])) == (p["video_id"], int(p["frame_t"])), i
        if a.exact:
            for k in GT_KEYS:
                if not np.array_equal(r[k], p[k]):
                    exact_bad[k] += 1
        if class_triplets(r) != class_triplets(p):
            cls_bad += 1
    out["exact_mismatch_frames_by_key"] = dict(exact_bad)
    out["class_triplet_mismatch_frames"] = cls_bad
    if a.cache:
        mine = pickle.load(open(a.cache, "rb"))
        theirs = pickle.load(open(a.published_cache, "rb"))
        out["cache_full_triplets"] = mine["stats"]["full_triplets"]
        out["published_full_triplets"] = theirs["stats"]["full_triplets"]
        assert mine["keys"] == theirs["keys"]
        ex_bad = sum(1 for x, y in zip(mine["extras"], theirs["extras"])
                     if Counter((e[0][2], e[1], e[2]) for e in x) != Counter((e[0][2], e[1], e[2]) for e in y))
        out["uncovered_triplet_mismatch_frames"] = ex_bad
        out["cache_stats_equal_except_order"] = {k: (mine["stats"].get(k), theirs["stats"].get(k))
                                                 for k in set(mine["stats"]) | set(theirs["stats"])
                                                 if mine["stats"].get(k) != theirs["stats"].get(k)}
    out["pass"] = (cls_bad == 0 and not exact_bad
                   and (not a.cache or (out["uncovered_triplet_mismatch_frames"] == 0
                                        and out["cache_full_triplets"] == out["published_full_triplets"])))
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    sys.exit(0 if out["pass"] else 1)


# ----------------------------------------------------------------------------------------------------------------
PRED_NAMES = ["predcls|covered|construction", "predcls|covered|sorted", "predcls|full|construction",
              "predcls|full|sorted"]
SG_NAMES = ["sgdet|covered|construction", "sgdet|covered|sorted", "sgdet|covered|sorted_pred",
            "sgdet|full_strict|construction", "sgdet|full_strict|sorted", "sgdet|full_rule|construction",
            "sgdet|full_rule|sorted"]
# Round 2 (full inventory): an unobserved GT object can be matched only by a slot that is not detected in
# that frame (a held slot, visibility_mask False); detected slots match visible objects by class + 2-D IoU.
SG_EXTRA = ["sgdet|heldonly|construction", "sgdet|heldonly|sorted"]


def slot_map_heldonly(srec, grec, stats):
    """cg.slot_map_common with the unobserved fallback restricted to undetected (held) SGDet slots."""
    gv = grec["valid_mask"].astype(bool)
    gvis = grec["visibility_mask"].astype(bool)
    by_class = defaultdict(list)
    for j in np.where(gv)[0]:
        by_class[int(grec["object_classes"][j])].append(j)
    pv = srec["pair_valid"].astype(bool)
    used = set(srec["person_idx"][pv].tolist()) | set(srec["object_idx"][pv].tolist())
    labels = srec.get("pred_labels", srec["object_classes"])
    svis = srec["visibility_mask"].astype(bool)
    m = {}
    for s in used:
        best, best_iou, hidden = -1, 0.0, -1
        for j in by_class.get(int(labels[s]), []):
            if gvis[j]:
                if svis[s]:
                    v = cg.iou(srec["bboxes_2d"][s], grec["bboxes_2d"][j])
                    if v >= 0.5 and v > best_iou:
                        best, best_iou = j, v
            elif hidden < 0 and not svis[s]:
                hidden = j
        if best >= 0:
            m[s] = best
        elif hidden >= 0:
            m[s] = hidden
    stats["heldonly_slots_mapped_unobserved"] += sum(1 for s, j in m.items() if not gvis[j])
    return m


def score(mode, dump_recs, ref_recs, cache):
    """rescore_rebuttal.run restricted to one mode. PredCls: the dump is its own GT (its layout must equal the
    reference's). SGDet: GT = the PredCls reference of the same feature box space."""
    names = PRED_NAMES if mode == "predcls" else SG_NAMES + SG_EXTRA
    sc = {n: cg.Scorer() for n in names}
    st = defaultdict(int)
    assert len(dump_recs) == len(ref_recs) == len(cache["keys"]), (len(dump_recs), len(ref_recs), len(cache["keys"]))
    for i, (d, g, ex, dr) in enumerate(zip(dump_recs, ref_recs, cache["extras"], cache["drops"])):
        key = (g["video_id"], int(g["frame_t"]))
        assert key == cache["keys"][i] == (d["video_id"], int(d["frame_t"])), i
        if mode == "predcls":
            for k in LAYOUT_KEYS:
                if not np.array_equal(d[k], g[k]):
                    raise AssertionError(f"PredCls dump layout differs from the reference at record {i} ({key}): {k}")
            g = d
        rels, tags = cg.ground_truth(g)
        dr = set(dr)
        frels = [r for gi, r in enumerate(rels) if gi not in dr] + [e[0] for e in ex]
        ftags = [t for gi, t in enumerate(tags) if gi not in dr] + [e[1] for e in ex]
        if mode == "predcls":
            ident = {j: j for j in np.where(g["valid_mask"].astype(bool))[0].tolist()}
            nc_p = cg.ranked_rows(g, "no")
            p_con, p_srt = rr.wc_rows(g, "construction"), rr.wc_rows(g, "sorted")
            if rels:
                sc["predcls|covered|construction"].add(rels, tags, {"wc": p_con, "nc": nc_p}, ident)
                sc["predcls|covered|sorted"].add(rels, tags, {"wc": p_srt}, ident)
            if frels:
                sc["predcls|full|construction"].add(frels, ftags, {"wc": p_con, "nc": nc_p}, ident)
                sc["predcls|full|sorted"].add(frels, ftags, {"wc": p_srt}, ident)
        else:
            s = d
            nc_s = cg.ranked_rows(s, "no")
            s_con, s_srt, s_pred = rr.wc_rows(s, "construction"), rr.wc_rows(s, "sorted"), rr.wc_rows(s, "sorted_pred")
            m_cov = cg.slot_map_common(s, g, st) if rels or frels else {}
            if rels:
                sc["sgdet|covered|construction"].add(rels, tags, {"wc": s_con, "nc": nc_s}, m_cov)
                sc["sgdet|covered|sorted"].add(rels, tags, {"wc": s_srt}, m_cov)
                sc["sgdet|covered|sorted_pred"].add(rels, tags, {"wc": s_pred}, m_cov)
            if frels:
                m_full = rr.slot_map_full(s, g, ex)
                st["slots_mapped_to_uncovered"] += sum(1 for v in m_full.values() if v >= rr.PSEUDO)
                sc["sgdet|full_strict|construction"].add(frels, ftags, {"wc": s_con, "nc": nc_s}, m_cov)
                sc["sgdet|full_strict|sorted"].add(frels, ftags, {"wc": s_srt}, m_cov)
                sc["sgdet|full_rule|construction"].add(frels, ftags, {"wc": s_con, "nc": nc_s}, m_full)
                sc["sgdet|full_rule|sorted"].add(frels, ftags, {"wc": s_srt}, m_full)
                if "visibility_mask" in s:
                    m_held = slot_map_heldonly(s, g, st)
                    sc["sgdet|heldonly|construction"].add(frels, ftags, {"wc": s_con, "nc": nc_s}, m_held)
                    sc["sgdet|heldonly|sorted"].add(frels, ftags, {"wc": s_srt}, m_held)
    res = {"slot_stats": dict(st)}
    for n, s_ in sc.items():
        if s_.n_frames == 0:
            continue
        summ = s_.summary()
        if n.endswith("sorted") or n.endswith("sorted_pred"):
            summ.pop("nc")
        res[n] = summ
    return res


def cmd_score(a):
    t0 = time.time()
    d = pickle.load(open(a.dump, "rb"))
    cache = pickle.load(open(a.cache, "rb"))
    ref = load(a.ref)
    if a.allow_subset and len(d["records"]) != len(cache["keys"]):
        # smoke dumps: keep the reference / cache frames the dump has (never for reported numbers)
        want = {(r["video_id"], int(r["frame_t"])) for r in d["records"]}
        idx = [i for i, k in enumerate(cache["keys"]) if k in want]
        ref = [ref[i] for i in idx]
        cache = dict(cache, keys=[cache["keys"][i] for i in idx], extras=[cache["extras"][i] for i in idx],
                     drops=[cache["drops"][i] for i in idx])
    res = {"dump": a.dump, "dump_meta": d.get("meta"), "ref": a.ref, "cache": a.cache,
           "gt_cache_stats": cache["stats"], "subset": bool(a.allow_subset)}
    res.update(score(a.mode, d["records"], ref, cache))
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    head = res[f"{a.mode}|covered|construction"]["wc"]["All"]["R"]
    print(f"[score] {a.dump} -> {a.out} ({res['seconds']}s) covered WC All R@20 {head[20]:.4f}")


def cmd_gate_scorer(a):
    """Published dumps + published cache through ``score`` == rescore/<exp>.json, cell for cell."""
    pdump, sdump = cg.DUMPS[a.exp]
    cache = pickle.load(open(PUBLISHED_CACHE, "rb"))
    prec, srec = load(pdump), load(sdump)
    mine = {}
    mine.update(score("predcls", prec, prec, cache))
    mine.update(score("sgdet", srec, prec, cache))
    theirs = json.load(open(f"{PUBLISHED_SCORES}/{a.exp}.json"))
    diff, n = 0.0, 0
    for name in PRED_NAMES + SG_NAMES:
        for c in mine[name]:
            for b in cg.BUCKETS:
                for met in ("R", "mR", "mR_supported", "pooled_R", "pooled_mR"):
                    for k in cg.KS:
                        x, y = mine[name][c][b][met][k], theirs[name][c][b][met][str(k)]
                        if x is None or y is None:
                            assert x is None and y is None, (name, c, b, met, k)
                            continue
                        diff = max(diff, abs(x - y))
                        n += 1
    print(f"[gate-scorer] {a.exp}: {n} cells, max |diff| = {diff:.3e}")
    sys.exit(0 if diff == 0.0 else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gt-ref")
    p.add_argument("--feature_model", required=True)
    p.add_argument("--slot_alignment", default="per_video_class")
    p.add_argument("--inventory", default="feature", choices=["feature", "full"])
    p.add_argument("--data_path", default="/data/rohith/ag")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build-gt")
    p.add_argument("--ref", required=True)
    p.add_argument("--cache", required=True)
    p = sub.add_parser("gate-ref")
    p.add_argument("--ref", required=True)
    p.add_argument("--published", required=True)
    p.add_argument("--exact", action="store_true")
    p.add_argument("--cache", default=None)
    p.add_argument("--published_cache", default=PUBLISHED_CACHE)
    p.add_argument("--out", default=None)
    p = sub.add_parser("gate-scorer")
    p.add_argument("--exp", default="w_sttran")
    p = sub.add_parser("score")
    p.add_argument("--mode", choices=["predcls", "sgdet"], required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow_subset", action="store_true")
    a = ap.parse_args()
    {"gt-ref": cmd_gt_ref, "build-gt": cmd_build_gt, "gate-ref": cmd_gate_ref, "gate-scorer": cmd_gate_scorer,
     "score": cmd_score}[a.cmd](a)
