"""INFER: score a new full-split prediction dump with the common-GT rules (CPU only).

Same ground truth and rules as ``tools/rebuttal/rescore_rebuttal.py`` (whose functions are reused): covered GT
(the PredCls ground truth of the common reference dump, ``rescore_common_gt.GT_REFERENCE``) and the full annotated
inventory (``rescore_rebuttal.GT_CACHE``), With-Constraint in construction order and sorted by confidence, No
Constraint, buckets All / OO / UO, per-frame R / mR (paper mR = sum over predicates / 26).

The PredCls GT of a frame is taken from the PredCls dump of the SAME feature family as the scored dump
(``rescore_common_gt.DUMPS[<gt row>][0]``, aligned by (video_id, frame_t)), exactly as ``rescore_common_gt.run``
pairs each SGDet dump with its own PredCls dump: the relation GT is identical across families (checked here against
the common reference ``GT_REFERENCE`` on every frame), but the PredCls slot boxes ``bboxes_2d`` that SGDet slots are
matched to at 2-D IoU >= 0.5 live in each feature family's own box space (ResNet-50 and DINOv3 PKLs differ in every
frame), so a DINOv3 SGDet dump scored against ResNet-50 PredCls boxes loses most visible matches. GT row: ``--gt-row``,
else ``--gate-exp``, else the Table 2 row of the dump's experiment (ablations -> worldwise_dinov3l), else by the
feature model (resnet50 -> w_sttran, dinov3l -> worldwise_dinov3l, dinov3_tok -> worldwise_plus_dinov3tok). The dump being scored may be a PredCls or an SGDet dump of the 1,511-video
test split in loader order.

  * PredCls dump: its GT keys (pairs, classes, labels, visibility) are checked against the reference frame by frame
    (any difference is a failed gate); scorers ``predcls|{covered,full}|{construction,sorted}``.
  * SGDet dump: ``sgdet|covered|{construction,sorted}``, ``sgdet|full_strict|sorted``,
    ``sgdet|full_rule|{construction,sorted}`` (see ``rescore_rebuttal`` for the two full variants).

``--gate-exp <exp>``: the covered/construction cells must reproduce ``common_gt/results/<exp>.json`` (the saved
dump's numbers); the max |diff| is written to the output JSON.

    WSGG_REPO=~/CODE/Scene4Cast_rebuttal $PY tools/rebuttal/infer_score.py --dump <pkl> --name <name> \
        [--gate-exp w_sttran] --out /data3/rohith/ag/runs/rebuttal/infer/scores
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.setdefault("WSGG_REPO", os.path.dirname(os.path.dirname(HERE)))
for p in (REPO, os.path.join(REPO, "tools"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import rescore_common_gt as cg  # noqa: E402
import rescore_rebuttal as rr  # noqa: E402

GT_KEYS = ("pair_valid", "person_idx", "object_idx", "object_classes", "gt_attention", "gt_spatial",
           "gt_contacting", "valid_mask", "visibility_mask")


FAMILY_ROW = {"resnet50": "w_sttran", "dinov3l": "worldwise_dinov3l", "dinov3_tok": "worldwise_plus_dinov3tok"}


def default_gt_row(meta):
    """GT row for a dump: the Table 2 row of its own experiment when there is one, else its feature family."""
    exp = meta.get("experiment", "")
    for prefix, row in (("worldwise_pp_dinov3_", "worldwise_pp_dinov3"), ("worldformer_c1_dinov3tok_", "worldwise_plus_dinov3tok"),
                        ("worldwise_plus_dinov3tok_", "worldwise_plus_dinov3tok"), ("worldwise_v2e_", "worldwise_dinov3l"),
                        ("worldwise_abl_", "worldwise_dinov3l"), ("w_sttran_pp_", "w_sttran_pp"), ("w_sttran_", "w_sttran"),
                        ("w_dsgdetr_pp_", "w_dsgdetr_pp"), ("w_dsgdetr_", "w_dsgdetr"), ("w_usg_", "w_usg")):
        if exp.startswith(prefix) and (row != "worldwise_dinov3l" or meta.get("feature_model") == "dinov3l"):
            return row
    return FAMILY_ROW[meta["feature_model"]]


REL_KEYS = ("pair_valid", "person_idx", "object_idx", "object_classes", "gt_attention", "gt_spatial",
            "gt_contacting", "valid_mask", "visibility_mask")


def load_reference(gt_row=None):
    """(GT records of the gt row's PredCls dump, full-GT cache, (video, frame_t) -> index). The relation GT of the
    gt row is checked against the common reference on every frame."""
    ref = cg.load(cg.GT_REFERENCE)
    with open(rr.GT_CACHE, "rb") as f:
        cache = pickle.load(f)
    assert len(ref) == len(cache["keys"]), (len(ref), len(cache["keys"]))
    gt = ref
    if gt_row is not None and cg.DUMPS[gt_row][0] != cg.GT_REFERENCE:
        gt = cg.load(cg.DUMPS[gt_row][0])
        assert len(gt) == len(ref), (gt_row, len(gt), len(ref))
        for i, (a, b) in enumerate(zip(gt, ref)):
            assert (a["video_id"], int(a["frame_t"])) == cache["keys"][i], (gt_row, i)
            for k in REL_KEYS:
                assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), (gt_row, i, k)
    index = {k: i for i, k in enumerate(cache["keys"])}
    return gt, cache, index


def input_visibility_stats(recs):
    st = defaultdict(int)
    for r in recs:
        vm = r["valid_mask"].astype(bool)
        vis = r["visibility_mask"].astype(bool)
        st["valid_slots"] += int(vm.sum())
        st["unobserved_slots"] += int((vm & ~vis).sum())
    return dict(st)


def score(dump, name, out_dir, gate_exp=None, full_split=True, gt_row=None):
    t0 = time.time()
    with open(dump, "rb") as f:
        d = pickle.load(f)
    meta, recs = d.get("meta", {}), d["records"]
    mode = meta.get("mode") or ("sgdet" if "pred_labels" in recs[0] else "predcls")
    gt_row = gt_row or gate_exp or default_gt_row(meta)
    ref, cache, index = load_reference(gt_row)
    if full_split:
        assert len(recs) == len(ref), f"{len(recs)} records, expected {len(ref)} (1,511-video split)"
    if mode == "predcls":
        names = ["predcls|covered|construction", "predcls|covered|sorted", "predcls|full|construction",
                 "predcls|full|sorted"]
    else:
        names = ["sgdet|covered|construction", "sgdet|covered|sorted", "sgdet|full_strict|sorted",
                 "sgdet|full_rule|construction", "sgdet|full_rule|sorted"]
    sc = {n: cg.Scorer() for n in names}
    st = defaultdict(int)
    gt_mismatch = defaultdict(int)
    for s in recs:
        i = index[(s["video_id"], int(s["frame_t"]))]
        g, ex, dr = ref[i], cache["extras"][i], set(cache["drops"][i])
        rels, tags = cg.ground_truth(g)
        frels = [r for gi, r in enumerate(rels) if gi not in dr] + [e[0] for e in ex]
        ftags = [t for gi, t in enumerate(tags) if gi not in dr] + [e[1] for e in ex]
        nc = cg.ranked_rows(s, "no")
        con, srt = rr.wc_rows(s, "construction"), rr.wc_rows(s, "sorted")
        if mode == "predcls":
            for k in GT_KEYS:
                if not np.array_equal(np.asarray(s[k]), np.asarray(g[k])):
                    gt_mismatch[k] += 1
            ident = {j: j for j in np.where(g["valid_mask"].astype(bool))[0].tolist()}
            if rels:
                sc["predcls|covered|construction"].add(rels, tags, {"wc": con, "nc": nc}, ident)
                sc["predcls|covered|sorted"].add(rels, tags, {"wc": srt}, ident)
            if frels:
                sc["predcls|full|construction"].add(frels, ftags, {"wc": con, "nc": nc}, ident)
                sc["predcls|full|sorted"].add(frels, ftags, {"wc": srt}, ident)
        else:
            m_cov = cg.slot_map_common(s, g, st) if rels or frels else {}
            if rels:
                sc["sgdet|covered|construction"].add(rels, tags, {"wc": con, "nc": nc}, m_cov)
                sc["sgdet|covered|sorted"].add(rels, tags, {"wc": srt}, m_cov)
            if frels:
                m_full = rr.slot_map_full(s, g, ex)
                st["slots_mapped_to_uncovered"] += sum(1 for v in m_full.values() if v >= rr.PSEUDO)
                sc["sgdet|full_strict|sorted"].add(frels, ftags, {"wc": srt}, m_cov)
                sc["sgdet|full_rule|construction"].add(frels, ftags, {"wc": con, "nc": nc}, m_full)
                sc["sgdet|full_rule|sorted"].add(frels, ftags, {"wc": srt}, m_full)
    res = {"name": name, "dump": dump, "meta": meta, "mode": mode, "n_records": len(recs),
           "gt_row": gt_row, "gt_dump": cg.DUMPS[gt_row][0], "gt_reference": cg.GT_REFERENCE, "gt_cache": rr.GT_CACHE, "slot_stats": dict(st),
           "input_visibility": input_visibility_stats(recs), "predcls_gt_key_mismatch_frames": dict(gt_mismatch)}
    for n, s_ in sc.items():
        summ = s_.summary()
        if n.endswith("sorted"):
            summ.pop("nc")
        res[n] = summ
    if gate_exp:
        saved = json.load(open(f"{cg.R}/common_gt/results/{gate_exp}.json"))
        mine, theirs = (("predcls|covered|construction", "predcls") if mode == "predcls"
                        else ("sgdet|covered|construction", "sgdet_common"))
        diff = 0.0
        for c in ("wc", "nc"):
            for b in cg.BUCKETS:
                for met in ("R", "mR", "pooled_R", "pooled_mR"):
                    for k in cg.KS:
                        a, bb = res[mine][c][b][met][k], saved[theirs][c][b][met][str(k)]
                        if a is None or bb is None:
                            if not (a is None and bb is None):
                                diff = float("inf")
                            continue
                        diff = max(diff, abs(a - bb))
        res["gate_exp"] = gate_exp
        res["gate_max_abs_diff_vs_common_gt"] = diff
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    msg = f"[{name}] {mode} n={len(recs)} {res['seconds']}s -> {out}"
    if gate_exp:
        msg += f"  gate max|diff| vs common_gt/{gate_exp} = {res['gate_max_abs_diff_vs_common_gt']:.2e}"
    if gt_mismatch:
        msg += f"  PREDCLS GT MISMATCH {dict(gt_mismatch)}"
    print(msg)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="/data3/rohith/ag/runs/rebuttal/infer/scores")
    ap.add_argument("--gate-exp", default=None, help="key of rescore_common_gt.DUMPS whose saved JSON must match")
    ap.add_argument("--subset", action="store_true", help="allow a dump of a subset of the 1,511 videos")
    ap.add_argument("--gt-row", default=None, help="key of rescore_common_gt.DUMPS whose PredCls dump gives the GT")
    a = ap.parse_args()
    r = score(a.dump, a.name, a.out, a.gate_exp, full_split=not a.subset, gt_row=a.gt_row)
    if a.gate_exp and r["gate_max_abs_diff_vs_common_gt"] > 1e-3:
        print("GATE FAIL")
        sys.exit(1)
    if r["predcls_gt_key_mismatch_frames"]:
        print("GATE FAIL (PredCls GT differs from the reference)")
        sys.exit(1)
