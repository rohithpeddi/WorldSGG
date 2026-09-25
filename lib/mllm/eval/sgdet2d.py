"""
Observed-only 2D SGDet recall for MLLM dump records (optional regime, ``score_run --sgdet2d``).
=============================================================================================

For methods that emit 2D boxes (SceneGraphVLM).  Identical GT-triplet construction and
ranking to ``recall3d.evaluate_record_3d`` (with / no constraint, only emitted pairs
produce triplets), with two changes:

1. **GT is restricted to the OO bucket with real boxes**: a GT pair is scored only if the
   object is observed (``visibility_mask``) and both the person and the object carry an
   annotated 2D box (``gt_has_box_2d``).  Unobserved objects have no 2D box, so a 2D
   matcher cannot score them; they are the Full-GT regimes' job.
2. **Matching** = class equality + 2D IoU >= ``iou_thr`` (default 0.5, the stock SGDet
   rule) on both endpoints, predicted boxes from ``pred_boxes_2d`` (Pi-3 space, same as
   ``gt_bboxes_2d``; IoU is invariant to the per-axis scale).  Predicted triplets whose
   subject or object has no box are dropped.

Slots are label-keyed (one per short label per frame, the WorldAG rule), so a method that
names the same class twice contributes one box per class (the first instance the run
stored).  Frames with no scorable GT pair are skipped (not counted as 0).
"""
from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional

import numpy as np

from lib.supervised.evaluation_recall import argsort_desc, intersect_2d
from lib.supervised.evaluation_recall_bucketed import BucketAccumulator, bucket_name


def iou2d(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def evaluate_record_2d(rec: Dict, evaluator, iou_thr: float = 0.5, acc: Optional[BucketAccumulator] = None,
                       max_pred: int = 100) -> Optional[List[List[int]]]:
    if "pred_boxes_2d" not in rec:
        return None
    pair_valid = rec["pair_valid"].astype(bool)
    ppv = rec.get("pred_pair_valid", pair_valid).astype(bool)
    person_idx = rec["person_idx"].astype(int)
    object_idx = rec["object_idx"].astype(int)
    classes = rec["object_classes"].astype(int)
    vis = rec.get("visibility_mask", np.ones(len(classes), bool)).astype(bool)
    gbox_ok = rec["gt_has_box_2d"].astype(bool)
    pbox_ok = rec["has_pred_2d"].astype(bool)
    gt_boxes = rec["gt_bboxes_2d"]
    pred_boxes = rec["pred_boxes_2d"]
    n_att = len(evaluator.AG_attention_predicates)
    n_spa = len(evaluator.AG_spatial_predicates)
    spa_off, con_off = n_att, n_att + n_spa
    num_rel = evaluator.num_rel

    gt_rels, tags = [], []
    for k in np.where(pair_valid)[0]:
        p, o = int(person_idx[k]), int(object_idx[k])
        if not (vis[p] and vis[o] and gbox_ok[p] and gbox_ok[o]):
            continue
        b = bucket_name(True, True)
        a = int(rec["gt_attention"][k])
        gt_rels.append((p, o, a)); tags.append((b, a))
        for s in np.where(rec["gt_spatial"][k] > 0.5)[0]:
            gt_rels.append((o, p, spa_off + int(s))); tags.append((b, spa_off + int(s)))
        for c in np.where(rec["gt_contacting"][k] > 0.5)[0]:
            gt_rels.append((p, o, con_off + int(c))); tags.append((b, con_off + int(c)))
    if not gt_rels:
        return None
    gt_rels = np.array(gt_rels, dtype=np.int64)

    ks = np.where(ppv)[0]
    if len(ks) == 0:
        pred_rels = np.zeros((0, 3), dtype=np.int64)
    else:
        pi, oi = person_idx[ks], object_idx[ks]
        att = rec["attention_distribution"][ks]
        spa = rec["spatial_distribution"][ks]
        con = rec["contacting_distribution"][ks]
        Kv = len(ks)
        n_con = con.shape[1]
        rel_inds = np.concatenate([np.stack([pi, oi], 1), np.stack([oi, pi], 1), np.stack([pi, oi], 1)], 0)
        rel_scores = np.concatenate([
            np.concatenate([att, np.zeros((Kv, n_spa + n_con))], 1),
            np.concatenate([np.zeros((Kv, n_att)), spa, np.zeros((Kv, n_con))], 1),
            np.concatenate([np.zeros((Kv, n_att + n_spa)), con], 1)], 0)
        obj_scores = rec.get("pred_scores", np.ones(len(classes))).astype(np.float64)
        if evaluator.constraint == "no":
            overall = obj_scores[rel_inds].prod(1)[:, None] * rel_scores
            si = argsort_desc(overall)[:max_pred]
            pred_rels = np.column_stack((rel_inds[si[:, 0]], si[:, 1]))
        else:
            keep = rel_scores.max(1) > 0
            pred_rels = np.column_stack((rel_inds[keep], rel_scores[keep].argmax(1)))
        ok = pbox_ok[pred_rels[:, 0]] & pbox_ok[pred_rels[:, 1]]
        pred_rels = pred_rels[ok]

    pred_to_gt: List[List[int]] = [[] for _ in range(len(pred_rels))]
    if len(pred_rels):
        gt_trip = np.column_stack((classes[gt_rels[:, 0]], gt_rels[:, 2], classes[gt_rels[:, 1]]))
        pr_trip = np.column_stack((classes[pred_rels[:, 0]], pred_rels[:, 2], classes[pred_rels[:, 1]]))
        keeps = intersect_2d(gt_trip, pr_trip)
        cache: Dict[tuple, float] = {}

        def iou(g, p):
            key = (int(g), int(p))
            if key not in cache:
                cache[key] = iou2d(gt_boxes[g], pred_boxes[p])
            return cache[key]

        for gi in np.where(keeps.any(1))[0]:
            for pj in np.where(keeps[gi])[0]:
                if iou(gt_rels[gi, 0], pred_rels[pj, 0]) >= iou_thr and \
                        iou(gt_rels[gi, 1], pred_rels[pj, 1]) >= iou_thr:
                    pred_to_gt[pj].append(int(gi))

    rd = evaluator.result_dict
    for k in rd[evaluator.mode + "_recall"]:
        match = reduce(np.union1d, pred_to_gt[:k]) if pred_to_gt[:k] else np.zeros(0)
        hit = [0] * num_rel
        cnt = [0] * num_rel
        for r in gt_rels[:, 2]:
            cnt[int(r)] += 1
        for m in match:
            hit[int(gt_rels[int(m), 2])] += 1
        for n in range(num_rel):
            if cnt[n] > 0:
                rd[evaluator.mode + "_mean_recall_collect"][k][n].append(hit[n] / cnt[n])
        rd[evaluator.mode + "_recall"][k].append(len(match) / gt_rels.shape[0])
    if acc is not None:
        acc.add_frame(tags, pred_to_gt)
    return pred_to_gt
