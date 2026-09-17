"""
Localised recall for MLLM dump records (B2): PredCls-3D and SGDet-3D regimes.
==============================================================================

Same GT-triplet construction, ranking (with / no constraint) and per-frame
recall bookkeeping as ``lib/supervised/evaluation_recall.py::evaluate_from_dict``,
with two changes needed for training-free MLLM predictions:

1. the triplet matcher uses **oriented 3D IoU** on the 8-corner OBBs in the
   canonical frame (``compute_iou_3d_obb``) instead of 2D IoU; at threshold
   ``0`` it degenerates to class matching ("unlocalised" sgdet), and in predcls
   pred corners == GT corners so every class match is a hit;
2. only pairs the model actually emitted (``pred_pair_valid``) produce
   predicted triplets -- an MLLM that never mentions an object should not get
   a free "argmax" triplet for it.

Results accumulate into a stock ``BasicSceneGraphEvaluator`` (``fetch_stats_json``
gives R/mR/hR@K) and, optionally, a ``BucketAccumulator`` (OO/OU split).
"""
from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional

import numpy as np

from lib.detector.monocular3d.evaluation.evaluate_3d import compute_iou_3d_obb
from lib.supervised.evaluation_recall import argsort_desc, intersect_2d
from lib.supervised.evaluation_recall_bucketed import BucketAccumulator, bucket_name

KS = (10, 20, 50, 100)


def _iou3d(c1: np.ndarray, c2: np.ndarray) -> float:
    if not np.any(c1) or not np.any(c2):
        return 0.0
    try:
        return float(compute_iou_3d_obb(c1, c2))
    except Exception:
        return 0.0


def evaluate_record_3d(rec: Dict, evaluator, iou_thr: float = 0.25, mode: str = "sgdet",
                       acc: Optional[BucketAccumulator] = None, require_pred_3d: bool = True,
                       max_pred: int = 100) -> Optional[List[List[int]]]:
    """Score one frame record. ``iou_thr <= 0`` -> class-only matching.

    ``require_pred_3d``: in sgdet, predicted triplets whose subject/object slot
    has no emitted 3D box are dropped when the threshold is > 0.
    """
    pair_valid = rec["pair_valid"].astype(bool)
    ppv = rec.get("pred_pair_valid", pair_valid).astype(bool)
    person_idx = rec["person_idx"].astype(int)
    object_idx = rec["object_idx"].astype(int)
    classes = rec["object_classes"].astype(int)
    gt_corners = rec["gt_corners"]
    pred_corners = rec["pred_corners"] if mode == "sgdet" else gt_corners
    has3d = rec.get("has_pred_3d", np.ones(len(classes), bool)).astype(bool)
    vis = rec.get("visibility_mask", np.ones(len(classes), bool)).astype(bool)
    n_att = len(evaluator.AG_attention_predicates)
    n_spa = len(evaluator.AG_spatial_predicates)
    spa_off, con_off = n_att, n_att + n_spa
    num_rel = evaluator.num_rel

    # ---- GT triplets (+ bucket tags), same order as evaluate_wsgg_video ----
    gt_rels, tags = [], []
    for k in np.where(pair_valid)[0]:
        p, o = int(person_idx[k]), int(object_idx[k])
        b = bucket_name(bool(vis[p]), bool(vis[o]))
        a = int(rec["gt_attention"][k])
        gt_rels.append((p, o, a)); tags.append((b, a))
        for s in np.where(rec["gt_spatial"][k] > 0.5)[0]:
            gt_rels.append((o, p, spa_off + int(s))); tags.append((b, spa_off + int(s)))
        for c in np.where(rec["gt_contacting"][k] > 0.5)[0]:
            gt_rels.append((p, o, con_off + int(c))); tags.append((b, con_off + int(c)))
    if not gt_rels:
        return None
    gt_rels = np.array(gt_rels, dtype=np.int64)

    # ---- predicted triplets: ranking identical to evaluate_from_dict ----
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
        else:  # "with": one predicate per (pair, head) in pair order (stock
            # behaviour: no re-ranking); heads the model left empty emit nothing
            keep = rel_scores.max(1) > 0
            pred_rels = np.column_stack((rel_inds[keep], rel_scores[keep].argmax(1)))
        if mode == "sgdet" and iou_thr > 0 and require_pred_3d:
            ok = has3d[pred_rels[:, 0]] & has3d[pred_rels[:, 1]]
            pred_rels = pred_rels[ok]

    # ---- match: class equality + 3D IoU on both endpoints ----
    pred_to_gt: List[List[int]] = [[] for _ in range(len(pred_rels))]
    if len(pred_rels):
        gt_trip = np.column_stack((classes[gt_rels[:, 0]], gt_rels[:, 2], classes[gt_rels[:, 1]]))
        pr_trip = np.column_stack((classes[pred_rels[:, 0]], pred_rels[:, 2], classes[pred_rels[:, 1]]))
        keeps = intersect_2d(gt_trip, pr_trip)
        iou_cache: Dict[tuple, float] = {}

        def iou(g, p):
            key = (int(g), int(p))
            if key not in iou_cache:
                iou_cache[key] = 1.0 if iou_thr <= 0 else _iou3d(gt_corners[g], pred_corners[p])
            return iou_cache[key]

        for gi in np.where(keeps.any(1))[0]:
            for pj in np.where(keeps[gi])[0]:
                if iou(gt_rels[gi, 0], pred_rels[pj, 0]) >= iou_thr and \
                        iou(gt_rels[gi, 1], pred_rels[pj, 1]) >= iou_thr:
                    pred_to_gt[pj].append(int(gi))

    # ---- recall bookkeeping (evaluate_from_dict) ----
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
