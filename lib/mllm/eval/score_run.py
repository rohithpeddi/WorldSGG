"""
Score one MLLM run on the worldbbox test set (B2 driver; B7 loops over runs).
=============================================================================

    python -m lib.mllm.eval.score_run --method zero_shot --model qwen25vl_7b --mode predcls \
        [--pred_dir <override>] [--out <json>] [--limit N]

Regimes reported (all on the locked 1,511-video split; videos without a run
output count as missing and are listed, not silently dropped):

  * ``wsgg``      WorldSGG protocol via the stock evaluator on the dump records:
                  R/mR/hR@{10,20,50,100}, with ("wc") and no ("nc") constraint,
                  all frames, plus the OO/OU bucket breakdown
                  (``evaluation_recall_bucketed``).            [predcls only]
  * ``loc3d``     lib/mllm/eval/recall3d.py: class match + oriented 3D-IoU on
                  OBB corners at thresholds {0 (unlocalised), 0.15, 0.25};
                  predcls uses GT corners for predictions (== relation recall
                  restricted to emitted pairs), sgdet needs emitted OBBs.
  * ``legacy``    vendored box-free P/R/F1 (U-WSGG) on wsg_2d_augmentations
                  restricted to the split.

The dump records are also written to ``<outputs.dumps>/<method>__<model>__<mode>.pkl``
(same container as tools/dump_predictions.py) so tools/bucketed_breakdown.py
and any future evaluator can re-slice them offline.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.mllm.core.config_loader import get_path, load_config              # noqa: E402
from lib.mllm.data.worldbbox import (                                       # noqa: E402
    ATTENTION_RELATIONSHIPS, CONTACTING_RELATIONSHIPS, OBJECT_CLASSES,
    SPATIAL_RELATIONSHIPS, WorldBBoxTestSet,
)
from lib.mllm.eval.dump_adapter import build_records, load_run_pkl, prediction_lookup  # noqa: E402
from lib.mllm.eval.recall3d import evaluate_record_3d                       # noqa: E402
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator, evaluate_wsgg_video  # noqa: E402
from lib.supervised.evaluation_recall_bucketed import (                     # noqa: E402
    BucketAccumulator, evaluate_wsgg_video_bucketed,
)

logger = logging.getLogger("score_run")
PRED_NAMES = list(ATTENTION_RELATIONSHIPS) + list(SPATIAL_RELATIONSHIPS) + list(CONTACTING_RELATIONSHIPS)
KS = (10, 20, 50, 100)


def make_evaluator(mode: str, constraint: str) -> BasicSceneGraphEvaluator:
    return BasicSceneGraphEvaluator(
        mode=mode, AG_object_classes=OBJECT_CLASSES, AG_all_predicates=PRED_NAMES,
        AG_attention_predicates=list(ATTENTION_RELATIONSHIPS),
        AG_spatial_predicates=list(SPATIAL_RELATIONSHIPS),
        AG_contacting_predicates=list(CONTACTING_RELATIONSHIPS),
        iou_threshold=0.5, save_file=os.devnull, constraint=constraint)


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def _stats(ev: BasicSceneGraphEvaluator) -> Dict[str, Any]:
    s = ev.fetch_stats_json()
    return {"R": {str(k): _r(s["recall"].get(k)) for k in KS},
            "mR": {str(k): _r(s["mean_recall"].get(k)) for k in KS},
            "hR": {str(k): _r(s["harmonic_mean_recall"].get(k)) for k in KS},
            "n_frames": len(ev.result_dict[ev.mode + "_recall"][KS[0]])}


def _buckets(acc: BucketAccumulator) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sizes": acc.bucket_sizes(), "buckets": {}}
    for b in ("OO", "OU", "UO", "UU"):
        if b not in acc.total:
            continue
        e: Dict[str, Any] = {"R": {}, "mR": {}, "n_gt": int(acc.total[b].sum())}
        for k in KS:
            r, _ = acc.recall(b, k)
            m, _ = acc.mean_recall(b, k)
            e["R"][str(k)] = _r(r)
            e["mR"][str(k)] = _r(m)
        out["buckets"][b] = e
    if "OU" in acc.total:
        nt: Dict[str, Any] = {}
        for name, kw in (("drop_trivial", dict(drop_trivial=True)),
                         ("drop_trivial_and_unsure", dict(drop_trivial=True, drop_unsure=True))):
            e = {"R": {}, "mR": {}, "n_gt": 0}
            for k in KS:
                r, t = acc.recall("OU", k, **kw)
                m, _ = acc.mean_recall("OU", k, **kw)
                e["R"][str(k)] = _r(r)
                e["mR"][str(k)] = _r(m)
                e["n_gt"] = t
            nt[name] = e
        out["ou_nontrivial"] = nt
    return out


def collect_records(model: str, mode: str, pred_dir: str, limit: int = 0, cfg=None,
                    video_list: str = None) -> Dict[str, Any]:
    ts = WorldBBoxTestSet(cfg)
    ids = ts.video_ids
    if video_list:
        keep = {Path(l.strip()).stem for l in open(video_list, encoding="utf-8") if l.strip()}
        ids = [v for v in ids if v in keep]
    ids = ids[:limit] if limit else ids
    pkl_dir = Path(pred_dir) / mode / model
    records: List[Dict[str, Any]] = []
    missing, errors = [], []
    t0 = time.time()
    for i, vid in enumerate(ids):
        p = pkl_dir / f"{vid}.mp4.pkl"
        if not p.exists():
            p = pkl_dir / f"{vid}.pkl"
        if not p.exists():
            missing.append(vid)
            continue
        try:
            preds = prediction_lookup(load_run_pkl(str(p)))
            records.extend(build_records(ts.load(vid), preds, mode=mode))
        except Exception as e:  # noqa
            errors.append((vid, repr(e)))
            logger.exception(f"[{vid}] failed")
        if (i + 1) % 200 == 0:
            logger.info(f"{i + 1}/{len(ids)} videos, {len(records)} frames, {time.time() - t0:.0f}s")
    return {"records": records, "missing": missing, "errors": errors,
            "n_videos": len(ids) - len(missing) - len(errors), "n_split": len(ids), "pred_dir": str(pkl_dir)}


def score_records(records: List[Dict[str, Any]], mode: str, iou_thrs=(0.0, 0.15, 0.25)) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if mode == "predcls":
        w: Dict[str, Any] = {}
        for cname, cval in (("wc", "with"), ("nc", "no")):
            ev = make_evaluator(mode, cval)
            acc = BucketAccumulator(ks=KS)
            for r in records:
                evaluate_wsgg_video(r, ev, mode=mode, verbose=False)
                evaluate_wsgg_video_bucketed(r, acc, ev, mode=mode)
            w[cname] = {**_stats(ev), "buckets": _buckets(acc)}
        out["wsgg"] = w
    loc: Dict[str, Any] = {}
    for thr in iou_thrs:
        key = "unloc" if thr <= 0 else f"iou{thr:g}"
        blk: Dict[str, Any] = {}
        for cname, cval in (("wc", "with"), ("nc", "no")):
            ev = make_evaluator(mode, cval)
            acc = BucketAccumulator(ks=KS)
            for r in records:
                evaluate_record_3d(r, ev, iou_thr=thr, mode=mode, acc=acc)
            blk[cname] = {**_stats(ev), "buckets": _buckets(acc)}
        loc[key] = blk
    out["loc3d"] = loc
    out["coverage"] = {
        "frames": len(records),
        "gt_pairs": int(sum(r["pair_valid"].sum() for r in records)),
        "pred_pairs": int(sum(r["pred_pair_valid"].sum() for r in records)),
        "gt_pairs_with_pred": int(sum((r["pair_valid"] & r["pred_pair_valid"]).sum() for r in records)),
        "object_slots": int(sum(r["valid_mask"].sum() - 1 for r in records)),
        "object_slots_with_pred_3d": int(sum(r["has_pred_3d"][1:].sum() for r in records)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="zero_shot | caption_all | rag_all | wsg_agent | track_a | track_b")
    ap.add_argument("--model", default="qwen25vl_7b")
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--pred_dir", default=None, help="run root (default: outputs.<method> from the config)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_legacy", action="store_true")
    ap.add_argument("--no_dump", action="store_true")
    ap.add_argument("--video_list", default=None,
                    help="restrict to these videos (e.g. the 442-video subset with pre-existing Stage-1 graphs)")
    ap.add_argument("--subset_tag", default="", help="suffix for output names when --video_list is used")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    pred_dir = args.pred_dir or get_path(cfg, f"outputs.{args.method}")
    tag = f"{args.method}__{args.model}__{args.mode}"
    suffix = (f"__lim{args.limit}" if args.limit else "") + (f"__{args.subset_tag}" if args.subset_tag else "")
    t0 = time.time()
    col = collect_records(args.model, args.mode, pred_dir, args.limit, cfg, video_list=args.video_list)
    logger.info(f"{tag}: {col['n_videos']}/{col['n_split']} videos, {len(col['records'])} frames, "
                f"{len(col['missing'])} missing, {len(col['errors'])} errors")
    res: Dict[str, Any] = {
        "run": {"method": args.method, "model": args.model, "mode": args.mode, "pred_dir": col["pred_dir"]},
        "n_videos": col["n_videos"], "n_split": col["n_split"], "n_missing": len(col["missing"]),
        "missing": col["missing"][:50], "errors": col["errors"][:20],
    }
    if col["records"]:
        res.update(score_records(col["records"], args.mode))
    if not args.no_legacy and not args.limit:
        try:
            from lib.mllm.eval.legacy_f1 import legacy_f1, summarize_legacy
            vids = None
            if args.video_list:
                vids = [Path(l.strip()).stem for l in open(args.video_list, encoding="utf-8") if l.strip()]
            res["legacy"] = summarize_legacy(legacy_f1(pred_dir, args.mode, args.model, cfg, video_ids=vids))
        except Exception as e:  # noqa
            logger.exception("legacy F1 failed")
            res["legacy"] = {"error": repr(e)}
    dumps_dir = get_path(cfg, "outputs.dumps") or "/data3/rohith/ag/runs/mllm/dumps/"
    os.makedirs(dumps_dir, exist_ok=True)
    if not args.no_dump and col["records"]:
        dp = os.path.join(dumps_dir, f"{tag}{suffix}.pkl")
        with open(dp, "wb") as f:
            pickle.dump({"meta": {"experiment": tag, "mode": args.mode, "method": args.method,
                                  "model": args.model, "frames": "all", "n_videos": col["n_videos"],
                                  "n_frames": len(col["records"])},
                         "records": col["records"]}, f, protocol=4)
        res["dump"] = dp
    res["seconds"] = round(time.time() - t0, 1)
    out = args.out or os.path.join(dumps_dir, f"{tag}{suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"wrote {out}")
    if "wsgg" in res:
        for c in ("wc", "nc"):
            s = res["wsgg"][c]
            print(f"[{tag}] wsgg {c}: R@20={s['R']['20']} mR@20={s['mR']['20']} "
                  f"R@50={s['R']['50']} mR@50={s['mR']['50']}")
    for k, v in res.get("loc3d", {}).items():
        print(f"[{tag}] loc3d {k}: wc R@20={v['wc']['R']['20']} mR@20={v['wc']['mR']['20']} | "
              f"nc R@50={v['nc']['R']['50']} mR@50={v['nc']['mR']['50']}")
    if "legacy" in res:
        g = res["legacy"].get("gt_plus_corrections", {})
        print(f"[{tag}] legacy F1 micro={g.get('micro_F1')} macro={g.get('macro_F1')} "
              f"videos={res['legacy'].get('n_videos_evaluated')}")


if __name__ == "__main__":
    main()
