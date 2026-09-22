"""
B4: build the annotation-INDEPENDENT caches for the worldbbox test videos.

    python -m lib.mllm.tools.build_caches --what bev,detections,lift --workers 6 [--limit N]

* ``bev``        : voxel cloud + base top-down render per video (lib/mllm/tools/bev.py)
* ``detections`` : normalised GDino detections per sampled frame (tools/detections.py)
* ``lift``       : 2D->3D OBB lift of every GDino detection on annotated frames
                   (tools/lift.py) -- the sgdet perception the tracks use
                   (annotated-frame *list* comes from frames_annotated/, not from
                   the annotation PKL, so this stays annotation-independent)

Each cache is registered in the manifest (tools/cache_manifest.py) with
``annotation_version=None``.  CPU only; safe to run next to the GPU queue.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

logger = logging.getLogger("build_caches")


def _paths(cfg):
    from lib.mllm.core.config_loader import get_path
    return {
        "bev": get_path(cfg, "worldbbox.caches.bev"),
        "detections": get_path(cfg, "worldbbox.caches.detections"),
        "lift": get_path(cfg, "worldbbox.caches.lifted3d"),
        "gdino": get_path(cfg, "annotations.detection_dynamic"),
    }


def _one(video_id: str, what: List[str], rebuild: bool) -> Dict[str, Any]:
    import numpy as np  # noqa
    from lib.mllm.core.config_loader import load_config
    from lib.mllm.data.worldbbox import WorldBBoxTestSet
    cfg = load_config()
    P = _paths(cfg)
    ts = WorldBBoxTestSet(cfg)
    v = ts.load(video_id)
    out: Dict[str, Any] = {"video": video_id}
    t0 = time.time()
    try:
        if "bev" in what:
            from lib.mllm.tools.bev import get_bev
            img, meta, cloud = get_bev(v, P["bev"], rebuild=rebuild)
            out["bev"] = {"size": img.size, "points": int(len(cloud["xyz"]))}
        if "detections" in what or "lift" in what:
            from lib.mllm.tools.detections import get_detections
            d = get_detections(v, P["gdino"], P["detections"], rebuild=rebuild)
            out["detections"] = {"frames": len(d["frames"]),
                                 "boxes": int(sum(len(x) for x in d["frames"].values()))}
        if "lift" in what:
            from lib.mllm.tools.lift import LiftCache
            lc = LiftCache(v, P["lift"])
            ann_dir = Path(ts.wb["frames_annotated"]) / v.vid_mp4
            ann_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".png")) if ann_dir.is_dir() else []
            n = n_ok = 0
            for ff in ann_files:
                for det in d["frames"].get(ff, []):
                    r = lc.get(ff, det["label"], det["bbox_pi3"])
                    n += 1
                    n_ok += r is not None
            lc.save()
            out["lift"] = {"boxes": n, "lifted": n_ok}
        out["seconds"] = round(time.time() - t0, 1)
    except Exception as e:  # noqa
        out["error"] = f"{e!r}\n{traceback.format_exc()[-800:]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="bev,detections,lift")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--status", default="/data3/rohith/ag/logs/b4_caches.status.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    what = [w.strip() for w in args.what.split(",") if w.strip()]
    from lib.mllm.core.config_loader import load_config
    from lib.mllm.data.worldbbox import WorldBBoxTestSet
    cfg = load_config()
    ts = WorldBBoxTestSet(cfg)
    ids = ts.video_ids[: args.limit] if args.limit else ts.video_ids
    P = _paths(cfg)
    t0 = time.time()
    done, errors = 0, []
    agg: Dict[str, Any] = {"bev_points": 0, "det_boxes": 0, "lift_boxes": 0, "lift_ok": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, vid, what, args.rebuild): vid for vid in ids}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if "error" in r:
                errors.append((r["video"], r["error"]))
                logger.error(f"[{r['video']}] {r['error'][:300]}")
            else:
                agg["bev_points"] += r.get("bev", {}).get("points", 0)
                agg["det_boxes"] += r.get("detections", {}).get("boxes", 0)
                agg["lift_boxes"] += r.get("lift", {}).get("boxes", 0)
                agg["lift_ok"] += r.get("lift", {}).get("lifted", 0)
            if done % 50 == 0 or done == len(ids):
                logger.info(f"{done}/{len(ids)} videos, {len(errors)} errors, {time.time() - t0:.0f}s")
                with open(args.status, "w") as f:
                    json.dump({"state": "running" if done < len(ids) else "done", "done": done,
                               "total": len(ids), "errors": len(errors), "agg": agg,
                               "seconds": round(time.time() - t0)}, f)
    # manifest registration (annotation-independent)
    try:
        from tools.cache_manifest import register
        commit_note = f"built by lib/mllm/tools/build_caches.py over {len(ids)} videos; {len(errors)} errors"
        if "bev" in what:
            register("mllm/bev", P["bev"], inputs=["pi3_dynamic/predictions.npz", "world_to_final"],
                     annotation_version=None, note="voxel cloud + base BEV render, canonical frame; " + commit_note)
        if "detections" in what:
            register("mllm/detections", P["detections"], inputs=["detection/gdino_bboxes"],
                     annotation_version=None, note="GDino boxes normalised to AG labels; " + commit_note)
        if "lift" in what:
            register("mllm/lifted3d", P["lift"], inputs=["mllm/detections", "pi3_dynamic/predictions.npz"],
                     annotation_version=None, note="2D->3D OBB lifts of GDino boxes on annotated frames; " + commit_note)
    except Exception as e:  # noqa
        logger.warning(f"manifest registration failed: {e!r}")
    with open(args.status, "w") as f:
        json.dump({"state": "done", "done": done, "total": len(ids), "errors": errors[:50],
                   "n_errors": len(errors), "agg": agg, "seconds": round(time.time() - t0)}, f, indent=1)
    logger.info(f"done: {agg}, errors={len(errors)}")


if __name__ == "__main__":
    main()
