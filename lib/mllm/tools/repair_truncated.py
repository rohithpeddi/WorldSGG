"""
Re-parse Track A/B run pickles whose responses were cut off at ``max_new_tokens``.

The generation is kept; only the parsing is redone with the salvage path added to
``extract_json`` (complete entries before the cut are recovered).  The per-frame
object list is rebuilt from the annotations exactly as ``build_payload`` orders it,
so ids in the stored response map to the same objects as during generation; a
video whose rebuilt ids differ from the stored ``ids`` map is left untouched.

    python -m lib.mllm.tools.repair_truncated --method track_a --mode predcls \
        --model qwen3vl_8b [--out_model qwen3vl_8b_repaired] [--limit N]
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.mllm.core.config_loader import get_path, load_config            # noqa: E402
from lib.mllm.data.worldbbox import WorldBBoxTestSet                     # noqa: E402
from lib.mllm.methods.track_a_prompt.runner import (                          # noqa: E402
    TrackAContext, extract_json, obb_dict, parse_objects,
)

logger = logging.getLogger("repair_truncated")


def frame_payload(fr, ctx=None):
    """The object list in build_payload's order (id = position); images are not needed
    to re-parse a stored response.  predcls: the frame's GT objects deduped by label;
    sgdet: the GDino proposals by descending score with their lifted 3D boxes."""
    objs = []
    if ctx is None or ctx.mode == "predcls":
        seen = set()
        for o in fr.objects:
            if o.label in seen:
                continue
            seen.add(o.label)
            objs.append({"id": len(objs) + 1, "label": o.label,
                         "bbox": None if o.bbox_2d is None else o.bbox_2d.tolist(),
                         "corners": o.corners_final, "obb": obb_dict(o.corners_final),
                         "visible": o.observed})
        return {"objects": objs}
    import numpy as np
    dets = ctx.dets["frames"].get(fr.file, [])
    for d in sorted((d for d in dets if d["label"] != "person"), key=lambda d: -d["score"]):
        r = ctx.lift.get(fr.file, d["label"], d["bbox_pi3"])
        corners = None if r is None else np.asarray(r["corners"])
        objs.append({"id": len(objs) + 1, "label": d["label"], "bbox": d["bbox"], "score": d["score"],
                     "corners": corners, "obb": obb_dict(corners), "visible": True})
    return {"objects": objs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="track_a", choices=["track_a", "track_b"])
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--out_model", default=None, help="write to this model dir (default: in place)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    src = Path(get_path(cfg, f"outputs.{args.method}")) / args.mode / args.model
    dst = src if not args.out_model else Path(get_path(cfg, f"outputs.{args.method}")) / args.mode / args.out_model
    dst.mkdir(parents=True, exist_ok=True)
    ts = WorldBBoxTestSet(cfg)
    files = sorted(src.glob("*.pkl"))
    if args.limit:
        files = files[:args.limit]
    t0 = time.time()
    n_vid = n_rep = n_skip_ids = 0
    f_before = f_after = o_before = o_after = 0
    for i, f in enumerate(files):
        rec = pickle.load(open(f, "rb"))
        vid = rec["video_id"].replace(".mp4", "")
        try:
            video = ts.load(vid)
        except Exception as e:                                      # noqa
            logger.warning(f"[{vid}] load failed: {e!r}")
            continue
        ctx = None
        if args.mode == "sgdet":
            ctx = TrackAContext(video, cfg, args.mode)
        payloads = {fr.file: frame_payload(fr, ctx) for fr in video.frames}
        changed = False
        for file, fdata in rec["frames"].items():
            had = bool(fdata.get("objects"))
            f_before += had
            o_before += len(fdata.get("objects") or {})
            if had or not fdata.get("raw_response"):
                o_after += len(fdata.get("objects") or {})
                f_after += had
                continue
            p = payloads.get(file)
            if p is None:
                continue
            ids = {o["id"]: o["label"] for o in p["objects"]}
            if fdata.get("ids") and {int(k): v for k, v in fdata["ids"].items()} != ids:
                n_skip_ids += 1
                continue
            d = extract_json(fdata["raw_response"])
            objs = parse_objects(d, p, args.mode)
            if objs:
                fdata["objects"] = objs
                fdata["repaired"] = True
                changed = True
            f_after += bool(objs)
            o_after += len(objs)
        if changed:
            rec["n_parsed"] = sum(1 for v in rec["frames"].values() if v.get("objects"))
            rec["repaired"] = True
            out = dst / f.name
            tmp = out.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(rec, fh)
            os.replace(tmp, out)
            n_rep += 1
        elif dst is not src:
            shutil.copy2(f, dst / f.name)
        n_vid += 1
        if (i + 1) % 200 == 0:
            logger.info(f"{i + 1}/{len(files)} videos, {n_rep} repaired, {time.time() - t0:.0f}s")
    logger.info(f"done: {n_vid} videos, {n_rep} repaired, {n_skip_ids} frames skipped (id mismatch), "
                f"{time.time() - t0:.0f}s")
    logger.info(f"frames with objects {f_before} -> {f_after};  object predictions {o_before} -> {o_after}")


if __name__ == "__main__":
    main()
