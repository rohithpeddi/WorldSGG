"""
B1 gate: exercise lib/mllm/data/worldbbox.py over the whole locked split.

Checks per video: annotation PKL loads; Pi3 npz exists; the Pi3 frame offset is
recovered by pose matching (not the fallback) and every annotated frame maps to a
Pi3 frame; bbox scale equals WorldAG's _compute_target_size rule; counts of
frames / objects / observed / unobserved / boxes without corners.

    python tools/check_worldbbox_mllm_adapter.py --out /data3/rohith/ag/logs/b1_adapter_check.json
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lib.mllm.data.worldbbox import WorldBBoxTestSet  # noqa: E402


def _target_size(ow, oh, pixel_limit=255000, patch=14):
    import math
    scale = math.sqrt(pixel_limit / (ow * oh))
    wt, ht = ow * scale, oh * scale
    k, m = round(wt / patch), round(ht / patch)
    while (k * patch) * (m * patch) > pixel_limit:
        if k / m > wt / ht:
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch, max(1, m) * patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data3/rohith/ag/logs/b1_adapter_check.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    ts = WorldBBoxTestSet()
    ids = ts.video_ids[: args.limit] if args.limit else ts.video_ids
    tot = dict(videos=0, frames=0, objects=0, observed=0, unobserved=0, no_bbox=0, no_corners=0,
               person_no_corners=0, frames_unmapped=0, videos_fallback_offset=0, videos_no_pi3=0,
               videos_scale_mismatch=0, videos_error=0)
    problems = []
    t0 = time.time()
    for i, vid in enumerate(ids):
        try:
            v = ts.load(vid)
            fr = v.frames
            tot["videos"] += 1
            tot["frames"] += len(fr)
            for f in fr:
                if f.person_corners_final is None:
                    tot["person_no_corners"] += 1
                for o in f.objects:
                    tot["objects"] += 1
                    tot["observed" if o.observed else "unobserved"] += 1
                    tot["no_bbox"] += o.bbox_2d is None
                    tot["no_corners"] += o.corners_final is None
            if not v.pi3_path.exists():
                tot["videos_no_pi3"] += 1
                problems.append((vid, "no_pi3"))
                continue
            # offset by pose voting?
            import logging as _l
            rec = []
            h = _l.Handler(); h.emit = lambda r: rec.append(r.getMessage())
            _l.getLogger("lib.mllm.data.worldbbox").addHandler(h)
            st = v.pi3_start
            _l.getLogger("lib.mllm.data.worldbbox").removeHandler(h)
            if any("annotated window" in m or "inconsistent" in m for m in rec):
                tot["videos_fallback_offset"] += 1
                problems.append((vid, "offset:" + "|".join(rec)[:200]))
            unm = sum(1 for f in fr if v.pi3_index_for_frame(f.frame_num) is None)
            tot["frames_unmapped"] += unm
            if unm:
                problems.append((vid, f"{unm} frames unmapped"))
            ow, oh = v.orig_size
            if tuple(v.pi3_size) != tuple(_target_size(ow, oh)):
                tot["videos_scale_mismatch"] += 1
                problems.append((vid, f"pi3 {v.pi3_size} vs target {_target_size(ow, oh)} orig {(ow, oh)}"))
        except Exception as e:  # noqa
            tot["videos_error"] += 1
            problems.append((vid, f"error: {e!r}"))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    out = {"totals": tot, "problems": problems[:500], "n_problems": len(problems)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(tot, indent=2))
    print(f"problems: {len(problems)} (first 10) {problems[:10]}")


if __name__ == "__main__":
    main()
