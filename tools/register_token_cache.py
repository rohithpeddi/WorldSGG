#!/usr/bin/env python3
"""
Register a finished token-cache split in the cache manifest (annotation-independent).

Checks that every video of the split has an npz (both shards' done lists), reports
sizes, and registers ``tokens/<stream>/<split>``.

    python tools/register_token_cache.py --stream dinov3l --split test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.preprocess.tokens.token_common import TOKEN_CACHE_ROOT, list_videos  # noqa: E402
from tools.cache_manifest import register  # noqa: E402

INPUTS = {
    "dinov3l": ["frames/<video>.mp4/<frames_annotated>", "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "features/roi_features/{predcls,sgdet}/dinov3l/<split> (Tier-1 boxes)"],
    "pi3": ["frames/<video>.mp4 (sampled_frames_idx clip, ag_pi3.py order)", "yyfz233/Pi3",
            "features/roi_features/{predcls,sgdet}/dinov3l/<split> (Tier-1 boxes)"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=["dinov3l", "pi3"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--force", action="store_true", help="register even if videos are missing")
    args = ap.parse_args()

    base = TOKEN_CACHE_ROOT / args.stream / args.split
    videos = list_videos(args.split)
    missing = [v for v in videos if not (base / f"{v}.npz").exists()]
    total = sum(p.stat().st_size for p in base.glob("*.npz"))
    print(f"{args.stream}/{args.split}: {len(videos) - len(missing)}/{len(videos)} videos cached, "
          f"{total / 1e9:.1f} GB, missing {len(missing)}")
    if missing[:10]:
        print("  e.g.", missing[:10])
    if missing and not args.force:
        sys.exit(1)
    register(f"tokens/{args.stream}/{args.split}", str(base), inputs=INPUTS[args.stream],
             annotation_version=None,
             note=f"{len(videos) - len(missing)} videos, {total / 1e9:.1f} GB; missing={len(missing)}")
    print("registered")


if __name__ == "__main__":
    main()
