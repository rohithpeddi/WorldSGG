#!/usr/bin/env python3
"""
Move pre-extracted PredCls ROI feature PKLs into train/ and test/ subdirectories.

Usage:
    python datasets/preprocess/features/move_predcls_to_splits.py \
        --data_path /data/rohith/ag \
        --features_dir /data/rohith/ag/features/roi_features/predcls/dinov2b
"""

import argparse
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataloader.ag_dataset import StandardAG


def main():
    parser = argparse.ArgumentParser(description="Move predcls PKLs into train/test splits")
    parser.add_argument("--data_path", type=str, required=True, help="AG dataset root")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory containing flat PKL files (e.g. .../predcls/dinov2b)")
    parser.add_argument("--dry_run", action="store_true", help="Print moves without executing")
    args = parser.parse_args()

    features_dir = args.features_dir

    # Load train/test datasets to get video ID lists
    print("Loading AG dataset to determine train/test splits...")
    train_ds = StandardAG(
        phase="train", mode="predcls", datasize="large",
        data_path=args.data_path,
        filter_nonperson_box_frame=True, filter_small_box=False,
    )
    test_ds = StandardAG(
        phase="test", mode="predcls", datasize="large",
        data_path=args.data_path,
        filter_nonperson_box_frame=True, filter_small_box=False,
    )

    train_videos = set()
    for idx in range(len(train_ds)):
        item = train_ds[idx]
        vid = item["video_id"].replace(".mp4", "")
        train_videos.add(vid)

    test_videos = set()
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        vid = item["video_id"].replace(".mp4", "")
        test_videos.add(vid)

    print(f"  Train videos: {len(train_videos)}")
    print(f"  Test videos:  {len(test_videos)}")

    # Create output dirs
    train_dir = os.path.join(features_dir, "train")
    test_dir = os.path.join(features_dir, "test")
    if not args.dry_run:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

    # Scan PKL files in the flat directory
    pkl_files = [f for f in os.listdir(features_dir) if f.endswith(".pkl")]
    print(f"  Found {len(pkl_files)} PKL files in {features_dir}")

    moved_train = 0
    moved_test = 0
    skipped = 0

    for pkl_file in sorted(pkl_files):
        video_stem = pkl_file.replace(".pkl", "")
        src = os.path.join(features_dir, pkl_file)

        if video_stem in train_videos:
            dst = os.path.join(train_dir, pkl_file)
            split = "train"
        elif video_stem in test_videos:
            dst = os.path.join(test_dir, pkl_file)
            split = "test"
        else:
            print(f"  ⚠️  {pkl_file}: not in train or test, skipping")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [DRY RUN] {pkl_file} → {split}/")
        else:
            shutil.move(src, dst)

        if split == "train":
            moved_train += 1
        else:
            moved_test += 1

    action = "Would move" if args.dry_run else "Moved"
    print(f"\n{action}: {moved_train} to train/, {moved_test} to test/, {skipped} skipped")


if __name__ == "__main__":
    main()
