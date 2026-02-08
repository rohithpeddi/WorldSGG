#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Set

# Adjust the import paths if your project structure differs.
from dataloader.standard.action_genome.ag_dataset import StandardAG


def _extract_video_ids(video_list: List[List[str]]) -> List[str]:
    ids: Set[str] = set()
    for frames in video_list:
        if not frames:
            continue
        # frame relpath looks like: video_id/frame_num.jpg (or similar)
        video_id = frames[0].split("/")[0]
        ids.add(video_id)
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/data/rohith/ag/video_splits.json",
        help="Output JSON path.",
    )

    args = parser.parse_args()

    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path="/data/rohith/ag",
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    test_video_ids = _extract_video_ids(test_dataset.video_list)

    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path="/data/rohith/ag",
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    train_video_ids = _extract_video_ids(train_dataset.video_list)

    out_obj = {
        "train": train_video_ids,
        "test": test_video_ids,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"Wrote {args.out} (train={len(out_obj['train'])}, test={len(out_obj['test'])})")


if __name__ == "__main__":
    main()
