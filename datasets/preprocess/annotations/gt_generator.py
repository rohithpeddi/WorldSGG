#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch

# import your dataset + constants
from dataloader.base_ag_dataset import BaseAG
from constants import Constants as const


def to_serializable(obj):
    """
    Convert anything that np.save doesn't like into something JSON/NPY friendly.
    We keep numpy arrays as-is, convert torch tensors to numpy, and leave scalars/strings.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (int, float, str)) or obj is None:
        return obj
    # for small containers (dict/list) we just return them and handle below
    return obj


def clean_frame_items(frame_items):
    """
    frame_items is the list you built in _build_dataset:
        [
          {PERSON_BOUNDING_BOX: ..., FRAME: ...},
          {<object1 stuff>},
          {<object2 stuff>},
          ...
        ]
    We want the same structure, but with tensors turned into numpy or lists.
    """
    cleaned = []
    for item in frame_items:
        new_item = {}
        for k, v in item.items():
            v = to_serializable(v)
            # if it's still a numpy array of objects, keep it;
            # if it's a numpy array of numbers/tensors it's fine.
            if isinstance(v, np.ndarray):
                new_item[k] = v
            else:
                # torch tensors were turned into numpy already,
                # but some of your fields are torch tensors of ints,
                # so we convert those to list to be safe
                if isinstance(v, (list, tuple)):
                    new_item[k] = v
                else:
                    new_item[k] = v
        cleaned.append(new_item)
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Export per-video GT annotations from Action Genome-style BaseAG dataset to .npy"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Root AG data directory (the one that has 'annotations', 'frames_annotated', etc.)",
    )
    parser.add_argument(
        "--phase",
        default="train",
        choices=["train", "test"],
        help="Which split to load from the PKLs",
    )
    parser.add_argument(
        "--datasize",
        default="full",
        choices=[const.FULL, const.MINI] if hasattr(const, "FULL") else ["full", "mini"],
        help="Dataset size flag used by your BaseAG",
    )
    parser.add_argument(
        "--output_dir",
        default="gt_output",
        help="Directory where per-video .npy files will be written",
    )
    parser.add_argument(
        "--filter_nonperson_box_frame",
        action="store_true",
        help="Use the same filtering as the original training loader"
    )
    parser.add_argument(
        "--enable_coco_gt",
        action="store_true",
        help="If you also want BaseAG to build internal COCO-format GT"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # instantiate your dataset
    ds = BaseAG(
        phase=args.phase,
        mode="",
        datasize=args.datasize,
        data_path=args.data_path,
        filter_nonperson_box_frame=args.filter_nonperson_box_frame,
        filter_small_box=False,
        enable_coco_gt=args.enable_coco_gt,
    )

    print(f"Loaded dataset with {len(ds)} videos")

    for vid_idx in range(len(ds)):
        # list of frame relpaths for this video: ["video_0001/000001.jpg", ...]
        frame_names = ds._video_list[vid_idx]
        # GT annotations for this video: list (per-frame) of list(dict)
        video_gt = ds.gt_annotations[vid_idx]

        # derive video_id from first frame
        first_frame = frame_names[0]
        video_id = first_frame.split("/")[0]

        # clean tensors inside
        cleaned_video_gt = []
        for frame_items in video_gt:
            cleaned_video_gt.append(clean_frame_items(frame_items))

        # we can store as an object array so structure is preserved
        out_path = os.path.join(args.output_dir, f"{video_id}.npy")
        np.save(out_path, np.array(cleaned_video_gt, dtype=object), allow_pickle=True)

        print(f"[{vid_idx+1}/{len(ds)}] saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
