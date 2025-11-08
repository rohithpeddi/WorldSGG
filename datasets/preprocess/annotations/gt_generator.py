#!/usr/bin/env python3
import json
import os

import numpy as np
import torch

from dataloader.base_ag_dataset import BaseAG


def to_jsonable(v):
    """Convert tensors / numpy to plain python so json.dump doesn't choke."""
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (str, int, float)) or v is None:
        return v
    return v


def clean_gt_frame_items(frame_items):
    """
    frame_items: list[dict] for ONE FRAME like the dataset builds.
    We convert every tensor/ndarray to a JSON-able list.
    """
    cleaned = []
    for item in frame_items:
        new_item = {}
        for k, v in item.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                new_item[k] = to_jsonable(v)
            elif isinstance(v, (list, tuple)):
                # may contain tensors inside
                new_item[k] = [to_jsonable(x) for x in v]
            else:
                new_item[k] = v
        cleaned.append(new_item)
    return cleaned


def main():
    ag_root_directory = "/data/rohith/ag/"
    output_directory = "/data/rohith/ag/ag4D/gt_annotations/"
    os.makedirs(output_directory, exist_ok=True)

    dataset = BaseAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
        enable_coco_gt=True,
    )

    print(f"Dataset loaded with {len(dataset)} videos")

    for vid_idx in range(len(dataset)):
        frame_names = dataset.video_list[vid_idx]
        gt_video_annotations = dataset.gt_annotations[vid_idx]

        video_id = frame_names[0].split("/")[0]
        video_dir = os.path.join(output_directory, video_id)
        os.makedirs(video_dir, exist_ok=True)

        images_json = []
        annotations_json = []
        ann_id = 1

        # We'll reuse the dataset's parsing logic to get per-frame boxes + cat_ids
        for frame_rel in frame_names:
            # create image entry
            image_id = len(images_json) + 1
            images_json.append({
                "id": image_id,
                "file_name": frame_rel,
            })

            # ds._parse_gt_for_frame(gt_video_annotations, frame_rel) returns (boxes_xyxy, cat_ids)
            boxes_xyxy, cat_ids = dataset.parse_gt_for_frame(gt_video_annotations, frame_rel)

            for b, cid in zip(boxes_xyxy, cat_ids):
                x1, y1, x2, y2 = b
                w = float(x2 - x1)
                h = float(y2 - y1)
                area = max(0.0, w) * max(0.0, h)
                annotations_json.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cid),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(area),
                    "iscrowd": 0,
                })
                ann_id += 1

        # clean the raw gt_annotations for this video
        cleaned_gt_ann = [clean_gt_frame_items(fitems) for fitems in gt_video_annotations]

        # write files
        with open(os.path.join(video_dir, "images.json"), "w") as f:
            json.dump(images_json, f, indent=2)

        with open(os.path.join(video_dir, "annotations.json"), "w") as f:
            json.dump(annotations_json, f, indent=2)

        with open(os.path.join(video_dir, "gt_annotations.json"), "w") as f:
            json.dump(cleaned_gt_ann, f, indent=2)

        print(f"[{vid_idx+1}/{len(dataset)}] wrote {video_dir}")

    print("Done.")


if __name__ == "__main__":
    main()