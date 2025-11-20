# This block of code takes in frame based annotations and converts them to world scene graph annotations.
# We follow the following procedure:
# First estimate all the unique objects in the video based on some tracking id.
# If an object does not appear in a frame, we add the bounding box corresponding to the last seen frame or the next seen frame.
# This ensures that each object has a bounding box in every frame of the video.
# Finally, we save the world scene graph annotations in a pkl file.

import json
import os
import argparse
import contextlib
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from dataloader.standard.action_genome.ag_dataset import StandardAG
from dataloader.base_ag_dataset import BaseAG


class FrameToWorldAnnotations:

    def __init__(
            self,
            ag_root_directory,
            dynamic_scene_dir_path,
    ):
        self.ag_root_directory = ag_root_directory
        self.dataset_classnames = [
            '__background__', 'person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair',
            'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
            'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 'paper/notebook',
            'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe',
            'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window'
        ]
        self.name_to_catid = {name: idx for idx, name in enumerate(self.dataset_classnames) if idx > 0}
        self.catid_to_name_map = {v: k for k, v in self.name_to_catid.items()}

        self.categories_json: List[Dict[str, Any]] = [
            {"id": cid, "name": name} for name, cid in self.name_to_catid.items()
        ]

        self.dynamic_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.static_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes_static'
        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = self.ag_root_directory / "sampled_frames_idx"

        self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        self.gt_annotations_root_dir = self.ag_root_directory / "gt_annotations"

        self.dynamic_masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.dynamic_masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.dynamic_masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.dynamic_masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        self.dynamic_masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.dynamic_masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.dynamic_masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        self.static_masks_im_dir_path = self.ag_root_directory / "segmentation_static" / 'masks' / 'image_based'
        self.static_masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / 'masks' / 'video_based'
        self.static_masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

        os.makedirs(self.bbox_3d_root_dir, exist_ok=True)


def load_dataset(ag_root_directory: str):
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False
    )

    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False
    )

    dataloader_train = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined: (a) floor-aligned 3D bbox generator + (b) SMPL↔PI3 human mesh aligner (sampled frames only)."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument("--dynamic_scene_dir_path", type=str,
                        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic")
    parser.add_argument("--output_world_annotations", type=str, default="/data/rohith/ag/ag4D/human/")
    parser.add_argument("--split", type=str, default="04")
    parser.add_argument("--include_dense", action="store_true",
                        help="use dense correspondences for human aligner")
    return parser.parse_args()

def main():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
    frame_to_world_generator.generate_gt_world_bb_annotations(dataloader=dataloader_train, split=args.split)
    frame_to_world_generator.generate_gt_world_bb_annotations(dataloader=dataloader_test, split=args.split)


def main_sample():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    video_id = "0DJ6R.mp4"
    frame_to_world_generator.generate_sample_gt_world_bb_annotations(video_id=video_id)


def main():
    ag_root_directory = "/data/rohith/ag/"
    dataset = BaseAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
        enable_coco_gt=True,
    )
    print(f"Dataset loaded with {len(dataset)} videos")