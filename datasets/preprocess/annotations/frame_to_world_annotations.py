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

from annotation_utils import get_video_belongs_to_split, _load_pkl_if_exists, _npz_open, _torch_inference_ctx, \
    _del_and_collect, _lift_2d_to_3d, _find_actor_index_in_frame, _choose_primary_actor, _build_frame_to_kps_map, \
    _robust_similarity_ransac, _faces_u32, _resize_mask_to, _mask_from_bbox, _resize_bbox_to, _xywh_to_xyxy, \
    _average_sims_robust, _finite_and_nonzero, _pinhole_from_fov, _is_empty_array


class FrameToWorldAnnotations:

    def __init__(
            self,
            ag_root_directory,
            dynamic_scene_dir_path,
    ):
        self.ag_root_directory = ag_root_directory
        self.dynamic_scene_dir_path = dynamic_scene_dir_path
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
        self.bbox_4d_root_dir = self.world_annotations_root_dir / "bbox_annotations_4d"
        os.makedirs(self.bbox_4d_root_dir, exist_ok=True)

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

    def get_video_gt_annotations(self, video_id):
        video_gt_annotations_json_path = self.gt_annotations_root_dir / video_id / "gt_annotations.json"
        if not video_gt_annotations_json_path.exists():
            raise FileNotFoundError(f"GT annotations file not found: {video_gt_annotations_json_path}")

        with open(video_gt_annotations_json_path, "r") as f:
            video_gt_annotations = json.load(f)

        video_gt_bboxes = {}
        for frame_idx, frame_items in enumerate(video_gt_annotations):
            frame_name = frame_items[0]["frame"].split("/")[-1]
            boxes = []
            labels = []
            for item in frame_items:
                if 'person_bbox' in item:
                    boxes.append(item['person_bbox'][0])
                    labels.append('person')
                    continue
                category_id = item['class']
                category_name = self.catid_to_name_map[category_id]
                if category_name:
                    if category_name == "closet/cabinet":
                        category_name = "closet"
                    elif category_name == "cup/glass/bottle":
                        category_name = "cup"
                    elif category_name == "paper/notebook":
                        category_name = "paper"
                    elif category_name == "sofa/couch":
                        category_name = "sofa"
                    elif category_name == "phone/camera":
                        category_name = "phone"
                    boxes.append(item['bbox'])
                    labels.append(category_name)
            if boxes:
                video_gt_bboxes[frame_name] = {
                    'boxes': boxes,
                    'labels': labels
                }

        return video_gt_bboxes, video_gt_annotations

    def get_video_gdino_annotations(self, video_id):
        video_dynamic_gdino_prediction_file_path = self.dynamic_detections_root_path / f"{video_id}.pkl"
        video_dynamic_predictions = _load_pkl_if_exists(video_dynamic_gdino_prediction_file_path)

        video_static_gdino_prediction_file_path = self.static_detections_root_path / f"{video_id}.pkl"
        video_static_predictions = _load_pkl_if_exists(video_static_gdino_prediction_file_path)

        if video_dynamic_predictions is None:
            video_dynamic_predictions = {}
        if video_static_predictions is None:
            video_static_predictions = {}

        if not video_dynamic_predictions and not video_static_predictions:
            raise ValueError(
                f"No GDINO predictions found for video {video_id}"
            )

        all_frame_names = set(video_dynamic_predictions.keys()) | set(video_static_predictions.keys())
        combined_gdino_predictions = {}
        for frame_name in all_frame_names:
            dyn_pred = video_dynamic_predictions.get(frame_name, None)
            stat_pred = video_static_predictions.get(frame_name, None)
            if dyn_pred is None:
                dyn_pred = {"boxes": [], "labels": [], "scores": []}
            if stat_pred is None:
                stat_pred = {"boxes": [], "labels": [], "scores": []}

            if _is_empty_array(dyn_pred["boxes"]) and _is_empty_array(stat_pred["boxes"]):
                combined_gdino_predictions[frame_name] = {
                    "boxes": [],
                    "labels": [],
                    "scores": [],
                }
                continue

            combined_boxes = []
            combined_labels = []
            combined_scores = []

            if not _is_empty_array(dyn_pred["boxes"]):
                combined_boxes += list(dyn_pred["boxes"])
                combined_labels += list(dyn_pred["labels"])
                combined_scores += list(dyn_pred["scores"])

            if not _is_empty_array(stat_pred["boxes"]):
                combined_boxes += list(stat_pred["boxes"])
                combined_labels += list(stat_pred["labels"])
                combined_scores += list(stat_pred["scores"])

            final_pred = {
                "boxes": combined_boxes,
                "labels": combined_labels,
                "scores": combined_scores,
            }

            combined_gdino_predictions[frame_name] = final_pred
        return combined_gdino_predictions

    def get_video_3d_annotations(self, video_id: str):
        out_path = self.bbox_3d_root_dir / f"{video_id[:-4]}.pkl"
        if not out_path.exists():
            raise FileNotFoundError(f"3D bbox annotations file not found: {out_path}")

        # Load 3D bbox annotations
        with open(out_path, "rb") as f:
            video_3d_annotations = pickle.load(f)
        return video_3d_annotations

    def get_video_dynamic_predictions(self, video_id: str):
        video_dynamic_gdino_prediction_file_path = self.dynamic_scene_dir_path / f"{video_id}_10" / "predictions.npz"
        video_dynamic_predictions = _npz_open(video_dynamic_gdino_prediction_file_path)
        return video_dynamic_predictions

    def generate_video_bb_annotations(
            self,
            video_id: str,
            video_id_gt_annotations: Dict,
            video_id_gdino_annotations: Dict,
            video_id_3d_annotations: Optional[Dict] = None,
            video_id_dynamic_predictions: Optional[Dict] = None,
            visualize: bool = False
    ) -> None:
        print(f"[{video_id}] Generating world SGG annotations for")

        # 1. Combine GT frame-wise 2D annotations with GT 3D bbox annotations

        # 2. Check for the change in camera_poses and judge if there is any camera motion or not.
        # If there is no camera motion, then object changes are most likely due to dynamic objects or occlusions or missing objects.
        # Determine a strategy to filter out each such case.

        # 3. Begin the process of creating the 4D world annotations.
        # a. Identify all the unique objects in the video based on some tracking id.
        # b. If an object does not appear in a frame, we add the bounding box corresponding to the last seen frame or the next seen frame.
        # This ensures that each object has a bounding box in every frame of the video.
        # c. Save the world scene graph annotations in a pkl file.

        pass

    def generate_sample_gt_world_4D_annotations(self, video_id: str) -> None:
        video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
        video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
        self.generate_video_bb_annotations(
            video_id,
            video_id_gt_annotations,
            video_id_gdino_annotations,
            visualize=True
        )


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
    frame_to_world_generator.generate_sample_gt_world_4D_annotations(video_id=video_id)


if __name__ == "__main__":
    # main()
    main_sample()