#!/usr/bin/env python3
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

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from dataloader.standard.action_genome.ag_dataset import StandardAG
from dataloader.base_ag_dataset import BaseAG

from annotation_utils import (
    get_video_belongs_to_split,
    _load_pkl_if_exists,
    _npz_open,
    _torch_inference_ctx,
    _del_and_collect,
    _lift_2d_to_3d,
    _find_actor_index_in_frame,
    _choose_primary_actor,
    _build_frame_to_kps_map,
    _robust_similarity_ransac,
    _faces_u32,
    _resize_mask_to,
    _mask_from_bbox,
    _resize_bbox_to,
    _xywh_to_xyxy,
    _average_sims_robust,
    _finite_and_nonzero,
    _pinhole_from_fov,
    _is_empty_array,
)
from concurrent.futures import ThreadPoolExecutor, as_completed


class FrameToWorldAnnotations:
    def __init__(self, ag_root_directory, dynamic_scene_dir_path):
        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)

        self.dataset_classnames = [
            "__background__",
            "person",
            "bag",
            "bed",
            "blanket",
            "book",
            "box",
            "broom",
            "chair",
            "closet/cabinet",
            "clothes",
            "cup/glass/bottle",
            "dish",
            "door",
            "doorknob",
            "doorway",
            "floor",
            "food",
            "groceries",
            "laptop",
            "light",
            "medicine",
            "mirror",
            "paper/notebook",
            "phone/camera",
            "picture",
            "pillow",
            "refrigerator",
            "sandwich",
            "shelf",
            "shoe",
            "sofa/couch",
            "table",
            "television",
            "towel",
            "vacuum",
            "window",
        ]
        self.name_to_catid = {
            name: idx for idx, name in enumerate(self.dataset_classnames) if idx > 0
        }
        self.catid_to_name_map = {v: k for k, v in self.name_to_catid.items()}

        self.categories_json: List[Dict[str, Any]] = [
            {"id": cid, "name": name} for name, cid in self.name_to_catid.items()
        ]

        self.dynamic_detections_root_path = (
            self.ag_root_directory / "detection" / "gdino_bboxes"
        )
        self.static_detections_root_path = (
            self.ag_root_directory / "detection" / "gdino_bboxes_static"
        )
        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = (
            self.ag_root_directory / "sampled_frames_idx"
        )

        self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        self.bbox_4d_root_dir = self.world_annotations_root_dir / "bbox_annotations_4d"
        os.makedirs(self.bbox_4d_root_dir, exist_ok=True)

        self.gt_annotations_root_dir = self.ag_root_directory / "gt_annotations"

        self.dynamic_masked_frames_im_dir_path = (
            self.ag_root_directory
            / "segmentation"
            / "masked_frames"
            / "image_based"
        )
        self.dynamic_masked_frames_vid_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_frames" / "video_based"
        )
        self.dynamic_masked_frames_combined_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_frames" / "combined"
        )
        self.dynamic_masked_videos_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_videos"
        )

        self.dynamic_masks_im_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "image_based"
        )
        self.dynamic_masks_vid_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "video_based"
        )
        self.dynamic_masks_combined_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "combined"
        )

        self.static_masks_im_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        )
        self.static_masks_vid_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        )
        self.static_masks_combined_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "combined"
        )

    # ----------------------------------------------------------------------------------
    # GT + GDINO + 3D annotations loaders
    # ----------------------------------------------------------------------------------

    def get_video_gt_annotations(self, video_id: str):
        video_gt_annotations_json_path = (
            self.gt_annotations_root_dir / video_id / "gt_annotations.json"
        )
        if not video_gt_annotations_json_path.exists():
            raise FileNotFoundError(
                f"GT annotations file not found: {video_gt_annotations_json_path}"
            )

        with open(video_gt_annotations_json_path, "r") as f:
            video_gt_annotations = json.load(f)

        video_gt_bboxes: Dict[str, Dict[str, Any]] = {}
        for frame_idx, frame_items in enumerate(video_gt_annotations):
            frame_name = frame_items[0]["frame"].split("/")[-1]
            boxes = []
            labels = []
            for item in frame_items:
                if "person_bbox" in item:
                    boxes.append(item["person_bbox"][0])
                    labels.append("person")
                    continue

                category_id = item["class"]
                category_name = self.catid_to_name_map[category_id]

                if category_name:
                    # Normalize some label names
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

                    boxes.append(item["bbox"])
                    labels.append(category_name)

            if boxes:
                video_gt_bboxes[frame_name] = {"boxes": boxes, "labels": labels}

        return video_gt_bboxes, video_gt_annotations

    def get_video_gdino_annotations(self, video_id: str):
        video_dynamic_gdino_prediction_file_path = (
            self.dynamic_detections_root_path / f"{video_id}.pkl"
        )
        video_dynamic_predictions = _load_pkl_if_exists(
            video_dynamic_gdino_prediction_file_path
        )

        video_static_gdino_prediction_file_path = (
            self.static_detections_root_path / f"{video_id}.pkl"
        )
        video_static_predictions = _load_pkl_if_exists(
            video_static_gdino_prediction_file_path
        )

        if video_dynamic_predictions is None:
            video_dynamic_predictions = {}
        if video_static_predictions is None:
            video_static_predictions = {}

        if not video_dynamic_predictions and not video_static_predictions:
            raise ValueError(f"No GDINO predictions found for video {video_id}")

        all_frame_names = set(video_dynamic_predictions.keys()) | set(
            video_static_predictions.keys()
        )
        combined_gdino_predictions: Dict[str, Dict[str, Any]] = {}

        for frame_name in all_frame_names:
            dyn_pred = video_dynamic_predictions.get(frame_name, None)
            stat_pred = video_static_predictions.get(frame_name, None)

            if dyn_pred is None:
                dyn_pred = {"boxes": [], "labels": [], "scores": []}
            if stat_pred is None:
                stat_pred = {"boxes": [], "labels": [], "scores": []}

            if _is_empty_array(dyn_pred["boxes"]) and _is_empty_array(
                stat_pred["boxes"]
            ):
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
            return None

        with open(out_path, "rb") as f:
            video_3d_annotations = pickle.load(f)
        return video_3d_annotations

    def get_video_dynamic_predictions(self, video_id: str):
        video_dynamic_3d_scene_path = (
            self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        )
        if not video_dynamic_3d_scene_path.exists():
            return None
        video_dynamic_predictions = _npz_open(video_dynamic_3d_scene_path)
        return video_dynamic_predictions

    # ----------------------------------------------------------------------------------
    # Data checks: 2D/3D consistency
    # ----------------------------------------------------------------------------------

    def check_2d_3d_annotation_consistency(
        self,
        video_id: str,
        video_id_gt_annotations: Dict,
        video_id_3d_bbox_predictions: Dict,
    ) -> int:
        # First check the total number of frames in both gt_2D annotations and the gt_3D annotations
        gt_2d_frames = set()
        for frame_items in video_id_gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]
            gt_2d_frames.add(frame_name)

        gt_3d_frames = set(video_id_3d_bbox_predictions["frames"].keys())

        # Filter out the frames in 3D that have zero or invalid bboxes
        valid_3d_frames = set()
        for frame_name, frame_data in video_id_3d_bbox_predictions["frames"].items():
            objects_3d = frame_data.get("objects", [])
            if len(objects_3d) > 0:
                valid_3d_frames.add(frame_name)

        # Check mismatched instances
        mismatched_frames = gt_2d_frames.symmetric_difference(valid_3d_frames)
        return len(mismatched_frames)

    def estimate_mismatched_annotations(
        self, train_dataloader: DataLoader, test_dataloader: DataLoader
    ) -> None:
        def _estimate_dataset_mismatches(
            dataloader: DataLoader,
        ) -> Tuple[Dict[int, int], int]:
            mismatch_count_map: Dict[int, int] = {}
            missing_3D_annotations = 0

            for data in tqdm(dataloader, desc="Estimating mismatched annotations"):
                video_id = data["video_id"]

                video_id_gt_bboxes, video_id_gt_annotations = (
                    self.get_video_gt_annotations(video_id)
                )
                video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)

                if video_id_3d_bbox_predictions is None:
                    missing_3D_annotations += 1
                    continue

                mismatched_count = self.check_2d_3d_annotation_consistency(
                    video_id,
                    video_id_gt_annotations,
                    video_id_3d_bbox_predictions,
                )
                mismatch_count_map[mismatched_count] = (
                    mismatch_count_map.get(mismatched_count, 0) + 1
                )

            return mismatch_count_map, missing_3D_annotations

        print("Estimating mismatched annotations in training dataset...")
        train_mismatch_count_map, train_missing_3D = _estimate_dataset_mismatches(
            train_dataloader
        )

        print("Estimating mismatched annotations in testing dataset...")
        test_mismatch_count_map, test_missing_3D = _estimate_dataset_mismatches(
            test_dataloader
        )

        # Create plots to display a histogram of mismatched counts
        import matplotlib.pyplot as plt

        def plot_mismatch_histogram(
            mismatch_count_map: Dict[int, int], title: str, output_path: str
        ) -> None:
            counts = list(mismatch_count_map.keys())
            frequencies = [mismatch_count_map[k] for k in counts]
            plt.figure(figsize=(10, 6))
            plt.bar(counts, frequencies, width=0.8, edgecolor="black")
            plt.xlabel("Number of Mismatched Frames")
            plt.ylabel("Number of Videos")
            plt.title(title)
            plt.xticks(counts)
            plt.grid(axis="y")
            plt.savefig(output_path)
            plt.close()

        plot_mismatch_histogram(
            train_mismatch_count_map,
            "Training Dataset Mismatched Annotations",
            self.world_annotations_root_dir / "train_mismatched_annotations_histogram.png",
        )
        plot_mismatch_histogram(
            test_mismatch_count_map,
            "Testing Dataset Mismatched Annotations",
            self.world_annotations_root_dir / "test_mismatched_annotations_histogram.png",
        )
        print(f"Training dataset missing 3D annotations for {train_missing_3D} videos.")
        print(f"Testing dataset missing 3D annotations for {test_missing_3D} videos.")

    # ----------------------------------------------------------------------------------
    # Camera motion estimation
    # ----------------------------------------------------------------------------------

    def construct_avg_motion_buckets(
        self, motion_magnitudes: List[float]
    ) -> Dict[str, int]:
        motion_buckets: Dict[str, int] = {
            "0.0-0.01": 0,
            "0.01-0.05": 0,
            "0.05-0.1": 0,
            "0.1-0.5": 0,
            "0.5+": 0,
        }
        for motion in motion_magnitudes:
            if motion < 0.01:
                motion_buckets["0.0-0.01"] += 1
            elif motion < 0.05:
                motion_buckets["0.01-0.05"] += 1
            elif motion < 0.1:
                motion_buckets["0.05-0.1"] += 1
            elif motion < 0.5:
                motion_buckets["0.1-0.5"] += 1
            else:
                motion_buckets["0.5+"] += 1
        return motion_buckets

    def estimate_camera_motion_for_video(
        self, video_id: str, video_dynamic_predictions: Dict[str, Any]
    ) -> Tuple[bool, float]:
        video_camera_poses = video_dynamic_predictions["camera_poses"]

        # Handle short sequences safely
        num_frames = len(video_camera_poses)
        if num_frames < 2:
            return False, 0.0

        poses = np.asarray(video_camera_poses, dtype=np.float32)  # (T, 4, 4)
        translations = poses[:, :3, 3]  # (T, 3)
        diffs = translations[1:] - translations[:-1]  # (T-1, 3)
        motion_magnitudes = np.linalg.norm(diffs, axis=-1)  # (T-1,)

        avg_motion = float(motion_magnitudes.mean())
        motion_threshold = 0.01  # threshold for significant motion
        has_motion = avg_motion > motion_threshold
        print(f"[{video_id}] Estimating camera motion... avg_motion={avg_motion:.6f}")
        return has_motion, avg_motion

    def _count_videos_with_camera_motion(
        self, split: str, dataloader: DataLoader
    ):
        """
        Iterate over a dataloader and compute camera motion stats for a given AG split.

        Returns:
            motion_count: number of videos whose avg motion > threshold
            total_count: total number of videos considered
            motion_buckets: histogram buckets of avg motion magnitudes
            avg_motion_list: list of avg motion magnitudes (one per video)
            video_avg_motion: dict[video_id] -> avg motion
        """
        motion_count = 0
        total_count = 0
        avg_motion_list: List[float] = []
        video_avg_motion: Dict[str, float] = {}

        for data in tqdm(dataloader, desc=f"Estimating camera motion (split={split})"):
            video_id = data["video_id"]

            # Filter by AG split (00–07 etc.)
            if get_video_belongs_to_split(video_id) != split:
                continue

            # Load dynamic scene / camera poses
            try:
                video_dynamic_predictions = self._load_points_for_video(video_id)
            except Exception as e:
                print(
                    f"[{video_id}] Skipping camera motion estimation due to load error: {e}"
                )
                continue

            if video_dynamic_predictions is None:
                continue

            has_motion, avg_motion = self.estimate_camera_motion_for_video(
                video_id, video_dynamic_predictions
            )

            if has_motion:
                motion_count += 1
            total_count += 1
            avg_motion_list.append(avg_motion)
            video_avg_motion[video_id] = avg_motion

        motion_buckets = self.construct_avg_motion_buckets(avg_motion_list)
        return (
            motion_count,
            total_count,
            motion_buckets,
            avg_motion_list,
            video_avg_motion,
        )

    def estimate_camera_motion_in_dataset(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        split: str,
    ) -> Tuple[int, int]:
        print(f"Estimating camera motion in training dataset for split {split}...")
        (
            train_motion_count,
            train_video_count,
            train_motion_buckets,
            train_avg_motion_list,
            train_video_avg_motion,
        ) = self._count_videos_with_camera_motion(split, train_dataloader)

        print(f"Estimating camera motion in testing dataset for split {split}...")
        (
            test_motion_count,
            test_video_count,
            test_motion_buckets,
            test_avg_motion_list,
            test_video_avg_motion,
        ) = self._count_videos_with_camera_motion(split, test_dataloader)

        print(
            f"Training dataset (split {split}): {train_motion_count}/{train_video_count} videos with camera motion."
        )
        print(
            f"Testing  dataset (split {split}): {test_motion_count}/{test_video_count} videos with camera motion."
        )

        # Plot the motion buckets (train/test separately)
        import matplotlib.pyplot as plt

        def plot_motion_buckets(
            motion_buckets: Dict[str, int], title: str, output_path: Path
        ) -> None:
            labels = list(motion_buckets.keys())
            counts = [motion_buckets[label] for label in labels]
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts, width=0.6, edgecolor="black")
            plt.xlabel("Average Camera Motion Magnitude Buckets")
            plt.ylabel("Number of Videos")
            plt.title(title)
            plt.grid(axis="y")
            plt.savefig(output_path)
            plt.close()

        # Per-split plots (train/test)
        plot_motion_buckets(
            train_motion_buckets,
            f"Training Dataset Camera Motion Buckets (split {split})",
            self.world_annotations_root_dir
            / f"train_camera_motion_buckets_split_{split}.png",
        )
        plot_motion_buckets(
            test_motion_buckets,
            f"Testing Dataset Camera Motion Buckets (split {split})",
            self.world_annotations_root_dir
            / f"test_camera_motion_buckets_split_{split}.png",
        )

        # ---- Save processed avg_motion results for each split (train/test separately) ----
        train_stats = {
            "split": split,
            "dataset_split": "train",
            "video_count": train_video_count,
            "motion_count": train_motion_count,
            "motion_buckets": train_motion_buckets,
            "video_avg_motion": train_video_avg_motion,  # per-video avg motion
        }
        test_stats = {
            "split": split,
            "dataset_split": "test",
            "video_count": test_video_count,
            "motion_count": test_motion_count,
            "motion_buckets": test_motion_buckets,
            "video_avg_motion": test_video_avg_motion,
        }

        train_stats_path = (
            self.world_annotations_root_dir / f"camera_motion_train_split_{split}.json"
        )
        test_stats_path = (
            self.world_annotations_root_dir / f"camera_motion_test_split_{split}.json"
        )

        with open(train_stats_path, "w") as f:
            json.dump(train_stats, f, indent=2)
        with open(test_stats_path, "w") as f:
            json.dump(test_stats, f, indent=2)

        print(f"[camera_motion] Saved train stats to {train_stats_path}")
        print(f"[camera_motion] Saved test stats to {test_stats_path}")

        # NOTE: no combining here — train and test stats are kept separate.
        return train_motion_count, test_motion_count

    def combine_camera_motion_stats_all_splits(self) -> None:
        """
        Load previously saved train/test camera motion stats for ALL splits,
        combine all TRAIN stats together and all TEST stats together,
        save two new combined JSONs, and create two combined plots.

        Expected existing files:
            world_annotations/camera_motion_train_split_*.json
            world_annotations/camera_motion_test_split_*.json
        """

        # ------------------------------------------------------------------
        # Helper: combine a set of jsons (train OR test) across all splits
        # ------------------------------------------------------------------
        def _combine_one_split_group(json_paths, dataset_split: str) -> None:
            if not json_paths:
                print(f"[camera_motion] No {dataset_split} stats found to combine.")
                return

            all_motion_values: List[float] = []
            total_motion_count: int = 0
            total_video_count: int = 0
            used_splits: List[str] = []

            for path in sorted(json_paths):
                with open(path, "r") as f:
                    stats = json.load(f)

                split_name = stats.get("split", None)
                if split_name is None:
                    # Try to parse from filename if not present in JSON
                    # e.g. camera_motion_train_split_04.json -> "04"
                    stem = Path(path).stem
                    # stem = "camera_motion_train_split_04"
                    parts = stem.split("_")
                    split_name = parts[-1] if parts else stem
                used_splits.append(split_name)

                video_avg_motion = stats.get("video_avg_motion", {})
                # ensure float conversion
                all_motion_values.extend(float(v) for v in video_avg_motion.values())

                total_motion_count += int(stats.get("motion_count", 0))
                total_video_count += int(stats.get("video_count", 0))

            # Construct combined buckets using *all* per-video motion magnitudes
            combined_motion_buckets = self.construct_avg_motion_buckets(
                all_motion_values
            )

            combined_stats = {
                "dataset_split": dataset_split,       # "train" or "test"
                "splits": used_splits,               # which splits were combined
                "video_count": total_video_count,
                "motion_count": total_motion_count,
                "motion_buckets": combined_motion_buckets,
            }

            # Save combined JSON
            combined_stats_path = (
                self.world_annotations_root_dir
                / f"camera_motion_{dataset_split}_all_splits.json"
            )
            with open(combined_stats_path, "w") as f:
                json.dump(combined_stats, f, indent=2)

            print(
                f"[camera_motion] Saved combined {dataset_split} stats to {combined_stats_path}"
            )

            # Plot combined buckets
            import matplotlib.pyplot as plt

            def plot_motion_buckets(
                motion_buckets: Dict[str, int], title: str, output_path: Path
            ) -> None:
                labels = list(motion_buckets.keys())
                counts = [motion_buckets[label] for label in labels]
                plt.figure(figsize=(10, 6))
                plt.bar(labels, counts, width=0.6, edgecolor="black")
                plt.xlabel("Average Camera Motion Magnitude Buckets")
                plt.ylabel("Number of Videos")
                plt.title(title)
                plt.grid(axis="y")
                plt.savefig(output_path)
                plt.close()

            out_png = (
                self.world_annotations_root_dir
                / f"{dataset_split}_camera_motion_buckets_all_splits.png"
            )
            plot_motion_buckets(
                combined_motion_buckets,
                f"{dataset_split.capitalize()} Dataset Camera Motion Buckets (all splits)",
                out_png,
            )
            print(
                f"[camera_motion] Saved combined {dataset_split} motion buckets plot "
                f"to {out_png}"
            )

        # ------------------------------------------------------------------
        # Discover all train/test jsons and combine separately
        # ------------------------------------------------------------------
        train_json_paths = sorted(
            self.world_annotations_root_dir.glob("camera_motion_train_split_*.json")
        )
        test_json_paths = sorted(
            self.world_annotations_root_dir.glob("camera_motion_test_split_*.json")
        )

        if not train_json_paths and not test_json_paths:
            print(
                "[camera_motion] No camera_motion_train_split_*.json or "
                "camera_motion_test_split_*.json files found to combine."
            )
            return

        # Combine all TRAIN stats across splits
        _combine_one_split_group(train_json_paths, dataset_split="train")

        # Combine all TEST stats across splits
        _combine_one_split_group(test_json_paths, dataset_split="test")

    # ----------------------------------------------------------------------------------
    # World 4D bbox annotations (skeleton — fill in with your logic)
    # ----------------------------------------------------------------------------------

    def _create_label_wise_masks_map(
        self, video_id: str, gt_annotations: List[Any]
    ):
        """
        PLACEHOLDER: use your existing implementation here.

        Should return:
            video_to_frame_to_label_mask, all_static_labels, all_dynamic_labels
        """
        raise NotImplementedError(
            "_create_label_wise_masks_map should be replaced with your existing implementation."
        )

    def generate_video_bb_annotations(
        self,
        video_id: str,
        video_id_gt_annotations: Dict,
        video_id_gdino_annotations: Dict,
        video_id_3d_bbox_predictions: Dict,
        visualize: bool = False,
    ) -> None:
        """
        video_id_3D_bbox_predictions format:

        {
            "video_id": video_id,
            "frames": out_frames,
            "per_frame_sims": per_frame_sims,
            "global_floor_sim": {
                "s": float(s_avg),
                "R": R_avg,
                "t": t_avg,
            },
            "primary_track_id_0": primary_track_id_0,
            "frame_bbox_meshes": frame_bbox_meshes,
            "gv": gv,
            "gf": gf,
            "gc": gc
        }
        """

        print(f"[{video_id}] Generating world SGG annotations")

        # Load 3D points for the video from dynamic scene predictions
        try:
            P = self._load_points_for_video(video_id)
            points_S = P["points"]  # (S,H,W,3)
            conf_S = P["conf"]  # (S,H,W) or None
            stems_S = P["frame_stems"]  # list of frame stems
            colors = P["colors"]  # (S,H,W,3)
            camera_poses = P["camera_poses"]  # (S,4,4)
            S, H, W, _ = points_S.shape
        except Exception as e:
            print(f"[{video_id}] Failed to load 3D points: {e}")
            return

        stem_to_idx = {stems_S[i]: i for i in range(S)}

        # Create label-wise masks map for segmentation
        (
            video_to_frame_to_label_mask,
            all_static_labels,
            all_dynamic_labels,
        ) = self._create_label_wise_masks_map(
            video_id=video_id, gt_annotations=video_id_gt_annotations
        )

        # Output structure for storing frame annotations
        out_frames: Dict[str, Dict[str, Any]] = {}

        # NOTE: Fill out_frames based on your 4D SGG logic (objects, tracks, interpolation, etc.)

        out_path = self.bbox_4d_root_dir / f"{video_id[:-4]}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "video_id": video_id,
                    "frames": out_frames,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(f"[{video_id}] Saved 4D world annotations to {out_path}")

    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        """Load 3D points from dynamic scene predictions (sampled to annotated frames)."""
        video_dynamic_3d_scene_path = (
            self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        )
        video_dynamic_predictions = np.load(
            video_dynamic_3d_scene_path, allow_pickle=True
        )
        video_dynamic_predictions = {
            k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files
        }

        points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
        imgs_f32 = video_dynamic_predictions["images"]
        camera_poses = video_dynamic_predictions["camera_poses"]
        colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

        conf = None
        if "conf" in video_dynamic_predictions:
            conf = video_dynamic_predictions["conf"]
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf[..., 0]

        S, H, W, _ = points.shape

        # Get frame mapping
        video_frames_annotated_dir_path = self.frame_annotated_dir_path / video_id
        annotated_frame_id_list = [
            f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith(".png")
        ]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))
        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        video_sampled_frames_npy_path = (
            self.sampled_frames_idx_root_dir / f"{video_id[:-4]}.npy"
        )
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(
            annotated_first_frame_id
        )
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(
            annotated_last_frame_id
        )
        sample_idx = list(
            range(
                an_first_id_in_vid_sam_frame_id_list,
                an_last_id_in_vid_sam_frame_id_list + 1,
            )
        )

        assert S == len(sample_idx)

        sampled_idx_frame_name_map: Dict[int, str] = {}
        frame_name_sampled_idx_map: Dict[str, int] = {}
        for idx_in_s, frame_idx in enumerate(sample_idx):
            frame_name = f"{video_sampled_frame_id_list[frame_idx]:06d}.png"
            sampled_idx_frame_name_map[idx_in_s] = frame_name
            frame_name_sampled_idx_map[frame_name] = idx_in_s

        annotated_idx_in_sampled_idx: List[int] = []
        for frame_name in annotated_frame_id_list:
            if frame_name in frame_name_sampled_idx_map:
                annotated_idx_in_sampled_idx.append(frame_name_sampled_idx_map[frame_name])

        points_sub = points[annotated_idx_in_sampled_idx]
        conf_sub = conf[annotated_idx_in_sampled_idx] if conf is not None else None
        stems_sub = [
            sampled_idx_frame_name_map[idx][:-4] for idx in annotated_idx_in_sampled_idx
        ]
        colors_sub = colors[annotated_idx_in_sampled_idx]
        camera_poses_sub = camera_poses[annotated_idx_in_sampled_idx]

        return {
            "points": points_sub,
            "conf": conf_sub,
            "frame_stems": stems_sub,
            "colors": colors_sub,
            "camera_poses": camera_poses_sub,
        }

    def generate_sample_gt_world_4D_annotations(self, video_id: str) -> None:
        video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(
            video_id
        )
        video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
        video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)
        self.generate_video_bb_annotations(
            video_id,
            video_id_gt_annotations,
            video_id_gdino_annotations,
            video_id_3d_bbox_predictions,
            visualize=True,
        )

    def generate_gt_world_bb_annotations(
        self, dataloader: DataLoader, split: str
    ) -> None:
        """
        PLACEHOLDER: your existing implementation that iterates over the dataloader
        and calls generate_video_bb_annotations for videos in the given AG split.
        """
        raise NotImplementedError(
            "generate_gt_world_bb_annotations should be replaced with your existing implementation."
        )


# --------------------------------------------------------------------------------------
# Dataset + CLI
# --------------------------------------------------------------------------------------


def load_dataset(ag_root_directory: str):
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    dataloader_train = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0,
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False,
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combined: "
            "(a) floor-aligned 3D bbox generator + "
            "(b) SMPL↔PI3 human mesh aligner (sampled frames only)."
        )
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument(
        "--output_world_annotations",
        type=str,
        default="/data/rohith/ag/ag4D/human/",
    )
    parser.add_argument("--split", type=str, default="04")
    parser.add_argument(
        "--include_dense",
        action="store_true",
        help="use dense correspondences for human aligner",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(
        args.ag_root_directory
    )
    frame_to_world_generator.generate_gt_world_bb_annotations(
        dataloader=dataloader_train, split=args.split
    )
    frame_to_world_generator.generate_gt_world_bb_annotations(
        dataloader=dataloader_test, split=args.split
    )


def main_sample():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    video_id = "0DJ6R.mp4"
    frame_to_world_generator.generate_sample_gt_world_4D_annotations(video_id=video_id)


def main_estimate_mismatches():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(
        args.ag_root_directory
    )
    frame_to_world_generator.estimate_mismatched_annotations(
        train_dataloader=dataloader_train, test_dataloader=dataloader_test
    )


def main_estimate_camera_motion():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(
        args.ag_root_directory
    )
    frame_to_world_generator.estimate_camera_motion_in_dataset(
        train_dataloader=dataloader_train,
        test_dataloader=dataloader_test,
        split=args.split,
    )


def main_combine_camera_motion_stats():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    frame_to_world_generator.combine_camera_motion_stats_all_splits()


if __name__ == "__main__":
    # main_estimate_mismatches()
    # main_estimate_camera_motion()
    # main_combine_camera_motion_stats()
    # main()
    main_sample()

