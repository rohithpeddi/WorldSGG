#!/usr/bin/env python3
import argparse
import contextlib
import gc
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.preprocess.annotations.frame_to_world_base import FrameToWorldBase

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from dataloader.standard.action_genome.ag_dataset import StandardAG

from annotation_utils import (
    get_video_belongs_to_split,
    _load_pkl_if_exists,
    _npz_open,
    _is_empty_array,
)


# --------------------------------------------------------------------------------------
# Stats estimator #1: 2D/3D mismatch estimation
# --------------------------------------------------------------------------------------

class MismatchStatsEstimator(FrameToWorldBase):

    def check_2d_3d_annotation_consistency(
        self,
        video_id: str,
        video_id_gt_annotations: List[Any],
        video_id_3d_bbox_predictions: Dict[str, Any],
    ) -> int:
        gt_2d_frames = set()
        for frame_items in video_id_gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]
            gt_2d_frames.add(frame_name)

        # Only keep 3D frames with at least one valid object
        valid_3d_frames = set()
        for frame_name, frame_data in video_id_3d_bbox_predictions["frames"].items():
            objects_3d = frame_data.get("objects", [])
            if len(objects_3d) > 0:
                valid_3d_frames.add(frame_name)

        mismatched_frames = gt_2d_frames.symmetric_difference(valid_3d_frames)
        return len(mismatched_frames)

    def estimate_mismatched_annotations(
        self, train_dataloader: DataLoader, test_dataloader: DataLoader
    ) -> Dict[str, Any]:
        def _estimate_dataset_mismatches(dataloader: DataLoader) -> Tuple[Dict[int, int], int]:
            mismatch_count_map: Dict[int, int] = {}
            missing_3d = 0

            for data in tqdm(dataloader, desc="Estimating mismatched annotations"):
                video_id = data["video_id"]

                _, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
                video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)

                if video_id_3d_bbox_predictions is None:
                    missing_3d += 1
                    continue

                mismatched_count = self.check_2d_3d_annotation_consistency(
                    video_id, video_id_gt_annotations, video_id_3d_bbox_predictions
                )
                mismatch_count_map[mismatched_count] = mismatch_count_map.get(mismatched_count, 0) + 1

            return mismatch_count_map, missing_3d

        print("Estimating mismatched annotations in training dataset...")
        train_map, train_missing = _estimate_dataset_mismatches(train_dataloader)

        print("Estimating mismatched annotations in testing dataset...")
        test_map, test_missing = _estimate_dataset_mismatches(test_dataloader)

        # Plot histograms
        import matplotlib.pyplot as plt

        def plot_hist(mismatch_count_map: Dict[int, int], title: str, output_path: Path) -> None:
            counts = sorted(mismatch_count_map.keys())
            freqs = [mismatch_count_map[k] for k in counts]
            plt.figure(figsize=(10, 6))
            plt.bar(counts, freqs, width=0.8, edgecolor="black")
            plt.xlabel("Number of Mismatched Frames")
            plt.ylabel("Number of Videos")
            plt.title(title)
            plt.xticks(counts if len(counts) <= 50 else counts[::2])
            plt.grid(axis="y")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path))
            plt.close()

        plot_hist(
            train_map,
            "Training Dataset Mismatched Annotations",
            self.world_annotations_root_dir / "train_mismatched_annotations_histogram.png",
        )
        plot_hist(
            test_map,
            "Testing Dataset Mismatched Annotations",
            self.world_annotations_root_dir / "test_mismatched_annotations_histogram.png",
        )

        stats = {
            "train": {"missing_3d": train_missing, "mismatch_hist": train_map},
            "test": {"missing_3d": test_missing, "mismatch_hist": test_map},
        }
        print(f"Training dataset missing 3D annotations for {train_missing} videos.")
        print(f"Testing dataset  missing 3D annotations for {test_missing} videos.")
        return stats


# --------------------------------------------------------------------------------------
# Stats estimator #2: camera motion estimation (per split)
# --------------------------------------------------------------------------------------


class CameraMotionStatsEstimator(FrameToWorldBase):



    def estimate_camera_motion_for_video(
        self, video_id: str, video_dynamic_predictions: Dict[str, Any]
    ) -> Tuple[bool, float]:
        video_camera_poses = video_dynamic_predictions["camera_poses"]

        num_frames = len(video_camera_poses)
        if num_frames < 2:
            return False, 0.0

        poses = np.asarray(video_camera_poses, dtype=np.float32)  # (T, 4, 4)
        translations = poses[:, :3, 3]  # (T, 3)
        diffs = translations[1:] - translations[:-1]  # (T-1, 3)
        motion_magnitudes = np.linalg.norm(diffs, axis=-1)  # (T-1,)

        avg_motion = float(motion_magnitudes.mean())
        motion_threshold = 0.01
        has_motion = avg_motion > motion_threshold
        print(f"[{video_id}] avg_motion={avg_motion:.6f}")
        return has_motion, avg_motion

    def _count_videos_with_camera_motion(self, split: str, dataloader: DataLoader):
        motion_count = 0
        total_count = 0
        avg_motion_list: List[float] = []
        video_avg_motion: Dict[str, float] = {}

        for data in tqdm(dataloader, desc=f"Estimating camera motion (split={split})"):
            video_id = data["video_id"]

            if get_video_belongs_to_split(video_id) != split:
                continue

            try:
                video_dynamic_predictions = self._load_points_for_video(video_id)
            except Exception as e:
                print(f"[{video_id}] Skipping (load error): {e}")
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
        return motion_count, total_count, motion_buckets, avg_motion_list, video_avg_motion

    def estimate_camera_motion_in_dataset(
        self, train_dataloader: DataLoader, test_dataloader: DataLoader, split: str
    ) -> Dict[str, Any]:
        print(f"Estimating camera motion in training dataset for split {split}...")
        train_motion_count, train_video_count, train_buckets, _, train_video_avg = (
            self._count_videos_with_camera_motion(split, train_dataloader)
        )

        print(f"Estimating camera motion in testing dataset for split {split}...")
        test_motion_count, test_video_count, test_buckets, _, test_video_avg = (
            self._count_videos_with_camera_motion(split, test_dataloader)
        )

        print(f"Train (split {split}): {train_motion_count}/{train_video_count} videos with motion.")
        print(f"Test  (split {split}): {test_motion_count}/{test_video_count} videos with motion.")

        # Plot buckets
        import matplotlib.pyplot as plt

        def plot_motion_buckets(motion_buckets: Dict[str, int], title: str, output_path: Path) -> None:
            labels = list(motion_buckets.keys())
            counts = [motion_buckets[l] for l in labels]
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts, width=0.6, edgecolor="black")
            plt.xlabel("Average Camera Motion Magnitude Buckets")
            plt.ylabel("Number of Videos")
            plt.title(title)
            plt.grid(axis="y")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path))
            plt.close()

        plot_motion_buckets(
            train_buckets,
            f"Training Dataset Camera Motion Buckets (split {split})",
            self.world_annotations_root_dir / f"train_camera_motion_buckets_split_{split}.png",
        )
        plot_motion_buckets(
            test_buckets,
            f"Testing Dataset Camera Motion Buckets (split {split})",
            self.world_annotations_root_dir / f"test_camera_motion_buckets_split_{split}.png",
        )

        # Save JSONs
        train_stats = {
            "split": split,
            "dataset_split": "train",
            "video_count": train_video_count,
            "motion_count": train_motion_count,
            "motion_buckets": train_buckets,
            "video_avg_motion": train_video_avg,
        }
        test_stats = {
            "split": split,
            "dataset_split": "test",
            "video_count": test_video_count,
            "motion_count": test_motion_count,
            "motion_buckets": test_buckets,
            "video_avg_motion": test_video_avg,
        }

        train_stats_path = self.world_annotations_root_dir / f"camera_motion_train_split_{split}.json"
        test_stats_path = self.world_annotations_root_dir / f"camera_motion_test_split_{split}.json"

        with open(train_stats_path, "w") as f:
            json.dump(train_stats, f, indent=2)
        with open(test_stats_path, "w") as f:
            json.dump(test_stats, f, indent=2)

        print(f"[camera_motion] Saved train stats to {train_stats_path}")
        print(f"[camera_motion] Saved test  stats to {test_stats_path}")

        return {"train": train_stats, "test": test_stats}


# --------------------------------------------------------------------------------------
# Stats estimator #3: combine previously saved camera motion stats across splits
# --------------------------------------------------------------------------------------

class CameraMotionStatsCombiner(FrameToWorldBase):


    def combine_camera_motion_stats_all_splits(self) -> None:
        def _combine_one_group(json_paths: List[Path], dataset_split: str) -> None:
            if not json_paths:
                print(f"[camera_motion] No {dataset_split} stats found to combine.")
                return

            all_motion_values: List[float] = []
            total_motion_count = 0
            total_video_count = 0
            used_splits: List[str] = []

            for path in sorted(json_paths):
                with open(path, "r") as f:
                    stats = json.load(f)

                split_name = stats.get("split", None)
                if split_name is None:
                    stem = Path(path).stem
                    split_name = stem.split("_")[-1] if stem else "unknown"
                used_splits.append(split_name)

                video_avg_motion = stats.get("video_avg_motion", {})
                all_motion_values.extend(float(v) for v in video_avg_motion.values())

                total_motion_count += int(stats.get("motion_count", 0))
                total_video_count += int(stats.get("video_count", 0))

            combined_buckets = self.construct_avg_motion_buckets(all_motion_values)

            combined_stats = {
                "dataset_split": dataset_split,
                "splits": used_splits,
                "video_count": total_video_count,
                "motion_count": total_motion_count,
                "motion_buckets": combined_buckets,
            }

            combined_stats_path = (
                self.world_annotations_root_dir / f"camera_motion_{dataset_split}_all_splits.json"
            )
            with open(combined_stats_path, "w") as f:
                json.dump(combined_stats, f, indent=2)

            print(f"[camera_motion] Saved combined {dataset_split} stats to {combined_stats_path}")

            # Plot
            import matplotlib.pyplot as plt

            labels = list(combined_buckets.keys())
            counts = [combined_buckets[l] for l in labels]
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts, width=0.6, edgecolor="black")
            plt.xlabel("Average Camera Motion Magnitude Buckets")
            plt.ylabel("Number of Videos")
            plt.title(f"{dataset_split.capitalize()} Dataset Camera Motion Buckets (all splits)")
            plt.grid(axis="y")
            out_png = self.world_annotations_root_dir / f"{dataset_split}_camera_motion_buckets_all_splits.png"
            plt.savefig(str(out_png))
            plt.close()
            print(f"[camera_motion] Saved combined {dataset_split} plot to {out_png}")

        train_json_paths = sorted(self.world_annotations_root_dir.glob("camera_motion_train_split_*.json"))
        test_json_paths = sorted(self.world_annotations_root_dir.glob("camera_motion_test_split_*.json"))

        if not train_json_paths and not test_json_paths:
            print("[camera_motion] No per-split camera motion jsons found to combine.")
            return

        _combine_one_group(train_json_paths, dataset_split="train")
        _combine_one_group(test_json_paths, dataset_split="test")

# --------------------------------------------------------------------------------------
# Dataset + orchestration
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
        num_workers=0,
    )
    return train_dataset, test_dataset, dataloader_train, dataloader_test


def run_statistics_estimations(
    *,
    ag_root_directory: str,
    dynamic_scene_dir_path: str,
    split: str,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    run_mismatch: bool = True,
    run_camera_motion: bool = True,
    run_combine_camera_motion: bool = False,
) -> Dict[str, Any]:
    """
    Single entrypoint that calls each estimation procedure and returns collected results.
    """
    results: Dict[str, Any] = {}

    if run_mismatch:
        mismatch_est = MismatchStatsEstimator(
            ag_root_directory=ag_root_directory,
            dynamic_scene_dir_path=dynamic_scene_dir_path,
        )
        results["mismatch"] = mismatch_est.estimate_mismatched_annotations(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )

    if run_camera_motion:
        cam_est = CameraMotionStatsEstimator(
            ag_root_directory=ag_root_directory,
            dynamic_scene_dir_path=dynamic_scene_dir_path,
        )
        results["camera_motion"] = cam_est.estimate_camera_motion_in_dataset(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            split=split,
        )

    if run_combine_camera_motion:
        comb = CameraMotionStatsCombiner(
            ag_root_directory=ag_root_directory,
            dynamic_scene_dir_path=dynamic_scene_dir_path,
        )
        comb.combine_camera_motion_stats_all_splits()
        results["camera_motion_combined"] = True

    return results


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="FrameToWorld stats estimators (refactored).")
    p.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    p.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    p.add_argument("--split", type=str, default="04")

    # what to run
    p.add_argument("--run_mismatch", action="store_true", help="run 2D/3D mismatch estimation")
    p.add_argument("--run_camera_motion", action="store_true", help="run camera motion estimation for --split")
    p.add_argument("--run_combine_camera_motion", action="store_true", help="combine camera motion stats across splits")
    p.add_argument(
        "--run_all",
        action="store_true",
        help="run mismatch + camera_motion + combine_camera_motion",
    )
    return p.parse_args()


def main():
    args = parse_args()
    _, _, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)

    if args.run_all:
        run_mismatch = True
        run_camera_motion = True
        run_combine = True
    else:
        # If user didn't specify anything, default to combine (matching your old __main__)
        any_flag = args.run_mismatch or args.run_camera_motion or args.run_combine_camera_motion
        run_mismatch = args.run_mismatch
        run_camera_motion = args.run_camera_motion
        run_combine = args.run_combine_camera_motion
        if not any_flag:
            run_combine = True

    results = run_statistics_estimations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        split=args.split,
        train_dataloader=dataloader_train,
        test_dataloader=dataloader_test,
        run_mismatch=run_mismatch,
        run_camera_motion=run_camera_motion,
        run_combine_camera_motion=run_combine,
    )

    print("\n=== Done ===")
    for k in results.keys():
        print(f"- {k}")


if __name__ == "__main__":
    main()
