#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import DataLoader

from annotation_utils import (
    _load_pkl_if_exists,
    _npz_open,
    _is_empty_array,
)
from dataloader.standard.action_genome.ag_dataset import StandardAG

# sys.path.insert(0, os.path.dirname(__file__) + "/..")


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
            video_id_gt_annotations,
            video_id_gdino_annotations,
            video_id_3d_bbox_predictions,
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

        # Create a label-wise masks map for segmentation
        (
            video_to_frame_to_label_mask,
            all_static_labels,
            all_dynamic_labels,
        ) = self._create_label_wise_masks_map(
            video_id=video_id, gt_annotations=video_id_gt_annotations
        )

        # Output structure for storing frame annotations
        video_3dgt_path = self.bbox_3d_root_dir / f"{video_id[:-4]}.pkl"
        if not video_3dgt_path.exists():
            print(f"[{video_id}] 3D bbox annotations not found at {video_3dgt_path}")
            return

        # Load the pkl file with 3D bbox predictions
        with open(video_3dgt_path, "rb") as f:
            video_3dgt = pickle.load(f)

        frame_3dbb_map = video_3dgt["frames"]
        per_frame_sims = video_3dgt["per_frame_sims"]
        global_floor_sim = video_3dgt["global_floor_sim"]
        primary_track_id_0 = video_3dgt["primary_track_id_0"]
        frame_bbox_meshes = video_3dgt["frame_bbox_meshes"]
        gv = video_3dgt["gv"]
        gf = video_3dgt["gf"]
        gc = video_3dgt["gc"]

        # frame_3dbb_map structure: {
        #   "000000.png": objects
        # }
        # Objects structure: [
        #   {
        #       "label": [...],
        #       "gt_bbox_xyxy": [...],
        #       "aabb_floor_aligned": [...],
        #       "multi_scale_candidates": [...],
        #       ...
        #   },
        #   ...
        # ]

        print(f"[{video_id}] Saved 4D world annotations to {video_3dgt_path}")

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
            # None, None, None,
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


if __name__ == "__main__":
    # main()
    main_sample()
