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
        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)
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
            return None

        with open(out_path, "rb") as f:
            video_3d_annotations = pickle.load(f)
        return video_3d_annotations

    def generate_video_bb_annotations(
            self,
            video_id: str,
            video_id_gt_annotations: Dict,
            video_id_gdino_annotations: Dict,
            video_id_3d_bbox_predictions: Dict,
            visualize: bool = False
    ) -> None:
        """
        Video_id_3D_bbox_predictions: {
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
        video_to_frame_to_label_mask, all_static_labels, all_dynamic_labels = self._create_label_wise_masks_map(
            video_id=video_id,
            gt_annotations=video_id_gt_annotations
        )

        # Output structure for storing frame annotations
        out_frames: Dict[str, Dict[str, Any]] = {}

        # Save to disk
        out_path = self.bbox_4d_root_dir / f"{video_id[:-4]}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "video_id": video_id,
                "frames": out_frames
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[{video_id}] Saved 4D world annotations to {out_path}")

    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        """Load 3D points from dynamic scene predictions."""
        video_dynamic_3d_scene_path = self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        video_dynamic_predictions = _npz_open(video_dynamic_3d_scene_path)

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
        annotated_frame_id_list = [f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith('.png')]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))
        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        video_sampled_frames_npy_path = self.sampled_frames_idx_root_dir / f"{video_id[:-4]}.npy"
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_first_frame_id)
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_last_frame_id)
        sample_idx = list(range(an_first_id_in_vid_sam_frame_id_list, an_last_id_in_vid_sam_frame_id_list + 1))

        assert S == len(sample_idx)

        sampled_idx_frame_name_map = {}
        frame_name_sampled_idx_map = {}
        for idx_in_s, frame_idx in enumerate(sample_idx):
            frame_name = f"{video_sampled_frame_id_list[frame_idx]:06d}.png"
            sampled_idx_frame_name_map[idx_in_s] = frame_name
            frame_name_sampled_idx_map[frame_name] = idx_in_s

        annotated_idx_in_sampled_idx = []
        for frame_name in annotated_frame_id_list:
            if frame_name in frame_name_sampled_idx_map:
                annotated_idx_in_sampled_idx.append(frame_name_sampled_idx_map[frame_name])

        points_sub = points[annotated_idx_in_sampled_idx]
        conf_sub = conf[annotated_idx_in_sampled_idx] if conf is not None else None
        stems_sub = [sampled_idx_frame_name_map[idx][:-4] for idx in annotated_idx_in_sampled_idx]
        colors_sub = colors[annotated_idx_in_sampled_idx]
        camera_poses_sub = camera_poses[annotated_idx_in_sampled_idx]

        return {
            "points": points_sub,
            "conf": conf_sub,
            "frame_stems": stems_sub,
            "colors": colors_sub,
            "camera_poses": camera_poses_sub
        }

    def _create_label_wise_masks_map(
            self,
            video_id: str,
            gt_annotations
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], set, set]:
        """Create label-wise masks for the video."""
        video_to_frame_to_label_mask: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        frame_stems = []
        for frame_items in gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]
            stem = Path(frame_name).stem
            frame_stems.append(stem)

        frame_map: Dict[str, Dict[str, np.ndarray]] = {}
        frame_map, all_static_labels = self._update_frame_map(
            frame_stems=frame_stems,
            video_id=video_id,
            frame_map=frame_map,
            is_static=True
        )
        frame_map, all_dynamic_labels = self._update_frame_map(
            frame_stems=frame_stems,
            video_id=video_id,
            frame_map=frame_map,
            is_static=False
        )

        if frame_map:
            video_to_frame_to_label_mask[video_id] = frame_map

        return video_to_frame_to_label_mask, all_static_labels, all_dynamic_labels

    def _update_frame_map(
            self,
            frame_stems: List[str],
            video_id: str,
            frame_map: Dict[str, Dict[str, np.ndarray]],
            is_static: bool
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], set]:
        """Update frame map with masks from image-based and video-based segmentation."""
        all_labels = set()
        for stem in frame_stems:
            lbls = self._labels_for_frame(video_id, stem, is_static)
            if not lbls:
                continue
            all_labels.update(lbls)
            if stem not in frame_map:
                frame_map[stem] = {}
            for lbl in lbls:
                m = self._get_union_mask(video_id, stem, lbl, is_static)
                if m is not None:
                    frame_map[stem][lbl] = m
        return frame_map, all_labels

    def _labels_for_frame(self, video_id: str, stem: str, is_static: bool) -> List[str]:
        """Get labels for a specific frame from mask files."""
        lbls = set()
        if is_static:
            image_root_dir_list = [self.static_masks_im_dir_path, self.static_masks_vid_dir_path]
        else:
            image_root_dir_list = [self.dynamic_masks_im_dir_path, self.dynamic_masks_vid_dir_path]

        for root in image_root_dir_list:
            vdir = root / video_id
            if not vdir.exists():
                continue
            for fn in os.listdir(vdir):
                if not fn.endswith(".png"):
                    continue
                if "__" in fn:
                    st, lbl = fn.split("__", 1)
                    lbl = lbl.rsplit(".png", 1)[0]
                    if st == stem:
                        lbls.add(lbl)
        return sorted(lbls)

    def _get_union_mask(self, video_id: str, stem: str, label: str, is_static: bool) -> Optional[np.ndarray]:
        """Get union of image-based and video-based masks for a label."""
        if is_static:
            im_p = self.static_masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.static_masks_vid_dir_path / video_id / f"{stem}__{label}.png"
        else:
            im_p = self.dynamic_masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.dynamic_masks_vid_dir_path / video_id / f"{stem}__{label}.png"

        m_im = cv2.imread(str(im_p), cv2.IMREAD_GRAYSCALE) if im_p.exists() else None
        m_vd = cv2.imread(str(vd_p), cv2.IMREAD_GRAYSCALE) if vd_p.exists() else None

        if m_im is None and m_vd is None:
            return None
        if m_im is None:
            m = (m_vd > 127)
        elif m_vd is None:
            m = (m_im > 127)
        else:
            m = (m_im > 127) | (m_vd > 127)
        return m.astype(bool)

    def _match_gdino_to_gt(
            self,
            gt_label: str,
            gt_xyxy: List[float],
            gd_boxes: List[List[float]],
            gd_labels: List[str],
            gd_scores: List[float],
            iou_thr: float = 0.3,
    ) -> List[float]:
        """Match GDINO detections to GT boxes using IoU."""
        candidates = [
            (b, s) for b, l, s in zip(gd_boxes, gd_labels, gd_scores)
            if (l == gt_label)
        ]
        if not candidates:
            return gt_xyxy

        # Keep boxes with IoU >= iou_thr (or top-1 if none pass)
        passing = [b for (b, s) in candidates if self._iou_xyxy(b, gt_xyxy) >= iou_thr]
        if passing:
            box = self._union_boxes_xyxy(passing)
            return box if box is not None else gt_xyxy

        # No IoU pass -> pick highest-score of same label
        best = max(candidates, key=lambda t: t[1])[0]
        return best

    @staticmethod
    def _xywh_to_xyxy(b):
        """Convert [x,y,w,h] to [x1,y1,x2,y2]."""
        x, y, w, h = [float(v) for v in b]
        return [x, y, x + w, y + h]

    @staticmethod
    def _area_xyxy(b):
        """Calculate area of a bounding box in xyxy format."""
        x1, y1, x2, y2 = b
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        """Calculate IoU between two boxes in xyxy format."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = FrameToWorldAnnotations._area_xyxy(a) + FrameToWorldAnnotations._area_xyxy(b) - inter
        return inter / max(ua, 1e-8)

    @staticmethod
    def _union_boxes_xyxy(boxes: List[List[float]]) -> Optional[List[float]]:
        """Compute union of multiple boxes."""
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return [x1, y1, x2, y2]

    @staticmethod
    def _mask_from_bbox(h: int, w: int, xyxy: List[float]) -> np.ndarray:
        """Create a binary mask from a bounding box."""
        m = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
        return m

    @staticmethod
    def _resize_mask_to(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        """Resize a mask to target dimensions."""
        th, tw = target_hw
        if mask.shape == (th, tw):
            return mask.astype(bool)
        return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)

    @staticmethod
    def _finite_and_nonzero(pts: np.ndarray) -> np.ndarray:
        """Identify finite and non-zero points."""
        good = np.isfinite(pts).all(axis=-1)
        if pts.ndim == 2:  # (N,3)
            nz = np.linalg.norm(pts, axis=-1) > 1e-12
        else:  # (H,W,3)
            nz = np.linalg.norm(pts, axis=-1) > 1e-12
        return good & nz

    def check_2d_3d_annotation_consistency(
            self,
            video_id: str,
            video_id_gt_annotations: Dict,
            video_id_3d_bbox_predictions: Dict
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
            bboxes_3d = frame_data.get("bboxes_3d", [])
            if bboxes_3d:
                valid_3d_frames.add(frame_name)

        # Check the total number of mismatched instances.
        mismatched_frames = gt_2d_frames.symmetric_difference(valid_3d_frames)
        return len(mismatched_frames)

    def estimate_mismatched_annotations(self, train_dataloader: DataLoader, test_dataloader: DataLoader) -> None:
        def _estimate_dataset_mismatches(dataloader: DataLoader) -> Dict[str, int]:
            mismatch_count_map = {}
            missing_3D_annotations = 0
            for data in tqdm(dataloader, desc="Estimating mismatched annotations"):
                video_id = data['video_id']
                video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
                video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)

                if video_id_3d_bbox_predictions is None:
                    missing_3D_annotations += 1
                    continue

                mismatched_count = self.check_2d_3d_annotation_consistency(
                    video_id,
                    video_id_gt_annotations,
                    video_id_3d_bbox_predictions
                )
                mismatch_count_map[mismatched_count] = mismatch_count_map.get(mismatched_count, 0) + 1
            return mismatch_count_map, missing_3D_annotations

        print("Estimating mismatched annotations in training dataset...")
        train_mismatch_count_map, train_missing_3D = _estimate_dataset_mismatches(train_dataloader)

        print("Estimating mismatched annotations in testing dataset...")
        test_mismatch_count_map, test_missing_3D = _estimate_dataset_mismatches(test_dataloader)

        # Create plots to display a histogram of mismatched counts
        import matplotlib.pyplot as plt
        def plot_mismatch_histogram(mismatch_count_map: Dict[int, int], title: str, output_path: str) -> None:
            counts = list(mismatch_count_map.keys())
            frequencies = [mismatch_count_map[k] for k in counts]
            plt.figure(figsize=(10, 6))
            plt.bar(counts, frequencies, width=0.8, color='skyblue', edgecolor='black')
            plt.xlabel('Number of Mismatched Frames')
            plt.ylabel('Number of Videos')
            plt.title(title)
            plt.xticks(counts)
            plt.grid(axis='y')
            plt.savefig(output_path)
            plt.close()

        plot_mismatch_histogram(
            train_mismatch_count_map,
            "Training Dataset Mismatched Annotations",
            self.world_annotations_root_dir / "train_mismatched_annotations_histogram.png"
        )
        plot_mismatch_histogram(
            test_mismatch_count_map,
            "Testing Dataset Mismatched Annotations",
            self.world_annotations_root_dir / "test_mismatched_annotations_histogram.png"
        )

    def generate_sample_gt_world_4D_annotations(self, video_id: str) -> None:
        video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
        video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
        video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)
        self.generate_video_bb_annotations(
            video_id,
            video_id_gt_annotations,
            video_id_gdino_annotations,
            video_id_3d_bbox_predictions,
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

def main_estimate_mismatches():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
    frame_to_world_generator.estimate_mismatched_annotations(
        train_dataloader=dataloader_train,
        test_dataloader=dataloader_test
    )


if __name__ == "__main__":
    # main()
    # main_sample()
    main_estimate_mismatches()