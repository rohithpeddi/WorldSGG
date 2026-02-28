#!/usr/bin/env python3
"""
ROI Feature Extraction — Shared Base Module
=============================================

Contains all reusable components for ROI feature extraction:
  - Constants (class names, label maps, normalization stats)
  - ExtractConfig dataclass
  - Helper functions (label normalization, GDino loading, image preprocessing)
  - BaseROIFeatureExtractor class with shared model init, preprocessing, and extraction loop

Subclasses (PredCls / SGDet) override `_extract_features_for_video()`.
"""

import argparse
import copy
import gc
import logging
import math
import os
import pickle
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.detector.monocular3d.models.dino_mono_3d import (
    DinoV3Monocular3D,
    _gather_intrinsics,
)

logger = logging.getLogger("roi_feature_extraction")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_TO_DIR = {"v2": "dinov2b", "v2l": "dinov2l", "v3l": "dinov3l"}

# Label normalization: GT compound names → short form
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}

# GDino expanded labels → GT short forms
GDINO_LABEL_TO_GT_LABEL = {
    "cabinet": "closet",
    "glass": "cup",
    "bottle": "cup",
    "notebook": "paper",
    "couch": "sofa",
    "camera": "phone",
}

DATASET_CLASSNAMES = [
    '__background__', 'person', 'bag', 'bed', 'blanket', 'book', 'box',
    'broom', 'chair', 'closet/cabinet', 'clothes', 'cup/glass/bottle',
    'dish', 'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries',
    'laptop', 'light', 'medicine', 'mirror', 'paper/notebook',
    'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich',
    'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel',
    'vacuum', 'window',
]

CATID_TO_NAME = {i: name for i, name in enumerate(DATASET_CLASSNAMES) if i > 0}

# Reverse mapping: normalized label string → AG class index.
LABEL_TO_CLASSIDX: Dict[str, int] = {}
for _idx, _name in enumerate(DATASET_CLASSNAMES):
    if _idx == 0:
        continue  # skip __background__
    LABEL_TO_CLASSIDX[_name] = _idx
    _norm = LABEL_NORMALIZE_MAP.get(_name, _name)
    LABEL_TO_CLASSIDX[_norm] = _idx

# DINOv2 normalization stats (same as ag_dataset_3d.py)
DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExtractConfig:
    """Configuration for ROI feature extraction."""
    # Detector
    model: str = "v2"
    experiment_name: str = "dinov2_separate"
    working_dir: str = ""
    ckpt: Optional[str] = None
    head_3d_mode: str = "separate"
    num_classes: int = 37
    pretrained: bool = True

    # Data
    data_path: str = "/data/rohith/ag"

    # Image processing (must match ag_dataset_3d.py / training config)
    pixel_limit: int = 255000
    patch_size: int = 14

    # Feature extraction
    output_dir: Optional[str] = None
    gdino_score_threshold: float = 0.3
    store_dtype: str = "float16"
    overwrite: bool = False
    gpu: int = 0
    video: Optional[str] = None
    split: Optional[str] = None

    # SGDet-specific
    score_threshold: float = 0.1
    nms_threshold: float = 0.5
    assign_iou_train: float = 0.5
    assign_iou_test: float = 0.3
    use_supply: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pkl_if_exists(path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _is_empty_array(arr) -> bool:
    if arr is None:
        return True
    if isinstance(arr, (list, tuple)):
        return len(arr) == 0
    if hasattr(arr, 'shape'):
        return arr.size == 0
    return False


def _normalize_label(label: str) -> str:
    """
    Normalize a label to the canonical GT short form.
    Handles both:
      1. GT compound class names (e.g. "closet/cabinet" → "closet")
      2. GDino expanded labels (e.g. "cabinet" → "closet", "glass" → "cup")
    """
    label = label.lower().strip()
    label = re.sub(r"^(a|an|the)\s+", "", label)
    label = LABEL_NORMALIZE_MAP.get(label, label)
    label = GDINO_LABEL_TO_GT_LABEL.get(label, label)
    return label


def _scale_bbox_to_pi3(
    bbox: List[float], scale_x: float, scale_y: float,
    target_w: int, target_h: int,
) -> List[float]:
    """Scale a single [x1, y1, x2, y2] bbox from original to Pi-3 coords."""
    x1 = max(0.0, min(float(bbox[0]) * scale_x, target_w - 1))
    y1 = max(0.0, min(float(bbox[1]) * scale_y, target_h - 1))
    x2 = max(0.0, min(float(bbox[2]) * scale_x, target_w - 1))
    y2 = max(0.0, min(float(bbox[3]) * scale_y, target_h - 1))
    return [x1, y1, x2, y2]


def scale_gt_annotations_to_pi3(
    gt_annotations: List[List[Dict[str, Any]]],
    scale_x: float,
    scale_y: float,
    target_w: int,
    target_h: int,
) -> List[List[Dict[str, Any]]]:
    """
    Deep-copy GT annotations with all bboxes scaled to Pi-3 space.

    Scales `person_bbox` and `bbox` fields. All other fields are copied as-is.
    The returned annotations have the exact same structure expected by
    `assign_relations()` and `_collect_bboxes_for_frame()`.
    """
    scaled_annotations = []
    for frame_items in gt_annotations:
        scaled_frame = []
        for item in frame_items:
            new_item = copy.copy(item)  # shallow copy — only bbox fields change

            if "person_bbox" in item and item["person_bbox"] is not None:
                pb = item["person_bbox"]
                if isinstance(pb, np.ndarray):
                    pb = pb.reshape(-1).tolist()
                elif isinstance(pb, (list, tuple)):
                    if isinstance(pb[0], (list, tuple, np.ndarray)):
                        pb = list(pb[0])
                    else:
                        pb = list(pb)
                new_item["person_bbox"] = _scale_bbox_to_pi3(
                    pb, scale_x, scale_y, target_w, target_h
                )

            if "bbox" in item and item["bbox"] is not None:
                bbox = item["bbox"]
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                new_item["bbox"] = np.array(
                    _scale_bbox_to_pi3(bbox, scale_x, scale_y, target_w, target_h),
                    dtype=np.float32,
                )

            scaled_frame.append(new_item)
        scaled_annotations.append(scaled_frame)
    return scaled_annotations


def _compute_target_size(
    orig_w: int, orig_h: int, pixel_limit: int = 255000, patch_size: int = 14
) -> Tuple[int, int]:
    """
    Compute annotation-consistent target size (identical to ag_dataset_3d.py).
    Aspect-ratio preserving, dimensions rounded to multiples of patch_size,
    total pixels <= pixel_limit.
    """
    scale = math.sqrt(pixel_limit / (orig_w * orig_h)) if orig_w * orig_h > 0 else 1
    w_target, h_target = orig_w * scale, orig_h * scale
    k = round(w_target / patch_size)
    m = round(h_target / patch_size)
    while (k * patch_size) * (m * patch_size) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch_size, max(1, m) * patch_size


def _load_gdino_predictions(data_path: str, video_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Load and merge dynamic + static GDino detections for a video.
    Same logic as BBox3DBase.get_video_gdino_annotations().
    """
    dynamic_path = Path(data_path) / "detection" / "gdino_bboxes" / f"{video_id}.pkl"
    static_path = Path(data_path) / "detection" / "gdino_bboxes_static" / f"{video_id}.pkl"

    dyn_preds = _load_pkl_if_exists(dynamic_path) or {}
    stat_preds = _load_pkl_if_exists(static_path) or {}

    if not dyn_preds and not stat_preds:
        return None

    all_frame_names = set(dyn_preds.keys()) | set(stat_preds.keys())
    combined: Dict[str, Dict[str, Any]] = {}

    for frame_name in all_frame_names:
        dyn = dyn_preds.get(frame_name, {"boxes": [], "labels": [], "scores": []})
        stat = stat_preds.get(frame_name, {"boxes": [], "labels": [], "scores": []})

        boxes, labels, scores = [], [], []
        if not _is_empty_array(dyn.get("boxes")):
            boxes.extend(list(dyn["boxes"]))
            labels.extend(list(dyn["labels"]))
            scores.extend(list(dyn["scores"]))
        if not _is_empty_array(stat.get("boxes")):
            boxes.extend(list(stat["boxes"]))
            labels.extend(list(stat["labels"]))
            scores.extend(list(stat["scores"]))

        combined[frame_name] = {"boxes": boxes, "labels": labels, "scores": scores}

    return combined


# ---------------------------------------------------------------------------
# Base ROI Feature Extractor
# ---------------------------------------------------------------------------

class BaseROIFeatureExtractor(ABC):
    """
    Base class for ROI feature extraction from trained DinoV3Monocular3D detectors.

    Subclasses must implement `_extract_features_for_video()`.
    """

    # Subclasses set this to define the mode name (used for output dir).
    MODE_NAME: str = "base"

    def __init__(self, cfg: ExtractConfig):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

        # Determine output directory
        model_dir = MODEL_TO_DIR.get(cfg.model, cfg.model)
        if cfg.output_dir:
            self.output_dir = Path(cfg.output_dir)
        else:
            self.output_dir = Path(cfg.data_path) / "features" / "roi_features" / self.MODE_NAME / model_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logging to file
        self.log_dir = self.output_dir / "logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # Build and load model
        logger.info(f"Building DinoV3Monocular3D (model={cfg.model}, head_3d_mode={cfg.head_3d_mode})")
        print(f"Building DinoV3Monocular3D (model={cfg.model}, head_3d_mode={cfg.head_3d_mode})...")
        self.model = DinoV3Monocular3D(
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            model=cfg.model,
            head_3d_mode=cfg.head_3d_mode,
        )

        # Load trained checkpoint
        if cfg.ckpt is not None:
            ckpt_path = os.path.join(
                cfg.working_dir, cfg.experiment_name, cfg.ckpt, "checkpoint_state.pth"
            )
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint: {ckpt_path}")
                logger.info(f"Loading checkpoint: {ckpt_path}")
                ckpt_state = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict(ckpt_state["model_state_dict"])
                del ckpt_state
                gc.collect()
                print("  ✓ Checkpoint loaded successfully")
                logger.info("Checkpoint loaded successfully")
            else:
                print(f"  ⚠️  Checkpoint not found at {ckpt_path}, using pretrained weights only")
                logger.warning(f"Checkpoint not found at {ckpt_path}, using pretrained weights only")
        else:
            print("  ℹ️  No checkpoint specified, using pretrained weights only")
            logger.info("No checkpoint specified, using pretrained weights only")

        self.model.to(self.device)
        self.model.eval()

        # Extract components for manual ROI feature extraction
        self.backbone = self.model.backbone
        self.transform = self.model.transform  # _NoOpRCNNTransform: batching/padding only

        # Get ROI pooler and box_head from the roi_heads
        if cfg.head_3d_mode == "unified":
            self.roi_pooler = self.model.roi_heads.base.box_roi_pool
            self.box_head = self.model.roi_heads.base.box_head
            self.pred_3d = self.model.roi_heads.pred_3d
        else:
            self.roi_pooler = self.model.roi_heads.box_roi_pool
            self.box_head = self.model.roi_heads.box_head
            self.pred_3d = self.model.head_3d_separate.pred_3d if self.model.head_3d_separate else None

        self.head_3d_version = self.model.head_3d_version

        # Store dtype
        self.store_dtype = np.float16 if cfg.store_dtype == "float16" else np.float32

        # Data paths
        self.frames_path = Path(cfg.data_path) / "frames"

        # Load AG dataset
        print("Loading Action Genome dataset...")
        from dataloader.ag_dataset import StandardAG

        self.train_dataset = StandardAG(
            phase="train", mode="predcls", datasize="large",
            data_path=cfg.data_path,
            filter_nonperson_box_frame=True, filter_small_box=False,
        )
        self.test_dataset = StandardAG(
            phase="test", mode="predcls", datasize="large",
            data_path=cfg.data_path,
            filter_nonperson_box_frame=True, filter_small_box=False,
        )

        print(f"  Train: {len(self.train_dataset)} videos")
        print(f"  Test:  {len(self.test_dataset)} videos")
        print(f"  Output: {self.output_dir}")
        logger.info(f"Dataset loaded: train={len(self.train_dataset)} videos, test={len(self.test_dataset)} videos")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Store dtype: {cfg.store_dtype}")
        logger.info(f"GDino score threshold: {cfg.gdino_score_threshold}")
        logger.info(f"Pixel limit: {cfg.pixel_limit}, Patch size: {cfg.patch_size}")

    # ------------------------------------------------------------------
    # Image loading & preprocessing (mirrors ag_dataset_3d.py exactly)
    # ------------------------------------------------------------------

    def _load_and_preprocess_frame(
        self, video_id: str, frame_file: str
    ) -> Optional[Tuple[torch.Tensor, int, int, int, int]]:
        """
        Load a single frame and preprocess exactly as ag_dataset_3d.py does.

        Returns:
            (img_chw, orig_w, orig_h, target_w, target_h) or None if image not found.
        """
        img_path = self.frames_path / video_id / frame_file
        if not img_path.exists():
            return None

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        target_w, target_h = _compute_target_size(
            orig_w, orig_h, self.cfg.pixel_limit, self.cfg.patch_size
        )

        img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        img_f32 = img_resized.astype(np.float32) / 255.0
        img_chw = torch.from_numpy(img_f32).permute(2, 0, 1).contiguous()

        img_chw = TF.normalize(img_chw, mean=list(DINO_IMAGE_MEAN), std=list(DINO_IMAGE_STD))

        return img_chw, orig_w, orig_h, target_w, target_h

    # ------------------------------------------------------------------
    # BBox collection (GT + GDino fill)
    # ------------------------------------------------------------------

    def _collect_bboxes_for_frame(
        self,
        gt_frame_items: List[Dict[str, Any]],
        gdino_frame: Optional[Dict[str, Any]],
        threshold: float,
        scale_x: float,
        scale_y: float,
        target_w: int,
        target_h: int,
        video_gt_labels: Optional[set] = None,
    ) -> Tuple[List[List[float]], List[str], List[str], List[float],
               List, List, List, List, List]:
        """
        Collect all bboxes for a frame: GT annotations + GDino fill for missing labels.

        All returned bboxes are in **Pi-3 scaled space** (resized image coordinates).
        GT bboxes and GDino bboxes are scaled from original coords to Pi-3 inline.

        Returns:
            bboxes_xyxy, labels, sources, gdino_scores,
            raw_gdino_dets, accepted_gdino, rej_score, rej_in_gt, rej_not_in_video
        """
        bboxes_xyxy = []
        labels = []
        sources = []
        gdino_scores_out = []

        gt_labels_in_frame = set()

        raw_gdino_detections = []
        filtered_gdino_accepted = []
        filtered_gdino_rejected_score = []
        filtered_gdino_rejected_in_gt = []
        filtered_gdino_rejected_not_in_video = []

        for item in gt_frame_items:
            if "person_bbox" in item:
                pb = item["person_bbox"]
                if isinstance(pb, np.ndarray):
                    pb = pb.reshape(-1, 4).tolist()
                elif isinstance(pb, list) and len(pb) > 0:
                    if not isinstance(pb[0], (list, tuple, np.ndarray)):
                        pb = [pb]
                if isinstance(pb, list) and len(pb) > 0:
                    for b in pb:
                        b_list = b if isinstance(b, list) else list(b)
                        bbox_orig = [float(v) for v in b_list[:4]]
                        bbox_pi3 = _scale_bbox_to_pi3(
                            bbox_orig, scale_x, scale_y, target_w, target_h
                        )
                        bboxes_xyxy.append(bbox_pi3)
                        labels.append("person")
                        sources.append("gt")
                        gdino_scores_out.append(-1.0)
                        gt_labels_in_frame.add("person")
            else:
                has_bbox = ("bbox" in item) and (item["bbox"] is not None)
                has_cls = ("class" in item) and (item["class"] is not None)
                if has_bbox and has_cls:
                    cls_idx = int(item["class"])
                    if cls_idx <= 0:
                        continue
                    label = CATID_TO_NAME.get(cls_idx, None)
                    if label is None:
                        continue
                    label_norm = _normalize_label(label)

                    bbox = item["bbox"]
                    if isinstance(bbox, np.ndarray):
                        bbox = bbox.tolist()
                    bbox_orig = [float(v) for v in bbox[:4]]
                    bbox_pi3 = _scale_bbox_to_pi3(
                        bbox_orig, scale_x, scale_y, target_w, target_h
                    )
                    bboxes_xyxy.append(bbox_pi3)
                    labels.append(label_norm)
                    sources.append("gt")
                    gdino_scores_out.append(-1.0)
                    gt_labels_in_frame.add(label_norm)

        # GDino fill: add detections for labels NOT in GT
        if gdino_frame is not None:
            gd_boxes = gdino_frame.get("boxes", [])
            gd_labels = gdino_frame.get("labels", [])
            gd_scores = gdino_frame.get("scores", [])

            best_per_label: Dict[str, Tuple[List[float], float]] = {}
            for gd_box, gd_label, gd_score in zip(gd_boxes, gd_labels, gd_scores):
                gd_score = float(gd_score)
                raw_gdino_detections.append((gd_label, gd_score))

                if gd_score < threshold:
                    filtered_gdino_rejected_score.append((gd_label, gd_score))
                    continue

                gd_label_norm = _normalize_label(gd_label)

                if gd_label_norm in gt_labels_in_frame:
                    filtered_gdino_rejected_in_gt.append((gd_label, gd_label_norm, gd_score))
                    continue

                if video_gt_labels is not None and gd_label_norm not in video_gt_labels:
                    filtered_gdino_rejected_not_in_video.append((gd_label, gd_label_norm, gd_score))
                    continue

                if (gd_label_norm not in best_per_label
                        or gd_score > best_per_label[gd_label_norm][1]):
                    if hasattr(gd_box, 'tolist'):
                        gd_box = gd_box.tolist()
                    # Scale GDino bbox to Pi-3 space
                    gd_box_pi3 = _scale_bbox_to_pi3(
                        [float(v) for v in gd_box], scale_x, scale_y, target_w, target_h
                    )
                    best_per_label[gd_label_norm] = (gd_box_pi3, gd_score)

            for gd_label, (gd_box, gd_score) in best_per_label.items():
                bboxes_xyxy.append(gd_box)
                labels.append(gd_label)
                sources.append("gdino")
                gdino_scores_out.append(gd_score)
                filtered_gdino_accepted.append((gd_label, gd_score))

        return (bboxes_xyxy, labels, sources, gdino_scores_out,
                raw_gdino_detections, filtered_gdino_accepted,
                filtered_gdino_rejected_score, filtered_gdino_rejected_in_gt,
                filtered_gdino_rejected_not_in_video)

    # ------------------------------------------------------------------
    # ROI feature extraction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_roi_and_union_features(
        self,
        img_chw: torch.Tensor,
        scaled_bboxes: List[List[float]],
        labels: List[str],
        label_ids: List[int],
    ) -> Tuple[torch.Tensor, Optional[np.ndarray], List[Tuple[int, int]]]:
        """
        Given a preprocessed image and scaled bboxes, extract ROI features
        and union features for (person, object) pairs.

        Returns:
            roi_features: (N, 1024) tensor on CPU
            union_features_np: (P, 1024) numpy array or None
            pair_indices: list of (person_label_id, object_label_id) tuples
        """
        # Run through transform (batching/padding only)
        images, _ = self.transform([img_chw], None)

        # Backbone + FPN
        features = self.backbone(images.tensors.to(self.device))
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        # ROI Pool + box_head → 1024-d features
        bboxes_tensor = torch.tensor(scaled_bboxes, dtype=torch.float32, device=self.device)
        roi_features = self.roi_pooler(features, [bboxes_tensor], [images.image_sizes[0]])
        roi_features = self.box_head(roi_features)  # (N, 1024)

        # Union features for (person, object) pairs
        person_indices = [i for i, l in enumerate(labels) if l == "person"]
        object_indices = [i for i, l in enumerate(labels) if l != "person"]

        union_features_np = None
        pair_indices = []

        if person_indices and object_indices:
            union_bboxes = []
            for p_idx in person_indices:
                p_box = scaled_bboxes[p_idx]
                p_label_id = label_ids[p_idx]
                for o_idx in object_indices:
                    o_box = scaled_bboxes[o_idx]
                    o_label_id = label_ids[o_idx]
                    union = [
                        min(p_box[0], o_box[0]),
                        min(p_box[1], o_box[1]),
                        max(p_box[2], o_box[2]),
                        max(p_box[3], o_box[3]),
                    ]
                    union_bboxes.append(union)
                    pair_indices.append((p_label_id, o_label_id))

            union_tensor = torch.tensor(union_bboxes, dtype=torch.float32, device=self.device)
            union_pooled = self.roi_pooler(features, [union_tensor], [images.image_sizes[0]])
            union_feats = self.box_head(union_pooled)  # (P, 1024)
            union_features_np = union_feats.cpu().numpy().astype(self.store_dtype)

        return roi_features, union_features_np, pair_indices

    @torch.no_grad()
    def _extract_roi_features_for_bboxes(
        self,
        features: Dict[str, torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        scaled_bboxes: List[List[float]],
    ) -> torch.Tensor:
        """
        Extract ROI features for given bboxes using pre-computed FPN features.

        Args:
            features: FPN feature maps from backbone
            image_sizes: list of (H, W) from the transform
            scaled_bboxes: bboxes in resized image coordinates

        Returns:
            roi_features: (N, 1024) tensor on device
        """
        bboxes_tensor = torch.tensor(scaled_bboxes, dtype=torch.float32, device=self.device)
        roi_features = self.roi_pooler(features, [bboxes_tensor], image_sizes)
        roi_features = self.box_head(roi_features)
        return roi_features

    @torch.no_grad()
    def _predict_3d_for_bboxes(
        self,
        roi_features: torch.Tensor,
        scaled_bboxes: List[List[float]],
        image_sizes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Run the 3D prediction head on ROI features + bboxes.

        Uses default intrinsics estimated from image size (no GT intrinsics).

        Args:
            roi_features: (N, 1024) from box_head
            scaled_bboxes: (N, 4) bboxes in resized image coordinates
            image_sizes: [(H, W)] from the transform

        Returns:
            boxes_3d: (N, 8, 3) predicted 3D corners
        """
        if self.pred_3d is None:
            return torch.empty((len(scaled_bboxes), 8, 3), device=self.device)

        bboxes_tensor = torch.tensor(scaled_bboxes, dtype=torch.float32, device=self.device)

        # Build default intrinsics from image size
        cat_intr = _gather_intrinsics(
            targets=None,
            proposals_or_boxes=[bboxes_tensor],
            device=self.device,
            image_shapes=image_sizes,
        )

        pred_3d, _ = self.pred_3d(roi_features, bboxes_tensor, cat_intr)
        return pred_3d.view(-1, 8, 3)

    # ------------------------------------------------------------------
    # Video GT label set computation
    # ------------------------------------------------------------------

    def _compute_video_gt_labels(self, gt_annotations: List[List[Dict[str, Any]]]) -> set:
        """Pre-compute the set of unique GT labels across all frames of a video."""
        video_gt_labels = set()
        for frame_items in gt_annotations:
            for item in frame_items:
                if "person_bbox" in item:
                    video_gt_labels.add("person")
                elif "class" in item and item["class"] is not None:
                    cls_idx = int(item["class"])
                    if cls_idx > 0:
                        label = CATID_TO_NAME.get(cls_idx, None)
                        if label is not None:
                            video_gt_labels.add(_normalize_label(label))
        return video_gt_labels



    # ------------------------------------------------------------------
    # Abstract method for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_features_for_video(
        self,
        video_id: str,
        gt_annotations: List[List[Dict[str, Any]]],
        frame_names: List[str],
        split_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features for all annotated frames of a video.
        Subclasses implement mode-specific logic (predcls vs sgdet).

        Args:
            video_id: e.g. "001YG.mp4"
            gt_annotations: list of per-frame annotation lists
            frame_names: list of frame file paths
            split_name: "train" or "test"

        Returns:
            dict with video metadata + per-frame features, or None
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Output directory (may differ per split for sgdet)
    # ------------------------------------------------------------------

    def _get_output_dir(self, split_name: str) -> Path:
        """
        Get the output directory for a given split.
        Subclasses can override to use per-split directories.
        """
        return self.output_dir

    # ------------------------------------------------------------------
    # Main extraction loop
    # ------------------------------------------------------------------

    def extract_all(self) -> Dict[str, bool]:
        """Extract ROI features for all videos."""
        results = {}
        global_stats = {
            "total_videos": 0, "success_videos": 0, "skipped_videos": 0,
            "total_frames": 0, "total_gt": 0, "total_gdino_added": 0,
        }
        start_time = time.time()

        datasets_to_process = []
        if self.cfg.split is None or self.cfg.split == "train":
            datasets_to_process.append(("train", self.train_dataset))
        if self.cfg.split is None or self.cfg.split == "test":
            datasets_to_process.append(("test", self.test_dataset))

        logger.info(f"Starting extraction: splits={[s for s, _ in datasets_to_process]}")

        for split_name, dataset in datasets_to_process:
            print(f"\n{'='*60}")
            print(f"Processing {split_name} split ({len(dataset)} videos)")
            print(f"{'='*60}")
            logger.info(f"{'='*60}")
            logger.info(f"Processing {split_name} split ({len(dataset)} videos)")
            logger.info(f"{'='*60}")

            split_output_dir = self._get_output_dir(split_name)
            os.makedirs(split_output_dir, exist_ok=True)

            for idx in tqdm(range(len(dataset)), desc=f"[{split_name}] ROI features"):
                item = dataset[idx]
                video_id = item["video_id"]
                gt_annotations = item["gt_annotations"]
                frame_names = item["frame_names"]

                # Single video filter
                if self.cfg.video and video_id != self.cfg.video:
                    continue

                # Output path
                video_stem = video_id.replace(".mp4", "")
                out_path = split_output_dir / f"{video_stem}.pkl"

                # Skip if exists
                if out_path.exists() and not self.cfg.overwrite:
                    results[video_id] = True
                    global_stats["skipped_videos"] += 1
                    continue

                global_stats["total_videos"] += 1

                try:
                    result = self._extract_features_for_video(
                        video_id, gt_annotations, frame_names, split_name,
                    )
                    if result is not None:
                        with open(out_path, "wb") as f:
                            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                        n_frames = len(result["frames"])
                        n_gt = sum(
                            sum(1 for s in fd["sources"] if s == "gt")
                            for fd in result["frames"].values()
                        )
                        n_gdino = sum(
                            sum(1 for s in fd["sources"] if s == "gdino")
                            for fd in result["frames"].values()
                        )
                        global_stats["success_videos"] += 1
                        global_stats["total_frames"] += n_frames
                        global_stats["total_gt"] += n_gt
                        global_stats["total_gdino_added"] += n_gdino
                        results[video_id] = True
                    else:
                        results[video_id] = False
                        logger.warning(f"[{video_id}] No features extracted")
                except Exception as e:
                    print(f"\n  ⚠️  [{video_id}] Error: {e}")
                    logger.error(f"[{video_id}] Error: {e}", exc_info=True)
                    results[video_id] = False

                # Periodic GPU cache cleanup
                if idx % 50 == 0:
                    torch.cuda.empty_cache()

        success = sum(1 for v in results.values() if v)
        total = len(results)
        elapsed = time.time() - start_time
        summary = (
            f"Done: {success}/{total} videos processed successfully in {elapsed:.1f}s\n"
            f"  Skipped (already exist): {global_stats['skipped_videos']}\n"
            f"  Total frames: {global_stats['total_frames']}\n"
            f"  Total GT objects: {global_stats['total_gt']}\n"
            f"  Total GDino additions: {global_stats['total_gdino_added']}\n"
            f"  Output: {self.output_dir}\n"
            f"  Log: {self.log_dir}"
        )
        print(f"\n{'='*60}")
        print(summary)
        print(f"{'='*60}")
        logger.info(f"{'='*60}")
        logger.info(summary)
        logger.info(f"{'='*60}")

        return results


# ---------------------------------------------------------------------------
# Config loading + CLI
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def build_parser(description: str = "Extract ROI features from trained DINOv2/v3 detectors") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )

    # Add all ExtractConfig fields as CLI overrides
    for f in dataclass_fields(ExtractConfig):
        arg_name = f"--{f.name}"
        field_type = f.type

        origin = getattr(field_type, "__origin__", None)
        type_args = getattr(field_type, "__args__", ())
        is_optional = origin is not None and type(None) in type_args
        inner_type = field_type
        if is_optional:
            inner_type = next((t for t in type_args if t is not type(None)), str)

        if inner_type is bool:
            parser.add_argument(arg_name, type=_str_to_bool, default=None)
        elif inner_type is int:
            parser.add_argument(arg_name, type=lambda v: None if v.lower() == "null" else int(v), default=None)
        elif inner_type is float:
            parser.add_argument(arg_name, type=float, default=None)
        else:
            parser.add_argument(arg_name, type=str, default=None)
    return parser


def _str_to_bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    elif v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def setup_logging(log_dir: Path, config_path: str, mode_name: str = "") -> Path:
    """
    Set up logging with both file and console handlers.

    File handler: DEBUG level (captures all per-frame details)
    Console handler: INFO level (milestones only)

    Args:
        log_dir: directory to store log files
        config_path: path to the config file (used for naming)
        mode_name: pipeline mode (e.g. "predcls", "sgdet") — included in filename

    Returns the log file path.
    """
    config_stem = Path(config_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{mode_name}_" if mode_name else ""
    log_file = log_dir / f"{prefix}{config_stem}_{timestamp}.log"

    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    logger.addHandler(ch)

    return log_file


def parse_config(args) -> ExtractConfig:
    """Parse YAML config + CLI overrides into an ExtractConfig."""
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)

    if not os.path.isfile(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    yaml_cfg = load_yaml_config(config_path)

    # Merge with CLI overrides
    merged = dict(yaml_cfg)
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None:
            merged[key] = val

    # Filter to ExtractConfig fields, handle null
    valid_fields = {f.name for f in dataclass_fields(ExtractConfig)}
    filtered = {}
    for k, v in merged.items():
        if k in valid_fields:
            if v == "null":
                v = None
            filtered[k] = v

    print(f"Config: {filtered}")
    return ExtractConfig(**filtered), config_path
