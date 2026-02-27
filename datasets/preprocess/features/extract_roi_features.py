#!/usr/bin/env python3
"""
ROI Feature Extraction for PredCls
====================================

Extracts pre-computed ROI features from trained DINOv2/v3 detectors for all
annotated objects (GT + GDino-filled missing objects) per frame.

The script:
  1. Loads a trained DinoV3Monocular3D detector from checkpoint.
  2. For each video, collects GT bboxes + GDino detections for missing labels.
  3. Preprocesses images identically to ag_dataset_3d.py (Pi3-compatible resize,
     DINOv2 normalization, bbox scaling).
  4. Runs frozen backbone+FPN → ROI pooling → box_head to get 1024-d features.
  5. Saves per-video PKL files.

Usage:
    python datasets/preprocess/features/extract_roi_features.py \
        --config configs/features/predcls/ex_roi_feat_v1_dinov2b_saurabh.yaml

    # Override via CLI:
    python datasets/preprocess/features/extract_roi_features.py \
        --config configs/features/predcls/ex_roi_feat_v1_dinov2b_saurabh.yaml \
        --model v2l --ckpt checkpoint_50 --video 001YG.mp4
"""

import argparse
import gc
import logging
import math
import os
import pickle
import re
import sys
import time
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
)

logger = logging.getLogger("roi_feature_extraction")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_TO_DIR = {"v2": "dinov2b", "v2l": "dinov2l", "v3l": "dinov3l"}

# Label normalization: GT compound names → short form
# (matches the normalization used in GDino detection scripts)
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}

# Reverse mapping: GDino detection scripts expand compound GT classes into
# individual words for the detection prompt (e.g., "closet/cabinet" → "closet", "cabinet").
# After detection, _normalize_label in base_ag_actor.py only handles the compound form
# and article stripping, so PKLs can contain expanded labels like "cabinet", "glass",
# "bottle", "notebook", "couch", "camera". This map normalizes them back to GT short forms.
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
# This mirrors `fetch_object_classes()` in BaseAG where compound names
# like 'closet/cabinet' occupy index 9. The short forms produced by
# _normalize_label (e.g., 'closet') map back to the same class index.
LABEL_TO_CLASSIDX: Dict[str, int] = {}
for _idx, _name in enumerate(DATASET_CLASSNAMES):
    if _idx == 0:
        continue  # skip __background__
    LABEL_TO_CLASSIDX[_name] = _idx  # e.g., 'closet/cabinet' → 9
    # Also register the normalized short form
    _norm = LABEL_NORMALIZE_MAP.get(_name, _name)
    LABEL_TO_CLASSIDX[_norm] = _idx  # e.g., 'closet' → 9

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
    # Strip articles (matches base_ag_actor._normalize_label)
    label = re.sub(r"^(a|an|the)\s+", "", label)
    # First try GT compound → short form
    label = LABEL_NORMALIZE_MAP.get(label, label)
    # Then try GDino expanded → GT short form
    label = GDINO_LABEL_TO_GT_LABEL.get(label, label)
    return label


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
# ROI Feature Extractor
# ---------------------------------------------------------------------------

class ROIFeatureExtractor:
    """
    Extracts ROI features from a trained DinoV3Monocular3D detector.

    Image preprocessing mirrors ag_dataset_3d.py exactly:
      1. Load image → uint8 RGB
      2. Resize to Pi3-compatible target size (_compute_target_size)
      3. Convert to float32 / 255.0
      4. DINOv2 normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
      5. Scale bboxes by (scale_x, scale_y) to match resized dims

    The model's _NoOpRCNNTransform only batches/pads — no further resize or normalization.
    """

    def __init__(self, cfg: ExtractConfig):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

        # Determine output directory
        model_dir = MODEL_TO_DIR.get(cfg.model, cfg.model)
        if cfg.output_dir:
            self.output_dir = Path(cfg.output_dir)
        else:
            self.output_dir = Path(cfg.data_path) / "features" / "roi_features" / "predcls" / model_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Also create sgdet placeholder
        sgdet_dir = Path(cfg.data_path) / "features" / "roi_features" / "sgdet"
        os.makedirs(sgdet_dir, exist_ok=True)

        # Setup logging to file (one log per config)
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
        else:
            self.roi_pooler = self.model.roi_heads.box_roi_pool
            self.box_head = self.model.roi_heads.box_head

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

        Pipeline:
          1. Load image as RGB uint8
          2. Compute Pi3-compatible target size via _compute_target_size
          3. Resize to target dims
          4. Convert to float32 / 255.0
          5. Apply DINOv2 normalization

        Returns:
            (img_chw, orig_w, orig_h, target_w, target_h) or None if image not found.
        """
        img_path = self.frames_path / video_id / frame_file
        if not img_path.exists():
            return None

        # Step 1: Load image (BGR → RGB)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Step 2: Compute Pi3-compatible target size
        target_w, target_h = _compute_target_size(
            orig_w, orig_h, self.cfg.pixel_limit, self.cfg.patch_size
        )

        # Step 3: Resize to target dims
        img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Step 4: Convert to float32 / 255.0, then to tensor (H, W, 3) → (3, H, W)
        img_f32 = img_resized.astype(np.float32) / 255.0
        img_chw = torch.from_numpy(img_f32).permute(2, 0, 1).contiguous()

        # Step 5: DINOv2 normalization (same as ag_dataset_3d.py)
        img_chw = TF.normalize(img_chw, mean=list(DINO_IMAGE_MEAN), std=list(DINO_IMAGE_STD))

        return img_chw, orig_w, orig_h, target_w, target_h

    # ------------------------------------------------------------------
    # BBox collection
    # ------------------------------------------------------------------

    def _collect_bboxes_for_frame(
        self,
        gt_frame_items: List[Dict[str, Any]],
        gdino_frame: Optional[Dict[str, Any]],
        threshold: float,
        video_gt_labels: Optional[set] = None,
    ) -> Tuple[List[List[float]], List[str], List[str], List[float]]:
        """
        Collect all bboxes for a frame: GT annotations + GDino fill for missing labels.

        GT bbox formats (from BaseAG.build_dataset):
          - person_bbox: already xyxy format (x1, y1, x2, y2)
          - object bbox:  already converted from xywh → xyxy by BaseAG.build_dataset()
                          stored as item["bbox"] = np.array([x1, y1, x2, y2])

        Returns:
            bboxes_xyxy: list of [x1, y1, x2, y2] in ORIGINAL image coordinates
            labels: list of normalized label strings
            sources: list of "gt" or "gdino"
            gdino_scores: list of float (-1.0 for GT)
        """
        bboxes_xyxy = []
        labels = []
        sources = []
        gdino_scores_out = []

        gt_labels_in_frame = set()

        # Track raw GDino info for logging
        raw_gdino_detections = []
        filtered_gdino_accepted = []
        filtered_gdino_rejected_score = []
        filtered_gdino_rejected_in_gt = []
        filtered_gdino_rejected_not_in_video = []

        for item in gt_frame_items:
            if "person_bbox" in item:
                # Person bbox — stored as xyxy by BaseAG
                pb = item["person_bbox"]
                if isinstance(pb, np.ndarray):
                    pb = pb.reshape(-1, 4).tolist()
                elif isinstance(pb, list) and len(pb) > 0:
                    # Handle nested list format
                    if not isinstance(pb[0], (list, tuple, np.ndarray)):
                        pb = [pb]  # Single person bbox as flat list
                if isinstance(pb, list) and len(pb) > 0:
                    for b in pb:
                        b_list = b if isinstance(b, list) else list(b)
                        bboxes_xyxy.append([float(v) for v in b_list[:4]])
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

                    # Object bbox — already xyxy from BaseAG.build_dataset()
                    bbox = item["bbox"]
                    if isinstance(bbox, np.ndarray):
                        bbox = bbox.tolist()
                    bboxes_xyxy.append([float(v) for v in bbox[:4]])
                    labels.append(label_norm)
                    sources.append("gt")
                    gdino_scores_out.append(-1.0)
                    gt_labels_in_frame.add(label_norm)

        # GDino fill: add detections for labels NOT in GT
        if gdino_frame is not None:
            gd_boxes = gdino_frame.get("boxes", [])
            gd_labels = gdino_frame.get("labels", [])
            gd_scores = gdino_frame.get("scores", [])

            # Group by label, keep highest-score per label
            best_per_label: Dict[str, Tuple[List[float], float]] = {}
            for gd_box, gd_label, gd_score in zip(gd_boxes, gd_labels, gd_scores):
                gd_score = float(gd_score)
                raw_gdino_detections.append((gd_label, gd_score))

                if gd_score < threshold:
                    filtered_gdino_rejected_score.append((gd_label, gd_score))
                    continue

                gd_label_norm = _normalize_label(gd_label)

                # Skip labels already in GT for this frame
                if gd_label_norm in gt_labels_in_frame:
                    filtered_gdino_rejected_in_gt.append((gd_label, gd_label_norm, gd_score))
                    continue

                # Only include GDino labels that exist in the video's GT annotations
                if video_gt_labels is not None and gd_label_norm not in video_gt_labels:
                    filtered_gdino_rejected_not_in_video.append((gd_label, gd_label_norm, gd_score))
                    continue

                if (gd_label_norm not in best_per_label
                        or gd_score > best_per_label[gd_label_norm][1]):
                    if hasattr(gd_box, 'tolist'):
                        gd_box = gd_box.tolist()
                    best_per_label[gd_label_norm] = ([float(v) for v in gd_box], gd_score)

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
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_features_for_video(
        self,
        video_id: str,
        gt_annotations: List[List[Dict[str, Any]]],
        frame_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract ROI features for all annotated frames of a video.
        """
        # Load GDino detections (keyed by bare frame name like "000063.png")
        gdino_preds = _load_gdino_predictions(self.cfg.data_path, video_id)

        # Pre-compute the set of unique GT labels across ALL frames of this video.
        # GDino detections will only be included if their label belongs to this set.
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

        logger.info(f"[{video_id}] Video GT labels ({len(video_gt_labels)}): {sorted(video_gt_labels)}")
        logger.info(f"[{video_id}] GDino detections available: {gdino_preds is not None}")
        if gdino_preds is not None:
            logger.info(f"[{video_id}] GDino frames available: {len(gdino_preds)}")

        # Per-video stats
        video_total_gt = 0
        video_total_gdino_added = 0
        video_total_gdino_raw = 0
        video_total_gdino_rejected_score = 0
        video_total_gdino_rejected_in_gt = 0
        video_total_gdino_rejected_not_in_video = 0

        frames_data = {}

        for frame_items in gt_annotations:
            if not frame_items:
                continue

            # Get frame name from the first item (format: "video_id/frame.png")
            first_item = frame_items[0]
            frame_relpath = first_item.get("frame", "")
            if "/" in frame_relpath:
                frame_file = frame_relpath.split("/")[-1]
            else:
                frame_file = frame_relpath

            if not frame_file:
                continue

            # Load and preprocess image (Pi3-compatible resize + DINOv2 normalization)
            result = self._load_and_preprocess_frame(video_id, frame_file)
            if result is None:
                continue
            img_chw, orig_w, orig_h, target_w, target_h = result

            # Compute bbox scale factors (same as ag_dataset_3d.py lines 397-398)
            scale_x = target_w / float(orig_w)
            scale_y = target_h / float(orig_h)

            # Collect bboxes (in original image coordinates)
            gdino_frame = gdino_preds.get(frame_file, None) if gdino_preds else None
            (
                bboxes_xyxy, labels, sources, gdino_scores,
                raw_gdino_dets, accepted_gdino, rej_score, rej_in_gt, rej_not_in_video
            ) = self._collect_bboxes_for_frame(
                frame_items, gdino_frame, self.cfg.gdino_score_threshold,
                video_gt_labels=video_gt_labels,
            )

            # Log per-frame details
            gt_labels_this_frame = [l for l, s in zip(labels, sources) if s == "gt"]
            gdino_labels_this_frame = [l for l, s in zip(labels, sources) if s == "gdino"]
            logger.debug(
                f"[{video_id}][{frame_file}] GT objects ({len(gt_labels_this_frame)}): {gt_labels_this_frame}"
            )
            if raw_gdino_dets:
                logger.debug(
                    f"[{video_id}][{frame_file}] Raw GDino detections ({len(raw_gdino_dets)}): "
                    f"{[(lbl, f'{sc:.3f}') for lbl, sc in raw_gdino_dets]}"
                )
            if rej_score:
                logger.debug(
                    f"[{video_id}][{frame_file}] GDino rejected (low score): "
                    f"{[(lbl, f'{sc:.3f}') for lbl, sc in rej_score]}"
                )
            if rej_in_gt:
                logger.debug(
                    f"[{video_id}][{frame_file}] GDino rejected (already in GT): "
                    f"{[(raw, norm, f'{sc:.3f}') for raw, norm, sc in rej_in_gt]}"
                )
            if rej_not_in_video:
                logger.debug(
                    f"[{video_id}][{frame_file}] GDino rejected (not in video GT labels): "
                    f"{[(raw, norm, f'{sc:.3f}') for raw, norm, sc in rej_not_in_video]}"
                )
            if accepted_gdino:
                logger.debug(
                    f"[{video_id}][{frame_file}] GDino ACCEPTED ({len(accepted_gdino)}): "
                    f"{[(lbl, f'{sc:.3f}') for lbl, sc in accepted_gdino]}"
                )

            # Accumulate video stats
            video_total_gt += len(gt_labels_this_frame)
            video_total_gdino_added += len(accepted_gdino)
            video_total_gdino_raw += len(raw_gdino_dets)
            video_total_gdino_rejected_score += len(rej_score)
            video_total_gdino_rejected_in_gt += len(rej_in_gt)
            video_total_gdino_rejected_not_in_video += len(rej_not_in_video)

            if not bboxes_xyxy:
                continue

            # Scale bboxes to match resized image dims + clamp to bounds
            # (same as ag_dataset_3d.py lines 411-414, 429-432)
            scaled_bboxes = []
            valid_indices = []
            for i, (x1, y1, x2, y2) in enumerate(bboxes_xyxy):
                x1_s = max(0.0, min(x1 * scale_x, target_w - 1))
                x2_s = max(0.0, min(x2 * scale_x, target_w - 1))
                y1_s = max(0.0, min(y1 * scale_y, target_h - 1))
                y2_s = max(0.0, min(y2 * scale_y, target_h - 1))
                # Filter degenerate boxes (same threshold as ag_dataset_3d.py line 416)
                if (x2_s - x1_s) >= 1 and (y2_s - y1_s) >= 1:
                    scaled_bboxes.append([x1_s, y1_s, x2_s, y2_s])
                    valid_indices.append(i)

            if not scaled_bboxes:
                continue

            # Filter labels/sources/scores to match valid bboxes
            labels = [labels[i] for i in valid_indices]
            sources = [sources[i] for i in valid_indices]
            gdino_scores = [gdino_scores[i] for i in valid_indices]
            bboxes_xyxy_orig = [bboxes_xyxy[i] for i in valid_indices]

            # Pass image through _NoOpRCNNTransform (batching/padding only, no resize/norm)
            images, _ = self.transform([img_chw], None)

            # Run backbone + FPN
            features = self.backbone(images.tensors.to(self.device))
            if isinstance(features, torch.Tensor):
                features = {"0": features}

            # Prepare bbox tensor (already in resized coordinates)
            bboxes_tensor = torch.tensor(scaled_bboxes, dtype=torch.float32, device=self.device)

            # ROI Pool + box_head → 1024-d features
            roi_features = self.roi_pooler(features, [bboxes_tensor], [images.image_sizes[0]])
            roi_features = self.box_head(roi_features)  # (N, 1024)

            # ---- Label IDs (AG class indices) ----
            label_ids = [LABEL_TO_CLASSIDX.get(l, 0) for l in labels]

            # ---- Union Features for (person, object) pairs ----
            person_indices = [i for i, l in enumerate(labels) if l == "person"]
            object_indices = [i for i, l in enumerate(labels) if l != "person"]

            union_features_np = None
            pair_indices = []

            if person_indices and object_indices:
                union_bboxes = []
                for p_idx in person_indices:
                    p_box = scaled_bboxes[p_idx]
                    p_label_id = label_ids[p_idx]  # always 1 (person)
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

                logger.debug(
                    f"[{video_id}][{frame_file}] Union features: {len(pair_indices)} pairs "
                    f"(persons={person_indices}, objects={object_indices})"
                )

            # Store
            frame_entry = {
                "roi_features": roi_features.cpu().numpy().astype(self.store_dtype),
                "bboxes_xyxy": np.array(bboxes_xyxy_orig, dtype=np.float32),
                "labels": labels,
                "label_ids": label_ids,
                "sources": sources,
                "gdino_scores": gdino_scores,
                "pair_indices": pair_indices,
            }
            if union_features_np is not None:
                frame_entry["union_features"] = union_features_np

            frames_data[frame_file] = frame_entry

        if not frames_data:
            logger.info(f"[{video_id}] No frames produced features")
            return None

        # Log per-video summary
        logger.info(
            f"[{video_id}] SUMMARY: frames={len(frames_data)}, "
            f"gt_objects={video_total_gt}, gdino_added={video_total_gdino_added}, "
            f"gdino_raw={video_total_gdino_raw}, "
            f"gdino_rej_score={video_total_gdino_rejected_score}, "
            f"gdino_rej_in_gt={video_total_gdino_rejected_in_gt}, "
            f"gdino_rej_not_in_video={video_total_gdino_rejected_not_in_video}"
        )

        return {
            "video_id": video_id,
            "model": self.cfg.model,
            "feature_dim": 1024,
            "checkpoint": os.path.join(
                self.cfg.working_dir or "", self.cfg.experiment_name,
                self.cfg.ckpt or "pretrained_only",
            ),
            "frames": frames_data,
        }

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
                out_path = self.output_dir / f"{video_stem}.pkl"

                # Skip if exists
                if out_path.exists() and not self.cfg.overwrite:
                    results[video_id] = True
                    global_stats["skipped_videos"] += 1
                    continue

                global_stats["total_videos"] += 1

                try:
                    result = self._extract_features_for_video(
                        video_id, gt_annotations, frame_names,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract ROI features from trained DINOv2/v3 detectors",
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

        # Handle Optional types
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


def _setup_logging(log_dir: Path, config_path: str) -> Path:
    """
    Set up logging with both file and console handlers.

    File handler: DEBUG level (captures all per-frame details)
    Console handler: INFO level (milestones only)

    Returns the log file path.
    """
    config_stem = Path(config_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{config_stem}_{timestamp}.log"

    # Reset any existing handlers
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # File handler — DEBUG level (all details)
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler — INFO level (milestones only)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    logger.addHandler(ch)

    return log_file


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load YAML config
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

    cfg = ExtractConfig(**filtered)

    # Determine output dir early to set up logging before model init
    model_dir = MODEL_TO_DIR.get(cfg.model, cfg.model)
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(cfg.data_path) / "features" / "roi_features" / "predcls" / model_dir
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = _setup_logging(log_dir, config_path)
    print(f"Log file: {log_file}")

    # Log full config
    logger.info(f"Config file: {config_path}")
    logger.info(f"Resolved config: {filtered}")
    logger.info(f"Log file: {log_file}")

    extractor = ROIFeatureExtractor(cfg)
    extractor.extract_all()


if __name__ == "__main__":
    main()

