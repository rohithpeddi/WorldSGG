#!/usr/bin/env python3
"""
ROI Feature Extraction for PredCls
====================================

Extracts pre-computed ROI features from trained DINOv2/v3 detectors for all
annotated objects (GT + GDino-filled missing objects) per frame.

The script:
  1. Loads a trained DinoV3Monocular3D detector from checkpoint.
  2. For each video, collects GT bboxes + GDino detections for missing labels.
  3. Runs frozen backbone+FPN → ROI pooling → box_head to get 1024-d features.
  4. Saves per-video PKL files.

Usage:
    python datasets/preprocess/features/extract_roi_features.py \
        --config configs/extract_roi_features.yaml

    # Override via CLI:
    python datasets/preprocess/features/extract_roi_features.py \
        --config configs/extract_roi_features.yaml \
        --model v2l --ckpt checkpoint_50 --video 001YG.mp4
"""

import argparse
import gc
import os
import pickle
import sys
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_TO_DIR = {"v2": "dinov2b", "v2l": "dinov2l", "v3l": "dinov3l"}

# Label normalization (must match corrected_world_bbox_generator.py)
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
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
    return LABEL_NORMALIZE_MAP.get(label, label)


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

        # Build and load model
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
                ckpt_state = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict(ckpt_state["model_state_dict"])
                del ckpt_state
                gc.collect()
                print("  ✓ Checkpoint loaded successfully")
            else:
                print(f"  ⚠️  Checkpoint not found at {ckpt_path}, using pretrained weights only")
        else:
            print("  ℹ️  No checkpoint specified, using pretrained weights only")

        self.model.to(self.device)
        self.model.eval()

        # Extract components for manual ROI feature extraction
        self.backbone = self.model.backbone
        self.transform = self.model.transform
        self.roi_heads = self.model.roi_heads if cfg.head_3d_mode == "unified" else self.model.roi_heads

        # Get ROI pooler and box_head from the roi_heads
        if cfg.head_3d_mode == "unified":
            self.roi_pooler = self.roi_heads.base.box_roi_pool
            self.box_head = self.roi_heads.base.box_head
        else:
            self.roi_pooler = self.roi_heads.box_roi_pool
            self.box_head = self.roi_heads.box_head

        # Store dtype
        self.store_dtype = np.float16 if cfg.store_dtype == "float16" else np.float32

        # Data paths
        self.frames_path = Path(cfg.data_path) / "frames"
        self.frames_annotated_path = Path(cfg.data_path) / "frames_annotated"

        # Load AG dataset
        print("Loading Action Genome dataset...")
        from dataloader.ag_dataset import StandardAG
        from torch.utils.data import DataLoader

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

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_frame_image(self, video_id: str, frame_name: str) -> Optional[torch.Tensor]:
        """
        Load a single frame image and prepare it for the detector.

        Returns a (3, H, W) float32 tensor normalized for DINOv2.
        """
        # Try frames_annotated first (annotated subset), then frames (all)
        frame_file = frame_name.split("/")[-1] if "/" in frame_name else frame_name
        img_path = self.frames_annotated_path / video_id / frame_file
        if not img_path.exists():
            img_path = self.frames_path / video_id / frame_file
        if not img_path.exists():
            return None

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None

        # Convert BGR → RGB, normalize to [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_f32 = img_rgb.astype(np.float32) / 255.0

        # To tensor: (H, W, 3) → (3, H, W)
        img_tensor = torch.from_numpy(img_f32).permute(2, 0, 1)
        return img_tensor

    # ------------------------------------------------------------------
    # BBox collection
    # ------------------------------------------------------------------

    def _collect_bboxes_for_frame(
        self,
        gt_frame_items: List[Dict[str, Any]],
        gdino_frame: Optional[Dict[str, Any]],
        threshold: float,
    ) -> Tuple[List[List[float]], List[str], List[str], List[float]]:
        """
        Collect all bboxes for a frame: GT annotations + GDino fill for missing labels.

        Returns:
            bboxes_xyxy: list of [x1, y1, x2, y2]
            labels: list of normalized label strings
            sources: list of "gt" or "gdino"
            gdino_scores: list of float (-1.0 for GT)
        """
        bboxes_xyxy = []
        labels = []
        sources = []
        gdino_scores_out = []

        gt_labels_in_frame = set()

        for item in gt_frame_items:
            if "person_bbox" in item:
                # Person bbox
                pb = item["person_bbox"]
                if isinstance(pb, np.ndarray):
                    pb = pb.tolist()
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
                if gd_score < threshold:
                    continue

                gd_label_norm = _normalize_label(gd_label)

                # Skip labels already in GT
                if gd_label_norm in gt_labels_in_frame:
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

        return bboxes_xyxy, labels, sources, gdino_scores_out

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
        # Load GDino detections
        gdino_preds = _load_gdino_predictions(self.cfg.data_path, video_id)

        frames_data = {}

        for frame_items in gt_annotations:
            if not frame_items:
                continue

            # Get frame name from the first item
            first_item = frame_items[0]
            frame_relpath = first_item.get("frame", "")
            if "/" in frame_relpath:
                frame_file = frame_relpath.split("/")[-1]
            else:
                frame_file = frame_relpath

            if not frame_file:
                continue

            # Load image
            img_tensor = self._load_frame_image(video_id, frame_file)
            if img_tensor is None:
                continue

            # Collect bboxes
            gdino_frame = gdino_preds.get(frame_file, None) if gdino_preds else None
            bboxes_xyxy, labels, sources, gdino_scores = self._collect_bboxes_for_frame(
                frame_items, gdino_frame, self.cfg.gdino_score_threshold,
            )

            if not bboxes_xyxy:
                continue

            # Prepare image through the detector's transform (batching, no resize/norm)
            img_list = [img_tensor]
            images, _ = self.transform(img_list, None)

            # Run backbone + FPN
            features = self.backbone(images.tensors.to(self.device))
            if isinstance(features, torch.Tensor):
                features = {"0": features}

            # Prepare bbox tensor
            bboxes_tensor = torch.tensor(bboxes_xyxy, dtype=torch.float32, device=self.device)

            # Scale bboxes if image was resized by transform
            orig_h, orig_w = img_tensor.shape[1], img_tensor.shape[2]
            trans_h, trans_w = images.image_sizes[0]
            if (trans_h != orig_h) or (trans_w != orig_w):
                scale_x = trans_w / orig_w
                scale_y = trans_h / orig_h
                bboxes_tensor[:, 0] *= scale_x
                bboxes_tensor[:, 2] *= scale_x
                bboxes_tensor[:, 1] *= scale_y
                bboxes_tensor[:, 3] *= scale_y

            # Clamp bboxes to image bounds
            bboxes_tensor[:, 0].clamp_(min=0, max=trans_w)
            bboxes_tensor[:, 2].clamp_(min=0, max=trans_w)
            bboxes_tensor[:, 1].clamp_(min=0, max=trans_h)
            bboxes_tensor[:, 3].clamp_(min=0, max=trans_h)

            # Ensure valid boxes (x2 > x1, y2 > y1)
            valid_mask = (bboxes_tensor[:, 2] > bboxes_tensor[:, 0]) & \
                         (bboxes_tensor[:, 3] > bboxes_tensor[:, 1])

            if not valid_mask.any():
                continue

            bboxes_tensor = bboxes_tensor[valid_mask]
            valid_indices = valid_mask.cpu().numpy()
            labels = [l for l, v in zip(labels, valid_indices) if v]
            sources = [s for s, v in zip(sources, valid_indices) if v]
            gdino_scores = [g for g, v in zip(gdino_scores, valid_indices) if v]
            bboxes_xyxy_kept = [b for b, v in zip(bboxes_xyxy, valid_indices) if v]

            # ROI Pool + box_head → 1024-d features
            roi_features = self.roi_pooler(features, [bboxes_tensor], [images.image_sizes[0]])
            roi_features = self.box_head(roi_features)  # (N, 1024)

            # Store
            frames_data[frame_file] = {
                "roi_features": roi_features.cpu().numpy().astype(self.store_dtype),
                "bboxes_xyxy": np.array(bboxes_xyxy_kept, dtype=np.float32),
                "labels": labels,
                "sources": sources,
                "gdino_scores": gdino_scores,
            }

        if not frames_data:
            return None

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

        datasets_to_process = []
        if self.cfg.split is None or self.cfg.split == "train":
            datasets_to_process.append(("train", self.train_dataset))
        if self.cfg.split is None or self.cfg.split == "test":
            datasets_to_process.append(("test", self.test_dataset))

        for split_name, dataset in datasets_to_process:
            print(f"\n{'='*60}")
            print(f"Processing {split_name} split ({len(dataset)} videos)")
            print(f"{'='*60}")

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
                    continue

                try:
                    result = self._extract_features_for_video(
                        video_id, gt_annotations, frame_names,
                    )
                    if result is not None:
                        with open(out_path, "wb") as f:
                            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                        n_frames = len(result["frames"])
                        n_total = sum(
                            len(fd["labels"]) for fd in result["frames"].values()
                        )
                        results[video_id] = True
                    else:
                        results[video_id] = False
                except Exception as e:
                    print(f"\n  ⚠️  [{video_id}] Error: {e}")
                    results[video_id] = False

                # Periodic GPU cache cleanup
                if idx % 50 == 0:
                    torch.cuda.empty_cache()

        success = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\n{'='*60}")
        print(f"Done: {success}/{total} videos processed successfully")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")

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
    extractor = ROIFeatureExtractor(cfg)
    extractor.extract_all()


if __name__ == "__main__":
    main()
