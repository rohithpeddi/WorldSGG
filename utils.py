#!/usr/bin/env python3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from typing import Optional

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


# ---------------------------
# Split logic (yours)
# ---------------------------

def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    """
    Get the split that the video belongs to based on its ID.
    Accepts either a bare ID (e.g., '0DJ6R') or a filename (e.g., '0DJ6R.mp4').
    """
    stem = Path(video_id).stem
    if not stem:
        return None
    first_letter = stem[0]
    if first_letter.isdigit() and int(first_letter) < 5:
        return "04"
    elif first_letter.isdigit() and int(first_letter) >= 5:
        return "59"
    elif first_letter in "ABCD":
        return "AD"
    elif first_letter in "EFGH":
        return "EH"
    elif first_letter in "IJKL":
        return "IL"
    elif first_letter in "MNOP":
        return "MP"
    elif first_letter in "QRST":
        return "QT"
    elif first_letter in "UVWXYZ":
        return "UZ"
    return None

# ------------------------------ Utilities (OLD) ------------------------------


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().tolist()
    return tensor


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    # im = im[:, :, ::-1]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    return im, im_scale


# ------------------------------ Utilities (NEW) ------------------------------
def _cxcywh_to_xyxy(norm_boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    Convert normalized [cx,cy,w,h] in [0,1] to absolute [x1,y1,x2,y2] in pixels.
    shape: [N,4]
    """
    cx, cy, w, h = norm_boxes.unbind(-1)
    x1 = (cx - 0.5 * w) * img_w
    y1 = (cy - 0.5 * h) * img_h
    x2 = (cx + 0.5 * w) * img_w
    y2 = (cy + 0.5 * h) * img_h
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    # clip to image
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h - 1)
    return boxes


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU between two sets of [N,4] and [M,4] boxes (xyxy).
    Returns [N,M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def get_color_map(num_colors):
    if num_colors <= 0: return []
    """Generates a list of distinct colors for visualization."""
    colors = plt.cm.get_cmap('hsv', num_colors)
    return [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors(range(num_colors))]


# ------------------------------ Logging ------------------------------

def _to_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.
    Handles numpy, torch tensors, and nested containers.
    """
    # Lazy imports to avoid hard deps if not needed
    import numpy as np
    try:
        import torch
    except ImportError:  # torch may not be available in all contexts
        torch = None

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    if torch is not None and isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, (np.generic,)):  # np.float32, np.int64, etc.
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Fall back to default
    return obj


class LocalLogger:
    """
    Simple JSON-lines logger for local experiments.

    Usage:
        logger = LocalLogger("/path/to/experiment")
        logger.log(log_type="epoch", epoch=0, mAP=0.42, train_total_loss=1.23)

    This will create:
        /path/to/experiment/json_logs/epoch.jsonl  (one JSON object per line)
    """

    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.log_dir = os.path.join(self.experiment_dir, "json_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Only rank 0 logs in distributed setups
        rank = int(os.environ.get("RANK", "0"))
        self.is_main_process = (rank == 0)

    def _get_log_path(self, log_type: str) -> str:
        """
        Return the path for a given log_type, e.g. 'epoch' -> epoch.jsonl
        """
        safe_type = log_type.replace("/", "_")
        return os.path.join(self.log_dir, f"{safe_type}.jsonl")

    def log(self, log_type: str, **kwargs: Any) -> None:
        """
        Append a JSON record for the given log_type.

        Example:
            log("epoch", epoch=0, mAP=0.42, train_total_loss=1.23)
        """
        if not self.is_main_process:
            # Avoid multiple processes writing to the same file
            return

        record: Dict[str, Any] = {
            "log_type": log_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        record.update(kwargs)

        record = _to_serializable(record)
        log_path = self._get_log_path(log_type)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
