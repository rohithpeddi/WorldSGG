import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataloader.coco.action_genome.ag_dataset import StandardAGCoCoDataset
from utils import get_color_map


# 1. Load the dataset, get the objects present in the dataset annotations.
# 2. Use gdino to extract bounding boxes.
# 3. Segmentation Route 1: Use SAM2 to get the masks for the objects in individual frames.
# 4. Segmentation Route 2:
#       (a) Identify the first frame occurrence of each object from annotations.
#       (b) Use SAM2 video mode to propagate and get the masks for each frame.
# 5. Take union of masks from both routes to get the final masks for each object in each frame.
# 6. Save masked frames and masked videos.

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


class BaseAgActor:

    def __init__(self, ag_root_directory):
        self.ag_root_directory = Path(ag_root_directory)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # temp JPG mirror for SAM2 video predictor (expects JPEG frames)
        self.sampled_frames_jpg = self.ag_root_directory / "sampled_frames_jpg"
        self.bbox_dir_path = self.ag_root_directory / "detection" / 'gdino_bboxes'

        self._ensure_dir(self.bbox_dir_path)
        self._ensure_dir(self.sampled_frames_jpg)

        self._dataloader_train = None
        self._dataloader_test = None
        self._test_dataset = None
        self._train_dataset = None

        self.load_dataset()

    # -------------------------------------- LOADING INFORMATION -------------------------------------- #
    def load_dataset(self):
        self._train_dataset = StandardAGCoCoDataset(
            phase="train",
            mode="sgdet",
            datasize="large",
            data_path=self.ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=False
        )

        self._test_dataset = StandardAGCoCoDataset(
            phase="test",
            mode="sgdet",
            datasize="large",
            data_path=self.ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=False
        )

        self._dataloader_train = DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=lambda b: b[0],  # you use batch_size=1; just pass the item through,
            pin_memory=False,
            num_workers=0
        )

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=lambda b: b[0],  # you use batch_size=1; just pass the item through,
            pin_memory=False
        )

    def _normalize_label(self, s: str) -> str:
        # keep your existing behavior if already defined elsewhere
        s = s.lower().strip()
        s = re.sub(r"^(a|an|the)\s+", "", s)
        alias = {
            "closet/cabinet": "closet",
            "cup/glass/bottle": "cup",
            "paper/notebook": "paper",
            "sofa/couch": "sofa",
            "phone/camera": "phone",
        }
        return alias.get(s, s)

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        # boxes: (..., 4) as [x, y, w, h] -> [x1, y1, x2, y2]
        out = boxes.clone()
        out[..., 2] = boxes[..., 0] + boxes[..., 2]
        out[..., 3] = boxes[..., 1] + boxes[..., 3]
        return out

    @staticmethod
    def _ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _binary_to_png(mask_bool: np.ndarray) -> np.ndarray:
        return mask_bool.astype(np.uint8) * 255

    @staticmethod
    def _apply_mask(img_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        # keep RGB where mask==1, black elsewhere
        out = img_np.copy()
        if out.ndim == 2:
            out = np.stack([out, out, out], axis=-1)
        out[~mask_bool] = 0
        return out

    def draw_and_save_bboxes(
            self,
            image_path: str,
            boxes: torch.Tensor,
            labels: List[str],
            output_dir: str,
            frame_name: str
    ):
        if not os.path.exists(image_path): return
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        unique_labels = sorted(list(set(labels)))

        if len(unique_labels) == 0: return

        color_map = get_color_map(len(unique_labels))
        label_to_color = {label: tuple(c) for label, c in zip(unique_labels, color_map)}

        for box, label in zip(boxes.tolist(), labels):
            color = label_to_color.get(label, "red")
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], box[1] - 10), label, fill=color)

        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, frame_name))

    def _mirror_pngs_to_jpg(self, frames_dir: Path, video_id: str) -> Tuple[Path, List[str]]:
        jpg_dir = self.sampled_frames_jpg / video_id
        self._ensure_dir(jpg_dir)

        fn_png = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        jpg_names = []
        for fn in fn_png:
            src = frames_dir / fn
            stem = Path(fn).stem
            dst = jpg_dir / f"{stem}.jpg"
            if not dst.exists():
                # convert (and also unify color)
                img = Image.open(src).convert("RGB")
                img.save(dst, format="JPEG", quality=95)
            jpg_names.append(dst.name)
        return jpg_dir, sorted(jpg_names)

    def _write_video_from_frames(self, frames_dir: Path, out_path: Path, fps: int = 15):
        img_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not img_files:
            return
        first = cv2.imread(str(frames_dir / img_files[0]), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for fn in img_files:
            frame = cv2.imread(str(frames_dir / fn), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
        writer.release()

