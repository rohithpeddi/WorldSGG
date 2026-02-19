import math
import os
import pickle
import zipfile
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

try:
    from torchvision.io import read_image, ImageReadMode
    _USE_TORCHVISION_IO = True
except ImportError:
    from PIL import Image
    _USE_TORCHVISION_IO = False

from ..constants import Constants as const


def _get_corners_world(frame_object: dict) -> Optional[np.ndarray]:
    """Get 8x3 corners from a frame object. Supports aabb_floor_aligned or obb_camera (top-level corners_world)."""
    # Format 1: aabb_floor_aligned["corners_world"]
    if "aabb_floor_aligned" in frame_object:
        aligned = frame_object["aabb_floor_aligned"]
        if isinstance(aligned, dict) and "corners_world" in aligned:
            arr = np.array(aligned["corners_world"], dtype=np.float32)
            if arr.shape == (8, 3):
                return arr
    # Format 2: obb_camera style — top-level "corners_world"
    if "corners_world" in frame_object:
        arr = np.array(frame_object["corners_world"], dtype=np.float32)
        if arr.shape == (8, 3):
            return arr
    return None


class ActionGenomeDataset3D(Dataset):
    """
    Action Genome dataset with direct resize to target_size x target_size.
    Frames and 2D annotations come from data_path (Action Genome). 3D GT for the loss
    comes from pkl files (folder of per-video .pkl).
    """

    def __init__(
            self,
            data_path: str,
            phase: str = 'train',
            datasize: str = 'full',
            filter_nonperson_box_frame: bool = True,
            filter_small_box: bool = False,
            pixel_limit: int = 255000,
            target_size: Optional[int] = None,
            world_3d_annotations_path: Optional[str] = "/data/rohith/ag/world_annotations/bbox_annotations_3d_obb_camera",
    ):
        self.data_path = data_path
        self.phase = phase
        self.datasize = datasize
        self.filter_nonperson_box_frame = filter_nonperson_box_frame
        self.filter_small_box = filter_small_box
        self.pixel_limit = pixel_limit
        self.target_size = target_size  # If set, forces square resize (legacy); otherwise uses pixel_limit scaling
        self.frames_path = os.path.join(self.data_path, const.FRAMES)

        # 3D GT from pkl: optional path; if None, use data_path/world_annotations/bbox_annotations_3d_final
        if world_3d_annotations_path is not None:
            self.world_3d_annotations = world_3d_annotations_path
        else:
            self.world_3d_annotations = os.path.join(self.data_path, const.WORLD_ANNOTATIONS, "bbox_annotations_3d_final")

        print("---------       Loading Annotation Files (3D)       ---------")
        self._fetch_object_classes()
        self.person_bbox, self.object_bbox = self._fetch_object_person_bboxes(filter_small_box)

        print("---------       Building Dataset (3D)       ---------")
        self._build_dataset()

        # Use only mean/std from the processor for normalization
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.image_mean: Tuple[float, float, float] = tuple(self.processor.image_mean)
        self.image_std: Tuple[float, float, float] = tuple(self.processor.image_std)

        print(f"Dataset (3D) initialized with {len(self)} frames")
        print(f"Object classes: {len(self.object_classes)}")

    def _fetch_object_classes(self):
        self.object_classes = [const.BACKGROUND]
        object_classes_path = os.path.join(self.data_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE)
        with open(object_classes_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.object_classes.append(line.strip('\n'))
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

    def _fetch_object_person_bboxes(self, filter_small_box=False):
        annotations_path = os.path.join(self.data_path, const.ANNOTATIONS)
        with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
            person_bbox = pickle.load(f)
        with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
            object_bbox = pickle.load(f)
        return person_bbox, object_bbox

    def _build_dataset(self):
        self.samples: Dict[str, Dict] = {}
        self.valid_nums = 0
        self.non_gt_human_nums = 0

        # 2D boxes
        for frame_name in self.person_bbox.keys():
            if self.object_bbox[frame_name][0][const.METADATA][const.SET] != self.phase:
                continue

            person_boxes = self.person_bbox[frame_name][const.BOUNDING_BOX]
            if self.filter_nonperson_box_frame and len(person_boxes) == 0:
                self.non_gt_human_nums += 1
                continue

            objects = []
            for obj in self.object_bbox[frame_name]:
                if obj[const.VISIBLE] and obj[const.BOUNDING_BOX] is not None:
                    class_idx = self.object_classes.index(obj[const.CLASS])
                    bbox = obj[const.BOUNDING_BOX]
                    bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    objects.append({'bbox': bbox_xyxy, 'class': class_idx})

            if len(objects) > 0 or len(person_boxes) > 0:
                self.samples[frame_name] = {
                    'filename': frame_name,
                    'person_boxes': person_boxes,
                    'objects': objects,
                    'person_boxes_3d': None,
                    'object_boxes_3d': []
                }
                self.valid_nums += 1

        print(f"Built dataset with {self.valid_nums} valid frames\n")
        print(f"Removed {self.non_gt_human_nums} frames without person boxes\n")

        # 3D Annotations (from pkl folder or .zip)
        def _iter_3d_pkls():
            if self.world_3d_annotations.lower().endswith('.zip') and os.path.isfile(self.world_3d_annotations):
                with zipfile.ZipFile(self.world_3d_annotations, 'r') as zf:
                    for name in zf.namelist():
                        if not name.endswith('.pkl'):
                            continue
                        with zf.open(name, 'r') as f:
                            yield os.path.basename(name), pickle.load(f)
            elif os.path.isdir(self.world_3d_annotations):
                for video_file in os.listdir(self.world_3d_annotations):
                    if not video_file.endswith('.pkl'):
                        continue
                    path = os.path.join(self.world_3d_annotations, video_file)
                    with open(path, 'rb') as f:
                        yield video_file, pickle.load(f)
            else:
                return

        if not (self.world_3d_annotations.lower().endswith('.zip') and os.path.isfile(self.world_3d_annotations)) and not os.path.isdir(self.world_3d_annotations):
            print(f"3D pkl folder/zip not found: {self.world_3d_annotations} — 3D loss GT will be zeros.")
        else:
            for video_file, video_3d_data in _iter_3d_pkls():
                video_id_raw = video_3d_data.get("video_id", video_file.replace(".pkl", ""))
                video_id = video_id_raw.replace(".mp4", "") if isinstance(video_id_raw, str) else str(video_id_raw)
                bbox_frames = video_3d_data.get("frames_final", {}).get("bbox_frames", {})

                for frame_name in bbox_frames.keys():
                    video_frame_name = f"{video_id}/{frame_name}"
                    frame_data = bbox_frames[frame_name]
                    frame_objects = frame_data.get("objects", [])

                    frame_person_3d_bboxes = None
                    frame_object_3d_bboxes = []

                    for frame_object in frame_objects:
                        label = frame_object.get("label")
                        if label is None:
                            continue
                        corners = _get_corners_world(frame_object)
                        if corners is None:
                            continue
                        if label == "person":
                            frame_person_3d_bboxes = corners
                        else:
                            class_idx = self.object_classes.index(label) if label in self.object_classes else -1
                            if class_idx != -1:
                                frame_object_3d_bboxes.append({"class": class_idx, "bbox_3d": corners})

                    if video_frame_name in self.samples:
                        self.samples[video_frame_name]["person_boxes_3d"] = frame_person_3d_bboxes
                        self.samples[video_frame_name]["object_boxes_3d"] = frame_object_3d_bboxes

        # Build indexable list of frame names
        self.frame_names: List[str] = sorted(self.samples.keys())

    @staticmethod
    def _compute_target_size(orig_w: int, orig_h: int, pixel_limit: int = 255000) -> Tuple[int, int]:
        """
        Compute annotation-consistent target size.
        Matches the scaling logic used during 3D annotation generation:
        aspect-ratio preserving, dimensions rounded to multiples of 14 (DINOv2 patch size),
        total pixels ≤ pixel_limit.
        """
        scale = math.sqrt(pixel_limit / (orig_w * orig_h)) if orig_w * orig_h > 0 else 1
        w_target, h_target = orig_w * scale, orig_h * scale
        k, m = round(w_target / 14), round(h_target / 14)
        while (k * 14) * (m * 14) > pixel_limit:
            if k / m > w_target / h_target:
                k -= 1
            else:
                m -= 1
        return max(1, k) * 14, max(1, m) * 14

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx: int):
        frame_name = self.frame_names[idx]
        sample = self.samples[frame_name]

        img_path = os.path.join(self.frames_path, sample['filename'])

        # Load image and get original dimensions
        if _USE_TORCHVISION_IO:
            img_tensor = read_image(img_path, mode=ImageReadMode.RGB)  # uint8 CHW
            _, orig_h, orig_w = img_tensor.shape
        else:
            pil_image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = pil_image.size

        # Compute target size using the same logic as annotation generation
        if self.target_size is not None:
            # Legacy: force square resize
            target_w, target_h = self.target_size, self.target_size
        else:
            # Annotation-consistent: aspect-ratio preserving, multiples of 14
            target_w, target_h = self._compute_target_size(orig_w, orig_h, self.pixel_limit)

        # Resize
        if _USE_TORCHVISION_IO:
            img_tensor = TF.resize(img_tensor, [target_h, target_w],
                                   interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            img_chw = img_tensor.float() / 255.0
        else:
            resized = pil_image.resize((target_w, target_h), Image.BILINEAR)
            img_np = np.array(resized).astype(np.float32) / 255.0
            img_chw = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)

        boxes: List[List[float]] = []
        labels: List[int] = []
        boxes_3d: List[np.ndarray] = []

        # Person
        person_boxes = np.array(sample['person_boxes'], dtype=np.float32).reshape(-1, 4)
        person_3d = sample.get('person_boxes_3d', None)

        for i, pb in enumerate(person_boxes):
            x1, y1, x2, y2 = pb.tolist()
            x1 = np.clip(x1 * scale_x, 0, target_w - 1)
            x2 = np.clip(x2 * scale_x, 0, target_w - 1)
            y1 = np.clip(y1 * scale_y, 0, target_h - 1)
            y2 = np.clip(y2 * scale_y, 0, target_h - 1)

            if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(1)

                if i == 0 and person_3d is not None:
                    boxes_3d.append(person_3d)
                else:
                    boxes_3d.append(np.zeros((8, 3), dtype=np.float32))

        # Objects
        available_3d_objects = list(sample.get('object_boxes_3d', []))

        for obj in sample['objects']:
            x1, y1, x2, y2 = np.array(obj['bbox'], dtype=np.float32).tolist()
            x1 = np.clip(x1 * scale_x, 0, target_w - 1)
            x2 = np.clip(x2 * scale_x, 0, target_w - 1)
            y1 = np.clip(y1 * scale_y, 0, target_h - 1)
            y2 = np.clip(y2 * scale_y, 0, target_h - 1)

            if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                cls_idx = obj['class']
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_idx)

                match_3d = None
                for i, obj3d in enumerate(available_3d_objects):
                    if obj3d['class'] == cls_idx:
                        match_3d = obj3d['bbox_3d']
                        available_3d_objects.pop(i)
                        break

                if match_3d is not None:
                    boxes_3d.append(match_3d)
                else:
                    boxes_3d.append(np.zeros((8, 3), dtype=np.float32))

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
            'boxes_3d': torch.tensor(np.array(boxes_3d), dtype=torch.float32) if boxes_3d else torch.empty((0, 8, 3),
                                                                                                           dtype=torch.float32),
        }

        return img_chw, target


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
