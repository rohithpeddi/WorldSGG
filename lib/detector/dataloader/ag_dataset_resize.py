import os
import pickle
from typing import Tuple, List, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from constant import const


class ActionGenomeDatasetResize(Dataset):
    """
    Action Genome dataset with direct resize to target_size x target_size (no padding, no crop).
    - Image is stretched to (target_size, target_size).
    - GT boxes are scaled with separate x and y factors.
    - Normalized using DINOv2 mean/std.
    """

    def __init__(
        self,
        data_path: str,
        phase: str = 'train',
        datasize: str = 'full',
        filter_nonperson_box_frame: bool = True,
        filter_small_box: bool = False,
        target_size: int = 224,
    ):
        self.data_path = data_path
        self.phase = phase
        self.datasize = datasize
        self.filter_nonperson_box_frame = filter_nonperson_box_frame
        self.filter_small_box = filter_small_box
        self.target_size = target_size
        self.frames_path = os.path.join(self.data_path, const.FRAMES)

        print("---------       Loading Annotation Files (Resize)       ---------")
        self._fetch_object_classes()
        self.person_bbox, self.object_bbox = self._fetch_object_person_bboxes(filter_small_box)

        print("---------       Building Dataset (Resize)       ---------")
        self._build_dataset()

        # Use only mean/std from the processor for normalization
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.image_mean: Tuple[float, float, float] = tuple(self.processor.image_mean)
        self.image_std: Tuple[float, float, float] = tuple(self.processor.image_std)

        print(f"Dataset (Resize) initialized with {len(self.samples)} frames")
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

    def _fetch_object_person_bboxes(self, filter_small_box: bool = False):
        annotations_path = os.path.join(self.data_path, const.ANNOTATIONS)
        with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
            person_bbox = pickle.load(f)
        with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
            object_bbox = pickle.load(f)
        return person_bbox, object_bbox

    def _build_dataset(self):
        self.samples: List[Dict] = []
        self.valid_nums = 0
        self.non_gt_human_nums = 0
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
                self.samples.append({'filename': frame_name, 'person_boxes': person_boxes, 'objects': objects})
                self.valid_nums += 1
        print(f"Built dataset with {self.valid_nums} valid frames\n")
        print(f"Removed {self.non_gt_human_nums} frames without person boxes\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = os.path.join(self.frames_path, sample['filename'])
        pil_image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_image.size

        # Stretch resize to target_size x target_size
        resized = pil_image.resize((self.target_size, self.target_size), Image.BILINEAR)
        scale_x = self.target_size / float(orig_w)
        scale_y = self.target_size / float(orig_h)

        boxes: List[List[float]] = []
        labels: List[int] = []

        person_boxes = np.array(sample['person_boxes'], dtype=np.float32).reshape(-1, 4)
        for pb in person_boxes:
            x1, y1, x2, y2 = pb.tolist()
            x1 = x1 * scale_x
            x2 = x2 * scale_x
            y1 = y1 * scale_y
            y2 = y2 * scale_y
            x1 = np.clip(x1, 0, self.target_size - 1)
            x2 = np.clip(x2, 0, self.target_size - 1)
            y1 = np.clip(y1, 0, self.target_size - 1)
            y2 = np.clip(y2, 0, self.target_size - 1)
            if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(1)

        for obj in sample['objects']:
            x1, y1, x2, y2 = np.array(obj['bbox'], dtype=np.float32).tolist()
            x1 = x1 * scale_x
            x2 = x2 * scale_x
            y1 = y1 * scale_y
            y2 = y2 * scale_y
            x1 = np.clip(x1, 0, self.target_size - 1)
            x2 = np.clip(x2, 0, self.target_size - 1)
            y1 = np.clip(y1, 0, self.target_size - 1)
            y2 = np.clip(y2, 0, self.target_size - 1)
            if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(obj['class'])

        # Convert image to CHW tensor and normalize with DINOv2 stats
        img_np = np.array(resized).astype(np.float32) / 255.0
        img_chw = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        # mean = torch.tensor(self.image_mean).view(3, 1, 1)
        # std = torch.tensor(self.image_std).view(3, 1, 1)
        # img_chw = (img_chw - mean) / std

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
        }

        return img_chw, target


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


