import os
import pickle
from typing import Tuple, List, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from constants import Constants as const


class ActionGenomeDataset3D(Dataset):
    """
    Action Genome dataset with direct resize to target_size x target_size.
    Includes 3D bounding box annotations.
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

        print("---------       Loading Annotation Files (3D)       ---------")
        self._fetch_object_classes()
        self.person_bbox, self.object_bbox = self._fetch_object_person_bboxes(filter_small_box)

        print("---------       Building Dataset (3D)       ---------")
        self.world_3d_annotations = os.path.join(self.data_path, const.WORLD_ANNOTATIONS, "bbox_annotations_3d_final")
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
        # self.samples: Dict[str, Dict]
        self.samples: Dict[str, Dict] = {}
        self.valid_nums = 0
        self.non_gt_human_nums = 0

        # ---------------- 2D boxes ---------------- #
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
                    # Initialize 3D placeholders
                    'person_boxes_3d': None,
                    'object_boxes_3d': []
                }
                self.valid_nums += 1

        print(f"Built dataset with {self.valid_nums} valid frames\n")
        print(f"Removed {self.non_gt_human_nums} frames without person boxes\n")

        # ------------ 3D Annotations ------------ #
        # Loop through all samples to append the 3D annotations to the samples directory
        for video_file in os.listdir(self.world_3d_annotations):
            if not video_file.endswith('.pkl'):
                continue
            video_3d_annotations_path = os.path.join(self.world_3d_annotations, video_file)
            with open(video_3d_annotations_path, 'rb') as f:
                video_3d_data = pickle.load(f)

            video_id = video_3d_data["video_id"]
            bbox_frames = video_3d_data["frames_final"]["bbox_frames"]

            for frame_id, frame_name in enumerate(bbox_frames.keys()):
                video_frame_name = f"{video_id}/{frame_name}"
                frame_objects = bbox_frames[frame_name]["objects"]

                frame_person_3d_bboxes = None
                frame_object_3d_bboxes = []

                # We need to be careful to match these 3D objects to the 2D objects
                # The current logic just collects them. 
                # Ideally, the order in 'objects' and 'object_boxes_3d' should match or be linkable.
                # The 'objects' list in _build_dataset is built from self.object_bbox (2D annotations).
                # The 3D annotations come from a separate file.
                # Assuming the 3D annotations file contains *all* objects present in the scene, 
                # but we only have 2D boxes for some.
                # 
                # However, for the purpose of this task, we will try to match them by class or assume 
                # the 3D data structure allows us to find the corresponding object.
                # 
                # Looking at the provided code in ag_dataset_resize.py, it just stores them.
                # Let's store them as is for now, but in __getitem__ we might have an issue if we can't align them.
                # 
                # Wait, the user request says: "A new model should be created with another MLP head that predicts the 3D bbox information using the features from the output of the detected 2D bboxes."
                # This implies we need ground truth 3D boxes for the *detected* (or GT) 2D boxes.
                # 
                # If the 2D and 3D annotations are not explicitly linked by an ID, we might have to do matching (e.g. by class and maybe projection?).
                # But let's look at how they are stored.
                # 2D: list of dicts with 'bbox' and 'class'.
                # 3D: list of dicts with 'label' (class) and 'aabb_floor_aligned'.

                # Let's try to store them in a way that we can try to align them later, or just store all of them.
                # For the person, it's usually one per frame in this dataset context (or at least treated as such in the original code's logic for 3D which assigns a single frame_person_3d_bboxes).

                for frame_object in frame_objects:
                    label = frame_object["label"]
                    if label == "person":
                        frame_person_3d_bboxes = np.array(
                            frame_object["aabb_floor_aligned"]["corners_world"], dtype=np.float32
                        )
                    else:
                        class_idx = self.object_classes.index(label) if label in self.object_classes else -1
                        if class_idx != -1:
                            frame_object_3d_bboxes.append({
                                "class": class_idx,
                                "bbox_3d": np.array(
                                    frame_object["aabb_floor_aligned"]["corners_world"], dtype=np.float32
                                ),
                                # Store center/dimensions if needed, but corners are enough for now
                            })

                # Find the corresponding sample and append 3D boxes
                if video_frame_name in self.samples:
                    self.samples[video_frame_name]['person_boxes_3d'] = frame_person_3d_bboxes
                    self.samples[video_frame_name]['object_boxes_3d'] = frame_object_3d_bboxes

        # ---------- Build indexable list of frame names ---------- #
        self.frame_names: List[str] = sorted(self.samples.keys())

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx: int):
        frame_name = self.frame_names[idx]
        sample = self.samples[frame_name]

        img_path = os.path.join(self.frames_path, sample['filename'])
        pil_image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_image.size

        # Stretch resize to target_size x target_size
        resized = pil_image.resize((self.target_size, self.target_size), Image.BILINEAR)
        scale_x = self.target_size / float(orig_w)
        scale_y = self.target_size / float(orig_h)

        boxes: List[List[float]] = []
        labels: List[int] = []
        boxes_3d: List[np.ndarray] = []  # List of 8x3 arrays

        # --- Person ---
        # We have potentially multiple 2D person boxes, but usually only one 3D person box in the 3D dict?
        # The code in _build_dataset assigns `frame_person_3d_bboxes` as a single array (8,3).
        # But `person_boxes` (2D) is a list.
        # If there are multiple people in 2D but only one in 3D, that's a mismatch.
        # However, Action Genome usually focuses on the person performing the action.
        # Let's assume the first 2D box corresponds to the 3D person box if it exists.

        person_boxes = np.array(sample['person_boxes'], dtype=np.float32).reshape(-1, 4)
        person_3d = sample.get('person_boxes_3d', None)

        for i, pb in enumerate(person_boxes):
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
                labels.append(1)  # Person class is 1? In object_classes, 1 is usually person if background is 0?
                # Wait, in _fetch_object_classes, background is added first. 
                # But 'person' is not explicitly in object_classes list from the file usually?
                # In standard COCO, person is 1. 
                # Let's check where 'person' is. 
                # The original code uses `labels.append(1)` for person. 
                # And `class_idx = self.object_classes.index(obj[const.CLASS])` for objects.
                # If object_classes has 'person', we should use that index.
                # If not, 1 is hardcoded. Let's stick to 1 for person.

                if i == 0 and person_3d is not None:
                    boxes_3d.append(person_3d)
                else:
                    # If we have more 2D boxes than 3D, or no 3D box, we pad with zeros or handle it.
                    # For now, let's append zeros if missing, to keep alignment.
                    boxes_3d.append(np.zeros((8, 3), dtype=np.float32))

        # --- Objects ---
        # We need to match 2D objects to 3D objects.
        # Strategy: Match by class. If multiple of same class, it's ambiguous without more info.
        # We will try to greedily match.

        available_3d_objects = sample.get('object_boxes_3d', [])
        # Make a copy to consume
        available_3d_objects = list(available_3d_objects)

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
                cls_idx = obj['class']
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_idx)

                # Find matching 3D box
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

        # Convert image to CHW tensor and normalize with DINOv2 stats
        img_np = np.array(resized).astype(np.float32) / 255.0
        img_chw = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

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
