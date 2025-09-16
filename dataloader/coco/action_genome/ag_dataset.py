import os
from typing import Dict, List, Any, Tuple

import numpy as np

from dataloader.base_ag_dataset import BaseAG


class StandardAGCoCoDataset(BaseAG):

    def __init__(
            self,
            phase="test",
            mode="sgdet",
            datasize="full",
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False
    ):
        super().__init__(phase, mode, datasize, data_path, filter_nonperson_box_frame, filter_small_box)
        self.gt_coco_dict = None
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

        self._ann_id_counter = 1
        self._min_box_area = 0.25 if filter_small_box else 0.0
        self._image_id_lookup: Dict[str, int] = {}

        self._images_json: List[Dict[str, Any]] = []
        self._annotations_json: List[Dict[str, Any]] = []

        self.build_gt_coco_annotations()

    # ------------------------------ GT parsing ------------------------------
    def parse_gt_for_frame(
            self,
            gt_video_annotations: List[List[Dict[str, Any]]],
            frame_relpath: str
    ) -> Tuple[List[List[float]], List[int]]:
        boxes_xyxy: List[List[float]] = []
        cat_ids: List[int] = []

        gt_frame_items = None
        for frame_items in gt_video_annotations:
            item = frame_items[0]
            if 'frame' in item and item['frame'] == frame_relpath:
                gt_frame_items = frame_items
                break

        if gt_frame_items is None:
            raise ValueError(f"No GT items found for frame {frame_relpath}")

        for item in gt_frame_items:
            if 'person_bbox' in item and item['person_bbox'] is not None:
                pb = item['person_bbox']
                if isinstance(pb, np.ndarray):
                    pb = pb.tolist()
                if isinstance(pb, list) and len(pb) > 0:
                    for b in pb:
                        b_list = b if isinstance(b, list) else list(b)
                        boxes_xyxy.append([float(b_list[0]), float(b_list[1]), float(b_list[2]), float(b_list[3])])
                        cat_ids.append(self.name_to_catid['person'])
            else:
                has_bbox = ('bbox' in item) and (item['bbox'] is not None)
                has_cls = ('class' in item) and (item['class'] is not None)
                matches_frame = ('frame' in item and item['frame'] == frame_relpath)

                if has_bbox and has_cls and (matches_frame or ('frame' not in item)):
                    b = item['bbox']
                    if isinstance(b, np.ndarray):
                        b = b.tolist()
                    b = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                    cls_idx = int(item['class'])
                    if cls_idx <= 0:
                        continue
                    class_name = self.dataset_classnames[cls_idx]
                    if class_name not in self.name_to_catid:
                        continue
                    boxes_xyxy.append(b)
                    cat_ids.append(self.name_to_catid[class_name])

        return boxes_xyxy, cat_ids

    def build_coco_gt_for_video(
            self,
            gt_video_annotations: List[List[Dict[str, Any]]],
            frame_names: List[str],
            video_id: str
    ):
        for frame_rel in frame_names:
            video_id2, frame_file = frame_rel.split('/')
            assert video_id2 == video_id
            frame_abs = os.path.join(self._data_path, "frames_annotated", video_id, frame_file)
            if not os.path.exists(frame_abs):
                continue

            if frame_rel not in self._image_id_lookup:
                self._image_id_lookup[frame_rel] = len(self._image_id_lookup) + 1
                self._images_json.append({"id": self._image_id_lookup[frame_rel], "file_name": frame_rel})
            image_id = self._image_id_lookup[frame_rel]

            gt_boxes_xyxy, gt_cat_ids = self.parse_gt_for_frame(gt_video_annotations, frame_rel)
            for b, cid in zip(gt_boxes_xyxy, gt_cat_ids):
                area = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
                if area < self._min_box_area:
                    continue
                self._annotations_json.append({
                    "id": self._ann_id_counter,
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                    "area": area,
                    "iscrowd": 0,
                })
                self._ann_id_counter += 1

    def build_gt_coco_annotations(self):
        for idx in range(len(self._video_list)):
            frame_names = self._video_list[idx]
            gt_video_annotations = self._gt_annotations[idx]
            video_id = frame_names[0].split('/')[0]
            self.build_coco_gt_for_video(gt_video_annotations, frame_names, video_id)

        self.gt_coco_dict = {
            "images": self._images_json,
            "annotations": self._annotations_json,
            "categories": self.categories_json,
            "info": {"description": "Action Genome detection eval", "version": "1.0"},
            "licenses": []
        }

    def __getitem__(self, index):
        frame_names = self._video_list[index]  # list of "video_id/frame.png" for one video
        gt_annotations = self._gt_annotations[index]  # dataset-provided annotations for that video
        video_id = frame_names[0].split('/')[0]

        return {
            'frame_names': frame_names,
            'gt_annotations': gt_annotations,
            'index': index,
            'video_id': video_id
        }
