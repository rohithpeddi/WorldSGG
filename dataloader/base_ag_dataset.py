import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Tuple

from constants import Constants as const
from utils import prep_im_for_blob, im_list_to_blob


class BaseAG(Dataset):

    def __init__(
            self,
            phase,
            mode,
            datasize,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
            enable_coco_gt=False
    ):

        self.invalid_video_names = None
        self.object_classes = None
        self.valid_video_names = None
        self.video_list = None
        self.video_size = None  # (w,h)
        self.gt_annotations = None
        self.non_gt_human_nums = None
        self.non_heatmap_nums = None
        self.non_person_video = None
        self.one_frame_video = None
        self.valid_nums = None
        self.invalid_videos = None
        self.relationship_classes = None
        root_path = data_path
        self._phase = phase
        self._mode = mode
        self._datasize = datasize
        self._data_path = data_path
        self._frames_path = os.path.join(root_path, const.FRAMES)

        # collect the object classes
        self.fetch_object_classes()

        # collect relationship classes
        self.fetch_relationship_classes()

        # Fetch object and person bounding boxes
        person_bbox, object_bbox = self._fetch_object_person_bboxes(self._datasize, filter_small_box)

        # collect valid frames
        video_dict, q = self._fetch_valid_frames(person_bbox, object_bbox)
        all_video_names = np.unique(q)

        # Build dataset
        self.build_dataset(video_dict, person_bbox, object_bbox, all_video_names, filter_nonperson_box_frame)

        self.gt_coco_dict = None
        self.dataset_classnames = [
            '__background__', 'person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair',
            'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
            'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 'paper/notebook',
            'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe',
            'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window'
        ]
        self.name_to_catid = {name: idx for idx, name in enumerate(self.dataset_classnames) if idx > 0}

        self.categories_json: List[Dict[str, Any]] = [
            {"id": cid, "name": name} for name, cid in self.name_to_catid.items()
        ]

        self._ann_id_counter = 1
        self._min_box_area = 0.25 if filter_small_box else 0.0
        self._image_id_lookup: Dict[str, int] = {}

        self._images_json: List[Dict[str, Any]] = []
        self._gt_coco_annotations_json: List[Dict[str, Any]] = []

        if enable_coco_gt:
            self.build_gt_coco_annotations()

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

    def extract_coco_gt_for_video(
            self,
            gt_video_annotations: List[List[Dict[str, Any]]],
            frame_names: List[str],
            video_id: str
    ):
        # Extract COCO-format GT annotations for a video
        images_json = []
        annotations_json = []
        for frame_rel in frame_names:
            video_id2, frame_file = frame_rel.split('/')
            assert video_id2 == video_id
            frame_abs = os.path.join(self._data_path, "frames_annotated", video_id, frame_file)
            if not os.path.exists(frame_abs):
                continue

            if frame_rel not in self._image_id_lookup:
                self._image_id_lookup[frame_rel] = len(self._image_id_lookup) + 1
                # self._images_json.append({"id": self._image_id_lookup[frame_rel], "file_name": frame_rel})
                images_json.append({"id": self._image_id_lookup[frame_rel], "file_name": frame_rel})
            image_id = self._image_id_lookup[frame_rel]

            gt_boxes_xyxy, gt_cat_ids = self.parse_gt_for_frame(gt_video_annotations, frame_rel)
            for b, cid in zip(gt_boxes_xyxy, gt_cat_ids):
                area = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
                if area < self._min_box_area:
                    continue
                annotations_json.append({
                    "id": self._ann_id_counter,
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                    "area": area,
                    "iscrowd": 0,
                })
                self._ann_id_counter += 1

        return images_json, annotations_json

    # def _build_coco_gt_for_video(
    #         self,
    #         gt_video_annotations: List[List[Dict[str, Any]]],
    #         frame_names: List[str],
    #         video_id: str
    # ):
    #     for frame_rel in frame_names:
    #         video_id2, frame_file = frame_rel.split('/')
    #         assert video_id2 == video_id
    #         frame_abs = os.path.join(self._data_path, "frames_annotated", video_id, frame_file)
    #         if not os.path.exists(frame_abs):
    #             continue
    #
    #         if frame_rel not in self._image_id_lookup:
    #             self._image_id_lookup[frame_rel] = len(self._image_id_lookup) + 1
    #             self._images_json.append({"id": self._image_id_lookup[frame_rel], "file_name": frame_rel})
    #         image_id = self._image_id_lookup[frame_rel]
    #
    #         gt_boxes_xyxy, gt_cat_ids = self._parse_gt_for_frame(gt_video_annotations, frame_rel)
    #         for b, cid in zip(gt_boxes_xyxy, gt_cat_ids):
    #             area = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    #             if area < self._min_box_area:
    #                 continue
    #             self._gt_coco_annotations_json.append({
    #                 "id": self._ann_id_counter,
    #                 "image_id": image_id,
    #                 "category_id": cid,
    #                 "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
    #                 "area": area,
    #                 "iscrowd": 0,
    #             })
    #             self._ann_id_counter += 1

    def aggregate_gt_coco_annotations(self, images_json, annotations_json):
        self._images_json.extend(images_json)
        self._gt_coco_annotations_json.extend(annotations_json)

    def build_gt_coco_annotations(self):
        for idx in range(len(self.video_list)):
            frame_names = self.video_list[idx]
            gt_video_annotations = self.gt_annotations[idx]
            video_id = frame_names[0].split('/')[0]
            images_json, annotations_json = self.extract_coco_gt_for_video(gt_video_annotations, frame_names, video_id)
            self.aggregate_gt_coco_annotations(images_json, annotations_json)

        self.gt_coco_dict = {
            "images": self._images_json,
            "annotations": self._gt_coco_annotations_json,
            "categories": self.categories_json,
            "info": {"description": "Action Genome detection eval", "version": "1.0"},
            "licenses": []
        }

    def fetch_object_classes(self):
        self.object_classes = [const.BACKGROUND]
        with open(os.path.join(self._data_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE), 'r',
                  encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

    def fetch_relationship_classes(self):
        self.relationship_classes = []
        with open(os.path.join(self._data_path, const.ANNOTATIONS, const.RELATIONSHIP_CLASSES_FILE), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'
        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]
        print('-------loading annotations---------slowly-----------')

    def _fetch_object_person_bboxes(self, datasize, filter_small_box=False):
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        if filter_small_box:
            with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()

        if datasize == const.MINI:
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:80000]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object

        return person_bbox, object_bbox

    def _fetch_valid_frames(self, person_bbox, object_bbox):
        video_dict = {}
        q = []
        for i in person_bbox.keys():
            if object_bbox[i][0][const.METADATA][const.SET] == self._phase:  # train or testing?
                video_name, frame_num = i.split('/')
                q.append(video_name)
                frame_valid = False
                for j in object_bbox[i]:  # the frame is valid if there is visible bbox
                    if j[const.VISIBLE]:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]
        return video_dict, q

    def fetch_video_data(self, index):
        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []
        for idx, name in enumerate(frame_names):
            im = cv2.imread(os.path.join(self._frames_path, name))  # channel h,w,3
            # im = im[:, :, ::-1]  # rgb -> bgr
            # cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000)
            im_scales.append(im_scale)
            processed_ims.append(im)
        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]], dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index

    def build_dataset(self, video_dict, person_bbox, object_bbox, all_video_names, filter_nonperson_box_frame=True):
        self.valid_video_names = []
        self.video_list = []
        self.video_size = []  # (w,h)
        self.gt_annotations = []
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0
        self.invalid_videos = []

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if person_bbox[j][const.BOUNDING_BOX].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1

                gt_annotation_frame = [
                    {
                        const.PERSON_BOUNDING_BOX: person_bbox[j][const.BOUNDING_BOX],
                        const.FRAME: j
                    }
                ]

                # each frame's objects and human
                for k in object_bbox[j]:
                    if k[const.VISIBLE]:
                        assert k[const.BOUNDING_BOX] is not None, 'warning! The object is visible without bbox'
                        k[const.CLASS] = self.object_classes.index(k[const.CLASS])
                        # from xywh to xyxy
                        k[const.BOUNDING_BOX] = np.array([
                            k[const.BOUNDING_BOX][0], k[const.BOUNDING_BOX][1],
                            k[const.BOUNDING_BOX][0] + k[const.BOUNDING_BOX][2],
                            k[const.BOUNDING_BOX][1] + k[const.BOUNDING_BOX][3]
                        ])

                        k[const.ATTENTION_RELATIONSHIP] = torch.tensor(
                            [self.attention_relationships.index(r) for r in k[const.ATTENTION_RELATIONSHIP]],
                            dtype=torch.long)
                        k[const.SPATIAL_RELATIONSHIP] = torch.tensor(
                            [self.spatial_relationships.index(r) for r in k[const.SPATIAL_RELATIONSHIP]],
                            dtype=torch.long)
                        k[const.CONTACTING_RELATIONSHIP] = torch.tensor(
                            [self.contacting_relationships.index(r) for r in k[const.CONTACTING_RELATIONSHIP]],
                            dtype=torch.long)
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                self.video_list.append(video)
                self.video_size.append(person_bbox[j][const.BOUNDING_BOX_SIZE])
                self.gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x' * 60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(
                self.non_heatmap_nums))
        print('x' * 60)

        self.invalid_video_names = np.setdiff1d(all_video_names, self.valid_video_names, assume_unique=False)

    def __len__(self):
        return len(self.video_list)