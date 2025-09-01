import collections
import os
import random

import cv2
import numpy as np
import torch
import json

from dataloader.base_ag_dataset import BaseAG
from constants import Constants as const
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from utils import NumpyEncoder


class LabelNoiseAG(BaseAG):

    def __init__(
            self,
            phase,
            mode,
            maintain_distribution,
            datasize,
            noise_percentage=30,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False
    ):
        super().__init__(phase, mode, datasize, data_path, filter_nonperson_box_frame, filter_small_box)
        self._maintain_distribution = maintain_distribution
        self._gt_annotations = self.corrupt_gt_annotations(noise_percentage)

    @staticmethod
    def estimate_rel_distribution(data):
        rel_counts = collections.Counter()
        total_annotations = 0

        for video in data:
            for frame in video:
                for rel in frame:
                    rel_counts[rel] += 1
                    total_annotations += 1

        distribution = {obj: count / total_annotations for obj, count in rel_counts.items()}
        return distribution, total_annotations, rel_counts

    def filter_annotations_preserve_distribution(self, data, label_noise_percentage):
        distribution, total_annotations, rel_counts = self.estimate_rel_distribution(data)

        if self._maintain_distribution:
            target_counts = {obj: int(round(count * label_noise_percentage)) for obj, count in rel_counts.items()}
        else:
            target_total_annotations = int(round(total_annotations * label_noise_percentage))
            objects = list(rel_counts.keys())
            total_relations = len(objects)

            # Generate random counts such that their sum equals target_total_annotations
            counts = np.random.multinomial(target_total_annotations, np.ones(total_relations) / total_relations)

            # Assign counts to corresponding objects
            target_counts = {obj: count for obj, count in zip(objects, counts)}

        # Collect positions of each object
        rel_positions = collections.defaultdict(list)
        for v_idx, video in enumerate(data):
            for f_idx, frame in enumerate(video):
                for rel_idx, rel in enumerate(frame):
                    rel_positions[rel].append((v_idx, f_idx, rel_idx))

        # For each relation, randomly select target_counts[obj] positions to keep
        positions_to_keep = set()
        for rel, positions in rel_positions.items():
            k = target_counts[rel]
            if k > len(positions):
                k = len(positions)
            selected_positions = random.sample(positions, k)
            positions_to_keep.update(selected_positions)

        # Reconstruct the data
        filtered_data = []
        for v_idx, video in enumerate(data):
            filtered_video = []
            for f_idx, frame in enumerate(video):
                filtered_frame = []
                for rel_idx, rel in enumerate(frame):
                    if (v_idx, f_idx, rel_idx) in positions_to_keep:
                        filtered_frame.append(rel)
                filtered_video.append(filtered_frame)
            filtered_data.append(filtered_video)

        return filtered_data

    @staticmethod
    def get_rel_class_list(gt_annotations, relationship_name):
        data_rel_class_list = []
        for video_id, video_annotation_dict in enumerate(gt_annotations):
            video_rel_list = []
            for video_frame_id, video_frame_annotation_dict in enumerate(video_annotation_dict):
                video_frame_gt_rel_id_list = []
                for frame_obj_id, frame_obj_dict in enumerate(video_frame_annotation_dict):
                    if frame_obj_id == 0:
                        # video_frame_gt_obj_id_list.append(0)
                        continue

                    rel_id_list = frame_obj_dict[relationship_name]

                    if isinstance(rel_id_list, torch.Tensor):
                        rel_id_list = rel_id_list.detach().cpu().numpy().tolist()

                    assert isinstance(rel_id_list, list)
                    video_frame_gt_rel_id_list.extend(rel_id_list)
                video_rel_list.append(video_frame_gt_rel_id_list)
            data_rel_class_list.append(video_rel_list)

        return data_rel_class_list

    def corrupt_gt_annotations(self, label_noise):
        # Load from cache if the partial file exists in the cache directory.
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        cache_file = os.path.join(
            annotations_path,
            const.LABEL_NOISE,
            f'{self._mode}_label_noise_{label_noise}_mdist_{self._maintain_distribution}.json'
        )

        if os.path.exists(cache_file):
            print(f"Loading corrupted ground truth annotations from {cache_file}")
            with open(cache_file, 'r') as file:
                corrupted_gt_annotations = json.loads(file.read())
            return corrupted_gt_annotations

        #--------------------------------------------------------------------------------------------
        # Generate corrupted ground truth annotations based on the label noise percentage.
        #--------------------------------------------------------------------------------------------
        print("--------------------------------------------------------------------------------")
        print("No file found in the cache directory.")
        print(f"Corrupting ground truth annotations based on label annotation ratio: {label_noise}%")
        print("--------------------------------------------------------------------------------")

        # 1. Estimate statistics of object class occurrences in the ground truth annotations.
        attention_rel_class_list = self.get_rel_class_list(self._gt_annotations, const.ATTENTION_RELATIONSHIP)
        spatial_rel_class_list = self.get_rel_class_list(self._gt_annotations, const.SPATIAL_RELATIONSHIP)
        contacting_rel_class_list = self.get_rel_class_list(self._gt_annotations, const.CONTACTING_RELATIONSHIP)

        # 2. Construct filter based on the probability distribution of the obj class occurrences.
        filtered_attention_rel_class_list = self.filter_annotations_preserve_distribution(
            data=attention_rel_class_list,
            label_noise_percentage=label_noise*0.01
        )

        filtered_spatial_rel_class_list = self.filter_annotations_preserve_distribution(
            data=spatial_rel_class_list,
            label_noise_percentage=label_noise*0.01
        )

        filtered_contacting_rel_class_list = self.filter_annotations_preserve_distribution(
            data=contacting_rel_class_list,
            label_noise_percentage=label_noise*0.01
        )

        # 3. Construct filtered ground truth annotations based on filtered obj class occurrences.
        corrupted_gt_annotations = []
        for video_id, video_annotation_dict in enumerate(self._gt_annotations):
            corrupted_video_annotation_dict = []
            for video_frame_id, video_frame_annotation_dict in enumerate(video_annotation_dict):
                corrupted_video_frame_annotation_dict = []
                for frame_obj_id, frame_obj_dict in enumerate(video_frame_annotation_dict):
                    if frame_obj_id == 0:
                        continue
                    attention_rel = frame_obj_dict[const.ATTENTION_RELATIONSHIP]
                    spatial_rel = frame_obj_dict[const.SPATIAL_RELATIONSHIP]
                    contacting_rel = frame_obj_dict[const.CONTACTING_RELATIONSHIP]

                    if isinstance(attention_rel, torch.Tensor):
                        attention_rel = attention_rel.detach().cpu().numpy().tolist()

                    if isinstance(spatial_rel, torch.Tensor):
                        spatial_rel = spatial_rel.detach().cpu().numpy().tolist()

                    if isinstance(contacting_rel, torch.Tensor):
                        contacting_rel = contacting_rel.detach().cpu().numpy().tolist()

                    corrupted_frame_obj_dict = frame_obj_dict.copy()
                    corrupted_frame_obj_dict[const.ATTENTION_RELATIONSHIP] = []
                    corrupted_frame_obj_dict[const.SPATIAL_RELATIONSHIP] = []
                    corrupted_frame_obj_dict[const.CONTACTING_RELATIONSHIP] = []

                    filtered_attention = filtered_attention_rel_class_list[video_id][video_frame_id]
                    filtered_spatial = filtered_spatial_rel_class_list[video_id][video_frame_id]
                    filtered_contacting = filtered_contacting_rel_class_list[video_id][video_frame_id]

                    for rel in attention_rel:
                        if rel in filtered_attention:
                            # Pick a random relationship from self.attention_relationships and add it to the list.
                            c_a_relationship = random.choice(self.attention_relationships)
                            corrupted_a_rel_id = self.attention_relationships.index(c_a_relationship)
                            corrupted_frame_obj_dict[const.ATTENTION_RELATIONSHIP].append(corrupted_a_rel_id)
                        else:
                            corrupted_frame_obj_dict[const.ATTENTION_RELATIONSHIP].append(rel)

                    for rel in spatial_rel:
                        if rel in filtered_spatial:
                            # Pick a random relationship from self.spatial_relationships and add it to the list.
                            c_s_relationship = random.choice(self.spatial_relationships)
                            corrupted_s_rel_id = self.spatial_relationships.index(c_s_relationship)
                            corrupted_frame_obj_dict[const.SPATIAL_RELATIONSHIP].append(corrupted_s_rel_id)
                        else:
                            corrupted_frame_obj_dict[const.SPATIAL_RELATIONSHIP].append(rel)

                    for rel in contacting_rel:
                        if rel in filtered_contacting:
                            # Pick a random relationship from self.contacting_relationships and add it to the list.
                            c_c_relationship = random.choice(self.contacting_relationships)
                            corrupted_c_rel_id = self.contacting_relationships.index(c_c_relationship)
                            corrupted_frame_obj_dict[const.CONTACTING_RELATIONSHIP].append(corrupted_c_rel_id)
                        else:
                            corrupted_frame_obj_dict[const.CONTACTING_RELATIONSHIP].append(rel)

                    assert len(corrupted_frame_obj_dict[const.ATTENTION_RELATIONSHIP]) == len(attention_rel)
                    assert len(corrupted_frame_obj_dict[const.SPATIAL_RELATIONSHIP]) == len(spatial_rel)
                    assert len(corrupted_frame_obj_dict[const.CONTACTING_RELATIONSHIP]) == len(contacting_rel)

                    corrupted_video_frame_annotation_dict.append(corrupted_frame_obj_dict)

                corrupted_video_frame_annotation_dict.insert(0, video_frame_annotation_dict[0])
                corrupted_video_annotation_dict.append(corrupted_video_frame_annotation_dict)

            # Don't change this logic as the ground truth annotations are loaded based on the video index
            # Number of gt annotations should remain the same as the original annotations.
            corrupted_gt_annotations.append(corrupted_video_annotation_dict)

        # 4. Save the filtered ground truth annotations to the cache directory.
        os.makedirs(os.path.join(annotations_path, const.LABEL_NOISE), exist_ok=True)
        filtered_gt_annotations_json = json.dumps(corrupted_gt_annotations, cls=NumpyEncoder)
        with open(cache_file, 'w') as f:
            f.write(filtered_gt_annotations_json)

        return corrupted_gt_annotations


    def __getitem__(self, index):
        frame_names = self._video_list[index]
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


def cuda_collate_fn(batch):
    """
    don't need to zip the tensor in the batch
    """
    return batch[0]
