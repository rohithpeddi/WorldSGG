import json
import os

import cv2
import numpy as np
import torch

from dataloader.base_ag_dataset import BaseAG
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob


class AgVideoRuleDataset(BaseAG):

    def __init__(
            self,
            phase,
            mode,
            datasize,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False
    ):
        super(AgVideoRuleDataset, self).__init__(
            phase,
            mode,
            datasize,
            data_path,
            filter_nonperson_box_frame,
            filter_small_box
        )
        print("Total number of ground truth annotations: ", len(self._gt_annotations))
        self._rule_embeddings_root_dir = os.path.join(os.path.dirname(__file__), '../../lattedata/embeddings')

    def __getitem__(self, index):
        frame_names = self._video_list[index]
        processed_ims = []
        im_scales = []
        for idx, name in enumerate(frame_names):
            im = cv2.imread(os.path.join(self._frames_path, name))  # channel h,w,3
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

        video_id = frame_names[0].split('/')[0]
        rule_embeddings_json_path = os.path.join(self._rule_embeddings_root_dir, f'{video_id[:-4]}_embeddings.json')
        if not os.path.exists(rule_embeddings_json_path):
            raise FileNotFoundError(f"Rule embeddings file not found: {rule_embeddings_json_path}")

        with open(rule_embeddings_json_path, 'r') as f:
            rule_embeddings = json.load(f)

        return img_tensor, im_info, gt_boxes, num_boxes, index, rule_embeddings


def cuda_collate_fn(batch):
    return batch[0]
