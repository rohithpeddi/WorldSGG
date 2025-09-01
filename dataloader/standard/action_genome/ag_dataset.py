import os

import cv2
import numpy as np
import torch

from dataloader.base_ag_dataset import BaseAG
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob


class StandardAG(BaseAG):

    def __init__(self, phase, mode, datasize, data_path=None, filter_nonperson_box_frame=True, filter_small_box=False):
        super().__init__(phase, mode, datasize, data_path, filter_nonperson_box_frame, filter_small_box)
        print("Total number of ground truth annotations: ", len(self._gt_annotations))

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
    don't need to zip the tensor

    """
    return batch[0]
