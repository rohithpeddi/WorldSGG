import os

import torch
from PIL.Image import open as Image_open
from dataloader.base_ag_dataset import BaseAG
from dataloader.corrupted.image_based.corruptions import *
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob

"""
Dataset Corruption Mode vs Video Corruption Mode
Possible settings of image corruptions:

1. Dataset Corruption Mode: FIXED ==> Video Corruption Mode: FIXED
2. Dataset Corruption Mode: MIXED 
    a. Video Corruption Mode: FIXED
    b. Video Corruption Mode: MIXED
"""


class ImageCorruptedAG(BaseAG):

    def __init__(
            self,
            phase,
            mode,
            datasize,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
            dataset_corruption_mode=None,
            video_corruption_mode=None,  # Can be FIXED or MIXED
            dataset_corruption_type=None,  # Can be FIXED or MIXED
            corruption_severity_level=None
    ):
        super().__init__(phase, mode, datasize, data_path, filter_nonperson_box_frame, filter_small_box)
        self._dataset_corruption_mode = dataset_corruption_mode
        self._dataset_corruption_type = dataset_corruption_type
        self._video_corruption_mode = video_corruption_mode
        self._corruption_severity_level = corruption_severity_level

        self._init_corruption_settings()

    def _init_corruption_settings(self):
        print(f"Dataset Corruption Mode: {self._dataset_corruption_mode}")
        if self._dataset_corruption_mode == const.FIXED:
            # 1. If dataset corruption mode is fixed
            # a. Fix the dataset corruption type
            # b. Initialize FixedDatasetCorruptionGenerator
            self.corruption_generator = FixedDatasetCorruptionGenerator(self._dataset_corruption_type,
                                                                        self._corruption_severity_level)
        elif self._dataset_corruption_mode == const.MIXED:
            # 2. If the dataset corruption mode is mixed and video corruption mode is fixed
            # a. Initialize FixedVideoCorruptionGenerator
            # 3. If the dataset corruption mode is mixed and video corruption mode is mixed
            # a. Initialize MixedVideoCorruptionGenerator
            if self._video_corruption_mode == const.FIXED:
                self.corruption_generator = FixedVideoCorruptionGenerator(self._corruption_severity_level)
            elif self._video_corruption_mode == const.MIXED:
                self.corruption_generator = MixedVideoCorruptionGenerator(self._corruption_severity_level)

    def __getitem__(self, index):
        frame_names = self._video_list[index]
        processed_ims = []
        im_scales = []
        # 1. Takes in number of frames, types of corruptions as input and outputs a list of image transforms
        corruption_transform_list = self.corruption_generator.fetch_corruption_transforms(len(frame_names))
        # to be applied to the frames
        for idx, name in enumerate(frame_names):
            im = Image_open(os.path.join(self._frames_path, name)).convert('RGB')
            corrupted_image = corruption_transform_list[idx](im, self._corruption_severity_level)
            corrupted_image = corrupted_image[:, :, ::-1]  # rgb -> bgr
            # cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            corrupted_image, corrupted_im_scale = prep_im_for_blob(corrupted_image, [[[102.9801, 115.9465, 122.7717]]], 600, 1000)
            im_scales.append(corrupted_im_scale)
            processed_ims.append(corrupted_image)
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
