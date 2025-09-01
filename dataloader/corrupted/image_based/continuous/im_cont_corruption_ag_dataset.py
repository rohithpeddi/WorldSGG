import torch
from PIL.Image import open as Image_open
from PIL import Image

from dataloader.base_ag_dataset import BaseAG
from dataloader.corrupted.image_based.util.corruptions import *
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob

class ImageContinuousCorruptedAG(BaseAG):

    def __init__(
            self,
            phase,
            mode,
            datasize,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
    ):
        super().__init__(phase, mode, datasize, data_path, filter_nonperson_box_frame, filter_small_box)
        self.corruption_generator = ContinuousImageCorruptionGenerator(mode=mode)

    def __getitem__(self, index):
        frame_names = self._video_list[index]
        processed_ims = []
        im_scales = []
        # 1. Takes in number of frames, types of corruptions as input and outputs a list of image transforms
        corruption_transform_list = self.corruption_generator.fetch_corruption_transforms(len(frame_names))
        # to be applied to the frames
        for idx, name in enumerate(frame_names):
            im = Image_open(os.path.join(self._frames_path, name)).convert('RGB')

            corrupted_image = None
            # Apply a series of transformations at different severity levels to the image
            im_corruption_transform_list = corruption_transform_list[idx]
            for i, (corruption_transform, severity_level) in enumerate(im_corruption_transform_list):
                # Apply the transformation to the image
                corrupted_image = corruption_transform(im, severity_level)
                # Transform the image to PIL format
                if i < len(im_corruption_transform_list)-1:
                    corrupted_image = Image.fromarray(corrupted_image, 'RGB')
                # print(f"Applying {corruption_transform} with severity level {severity_level} to frame {idx + 1}/{len(frame_names)}")

            assert corrupted_image is not None, "Corrupted image is None"
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
