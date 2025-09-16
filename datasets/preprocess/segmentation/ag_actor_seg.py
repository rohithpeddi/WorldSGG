from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from dataloader.standard.action_genome.ag_dataset import StandardAG, cuda_collate_fn


# 1. Load the dataset, get the objects present in the dataset annotations.
# 2. Use gdino to extract bounding boxes.
# 3. Segmentation Route 1: Use SAM2 to get the masks for the objects in individual frames.
# 4. Segmentation Route 2:
#       (a) Identify the first frame occurrence of each object from annotations.
#       (b) Use SAM2 video mode to propagate and get the masks for each frame.
# 5. Take union of masks from both routes to get the final masks for each object in each frame.
# 6. Save masked frames and masked videos.


class AgActorSegmentation:

    def __init__(self, data_dir_path):
        self.gdino_object_labels = None
        self.gdino_model = None
        self.gdino_processor = None
        self.gdino_device = None
        self.gdino_model_id = None
        self._dataloader_train = None
        self._dataloader_test = None
        self._test_dataset = None
        self._train_dataset = None

        self.data_dir_path = Path(data_dir_path)
        self.bbox_dir_path = self.data_dir_path / 'bboxes'
        self.masked_frames_im_dir_path = self.data_dir_path / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.data_dir_path / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.data_dir_path / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.data_dir_path / 'masked_videos'
        self.masked_frames_im_dir_path.mkdir(parents=True, exist_ok=True)
        self.masked_frames_vid_dir_path.mkdir(parents=True, exist_ok=True)
        self.masked_videos_dir_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_dataset()
        self.load_gdino_model()
        self.load_sam2_model()

    def load_dataset(self):
        self._train_dataset = StandardAG(
            phase="train",
            mode="sgdet",
            datasize="large",
            data_path=self.data_dir_path,
            filter_nonperson_box_frame=True,
            filter_small_box=True
        )

        self._test_dataset = StandardAG(
            phase="test",
            mode="sgdet",
            datasize="large",
            data_path=self.data_dir_path,
            filter_nonperson_box_frame=True,
            filter_small_box=True
        )

        self._dataloader_train = DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=cuda_collate_fn,
            pin_memory=True,
            num_workers=0
        )

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=cuda_collate_fn,
            pin_memory=False
        )

    def load_gdino_model(self):
        # Load GDINO model for bounding box extraction
        self.gdino_model_id = "IDEA-Research/grounding-dino-base"
        self.gdino_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_model_id).to(self.device)
        self.gdino_object_labels = [
                "a person", "a bag", "a blanket", "a book", "a box", "a broom", "a chair", "a clothes",
                "a cup", "a dish", "a food", "a laptop", "a paper", "a phone", "a picture", "a pillow",
                "a sandwich", "a shoe", "a towel", "a vacuum", "a glass", "a bottle", "a notebook", "a camera",
                "a bed", "a closet", "a cabinet", "a door", "a doorknob", "a groceries", "a mirror", "a refrigerator",
                "a sofa", "a couch", "a table", "a television", "a window"
            ]

    def load_sam2_model(self):
        # Load SAM2 model for segmentation
        pass

    def extract_bounding_boxes(self, video_id):
        # Use GDINO to extract bounding boxes for objects in frames
        pass

    def segment_with_sam2(self, video_id):
        # Use SAM2 to get masks for objects in individual frames
        pass

    def segment_with_sam2_video_mode(self, video_id):
        # Identify first frame occurrence of each object and use SAM2 video mode to propagate masks
        pass

    def combine_masks(self, video_id):
        # Take union of masks from both segmentation routes
        pass

    def save_masked_frames_and_videos(self, video_id):
        # Save the final masked frames and videos to the output path
        pass

    def process(self, video_id):
        self.load_dataset()

        video_id_list = [""]

        self.extract_bounding_boxes(video_id)
        self.segment_with_sam2(video_id)
        self.segment_with_sam2_video_mode(video_id)
        self.combine_masks(video_id)
        self.save_masked_frames_and_videos(video_id)
