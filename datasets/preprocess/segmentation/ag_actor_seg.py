import os
import pickle
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_color_map
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from dataloader.coco.action_genome.ag_dataset import StandardAGCoCoDataset


# 1. Load the dataset, get the objects present in the dataset annotations.
# 2. Use gdino to extract bounding boxes.
# 3. Segmentation Route 1: Use SAM2 to get the masks for the objects in individual frames.
# 4. Segmentation Route 2:
#       (a) Identify the first frame occurrence of each object from annotations.
#       (b) Use SAM2 video mode to propagate and get the masks for each frame.
# 5. Take union of masks from both routes to get the final masks for each object in each frame.
# 6. Save masked frames and masked videos.


class AgActorSegmentation:

    def __init__(self, ag_root_directory):
        self.gdino_object_labels = None
        self.gdino_model = None
        self.gdino_processor = None
        self.gdino_device = None
        self.gdino_model_id = None
        self._dataloader_train = None
        self._dataloader_test = None
        self._test_dataset = None
        self._train_dataset = None

        self.ag_root_directory = Path(ag_root_directory)
        self.bbox_dir_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.gdino_vis_path = self.ag_root_directory / "detection" / 'gdino_vis'
        self.masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.ag_root_directory / "segmentation" / 'masked_videos'
        self.masked_frames_im_dir_path.mkdir(parents=True, exist_ok=True)
        self.masked_frames_vid_dir_path.mkdir(parents=True, exist_ok=True)
        self.masked_videos_dir_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_dataset()
        self.load_gdino_model()
        self.load_sam2_model()

        self.video_id_active_objects_map = {}
        self.process_video_id_active_objects_map()

    # -------------------------------------- LOADING INFORMATION -------------------------------------- #

    def load_dataset(self):
        self._train_dataset = StandardAGCoCoDataset(
            phase="train",
            mode="sgdet",
            datasize="large",
            data_path=self.ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=True
        )

        self._test_dataset = StandardAGCoCoDataset(
            phase="test",
            mode="sgdet",
            datasize="large",
            data_path=self.ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=True
        )

        self._dataloader_train = DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=lambda b: b[0],  # you use batch_size=1; just pass the item through,
            pin_memory=False,
            num_workers=0
        )

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=lambda b: b[0],  # you use batch_size=1; just pass the item through,
            pin_memory=False
        )

    def process_video_id_active_objects_map(self):
        for data in self._dataloader_train:
            video_id = data['video_id']
            gt_annotations = data['gt_annotations']
            active_objects = set()
            for frame_items in gt_annotations:
                for item in frame_items:
                    category_id = item['class']
                    category_name = self._train_dataset.catid_to_name_map[category_id]
                    if category_name:
                        active_objects.add(category_name)

            active_objects.add("person")  # Ensure 'person' is always included
            self.video_id_active_objects_map[video_id] = sorted(list(active_objects))

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

    # -------------------------------------- DETECTION MODULES -------------------------------------- #

    def draw_and_save_bboxes(
            self,
            image_path: str,
            boxes: torch.Tensor,
            labels: List[str],
            output_dir: str,
            frame_name: str
    ):
        if not os.path.exists(image_path): return
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        unique_labels = sorted(list(set(labels)))

        if len(unique_labels) == 0: return

        color_map = get_color_map(len(unique_labels))
        label_to_color = {label: tuple(c) for label, c in zip(unique_labels, color_map)}

        for box, label in zip(boxes.tolist(), labels):
            color = label_to_color.get(label, "red")
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], box[1] - 10), label, fill=color)

        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, frame_name))

    def extract_bounding_boxes(self, video_id, visualize=False):
        # Use GDINO to extract bounding boxes for objects in frames
        video_frames_dir_path = os.path.join(self.ag_root_directory, "sampled_frames", video_id)
        video_output_file_path = os.path.join(self.bbox_dir_path, f"{video_id}.pkl")

        # Loads object labels corresponding to active objects in the dataset
        # video_object_labels = self.gdino_object_labels
        video_object_labels = self.video_id_active_objects_map[video_id]

        if os.path.exists(video_output_file_path):
            print(f"Bounding boxes for video {video_id} already exist. Skipping detection...")
            return

        video_predictions = {}
        video_frames = sorted([f for f in os.listdir(video_frames_dir_path) if f.endswith('.png')])
        for video_frame_name in tqdm(video_frames, desc=f"Detecting objects in {video_id}"):
            frame_path = os.path.join(video_frames_dir_path, video_frame_name)
            if not os.path.exists(frame_path): continue
            image = Image.open(frame_path).convert("RGB")
            inputs = self.gdino_processor(
                images=image,
                text=". ".join(video_object_labels),
                return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.gdino_model(**inputs)

            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]])[0]

            video_predictions[video_frame_name] = {
                'boxes': results['boxes'],
                'scores': results['scores'],
                'labels': results['labels']
            }

            if visualize:
                vis_dir = os.path.join(self.gdino_vis_path, video_id)
                self.draw_and_save_bboxes(frame_path, results['boxes'], results['labels'], vis_dir, video_frame_name)

        with open(video_output_file_path, 'wb') as file:
            pickle.dump(video_predictions, file)

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

    def process(self):
        self.load_dataset()

        # video_id_list = os.listdir(self.data_dir_path / "videos")
        video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]

        for video_id in tqdm(video_id_list):
            self.extract_bounding_boxes(video_id)
            self.segment_with_sam2(video_id)
            self.segment_with_sam2_video_mode(video_id)
            self.combine_masks(video_id)
            self.save_masked_frames_and_videos(video_id)


def main():
    data_dir_path = "/data/rohith/ag/"
    ag_actor_segmentation = AgActorSegmentation(data_dir_path)
    # ag_actor_segmentation.process()


if __name__ == "__main__":
    main()
