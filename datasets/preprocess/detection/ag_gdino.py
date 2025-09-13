import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class AgGDino:

    def __init__(
            self,
            ag_root_directory
    ):
        self.ag_root_directory = ag_root_directory

        self.model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

        self.object_labels = [
            "person", "bag", "blanket", "book", "box", "broom", "chair", "clothes", "cup", "dish", "food", "laptop",
            "paper", "phone", "picture", "pillow", "sandwich", "shoe", "towel", "vacuum", "glass", "bottle", "notebook",
            "camera"
        ]

        self.gdino_output_path = os.path.join(self.ag_root_directory, "detection", "gdino")
        if not os.path.exists(self.gdino_output_path):
            os.makedirs(self.gdino_output_path)

        self.video_id_frames_dict = None
        json_file_path = os.path.join(os.getcwd(), "..", "..", "4d_video_frame_id_list.json")
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                self.video_id_frames_dict = json.load(file)
        else:
            # If the JSON file does not exist, create it from the pickle file
            pkl_file_path = os.path.join(os.getcwd(), "..", "..", "4d_video_frame_id_list.pkl")
            with open(pkl_file_path, 'rb') as pkl_file:
                self.video_id_frames_dict = pickle.load(pkl_file)

            # Save the dictionary to a JSON file
            with open(json_file_path, 'w') as json_file:
                json.dump(self.video_id_frames_dict, json_file)

        assert self.video_id_frames_dict is not None, "video_id_frames_dict should not be None"

    @staticmethod
    def visualize_gdino_predictions(
            image: Image.Image,
            results,
            score_threshold: float = 0.3,
            frame_idx: int = 0,
            output_dir: Optional[Path] = None,
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for score, text_label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
            if score < score_threshold:
                continue
            box = [round(x, 2) for x in box.tolist()]
            # draw rectangle
            draw.rectangle(box, outline="red", width=2)
            # find a human‐readable name: for Grounding DINO, label is the index of the text prompt
            # so we just re‐use the prompt
            caption = f"{text_label}: {score:.2f}"
            text_w = font.getbbox(text_label)[2] - font.getbbox(text_label)[0]
            text_h = font.getbbox(text_label)[3] - font.getbbox(text_label)[1]
            # background for text
            draw.rectangle(
                [box[0], box[1] - text_h, box[0] + text_w, box[1]],
                fill="red"
            )
            draw.text((box[0], box[1] - text_h), caption, fill="white", font=font)

            if output_dir is not None:
                out_path = output_dir / f"frame_{frame_idx:06d}.png"
                image.save(out_path)

        return image

    def video_gdino_processor(self, video_id):
        video_frames_path = os.path.join(self.ag_root_directory, "frames", video_id)

        if video_id not in self.video_id_frames_dict:
            print(f"Video ID {video_id} not found in video_id_frames_dict. Skipping...")
            return

        video_frame_numbers = self.video_id_frames_dict[video_id]
        # vis_video_output_path = os.path.join(self.ag_root_directory, "detection", "gdino_vis", video_id)
        # if not os.path.exists(vis_video_output_path):
        # 	os.makedirs(vis_video_output_path)

        video_output_file_path = os.path.join(self.gdino_output_path, f"{video_id[:-4]}.pkl")
        if os.path.exists(video_output_file_path):
            print(f"Video {video_id} already processed. Skipping...")
            return

        video_predictions = {}
        for frame_number in tqdm(video_frame_numbers):
            video_frame_path = os.path.join(video_frames_path, f"{frame_number:06d}.png")

            if not os.path.exists(video_frame_path):
                print(f"Frame {video_frame_path} does not exist. Skipping...")
                continue

            video_frame_file = Image.open(video_frame_path).convert("RGB")
            inputs = self.processor(
                video_frame_file,
                return_tensors="pt",
                padding=True,
                text=self.object_labels
            ).to(self.device)
            with torch.no_grad():
                outputs = self.gdino_model(**inputs)
            predictions = self.processor.post_process_grounded_object_detection(
                outputs,
                target_sizes=[video_frame_file.size[::-1]],
                threshold=0.3,
            )
            video_predictions[frame_number] = predictions[0]

        # self.visualize_gdino_predictions(
        # 	image=video_frames[0],
        # 	results=predictions[0],
        # 	output_dir=Path(vis_video_output_path)
        # )

        # Save the predictions to pkl file
        with open(video_output_file_path, 'wb') as file:
            torch.save(video_predictions, file)

    def process_all_videos(self):
        video_ids = sorted(os.listdir(os.path.join(self.ag_root_directory, "frames")))
        for video_id in tqdm(video_ids):
            self.video_gdino_processor(video_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ag_root_directory",
        type=str,
        default="/data/rohith/ag/",
        help="Root dir with 'frames/' subfolder"
    )
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    ag_gdino = AgGDino(args.ag_root_directory)
    ag_gdino.process_all_videos()


if __name__ == "__main__":
    main()
