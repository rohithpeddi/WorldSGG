import torch
import os
from PIL import Image
from typing import List
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import numpy as np
import argparse
from pathlib import Path


class MovingObjectIdentifier:
    """
    Uses a Vision Language Model to identify moving objects in a video.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load a powerful video-VLM and its processor
        self.model_id = "microsoft/Video-LLaVA-NeXT-Llama-3-8B-4-frames"

        # Load in 4-bit for efficiency
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print("MovingObjectIdentifier initialized on device:", self.device)

    def identify_moving_objects(
            self,
            video_frames_dir_path: str,
            candidate_labels: List[str]
    ) -> List[str]:
        """
        Analyzes video frames and returns a list of moving objects from the candidate list.

        Args:
            video_frames_dir_path: Path to the directory of sampled video frames.
            candidate_labels: The full list of potential object labels.

        Returns:
            A filtered list of labels that the VLM identified as moving.
        """
        if not os.path.exists(video_frames_dir_path):
            print(f"Error: Frame directory not found at {video_frames_dir_path}")
            return []

        frame_files = sorted([f for f in os.listdir(video_frames_dir_path) if f.endswith('.png')])
        if len(frame_files) < 4:
            print(
                f"Warning: Not enough frames in {video_frames_dir_path} for full analysis. Using all {len(frame_files)} available frames.")
            if not frame_files:
                print("Error: No frames found.")
                return []
            selected_frames = frame_files
        else:
            # Sample 4 frames evenly from the video
            indices = np.linspace(0, len(frame_files) - 1, 4, dtype=int)
            selected_frames = [frame_files[i] for i in indices]

        video_frames = [Image.open(os.path.join(video_frames_dir_path, f)).convert("RGB") for f in selected_frames]

        # Construct a detailed prompt for the VLM
        candidate_list_str = ", ".join(candidate_labels)
        prompt = (
            "USER: <video>\nHere are four frames from a video. Your task is to identify all objects that are moving. "
            "The movement could be natural (like a person walking) or caused by an actor (like a person picking up a cup). "
            f"From the following list, please identify ONLY the objects that are in motion in the video.\n"
            f"List of candidate objects: [{candidate_list_str}]\n\n"
            "Respond with a comma-separated list of the moving objects and nothing else.\n"
            "ASSISTANT:"
        )

        try:
            inputs = self.processor(text=prompt, images=video_frames, return_tensors="pt").to(self.device)

            # Generate the response
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=100)

            response_text = self.processor.decode(output[0], skip_special_tokens=True)
            # Extract the part of the response after "ASSISTANT":
            assistant_response = response_text.split("ASSISTANT:")[-1].strip().lower()

            # Filter the candidate list based on the VLM's response
            moving_objects = [
                label for label in candidate_labels if label.lower() in assistant_response
            ]

            return list(set(moving_objects))  # Return unique objects

        except Exception as e:
            print(f"An error occurred during VLM processing: {e}")
            return []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify moving objects in videos using a Vision Language Model."
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data/rohith/ag",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # List of video files to process
    video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]
    output_dir = data_dir / "moving_objects"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    candidate_labels = ["a person", "a bag", "a blanket", "a book", "a box", "a broom", "a chair", "a clothes",
                        "a cup", "a dish", "a food", "a laptop", "a paper", "a phone", "a picture", "a pillow",
                        "a sandwich", "a shoe", "a towel", "a vacuum", "a glass", "a bottle", "a notebook", "a camera"]

    # Initialize the model
    identifier = MovingObjectIdentifier()

    # Process each video
    for video_id in video_id_list:
        video_name = Path(video_id).stem
        frames_dir = data_dir / "frames" / video_name

        print(f"\nProcessing video: {video_name}")

        moving_objects = identifier.identify_moving_objects(str(frames_dir), candidate_labels)

        print(f"Identified moving objects: {moving_objects}")

        # Save the results to a text file
        output_file_path = output_dir / f"{video_name}.txt"
        with open(output_file_path, 'w') as f:
            if moving_objects:
                f.write("\n".join(moving_objects))
            else:
                f.write("")  # Write an empty file if no objects were found

        print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
