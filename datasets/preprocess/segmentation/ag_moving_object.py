import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor


class MovingObjectIdentifier:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load a powerful video-VLM and its processor
        self.model_id = "LanguageBind/Video-LLaVA-7B-hf"

        # Load in 4-bit for efficiency
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto")

        self.processor = VideoLlavaProcessor.from_pretrained(self.model_id)
        print("MovingObjectIdentifier initialized on device:", self.device)

    def identify_moving_objects(
            self,
            video_frames_dir_path: str,
            candidate_labels: List[str]
    ) -> List[str]:
        """
        Analyzes video frames in chunks of up to 50 and returns a list of moving objects.
        For each chunk, it samples 4 representative frames to pass to the model.
        """
        if not os.path.exists(video_frames_dir_path):
            print(f"Error: Frame directory not found at {video_frames_dir_path}")
            return []

        frame_files = sorted([f for f in os.listdir(video_frames_dir_path) if f.endswith('.png')])
        if not frame_files:
            print("Error: No frames found.")
            return []

        all_moving_objects = []
        candidate_list_str = ", ".join(candidate_labels)
        # The prompt is slightly adjusted to reflect that the frames are sampled from a wider segment.
        prompt = (
            "USER: <video>\nHere are four frames sampled from a video segment. Your task is to identify all objects that are moving. "
            "The movement could be natural (like a person walking) or caused by an actor (like a person picking up a cup). "
            f"From the following list, please identify ONLY the objects that are in motion in the video.\n"
            f"List of candidate objects: [{candidate_list_str}]\n\n"
            "Respond with a comma-separated list of the moving objects and nothing else.\n"
            "ASSISTANT:"
        )

        # Process the video in large chunks
        chunk_size = 8
        frame_chunks_to_process = []
        for i in range(0, len(frame_files), chunk_size):
            frame_chunks_to_process.append(frame_files[i:i + chunk_size])

        print(f"Total frames: {len(frame_files)}. "
              f"Processing in {len(frame_chunks_to_process)} chunks of up to {chunk_size} frames each.")

        for i, frame_chunk in enumerate(frame_chunks_to_process):
            print(f"  - Processing chunk {i + 1}/{len(frame_chunks_to_process)} ({len(frame_chunk)} frames)...")

            selected_frames_files = []
            # The model requires exactly 4 frames.
            if len(frame_chunk) < 8:
                # If a chunk has fewer than 4 frames (e.g., at the end of a video), pad it.
                if not frame_chunk: continue
                selected_frames_files = frame_chunk + [frame_chunk[-1]] * (8 - len(frame_chunk))
            else:
                # For chunks with 4 or more frames, sample 4 frames evenly.
                indices = np.linspace(0, len(frame_chunk) - 1, 8, dtype=int)
                selected_frames_files = [frame_chunk[idx] for idx in indices]

            video_frames = [Image.open(os.path.join(video_frames_dir_path, f)).convert("RGB") for f in
                            selected_frames_files]

            try:
                inputs = self.processor(text=prompt, images=video_frames, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=100)

                response_text = self.processor.decode(output[0], skip_special_tokens=True)
                assistant_response = response_text.split("ASSISTANT:")[-1].strip().lower()

                batch_moving_objects = [
                    label for label in candidate_labels if label.lower() in assistant_response
                ]
                if batch_moving_objects:
                    all_moving_objects.extend(batch_moving_objects)

            except Exception as e:
                print(f"An error occurred during VLM processing for chunk {i + 1}: {e}")
                continue

        # Return a unique, compiled list of all identified objects
        return list(set(all_moving_objects))


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
    # video_id_list = [v for v in os.listdir(data_dir / "videos") if v.endswith('.mp4')]
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
        output_file_path = output_dir / f"{video_name}.txt"

        # If the output file already exists, skip processing this video
        if output_file_path.exists():
            print(f"Output for {video_name} already exists. Skipping.")
            continue

        frames_dir = data_dir / "sampled_frames" / video_name
        print(f"\nProcessing video: {video_name}")

        moving_objects = identifier.identify_moving_objects(str(frames_dir), candidate_labels)

        print(f"Identified moving objects: {moving_objects}")

        # Save the results to a text file
        with open(output_file_path, 'w') as f:
            if moving_objects:
                f.write("\n".join(moving_objects))
            else:
                f.write("")  # Write an empty file if no objects were found

        print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
