import argparse
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def _normalize_label(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'^(a|an|the)\s+', '', s)  # drop leading articles
    s = re.sub(r'[^a-z0-9\s]+', '', s)   # strip punctuation
    return s.strip()


class MovingObjectIdentifier:
    """
    Identify moving objects from sampled video frames using Phi-3-Vision.
    Processes frames in chunks (default 8), builds a multi-image prompt with <|image_i|> placeholders,
    and extracts the set of moving objects from the model's response.
    """

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_id = "microsoft/Phi-3-vision-128k-instruct"

        # QUICK FIX: force PyTorch SDPA kernels (no flash-attention, avoids GLIBC mismatch)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        print("MovingObjectIdentifier initialized on device:", self.device)
        print("Using attention implementation: sdpa")

    def _build_messages(self, num_images: int, candidate_labels: List[str]) -> List[dict]:
        candidate_list_str = ", ".join(candidate_labels)
        # Create one user message that references all images in order
        placeholders = "".join(f"<|image_{i+1}|>\n" for i in range(num_images))

        # Phi-3-Vision expects the number/order of placeholders to match the images list
        content = (
            f"{placeholders}"
            f"You are given {num_images} frames sampled from a short video segment. "
            "Identify ONLY the objects that are visibly moving across these frames. "
            "Movement can be natural (e.g., a person walking) or caused by an actor (e.g., picking up a cup). "
            f"From the following list, return ONLY those that are moving:\n"
            f"[{candidate_list_str}]\n\n"
            "Respond with a comma-separated list of the moving objects and nothing else."
        )
        return [{"role": "user", "content": content}]

    def identify_moving_objects(
        self,
        video_frames_dir_path: str,
        candidate_labels: List[str],
        chunk_size: int = 8,
        max_new_tokens: int = 128,
    ) -> List[str]:
        """
        Analyzes video frames in chunks of up to `chunk_size` and returns unique moving objects.
        For each chunk, it samples `chunk_size` representative frames (padding or downsampling as needed).
        """
        if not os.path.exists(video_frames_dir_path):
            print(f"Error: Frame directory not found at {video_frames_dir_path}")
            return []

        frame_files = sorted([f for f in os.listdir(video_frames_dir_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not frame_files:
            print("Error: No frames found.")
            return []

        # Precompute normalized candidate label map to recover canonical output strings
        normalized_to_canonical = {_normalize_label(lbl): lbl for lbl in candidate_labels}

        # Split into chunks
        frame_chunks_to_process = [frame_files[i:i + chunk_size] for i in range(0, len(frame_files), chunk_size)]
        print(f"Total frames: {len(frame_files)}. "
              f"Processing in {len(frame_chunks_to_process)} chunks of up to {chunk_size} frames each.")

        all_moving_objects = set()

        for ci, frame_chunk in enumerate(frame_chunks_to_process, start=1):
            print(f"  - Processing chunk {ci}/{len(frame_chunks_to_process)} ({len(frame_chunk)} frames)...")

            # Ensure exactly `chunk_size` frames per prompt: pad or evenly sample
            if len(frame_chunk) == 0:
                continue
            if len(frame_chunk) < chunk_size:
                selected = frame_chunk + [frame_chunk[-1]] * (chunk_size - len(frame_chunk))
            else:
                idxs = np.linspace(0, len(frame_chunk) - 1, chunk_size, dtype=int)
                selected = [frame_chunk[idx] for idx in idxs]

            # Load images in the same order as placeholders
            images = [
                Image.open(os.path.join(video_frames_dir_path, fname)).convert("RGB")
                for fname in selected
            ]

            try:
                messages = self._build_messages(num_images=len(images), candidate_labels=candidate_labels)
                prompt = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)
                generation_args = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "do_sample": False,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                }

                with torch.no_grad():
                    out_ids = self.model.generate(**inputs, **generation_args)

                # Strip the input tokens to get only the generated completion
                gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
                text = self.processor.batch_decode(
                    gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                response = text.strip().lower()

                # Parse a comma-separated list; normalize and map back to canonical labels
                parts = [p.strip() for p in response.split(",") if p.strip()]
                for p in parts:
                    key = _normalize_label(p)
                    if key in normalized_to_canonical:
                        all_moving_objects.add(normalized_to_canonical[key])

                # Fallback: if the model wrote a sentence, also do a soft contains check
                if not parts:
                    for lbl in candidate_labels:
                        if _normalize_label(lbl) in response:
                            all_moving_objects.add(lbl)

            except Exception as e:
                print(f"An error occurred during Phi-3 processing for chunk {ci}: {e}")
                continue

        return sorted(all_moving_objects)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify moving objects in videos using Phi-3-Vision-128k-Instruct (SDPA attention)."
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
    os.makedirs(output_dir, exist_ok=True)

    candidate_labels = [
        "a person", "a bag", "a blanket", "a book", "a box", "a broom", "a chair", "a clothes",
        "a cup", "a dish", "a food", "a laptop", "a paper", "a phone", "a picture", "a pillow",
        "a sandwich", "a shoe", "a towel", "a vacuum", "a glass", "a bottle", "a notebook", "a camera"
    ]

    identifier = MovingObjectIdentifier()

    for video_id in video_id_list:
        video_name = Path(video_id).stem
        output_file_path = output_dir / f"{video_name}.txt"

        if output_file_path.exists():
            print(f"Output for {video_name} already exists. Skipping.")
            continue

        frames_dir = data_dir / "sampled_frames" / video_id
        print(f"\nProcessing video: {video_name}")

        moving_objects = identifier.identify_moving_objects(str(frames_dir), candidate_labels)

        print(f"Identified moving objects: {moving_objects}")

        with open(output_file_path, "w") as f:
            if moving_objects:
                f.write("\n".join(moving_objects))
            else:
                f.write("")

        print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
