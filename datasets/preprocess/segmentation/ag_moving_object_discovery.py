import argparse
import json
import os
import re
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _normalize_label(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'^(a|an|the)\s+', '', s)  # drop leading articles
    s = re.sub(r'[^a-z0-9\s]+', '', s)  # strip punctuation
    return s.strip()


def load_all_captions(json_files: List[Path]) -> List[dict]:
    all_caption_data = []
    print("Pre-loading all caption files into memory...")
    for file_path in json_files:
        if not file_path.exists():
            print(f"Warning: Caption file not found: {file_path}")
            continue
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_caption_data.append(data)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}")
    print("Caption loading complete.")
    return all_caption_data


def get_combined_caption_from_memory(video_id: str, all_caption_data: List[dict]) -> str:
    captions = []
    for data in all_caption_data:
        caption = data.get(video_id)
        if caption and isinstance(caption, str):
            captions.append(caption.strip())
    return " ".join(captions)


class MovingObjectLLMIdentifier:

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        print(f"MovingObjectLLMIdentifier initialized on device: {self.device}")
        print(f"Using model: {self.model_id}")

    def _build_messages(self, context: str, candidate_labels: List[str]) -> List[dict]:
        candidate_list_str = ", ".join(candidate_labels)
        content = (
            f"Given the following description of a video:\n"
            f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
            "Based ONLY on the context above, identify which objects from the list below are most likely moving "
            "or being directly interacted with by a person. Interaction implies movement (e.g., picking up a cup, "
            "opening a book, putting on a shoe).\n\n"
            f"Candidate Objects: [{candidate_list_str}]\n\n"
            "Respond with a comma-separated list of ONLY the moving objects. Do not add any other text or explanation."
        )
        return [{"role": "user", "content": content}]

    def identify_moving_objects_from_captions(
            self,
            context: str,
            candidate_labels: List[str],
            max_new_tokens: int = 128,
    ) -> List[str]:
        if not context:
            print("Error: Input context is empty.")
            return []

        normalized_to_canonical = {_normalize_label(lbl): lbl for lbl in candidate_labels}
        all_moving_objects = set()

        messages = self._build_messages(context, candidate_labels)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": terminators,
        }

        with torch.no_grad():
            out_ids = self.model.generate(inputs, **generation_args)

        gen_only = out_ids[:, inputs.shape[1]:]
        text = self.tokenizer.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        response = text.strip().lower()

        parts = [p.strip() for p in response.split(",") if p.strip()]
        for p in parts:
            key = _normalize_label(p)
            if key in normalized_to_canonical:
                all_moving_objects.add(normalized_to_canonical[key])
        if not parts:
            for lbl in candidate_labels:
                if _normalize_label(lbl) in response:
                    all_moving_objects.add(lbl)

        return sorted(all_moving_objects)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify moving objects from video captions using a Llama 3.1 LLM."
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data/rohith/ag",
        help="Path to the root dataset directory (must contain a 'captions' subdirectory)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]
    output_dir = data_dir / "moving_objects"
    os.makedirs(output_dir, exist_ok=True)

    caption_files = [
        data_dir / "captions" / "longvu.json",
        data_dir / "captions" / "charades.json",
        data_dir / "captions" / "chatuniv.json",
    ]
    candidate_labels = [
        "a person", "a bag", "a blanket", "a book", "a box", "a broom", "a chair", "a clothes",
        "a cup", "a dish", "a food", "a laptop", "a paper", "a phone", "a picture", "a pillow",
        "a sandwich", "a shoe", "a towel", "a vacuum", "a glass", "a bottle", "a notebook", "a camera"
    ]

    # Preload all caption data into memory at once
    all_captions_data = load_all_captions(caption_files)
    identifier = MovingObjectLLMIdentifier()
    for video_id in video_id_list:
        video_name = Path(video_id).stem
        output_file_path = output_dir / f"{video_name}.txt"

        if output_file_path.exists():
            print(f"Output for {video_name} already exists. Skipping.")
            continue

        print(f"\nProcessing video: {video_name}")

        # Get combined caption from preloaded data in memory
        combined_caption = get_combined_caption_from_memory(video_id, all_captions_data)

        if not combined_caption:
            print(f"No captions found for {video_id}. Saving empty file.")
            with open(output_file_path, "w") as f:
                f.write("")
            continue

        print(f"  - Combined Context: \"{combined_caption}...\"")
        moving_objects = identifier.identify_moving_objects_from_captions(combined_caption, candidate_labels)
        print(f"  - Identified moving objects: {moving_objects}")

        with open(output_file_path, "w") as f:
            if moving_objects:
                f.write("\n".join(moving_objects))
            else:
                f.write("")
        print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    main()