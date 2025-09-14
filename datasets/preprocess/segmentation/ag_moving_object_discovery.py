import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

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
    """
    Multi-stage (deliberate) identifier that keeps its intermediate reasoning STRUCTURED and INTERNAL.
    The model is prompted to reason over multiple steps but is instructed to ONLY return compact JSON
    at each stage. We then combine the stages and output just the final labels.
    """

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
        # Ensure we have a pad token to avoid warnings during generation
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"MovingObjectLLMIdentifier initialized on device: {self.device}")
        print(f"Using model: {self.model_id}")

    # ---------------------------
    # Low-level generation helpers
    # ---------------------------
    def _generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        # Keep tensors on CPU and let accelerate handle device placement for sharded models
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            eos_token_id=terminators,
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(inputs, **gen_kwargs)
        gen_only = out_ids[:, inputs.shape[1]:]
        text = self.tokenizer.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return text.strip()

    @staticmethod
    def _safe_json_loads(s: str) -> Dict[str, Any]:
        """Try to parse JSON; if there is extra text, extract the first JSON object candidate."""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # Attempt to extract the first {...} block
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {}

    # ---------------------------
    # Stage prompts
    # ---------------------------
    def _stage1_extract_interactions(self, context: str) -> Dict[str, Any]:
        """
        Stage 1: From the raw caption context, extract concise, STRUCTURED signals about actions and
        potentially moved/handled objects. The model is told to think privately and return ONLY JSON.
        """
        sys = {
            "role": "system",
            "content": (
                "You are an expert video-caption analyst. You may think step-by-step in a hidden scratchpad, "
                "but DO NOT reveal your thoughts. Return ONLY compact JSON that follows the schema exactly."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Read the CONTEXT and extract three lists (up to 20 items each).\n\n"
                "Return JSON ONLY with keys: actions, objects, objects_interacted.\n\n"
                "- actions: distinct short verb or verb-phrase strings describing actions in the context.\n"
                "- objects: distinct short noun phrases of physical objects mentioned.\n"
                "- objects_interacted: subset of objects that are likely moved or directly manipulated by a person\n"
                "  (e.g., picked up, opened, worn, carried, cleaned).\n\n"
                "STRICT OUTPUT REQUIREMENTS:\n"
                "- Output ONLY valid JSON with these keys.\n"
                "- Do NOT include explanations, reasoning, or extra keys.\n\n"
                f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
            )
        }
        raw = self._generate([sys, user], max_new_tokens=1000)
        return self._safe_json_loads(raw)

    def _stage2_map_to_labels(self, interacted: List[str], candidate_labels: List[str]) -> Dict[str, Any]:
        """
        Stage 2: Map interacted object phrases to the provided candidate label set. The model returns
        ONLY JSON with the selected subset (synonym-aware), no reasoning.
        """
        sys = {
            "role": "system",
            "content": (
                "You map phrases to a provided label set. Think privately; DO NOT reveal thoughts. "
                "Return ONLY JSON that matches the schema."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Given the list of interacted object phrases and the allowed candidate labels, select the subset\n"
                "of labels that best match the interacted objects (use synonyms/singular-plural as needed).\n\n"
                "Return JSON ONLY with: {\"selected_labels\": [<labels from the allowed set>]}\n"
                "- Include a label only if there is a clear match.\n"
                "- Do NOT output labels not in the allowed set.\n\n"
                f"interacted_phrases = {json.dumps(interacted, ensure_ascii=False)}\n"
                f"allowed_labels = {json.dumps(candidate_labels, ensure_ascii=False)}\n"
            )
        }
        raw = self._generate([sys, user], max_new_tokens=1000)
        return self._safe_json_loads(raw)

    # ---------------------------
    # Public API
    # ---------------------------
    def identify_moving_objects_from_captions(
        self,
        context: str,
        candidate_labels: List[str],
        max_new_tokens: int = 128,
    ) -> List[str]:
        """
        Multi-stage pipeline:
          1) Extract actions/objects/objects_interacted (JSON only).
          2) Map objects_interacted -> candidate labels (JSON only).
          3) Fallback to deterministic normalization if needed.
        """
        if not context:
            print("Error: Input context is empty.")
            return []

        normalized_to_canonical = {_normalize_label(lbl): lbl for lbl in candidate_labels}

        # Stage 1
        s1 = self._stage1_extract_interactions(context)
        interacted = s1.get("objects_interacted") if isinstance(s1, dict) else None
        if not interacted or not isinstance(interacted, list):
            interacted = []

        # Stage 2
        s2 = self._stage2_map_to_labels(interacted, candidate_labels)
        picked = []
        if isinstance(s2, dict) and isinstance(s2.get("selected_labels"), list):
            picked = [lbl for lbl in s2["selected_labels"] if isinstance(lbl, str)]

        # If model mapping failed or returned empty, fall back to deterministic string matching
        if not picked:
            seen = set()
            for phrase in interacted:
                key = _normalize_label(phrase)
                if key in normalized_to_canonical:
                    seen.add(normalized_to_canonical[key])
                else:
                    # heuristic partial match against candidate labels
                    for nl, canon in normalized_to_canonical.items():
                        if nl in key or key in nl:
                            seen.add(canon)
            picked = sorted(seen)

        # Final sanity: keep only items from the allowed set
        allowed = set(candidate_labels)
        final_labels = sorted([p for p in picked if p in allowed])
        return final_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify moving objects from video captions using a multi-stage Llama 3.1 pipeline."
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data/rohith/ag",
        help="Path to the root dataset directory (must contain a 'captions' subdirectory)."
    )
    return parser.parse_args()


essential_video_ids = [
    "0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4",
    "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"
]


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    video_id_list = essential_video_ids
    # video_id_list = os.listdir(data_dir / "videos")
    output_dir = data_dir / "moving_objects" / "llama3.1"
    os.makedirs(output_dir, exist_ok=True)

    caption_files = [
        data_dir / "captions" / "longvu.json",
        data_dir / "captions" / "charades.json",
        data_dir / "captions" / "chatuniv.json",
    ]
    candidate_labels = [
        "person", "bag", "blanket", "book", "box", "broom", "chair", "clothes",
        "cup", "dish", "food", "laptop", "paper", "phone", "picture", "pillow",
        "sandwich", "shoe", "towel", "vacuum", "glass", "bottle", "notebook", "camera",
        "bed", "closet", "cabinet", "door", "doorknob", "groceries", "mirror", "refrigerator",
        "sofa", "couch", "table", "television", "window"
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
        combined_caption = get_combined_caption_from_memory(video_name, all_captions_data)
        if not combined_caption:
            print(f"No captions found for {video_name}. Saving empty file.")
            with open(output_file_path, "w") as f:
                f.write("")
            continue

        # Multi-stage deliberate reasoning (intermediate JSON kept internal)
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
