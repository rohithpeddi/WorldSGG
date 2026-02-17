#!/usr/bin/env python3
"""
lora_ag.py
==========
LoRA-based fine-tuning of MLLMs (InternVL2.5, QwenVL2.5, KimiVL) for
scene graph generation on the Action Genome dataset.

Three LoRA fine-tuning variants (--lora_target):
  • language  — LoRA on language model attention layers only
  • vision    — LoRA on vision encoder attention layers only
  • full      — LoRA on both language + vision layers

Usage:
    python lora_ag.py \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --ag_root_directory /data/rohith/ag \
        --lora_target language \
        --output_dir /data/rohith/ag/lora_checkpoints/ \
        --num_epochs 3 \
        --batch_size 4
"""

import os
import sys
import json
import pickle
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relationship label sets (Action Genome vocabulary)
# ---------------------------------------------------------------------------

ATTENTION_RELATIONSHIPS = [
    "looking_at", "not_looking_at", "unsure",
]

CONTACTING_RELATIONSHIPS = [
    "carrying", "covered_by", "drinking_from", "eating",
    "have_it_on_the_back", "holding", "leaning_on", "lying_on",
    "not_contacting", "other_relationship", "sitting_on", "standing_on",
    "touching", "twisting", "wearing", "wiping", "writing_on",
]

SPATIAL_RELATIONSHIPS = [
    "above", "beneath", "in_front_of", "behind", "on_the_side_of", "in",
]

# Class name list from AG (index 0 is background, 1 is person, then objects)
OBJECT_CLASSES = [
    '__background__', 'person', 'bag', 'bed', 'blanket', 'book', 'box',
    'broom', 'chair', 'closet/cabinet', 'clothes', 'cup/glass/bottle',
    'dish', 'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries',
    'laptop', 'light', 'medicine', 'mirror', 'paper/notebook',
    'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich',
    'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel',
    'vacuum', 'window',
]

# ---------------------------------------------------------------------------
# Prompt / Response templates
# ---------------------------------------------------------------------------

SG_QUERY_TEMPLATE = """You are analyzing a video frame from a scene. The object "{object_name}" IS visible in the current frame. A person IS also visible in this frame.

Based on the visual context, predict the relationships between the person and the object "{object_name}".

You must answer three questions:

1. ATTENTION relationship (how is the person attending to the "{object_name}"?):
   Pick EXACTLY ONE label from: {attention_labels}

2. CONTACTING relationship (what physical contact exists between the person and the "{object_name}"?):
   Pick ONE OR MORE labels from: {contacting_labels}

3. SPATIAL relationship (where is the "{object_name}" relative to the person?):
   Pick ONE OR MORE labels from: {spatial_labels}

Respond ONLY in the following JSON format (attention is a single string; contacting and spatial are lists of strings):
{{
    "attention": "<label>",
    "contacting": ["<label>", ...],
    "spatial": ["<label>", ...]
}}"""


def build_gt_response_json(
    attention_indices: torch.Tensor,
    contacting_indices: torch.Tensor,
    spatial_indices: torch.Tensor,
) -> str:
    """Convert GT relationship index tensors to a JSON string answer."""
    att_labels = [ATTENTION_RELATIONSHIPS[i] for i in attention_indices.tolist()]
    cont_labels = [CONTACTING_RELATIONSHIPS[i] for i in contacting_indices.tolist()]
    spa_labels = [SPATIAL_RELATIONSHIPS[i] for i in spatial_indices.tolist()]

    response = {
        "attention": att_labels[0] if att_labels else "unsure",
        "contacting": cont_labels if cont_labels else ["not_contacting"],
        "spatial": spa_labels if spa_labels else ["in_front_of"],
    }
    return json.dumps(response)


# ---------------------------------------------------------------------------
# Per-model LoRA target module configs
# ---------------------------------------------------------------------------

LORA_TARGET_MODULES = {
    # InternVL2.5-8B
    "internvl": {
        "language": [
            "language_model.model.layers.*.self_attn.q_proj",
            "language_model.model.layers.*.self_attn.k_proj",
            "language_model.model.layers.*.self_attn.v_proj",
            "language_model.model.layers.*.self_attn.o_proj",
        ],
        "vision": [
            "vision_model.encoder.layers.*.attn.qkv",
            "vision_model.encoder.layers.*.attn.proj",
        ],
    },
    # QwenVL 2.5
    "qwenvl": {
        "language": [
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.self_attn.o_proj",
        ],
        "vision": [
            "visual.blocks.*.attn.qkv",
            "visual.blocks.*.attn.proj",
        ],
    },
    # KimiVL
    "kimikvl": {
        "language": [
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.self_attn.o_proj",
        ],
        "vision": [
            "vision_model.encoder.layers.*.self_attn.q_proj",
            "vision_model.encoder.layers.*.self_attn.k_proj",
            "vision_model.encoder.layers.*.self_attn.v_proj",
            "vision_model.encoder.layers.*.self_attn.out_proj",
        ],
    },
}


def get_target_modules(model_key: str, lora_target: str) -> List[str]:
    """
    Return the list of target module name patterns for LoRA.

    Args:
        model_key: One of 'internvl', 'qwenvl', 'kimikvl'
        lora_target: One of 'language', 'vision', 'full'
    """
    if model_key not in LORA_TARGET_MODULES:
        raise ValueError(
            f"Unknown model_key '{model_key}'. "
            f"Choose from: {list(LORA_TARGET_MODULES.keys())}"
        )

    cfg = LORA_TARGET_MODULES[model_key]
    if lora_target == "language":
        return cfg["language"]
    elif lora_target == "vision":
        return cfg["vision"]
    elif lora_target == "full":
        return cfg["language"] + cfg["vision"]
    else:
        raise ValueError(f"Unknown lora_target '{lora_target}'. Choose from: language, vision, full")


def resolve_model_key(model_name: str) -> str:
    """Map model_name to a canonical key for LORA_TARGET_MODULES lookup."""
    name_lower = model_name.lower()
    if "internvl" in name_lower:
        return "internvl"
    elif "qwen" in name_lower:
        return "qwenvl"
    elif "kimi" in name_lower:
        return "kimikvl"
    else:
        raise ValueError(
            f"Cannot determine model type from '{model_name}'. "
            f"Name must contain 'internvl', 'qwen', or 'kimi'."
        )


# ---------------------------------------------------------------------------
# Dataset: Action Genome Scene Graph for LoRA Fine-Tuning
# ---------------------------------------------------------------------------

class AGSceneGraphDataset(Dataset):
    """
    PyTorch Dataset for Action Genome scene graph generation fine-tuning.

    Each sample is a (image, query_prompt, gt_response_json) tuple for one
    (frame, object) pair. The query asks for relationship predictions;
    the response is the GT relationship labels in JSON format.
    """

    def __init__(
        self,
        ag_root_directory: str,
        phase: str = "train",
        max_objects_per_frame: int = 10,
        max_videos: Optional[int] = None,
    ):
        super().__init__()
        self.ag_root = Path(ag_root_directory)
        self.frames_path = self.ag_root / "frames"
        self.phase = phase

        # Load annotations via the existing BaseAG mechanism
        # We build a flat list of (frame_path, object_name, gt_response) entries
        self.samples: List[Dict[str, Any]] = []

        self._build_samples(max_objects_per_frame, max_videos)
        logger.info(f"AGSceneGraphDataset [{phase}]: {len(self.samples)} samples from AG")

    def _build_samples(self, max_objects_per_frame: int, max_videos: Optional[int]):
        """
        Load AG GT annotations and build (frame_image_path, query, gt_response)
        triples for training.
        """
        annotations_path = self.ag_root / "annotations"

        # Load person and object bboxes + relationship annotations
        person_bbox_path = annotations_path / "person_bbox.pkl"
        object_bbox_path = annotations_path / "object_bbox_and_relationship.pkl"

        if not person_bbox_path.exists() or not object_bbox_path.exists():
            logger.error(f"Missing annotation files in {annotations_path}")
            return

        with open(person_bbox_path, "rb") as f:
            person_bbox = pickle.load(f)
        with open(object_bbox_path, "rb") as f:
            object_bbox = pickle.load(f)

        # Load relationship class names
        attention_rels = list(ATTENTION_RELATIONSHIPS)
        contacting_rels = list(CONTACTING_RELATIONSHIPS)
        spatial_rels = list(SPATIAL_RELATIONSHIPS)

        attention_labels_str = ", ".join(attention_rels)
        contacting_labels_str = ", ".join(contacting_rels)
        spatial_labels_str = ", ".join(spatial_rels)

        # Filter frames by phase (train/test)
        phase_key = "train" if self.phase == "train" else "testing"
        video_frames: Dict[str, List[str]] = {}

        for frame_key in person_bbox.keys():
            # Check phase
            if not object_bbox.get(frame_key):
                continue
            metadata = object_bbox[frame_key][0].get("metadata", {})
            if metadata.get("set") != phase_key:
                continue
            # Check person bbox exists
            if person_bbox[frame_key]["bbox"].shape[0] == 0:
                continue

            video_name = frame_key.split("/")[0]
            if video_name not in video_frames:
                video_frames[video_name] = []
            video_frames[video_name].append(frame_key)

        # Optionally limit number of videos
        video_names = sorted(video_frames.keys())
        if max_videos is not None:
            video_names = video_names[:max_videos]

        for video_name in video_names:
            frames = video_frames[video_name]
            for frame_key in frames:
                frame_path = str(self.frames_path / frame_key)

                # Get visible objects with relationships
                obj_count = 0
                for obj_ann in object_bbox[frame_key]:
                    if not obj_ann.get("visible", False):
                        continue
                    if obj_ann.get("bbox") is None:
                        continue

                    cls_idx = obj_ann.get("class")
                    if cls_idx is None or cls_idx == 0:
                        continue

                    # Get object name
                    if isinstance(cls_idx, int) and cls_idx < len(OBJECT_CLASSES):
                        object_name = OBJECT_CLASSES[cls_idx]
                    elif isinstance(cls_idx, str):
                        object_name = cls_idx
                    else:
                        continue

                    if object_name == "person" or object_name == "__background__":
                        continue

                    # Extract GT relationship labels
                    att_rel = obj_ann.get("attention_relationship", [])
                    cont_rel = obj_ann.get("contacting_relationship", [])
                    spa_rel = obj_ann.get("spatial_relationship", [])

                    # Build GT response
                    gt_attention = att_rel[0] if att_rel else "unsure"
                    gt_contacting = cont_rel if cont_rel else ["not_contacting"]
                    gt_spatial = spa_rel if spa_rel else ["in_front_of"]

                    gt_response = json.dumps({
                        "attention": gt_attention,
                        "contacting": gt_contacting,
                        "spatial": gt_spatial,
                    })

                    # Build query prompt
                    query = SG_QUERY_TEMPLATE.format(
                        object_name=object_name,
                        attention_labels=attention_labels_str,
                        contacting_labels=contacting_labels_str,
                        spatial_labels=spatial_labels_str,
                    )

                    self.samples.append({
                        "frame_path": frame_path,
                        "video_name": video_name,
                        "frame_key": frame_key,
                        "object_name": object_name,
                        "query": query,
                        "gt_response": gt_response,
                    })

                    obj_count += 1
                    if obj_count >= max_objects_per_frame:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Model loading + LoRA application
# ---------------------------------------------------------------------------

def load_model_for_lora(
    model_name: str,
    lora_target: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_qlora: bool = False,
    gradient_checkpointing: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    Load the base model (HuggingFace) and wrap it with LoRA adapters.

    Returns (model_with_lora, processor/tokenizer, model_key)
    """
    from peft import LoraConfig, get_peft_model, TaskType

    model_key = resolve_model_key(model_name)
    target_modules = get_target_modules(model_key, lora_target)

    # Convert glob-style patterns to regex for PEFT
    # e.g., "model.layers.*.self_attn.q_proj" → regex matching
    target_module_names = []
    for pattern in target_modules:
        # Extract the final module name for PEFT's target_modules parameter
        # PEFT uses simple string matching on module names
        parts = pattern.split(".")
        target_module_names.append(parts[-1])
    # Deduplicate
    target_module_names = list(set(target_module_names))

    logger.info(f"Model key: {model_key}")
    logger.info(f"LoRA target: {lora_target}")
    logger.info(f"Target module names: {target_module_names}")

    # --- Load base model ---
    quantization_config = None
    if use_qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = None
    tokenizer = None

    if model_key == "internvl":
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"Loading InternVL model from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False,
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto" if use_qlora else None,
        )
        if not use_qlora:
            model = model.cuda()

    elif model_key == "qwenvl":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        logger.info(f"Loading QwenVL model from {model_name}")
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto" if use_qlora else None,
        )
        if not use_qlora:
            model = model.cuda()

    elif model_key == "kimikvl":
        from transformers import AutoModelForCausalLM, AutoProcessor
        logger.info(f"Loading KimiVL model from {model_name}")
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto" if use_qlora else None,
        )
        if not use_qlora:
            model = model.cuda()

    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Configure and apply LoRA ---
    # Determine which modules to target based on lora_target
    modules_to_save = None
    if lora_target == "vision" or lora_target == "full":
        # When targeting vision, we may also want to save the vision projection
        modules_to_save = []

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_module_names,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    effective_processor = processor if processor is not None else tokenizer
    return model, effective_processor, model_key


# ---------------------------------------------------------------------------
# Collate function for training
# ---------------------------------------------------------------------------

def build_conversation_text(
    query: str,
    gt_response: str,
    model_key: str,
) -> str:
    """
    Build a single training conversation string that can be tokenized.
    The format follows the model's chat template pattern.
    """
    # Simple instruction-response format that works across models
    return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{gt_response}<|im_end|>"


def collate_fn_factory(processor, model_key: str, max_length: int = 2048):
    """
    Create a collate function that prepares batches for training.

    For simplicity, we treat each sample as a text-only fine-tuning task
    (the frame path can be used for image-conditioned training if the
    model/processor supports it, but the text-based approach is more
    portable across all three model families).
    """
    from PIL import Image

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images_list = []

        for sample in batch:
            query = sample["query"]
            gt_response = sample["gt_response"]

            # Load the frame image
            frame_path = sample["frame_path"]
            try:
                image = Image.open(frame_path).convert("RGB")
                # Resize for memory efficiency
                image = image.resize((448, 448))
            except Exception as e:
                logger.warning(f"Failed to load image {frame_path}: {e}. Using blank.")
                image = Image.new("RGB", (448, 448))

            images_list.append(image)

            # Build conversation text
            conv_text = build_conversation_text(query, gt_response, model_key)
            texts.append(conv_text)

        # Tokenize
        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor

        # For image-conditioned training, we tokenize text + image together
        # For text-only fallback, just tokenize the text
        try:
            # Try multi-modal encoding (works for QwenVL, KimiVL)
            encodings = processor(
                text=texts,
                images=images_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        except (TypeError, AttributeError):
            # Fallback: text-only tokenization
            encodings = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

        # Create labels: same as input_ids but with padding tokens set to -100
        labels = encodings["input_ids"].clone()
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        encodings["labels"] = labels
        return encodings

    return collate_fn


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    processor,
    model_key: str,
    train_dataset: AGSceneGraphDataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    max_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 500,
    logging_steps: int = 10,
    use_deepspeed: bool = False,
    deepspeed_config: Optional[str] = None,
):
    """Fine-tune the model with LoRA using HuggingFace Trainer."""
    from transformers import TrainingArguments, Trainer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    collate = collate_fn_factory(processor, model_key, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=deepspeed_config if use_deepspeed else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
    )

    logger.info("Starting LoRA fine-tuning...")
    logger.info(f"  Num samples: {len(train_dataset)}")
    logger.info(f"  Num epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Output dir: {output_path}")

    train_result = trainer.train()

    # Save final LoRA adapter
    final_path = output_path / "final_adapter"
    model.save_pretrained(str(final_path))
    logger.info(f"Final LoRA adapter saved to {final_path}")

    # Save training metrics
    metrics = train_result.metrics
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    return train_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LoRA fine-tuning of MLLMs for scene graph generation "
            "on the Action Genome dataset."
        ),
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help=(
            "HuggingFace model name/path. Examples: "
            "OpenGVLab/InternVL2_5-8B, "
            "Qwen/Qwen2.5-VL-7B-Instruct, "
            "moonshotai/Kimi-VL-A3B-Instruct"
        ),
    )
    parser.add_argument(
        "--ag_root_directory", type=str,
        default="/data/rohith/ag",
        help="Root directory of the Action Genome dataset",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/data/rohith/ag/lora_checkpoints/",
        help="Output directory for LoRA adapter checkpoints",
    )
    parser.add_argument(
        "--lora_target", type=str,
        default="language",
        choices=["language", "vision", "full"],
        help=(
            "Which parts of the model to apply LoRA to: "
            "'language' (LLM attention only), "
            "'vision' (vision encoder only), "
            "'full' (both language + vision)"
        ),
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="LoRA rank (r). Higher = more capacity but more parameters.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA alpha (scaling factor). Typically 2x rank.",
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="Dropout probability for LoRA layers.",
    )
    parser.add_argument(
        "--use_qlora", action="store_true", default=False,
        help="Enable QLoRA: 4-bit quantization + LoRA for lower memory usage.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--max_videos", type=int, default=None,
        help="Limit the number of training videos (for dev/debug).",
    )
    parser.add_argument(
        "--max_objects_per_frame", type=int, default=10,
        help="Maximum number of objects to process per frame.",
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", default=False,
        help="Enable DeepSpeed for distributed training.",
    )
    parser.add_argument(
        "--deepspeed_config", type=str, default=None,
        help="Path to DeepSpeed configuration JSON file.",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=True,
        help="Enable gradient checkpointing to reduce memory.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve output directory with model and target info
    model_key = resolve_model_key(args.model_name)
    output_dir = os.path.join(
        args.output_dir,
        f"{model_key}_{args.lora_target}_r{args.lora_rank}",
    )

    logger.info("=" * 60)
    logger.info("LoRA Fine-Tuning for Action Genome Scene Graph Generation")
    logger.info("=" * 60)
    logger.info(f"  Model: {args.model_name} ({model_key})")
    logger.info(f"  LoRA target: {args.lora_target}")
    logger.info(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    logger.info(f"  QLoRA: {args.use_qlora}")
    logger.info(f"  Output: {output_dir}")

    # 1. Build dataset
    logger.info("Building training dataset...")
    train_dataset = AGSceneGraphDataset(
        ag_root_directory=args.ag_root_directory,
        phase="train",
        max_objects_per_frame=args.max_objects_per_frame,
        max_videos=args.max_videos,
    )

    if len(train_dataset) == 0:
        logger.error("No training samples found! Check AG root directory and annotations.")
        return

    # 2. Load model with LoRA
    logger.info("Loading model and applying LoRA adapters...")
    model, processor, model_key = load_model_for_lora(
        model_name=args.model_name,
        lora_target=args.lora_target,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_qlora=args.use_qlora,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # 3. Train
    train(
        model=model,
        processor=processor,
        model_key=model_key,
        train_dataset=train_dataset,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        use_deepspeed=args.use_deepspeed,
        deepspeed_config=args.deepspeed_config,
    )

    # 4. Save config
    config_path = os.path.join(output_dir, "lora_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "model_key": model_key,
            "lora_target": args.lora_target,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "use_qlora": args.use_qlora,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ag_root_directory": args.ag_root_directory,
            "max_videos": args.max_videos,
        }, f, indent=2)
    logger.info(f"LoRA config saved to {config_path}")
    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    main()
