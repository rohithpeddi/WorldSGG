#!/usr/bin/env python3
"""
zero_shot_ag.py
===============
Zero-shot scene graph generation for the Action Genome dataset.

For each video:
    1. Load GT annotations to get **observed** objects in each frame.
    2. Per frame, for every observed object, generate a combined relationship
       prompt (attention / contacting / spatial) between the person and that
       object.
    3. Answer each prompt zero-shot from the relevant video clip.
       - Standard: load a short clip centered on the annotated frame.
       - Fast (default): load one sampled video tensor once; slice centered
                         chunks per frame; batch prompts via mllm_batch_response.
    4. Save per-video results as a pickle file.

Optionally loads LoRA adapters for fine-tuned model inference (--lora_path).
"""

import os
import sys
import json
import pickle
import argparse
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple

import torch
from torchvision import transforms
from tqdm import tqdm

# Allow imports from the parent package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.vgent import Vgent
from core.ag_data import AgDataBBAnnotations
from core.logger_utils import setup_logging
from models.utils import fetch_video, resize_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relationship label sets (Action Genome vocabulary)
# ---------------------------------------------------------------------------

ATTENTION_RELATIONSHIPS = [
    "looking_at",
    "not_looking_at",
    "unsure",
]

CONTACTING_RELATIONSHIPS = [
    "carrying",
    "covered_by",
    "drinking_from",
    "eating",
    "have_it_on_the_back",
    "holding",
    "leaning_on",
    "lying_on",
    "not_contacting",
    "other_relationship",
    "sitting_on",
    "standing_on",
    "touching",
    "twisting",
    "wearing",
    "wiping",
    "writing_on",
]

SPATIAL_RELATIONSHIPS = [
    "above",
    "beneath",
    "in_front_of",
    "behind",
    "on_the_side_of",
    "in",
]

# ---------------------------------------------------------------------------
# Prompt template for scene graph relationship prediction
# ---------------------------------------------------------------------------

AG_SCENE_GRAPH_PROMPT = """You are analyzing a video clip from a scene. The object "{object_name}" IS visible in the current frame. A person IS also visible in this frame.

Based on the video context, predict the relationships between the person and the object "{object_name}".

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

# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

_SPLIT_JSON_PATH = Path(__file__).parent / "video_splits.json"


def load_split_video_ids(split_name: str) -> set:
    """Load video IDs for *split_name* from ``video_splits.json``."""
    with open(_SPLIT_JSON_PATH, "r") as f:
        splits = json.load(f)
    if split_name not in splits:
        raise ValueError(
            f"Split '{split_name}' not found in {_SPLIT_JSON_PATH}. "
            f"Available splits: {list(splits.keys())}"
        )
    return {Path(v).stem for v in splits[split_name]}


def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    """First-letter bucketing for split assignment."""
    stem = Path(video_id).stem
    if not stem:
        return None
    c = stem[0]
    if c.isdigit() and int(c) < 5:
        return "04"
    elif c.isdigit() and int(c) >= 5:
        return "59"
    for bucket, letters in [
        ("AD", "ABCD"), ("EH", "EFGH"), ("IL", "IJKL"),
        ("MP", "MNOP"), ("QT", "QRST"), ("UZ", "UVWXYZ"),
    ]:
        if c in letters:
            return bucket
    return None


# ---------------------------------------------------------------------------
# Core processor class
# ---------------------------------------------------------------------------

class ActionGenomeSceneGraphProcessor:
    """Zero-shot scene graph generation: predicts relationships
    for **observed** objects in each frame of Action Genome videos."""

    def __init__(
        self,
        ag_root_directory: str,
        output_dir: str,
        model_name: str,
        split: str,
        tensor_parallel_size: int = 1,
        use_vllm: bool = True,
        batch_size: int = 32,
        lora_path: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # AG data loader
        self.ag_data = AgDataBBAnnotations(
            ag_root_directory=ag_root_directory
        )

        # Frame directories
        self.frames_dir = Path(ag_root_directory) / "frames"
        self.frames_annotated_dir = Path(ag_root_directory) / "frames_annotated"

        self.batch_size = int(batch_size)
        self.lora_path = lora_path

        # Build Vgent args
        class Args:
            def __init__(self_inner):
                self_inner.model_name = model_name
                self_inner.vision_encoder_path = "google/siglip-so400m-patch14-384"
                self_inner.vision_feature_layer = -1
                self_inner.vision_feature_select_layer = -1
                self_inner.use_flash_attn = False
                self_inner.fps = 1
                self_inner.chunk_size = 128
                self_inner.total_pixels = 128000
                self_inner.split = split
                self_inner.tensor_parallel_size = tensor_parallel_size
                self_inner.use_vllm = use_vllm
                self_inner.n_retrieval = 20
                self_inner.n_refine = 5
                self_inner.uniform_frame = 450

        self.args = Args()

        # If LoRA path provided, load model with LoRA adapters (non-vLLM only)
        if self.lora_path:
            logger.info(f"LoRA path provided: {self.lora_path}. Loading model with LoRA adapters...")
            self.args.use_vllm = False
            self.vgent = Vgent(self.args)
            self._apply_lora_adapter()
        else:
            self.vgent = Vgent(self.args)

        self.fast_mode = True  # default: fast mode enabled

        logger.info("Model loaded successfully (scene graph zero-shot mode).")

    def _apply_lora_adapter(self):
        """Apply LoRA adapter weights to the loaded base model."""
        try:
            from peft import PeftModel
            base_model = self.vgent.model.model
            self.vgent.model.model = PeftModel.from_pretrained(
                base_model, self.lora_path
            )
            self.vgent.model.model.eval()
            logger.info(f"LoRA adapter loaded from {self.lora_path}")
        except ImportError:
            logger.error("peft is required for LoRA inference. Install with: pip install peft")
            raise
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            raise

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_objects_from_frame(frame_data) -> Set[str]:
        """Return the set of non-person object labels present in a single frame."""
        if isinstance(frame_data, dict):
            objects_list = frame_data.get("objects", [])
        elif isinstance(frame_data, list):
            objects_list = frame_data
        else:
            return set()

        labels = set()
        for obj in objects_list:
            if isinstance(obj, dict):
                label = obj.get("label", None)
                if label and label != "person":
                    labels.add(label)
        return labels

    def get_video_objects(self, video_data: Dict[str, Any]) -> Set[str]:
        """Collect all unique non-person object labels across all frames."""
        bbox_frames = video_data.get("bbox_frames", {})
        all_objects: Set[str] = set()
        for _, frame_data in bbox_frames.items():
            all_objects.update(self._extract_objects_from_frame(frame_data))
        return all_objects

    def get_observed_objects_per_frame(
        self,
        video_data: Dict[str, Any],
    ) -> Dict[str, Set[str]]:
        """
        For every annotated frame, return the set of non-person objects
        that are observed (present) in that frame.

        Returns {frame_stem: set_of_observed_object_labels}.
        """
        bbox_frames = video_data.get("bbox_frames", {})
        observed_map: Dict[str, Set[str]] = {}
        for frame_stem, frame_data in bbox_frames.items():
            frame_objects = self._extract_objects_from_frame(frame_data)
            if frame_objects:
                observed_map[frame_stem] = frame_objects
        return observed_map

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_relationship_queries(
        observed_objects: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        For each observed object, create a combined relationship query prompt.

        Returns a list of dicts: {object_name: str, prompt: str}
        """
        queries: List[Dict[str, Any]] = []
        attention_labels = ", ".join(ATTENTION_RELATIONSHIPS)
        contacting_labels = ", ".join(CONTACTING_RELATIONSHIPS)
        spatial_labels = ", ".join(SPATIAL_RELATIONSHIPS)

        for obj in sorted(observed_objects):
            prompt = AG_SCENE_GRAPH_PROMPT.format(
                object_name=obj,
                attention_labels=attention_labels,
                contacting_labels=contacting_labels,
                spatial_labels=spatial_labels,
            )
            queries.append({"object_name": obj, "prompt": prompt})
        return queries

    # ------------------------------------------------------------------
    # Frame / clip loading
    # ------------------------------------------------------------------

    def get_all_frames(self, video_id: str) -> List[int]:
        video_dir = self.frames_dir / video_id
        if not video_dir.exists():
            return []
        frames = []
        for fn in os.listdir(video_dir):
            if fn.endswith((".png", ".jpg")):
                m = re.search(r"(\d+)", fn)
                if m:
                    frames.append(int(m.group(1)))
        return sorted(frames)

    def _build_frame_map(self, video_id: str) -> Dict[int, str]:
        """Map frame index → filename inside frames_dir/video_id."""
        video_dir = self.frames_dir / video_id
        fmap: Dict[int, str] = {}
        if not video_dir.exists():
            return fmap
        for fn in os.listdir(video_dir):
            if fn.endswith((".png", ".jpg")):
                m = re.search(r"(\d+)", fn)
                if m:
                    fmap[int(m.group(1))] = fn
        return fmap

    def load_video_clip(self, image_paths: List[str]):
        """Load a list of frame image paths as a single video tensor [T,C,H,W]."""
        if not image_paths:
            return None
        input_data = {
            "video": image_paths,
            "min_pixels": 28 * 28,
            "total_pixels": self.args.total_pixels,
        }
        try:
            images = fetch_video(input_data, resize=False)
        except Exception as e:
            logger.error(f"Error fetching video images: {e}")
            return None
        if not images:
            return None
        try:
            tensor = torch.stack(
                [transforms.PILToTensor()(img) for img in images]
            ).float()
        except Exception as e:
            logger.error(f"Error stacking images to tensor: {e}")
            return None

        video_tensor, _ = resize_video(
            tensor, self.args.fps, total_pixels=self.args.total_pixels
        )
        return [video_tensor]

    def _load_clip_around_frame(
        self,
        frame_idx: int,
        frame_map: Dict[int, str],
        video_id: str,
        window: int = 15,
    ):
        """Load a short clip centred on frame_idx (±window frames)."""
        video_dir = self.frames_dir / video_id
        sorted_indices = sorted(frame_map.keys())
        if not sorted_indices:
            return None

        lo = max(sorted_indices[0], frame_idx - window)
        hi = min(sorted_indices[-1], frame_idx + window)
        paths = []
        for idx in range(lo, hi + 1, 2):  # every other frame
            if idx in frame_map:
                paths.append(str(video_dir / frame_map[idx]))
        if not paths:
            return None
        return self.load_video_clip(paths)

    # ------------------------------------------------------------------
    # Clip slicing helpers (fast mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _temporal_downsample(video_tensor: torch.Tensor, max_frames: int) -> torch.Tensor:
        """Evenly downsample along time dimension to <= max_frames."""
        if video_tensor is None:
            return video_tensor
        t = int(video_tensor.shape[0])
        if t <= max_frames:
            return video_tensor
        idx = torch.linspace(0, t - 1, steps=max_frames, device=video_tensor.device).long()
        return video_tensor.index_select(0, idx)

    def _slice_centered_chunk(
        self,
        video_tensor: torch.Tensor,
        center_t: int,
        max_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Slice a chunk (<= max_frames) centered at center_t from a [T,C,H,W] tensor.
        """
        if max_frames is None:
            max_frames = int(self.args.chunk_size)

        T = int(video_tensor.shape[0])
        if T <= max_frames:
            return video_tensor

        half = max_frames // 2
        start = max(0, int(center_t) - half)
        end = start + max_frames
        if end > T:
            end = T
            start = max(0, end - max_frames)
        chunk = video_tensor[start:end]
        if chunk.shape[0] > max_frames:
            chunk = self._temporal_downsample(chunk, max_frames)
        return chunk

    # ------------------------------------------------------------------
    # Zero-shot inference
    # ------------------------------------------------------------------

    def _answer_query_zero_shot(
        self,
        query_prompt: str,
        clip_tensor: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> Optional[str]:
        """
        Zero-shot: directly ask the VLM on the given clip.
        """
        try:
            clip_tensor = self._temporal_downsample(clip_tensor, int(self.args.chunk_size))
            resp = self.vgent.model.mllm_response(
                query_prompt, [clip_tensor], max_new_tokens=max_new_tokens
            )
            return resp.strip() if resp else None
        except Exception as e:
            logger.error(f"Zero-shot inference error: {e}")
            return None

    def _batch_answer_zero_shot(
        self,
        prompts_and_clips: List[Tuple[str, torch.Tensor]],
        max_new_tokens: int = 128,
    ) -> List[Optional[str]]:
        """
        Batched zero-shot inference using mllm_batch_response, chunked to self.batch_size.
        """
        if not prompts_and_clips:
            return []

        out: List[Optional[str]] = []
        for i in range(0, len(prompts_and_clips), self.batch_size):
            chunk = prompts_and_clips[i:i + self.batch_size]
            batch_prompts = []
            for prompt_text, clip_tensor in chunk:
                clip_tensor = self._temporal_downsample(clip_tensor, int(self.args.chunk_size))
                batch_prompts.append({
                    "text": prompt_text,
                    "video_inputs": [clip_tensor],
                    "max_new_tokens": max_new_tokens,
                })
            try:
                responses = self.vgent.model.mllm_batch_response(batch_prompts)
                out.extend([r.strip() if r else None for r in responses])
            except Exception as e:
                logger.error(f"Zero-shot batch inference error (chunk {i}): {e}")
                out.extend([None] * len(chunk))
        return out

    # ------------------------------------------------------------------
    # Parse LLM response into validated labels
    # ------------------------------------------------------------------

    @staticmethod
    def parse_relationship_response(response: Optional[str]) -> Dict[str, Any]:
        """
        Parse a JSON response from the LLM and validate that each
        relationship label belongs to its allowed set.

        Returns:
            {
                "attention": str,
                "contacting": list[str],
                "spatial": list[str],
            }
        """
        default: Dict[str, Any] = {
            "attention": "unknown",
            "contacting": ["unknown"],
            "spatial": ["unknown"],
        }
        if not response:
            return default

        # Try to extract JSON from (possibly noisy) LLM output
        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    return default
            else:
                return default

        result: Dict[str, Any] = dict(default)

        # attention (single)
        att = parsed.get("attention", "")
        if isinstance(att, list):
            att = att[0] if att else ""
        att = str(att).strip().lower()
        if att in ATTENTION_RELATIONSHIPS:
            result["attention"] = att

        # contacting (multi)
        raw_cont = parsed.get("contacting", [])
        if isinstance(raw_cont, str):
            raw_cont = [raw_cont]
        valid_cont = [
            str(c).strip().lower()
            for c in raw_cont
            if str(c).strip().lower() in CONTACTING_RELATIONSHIPS
        ]
        if valid_cont:
            result["contacting"] = valid_cont

        # spatial (multi)
        raw_spa = parsed.get("spatial", [])
        if isinstance(raw_spa, str):
            raw_spa = [raw_spa]
        valid_spa = [
            str(s).strip().lower()
            for s in raw_spa
            if str(s).strip().lower() in SPATIAL_RELATIONSHIPS
        ]
        if valid_spa:
            result["spatial"] = valid_spa

        return result

    # ------------------------------------------------------------------
    # Per-video processing (standard)
    # ------------------------------------------------------------------

    def process_video(self, video_id: str):
        """Standard: per-frame clip loading (centered), per-query zero-shot."""
        save_dir = self.output_dir / self.args.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{video_id}.pkl"

        if save_path.exists():
            logger.info(f"Skipping {video_id}: output already exists at {save_path}")
            return

        # 1. Load GT annotations
        vid_key = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
        try:
            video_data = self.ag_data.get_final_data_lite(vid_key)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {video_id}: {e}")
            return

        # 2. Get observed objects per frame
        observed_map = self.get_observed_objects_per_frame(video_data)
        if not observed_map:
            logger.info(f"Skipping {video_id}: no observed objects in any frame")
            return

        video_objects = self.get_video_objects(video_data)
        logger.info(
            f"[{video_id}] video-level objects={len(video_objects)}, "
            f"frames_with_objects={len(observed_map)}"
        )

        # 3. Build frame map & load a fallback representative video clip
        frame_map = self._build_frame_map(video_id)
        all_frames = self.get_all_frames(video_id)
        if not all_frames or not frame_map:
            logger.warning(f"Skipping {video_id}: no frames on disk")
            return

        # Fallback clip for when per-frame clip fails
        all_paths = [str(self.frames_dir / video_id / frame_map[idx]) for idx in sorted(frame_map.keys())]
        sampled_paths = all_paths[::4] if len(all_paths) > 120 else all_paths
        video_inputs = self.load_video_clip(sampled_paths)
        if video_inputs is None or video_inputs[0] is None:
            logger.warning(f"Skipping {video_id}: failed to load fallback video clip")
            return

        # 4. Iterate over frames with observed objects
        frame_results: Dict[str, Any] = {}

        for frame_stem, observed_objs in tqdm(
            observed_map.items(),
            desc=f"Frames [{video_id}]",
            leave=False,
        ):
            queries = self.generate_relationship_queries(observed_objs)

            # Resolve frame index
            m = re.search(r"(\d+)", frame_stem)
            frame_idx = int(m.group(1)) if m else 0

            # Load a small clip centered on this frame
            clip_inputs = self._load_clip_around_frame(frame_idx, frame_map, video_id, window=15)
            active_inputs = clip_inputs if (clip_inputs is not None and clip_inputs[0] is not None) else video_inputs
            clip_tensor = active_inputs[0]

            query_results = []
            for q in queries:
                raw_response = self._answer_query_zero_shot(q["prompt"], clip_tensor, max_new_tokens=128)
                parsed = self.parse_relationship_response(raw_response)
                query_results.append({
                    "object_name": q["object_name"],
                    "raw_response": raw_response,
                    "attention": parsed["attention"],
                    "contacting": parsed["contacting"],
                    "spatial": parsed["spatial"],
                })

            frame_results[frame_stem] = {
                "observed_objects": sorted(observed_objs),
                "predictions": query_results,
            }

        # 5. Save results
        output_record = {
            "video_id": video_id,
            "video_objects": sorted(video_objects),
            "num_frames_processed": len(frame_results),
            "frames": frame_results,
        }
        with open(save_path, "wb") as f:
            pickle.dump(output_record, f)
        logger.info(f"Saved {len(frame_results)} frame results for {video_id} → {save_path}")

    # ------------------------------------------------------------------
    # Per-video processing (fast)
    # ------------------------------------------------------------------

    def process_video_fast(self, video_id: str):
        """
        Fast: load one sampled video tensor once; slice centered chunks per frame;
        batch all prompts.
        """
        save_dir = self.output_dir / self.args.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{video_id}.pkl"

        if save_path.exists():
            logger.info(f"Skipping {video_id}: output already exists at {save_path}")
            return

        # 1. Load GT annotations
        vid_key = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
        try:
            video_data = self.ag_data.get_final_data_lite(vid_key)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {video_id}: {e}")
            return

        # 2. Get observed objects per frame
        observed_map = self.get_observed_objects_per_frame(video_data)
        if not observed_map:
            logger.info(f"Skipping {video_id}: no observed objects")
            return

        video_objects = self.get_video_objects(video_data)

        # 3. Load video tensor ONCE (sampled)
        frame_map = self._build_frame_map(video_id)
        all_frames = self.get_all_frames(video_id)
        if not all_frames or not frame_map:
            logger.warning(f"Skipping {video_id}: no frames on disk")
            return

        sorted_indices = sorted(frame_map.keys())
        step = 4 if len(sorted_indices) > 120 else 1
        sampled_indices = [sorted_indices[i] for i in range(0, len(sorted_indices), step)]

        sampled_paths = [str(self.frames_dir / video_id / frame_map[idx]) for idx in sampled_indices]
        logger.info(f"[{video_id}][FAST] Loading video tensor ({len(sampled_paths)} frames) …")

        video_inputs = self.load_video_clip(sampled_paths)
        if video_inputs is None or video_inputs[0] is None:
            logger.warning(f"Skipping {video_id}: failed to load video clip")
            return

        video_tensor = video_inputs[0]
        logger.info(f"[{video_id}][FAST] Video tensor loaded: shape={tuple(video_tensor.shape)}")

        # Helper: find nearest sampled position for an original frame index
        def nearest_sample_pos(frame_idx: int) -> int:
            import bisect
            j = bisect.bisect_left(sampled_indices, frame_idx)
            if j <= 0:
                return 0
            if j >= len(sampled_indices):
                return len(sampled_indices) - 1
            before = sampled_indices[j - 1]
            after = sampled_indices[j]
            return (j - 1) if (frame_idx - before) <= (after - frame_idx) else j

        # 4. Collect all (frame_stem, object_name, prompt, clip_tensor)
        all_entries: List[Tuple[str, str, str, torch.Tensor]] = []
        frame_query_counts: Dict[str, Tuple[Set[str], int]] = {}

        for frame_stem, observed_objs in observed_map.items():
            qs = self.generate_relationship_queries(observed_objs)
            m = re.search(r"(\d+)", frame_stem)
            frame_idx = int(m.group(1)) if m else 0

            # slice a centered chunk from the global sampled tensor
            center_pos = nearest_sample_pos(frame_idx)
            clip_tensor = self._slice_centered_chunk(video_tensor, center_pos, max_frames=int(self.args.chunk_size))

            for q in qs:
                all_entries.append((frame_stem, q["object_name"], q["prompt"], clip_tensor))

            frame_query_counts[frame_stem] = (observed_objs, len(qs))

        logger.info(f"[{video_id}][FAST] {len(observed_map)} frames, {len(all_entries)} total queries — batching …")

        # 5. Batched zero-shot inference
        prompts_and_clips = [(prompt, clip) for (_, _, prompt, clip) in all_entries]
        raw_responses = self._batch_answer_zero_shot(prompts_and_clips, max_new_tokens=128)

        # 6. Redistribute back to per-frame structure
        frame_results: Dict[str, Any] = {}
        idx = 0
        for frame_stem in observed_map:
            observed_objs, qcount = frame_query_counts[frame_stem]
            query_results = []
            for _ in range(qcount):
                _, object_name, _, _ = all_entries[idx]
                raw_resp = raw_responses[idx] if idx < len(raw_responses) else None
                parsed = self.parse_relationship_response(raw_resp)
                query_results.append({
                    "object_name": object_name,
                    "raw_response": raw_resp,
                    "attention": parsed["attention"],
                    "contacting": parsed["contacting"],
                    "spatial": parsed["spatial"],
                })
                idx += 1

            frame_results[frame_stem] = {
                "observed_objects": sorted(observed_objs),
                "predictions": query_results,
            }

        # 7. Save results
        output_record = {
            "video_id": video_id,
            "video_objects": sorted(video_objects),
            "num_frames_processed": len(frame_results),
            "frames": frame_results,
        }
        with open(save_path, "wb") as f:
            pickle.dump(output_record, f)
        logger.info(f"Saved {len(frame_results)} frame results for {video_id} → {save_path}")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self, limit: Optional[int] = None):
        """Process all videos (optionally filtered by split and limit)."""
        if not self.frames_annotated_dir.exists():
            logger.error(f"Annotated frames dir not found: {self.frames_annotated_dir}")
            return

        video_ids = sorted(
            d for d in os.listdir(self.frames_annotated_dir)
            if (self.frames_annotated_dir / d).is_dir()
        )

        # Split filtering
        split = self.args.split
        if split in ("test", "train"):
            split_ids = load_split_video_ids(split)
            video_ids = [v for v in video_ids if Path(v).stem in split_ids]
            logger.info(f"Filtered to {len(video_ids)} videos for '{split}' split (video_splits.json)")
        elif split:
            video_ids = [v for v in video_ids if get_video_belongs_to_split(v) == split]
            logger.info(f"Filtered to {len(video_ids)} videos for '{split}' split (first-letter logic)")

        if limit:
            video_ids = video_ids[:limit]

        mode_label = "FAST" if self.fast_mode else "STANDARD"
        lora_label = f" [LoRA: {self.lora_path}]" if self.lora_path else ""
        logger.info(f"Processing {len(video_ids)} videos [{mode_label}]{lora_label} (scene graph zero-shot) …")

        process_fn = self.process_video_fast if self.fast_mode else self.process_video

        for i, video_id in enumerate(tqdm(video_ids, desc="Videos")):
            logger.info(f"[{i + 1}/{len(video_ids)}] {video_id}")
            try:
                process_fn(video_id)
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot scene graph generation for Action Genome videos. "
            "Predicts relationships for observed objects using a VLM (via Vgent)."
        ),
    )
    parser.add_argument(
        "--ag_root_directory", type=str,
        default="/data/rohith/ag",
        help="Root directory of the Action Genome dataset",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/data/rohith/ag/sg_results/",
        help="Output directory for scene graph predictions",
    )
    parser.add_argument(
        "--model_name", type=str, default="kimikvl",
        help="Name of the VLM model to use (kimikvl | internvl | qwen3vl)",
    )
    parser.add_argument(
        "--split", type=str, default="04",
        help=(
            "Process only videos in this split. "
            "'test'/'train' use video_splits.json; "
            "otherwise first-letter buckets: 04, 59, AD, EH, IL, MP, QT, UZ"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many videos (for dev/debug)",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--use_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM for inference (default: True). --no-use_vllm for HF.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help=(
            "Enable fast processing mode: load video tensor once, slice per-frame chunks, "
            "and batch all zero-shot prompts. (Default: enabled)"
        ),
    )
    parser.add_argument(
        "--no-fast",
        action="store_false",
        dest="fast",
        help="Disable fast mode (use standard per-frame processing).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for --fast zero-shot mllm_batch_response calls.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help=(
            "Path to LoRA adapter weights. When provided, forces --no-use_vllm "
            "and loads the fine-tuned adapter on top of the base model."
        ),
    )

    args = parser.parse_args()
    setup_logging(args.output_dir, f"ag_sg_zeroshot_{args.model_name}.log")

    # LoRA forces non-vLLM mode
    use_vllm = args.use_vllm
    if args.lora_path:
        use_vllm = False
        logger.info("LoRA path provided → forcing --no-use_vllm")

    processor = ActionGenomeSceneGraphProcessor(
        ag_root_directory=args.ag_root_directory,
        output_dir=args.output_dir,
        model_name=args.model_name,
        split=args.split,
        tensor_parallel_size=args.tensor_parallel_size,
        use_vllm=use_vllm,
        batch_size=args.batch_size,
        lora_path=args.lora_path,
    )
    processor.fast_mode = args.fast
    processor.run(limit=args.limit)


if __name__ == "__main__":
    main()
