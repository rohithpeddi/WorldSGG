
import os
import sys
import json
import pickle
import argparse
import random
import re
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import logging
from tqdm import tqdm
from lib.mllm.core.logger_utils import setup_logging
from lib.mllm.core.config_loader import load_config, get_path, get_inference_defaults

logger = logging.getLogger(__name__)


from lib.mllm.core.vgent import Vgent
from lib.mllm.core.prompts import CAPTION_GENERATION_PROMPT, GRAPH_PROMPT, strip_thinking_tags
from lib.mllm.models.utils import fetch_video, resize_video


# ---------------------------
# Split logic
# ---------------------------

# Path to the JSON file that maps split names ("train", "test") to video filenames
_SPLIT_JSON_PATH = Path(__file__).resolve().parents[2] / "video_splits.json"


def load_split_video_ids(split_name: str) -> set:
    """
    Load video IDs belonging to *split_name* (e.g. "test", "train")
    from video_splits.json.  Returns a set of bare IDs (stems, without .mp4).
    """
    with open(_SPLIT_JSON_PATH, "r") as f:
        splits = json.load(f)
    if split_name not in splits:
        raise ValueError(
            f"Split '{split_name}' not found in {_SPLIT_JSON_PATH}. "
            f"Available splits: {list(splits.keys())}"
        )
    return {Path(v).stem for v in splits[split_name]}


def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    """
    Get the split that the video belongs to based on its ID.
    Accepts either a bare ID (e.g., '0DJ6R') or a filename (e.g., '0DJ6R.mp4').
    """
    stem = Path(video_id).stem
    if not stem:
        return None
    first_letter = stem[0]
    if first_letter.isdigit() and int(first_letter) < 5:
        return "04"
    elif first_letter.isdigit() and int(first_letter) >= 5:
        return "59"
    elif first_letter in "ABCD":
        return "AD"
    elif first_letter in "EFGH":
        return "EH"
    elif first_letter in "IJKL":
        return "IL"
    elif first_letter in "MNOP":
        return "MP"
    elif first_letter in "QRST":
        return "QT"
    elif first_letter in "UVWXYZ":
        return "UZ"
    return None

class ActionGenomeProcessor:
    def __init__(self, data_dir, output_dir, model_name, split, frame_bbox_mode=False, tensor_parallel_size=1, use_vllm=True, model_weights_dir=None):
        self.data_dir = Path(data_dir)
        self.frames_dir = self.data_dir / "frames"
        self.frames_annotated_dir = self.data_dir / "frames_annotated"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Vgent arguments mock
        from lib.mllm.core.config_loader import get_vllm_engine_settings
        vllm_cfg = get_vllm_engine_settings(load_config())

        class Args:
            def __init__(self):
                self.model_name = model_name
                self.vision_encoder_path = "google/siglip-so400m-patch14-384"
                self.vision_feature_layer = -1
                self.vision_feature_select_layer = -1
                self.use_flash_attn = False
                self.fps = 1  
                self.chunk_size = 128
                self.total_pixels = 128000
                self.frame_bbox_mode = frame_bbox_mode
                self.split = split
                self.tensor_parallel_size = tensor_parallel_size
                self.use_vllm = use_vllm
                self.model_weights_dir = model_weights_dir
                # vLLM engine settings (from config vllm: section)
                self.gpu_memory_utilization = vllm_cfg.get("gpu_memory_utilization", 0.90)
                self.max_model_len = vllm_cfg.get("max_model_len", 32768)
                self.max_num_seqs = vllm_cfg.get("max_num_seqs", 32)
                self.enable_chunked_prefill = vllm_cfg.get("enable_chunked_prefill", True)
                self.dtype = vllm_cfg.get("dtype", "bfloat16")

        self.args = Args()
        self.vgent = Vgent(self.args)
        self.fast_mode = False  # toggled by --fast CLI flag
        self._failed_log_lock = threading.Lock()

        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Failure logging
    # ------------------------------------------------------------------

    def _log_failed_video(self, video_id: str, reason: str) -> None:
        """Append a failure record to ``failed_videos.txt`` inside the
        model-specific output directory.  Thread-safe."""
        log_path = self.output_dir / self.args.model_name / "failed_videos.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = f"{video_id}\t{reason}\n"
        with self._failed_log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        logger.warning(f"Logged failure: {video_id} — {reason}")

    def get_annotated_frames(self, video_id):
        video_annotated_dir = self.frames_annotated_dir / video_id
        if not video_annotated_dir.exists():
            return []
        
        frames = []
        for filename in os.listdir(video_annotated_dir):
            if filename.endswith(".png"):
                match = re.search(r"(\d+)", filename)
                if match:
                    frames.append(int(match.group(1)))
        
        return sorted(frames)

    def get_all_frames(self, video_id):
        video_frames_dir = self.frames_dir / video_id
        if not video_frames_dir.exists():
            return []
        
        frames = []
        for filename in os.listdir(video_frames_dir):
             if filename.endswith(".png") or filename.endswith(".jpg"):
                 match = re.search(r"(\d+)", filename)
                 if match:
                     frames.append(int(match.group(1)))
        
        return sorted(frames)
    
    def get_clip_intervals(self, annotated_frames, total_frames_range, video_id=""):
        if not annotated_frames:
            return []
            
        intervals = []
        midpoints = []
        
        for i in tqdm(range(len(annotated_frames) - 1), desc=f"[{video_id}] Processing frames", leave=False):
            mid = (annotated_frames[i] + annotated_frames[i+1]) // 2
            midpoints.append(mid)
            
        video_start = total_frames_range[0]
        video_end = total_frames_range[-1]
        
        start = video_start
        for i, frame in enumerate(tqdm(annotated_frames, desc=f"[{video_id}] Processing clips", leave=False)):
            if i < len(midpoints):
                end = midpoints[i]
            else:
                end = video_end
            
            intervals.append({
                "annotated_frame": frame,
                "start_frame": start,
                "end_frame": end
            })
            start = end + 1
            
        if intervals:
            # 1. Limit the FIRST clip start
            first_clip = intervals[0]
            annotated_frame = first_clip["annotated_frame"]
            # Start at most 30 frames before annotated frame, but not before video start
            new_start = max(first_clip["start_frame"], annotated_frame - 30)
            first_clip["start_frame"] = new_start
            
            # 2. Limit the LAST clip end
            last_clip = intervals[-1]
            annotated_frame = last_clip["annotated_frame"]
            # End at most 30 frames after annotated frame, but not after video end
            new_end = min(last_clip["end_frame"], annotated_frame + 30)
            last_clip["end_frame"] = new_end
            
        return intervals

    def load_video_clip(self, image_paths):
        if not image_paths:
            logger.warning("Empty image_paths provided to load_video_clip")
            return None

        # Fetch images (resized to factor)
        # fetch_video returns list[Image.Image] for list input
        input_data = {
            "video": image_paths, 
            "min_pixels": 28*28, 
            "total_pixels": self.args.total_pixels
        }
        try:
            images = fetch_video(input_data, resize=False) # resize arg doesn't affect list input logic in fetch_video
        except Exception as e:
            logger.error(f"Error fetching video images: {e}")
            return None
        
        if not images:
            logger.warning("No images returned from fetch_video")
            return None

        # Convert to tensor (T, C, H, W) in range [0, 255] float
        try:
            tensor = torch.stack([transforms.PILToTensor()(img) for img in images]).float()
        except Exception as e:
            logger.error(f"Error stacking images to tensor: {e}")
            return None
        
        # Resize video to meet total_pixels constraint and handle fps
        video_tensor, _ = resize_video(tensor, self.args.fps, total_pixels=self.args.total_pixels)
        
        return [video_tensor]

    def process_video(self, video_id):
        save_path_dir = self.output_dir / self.args.model_name
        save_path = save_path_dir / f"{video_id}.pkl"
        if save_path.exists():
            logger.info(f"Skipping {video_id} as output already exists at {save_path}")
            return

        annotated_frames = self.get_annotated_frames(video_id)
        if not annotated_frames:
            # print(f"No annotations for {video_id}")
            return

        all_frames = self.get_all_frames(video_id)
        if not all_frames:
             logger.warning(f"No frames found for {video_id} in frames dir")
             return

        frame_map = {}
        video_frames_dir = self.frames_dir / video_id
        for filename in os.listdir(video_frames_dir):
             if filename.endswith(".png") or filename.endswith(".jpg"):
                 match = re.search(r"(\d+)", filename)
                 if match:
                     frame_map[int(match.group(1))] = filename

        intervals = self.get_clip_intervals(annotated_frames, all_frames, video_id=video_id)
        
        results = []
        
        for clip in intervals:
            start = clip["start_frame"]
            end = clip["end_frame"]
            
            clip_images_paths = []
            # Use step=2 for alternate frames
            for f_idx in range(start, end + 1, 2):
                if f_idx in frame_map:
                    clip_images_paths.append(str(video_frames_dir / frame_map[f_idx]))
            
            if not clip_images_paths:
                continue

            try:
                # Load video clip as tensor
                video_inputs_list = self.load_video_clip(clip_images_paths)
                if video_inputs_list is None:
                     logger.warning(f"Failed to load video clip frames {start}-{end} for {video_id}, skipping.")
                     continue
                
                # 1. Generate Caption
                caption_raw = self.vgent.model.mllm_response(
                    CAPTION_GENERATION_PROMPT, 
                    video_inputs_list[0],  # Pass the tensor
                    max_new_tokens=100
                )
                caption = strip_thinking_tags(caption_raw)
                
                # 2. Construct Graph
                clip_captions = [(0, caption)]
                # construct_graph expects [video_tensor]
                final_graph, _ = self.vgent.construct_graph(video_inputs_list, clip_captions)
                
                results.append({
                    "clip_metadata": clip,
                    "caption": caption,
                    "graph": final_graph
                })
                
            except Exception as e:
                logger.error(f"Error processing clip {start}-{end} for video {video_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self._log_failed_video(
                    video_id,
                    f"clip {start}-{end} error: {e}",
                )

        if results:
            save_path_dir = self.output_dir / self.args.model_name
            os.makedirs(save_path_dir, exist_ok=True)
            save_path = save_path_dir / f"{video_id}.pkl"

            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Saved {len(results)} clips to {save_path}")

    # ------------------------------------------------------------------
    # Fast path: batched LLM calls
    # ------------------------------------------------------------------

    def process_video_fast(self, video_id):
        """Fast-path: same output as ``process_video`` but batches all
        caption-generation and entity-extraction LLM calls into two
        large ``mllm_batch_response`` calls per video."""
        save_path_dir = self.output_dir / self.args.model_name
        save_path = save_path_dir / f"{video_id}.pkl"
        if save_path.exists():
            logger.info(f"Skipping {video_id} as output already exists at {save_path}")
            return

        annotated_frames = self.get_annotated_frames(video_id)
        if not annotated_frames:
            return

        all_frames = self.get_all_frames(video_id)
        if not all_frames:
            logger.warning(f"No frames found for {video_id} in frames dir")
            return

        frame_map = {}
        video_frames_dir = self.frames_dir / video_id
        for filename in os.listdir(video_frames_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                match = re.search(r"(\d+)", filename)
                if match:
                    frame_map[int(match.group(1))] = filename

        intervals = self.get_clip_intervals(annotated_frames, all_frames, video_id=video_id)

        # ------ Phase 1: Pre-load ALL clip tensors -----------------------
        logger.info(f"[{video_id}][FAST] Loading {len(intervals)} clip tensors …")
        clip_data: List[Tuple[Dict, Any]] = []  # (clip_meta, video_inputs)
        for clip in intervals:
            start, end = clip["start_frame"], clip["end_frame"]
            clip_paths = [
                str(video_frames_dir / frame_map[f_idx])
                for f_idx in range(start, end + 1, 2)
                if f_idx in frame_map
            ]
            if not clip_paths:
                continue
            video_inputs = self.load_video_clip(clip_paths)
            if video_inputs is None:
                logger.warning(
                    f"Failed to load clip {start}-{end} for {video_id}, skipping."
                )
                continue
            clip_data.append((clip, video_inputs))

        if not clip_data:
            logger.warning(f"No valid clips for {video_id}")
            return

        n = len(clip_data)
        logger.info(f"[{video_id}][FAST] {n} clips loaded.")

        # ------ Phase 2: ONE batched caption generation -----------------
        logger.info(f"[{video_id}][FAST] [Step 1/2] Batched caption generation ({n} clips) …")
        caption_prompts = [
            {
                "text": CAPTION_GENERATION_PROMPT,
                "video_inputs": [vi[0]],  # tensor
                "max_new_tokens": 100,
            }
            for _, vi in clip_data
        ]
        captions = self._batched_with_bisect(
            caption_prompts, label=f"[{video_id}][FAST] captions"
        )
        logger.info(f"[{video_id}][FAST] [Step 1/2] Captions done.")

        # Log clips with failed caption generation
        failed_sub_indices = [i for i, s in enumerate(captions) if s is None]
        if failed_sub_indices:
            self._log_failed_video(
                video_id,
                f"caption generation returned None for {len(failed_sub_indices)}/{n} clips "
                f"(indices: {failed_sub_indices})",
            )

        # ------ Phase 3: ONE batched entity extraction -------------------
        # For each clip, construct_graph splits into chunks. Most clips
        # are <=128 frames (chunk_size), so 1 chunk per clip.  We prepare
        # the entity-extraction prompts for each chunk across all clips.
        logger.info(f"[{video_id}][FAST] [Step 2/2] Batched entity extraction …")

        entity_prompts: List[Dict[str, Any]] = []
        entity_mapping: List[Tuple[int, int]] = []  # (clip_idx, chunk_idx)
        chunk_metadata: Dict[int, List[List[int]]] = {}  # clip_idx -> list of sampled_indices per chunk

        for clip_idx, (_, video_inputs) in enumerate(clip_data):
            split_tensors = torch.split(video_inputs[0], self.args.chunk_size, dim=0)
            sampled_per_chunk = []

            # Include caption context to ground entity extraction
            clip_caption = captions[clip_idx] if captions[clip_idx] else ""
            if clip_caption:
                grounded_prompt = (
                    f"Context: The following caption describes this video clip: "
                    f"\"{clip_caption}\"\n\n{GRAPH_PROMPT}"
                )
            else:
                grounded_prompt = GRAPH_PROMPT

            for chunk_idx, chunk_tensor in enumerate(split_tensors):
                # Same downsampling as construct_graph
                if chunk_tensor.shape[0] > 30:
                    indices = torch.linspace(0, chunk_tensor.shape[0] - 1, 30).long()
                    chunk_tensor = chunk_tensor[indices]
                    sampled = indices.tolist()
                else:
                    sampled = list(range(chunk_tensor.shape[0]))
                sampled_per_chunk.append(sampled)

                entity_prompts.append({
                    "text": grounded_prompt,
                    "video_inputs": [chunk_tensor],
                    "max_new_tokens": 512,
                })
                entity_mapping.append((clip_idx, chunk_idx))
            chunk_metadata[clip_idx] = sampled_per_chunk

        logger.info(
            f"[{video_id}][FAST]   {len(entity_prompts)} entity prompts "
            f"across {n} clips …"
        )

        entity_responses = self._batched_with_bisect(
            entity_prompts, label=f"[{video_id}][FAST] entities"
        )
        logger.info(f"[{video_id}][FAST] [Step 2/2] Entity extraction done.")

        # ------ Phase 4: Parse responses & retry failures ----------------
        # Group entity responses by clip
        clip_chunks: Dict[int, Dict[int, Any]] = {i: {} for i in range(n)}
        failed: List[int] = []  # indices into entity_prompts for retry

        for prompt_idx, ((clip_idx, chunk_idx), resp) in enumerate(
            zip(entity_mapping, entity_responses)
        ):
            parsed = self._try_parse_entities(resp)
            if parsed is not None:
                clip_chunks[clip_idx][chunk_idx] = parsed
            else:
                failed.append(prompt_idx)

        # Retry failed parses individually (up to 4 more attempts each)
        if failed:
            logger.info(
                f"[{video_id}][FAST] Retrying {len(failed)} failed entity parses …"
            )
            for fidx in failed:
                clip_idx, chunk_idx = entity_mapping[fidx]
                prompt = entity_prompts[fidx]
                for attempt in range(4):
                    try:
                        resp = self.vgent.model.mllm_response(
                            prompt["text"],
                            prompt["video_inputs"],
                            max_new_tokens=512,
                        )
                        parsed = self._try_parse_entities(resp)
                        if parsed is not None:
                            clip_chunks[clip_idx][chunk_idx] = parsed
                            break
                    except Exception:
                        pass

        # Log clips whose entity chunks are still missing after retries
        still_failed = [
            fidx for fidx in failed
            if entity_mapping[fidx][1] not in clip_chunks[entity_mapping[fidx][0]]
        ]
        if still_failed:
            self._log_failed_video(
                video_id,
                f"entity extraction failed for {len(still_failed)} chunks after retries "
                f"(prompt indices: {still_failed})",
            )
        # ------ Phase 5: Build graphs + assemble results -----------------
        logger.info(f"[{video_id}][FAST] Building graphs …")
        import networkx as nx
        from collections import defaultdict as dd

        results = []
        for clip_idx, (clip_meta, video_inputs) in enumerate(clip_data):
            caption = captions[clip_idx] if captions[clip_idx] else ""

            video_graph = nx.DiGraph()
            entity_graph = dd(set)

            n_chunks = len(chunk_metadata[clip_idx])
            for chunk_idx in range(n_chunks):
                sampled_indices = chunk_metadata[clip_idx][chunk_idx]
                parsed = clip_chunks[clip_idx].get(chunk_idx)
                if parsed is not None:
                    entities, actions, scenes = parsed
                else:
                    entities, actions, scenes = [], [], []

                current_captions = [(0, caption)] if chunk_idx == 0 else []
                video_graph.add_node(
                    chunk_idx,
                    actions=actions,
                    scenes=scenes,
                    entities=entities,
                    captions=[s[1] for s in current_captions] if current_captions else None,
                    sampled_indices=sampled_indices,
                )
                self.vgent._update_entity_graph(
                    entity_graph, video_graph, chunk_idx, entities, actions, scenes,
                )

            results.append({
                "clip_metadata": clip_meta,
                "caption": caption,
                "graph": video_graph,
            })

        # ------ Phase 6: Save --------------------------------------------
        if results:
            os.makedirs(save_path_dir, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Saved {len(results)} clips to {save_path}")

    def _batched_with_bisect(
        self,
        prompts: List[Dict[str, Any]],
        label: str = "batch",
    ) -> List[Optional[str]]:
        """Try ``mllm_batch_response``; on failure bisect and retry.

        If the full batch exceeds the model's context window, this
        recursively splits the prompt list in half until it succeeds or
        falls back to individual ``mllm_response`` calls.  Only truly
        oversized individual prompts will produce ``None``.
        """
        n = len(prompts)
        if n == 0:
            return []

        # --- single prompt: fall back to individual call ---
        #     If the clip is still too large, progressively subsample
        #     frames (halve each attempt) up to MAX_REDUCTIONS times.
        if n == 1:
            MAX_REDUCTIONS = 3
            p = prompts[0]
            video_input = p["video_inputs"]
            for attempt in range(MAX_REDUCTIONS + 1):
                try:
                    resp = self.vgent.model.mllm_response(
                        p["text"],
                        video_input,
                        max_new_tokens=p.get("max_new_tokens", 100),
                    )
                    return [strip_thinking_tags(resp) if resp else None]
                except Exception as e:
                    err_msg = str(e)
                    is_too_long = (
                        "longer than the maximum model length" in err_msg
                        or "max_model_len" in err_msg
                    )
                    # vLLM cache corruption: after a failed generate() the
                    # mm_receiver_cache may drop entries; retrying with a
                    # modified tensor triggers
                    # "AssertionError: Expected a cached item for mm_hash=…"
                    # This is unrecoverable within the same engine session.
                    is_cache_error = (
                        "Expected a cached item for mm_hash" in err_msg
                    )
                    if is_cache_error:
                        logger.warning(
                            f"{label}: vLLM multimodal cache error on "
                            f"attempt {attempt + 1}, skipping prompt: {e}"
                        )
                        return [None]
                    if not is_too_long or attempt == MAX_REDUCTIONS:
                        logger.error(
                            f"{label}: single-prompt fallback failed "
                            f"(attempt {attempt + 1}): {e}"
                        )
                        return [None]
                    # Halve the frames and retry — use .clone() so vLLM
                    # sees a brand-new tensor (not a view of the old one)
                    # and hashes it independently.
                    tensor = (
                        video_input[0]
                        if isinstance(video_input, list)
                        else video_input
                    )
                    T = tensor.shape[0]
                    indices = torch.linspace(0, T - 1, max(1, T // 2)).long()
                    reduced = tensor[indices].clone()
                    video_input = [reduced] if isinstance(p["video_inputs"], list) else reduced
                    logger.warning(
                        f"{label}: prompt too long ({T} frames), "
                        f"subsampling to {reduced.shape[0]} frames "
                        f"(attempt {attempt + 2}/{MAX_REDUCTIONS + 1}) …"
                    )

        # --- try the full batch first ---
        try:
            raw = self.vgent.model.mllm_batch_response(prompts)
            return [
                strip_thinking_tags(s) if s else None for s in raw
            ]
        except Exception as e:
            err_msg = str(e)
            is_cache_error = (
                "Expected a cached item for mm_hash" in err_msg
            )
            if is_cache_error:
                # Cache corruption in batch mode — fall back to sequential
                logger.warning(
                    f"{label}: vLLM cache error in batch of {n}, "
                    f"falling back to sequential: {e}"
                )
                results = []
                for p in prompts:
                    results.extend(
                        self._batched_with_bisect([p], label=label)
                    )
                return results
            mid = n // 2
            logger.warning(
                f"{label}: batch of {n} failed ({e}), "
                f"bisecting into {mid} + {n - mid} …"
            )
            left = self._batched_with_bisect(prompts[:mid], label=label)
            right = self._batched_with_bisect(prompts[mid:], label=label)
            return left + right

    @staticmethod
    def _flatten(lst):
        """Recursively flatten nested lists into a single list of non-list items."""
        out = []
        for item in lst:
            if isinstance(item, list):
                out.extend(ActionGenomeProcessor._flatten(item))
            else:
                out.append(item)
        return out

    @staticmethod
    def _try_parse_entities(response):
        """Try to parse a GRAPH_PROMPT response into (entities, actions, scenes).
        Returns None on failure."""
        if not response:
            return None
        response = strip_thinking_tags(response)
        try:
            info = json.loads(
                response.replace("```json", "").replace("```", "").strip()
            )

            # --- Normalise varied Kimi output shapes to a single dict ---
            # Case 1: top-level list
            if isinstance(info, list):
                # Unwrap single-element wrapper: [{"entities": [...], ...}]
                if len(info) == 1 and isinstance(info[0], dict):
                    info = info[0]
                # Merge list of dicts that each carry their own keys
                elif all(isinstance(item, dict) for item in info):
                    merged: Dict[str, list] = {}
                    for item in info:
                        for k, v in item.items():
                            merged.setdefault(k, [])
                            if isinstance(v, list):
                                merged[k].extend(v)
                            else:
                                merged[k].append(v)
                    info = merged
                else:
                    # Plain list of entity dicts (no wrapper keys)
                    info = {"entities": info}

            if not isinstance(info, dict):
                return None

            # Flatten any accidentally nested lists inside each key
            raw_entities = ActionGenomeProcessor._flatten(info.get("entities", []))
            raw_actions = ActionGenomeProcessor._flatten(info.get("actions", []))
            raw_scenes = ActionGenomeProcessor._flatten(info.get("scenes", []))

            entities = [
                f"{e['entity name']}, {e['description']}"
                for e in raw_entities
                if isinstance(e, dict) and "entity name" in e and "description" in e
            ]
            actions = [
                f"{e['entity name']}, {e['action description']}"
                for e in raw_actions
                if isinstance(e, dict) and "entity name" in e and "action description" in e
            ]
            scenes = [
                s["location"] for s in raw_scenes if isinstance(s, dict) and "location" in s
            ]
            return entities, actions, scenes
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return None

    def run(self, limit=None, video_list=None):
        video_ids = sorted(d for d in os.listdir(self.frames_annotated_dir)
                           if (self.frames_annotated_dir / d).is_dir())
        if video_list:
            with open(video_list, "r", encoding="utf-8") as f:
                keep = {Path(l.strip()).stem for l in f if l.strip()}
            video_ids = [v for v in video_ids if Path(v).stem in keep]
            logger.info(f"Filtered to {len(video_ids)} videos from list {video_list}")
        
        # --- split filtering (skipped when an explicit video list is given) ----
        split = None if video_list else self.args.split
        if split in ("test", "train"):
            # Use the JSON-defined split list
            split_ids = list(load_split_video_ids(split))
            video_ids = [v for v in video_ids if Path(v).stem in split_ids]
            logger.info(f"Filtered to {len(video_ids)} videos for '{split}' split (from video_splits.json)")
        elif split:
            # Fallback: use the first-letter bucketing logic
            video_ids = [v for v in video_ids if get_video_belongs_to_split(v) == split]
            logger.info(f"Filtered to {len(video_ids)} videos for '{split}' split (first-letter logic)")
        # -----------------------------------------------------------------------

        if limit:
            video_ids = video_ids[:limit]

        # Shuffle so multiple GPUs process different videos concurrently
        random.shuffle(video_ids)

        mode_label = "FAST" if self.fast_mode else "STANDARD"
        process_fn = self.process_video_fast if self.fast_mode else self.process_video

        logger.info(f"Processing {len(video_ids)} videos [{mode_label}] …")
        pbar = tqdm(video_ids, desc="Processing videos")
        for i, video_id in enumerate(pbar):
            pbar.set_description(f"Processing videos [{video_id}]")
            logger.info(f"[{i+1}/{len(video_ids)}] Processing {video_id}")
            try:
                process_fn(video_id)
            except Exception as e:
                logger.error(f"[{video_id}] Error processing video: {e}")
                import traceback
                logger.error(traceback.format_exc())

def main():
    cfg = load_config()
    inf = get_inference_defaults(cfg)
    parser = argparse.ArgumentParser(description="Process Action Genome videos to generate graphs and captions.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: auto-discovered)")
    parser.add_argument("--data_dir", type=str, default=get_path(cfg, "ag_root"), help="Path to Action Genome dataset")
    parser.add_argument("--output_dir", type=str, default=get_path(cfg, "outputs.graphs"), help="Output directory for graphs")
    parser.add_argument("--model_name", type=str, default=inf.get("default_model", "qwen25vl_7b"), help="Path or name of the video model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument("--tensor_parallel_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--split", type=str, default="04", help="Process only videos belonging to a specific split. Use 'test' or 'train' to filter by video_splits.json, or a first-letter bucket like '04', '59', 'AD', 'EH', 'IL', 'MP', 'QT', 'UZ'.")
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=inf.get("use_vllm", True), help="Use vLLM for inference (default: True). Pass --no-use_vllm to load model directly via HuggingFace Transformers.")
    parser.add_argument("--video_list", type=str, default=None,
                        help="Text file of video stems to process (e.g. the worldbbox test split)")
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Use batched LLM calls for faster processing")
    parser.add_argument("--model_weights_dir", type=str, default=get_path(cfg, "model_weights") or None,
                        help="Directory containing pre-downloaded model weights (from download_models.py)")
    
    args = parser.parse_args()
    setup_logging(args.output_dir, f"{args.model_name}.log")
    
    processor = ActionGenomeProcessor(args.data_dir, args.output_dir, args.model_name, args.split, tensor_parallel_size=args.tensor_parallel_size, use_vllm=args.use_vllm, model_weights_dir=args.model_weights_dir)
    processor.fast_mode = args.fast
    processor.run(limit=args.limit, video_list=args.video_list)

if __name__ == "__main__":
    main()
