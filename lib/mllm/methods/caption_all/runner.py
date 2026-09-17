#!/usr/bin/env python3
"""
process_ag_caption_all.py
==========================
Variant of ``process_ag_caption.py`` that queries *all* video-level
objects on every annotated frame (not just missing ones), using
captions + MLLM only (no RAG pipeline).

Usage
-----
::

    python process_ag_caption_all.py \
        --model_name kimikvl --split 04

Paths (`ag_root`, `outputs.caption_all`, `graphs`, etc.) are resolved
from `config_pragya.yaml` / `config_utd.yaml` via `load_config()`.

Output pkl format
-----------------
One pkl file per video, saved at
``<output_dir>/<mode>/<model_name>/<video_id>.pkl``.

Each file contains a single pickled ``dict`` with the following keys::

    {
        "video_id":             str,        # e.g. "00012"
        "mode":                 str,        # "predcls" or "sgdet"
        "model_name":           str,        # e.g. "kimikvl"
        "estimation_meta":      dict|None,  # sgdet estimation info (None for predcls)
        "video_objects":        list[str],  # sorted list of all object names used
        "num_frames_processed": int,        # number of annotated frames processed
        "frames": {                         # keyed by frame stem, e.g. "000012.png"
            "<frame_stem>": {
                "objects":     list[str],   # sorted list of object names for this frame
                "predictions": [            # one entry per object (same order as objects)
                    {
                        "object":     str,          # object name
                        "raw_response": str|None,   # raw LLM response text
                        "attention":  dict,         # {label: score} for attention
                        "contacting": dict,         # {label: score} for contacting
                        "spatial":    dict,          # {label: score} for spatial
                    },
                    ...
                ]
            },
            ...
        }
    }
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
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Allow imports from the parent package

from lib.mllm.core.vgent import Vgent
from lib.mllm.core.ag_data import AgDataBBAnnotations
from lib.mllm.core.logger_utils import setup_logging
from lib.mllm.core.config_loader import load_config, get_path, get_inference_defaults


from lib.mllm.base_processor import (
    ActionGenomeBaseProcessor,
    ATTENTION_RELATIONSHIPS,
    CONTACTING_RELATIONSHIPS,
    SPATIAL_RELATIONSHIPS,
    load_split_video_ids,
    get_video_belongs_to_split,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (all-objects variant -- uses "object_name" not
# "missing_object")
# ---------------------------------------------------------------------------

AG_RELATIONSHIP_QUERY_PROMPT = """\
You are analyzing a video of a scene. The first frame is the specific \
moment to analyze. The remaining frames provide surrounding context. \
The object "{object_name}" is one of the objects present in this video. \
It may or may not be visible in the target frame. A person IS visible \
in the target frame.

Based on the video context, predict the relationships between the person \
and the object "{object_name}" at the target frame.

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

# Pre-computed label strings (avoids re-joining constants per call)
_ATTENTION_LABELS_STR = ", ".join(ATTENTION_RELATIONSHIPS)
_CONTACTING_LABELS_STR = ", ".join(CONTACTING_RELATIONSHIPS)
_SPATIAL_LABELS_STR = ", ".join(SPATIAL_RELATIONSHIPS)


# ---------------------------------------------------------------------------
# Core processor class
# ---------------------------------------------------------------------------

class ActionGenomeCaptionAllObjectsProcessor(ActionGenomeBaseProcessor):
    """Predict relationships for ALL video-level objects on every frame
    using captions + MLLM only.

    No graph construction, no keyword extraction, no node
    retrieval/refinement.  Captions are loaded from the same pickle
    files that ``process_action_genome.py`` produces, but the graph
    portion is ignored.
    """

    def __init__(
        self,
        ag_root_directory: str,
        output_dir: str,
        graph_dir: str,
        model_name: str,
        split: str,
        tensor_parallel_size: int = 1,
        use_vllm: bool = True,
        mode: str = "predcls",
        model_weights_dir: str = None,
    ):
        super().__init__(
            ag_root_directory=ag_root_directory,
            output_dir=output_dir,
            graph_dir=graph_dir,
            model_name=model_name,
            split=split,
            tensor_parallel_size=tensor_parallel_size,
            use_vllm=use_vllm,
            mode=mode,
            model_weights_dir=model_weights_dir,
        )

    # ------------------------------------------------------------------
    # Load captions + clip intervals from precomputed pickle
    # ------------------------------------------------------------------

    def load_captions(
        self, video_id: str,
    ) -> Tuple[List[Tuple[int, str]], List[Dict[str, int]]]:
        """Load captions and clip intervals from the pickle saved by
        ``process_action_genome.py``.

        Returns
        -------
        captions : list[(int, str)]
        clip_intervals : list[dict]
            Each dict has ``annotated_frame``, ``start_frame``, ``end_frame``.
        """
        graph_pkl = self.graph_dir / self.args.model_name / f"{video_id}.pkl"
        if not graph_pkl.exists():
            logger.warning(
                f"No precomputed pickle for {video_id} at {graph_pkl} -- "
                f"proceeding without captions"
            )
            return [], []

        with open(graph_pkl, "rb") as f:
            clip_results: List[Dict[str, Any]] = pickle.load(f)

        if not clip_results:
            return [], []

        captions: List[Tuple[int, str]] = []
        clip_intervals: List[Dict[str, int]] = []

        for clip_idx, clip in enumerate(clip_results):
            sub = clip.get("caption") or clip.get("caption", "")
            # Defensively handle caption stored as list instead of str
            if isinstance(sub, list):
                sub = " ".join(str(s) for s in sub)
            clip_meta = clip.get("clip_metadata", {})
            annotated_frame = clip_meta.get("annotated_frame", clip_idx)
            if sub:
                captions.append((annotated_frame, sub))

            clip_intervals.append({
                "annotated_frame": annotated_frame,
                "start_frame": clip_meta.get("start_frame", 0),
                "end_frame": clip_meta.get("end_frame", 0),
            })

        logger.info(
            f"Loaded {len(captions)} captions, "
            f"{len(clip_intervals)} clip intervals for {video_id}"
        )
        return captions, clip_intervals

    # ------------------------------------------------------------------
    # Query generation (ALL objects, not just missing)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_relationship_queries(
        objects: Set[str],
    ) -> List[Dict[str, Any]]:
        """Create a combined relationship query per object."""
        queries: List[Dict[str, Any]] = []

        for obj in sorted(objects):
            prompt = AG_RELATIONSHIP_QUERY_PROMPT.format(
                object_name=obj,
                attention_labels=_ATTENTION_LABELS_STR,
                contacting_labels=_CONTACTING_LABELS_STR,
                spatial_labels=_SPATIAL_LABELS_STR,
            )
            queries.append({
                "object": obj,
                "prompt": prompt,
            })
        return queries

    # ------------------------------------------------------------------
    # Build caption context string
    # ------------------------------------------------------------------

    @staticmethod
    def _format_caption_context(
        captions: List[Tuple[int, str]],
        frame_idx: Optional[int] = None,
    ) -> str:
        """Format captions into a context string for the prompt."""
        if not captions:
            return ""

        if frame_idx is not None:
            ordered = sorted(captions, key=lambda s: abs(s[0] - frame_idx))
        else:
            ordered = sorted(captions, key=lambda s: s[0])

        lines = [f"[Frame {idx}] {text}" for idx, text in ordered]
        return (
            "The following captions describe what happens in this video:\n"
            + "\n".join(lines)
            + "\n\n"
        )

    # ------------------------------------------------------------------
    # Per-video processing
    # ------------------------------------------------------------------

    def process_video(self, video_id: str):
        """Process a single video end-to-end.

        Loads the video tensor once, then batches ALL queries across
        ALL frames with annotation-driven per-frame clips.
        """
        save_dir = self.output_dir / self.mode / self.args.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{video_id}.pkl"

        if save_path.exists():
            logger.info(f"Skipping {video_id}: output already exists at {save_path}")
            return

        # 1. Load GT annotations -------------------------------------------
        vid_key = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
        try:
            video_data = self.ag_data.get_final_data_lite(vid_key)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {video_id}: {e}")
            return

        # 2. Defer object set until after video + captions are loaded
        annotated_objects = self.get_video_objects(video_data)
        if not annotated_objects:
            logger.info(f"Skipping {video_id}: no objects found")
            return

        # 3. Load captions + clip intervals --------------------------------
        logger.info(f"[{video_id}] Loading captions ...")
        captions, clip_intervals = self.load_captions(video_id)

        # 4. Load video tensor ONCE ----------------------------------------
        frame_map = self._build_frame_map(video_id)
        if not frame_map:
            logger.warning(f"Skipping {video_id}: no frames on disk")
            return

        all_paths = [
            str(self.frames_dir / video_id / frame_map[idx])
            for idx in sorted(frame_map.keys())
        ]
        sampled_paths = all_paths[::8] if len(all_paths) > 120 else all_paths
        logger.info(
            f"[{video_id}] Loading video tensor ({len(sampled_paths)} frames) ..."
        )
        video_inputs = self.load_video_clip(sampled_paths)
        if video_inputs is None:
            logger.warning(f"Skipping {video_id}: failed to load video clip")
            return
        logger.info(
            f"[{video_id}] Video tensor loaded: shape={video_inputs[0].shape}"
        )

        # 4b. Pre-load annotation-driven per-frame clips -------------------
        video_frames_dir = self.frames_dir / video_id
        frame_to_clip: Dict[int, Any] = {}

        # Clips are only consumed by the Yes/No verification pass (skipped with
        # --skip-verification): do not decode them in that case.
        if clip_intervals and not self.skip_verification:
            logger.info(
                f"[{video_id}] Loading {len(clip_intervals)} "
                f"annotation-driven clips ..."
            )
            for interval in clip_intervals:
                ann_frame = interval["annotated_frame"]
                start = interval["start_frame"]
                end = interval["end_frame"]
                clip_paths = [
                    str(video_frames_dir / frame_map[f_idx])
                    for f_idx in range(start, end + 1, 2)
                    if f_idx in frame_map
                ]
                if not clip_paths:
                    continue
                clip_inputs = self.load_video_clip(clip_paths)
                if clip_inputs is not None:
                    frame_to_clip[ann_frame] = clip_inputs
            logger.info(
                f"[{video_id}] Loaded {len(frame_to_clip)}/"
                f"{len(clip_intervals)} clips."
            )

        # 5. Resolve object set (predcls vs sgdet) -------------------------
        video_objects, estimation_meta = self.get_objects_for_mode(
            video_data, captions, video_inputs,
        )
        if not video_objects:
            logger.info(f"Skipping {video_id}: no objects after mode={self.mode} filtering")
            return

        # 6. Collect ALL queries across ALL frames -------------------------
        bbox_frames = video_data.get("bbox_frames", {})
        all_entries: List[Tuple[str, Dict[str, Any], Optional[int]]] = []
        frame_query_counts: Dict[str, int] = {}

        for frame_stem in sorted(bbox_frames.keys()):
            queries = self.generate_relationship_queries(video_objects)
            m = re.search(r"(\d+)", frame_stem)
            fidx = int(m.group(1)) if m else None
            for q in queries:
                all_entries.append((frame_stem, q, fidx))
            frame_query_counts[frame_stem] = len(queries)

        n = len(all_entries)
        logger.info(
            f"[{video_id}] {len(bbox_frames)} frames x "
            f"{len(video_objects)} objects = "
            f"{n} total queries -- processing in single batch"
        )

        # 6b. Build annotated-frames context + precomputed query tensors ----
        annotated_ctx = self._build_annotated_context(bbox_frames, video_id)
        if annotated_ctx is None:
            annotated_ctx = video_inputs[0]  # fallback to full video
        query_ctx_map = self._build_query_context_map(
            bbox_frames, video_id, annotated_ctx,
        )

        # 7. Build batch prompts — annotated context + target frame ---------
        # Pre-compute caption context per unique frame_idx (avoids
        # re-sorting captions for every (frame, object) pair).
        caption_ctx_cache: Dict[Optional[int], str] = {}
        batch_prompts: List[Dict[str, Any]] = []
        for frame_stem, q, fidx in all_entries:
            if fidx not in caption_ctx_cache:
                caption_ctx_cache[fidx] = self._format_caption_context(captions, fidx)
            full_prompt = caption_ctx_cache[fidx] + q["prompt"]
            query_tensor = query_ctx_map.get(frame_stem, annotated_ctx)
            batch_prompts.append({
                "text": full_prompt,
                "video_inputs": [query_tensor],
                "max_new_tokens": 128,
            })

        # 8. ONE batched LLM call ------------------------------------------
        self._log_prompts(video_id, batch_prompts, tag="caption")
        logger.info(f"[{video_id}] Batched MLLM call: {n} prompts ...")
        try:
            raw_responses = self._chunked_batch_response(batch_prompts)
            raw_responses = [
                r.strip() if r else None for r in raw_responses
            ]
        except Exception as e:
            logger.error(f"Batched MLLM error: {e}")
            raw_responses = [None] * n
        logger.info(
            f"[{video_id}] Got {len(raw_responses)} responses."
        )

        # 9. Parse responses & collect verification entries -----------------
        parsed_entries = []
        verification_entries = []

        for i, (frame_stem, q, fidx_entry) in enumerate(all_entries):
            raw_resp = raw_responses[i] if i < len(raw_responses) else None
            parsed = self.parse_relationship_response(raw_resp)
            parsed_entries.append((raw_resp, parsed))
            caption_ctx = caption_ctx_cache.get(fidx_entry, "")
            # Prepend annotated frame to clip for verification
            if fidx_entry is not None and fidx_entry in frame_to_clip:
                base_clip = frame_to_clip[fidx_entry][0]
            else:
                base_clip = video_inputs[0]
            verify_tensor = self._prepend_target_frame(
                frame_stem, video_id, base_clip,
            )
            verification_entries.append((
                parsed, q["object"], [verify_tensor], caption_ctx,
            ))

        # 10. Verification (optional) -------------------------------------
        if self.skip_verification:
            logger.info(f"[{video_id}] Skipping verification (--skip-verification)")
            all_scored = [self._default_scored_from_parsed(p) for _, p in parsed_entries]
        else:
            all_scored = self.verify_all_relationships_bulk(verification_entries)

        # 11. Redistribute scored results to per-frame structure ------------
        frame_results: Dict[str, Any] = {}
        idx = 0
        for frame_stem in sorted(bbox_frames.keys()):
            qcount = frame_query_counts[frame_stem]
            query_results = []
            for _ in range(qcount):
                _, q, _ = all_entries[idx]
                raw_resp, _ = parsed_entries[idx]
                scored = all_scored[idx]
                query_results.append({
                    "object": q["object"],
                    "raw_response": raw_resp,
                    "attention": scored["attention"],
                    "contacting": scored["contacting"],
                    "spatial": scored["spatial"],
                })
                idx += 1

            frame_results[frame_stem] = {
                "objects": sorted(video_objects),
                "predictions": query_results,
            }

        # 12. Save results --------------------------------------------------
        output_record = {
            "video_id": video_id,
            "mode": self.mode,
            "model_name": self.args.model_name,
            "estimation_meta": estimation_meta,
            "video_objects": sorted(video_objects),
            "num_frames_processed": len(frame_results),
            "frames": frame_results,
        }
        with open(save_path, "wb") as f:
            pickle.dump(output_record, f)
        logger.info(
            f"Saved {len(frame_results)} frame results for {video_id} -> {save_path}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    inf = get_inference_defaults(cfg)
    parser = argparse.ArgumentParser(
        description=(
            "Generate relationship queries for ALL objects in Action Genome "
            "videos (not just missing ones) and answer them with a "
            "caption-enriched MLLM call (no RAG pipeline)."
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: auto-discovered)",
    )
    parser.add_argument(
        "--model_name", type=str, default=inf.get("default_model", "qwen25vl_7b"),
        help="Name of the VLM model key (see config.yaml models section)",
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
        "--video_id", type=str, default=None,
        help=(
            "Process only this single video ID (e.g. '0DJ6R'). "
            "Bypasses split/limit/shuffle for quick inspection."
        ),
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--use_vllm",
        action=argparse.BooleanOptionalAction,
        default=inf.get("use_vllm", True),
        help="Use vLLM for inference (default: True). --no-use_vllm for HF.",
    )
    parser.add_argument(
        "--mode", type=str, default="predcls",
        choices=["predcls", "sgdet"],
        help=(
            "Evaluation mode. "
            "'predcls' (default): use annotation objects directly. "
            "'sgdet': estimate objects from captions, intersect with annotations."
        ),
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        default=False,
        help=(
            "Skip the Yes/No verification step (~5x fewer LLM calls). "
            "Raw responses are still stored for later verification."
        ),
    )
    parser.add_argument(
        "--print_prompts",
        action="store_true",
        default=False,
        help="Print all constructed prompts for each video (debug).",
    )
    parser.add_argument(
        "--video_list", type=str, default=None,
        help="Text file of video stems to process (e.g. the worldbbox test split).",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        default=False,
        help="Shuffle video processing order (for multi-GPU concurrency).",
    )
    args = parser.parse_args()
    setup_logging(get_path(cfg, "outputs.caption_all"), f"ag_caption_all_objects_{args.mode}_{args.model_name}.log")

    processor = ActionGenomeCaptionAllObjectsProcessor(
        ag_root_directory=get_path(cfg, "ag_root"),
        output_dir=get_path(cfg, "outputs.caption_all"),
        graph_dir=get_path(cfg, "graphs"),
        model_name=args.model_name,
        split=args.split,
        tensor_parallel_size=args.tensor_parallel_size,
        use_vllm=args.use_vllm,
        mode=args.mode,
        model_weights_dir=get_path(cfg, "model_weights") or None,
    )
    processor.skip_verification = args.skip_verification
    processor.print_prompts = args.print_prompts
    processor.run(limit=args.limit, video_id=args.video_id, randomize=args.randomize,
                  video_list=args.video_list)


if __name__ == "__main__":
    main()
