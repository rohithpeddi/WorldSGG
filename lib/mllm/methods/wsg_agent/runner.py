#!/usr/bin/env python3
"""
process_ag_wsg_agent.py
=======================
**WorldSGG Agent** — extends the RAG-all pipeline with an agentic
*Strategy Router* that decides, per object, which inference strategy
to use before querying the VLM for relationship predictions.

Strategies
----------
- **DIRECT**: Object is likely visible and actively interacted with.
  Answer directly from the annotation-driven frame clip (no graph
  retrieval).  Fastest path.
- **RAG**: Object may be occluded or requires broader scene context.
  Full graph-retrieval pipeline (keyword extraction → node retrieval
  → node verification → context-enriched answer).
- **TEMPORAL**: Interaction spans multiple time segments.  Aggregate
  evidence from the target frame's clip *plus* its temporal
  neighbours.

Usage
-----
::

    python process_ag_wsg_agent.py \\
        --model_name kimikvl --split 04

Paths (``ag_root``, ``outputs.wsg_agent``, ``graphs``, etc.) are resolved
from ``config_pragya.yaml`` / ``config_utd.yaml`` via ``load_config()``.

Output pkl format is identical to ``process_ag_rag_all.py`` with an
additional ``strategy_assignments`` key recording the router decision
per object.
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
import networkx as nx
from collections import defaultdict
from tqdm import tqdm


from lib.mllm.core.vgent import Vgent
from lib.mllm.core.ag_data import AgDataBBAnnotations
from lib.mllm.core.logger_utils import setup_logging
from lib.mllm.core.config_loader import load_config, get_path, get_inference_defaults
from lib.mllm.core.prompts import SQL_ANSWER_PROMPT


from lib.mllm.base_processor import (
    ActionGenomeBaseProcessor,
    ATTENTION_RELATIONSHIPS,
    CONTACTING_RELATIONSHIPS,
    SPATIAL_RELATIONSHIPS,
    load_split_video_ids,
    get_video_belongs_to_split,
)

from lib.mllm.methods.rag_all.runner import (
    ActionGenomeRAGAllObjectsProcessor,
    AG_RELATIONSHIP_QUERY_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy Router prompt
# ---------------------------------------------------------------------------

STRATEGY_ROUTER_PROMPT = """\
You are a planning agent for video scene-graph generation. Given a list of \
objects and scene context from a video, decide the best inference strategy \
for each object.

Scene context (from video captions):
{scene_context}

Objects present in this video: {object_list}

For EACH object, choose exactly ONE strategy:

- DIRECT — The object is commonly held, worn, or actively manipulated by \
the person (e.g., phone, cup, food). Direct visual inspection of the \
relevant frame is sufficient to determine the relationship.
- RAG — The object may be in the background, partially occluded, or its \
relationship to the person depends on broader scene understanding \
(e.g., table, door, window). Graph-based retrieval of scene context \
will help.
- TEMPORAL — The person's interaction with this object changes over time \
or spans multiple segments (e.g., picking up then putting down). \
Multiple temporal clips are needed.

Respond ONLY with a JSON object mapping each object name to its strategy. \
Example:
{{"cup": "DIRECT", "table": "RAG", "blanket": "TEMPORAL"}}"""


# ---------------------------------------------------------------------------
# Agent processor
# ---------------------------------------------------------------------------

class WorldSGGAgentProcessor(ActionGenomeRAGAllObjectsProcessor):
    """Extends the RAG-all processor with a Strategy Router that adds
    an agentic planning step before relationship prediction."""

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
        # Skip RAGAll's __init__ (which drops model_weights_dir) and call
        # the base class directly so local weights are respected.
        ActionGenomeBaseProcessor.__init__(
            self,
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
    # Strategy routing
    # ------------------------------------------------------------------

    def route_strategies(
        self,
        objects: Set[str],
        captions: List[Tuple[int, str]],
        video_inputs,
    ) -> Dict[str, str]:
        """Ask the VLM to classify each object into DIRECT/RAG/TEMPORAL.

        Returns ``{"cup": "DIRECT", "table": "RAG", ...}``.
        Falls back to RAG for any object not in the response.
        """
        if not objects:
            return {}

        # Build scene context from captions
        if captions:
            ordered = sorted(captions, key=lambda s: s[0])
            lines = [f"[Frame {idx}] {text}" for idx, text in ordered]
            scene_ctx = "\n".join(lines)
        else:
            scene_ctx = "(No captions available.)"

        obj_list = ", ".join(f'"{o}"' for o in sorted(objects))
        prompt = STRATEGY_ROUTER_PROMPT.format(
            scene_context=scene_ctx,
            object_list=obj_list,
        )

        # Single text-only LLM call (no video frames needed)
        try:
            response = self.vgent.model.mllm_response(
                prompt, None, max_new_tokens=256,
            )
            cleaned = response.replace("```json", "").replace("```", "").strip()
            assignments = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError, Exception) as e:
            logger.warning(f"Strategy router parse error: {e}, defaulting all to RAG")
            assignments = {}

        # Validate and normalise
        valid_strategies = {"DIRECT", "RAG", "TEMPORAL"}
        result: Dict[str, str] = {}
        for obj in sorted(objects):
            raw = assignments.get(obj, "RAG").upper().strip()
            result[obj] = raw if raw in valid_strategies else "RAG"

        counts = defaultdict(int)
        for s in result.values():
            counts[s] += 1
        logger.info(
            f"  [Strategy Router] {dict(counts)} — "
            + ", ".join(f"{o}={s}" for o, s in result.items())
        )
        return result

    # ------------------------------------------------------------------
    # DIRECT strategy: no graph retrieval
    # ------------------------------------------------------------------

    def _process_direct_entries(
        self,
        entries: List[Tuple[str, Dict[str, Any], int]],
        frame_to_clip: Dict[int, Any],
        video_inputs,
        captions: List[Tuple[int, str]],
        video_id: str = None,
        query_ctx_map: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Optional[str]], List[Any]]:
        """Process DIRECT-strategy entries: use the annotation-driven
        clip for each frame, no graph context.

        Returns (responses, clip_tensors) — clip tensors are the
        annotation-driven clips used for later verification.
        """
        if not entries:
            return [], []

        # Build caption context once
        sub_ctx = ""
        if captions:
            ordered = sorted(captions, key=lambda s: s[0])
            lines = [f"[Frame {idx}] {text}" for idx, text in ordered]
            sub_ctx = (
                "The following captions describe what happens:\n"
                + "\n".join(lines) + "\n\n"
            )

        batch_prompts: List[Dict[str, Any]] = []
        verify_clips: List[Any] = []
        for frame_stem, q, fidx in entries:
            # Store annotation clip for verification
            clip = frame_to_clip.get(fidx, video_inputs)
            if isinstance(clip, list):
                verify_clips.append(clip[0])
            else:
                verify_clips.append(clip)
            # Use precomputed annotated-frames context
            if query_ctx_map and frame_stem in query_ctx_map:
                query_tensor = query_ctx_map[frame_stem]
            else:
                query_tensor = video_inputs[0]
            full_prompt = sub_ctx + q["prompt"]
            batch_prompts.append({
                "text": full_prompt,
                "video_inputs": [query_tensor],
                "max_new_tokens": 128,
            })

        logger.info(f"  [DIRECT] Processing {len(batch_prompts)} entries …")
        self._log_prompts(video_id or "unknown", batch_prompts, tag="DIRECT")
        try:
            responses = self._chunked_batch_response(batch_prompts)
            return [r.strip() if r else None for r in responses], verify_clips
        except Exception as e:
            logger.error(f"DIRECT batch error: {e}")
            return [None] * len(entries), verify_clips

    # ------------------------------------------------------------------
    # TEMPORAL strategy: multi-clip aggregation
    # ------------------------------------------------------------------

    def _process_temporal_entries(
        self,
        entries: List[Tuple[str, Dict[str, Any], int]],
        frame_to_clip: Dict[int, Any],
        clip_intervals: List[Dict[str, int]],
        video_inputs,
        captions: List[Tuple[int, str]],
        video_id: str = None,
        query_ctx_map: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Optional[str]], List[Any]]:
        """Process TEMPORAL-strategy entries: concatenate the target
        frame's clip with its temporal neighbours.

        Returns (responses, clip_tensors) — clip tensors are the
        temporal clips used for later verification.
        """
        if not entries:
            return [], []

        # Build ordered list of annotated frames for neighbour lookup
        ann_frames = sorted(frame_to_clip.keys())

        sub_ctx = ""
        if captions:
            ordered = sorted(captions, key=lambda s: s[0])
            lines = [f"[Frame {idx}] {text}" for idx, text in ordered]
            sub_ctx = (
                "The following captions describe what happens:\n"
                + "\n".join(lines) + "\n\n"
            )

        batch_prompts: List[Dict[str, Any]] = []
        verify_clips: List[Any] = []
        for frame_stem, q, fidx in entries:
            # Find this frame + its immediate temporal neighbours
            clip_tensors_to_cat = []

            closest = min(ann_frames, key=lambda af: abs(af - fidx)) if ann_frames else None

            if closest is not None:
                ci = ann_frames.index(closest)
                for offset in [-1, 0, 1]:
                    ni = ci + offset
                    if 0 <= ni < len(ann_frames) and ann_frames[ni] in frame_to_clip:
                        ct = frame_to_clip[ann_frames[ni]]
                        tensor = ct[0] if isinstance(ct, list) else ct
                        clip_tensors_to_cat.append(tensor)

            if clip_tensors_to_cat:
                combined = torch.cat(clip_tensors_to_cat, dim=0)
                if combined.shape[0] > 60:
                    indices = torch.linspace(0, combined.shape[0] - 1, 60).long()
                    combined = combined[indices]
                temporal_clip = combined
            else:
                temporal_clip = video_inputs[0] if video_inputs else None

            # Store temporal clip for verification
            verify_clips.append(temporal_clip)

            if temporal_clip is None:
                batch_prompts.append({
                    "text": q["prompt"],
                    "video_inputs": video_inputs,
                    "max_new_tokens": 128,
                })
            else:
                # Use precomputed annotated-frames context for query
                if query_ctx_map and frame_stem in query_ctx_map:
                    query_tensor = query_ctx_map[frame_stem]
                else:
                    query_tensor = video_inputs[0]
                full_prompt = sub_ctx + q["prompt"]
                batch_prompts.append({
                    "text": full_prompt,
                    "video_inputs": [query_tensor],
                    "max_new_tokens": 128,
                })

        logger.info(f"  [TEMPORAL] Processing {len(batch_prompts)} entries …")
        self._log_prompts(video_id or "unknown", batch_prompts, tag="TEMPORAL")
        try:
            responses = self._chunked_batch_response(batch_prompts)
            return [r.strip() if r else None for r in responses], verify_clips
        except Exception as e:
            logger.error(f"TEMPORAL batch error: {e}")
            return [None] * len(entries), verify_clips

    # ------------------------------------------------------------------
    # Main per-video processing (overrides parent)
    # ------------------------------------------------------------------

    def process_video(self, video_id: str):
        """Process a single video with the agentic Strategy Router.

        Flow:
        1. Load annotations, graphs, video tensor, objects
        2. **PLAN**: Route each object to DIRECT / RAG / TEMPORAL
        3. **ACT**: Execute the assigned strategy per object
        4. Verify, parse, and save
        """
        save_dir = self.output_dir / self.mode / self.args.model_name
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

        annotated_objects = self.get_video_objects(video_data)
        if not annotated_objects:
            logger.info(f"Skipping {video_id}: no objects found")
            return

        # 2. Precomputed graph & captions
        logger.info(f"[{video_id}][AGENT] Loading precomputed graph …")
        video_graph, entity_graph, captions, clip_intervals = (
            self.load_precomputed_graphs(video_id)
        )
        if video_graph is None or video_graph.number_of_nodes() == 0:
            logger.warning(f"Skipping {video_id}: no precomputed graph")
            return

        # 3. Pre-compute graph embeddings
        logger.info(f"[{video_id}][AGENT] Pre-computing graph embeddings …")
        embedding_cache = self.vgent.precompute_graph_embeddings(
            video_graph, entity_graph,
        )

        # 4. Load video tensor
        frame_map = self._build_frame_map(video_id)
        all_frames = self.get_all_frames(video_id)
        if not all_frames or not frame_map:
            logger.warning(f"Skipping {video_id}: no frames on disk")
            return

        all_paths = [
            str(self.frames_dir / video_id / frame_map[idx])
            for idx in sorted(frame_map.keys())
        ]
        sampled_paths = all_paths[::8] if len(all_paths) > 120 else all_paths
        logger.info(f"[{video_id}][AGENT] Loading video tensor ({len(sampled_paths)} frames) …")
        video_inputs = self.load_video_clip(sampled_paths)
        if video_inputs is None:
            logger.warning(f"Skipping {video_id}: failed to load video clip")
            return

        # 5. Load annotation-driven per-frame clips
        video_frames_dir = self.frames_dir / video_id
        frame_to_clip: Dict[int, Any] = {}
        if clip_intervals:
            for interval in clip_intervals:
                ann_frame = interval["annotated_frame"]
                start, end = interval["start_frame"], interval["end_frame"]
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

        # 6. Resolve object set
        video_objects, estimation_meta = self.get_objects_for_mode(
            video_data, captions, video_inputs,
        )
        if not video_objects:
            logger.info(f"Skipping {video_id}: no objects after mode={self.mode} filtering")
            return

        # =================================================================
        # 7. PLANNING STEP — Strategy Router
        # =================================================================
        logger.info(f"[{video_id}][AGENT] Planning: routing {len(video_objects)} objects …")
        strategy_map = self.route_strategies(
            video_objects, captions, video_inputs,
        )

        # 8. Build ALL queries and partition by strategy
        bbox_frames = video_data.get("bbox_frames", {})
        direct_entries: List[Tuple[str, Dict[str, Any], int]] = []
        rag_entries: List[Tuple[str, Dict[str, Any], int]] = []
        temporal_entries: List[Tuple[str, Dict[str, Any], int]] = []

        # Track original indices for reassembly
        entry_order: List[Tuple[str, int]] = []  # (strategy, idx_in_strategy_list)
        frame_query_counts: Dict[str, int] = {}
        all_entries_ordered: List[Tuple[str, Dict[str, Any], int]] = []

        for frame_stem in sorted(bbox_frames.keys()):
            queries = self.generate_relationship_queries(video_objects)
            m = re.search(r"(\d+)", frame_stem)
            fidx = int(m.group(1)) if m else 0
            frame_query_counts[frame_stem] = len(queries)
            for q in queries:
                strategy = strategy_map.get(q["object"], "RAG")
                all_entries_ordered.append((frame_stem, q, fidx))
                if strategy == "DIRECT":
                    entry_order.append(("DIRECT", len(direct_entries)))
                    direct_entries.append((frame_stem, q, fidx))
                elif strategy == "TEMPORAL":
                    entry_order.append(("TEMPORAL", len(temporal_entries)))
                    temporal_entries.append((frame_stem, q, fidx))
                else:
                    entry_order.append(("RAG", len(rag_entries)))
                    rag_entries.append((frame_stem, q, fidx))

        # 8b. Build annotated-frames context + precomputed query tensors ---
        annotated_ctx = self._build_annotated_context(bbox_frames, video_id)
        if annotated_ctx is None:
            annotated_ctx = video_inputs[0]  # fallback to full video
        query_ctx_map = self._build_query_context_map(
            bbox_frames, video_id, annotated_ctx,
        )

        n_total = len(all_entries_ordered)
        logger.info(
            f"[{video_id}][AGENT] {n_total} total queries — "
            f"DIRECT={len(direct_entries)}, "
            f"RAG={len(rag_entries)}, "
            f"TEMPORAL={len(temporal_entries)}"
        )

        # =================================================================
        # 9. EXECUTION — run each strategy
        # =================================================================

        # 9a. DIRECT
        direct_responses, direct_clips = self._process_direct_entries(
            direct_entries, frame_to_clip, video_inputs, captions,
            video_id=video_id, query_ctx_map=query_ctx_map,
        )

        # 9b. RAG (reuse parent's batched RAG pipeline)
        if rag_entries:
            rag_responses, rag_clips, rag_contexts = self._batch_all_video_queries(
                all_entries=rag_entries,
                video_inputs=video_inputs,
                captions=captions,
                video_graph=video_graph,
                entity_graph=entity_graph,
                embedding_cache=embedding_cache,
                frame_to_clip=frame_to_clip,
                video_id=video_id,
                query_ctx_map=query_ctx_map,
            )
        else:
            rag_responses, rag_clips, rag_contexts = [], [], []

        # 9c. TEMPORAL
        temporal_responses, temporal_clips = self._process_temporal_entries(
            temporal_entries, frame_to_clip, clip_intervals,
            video_inputs, captions, video_id=video_id,
            query_ctx_map=query_ctx_map,
        )

        # =================================================================
        # 10. Reassemble responses in original order
        # =================================================================
        raw_responses: List[Optional[str]] = []
        node_contexts: List[str] = []
        clip_tensors: List[Any] = []

        for strategy, idx in entry_order:
            if strategy == "DIRECT":
                raw_responses.append(
                    direct_responses[idx] if idx < len(direct_responses) else None
                )
                node_contexts.append("")
                clip_tensors.append(
                    direct_clips[idx] if idx < len(direct_clips) else None
                )
            elif strategy == "TEMPORAL":
                raw_responses.append(
                    temporal_responses[idx] if idx < len(temporal_responses) else None
                )
                node_contexts.append("")
                clip_tensors.append(
                    temporal_clips[idx] if idx < len(temporal_clips) else None
                )
            else:
                raw_responses.append(
                    rag_responses[idx] if idx < len(rag_responses) else None
                )
                node_contexts.append(
                    rag_contexts[idx] if idx < len(rag_contexts) else ""
                )
                clip_tensors.append(
                    rag_clips[idx] if idx < len(rag_clips) else None
                )

        # 11. Parse & verify
        sub_ctx = ""
        if captions:
            ordered = sorted(captions, key=lambda s: s[0])
            lines = [f"[Frame {si}] {st}" for si, st in ordered]
            sub_ctx = (
                "The following captions describe what happens:\n"
                + "\n".join(lines) + "\n\n"
            )

        parsed_entries = []
        verification_entries = []
        for i, (frame_stem, q, _) in enumerate(all_entries_ordered):
            raw_resp = raw_responses[i] if i < len(raw_responses) else None
            parsed = self.parse_relationship_response(raw_resp)
            parsed_entries.append((raw_resp, parsed))
            nctx = node_contexts[i] if i < len(node_contexts) else ""
            full_ctx = sub_ctx + nctx
            clip = clip_tensors[i] if i < len(clip_tensors) and clip_tensors[i] is not None else None
            _base = clip if clip is not None else video_inputs[0]
            _verify_tensor = self._prepend_target_frame(
                frame_stem, video_id, _base,
            )
            verification_entries.append((parsed, q["object"], [_verify_tensor], full_ctx))

        if self.skip_verification:
            all_scored = [self._default_scored_from_parsed(p) for _, p in parsed_entries]
        else:
            all_scored = self.verify_all_relationships_bulk(verification_entries)

        # 12. Redistribute to per-frame structure
        frame_results: Dict[str, Any] = {}
        idx = 0
        for frame_stem in sorted(bbox_frames.keys()):
            qcount = frame_query_counts[frame_stem]
            query_results = []
            for _ in range(qcount):
                _, q, _ = all_entries_ordered[idx]
                raw_resp, _ = parsed_entries[idx]
                scored = all_scored[idx]
                query_results.append({
                    "object": q["object"],
                    "raw_response": raw_resp,
                    "attention": scored["attention"],
                    "contacting": scored["contacting"],
                    "spatial": scored["spatial"],
                    "strategy": strategy_map.get(q["object"], "RAG"),
                })
                idx += 1
            frame_results[frame_stem] = {
                "objects": sorted(video_objects),
                "predictions": query_results,
            }

        # 13. Save
        output_record = {
            "video_id": video_id,
            "mode": self.mode,
            "model_name": self.args.model_name,
            "estimation_meta": estimation_meta,
            "video_objects": sorted(video_objects),
            "strategy_assignments": strategy_map,
            "num_frames_processed": len(frame_results),
            "frames": frame_results,
        }
        with open(save_path, "wb") as f:
            pickle.dump(output_record, f)
        logger.info(
            f"Saved {len(frame_results)} frame results for {video_id} → {save_path}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    inf = get_inference_defaults(cfg)
    parser = argparse.ArgumentParser(
        description=(
            "WorldSGG Agent: relationship prediction with an agentic "
            "Strategy Router (DIRECT / RAG / TEMPORAL) per object."
        ),
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model_name", type=str,
        default=inf.get("default_model", "qwen25vl_7b"),
    )
    parser.add_argument(
        "--split", type=str, default="04",
        help="Split filter (test/train or first-letter bucket)",
    )
    parser.add_argument("--limit", type=int, default=None)
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
    )
    parser.add_argument(
        "--use_vllm",
        action=argparse.BooleanOptionalAction,
        default=inf.get("use_vllm", True),
    )
    parser.add_argument(
        "--mode", type=str, default="predcls",
        choices=["predcls", "sgdet"],
    )
    parser.add_argument(
        "--skip-verification", action="store_true", default=False,
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
    setup_logging(
        get_path(cfg, "outputs.wsg_agent"),
        f"wsg_agent_{args.mode}_{args.model_name}.log",
    )

    processor = WorldSGGAgentProcessor(
        ag_root_directory=get_path(cfg, "ag_root"),
        output_dir=get_path(cfg, "outputs.wsg_agent"),
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
