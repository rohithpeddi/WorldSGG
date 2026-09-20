#!/usr/bin/env python3
"""
process_ag_rag_all.py
=====================
Variant of ``process_ag_rag.py`` that queries *all* video-level objects on
every annotated frame (not just missing ones).

This gives full relationship annotations for every object on every frame,
useful for training and dense evaluation.

Usage
-----
Standard (per-query sequential)::

    python process_ag_rag_all.py \
        --model_name kimikvl --split 04

Paths (``ag_root``, ``outputs.rag_all``, ``graphs``, etc.) are resolved
from ``config_pragya.yaml`` / ``config_utd.yaml`` via ``load_config()``.

Fast mode (all queries batched in one LLM call per video)::

    python process_ag_rag_all.py --fast \
        --model_name kimikvl --split 04

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
import networkx as nx
from collections import defaultdict
from torchvision import transforms
from tqdm import tqdm

# Allow imports from the parent package

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates for relationship prediction (all-objects variant)
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


# ---------------------------------------------------------------------------
# Core processor class
# ---------------------------------------------------------------------------

class ActionGenomeRAGAllObjectsProcessor(ActionGenomeBaseProcessor):
    """Generates and answers relationship queries for ALL video-level objects
    on every annotated frame, using the Vgent RAG pipeline."""

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
    # Load precomputed graphs / captions
    # ------------------------------------------------------------------

    def load_precomputed_graphs(
        self, video_id: str,
    ) -> Tuple[
        Optional["nx.DiGraph"],
        Optional["defaultdict"],
        List[Tuple[int, str]],
        List[Dict[str, int]],
    ]:
        """
        Load the precomputed graph and captions saved by
        ``process_action_genome.py``.

        Returns
        -------
        video_graph : nx.DiGraph | None
        entity_graph : defaultdict(set) | None
        captions : list[(int, str)]
        clip_intervals : list[dict]
            Each dict has ``annotated_frame``, ``start_frame``, ``end_frame``
            matching the annotation-driven segmentation from graph construction.
        """
        graph_pkl = self.graph_dir / self.args.model_name / f"{video_id}.pkl"
        if not graph_pkl.exists():
            logger.warning(f"Precomputed graph not found: {graph_pkl}")
            return None, None, [], []

        with open(graph_pkl, "rb") as f:
            clip_results: List[Dict[str, Any]] = pickle.load(f)

        if not clip_results:
            logger.warning(f"Empty graph pickle for {video_id}")
            return None, None, [], []

        # --- Merge per-clip graphs into a single video-level graph ---------
        video_graph = nx.DiGraph()
        entity_graph: defaultdict = defaultdict(set)
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

            # Extract clip interval (same segmentation as graph construction)
            clip_intervals.append({
                "annotated_frame": annotated_frame,
                "start_frame": clip_meta.get("start_frame", 0),
                "end_frame": clip_meta.get("end_frame", 0),
            })

            clip_graph: nx.DiGraph = clip.get("graph", None)
            if clip_graph is None or not isinstance(clip_graph, nx.DiGraph):
                continue

            # Re-index clip nodes so they don't collide across clips.
            offset = len(video_graph)
            for node_id in clip_graph.nodes:
                new_id = node_id + offset
                video_graph.add_node(
                    new_id, **clip_graph.nodes[node_id],
                    source_annotated_frame=annotated_frame,
                )
                # Update entity graph
                node_data = clip_graph.nodes[node_id]
                for entity in (
                    node_data.get("entities", [])
                    + node_data.get("actions", [])
                    + node_data.get("scenes", [])
                ):
                    entity_name = entity.split(",")[0].lower().strip()
                    entity_graph[entity_name].add(new_id)

            for u, v, data in clip_graph.edges(data=True):
                video_graph.add_edge(u + offset, v + offset, **data)

        logger.info(
            f"Loaded precomputed graph for {video_id}: "
            f"{video_graph.number_of_nodes()} nodes, "
            f"{video_graph.number_of_edges()} edges, "
            f"{len(captions)} captions, "
            f"{len(clip_intervals)} clip intervals"
        )
        return video_graph, entity_graph, captions, clip_intervals

    # ------------------------------------------------------------------
    # Query generation (ALL objects, not just missing)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_relationship_queries(
        objects: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        For each object, create a *combined* relationship query
        asking the LLM to predict attention, contacting, and spatial
        relationships between the object and the person.

        Returns a list of query dicts, each containing:
          - object: str
          - prompt: str  (the fully-formatted prompt)
        """
        queries: List[Dict[str, Any]] = []
        attention_labels = ", ".join(ATTENTION_RELATIONSHIPS)
        contacting_labels = ", ".join(CONTACTING_RELATIONSHIPS)
        spatial_labels = ", ".join(SPATIAL_RELATIONSHIPS)

        for obj in sorted(objects):
            prompt = AG_RELATIONSHIP_QUERY_PROMPT.format(
                object_name=obj,
                attention_labels=attention_labels,
                contacting_labels=contacting_labels,
                spatial_labels=spatial_labels,
            )
            queries.append({
                "object": obj,
                "prompt": prompt,
            })
        return queries

    # ------------------------------------------------------------------
    # Templatized refinement questions (replaces LLM-generated ones)
    # ------------------------------------------------------------------

    @staticmethod
    def _templatized_refine_questions(object_name: str) -> Dict[str, str]:
        """Fixed sub-questions for relationship queries."""
        return {
            "Q1": f"Is the '{object_name}' visible or interacted with in this video segment?",
            "Q2": "Is there a person visible in this video segment?",
        }


    def _batch_all_video_queries(
        self,
        all_entries: List[Tuple[str, Dict[str, Any], int]],
        video_inputs,
        captions,
        video_graph,
        entity_graph,
        embedding_cache,
        frame_to_clip: Optional[Dict[int, Any]] = None,
        video_id: Optional[str] = None,
        query_ctx_map: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Optional[str]], List[Any], List[str]]:
        """Process *every* relationship query for an entire video.

        **Optimised pipeline:**

        1. **Deduplicated** keyword extraction (unique prompts only)
        2. **Deduplicated** cached node retrieval (unique prompts only)
        3. **Deduplicated** batched node checking (unique prompts only)
        4. **Frame-specific** batched final answer — each (frame, object)
           pair is answered with the annotation-driven clip for that frame
           (same segmentation as graph construction), so predictions
           vary across frames.
        """
        if not all_entries:
            return [], [], []

        n = len(all_entries)
        all_prompts = [q["prompt"] for _, q, _ in all_entries]

        if video_inputs is None or video_inputs[0] is None:
            return [None] * n, [None] * n, [""] * n

        split_clips = torch.split(
            video_inputs[0], self.args.chunk_size, dim=0,
        )

        # ==================================================================
        # Steps 1–3: DEDUPLICATED — run only for unique object prompts
        # ==================================================================
        unique_prompts = list(dict.fromkeys(all_prompts))  # preserves order
        prompt_to_uidx = {p: i for i, p in enumerate(unique_prompts)}
        n_unique = len(unique_prompts)

        logger.info(
            f"  [Dedup] {n} total entries → {n_unique} unique prompts "
            f"(~{n // max(n_unique, 1)}× dedup ratio)"
        )

        # ---- Step 1: ONE batched keyword extraction (unique only) --------
        logger.info(f"  [Step 1/4] Keyword extraction for {n_unique} unique prompts …")
        unique_kw_results = self.vgent.batch_extract_keywords(unique_prompts)
        logger.info(f"  [Step 1/4] Keyword extraction done.")

        # ---- Step 2: Cached node retrieval (unique only, no LLM) ---------
        logger.info(f"  [Step 2/4] Cached node retrieval for {n_unique} unique queries …")
        unique_retrieved = []
        for qi, (prompt, (query_list, llm_info)) in enumerate(
            zip(unique_prompts, unique_kw_results)
        ):
            if qi % 10 == 0:
                logger.info(f"    retrieval {qi+1}/{n_unique} …")
            retrieved = self.vgent.retrieve_nodes_with_cache(
                prompt, query_list, video_inputs,
                candidates=[], video_graph=video_graph,
                entity_graph=entity_graph, captions=captions,
                llm_info=llm_info, embedding_cache=embedding_cache,
            )
            unique_retrieved.append(retrieved)
        logger.info(f"  [Step 2/4] Node retrieval done.")

        # ---- Step 3: Templatized refinement + ONE batched node check -----
        # Build check prompts only for unique queries
        check_prompts: List[Dict[str, Any]] = []
        check_mapping: List[Tuple[int, int]] = []   # (unique_idx, node)

        # Build object names for unique prompts (extract from all_entries)
        unique_objects = []
        for up in unique_prompts:
            # Find the first entry with this prompt to get object name
            for _, q, _ in all_entries:
                if q["prompt"] == up:
                    unique_objects.append(q["object"])
                    break

        ftc_check = frame_to_clip or {}
        for uidx, obj_name in enumerate(unique_objects):
            nodes = unique_retrieved[uidx].get("nodes", [])
            info = self._templatized_refine_questions(obj_name)

            for node in nodes:
                # Resolve the correct video clip for this graph node
                nd = video_graph.nodes.get(node, {}) if video_graph is not None else {}
                ann_frame = nd.get("source_annotated_frame")
                if ann_frame is not None and ann_frame in ftc_check:
                    node_clip = ftc_check[ann_frame][0]
                elif node < len(split_clips):
                    node_clip = split_clips[node]
                else:
                    continue

                caption_prompt = ""
                if captions is not None:
                    if ann_frame is not None:
                        sel = [t for ts, t in captions if ts == ann_frame]
                    else:
                        t0 = node * self.args.chunk_size // self.args.fps
                        t1 = (node + 1) * self.args.chunk_size // self.args.fps
                        sel = [t for ts, t in captions if t0 <= ts < t1]
                    if sel:
                        caption_prompt = (
                            " This video's captions are listed below:\n"
                            + " ".join(sel) + "\n"
                        )

                instruct = SQL_ANSWER_PROMPT.format(questions=info) + caption_prompt
                check_prompts.append({
                    "text": instruct,
                    "video_inputs": [node_clip],
                    "max_new_tokens": getattr(self, "gen_max_tokens", None) or 256,
                })
                check_mapping.append((uidx, node))

        logger.info(
            f"  [Step 3/4] Batched node checking: {len(check_prompts)} prompts "
            f"across {n_unique} unique queries …"
        )
        if check_prompts:
            try:
                check_responses = self._chunked_batch_response(check_prompts)
            except Exception as e:
                logger.error(f"Batched node-check error: {e}")
                check_responses = [None] * len(check_prompts)
        else:
            check_responses = []
        logger.info(f"  [Step 3/4] Node checking done.")

        # Group check results by unique idx and rank nodes
        unique_entry_results: Dict[int, Dict[int, Any]] = defaultdict(dict)
        for (uidx, node), resp in zip(check_mapping, check_responses):
            try:
                pred = json.loads(
                    resp.replace("```json", "").replace("```", "").strip()
                ) if resp else None
            except (json.JSONDecodeError, TypeError, AttributeError):
                pred = None
            unique_entry_results[uidx][node] = pred

        from lib.mllm.core.retrieval import count_and_sort_filtered

        unique_clip_indices: List[Optional[int]] = []
        for uidx in range(n_unique):
            cr = unique_entry_results.get(uidx, {})
            if cr:
                _, sorted_nodes = count_and_sort_filtered(cr)
                if sorted_nodes and sorted_nodes[0] < len(split_clips):
                    unique_clip_indices.append(sorted_nodes[0])
                else:
                    nodes = unique_retrieved[uidx].get("nodes", [])
                    unique_clip_indices.append(
                        nodes[0] if nodes and nodes[0] < len(split_clips) else None
                    )
            else:
                nodes = unique_retrieved[uidx].get("nodes", [])
                unique_clip_indices.append(
                    nodes[0] if nodes and nodes[0] < len(split_clips) else None
                )

        # ---- Extract node contexts (unique, then broadcast) --------------
        unique_node_contexts: List[str] = []
        for uidx in range(n_unique):
            nodes = unique_retrieved[uidx].get("nodes", [])
            node_ctx = ""
            if nodes and video_graph is not None:
                top_node = nodes[0]
                if top_node in video_graph.nodes:
                    nd = video_graph.nodes[top_node]
                    parts = (
                        nd.get("entities", [])
                        + nd.get("actions", [])
                        + nd.get("scenes", [])
                    )
                    if parts:
                        node_ctx = (
                            "Relevant scene context from video analysis:\n"
                            + "; ".join(parts) + "\n\n"
                        )
            unique_node_contexts.append(node_ctx)

        # Broadcast deduped results back to all entries
        node_contexts = [unique_node_contexts[prompt_to_uidx[p]] for p in all_prompts]

        # ==================================================================
        # Step 4: FRAME-SPECIFIC — annotated context for queries, clips for verify
        # ==================================================================
        ftc = frame_to_clip or {}
        batch_prompts: List[Dict[str, Any]] = []
        clip_tensors: List[Any] = []  # stored for verification (annotation/RAG clips)
        for idx, (prompt, (frame_stem, _, fidx)) in enumerate(zip(all_prompts, all_entries)):
            # Store annotation/RAG clip for later verification
            if fidx in ftc:
                clip_tensor = ftc[fidx][0]  # load_video_clip returns [tensor]
            else:
                uidx = prompt_to_uidx[prompt]
                clip_idx = unique_clip_indices[uidx]
                clip_tensor = (
                    split_clips[clip_idx]
                    if clip_idx is not None and clip_idx < len(split_clips)
                    else video_inputs[0]
                )
            clip_tensors.append(clip_tensor)
            # Use precomputed query tensor (annotated context + target frame)
            if query_ctx_map and frame_stem in query_ctx_map:
                query_visual = query_ctx_map[frame_stem]
            else:
                query_visual = video_inputs[0]
            # Prepend graph context so the VLM has scene knowledge
            graph_ctx = node_contexts[idx]
            full_prompt = graph_ctx + prompt if graph_ctx else prompt
            batch_prompts.append({
                "text": full_prompt,
                "video_inputs": [query_visual],
                "max_new_tokens": getattr(self, "gen_max_tokens", None) or 128,
            })

        self._log_prompts(video_id or "unknown", batch_prompts, tag="rag")
        logger.info(f"  [Step 4/4] Batched final answer: {len(batch_prompts)} prompts …")
        try:
            responses = self._chunked_batch_response(batch_prompts)
            logger.info(f"  [Step 4/4] Final answer done. Got {len(responses)} responses.")
            return [r.strip() if r else None for r in responses], clip_tensors, node_contexts
        except Exception as e:
            logger.error(f"Batched final-answer error: {e}")
            return [None] * n, clip_tensors, node_contexts

    # ------------------------------------------------------------------
    # Per-video processing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Thinking-model support
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_thinking(text):
        """Drop a Qwen3-VL-Thinking reasoning trace, keeping the final answer.

        ``parse_relationship_response`` falls back to a greedy ``{.*}`` search,
        which would otherwise span from the first brace inside the trace to the
        last brace of the answer and fail to decode.
        """
        if not text:
            return text
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        return text.strip()

    def _chunked_batch_response(self, prompts, *a, **kw):
        return [self._strip_thinking(r)
                for r in super()._chunked_batch_response(prompts, *a, **kw)]

    def process_video(self, video_id: str):
        """Process a single video end-to-end.

        Loads the video tensor once, then processes ALL queries across
        ALL frames with deduplicated retrieval and frame-specific
        annotation-driven clips.
        """
        save_dir = (self.output_dir / self.mode
                    / (self.args.model_name + getattr(self, "run_tag", "")))
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

        # 2. Video-level objects (ALL of them) -----------------------------
        annotated_objects = self.get_video_objects(video_data)
        if not annotated_objects:
            logger.info(f"Skipping {video_id}: no objects found")
            return

        # 3. Precomputed graph & captions ---------------------------------
        logger.info(f"[{video_id}][FAST] Loading precomputed graph …")
        video_graph, entity_graph, captions, clip_intervals = self.load_precomputed_graphs(
            video_id,
        )
        if video_graph is None or video_graph.number_of_nodes() == 0:
            logger.warning(
                f"Skipping {video_id}: no precomputed graph available"
            )
            return

        # 4. Pre-compute graph embeddings once -----------------------------
        logger.info(f"[{video_id}][FAST] Pre-computing graph embeddings (nodes={video_graph.number_of_nodes()}) …")
        embedding_cache = self.vgent.precompute_graph_embeddings(
            video_graph, entity_graph,
        )
        logger.info(f"[{video_id}][FAST] Graph embeddings cached.")

        # 5. Load video tensor ONCE ----------------------------------------
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
        logger.info(f"[{video_id}][FAST] Loading video tensor ({len(sampled_paths)} frames) …")
        video_inputs = self.load_video_clip(sampled_paths)
        if video_inputs is None:
            logger.warning(f"Skipping {video_id}: failed to load video clip")
            return
        logger.info(f"[{video_id}][FAST] Video tensor loaded: shape={video_inputs[0].shape}")

        # 5b. Pre-load annotation-driven per-frame clips -------------------
        # Uses the same intervals as graph construction (process_ag_graphs.py)
        video_frames_dir = self.frames_dir / video_id
        frame_to_clip: Dict[int, Any] = {}  # annotated_frame → video tensor

        if clip_intervals:
            logger.info(
                f"[{video_id}][FAST] Loading {len(clip_intervals)} "
                f"annotation-driven clips …"
            )
            for interval in clip_intervals:
                ann_frame = interval["annotated_frame"]
                start = interval["start_frame"]
                end = interval["end_frame"]
                # Same frame sampling as graph construction: step=2
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
                f"[{video_id}][FAST] Loaded {len(frame_to_clip)}/"
                f"{len(clip_intervals)} clips."
            )

        # 5c. Resolve object set (predcls vs sgdet) -------------------------
        video_objects, estimation_meta = self.get_objects_for_mode(
            video_data, captions, video_inputs,
        )
        if not video_objects:
            logger.info(f"Skipping {video_id}: no objects after mode={self.mode} filtering")
            return

        # 6. Collect ALL queries across ALL frames -------------------------
        bbox_frames = video_data.get("bbox_frames", {})
        all_entries: List[Tuple[str, Dict[str, Any], int]] = []
        frame_query_counts: Dict[str, int] = {}  # frame -> count

        for frame_stem in sorted(bbox_frames.keys()):
            queries = self.generate_relationship_queries(video_objects)
            m = re.search(r"(\d+)", frame_stem)
            fidx = int(m.group(1)) if m else 0
            for q in queries:
                all_entries.append((frame_stem, q, fidx))
            frame_query_counts[frame_stem] = len(queries)

        logger.info(
            f"[{video_id}][FAST] {len(bbox_frames)} frames × "
            f"{len(video_objects)} objects = "
            f"{len(all_entries)} total queries — processing in bulk"
        )

        # 6b. Build annotated-frames context + precomputed query tensors ----
        annotated_ctx = self._build_annotated_context(bbox_frames, video_id)
        if annotated_ctx is None:
            annotated_ctx = video_inputs[0]  # fallback to full video
        query_ctx_map = self._build_query_context_map(
            bbox_frames, video_id, annotated_ctx,
        )

        # 7. Process ALL queries in batched LLM calls ----------------------
        raw_responses, clip_tensors, node_contexts = self._batch_all_video_queries(
            all_entries=all_entries,
            video_inputs=video_inputs,
            captions=captions,
            video_graph=video_graph,
            entity_graph=entity_graph,
            embedding_cache=embedding_cache,
            frame_to_clip=frame_to_clip,
            video_id=video_id,
            query_ctx_map=query_ctx_map,
        )

        # 8. Parse responses & collect verification entries -----------------
        parsed_entries = []
        verification_entries = []

        # Build caption context once (same for all queries in fast mode)
        _sub_ctx = ""
        if captions:
            _ordered = sorted(captions, key=lambda s: s[0])
            _lines = [f"[Frame {si}] {st}" for si, st in _ordered]
            _sub_ctx = (
                "The following captions describe what happens in this video:\n"
                + "\n".join(_lines) + "\n\n"
            )

        for i, (entry_frame, q, _) in enumerate(all_entries):
            raw_resp = raw_responses[i] if i < len(raw_responses) else None
            parsed = self.parse_relationship_response(raw_resp)
            parsed_entries.append((raw_resp, parsed))
            # Combine caption context with per-query graph-node context
            _node_ctx = node_contexts[i] if i < len(node_contexts) else ""
            _full_ctx = _sub_ctx + _node_ctx
            # Prepend annotated frame to clip for verification
            _clip = clip_tensors[i] if i < len(clip_tensors) and clip_tensors[i] is not None else None
            _base = _clip if _clip is not None else video_inputs[0]
            _verify_tensor = self._prepend_target_frame(
                entry_frame, video_id, _base,
            )
            verification_entries.append((
                parsed, q["object"], [_verify_tensor], _full_ctx,
            ))

        # 9. Verification (optional) --------------------------------------
        if self.skip_verification:
            logger.info(f"[{video_id}] Skipping verification (--skip-verification)")
            all_scored = [self._default_scored_from_parsed(p) for _, p in parsed_entries]
        else:
            all_scored = self.verify_all_relationships_bulk(verification_entries)

        # 10. Redistribute scored results to per-frame structure -----------
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

        # 9. Save results --------------------------------------------------
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
            "Generate relationship queries for ALL objects in Action Genome "
            "videos (not just missing ones) and answer them with a "
            "vLLM-backed RAG pipeline."
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
        "--max_new_tokens", type=int, default=None,
        help=(
            "Generation budget per call (default: the built-in 256 for node "
            "checks / 128 for final answers). Thinking models need ~4096: the "
            "reasoning trace is emitted before the answer."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (Qwen3-VL-Thinking wants 0.6).",
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Nucleus sampling top-p (Qwen3-VL-Thinking wants 0.95).",
    )
    parser.add_argument(
        "--tag", default="",
        help="Suffix for the output model dir (prompt/ablation variants).",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        default=False,
        help="Shuffle video processing order (for multi-GPU concurrency).",
    )
    args = parser.parse_args()
    setup_logging(get_path(cfg, "outputs.rag_all"), f"ag_rag_all_objects_{args.mode}_{args.model_name}.log")

    processor = ActionGenomeRAGAllObjectsProcessor(
        ag_root_directory=get_path(cfg, "ag_root"),
        output_dir=get_path(cfg, "outputs.rag_all"),
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
    processor.gen_max_tokens = args.max_new_tokens
    processor.run_tag = args.tag
    processor.args.temperature = args.temperature
    processor.args.top_p = args.top_p
    processor.run(limit=args.limit, video_id=args.video_id, randomize=args.randomize,
                  video_list=args.video_list)


if __name__ == "__main__":
    main()
