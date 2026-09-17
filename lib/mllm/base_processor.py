#!/usr/bin/env python3
"""
process_ag_base.py
==================
Abstract base class for Action Genome relationship-prediction processors.

Provides all shared infrastructure:
  - Relationship label vocabularies and the object-estimation prompt
  - Split-filtering helpers
  - Annotation helpers (object extraction, missing-object computation)
  - Frame / clip loading utilities
  - SGDet / PredCls mode-aware object resolution
  - Response parsing and validation
  - The main ``run()`` loop

Concrete subclasses must implement:
  - ``process_video(video_id)``
"""

import os
import sys
import json
import pickle
import random
import re
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple

import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Allow imports from the parent package

from lib.mllm.core.vgent import Vgent
from lib.mllm.core.ag_data import AgDataBBAnnotations  # noqa: F401 (legacy source)
from lib.mllm.data.worldbbox import make_ag_data
from lib.mllm.core.logger_utils import setup_logging
from lib.mllm.models.utils import fetch_video, resize_video


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
# SGDet object-estimation prompt (shared across all scripts)
# ---------------------------------------------------------------------------

AG_OBJECT_ESTIMATION_PROMPT = """\
You are analyzing a video of a person performing an activity.

{caption_context}

Here is a list of candidate objects that may be present in the video:
{candidate_objects}

Based on the video and the captions above, select ALL objects from the candidate list (excluding "person") that the person interacts with or that are relevant to the activity. You may also include additional objects not in the candidate list if they are clearly present.

Respond ONLY with a JSON list of object name strings. Use short, lowercase, singular noun phrases (e.g., "cup", "laptop", "blanket"). Prefer using the exact labels from the candidate list when possible. Example:
["cup", "laptop", "blanket", "table"]"""

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


def load_video_list(path: str) -> set:
    """Load video stems (one per line, ``.mp4`` optional) from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return {Path(line.strip()).stem for line in f if line.strip()}


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
# Abstract base processor
# ---------------------------------------------------------------------------

class ActionGenomeBaseProcessor(ABC):
    """Shared infrastructure for AG relationship-prediction processors.

    Subclasses must implement ``process_video``.
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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- AG root (needed for object_classes.txt) -----------------------
        self.ag_root_directory = Path(ag_root_directory)

        # ---- Precomputed graph / caption source directory -----------------
        self.graph_dir = Path(graph_dir)

        # ---- AG data loader ------------------------------------------------
        # worldbbox test PKLs by default (inference.annotation_source), legacy
        # bbox_annotations_3d_obb_final when set to "legacy".
        self.ag_data = make_ag_data(ag_root_directory)

        # ---- Frame directories ---------------------------------------------
        self.frames_dir = Path(ag_root_directory) / "frames"
        self.frames_annotated_dir = Path(ag_root_directory) / "frames_annotated"

        # ---- Object vocabulary (loaded once, shared across all videos) -----
        self.object_vocabulary = self._load_object_vocabulary()

        # ---- Vgent model ---------------------------------------------------
        # Load vLLM engine settings from config (pragya vs utd)
        from lib.mllm.core.config_loader import load_config, get_vllm_engine_settings
        vllm_cfg = get_vllm_engine_settings(
            load_config() if not hasattr(self, '_cfg') else self._cfg
        )

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
                self.split = split
                self.tensor_parallel_size = tensor_parallel_size
                self.use_vllm = use_vllm
                self.model_weights_dir = model_weights_dir
                # RAG-specific args kept for Vgent init compatibility
                self.n_retrieval = 20
                self.n_refine = 5
                self.uniform_frame = 450
                # vLLM engine settings (from config vllm: section)
                self.gpu_memory_utilization = vllm_cfg.get("gpu_memory_utilization", 0.90)
                self.max_model_len = vllm_cfg.get("max_model_len", 32768)
                self.max_num_seqs = vllm_cfg.get("max_num_seqs", 32)
                self.enable_chunked_prefill = vllm_cfg.get("enable_chunked_prefill", True)
                self.dtype = vllm_cfg.get("dtype", "bfloat16")

        self.args = Args()
        self.vgent = Vgent(self.args)
        self.mode = mode  # "predcls" or "sgdet"
        self.skip_verification: bool = False
        self.print_prompts: bool = False
        logger.info(f"Vgent model loaded. Mode={self.mode}")

    def _log_prompts(
        self,
        video_id: str,
        batch_prompts: List[Dict[str, Any]],
        tag: str = "",
    ) -> None:
        """Log all constructed prompts for a video when ``--print_prompts`` is set."""
        if not self.print_prompts:
            return
        label = f"[{video_id}]" + (f"[{tag}]" if tag else "")
        logger.info(f"{'='*80}")
        logger.info(f"{label} {len(batch_prompts)} prompts")
        logger.info(f"{'='*80}")
        for i, p in enumerate(batch_prompts):
            vi = p.get("video_inputs", [])
            shape_str = ""
            if vi and hasattr(vi[0], "shape"):
                shape_str = f"  [tensor: {tuple(vi[0].shape)}]"
            logger.info(f"--- Prompt {i+1}/{len(batch_prompts)}{shape_str} ---")
            logger.info(p.get("text", ""))
        logger.info(f"{'='*80}")

    # ------------------------------------------------------------------
    # Object vocabulary (from AG annotations/object_classes.txt)
    # ------------------------------------------------------------------

    def _load_object_vocabulary(self) -> List[str]:
        """Load the canonical AG object class names from
        ``annotations/object_classes.txt`` and apply the standard
        label normalizations (matching ``VideoAGLoader``).

        Returns a list of class names (excluding "background").
        """
        obj_cls_path = self.ag_root_directory / "annotations" / "object_classes.txt"
        if not obj_cls_path.exists():
            logger.warning(
                f"object_classes.txt not found at {obj_cls_path} — "
                f"object vocabulary will be empty"
            )
            return []

        classes = ["background"]
        with open(obj_cls_path, "r", encoding="utf-8") as f:
            for line in f:
                classes.append(line.strip("\n"))

        # Label normalizations (same as VideoAGLoader.fetch_object_classes)
        if len(classes) > 31:
            classes[9] = "closet/cabinet"
            classes[11] = "cup/glass/bottle"
            classes[23] = "paper/notebook"
            classes[24] = "phone/camera"
            classes[31] = "sofa/couch"

        # Drop "background" for prompting purposes
        vocab = [c for c in classes if c != "background"]
        logger.info(f"Loaded AG object vocabulary: {len(vocab)} classes")
        return vocab

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_objects_from_frame(frame_data) -> Set[str]:
        """Return the set of object labels present in a single frame."""
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
        for frame_stem, frame_data in bbox_frames.items():
            all_objects.update(self._extract_objects_from_frame(frame_data))
        return all_objects

    def get_missing_objects_per_frame(
        self,
        video_data: Dict[str, Any],
        video_objects: Set[str],
    ) -> Dict[str, Set[str]]:
        """For every annotated frame, compute objects present in the
        video but absent in that particular frame.

        Returns ``{frame_stem: set_of_missing_object_labels}``.
        """
        bbox_frames = video_data.get("bbox_frames", {})
        missing_map: Dict[str, Set[str]] = {}
        for frame_stem, frame_data in bbox_frames.items():
            frame_objects = self._extract_objects_from_frame(frame_data)
            missing = video_objects - frame_objects
            if missing:
                missing_map[frame_stem] = missing
        return missing_map

    # ------------------------------------------------------------------
    # SGDet: LLM-based object estimation from captions
    # ------------------------------------------------------------------

    def estimate_objects_from_captions(
        self,
        captions: List[Tuple[int, str]],
        video_inputs,
    ) -> Set[str]:
        """Ask the LLM to estimate which objects are involved in the
        activity, based on captions and the video clip.

        The full AG object vocabulary (``self.object_vocabulary``) is
        included in the prompt as a candidate list so the LLM
        preferentially uses the canonical labels (improves
        intersection-based evaluation in sgdet mode).

        Returns a set of lowercase object name strings.
        """
        if captions:
            ordered = sorted(captions, key=lambda s: s[0])
            lines = [f"[Frame {idx}] {text}" for idx, text in ordered]
            caption_ctx = (
                "The following captions describe what happens in this video:\n"
                + "\n".join(lines)
            )
        else:
            caption_ctx = "(No captions available for this video.)"

        if self.object_vocabulary:
            candidate_str = ", ".join(f'"{o}"' for o in self.object_vocabulary)
        else:
            candidate_str = "(No candidate list available.)"

        prompt = AG_OBJECT_ESTIMATION_PROMPT.format(
            caption_context=caption_ctx,
            candidate_objects=candidate_str,
        )

        try:
            response = self.vgent.model.mllm_response(
                prompt, video_inputs, max_new_tokens=256,
            )
            response = response.strip()
        except Exception as e:
            logger.error(f"Object estimation LLM error: {e}")
            return set()

        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse object estimation response: {response[:200]}")
                    return set()
            else:
                logger.warning(f"No JSON list found in object estimation response: {response[:200]}")
                return set()

        if not isinstance(parsed, list):
            logger.warning(f"Object estimation returned non-list: {type(parsed)}")
            return set()

        objects = {str(o).strip().lower() for o in parsed if isinstance(o, str) and str(o).strip()}
        logger.info(f"LLM estimated {len(objects)} objects: {sorted(objects)}")
        return objects

    def get_objects_for_mode(
        self,
        video_data: Dict[str, Any],
        captions: List[Tuple[int, str]],
        video_inputs,
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """Return the object set and estimation metadata.

        - **predcls**: annotation objects (no estimation needed)
        - **sgdet**: LLM-estimated objects ∩ AG vocabulary, with
          discriminative Yes/No verification scores

        Returns ``(object_set, estimation_meta)`` where
        ``estimation_meta`` is empty for predcls.
        """
        annotated_objects = self.get_video_objects(video_data)

        if self.mode == "predcls":
            return annotated_objects, {}

        # --- sgdet ---
        estimated_objects = self.estimate_objects_from_captions(
            captions, video_inputs,
        )

        # Intersect with the canonical AG vocabulary (NOT with GT annotations)
        vocab_set = {v.lower() for v in self.object_vocabulary}
        vocab_filtered = estimated_objects & vocab_set
        logger.info(
            f"[SGDet] raw_estimated={len(estimated_objects)}, "
            f"vocab_filtered={len(vocab_filtered)}, "
            f"annotated(ref)={len(annotated_objects)}: {sorted(vocab_filtered)}"
        )

        # Verify each object via Yes/No classification
        object_scores = self.verify_objects_batch(
            vocab_filtered, video_inputs, captions=captions,
        )

        estimation_meta = {
            "annotated_objects": sorted(annotated_objects),
            "raw_estimated": sorted(estimated_objects),
            "vocab_filtered": sorted(vocab_filtered),
            "object_scores": object_scores,
            "video_objects": sorted(vocab_filtered),
        }

        return vocab_filtered, estimation_meta

    # ------------------------------------------------------------------
    # Discriminative object verification
    # ------------------------------------------------------------------

    def verify_objects_batch(
        self,
        objects: Set[str],
        video_inputs,
        captions: Optional[List[Tuple[int, str]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """For each estimated object, ask the VLM:
            'Is there a {object} in this scene? Answer only Yes or No.'

        Uses ``mllm_yes_no_batch()`` for batched single-token inference
        with logprob extraction.

        If *captions* are provided, the caption context (the same one
        used during object estimation) is prepended to each query so the
        VLM is grounded in the same information.

        Returns ``{"cup": {"answer": "Yes", "yes_prob": 0.92, ...}, ...}``
        """
        if not objects:
            return {}

        # Build caption context prefix (matches estimate_objects_from_captions)
        caption_prefix = ""
        if captions:
            ordered = sorted(captions, key=lambda s: s[0])
            lines = [f"[Frame {idx}] {text}" for idx, text in ordered]
            caption_prefix = (
                "The following captions describe what happens in this video:\n"
                + "\n".join(lines)
                + "\n\n"
            )

        prompts = []
        obj_list = sorted(objects)
        for obj in obj_list:
            prompts.append({
                "text": (
                    f"{caption_prefix}"
                    f"Is there a {obj} in this scene? "
                    f"Answer only Yes or No."
                ),
                "video_inputs": video_inputs,
            })

        try:
            results = self.vgent.model.mllm_yes_no_batch(prompts)
        except Exception as e:
            logger.error(f"verify_objects_batch error: {e}")
            results = [
                {"answer": "No", "yes_prob": 0.0, "logprob": 0.0}
                for _ in obj_list
            ]

        scores = {}
        for obj, res in zip(obj_list, results):
            scores[obj] = res
        logger.info(
            f"[Object Verification] "
            + ", ".join(f"{o}={s['yes_prob']:.3f}" for o, s in scores.items())
        )
        return scores

    # ------------------------------------------------------------------
    # Discriminative relationship verification – composable helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rel_verification_prompts(
        parsed: Dict[str, Any],
        object_name: str,
        video_inputs,
        context_prefix: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Any, str]], Dict[str, Any]]:
        """Build Yes/No prompts for one parsed relationship prediction.

        Returns ``(prompts, keys, default_scored)`` where:
        - *prompts*: list of prompt dicts ready for ``mllm_yes_no_batch``
        - *keys*: parallel list of ``(rel_type, index_or_None, label)``
        - *default_scored*: the scored dict initialised to 0.0
        """
        prompts: List[Dict[str, Any]] = []
        keys: List[Tuple[str, Any, str]] = []

        att_label = parsed.get("attention", "unknown")
        if att_label and att_label != "unknown":
            prompts.append({
                "text": (
                    f"{context_prefix}"
                    f"At the target frame (the first frame of this clip), "
                    f"is the person {att_label.replace('_', ' ')} "
                    f"the {object_name}? Answer only Yes or No."
                ),
                "video_inputs": video_inputs,
            })
            keys.append(("attention", None, att_label))

        cont_labels = parsed.get("contacting", ["unknown"])
        for i, label in enumerate(cont_labels):
            if label and label != "unknown":
                prompts.append({
                    "text": (
                        f"{context_prefix}"
                        f"At the target frame (the first frame of this clip), "
                        f"is the person "
                        f"{label.replace('_', ' ')} the {object_name}? "
                        f"Answer only Yes or No."
                    ),
                    "video_inputs": video_inputs,
                })
                keys.append(("contacting", i, label))

        spa_labels = parsed.get("spatial", ["unknown"])
        for i, label in enumerate(spa_labels):
            if label and label != "unknown":
                prompts.append({
                    "text": (
                        f"{context_prefix}"
                        f"At the target frame (the first frame of this clip), "
                        f"is the {object_name} "
                        f"{label.replace('_', ' ')} the person? "
                        f"Answer only Yes or No."
                    ),
                    "video_inputs": video_inputs,
                })
                keys.append(("spatial", i, label))

        default_scored = {
            "attention": {"label": att_label, "yes_prob": 0.0},
            "contacting": [{"label": l, "yes_prob": 0.0} for l in cont_labels],
            "spatial": [{"label": l, "yes_prob": 0.0} for l in spa_labels],
        }
        return prompts, keys, default_scored

    @staticmethod
    def _apply_rel_verification_results(
        keys: List[Tuple[str, Any, str]],
        results: List[Dict[str, Any]],
        default_scored: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map batch results back onto a scored dict."""
        scored = default_scored  # already a fresh copy per call
        for (rel_type, idx, label), res in zip(keys, results):
            if rel_type == "attention":
                scored["attention"] = {"label": label, "yes_prob": res["yes_prob"]}
            elif rel_type == "contacting":
                scored["contacting"][idx] = {"label": label, "yes_prob": res["yes_prob"]}
            elif rel_type == "spatial":
                scored["spatial"][idx] = {"label": label, "yes_prob": res["yes_prob"]}
        return scored

    # convenience wrapper (still useful for the sequential process_video path)
    def verify_relationships_batch(
        self,
        parsed: Dict[str, Any],
        object_name: str,
        video_inputs,
        context_prefix: str = "",
    ) -> Dict[str, Any]:
        """Verify each predicted relationship via Yes/No – **one query at a
        time**.  For bulk verification across many queries, use
        ``verify_all_relationships_bulk()`` instead.
        """
        prompts, keys, default_scored = self._build_rel_verification_prompts(
            parsed, object_name, video_inputs, context_prefix,
        )
        if not prompts:
            return default_scored

        try:
            results = self.vgent.model.mllm_yes_no_batch(prompts)
        except Exception as e:
            logger.error(f"verify_relationships_batch error: {e}")
            results = [{"answer": "No", "yes_prob": 0.0, "logprob": 0.0}] * len(prompts)

        return self._apply_rel_verification_results(keys, results, default_scored)

    # ------------------------------------------------------------------
    # Batch chunking helpers
    # ------------------------------------------------------------------

    BATCH_CHUNK_SIZE = 64  # max prompts per vLLM generate() call (== vllm.max_num_seqs)

    def _chunked_batch_response(
        self, prompts: List[Dict[str, Any]],
        chunk_size: int = BATCH_CHUNK_SIZE,
    ) -> List[str]:
        """Split large batches into sub-batches for mllm_batch_response."""
        if not prompts:
            return []
        results: List[str] = []
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i + chunk_size]
            logger.info(
                f"  [chunk {i // chunk_size + 1}/"
                f"{(len(prompts) + chunk_size - 1) // chunk_size}] "
                f"{len(chunk)} prompts ..."
            )
            results.extend(self.vgent.model.mllm_batch_response(chunk))
        return results

    def _chunked_yes_no_batch(
        self, prompts: List[Dict[str, Any]],
        chunk_size: int = BATCH_CHUNK_SIZE,
    ) -> List[Dict[str, Any]]:
        """Split large batches into sub-batches for mllm_yes_no_batch."""
        if not prompts:
            return []
        results: List[Dict[str, Any]] = []
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i + chunk_size]
            logger.info(
                f"  [yes/no chunk {i // chunk_size + 1}/"
                f"{(len(prompts) + chunk_size - 1) // chunk_size}] "
                f"{len(chunk)} prompts ..."
            )
            results.extend(self.vgent.model.mllm_yes_no_batch(chunk))
        return results

    # ------------------------------------------------------------------
    # Default scored dict (when verification is skipped)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_scored_from_parsed(
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a scored dict that trusts the initial prediction.

        Used when ``--skip-verification`` is set.  Each predicted label
        gets ``yes_prob=1.0`` so downstream consumers can treat them
        identically to verified predictions.
        """
        att_label = parsed.get("attention", "unknown")
        cont_labels = parsed.get("contacting", ["unknown"])
        spa_labels = parsed.get("spatial", ["unknown"])
        return {
            "attention": {"label": att_label, "yes_prob": 1.0},
            "contacting": [{"label": l, "yes_prob": 1.0} for l in cont_labels],
            "spatial": [{"label": l, "yes_prob": 1.0} for l in spa_labels],
        }

    # ------------------------------------------------------------------
    # Bulk relationship verification (one LLM call for an entire video)
    # ------------------------------------------------------------------

    def verify_all_relationships_bulk(
        self,
        entries: List[Tuple[Dict[str, Any], str, Any, str]],
    ) -> List[Dict[str, Any]]:
        """Verify ALL relationship predictions in **one** batched LLM call.

        Parameters
        ----------
        entries : list of ``(parsed, object_name, video_inputs, context_prefix)``
            One tuple per query (across all frames).

        Returns
        -------
        list of scored dicts, one per entry, in the same order.
        """
        # 1. Build every prompt + track slice boundaries
        all_prompts: List[Dict[str, Any]] = []
        meta: List[Tuple[List[Tuple[str, Any, str]], Dict[str, Any], int]] = []

        for parsed, obj_name, vid_in, ctx in entries:
            prompts, keys, default_scored = self._build_rel_verification_prompts(
                parsed, obj_name, vid_in, ctx,
            )
            meta.append((keys, default_scored, len(prompts)))
            all_prompts.extend(prompts)

        if not all_prompts:
            return [m[1] for m in meta]  # all defaults

        logger.info(
            f"[Bulk Rel Verification] {len(entries)} queries -> "
            f"{len(all_prompts)} Yes/No prompts"
        )

        # 2. Chunked batched call
        try:
            all_results = self._chunked_yes_no_batch(all_prompts)
        except Exception as e:
            logger.error(f"verify_all_relationships_bulk error: {e}")
            all_results = [
                {"answer": "No", "yes_prob": 0.0, "logprob": 0.0}
            ] * len(all_prompts)

        # 3. Redistribute
        scored_list: List[Dict[str, Any]] = []
        offset = 0
        for keys, default_scored, count in meta:
            chunk = all_results[offset:offset + count]
            offset += count
            if chunk:
                scored_list.append(
                    self._apply_rel_verification_results(keys, chunk, default_scored)
                )
            else:
                scored_list.append(default_scored)

        return scored_list

    # ------------------------------------------------------------------
    # Frame / clip loading
    # ------------------------------------------------------------------

    def get_annotated_frames(self, video_id: str) -> List[int]:
        video_dir = self.frames_annotated_dir / video_id
        if not video_dir.exists():
            return []
        frames = []
        for fn in os.listdir(video_dir):
            if fn.endswith(".png"):
                m = re.search(r"(\d+)", fn)
                if m:
                    frames.append(int(m.group(1)))
        return sorted(frames)

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
        """Load a list of frame image paths as a single video tensor."""
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
            tensor, self.args.fps, total_pixels=self.args.total_pixels,
        )
        return [video_tensor]

    def _load_clip_around_frame(
        self,
        frame_idx: int,
        frame_map: Dict[int, str],
        video_id: str,
        window: int = 15,
    ):
        """Load a short clip centred on *frame_idx* (±window frames)."""
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

    def _prepend_target_frame(
        self,
        frame_stem: str,
        video_id: str,
        base_tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        """Load the annotated frame and prepend it to a video tensor.

        Returns a tensor of shape ``(T+1, C, H, W)`` where the first
        frame is the target annotated frame.  Falls back to the
        unmodified *base_tensor* if the frame cannot be loaded.
        """
        frame_path = self.frames_annotated_dir / video_id / frame_stem
        if not frame_path.exists():
            frame_path = self.frames_dir / video_id / frame_stem
        if not frame_path.exists():
            logger.warning(
                f"Target frame not found: {frame_stem} for {video_id}, "
                f"using base tensor as-is"
            )
            return base_tensor

        from PIL import Image
        img = Image.open(frame_path).convert("RGB")
        frame_tensor = transforms.PILToTensor()(img).float().unsqueeze(0)

        if frame_tensor.shape[-2:] != base_tensor.shape[-2:]:
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor,
                size=base_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return torch.cat([frame_tensor, base_tensor], dim=0)

    def _build_annotated_context(
        self,
        bbox_frames: Dict[str, Any],
        video_id: str,
        max_context_frames: int = 15,
    ) -> "torch.Tensor":
        """Load annotated frames into a single context tensor (subsampled to max_context_frames).

        Returns a tensor of shape ``(N_annotated, C, H, W)`` built from
        ``frames_annotated/`` (fallback ``frames/``) for each frame_stem
        in *bbox_frames*.  Returns ``None`` if no frames can be loaded.

        The tensor is downscaled via ``resize_video`` (same as
        ``load_video_clip``) to keep visual-token counts within the
        vLLM KV-cache budget.
        """
        from PIL import Image
        frame_tensors = []

        frame_stems = sorted(bbox_frames.keys())
        if len(frame_stems) > max_context_frames:
            indices = np.linspace(0, len(frame_stems) - 1, max_context_frames, dtype=int)
            frame_stems = [frame_stems[i] for i in indices]

        for frame_stem in frame_stems:
            path = self.frames_annotated_dir / video_id / frame_stem
            if not path.exists():
                path = self.frames_dir / video_id / frame_stem
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            frame_tensors.append(
                transforms.PILToTensor()(img).float().unsqueeze(0)
            )
        if not frame_tensors:
            return None
        # Resize all to match the first frame's spatial dims
        h, w = frame_tensors[0].shape[-2:]
        resized = []
        for ft in frame_tensors:
            if ft.shape[-2:] != (h, w):
                ft = torch.nn.functional.interpolate(
                    ft, size=(h, w), mode="bilinear", align_corners=False,
                )
            resized.append(ft)
        context = torch.cat(resized, dim=0)

        # ---- Downscale to stay within the vLLM KV-cache budget ----------
        # Without this, raw HD frames (e.g. 1920×1080) produce too many
        # visual tokens per prompt, causing the vLLM scheduler to OOM
        # and hang at "Processed prompts: 0%".
        #
        # We use a HIGHER pixel budget than load_video_clip (4×) because
        # the context frames are used for relationship reasoning and need
        # enough spatial detail to identify objects.  The cap (512k) keeps
        # things safe for 80 GB A100s with max_model_len=32768.
        context_total_pixels = min(self.args.total_pixels * 4, 512_000)
        orig_shape = context.shape
        context, _ = resize_video(
            context, self.args.fps,
            total_pixels=context_total_pixels,
        )
        if context.shape != orig_shape:
            logger.info(
                f"[{video_id}] Downscaled annotated context: "
                f"{orig_shape} -> {context.shape}"
            )
        return context

    def _build_query_context_map(
        self,
        bbox_frames: Dict[str, Any],
        video_id: str,
        annotated_context: "torch.Tensor",
    ) -> Dict[str, "torch.Tensor"]:
        """Precompute one ``[target_frame + annotated_context]`` tensor
        per unique frame_stem.

        All objects queried on the same frame share the same tensor
        reference (no per-query copies).
        """
        context_map: Dict[str, torch.Tensor] = {}
        for frame_stem in sorted(bbox_frames.keys()):
            prepended = self._prepend_target_frame(
                frame_stem, video_id, annotated_context,
            )
            context_map[frame_stem] = prepended
        logger.info(
            f"[{video_id}] Built query context map: "
            f"{len(context_map)} unique frames, "
            f"annotated context shape={annotated_context.shape}"
        )
        return context_map

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_relationship_response(response: Optional[str]) -> Dict[str, Any]:
        """
        Parse a JSON response from the LLM and validate that each
        relationship label belongs to its allowed set.

        Returns::

            {
                "attention": str,            # single label or "unknown"
                "contacting": list[str],     # one-or-more validated labels
                "spatial": list[str],         # one-or-more validated labels
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

        if not isinstance(parsed, dict):
            return default

        result: Dict[str, Any] = dict(default)

        # --- attention (single label) ---
        att = parsed.get("attention", "")
        if isinstance(att, list):
            att = att[0] if att else ""
        att = str(att).strip().lower().replace(" ", "_")
        if att in ATTENTION_RELATIONSHIPS:
            result["attention"] = att

        # --- contacting (multi-label) ---
        raw_cont = parsed.get("contacting", [])
        if isinstance(raw_cont, str):
            raw_cont = [raw_cont]
        valid_cont = [
            str(c).strip().lower().replace(" ", "_")
            for c in raw_cont
            if str(c).strip().lower().replace(" ", "_") in CONTACTING_RELATIONSHIPS
        ]
        if valid_cont:
            result["contacting"] = valid_cont

        # --- spatial (multi-label) ---
        raw_spa = parsed.get("spatial", [])
        if isinstance(raw_spa, str):
            raw_spa = [raw_spa]
        valid_spa = [
            str(s).strip().lower().replace(" ", "_")
            for s in raw_spa
            if str(s).strip().lower().replace(" ", "_") in SPATIAL_RELATIONSHIPS
        ]
        if valid_spa:
            result["spatial"] = valid_spa

        return result

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def process_video(self, video_id: str):
        """Process a single video end-to-end."""
        ...

    # ------------------------------------------------------------------
    # Run loop (identical across all subclasses)
    # ------------------------------------------------------------------

    def run(
        self,
        limit: Optional[int] = None,
        video_id: Optional[str] = None,
        randomize: bool = False,
        video_list: Optional[str] = None,
    ):
        """Process videos (optionally filtered by split, limit, or single ID).

        Parameters
        ----------
        limit : int, optional
            Process at most this many videos.
        video_id : str, optional
            Process only this single video ID (bypasses split/limit/shuffle).
        randomize : bool
            If True, shuffle video IDs with a time-based seed so multiple
            GPUs process different videos concurrently.  Default is False
            (deterministic sorted order).
        video_list : str, optional
            Text file with one video stem per line (e.g. the locked
            worldbbox test split).  Applied after split filtering.
        """
        # ---- Single-video shortcut ------------------------------------------
        if video_id:
            logger.info(f"Processing single video: {video_id}")
            # Dispatch to fast path when --fast is set and the subclass provides it
            if getattr(self, 'fast_mode', False) and hasattr(self, 'process_video_fast'):
                process_fn = self.process_video_fast
            else:
                process_fn = self.process_video
            try:
                process_fn(video_id)
            except Exception as e:
                logger.error(f"[{video_id}] Error processing: {e}")
                import traceback
                logger.error(traceback.format_exc())
            return

        # ---- Full batch processing ------------------------------------------
        if not self.frames_annotated_dir.exists():
            logger.error(
                f"Annotated frames dir not found: {self.frames_annotated_dir}"
            )
            return

        video_ids = sorted(
            d for d in os.listdir(self.frames_annotated_dir)
            if (self.frames_annotated_dir / d).is_dir()
        )

        # ---- Split filtering (skipped when an explicit video list is given) --
        split = None if video_list else self.args.split
        if split in ("test", "train"):
            split_ids = load_split_video_ids(split)
            video_ids = [v for v in video_ids if Path(v).stem in split_ids]
            logger.info(
                f"Filtered to {len(video_ids)} videos for '{split}' split "
                f"(from video_splits.json)"
            )
        elif split:
            video_ids = [
                v for v in video_ids
                if get_video_belongs_to_split(v) == split
            ]
            logger.info(
                f"Filtered to {len(video_ids)} videos for '{split}' split "
                f"(first-letter logic)"
            )

        if video_list:
            keep = load_video_list(video_list)
            video_ids = [v for v in video_ids if Path(v).stem in keep]
            logger.info(
                f"Filtered to {len(video_ids)} videos from list {video_list} "
                f"({len(keep)} stems)"
            )

        if limit:
            video_ids = video_ids[:limit]

        # Optionally shuffle so multiple GPUs process different videos.
        if randomize:
            import time, random
            _shuffle_seed = time.time_ns()
            logger.info(f"Shuffling {len(video_ids)} video IDs with seed={_shuffle_seed}")
            random.Random(_shuffle_seed).shuffle(video_ids)
        else:
            logger.info(f"Processing {len(video_ids)} video IDs in sorted order")

        # Dispatch to fast path when --fast is set and the subclass provides it
        if getattr(self, 'fast_mode', False) and hasattr(self, 'process_video_fast'):
            process_fn = self.process_video_fast
            mode_label = "FAST"
        else:
            process_fn = self.process_video
            mode_label = "STANDARD"

        logger.info(f"Processing {len(video_ids)} videos [{mode_label}] …")

        pbar = tqdm(video_ids, desc="Videos")
        for i, video_id in enumerate(pbar):
            pbar.set_description(f"Videos [{video_id}]")
            logger.info(f"[{i + 1}/{len(video_ids)}] {video_id}")
            try:
                process_fn(video_id)
            except Exception as e:
                logger.error(f"[{video_id}] Error processing: {e}")
                import traceback
                logger.error(traceback.format_exc())
