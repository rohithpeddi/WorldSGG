#!/usr/bin/env python3
"""
Compute AG4D Dataset Statistics
=================================

Iterates over the WorldAG dataset (train + test) and computes all
statistics required by sup_ag4d_statistics.tex.

This script works directly with the annotation and feature PKLs to
compute exact counts without building heavy tensors.

Usage:
    python analysis/compute_ag4d_statistics.py \
        --data_path /data/rohith/ag \
        --mode predcls \
        --feature_model dinov2l

Output: formatted statistics printed to stdout + saved to a JSON file.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add project root to sys.path so we can import from dataloader
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.world_ag_dataset import (
    ATTENTION_RELATIONSHIPS,
    CONTACTING_RELATIONSHIPS,
    LABEL_NORMALIZE_MAP,
    OBJECT_CLASSES,
    SPATIAL_RELATIONSHIPS,
    WorldAG,
    _to_short,
    world_collate_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(n) -> str:
    """Format an integer with commas for display."""
    return f"{n:,}"


def fmt_f(n, decimals=1) -> str:
    """Format a float with commas and fixed decimals."""
    return f"{n:,.{decimals}f}"


def load_pkl(path: Path) -> Dict[str, Any]:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Core statistics computation (directly from PKLs, no tensor construction)
# ---------------------------------------------------------------------------

def compute_split_statistics(
    data_path: Path,
    phase: str,
    mode: str,
    feature_model: str,
) -> Dict[str, Any]:
    """Compute all statistics for a single split (train or test).

    Works directly with the annotation and feature PKLs to avoid
    the overhead of building padded tensors.

    Returns a dict with all per-split statistics.
    """
    feat_dir = data_path / "features" / "roi_features" / mode / feature_model / phase
    annot_dir = data_path / "world4d_rel_annotations" / phase

    if not feat_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feat_dir}")
    if not annot_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annot_dir}")

    # Discover common videos
    feat_videos = {p.stem for p in feat_dir.glob("*.pkl")}
    annot_videos = set()
    annot_name_map = {}  # canonical_name -> actual pkl filename
    for p in annot_dir.glob("*.pkl"):
        name = p.name.replace(".pkl", "")
        annot_videos.add(name)
        annot_name_map[name] = p

    # Match with/without .mp4 suffix
    common_videos = set()
    video_to_annot_name = {}
    for vid in feat_videos:
        if vid in annot_videos:
            common_videos.add(vid)
            video_to_annot_name[vid] = vid
        elif vid + ".mp4" in annot_videos:
            common_videos.add(vid)
            video_to_annot_name[vid] = vid + ".mp4"

    logger.info(
        f"[{phase}] Found {len(common_videos)} common videos "
        f"(feat={len(feat_videos)}, annot={len(annot_videos)})"
    )

    # Accumulators
    stats = {
        "phase": phase,
        "num_videos": 0,
        "num_frames": 0,
        "num_objects_observed": 0,       # visible objects across all frames
        "num_objects_unobserved": 0,     # invisible objects across all frames
        "num_objects_total": 0,
        "num_rels_observed": 0,          # relationship triplets for observed pairs
        "num_rels_unobserved": 0,        # relationship triplets for unobserved pairs
        "num_rels_total": 0,
        "num_3d_obbs": 0,               # objects with valid 3D OBBs
        "objects_per_frame_world": [],   # for computing average
        "objects_per_frame_observed": [],
        "rels_per_frame_world": [],
        "rels_per_frame_observed": [],
        # Predicate distributions
        "att_dist": Counter(),
        "spa_dist": Counter(),
        "con_dist": Counter(),
        # Per-object-class count
        "object_class_dist": Counter(),
        # Frames per video
        "frames_per_video": [],
    }

    valid_videos = 0
    skipped_videos = 0

    for vid in tqdm(sorted(common_videos), desc=f"Processing {phase}"):
        try:
            feat_path = feat_dir / f"{vid}.pkl"
            annot_name = video_to_annot_name[vid]
            annot_path = annot_dir / f"{annot_name}.pkl"

            feat_data = load_pkl(feat_path)
            annot_data = load_pkl(annot_path)

            feat_frames = feat_data.get("frames", {})
            annot_frames = annot_data.get("frames", {})

            # Align frames (same logic as WorldAG._align_frames)
            feat_frame_set = set(feat_frames.keys())
            annot_frame_to_key = {}
            for key in annot_frames.keys():
                frame_file = key.split("/")[-1] if "/" in key else key
                annot_frame_to_key[frame_file] = key

            common_frame_names = sorted(
                feat_frame_set & set(annot_frame_to_key.keys())
            )

            if len(common_frame_names) < 2:
                skipped_videos += 1
                continue

            valid_videos += 1
            T = len(common_frame_names)
            stats["frames_per_video"].append(T)

            for frame_file in common_frame_names:
                stats["num_frames"] += 1

                ff = feat_frames[frame_file]
                af_key = annot_frame_to_key.get(frame_file, frame_file)
                af = annot_frames.get(af_key, {})

                # --- Objects ---
                feat_labels = ff.get("labels", [])
                feat_label_ids = ff.get("label_ids", [])
                feat_sources = ff.get("sources", [])
                N = len(feat_labels)

                n_observed = 0
                n_unobserved = 0
                n_with_obb = 0

                # Annotation data
                person_info = af.get("person_info", {})
                object_info_list = af.get("object_info_list", [])

                # Build annot lookup
                annot_by_short_label = {}
                for obj in object_info_list:
                    short = obj.get("label", _to_short(obj.get("class", "")))
                    if short not in annot_by_short_label:
                        annot_by_short_label[short] = obj

                for i in range(N):
                    src = feat_sources[i] if i < len(feat_sources) else "gt"
                    is_visible = src not in ("rag", "gdino", "correction")

                    if is_visible:
                        n_observed += 1
                    else:
                        n_unobserved += 1

                    # Count 3D OBBs
                    label_str = feat_labels[i] if i < len(feat_labels) else ""
                    short_label = _to_short(label_str)

                    # Check if this object has valid 3D corners
                    annot_obj = annot_by_short_label.get(short_label)
                    if annot_obj is not None:
                        c = annot_obj.get("corners_final", None)
                        if c is not None:
                            c = np.asarray(c, dtype=np.float32)
                            if c.shape == (8, 3) and not np.allclose(c, 0):
                                n_with_obb += 1

                    # Object class distribution
                    class_idx = (
                        feat_label_ids[i] if i < len(feat_label_ids) else 0
                    )
                    if 0 < class_idx < len(OBJECT_CLASSES):
                        stats["object_class_dist"][OBJECT_CLASSES[class_idx]] += 1

                # Person 3D OBB
                person_corners = person_info.get("corners_final", None)
                if person_corners is not None:
                    pc = np.asarray(person_corners, dtype=np.float32)
                    if pc.shape == (8, 3) and not np.allclose(pc, 0):
                        n_with_obb += 1

                stats["num_objects_observed"] += n_observed
                stats["num_objects_unobserved"] += n_unobserved
                stats["num_objects_total"] += N
                stats["num_3d_obbs"] += n_with_obb
                stats["objects_per_frame_world"].append(N)
                stats["objects_per_frame_observed"].append(n_observed)

                # --- Relationships ---
                feat_pair_indices = ff.get("pair_indices", [])
                K = len(feat_pair_indices)

                n_rels_observed = 0
                n_rels_unobserved = 0

                for p_label_id, o_label_id in feat_pair_indices:
                    # Find object source to determine observed/unobserved
                    # Map label_id back to position to get source
                    o_pos = None
                    for idx in range(N):
                        lid = (
                            feat_label_ids[idx]
                            if idx < len(feat_label_ids) else 0
                        )
                        if lid == o_label_id:
                            o_pos = idx
                            break

                    if o_pos is not None:
                        src = (
                            feat_sources[o_pos]
                            if o_pos < len(feat_sources) else "gt"
                        )
                        is_obj_visible = src not in ("rag", "gdino", "correction")
                    else:
                        is_obj_visible = True  # default assume visible

                    if is_obj_visible:
                        n_rels_observed += 1
                    else:
                        n_rels_unobserved += 1

                    # Predicate distributions
                    # Get the annotation object for this pair
                    if o_pos is not None:
                        obj_label = (
                            feat_labels[o_pos]
                            if o_pos < len(feat_labels) else ""
                        )
                        short_label = _to_short(obj_label)
                        annot_obj = annot_by_short_label.get(short_label)

                        if annot_obj is not None:
                            att_rels = annot_obj.get(
                                "attention_relationship", []
                            )
                            spa_rels = annot_obj.get(
                                "spatial_relationship", []
                            )
                            con_rels = annot_obj.get(
                                "contacting_relationship", []
                            )

                            for r in att_rels:
                                stats["att_dist"][r] += 1
                            for r in spa_rels:
                                stats["spa_dist"][r] += 1
                            for r in con_rels:
                                stats["con_dist"][r] += 1

                stats["num_rels_observed"] += n_rels_observed
                stats["num_rels_unobserved"] += n_rels_unobserved
                stats["num_rels_total"] += K
                stats["rels_per_frame_world"].append(K)
                stats["rels_per_frame_observed"].append(n_rels_observed)

        except Exception as e:
            logger.warning(f"Error processing video {vid}: {e}")
            skipped_videos += 1
            continue

    stats["num_videos"] = valid_videos

    if skipped_videos > 0:
        logger.info(f"Skipped {skipped_videos} videos in {phase}")

    # Compute averages
    if stats["objects_per_frame_world"]:
        stats["avg_objects_per_frame_world"] = np.mean(
            stats["objects_per_frame_world"]
        )
        stats["avg_objects_per_frame_observed"] = np.mean(
            stats["objects_per_frame_observed"]
        )
    else:
        stats["avg_objects_per_frame_world"] = 0.0
        stats["avg_objects_per_frame_observed"] = 0.0

    if stats["rels_per_frame_world"]:
        stats["avg_rels_per_frame_world"] = np.mean(
            stats["rels_per_frame_world"]
        )
        stats["avg_rels_per_frame_observed"] = np.mean(
            stats["rels_per_frame_observed"]
        )
    else:
        stats["avg_rels_per_frame_world"] = 0.0
        stats["avg_rels_per_frame_observed"] = 0.0

    if stats["frames_per_video"]:
        stats["avg_frames_per_video"] = np.mean(stats["frames_per_video"])
    else:
        stats["avg_frames_per_video"] = 0.0

    return stats


# ---------------------------------------------------------------------------
# Printing / formatting
# ---------------------------------------------------------------------------

def print_separator(char="=", width=72):
    print(char * width)


def print_statistics(train_stats, test_stats):
    """Pretty-print all statistics for the LaTeX file."""

    total_videos = train_stats["num_videos"] + test_stats["num_videos"]
    total_frames = train_stats["num_frames"] + test_stats["num_frames"]
    total_obbs = train_stats["num_3d_obbs"] + test_stats["num_3d_obbs"]
    total_rels = train_stats["num_rels_total"] + test_stats["num_rels_total"]
    total_rels_obs = (
        train_stats["num_rels_observed"] + test_stats["num_rels_observed"]
    )
    total_rels_unobs = (
        train_stats["num_rels_unobserved"] + test_stats["num_rels_unobserved"]
    )
    total_obj = (
        train_stats["num_objects_total"] + test_stats["num_objects_total"]
    )
    total_obj_obs = (
        train_stats["num_objects_observed"] + test_stats["num_objects_observed"]
    )
    total_obj_unobs = (
        train_stats["num_objects_unobserved"]
        + test_stats["num_objects_unobserved"]
    )

    # Overall averages (weighted by frame count)
    all_obj_world = (
        train_stats["objects_per_frame_world"]
        + test_stats["objects_per_frame_world"]
    )
    all_obj_obs = (
        train_stats["objects_per_frame_observed"]
        + test_stats["objects_per_frame_observed"]
    )
    all_rels_world = (
        train_stats["rels_per_frame_world"]
        + test_stats["rels_per_frame_world"]
    )
    all_rels_obs = (
        train_stats["rels_per_frame_observed"]
        + test_stats["rels_per_frame_observed"]
    )

    avg_obj_world = np.mean(all_obj_world) if all_obj_world else 0.0
    avg_obj_obs = np.mean(all_obj_obs) if all_obj_obs else 0.0
    avg_rels_world = np.mean(all_rels_world) if all_rels_world else 0.0
    avg_rels_obs = np.mean(all_rels_obs) if all_rels_obs else 0.0

    print_separator()
    print("ActionGenome4D Dataset Statistics")
    print_separator()

    # -----------------------------------------------------------------------
    # Table 1: AG vs AG4D (Scale section)
    # -----------------------------------------------------------------------
    print("\n[Table 1: AG vs AG4D — Scale section]")
    print(f"  Videos                  : {fmt(total_videos)}")
    print(
        f"  Train / Test videos     : "
        f"{fmt(train_stats['num_videos'])} / {fmt(test_stats['num_videos'])}"
    )
    print(f"  Annotated frames        : {fmt(total_frames)}")

    # -----------------------------------------------------------------------
    # Geometric Annotation Statistics
    # -----------------------------------------------------------------------
    print("\n[Geometric Annotation Statistics]")
    print(f"  Total 3D OBBs           : {fmt(total_obbs)}")

    # -----------------------------------------------------------------------
    # Semantic Annotation Statistics
    # -----------------------------------------------------------------------
    print("\n[Semantic Annotation Statistics]")
    print(f"  Total relationship triplets (AG4D)    : {fmt(total_rels)}")
    print(f"  Observed pairs (both visible)         : {fmt(total_rels_obs)}")
    print(
        f"  Unobserved pairs (≥1 invisible, new)  : {fmt(total_rels_unobs)}"
    )
    print(
        f"  Avg. objects per frame (world state)   : {fmt_f(avg_obj_world)}"
    )
    print(
        f"  Avg. objects per frame (observed only) : {fmt_f(avg_obj_obs)}"
    )
    print(
        f"  Avg. rels per frame (world state)      : {fmt_f(avg_rels_world)}"
    )
    print(
        f"  Avg. rels per frame (observed only)    : {fmt_f(avg_rels_obs)}"
    )

    # -----------------------------------------------------------------------
    # Table 2: Train/Test Split Statistics
    # -----------------------------------------------------------------------
    print("\n[Table 2: Train/Test Split Statistics]")
    header = f"  {'Statistic':<35s} {'Train':>12s} {'Test':>12s}"
    print(header)
    print("  " + "-" * 60)

    rows = [
        ("Videos", train_stats["num_videos"], test_stats["num_videos"]),
        ("Annotated frames", train_stats["num_frames"], test_stats["num_frames"]),
        (
            "Object instances (observed)",
            train_stats["num_objects_observed"],
            test_stats["num_objects_observed"],
        ),
        (
            "Object instances (unobserved)",
            train_stats["num_objects_unobserved"],
            test_stats["num_objects_unobserved"],
        ),
        (
            "Object instances (total)",
            train_stats["num_objects_total"],
            test_stats["num_objects_total"],
        ),
        (
            "Rel triplets (observed)",
            train_stats["num_rels_observed"],
            test_stats["num_rels_observed"],
        ),
        (
            "Rel triplets (unobserved)",
            train_stats["num_rels_unobserved"],
            test_stats["num_rels_unobserved"],
        ),
        (
            "Rel triplets (total)",
            train_stats["num_rels_total"],
            test_stats["num_rels_total"],
        ),
        ("3D OBBs", train_stats["num_3d_obbs"], test_stats["num_3d_obbs"]),
    ]
    for label, tr, te in rows:
        print(f"  {label:<35s} {fmt(tr):>12s} {fmt(te):>12s}")

    # -----------------------------------------------------------------------
    # Predicate Distributions
    # -----------------------------------------------------------------------
    print("\n[Predicate Label Distributions (Train + Test)]")

    # Merge train + test distributions
    att_dist = train_stats["att_dist"] + test_stats["att_dist"]
    spa_dist = train_stats["spa_dist"] + test_stats["spa_dist"]
    con_dist = train_stats["con_dist"] + test_stats["con_dist"]

    print("\n  Attention Predicate Distribution:")
    att_total = sum(att_dist.values())
    for r in ATTENTION_RELATIONSHIPS:
        c = att_dist.get(r, 0)
        pct = 100.0 * c / att_total if att_total > 0 else 0.0
        print(f"    {r:<25s} {fmt(c):>12s}  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<25s} {fmt(att_total):>12s}")

    print("\n  Spatial Predicate Distribution:")
    spa_total = sum(spa_dist.values())
    for r in SPATIAL_RELATIONSHIPS:
        c = spa_dist.get(r, 0)
        pct = 100.0 * c / spa_total if spa_total > 0 else 0.0
        print(f"    {r:<25s} {fmt(c):>12s}  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<25s} {fmt(spa_total):>12s}")

    print("\n  Contacting Predicate Distribution:")
    con_total = sum(con_dist.values())
    for r in CONTACTING_RELATIONSHIPS:
        c = con_dist.get(r, 0)
        pct = 100.0 * c / con_total if con_total > 0 else 0.0
        print(f"    {r:<25s} {fmt(c):>12s}  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<25s} {fmt(con_total):>12s}")

    # -----------------------------------------------------------------------
    # Object Class Distribution
    # -----------------------------------------------------------------------
    print("\n[Object Class Distribution (Train + Test)]")
    obj_dist = train_stats["object_class_dist"] + test_stats["object_class_dist"]
    obj_total = sum(obj_dist.values())
    for cls in OBJECT_CLASSES[1:]:  # skip __background__
        c = obj_dist.get(cls, 0)
        pct = 100.0 * c / obj_total if obj_total > 0 else 0.0
        print(f"    {cls:<25s} {fmt(c):>12s}  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<25s} {fmt(obj_total):>12s}")

    # -----------------------------------------------------------------------
    # Additional useful statistics
    # -----------------------------------------------------------------------
    print("\n[Additional Statistics]")
    print(
        f"  Avg. frames per video (train)  : "
        f"{fmt_f(train_stats['avg_frames_per_video'])}"
    )
    print(
        f"  Avg. frames per video (test)   : "
        f"{fmt_f(test_stats['avg_frames_per_video'])}"
    )

    all_fpv = train_stats["frames_per_video"] + test_stats["frames_per_video"]
    print(
        f"  Avg. frames per video (all)    : "
        f"{fmt_f(np.mean(all_fpv)) if all_fpv else 'N/A'}"
    )
    print(
        f"  Min frames per video           : "
        f"{min(all_fpv) if all_fpv else 'N/A'}"
    )
    print(
        f"  Max frames per video           : "
        f"{max(all_fpv) if all_fpv else 'N/A'}"
    )

    print_separator()

    # -----------------------------------------------------------------------
    # LaTeX-ready values
    # -----------------------------------------------------------------------
    print("\n[LaTeX-Ready Replacement Values]")
    print("Copy these into sup_ag4d_statistics.tex:\n")

    print("% Table 1: AG vs AG4D — Scale section")
    print(f"% Videos:                {fmt(total_videos)}")
    print(
        f"% Train / Test videos:   "
        f"{fmt(train_stats['num_videos'])} / {fmt(test_stats['num_videos'])}"
    )
    print(f"% Annotated frames:      {fmt(total_frames)}")

    print("\n% Geometric stats")
    print(f"% Total 3D OBBs:        {fmt(total_obbs)}")

    print("\n% Semantic stats")
    print(f"% Total rel triplets:    {fmt(total_rels)}")
    print(f"% Observed pairs:        {fmt(total_rels_obs)}")
    print(f"% Unobserved pairs:      {fmt(total_rels_unobs)}")
    print(f"% Avg obj/frame (world): {fmt_f(avg_obj_world)}")
    print(f"% Avg obj/frame (obs):   {fmt_f(avg_obj_obs)}")
    print(f"% Avg rel/frame (world): {fmt_f(avg_rels_world)}")
    print(f"% Avg rel/frame (obs):   {fmt_f(avg_rels_obs)}")

    print("\n% Table 2: per-split")
    print(
        f"% Train videos:          {fmt(train_stats['num_videos'])}"
    )
    print(
        f"% Test videos:           {fmt(test_stats['num_videos'])}"
    )
    print(
        f"% Train frames:          {fmt(train_stats['num_frames'])}"
    )
    print(
        f"% Test frames:           {fmt(test_stats['num_frames'])}"
    )
    print(
        f"% Train obj (obs):       {fmt(train_stats['num_objects_observed'])}"
    )
    print(
        f"% Test obj (obs):        {fmt(test_stats['num_objects_observed'])}"
    )
    print(
        f"% Train obj (unobs):     {fmt(train_stats['num_objects_unobserved'])}"
    )
    print(
        f"% Test obj (unobs):      {fmt(test_stats['num_objects_unobserved'])}"
    )
    print(
        f"% Train obj (total):     {fmt(train_stats['num_objects_total'])}"
    )
    print(
        f"% Test obj (total):      {fmt(test_stats['num_objects_total'])}"
    )
    print(
        f"% Train rels (obs):      {fmt(train_stats['num_rels_observed'])}"
    )
    print(
        f"% Test rels (obs):       {fmt(test_stats['num_rels_observed'])}"
    )
    print(
        f"% Train rels (unobs):    {fmt(train_stats['num_rels_unobserved'])}"
    )
    print(
        f"% Test rels (unobs):     {fmt(test_stats['num_rels_unobserved'])}"
    )
    print(
        f"% Train rels (total):    {fmt(train_stats['num_rels_total'])}"
    )
    print(
        f"% Test rels (total):     {fmt(test_stats['num_rels_total'])}"
    )
    print(
        f"% Train OBBs:           {fmt(train_stats['num_3d_obbs'])}"
    )
    print(
        f"% Test OBBs:            {fmt(test_stats['num_3d_obbs'])}"
    )

    print_separator()


def save_results_json(train_stats, test_stats, output_path: Path):
    """Save results to a JSON file for later use."""

    def _clean(stats):
        """Remove large per-frame lists, keep only summaries."""
        out = {}
        skip_keys = {
            "objects_per_frame_world", "objects_per_frame_observed",
            "rels_per_frame_world", "rels_per_frame_observed",
            "frames_per_video",
        }
        for k, v in stats.items():
            if k in skip_keys:
                continue
            if isinstance(v, Counter):
                out[k] = dict(v)
            elif isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    result = {
        "train": _clean(train_stats),
        "test": _clean(test_stats),
        "combined": {
            "total_videos": (
                train_stats["num_videos"] + test_stats["num_videos"]
            ),
            "total_frames": (
                train_stats["num_frames"] + test_stats["num_frames"]
            ),
            "total_3d_obbs": (
                train_stats["num_3d_obbs"] + test_stats["num_3d_obbs"]
            ),
            "total_rels": (
                train_stats["num_rels_total"] + test_stats["num_rels_total"]
            ),
            "total_rels_observed": (
                train_stats["num_rels_observed"]
                + test_stats["num_rels_observed"]
            ),
            "total_rels_unobserved": (
                train_stats["num_rels_unobserved"]
                + test_stats["num_rels_unobserved"]
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved results to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute AG4D dataset statistics for the paper"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/rohith/ag",
        help="Root directory of Action Genome dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="predcls",
        choices=["predcls", "sgdet"],
        help="Mode (predcls or sgdet)",
    )
    parser.add_argument(
        "--feature_model",
        type=str,
        default="dinov2l",
        help="Feature model directory name (e.g. dinov2l, dinov2b)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (default: analysis/ag4d_statistics.json)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if args.output is None:
        output_path = PROJECT_ROOT / "analysis" / "ag4d_statistics.json"
    else:
        output_path = Path(args.output)

    logger.info(f"Data path: {data_path}")
    logger.info(f"Mode: {args.mode}, Feature model: {args.feature_model}")

    # Compute statistics for both splits
    train_stats = compute_split_statistics(
        data_path, "train", args.mode, args.feature_model
    )
    test_stats = compute_split_statistics(
        data_path, "test", args.mode, args.feature_model
    )

    # Print formatted output
    print_statistics(train_stats, test_stats)

    # Save to JSON
    save_results_json(train_stats, test_stats, output_path)


if __name__ == "__main__":
    main()
