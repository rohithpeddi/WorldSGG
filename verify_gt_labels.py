#!/usr/bin/env python3
"""
Verify GT label correctness: check that each paired object's GT relationships
match between the feature PKL labels and annotation PKL.

Shows the full mapping chain:
  pair_indices (class IDs) → positional index → feat_label → short_label
  → annot_obj lookup → relationship strings → encoded labels

Usage:
    python verify_gt_labels.py --data_path /data/rohith/ag --mode predcls
"""
import logging
import sys
import os
import pickle
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Import the same constants as the dataloader
from dataloader.world_ag_dataset import (
    OBJECT_CLASSES, ATTENTION_RELATIONSHIPS, SPATIAL_RELATIONSHIPS,
    CONTACTING_RELATIONSHIPS, LABEL_NORMALIZE_MAP, _to_short,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/rohith/ag")
    parser.add_argument("--mode", type=str, default="predcls")
    parser.add_argument("--feature_model", type=str, default="dinov3l")
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--n_videos", type=int, default=3)
    args = parser.parse_args()

    # Load raw PKLs directly to inspect the data
    feat_dir = Path(args.data_path) / "features" / "roi_features" / "predcls" / args.feature_model / args.phase
    annot_dir = Path(args.data_path) / "world4d_rel_annotations" / args.phase

    print(f"Feature dir: {feat_dir}")
    print(f"Annot dir:   {annot_dir}")

    feat_pkls = sorted(feat_dir.glob("*.pkl"))[:args.n_videos]

    for pkl_path in feat_pkls:
        vid = pkl_path.stem
        print(f"\n{'='*70}")
        print(f"VIDEO: {vid}")
        print(f"{'='*70}")

        with open(pkl_path, "rb") as f:
            feat_data = pickle.load(f)

        # Find annotation
        annot_path = None
        for suffix in ["", ".mp4"]:
            p = annot_dir / f"{vid}{suffix}.pkl"
            if p.exists():
                annot_path = p
                break
        if annot_path is None:
            print(f"  *** Annotation not found for {vid}")
            continue

        with open(annot_path, "rb") as f:
            annot_data = pickle.load(f)

        feat_frames = feat_data.get("frames", {})
        annot_frames = annot_data.get("frames", {})

        # Align frames
        annot_frame_to_key = {}
        for key in annot_frames:
            frame_file = key.split("/")[-1] if "/" in key else key
            annot_frame_to_key[frame_file] = key

        common = sorted(set(feat_frames.keys()) & set(annot_frame_to_key.keys()))
        print(f"  Common frames: {len(common)}")

        # Show first 2 frames in detail
        for frame_file in common[:2]:
            ff = feat_frames[frame_file]
            af_key = annot_frame_to_key[frame_file]
            af = annot_frames[af_key]

            feat_labels = ff.get("labels", [])
            feat_label_ids = ff.get("label_ids", [])
            feat_sources = ff.get("sources", [])
            feat_pairs = ff.get("pair_indices", [])

            person_info = af.get("person_info", {})
            object_info_list = af.get("object_info_list", [])

            print(f"\n  --- Frame: {frame_file} ---")
            print(f"  Feature objects ({len(feat_labels)}):")
            for i, (lbl, lid, src) in enumerate(zip(feat_labels, feat_label_ids, feat_sources)):
                print(f"    pos[{i}]: label='{lbl}', class_id={lid}, source={src}")

            print(f"  Annotation objects ({len(object_info_list)}):")
            for i, obj in enumerate(object_info_list):
                cls = obj.get("class", "?")
                lbl = obj.get("label", _to_short(cls))
                vis = obj.get("visible", True)
                att = obj.get("attention_relationship", [])
                spa = obj.get("spatial_relationship", [])
                con = obj.get("contacting_relationship", [])
                src = obj.get("source", "gt")
                print(f"    annot[{i}]: class='{cls}', label='{lbl}', visible={vis}, source={src}")
                print(f"      att={att}, spa={spa}, con={con}")

            print(f"  Raw pair_indices: {feat_pairs}")

            # Build label_id_to_pos mapping (same as dataloader)
            N = len(feat_labels)
            label_id_to_pos = {}
            for i in range(N):
                lid = feat_label_ids[i] if i < len(feat_label_ids) else 0
                if lid not in label_id_to_pos:
                    label_id_to_pos[lid] = i

            print(f"  label_id_to_pos: {label_id_to_pos}")

            # Build annot lookup (same as dataloader)
            annot_by_short = {}
            for obj in object_info_list:
                short = obj.get("label", _to_short(obj.get("class", "")))
                if short not in annot_by_short:
                    annot_by_short[short] = obj

            print(f"  annot_by_short keys: {list(annot_by_short.keys())}")

            # Trace each pair
            print(f"\n  Pair resolution:")
            for p_lid, o_lid in feat_pairs:
                p_pos = label_id_to_pos.get(p_lid)
                o_pos = label_id_to_pos.get(o_lid)
                p_label = feat_labels[p_pos] if p_pos is not None and p_pos < len(feat_labels) else "?"
                o_label = feat_labels[o_pos] if o_pos is not None and o_pos < len(feat_labels) else "?"
                o_short = _to_short(o_label)
                annot_match = annot_by_short.get(o_short)

                status = "✓ MATCH" if annot_match else "✗ NO MATCH"
                print(f"    pair({p_lid},{o_lid}) → pos({p_pos},{o_pos})")
                print(f"      person='{p_label}', object='{o_label}' → short='{o_short}' → {status}")
                if annot_match:
                    print(f"      GT att={annot_match.get('attention_relationship', [])}")
                    print(f"      GT spa={annot_match.get('spatial_relationship', [])}")
                    print(f"      GT con={annot_match.get('contacting_relationship', [])}")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
