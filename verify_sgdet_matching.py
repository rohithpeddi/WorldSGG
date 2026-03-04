#!/usr/bin/env python3
"""
Verify SGDet GT matching in the dataloader.

Loads raw PKL files (features + annotations) for SGDet and traces:
1. What `detector_found_idx` maps each detection to
2. Whether detector labels match GT labels (mismatches = the bug we fixed)
3. What GT relationships are resolved via detector_found_idx vs label string matching
4. Full dataloader output sanity (pair_valid, GT tensors)

Usage:
    python verify_sgdet_matching.py \
        --data_path /data/rohith/ag \
        --feature_model dinov3l \
        --n_videos 3
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Reuse dataloader label maps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader.world_ag_dataset import (
    LABEL_NORMALIZE_MAP, OBJECT_CLASSES, WorldAG, world_collate_fn,
    _to_short, _attention_label_to_idx,
)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def verify_video_raw(feat_path, annot_path, video_id):
    """Trace GT matching for one video at the raw PKL level."""
    feat_data = load_pkl(feat_path)
    annot_data = load_pkl(annot_path)

    feat_frames = feat_data.get("frames", {})
    annot_frames_raw = annot_data.get("frames", {})

    # Annotation frame keys may be "video_id/frame.png"; extract short name
    # Same logic as WorldAG._align_frames
    annot_short_to_key = {}
    for key in annot_frames_raw:
        short = key.split("/")[-1] if "/" in key else key
        annot_short_to_key[short] = key

    # Find common frames (feature keys are short names like "000001.png")
    common = sorted(set(feat_frames.keys()) & set(annot_short_to_key.keys()))
    if not common:
        print(f"  WARNING: No common frames between features and annotations!")
        print(f"    Feature keys sample: {list(feat_frames.keys())[:3]}")
        print(f"    Annot keys sample:   {list(annot_frames_raw.keys())[:3]}")
        return 0, 0

    total_pairs = 0
    total_mismatches = 0

    for frame_file in common[:5]:  # Show first 5 frames
        ff = feat_frames[frame_file]
        annot_key = annot_short_to_key[frame_file]
        af = annot_frames_raw.get(annot_key, {})

        labels = ff.get("labels", [])
        label_ids = ff.get("label_ids", [])
        sources = ff.get("sources", [])
        detector_found_idx = ff.get("detector_found_idx", [])
        pair_indices = ff.get("pair_indices", [])

        person_info = af.get("person_info", {})
        object_info_list = af.get("object_info_list", [])

        # Build annot_by_short_label (old approach)
        annot_by_short_label = {}
        for obj in object_info_list:
            short = obj.get("label", _to_short(obj.get("class", "")))
            if short not in annot_by_short_label:
                annot_by_short_label[short] = obj

        # Build det_pos_to_annot (new approach)
        det_pos_to_annot = {}
        for k, det_pos in enumerate(detector_found_idx):
            if k == 0:
                det_pos_to_annot[det_pos] = person_info
            elif (k - 1) < len(object_info_list):
                det_pos_to_annot[det_pos] = object_info_list[k - 1]

        print(f"\n  --- Frame: {frame_file} ---")
        print(f"  Detections ({len(labels)}):")
        for i, (lbl, lid, src) in enumerate(zip(labels, label_ids, sources)):
            gt_annot = det_pos_to_annot.get(i)
            if gt_annot is not None:
                if i == detector_found_idx[0] if detector_found_idx else False:
                    gt_class = "person"
                else:
                    gt_class = gt_annot.get("label", gt_annot.get("class", "?"))
                match_str = f"→ GT: {gt_class}"
                if _to_short(lbl) != _to_short(str(gt_class)):
                    match_str += " ⚠️ MISMATCH"
                    total_mismatches += 1
                else:
                    match_str += " ✓"
            else:
                match_str = "→ NO GT MATCH (unmatched detection)"
            print(f"    pos[{i}]: det_label='{lbl}' (id={lid}), src={src}  {match_str}")

        print(f"  GT Annotations ({len(object_info_list)} objects):")
        for j, obj in enumerate(object_info_list):
            att = obj.get("attention_relationship", [])
            spa = obj.get("spatial_relationship", [])
            con = obj.get("contacting_relationship", [])
            print(f"    annot[{j}]: class='{obj.get('class', '?')}', "
                  f"label='{obj.get('label', '?')}', "
                  f"att={att}, spa_cnt={len(spa)}, con_cnt={len(con)}")

        print(f"  detector_found_idx: {detector_found_idx}")
        print(f"  Pairs ({len(pair_indices)}):")

        for pi, (p_lid, o_lid) in enumerate(pair_indices):
            # Find positions
            p_pos = next((i for i, lid in enumerate(label_ids) if lid == p_lid), None)
            o_pos = next((i for i, lid in enumerate(label_ids) if lid == o_lid), None)

            if o_pos is None:
                print(f"    pair[{pi}]: ({p_lid},{o_lid}) → o_pos=None, SKIP")
                continue

            # OLD: label string matching
            old_label = labels[o_pos] if o_pos < len(labels) else ""
            old_short = _to_short(old_label)
            old_annot = annot_by_short_label.get(old_short)
            old_att = old_annot.get("attention_relationship", []) if old_annot else []

            # NEW: detector_found_idx matching
            new_annot = det_pos_to_annot.get(o_pos)
            if new_annot is None:
                # Fallback for supply boxes
                new_annot = annot_by_short_label.get(old_short)
            new_att = new_annot.get("attention_relationship", []) if new_annot else []

            same = "✓ SAME" if old_att == new_att else "⚠️ DIFFERENT"
            total_pairs += 1

            if old_att != new_att:
                print(f"    pair[{pi}]: det[{o_pos}]='{labels[o_pos]}' "
                      f"OLD_att={old_att} vs NEW_att={new_att}  {same}")

    # Count all pairs
    for frame_file in common:
        ff = feat_frames[frame_file]
        total_pairs += len(ff.get("pair_indices", []))

    return total_pairs, total_mismatches


def verify_dataloader(data_path, feature_model, n_videos):
    """Load through the actual dataloader and verify outputs."""
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("DATALOADER OUTPUT VERIFICATION (SGDet)")
    print("=" * 70)

    ds = WorldAG(
        phase="train",
        data_path=data_path,
        feature_model=feature_model,
        mode="sgdet",
        include_invisible=True,
    )

    loader = DataLoader(ds, batch_size=1, collate_fn=world_collate_fn, num_workers=0)

    for idx, batch in enumerate(loader):
        if idx >= n_videos:
            break

        vid = batch["video_id"]
        T = batch["T"]
        N_max = batch["N_max"]
        K_max = batch["K_max"]
        pair_valid = batch["pair_valid"]
        gt_att = batch["gt_attention"]
        gt_spa = batch["gt_spatial"]
        gt_con = batch["gt_contacting"]
        person_idx = batch["person_idx"]
        object_idx = batch["object_idx"]
        obj_classes = batch["object_classes"]
        valid_mask = batch["valid_mask"]

        total_valid_pairs = pair_valid.sum().item()
        total_valid_objs = valid_mask.sum().item()

        print(f"\n{'=' * 60}")
        print(f"Video: {vid} | T={T}, N_max={N_max}, K_max={K_max}")
        print(f"  Total valid objects: {total_valid_objs}")
        print(f"  Total valid pairs:   {total_valid_pairs}")
        print(f"  pair_valid any True: {pair_valid.any().item()}")

        if total_valid_pairs == 0:
            print("  ⚠️ NO VALID PAIRS!")
            continue

        # Check first 3 frames
        for t in range(min(3, T)):
            pv = pair_valid[t]
            K_v = pv.sum().item()
            N_v = valid_mask[t].sum().item()

            if K_v == 0:
                print(f"  Frame {t}: K_valid=0")
                continue

            valid_classes = obj_classes[t][valid_mask[t]].tolist()
            p_idx = person_idx[t][:K_v].tolist()
            o_idx = object_idx[t][:K_v].tolist()
            att_vals = gt_att[t][:K_v].tolist()
            spa_nonzero = [gt_spa[t][k].sum().item() > 0 for k in range(K_v)]
            con_nonzero = [gt_con[t][k].sum().item() > 0 for k in range(K_v)]

            print(f"  Frame {t}: K_valid={K_v}, N_valid={N_v}")
            print(f"    classes[valid]: {valid_classes}")
            print(f"    person_idx: {p_idx}")
            print(f"    object_idx: {o_idx}")
            print(f"    gt_attention: {att_vals}")
            print(f"    gt_spatial_nonzero: {spa_nonzero}")
            print(f"    gt_contacting_nonzero: {con_nonzero}")

            # Flag if any GT is all-zero (possibly broken matching)
            if not any(spa_nonzero):
                print(f"    ⚠️ ALL spatial GT are zero!")
            if not any(con_nonzero):
                print(f"    ⚠️ ALL contacting GT are zero!")


def main():
    parser = argparse.ArgumentParser(description="Verify SGDet GT matching")
    parser.add_argument("--data_path", default="/data/rohith/ag")
    parser.add_argument("--feature_model", default="dinov3l")
    parser.add_argument("--n_videos", type=int, default=50)
    parser.add_argument("--phase", default="train",
                        help="Which split to check (train or test)")
    args = parser.parse_args()

    feat_dir = Path(args.data_path) / "features" / "roi_features" / "sgdet" / args.feature_model / args.phase
    annot_dir = Path(args.data_path) / "world4d_rel_annotations" / args.phase

    print(f"Feature dir: {feat_dir}")
    print(f"Annot dir:   {annot_dir}")

    if not feat_dir.exists():
        print(f"ERROR: Feature dir does not exist: {feat_dir}")
        print("SGDet features may not have been extracted yet.")
        return

    feat_files = sorted(feat_dir.glob("*.pkl"))
    annot_files = {p.stem: p for p in annot_dir.glob("*.pkl")}

    print(f"\nFound {len(feat_files)} feature PKLs, {len(annot_files)} annotation PKLs")

    total_pairs = 0
    total_mismatches = 0

    for fi, feat_path in enumerate(feat_files[:args.n_videos]):
        video_id = feat_path.stem
        annot_path = annot_files.get(f"{video_id}.mp4")

        if annot_path is None:
            print(f"\n{'=' * 70}")
            print(f"VIDEO: {video_id} — NO MATCHING ANNOTATION PKL")
            continue

        print(f"\n{'=' * 70}")
        print(f"VIDEO: {video_id}")
        print(f"{'=' * 70}")

        vp, vm = verify_video_raw(feat_path, annot_path, video_id)
        total_pairs += vp
        total_mismatches += vm

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_pairs} total pairs, {total_mismatches} label mismatches "
          f"(detector label ≠ GT label, fixed by detector_found_idx)")
    print(f"{'=' * 70}")

    # Also run through the actual dataloader
    if feat_dir.exists():
        verify_dataloader(args.data_path, args.feature_model, args.n_videos)


if __name__ == "__main__":
    main()
