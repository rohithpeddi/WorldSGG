#!/usr/bin/env python3
"""
Quick diagnostic to verify the WorldAG dataloader produces correct pair tensors.

Loads a few videos from the test split and prints pair_valid, person_idx,
object_idx, gt_attention, etc. to confirm they are non-zero after the fix.

Usage:
    python test_dataloader_pairs.py --data_path /data/rohith/ag --mode predcls
"""
import logging
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/rohith/ag")
    parser.add_argument("--mode", type=str, default="predcls")
    parser.add_argument("--feature_model", type=str, default="dinov3l")
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--n_videos", type=int, default=5)
    args = parser.parse_args()

    from dataloader.world_ag_dataset import WorldAG, world_collate_fn
    from torch.utils.data import DataLoader

    dataset = WorldAG(
        phase=args.phase,
        data_path=args.data_path,
        mode=args.mode,
        feature_model=args.feature_model,
        include_invisible=True,
        max_objects=64,
    )
    print(f"Dataset: {len(dataset)} videos ({args.phase}, {args.mode})")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=world_collate_fn)

    for i, batch in enumerate(loader):
        if i >= args.n_videos:
            break

        vid = batch.get("video_id", f"vid_{i}")
        T = batch["visual_features"].shape[0]
        K_max = batch["pair_valid"].shape[1]
        N_max = batch["valid_mask"].shape[1]

        pair_valid = batch["pair_valid"]        # (T, K_max)
        person_idx = batch["person_idx"]        # (T, K_max)
        object_idx = batch["object_idx"]        # (T, K_max)
        gt_att = batch["gt_attention"]           # (T, K_max)
        gt_spa = batch["gt_spatial"]             # (T, K_max, 6)
        gt_con = batch["gt_contacting"]          # (T, K_max, 17)
        obj_classes = batch["object_classes"]    # (T, N_max)
        valid_mask = batch["valid_mask"]         # (T, N_max)

        total_valid_pairs = pair_valid.sum().item()
        total_valid_objects = valid_mask.sum().item()

        print(f"\n{'='*60}")
        print(f"Video: {vid} | T={T}, N_max={N_max}, K_max={K_max}")
        print(f"  Total valid objects: {total_valid_objects}")
        print(f"  Total valid pairs:   {total_valid_pairs}")
        print(f"  pair_valid any True: {pair_valid.any().item()}")

        if total_valid_pairs == 0:
            print(f"  *** WARNING: NO VALID PAIRS! ***")
            # Print frame details for debugging
            for t in range(min(T, 3)):
                n_valid = valid_mask[t].sum().item()
                classes = obj_classes[t][valid_mask[t]].tolist()
                print(f"    Frame {t}: N_valid={n_valid}, classes={classes}")
            continue

        # Show per-frame details for first 3 frames
        for t in range(min(T, 3)):
            pv = pair_valid[t]
            k_v = pv.sum().item()
            if k_v == 0:
                continue
            n_valid = valid_mask[t].sum().item()
            classes = obj_classes[t][valid_mask[t]].tolist()
            p_idx = person_idx[t][pv].tolist()
            o_idx = object_idx[t][pv].tolist()
            att = gt_att[t][pv].tolist()
            spa_any = (gt_spa[t][pv].sum(dim=-1) > 0).tolist()
            con_any = (gt_con[t][pv].sum(dim=-1) > 0).tolist()

            print(f"  Frame {t}: K_valid={k_v}, N_valid={n_valid}")
            print(f"    classes[valid]: {classes}")
            print(f"    person_idx: {p_idx}")
            print(f"    object_idx: {o_idx}")
            print(f"    gt_attention: {att}")
            print(f"    gt_spatial_nonzero: {spa_any}")
            print(f"    gt_contacting_nonzero: {con_any}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
