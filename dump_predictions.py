#!/usr/bin/env python3
"""
Dump WSGG Predictions from Checkpoints
========================================

Load each checkpoint (epoch >= min_epoch) for a given config, run inference
on the test set, and save per-video prediction PKLs.

The model's forward() returns predictions for ALL T frames with shape
(T, K_max, C). We dump every frame that has valid pairs, matching
predictions with batch GT labels using consistent K_max indexing.

Output structure::

    <output_root>/<mode>/<experiment_name>/epoch_<N>/<video_id>.pkl

Each PKL stores:
    - "video_id": str
    - "frames": dict[frame_idx → per-frame dict with preds + GT]

Usage::

    python dump_predictions.py --config configs/methods/predcls/gl_stgn_predcls_dinov2b.yaml
"""

import gc
import logging
import os
import pickle
import re

import numpy as np
import torch
from tqdm import tqdm

from wsgg_base import load_wsgg_config

logger = logging.getLogger(__name__)


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _to_device(batch, device):
    """Move batch tensors to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def discover_checkpoints(experiment_dir: str, min_epoch: int):
    pattern = re.compile(r"^checkpoint_(\d+)$")
    hits = []
    if not os.path.isdir(experiment_dir):
        return hits
    for name in os.listdir(experiment_dir):
        m = pattern.match(name)
        if m:
            epoch = int(m.group(1))
            ckpt_file = os.path.join(experiment_dir, name, "checkpoint_state.pth")
            if epoch >= min_epoch and os.path.isfile(ckpt_file):
                hits.append((epoch, name))
    return sorted(hits)


def dump_one_epoch(tester, epoch, ckpt_name, experiment_dir, output_dir, device, mode):
    ckpt_path = os.path.join(experiment_dir, ckpt_name, "checkpoint_state.pth")
    logger.info(f"  Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    tester._model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    del checkpoint
    gc.collect()

    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    tester._model.eval()
    test_loader = tester._dataloader_test

    n_saved = 0
    n_skipped = 0
    n_debug = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  Epoch {epoch}", leave=False):
            video_id = batch.get("video_id", f"unknown_{n_saved}")
            out_path = os.path.join(epoch_dir, f"{video_id}.pkl")

            if os.path.exists(out_path):
                n_skipped += 1
                continue

            # Move batch to device
            b = _to_device(batch, device)

            # Call model.forward() directly to get ALL temporal predictions
            # Shape: (T, K_max, C_att), (T, K_max, C_spa), (T, K_max, C_con)
            pred = tester._model.forward(
                visual_features_seq=b["visual_features"],
                corners_seq=b["corners"],
                valid_mask_seq=b["valid_mask"],
                visibility_mask_seq=b["visibility_mask"],
                person_idx_seq=b["person_idx"],
                object_idx_seq=b["object_idx"],
                pair_valid=b["pair_valid"],
                camera_pose_seq=b.get("camera_poses"),
            )

            if pred is None:
                continue

            T = batch["visual_features"].shape[0]

            # Get full temporal predictions: (T, K_max, ...)
            att_dist_all = _to_numpy(pred["attention_distribution"])   # (T, K_max, 3)
            spa_dist_all = _to_numpy(pred["spatial_distribution"])     # (T, K_max, 6)
            con_dist_all = _to_numpy(pred["contacting_distribution"]) # (T, K_max, 17)

            frames_data = {}
            total_valid = 0

            for t in range(T):
                pair_valid = _to_numpy(batch["pair_valid"][t]).astype(bool)
                K_valid = pair_valid.sum()

                if K_valid == 0:
                    continue

                total_valid += K_valid

                frame_data = {
                    # Predictions (K_max, ...)
                    "attention_distribution": att_dist_all[t],
                    "spatial_distribution": spa_dist_all[t],
                    "contacting_distribution": con_dist_all[t],
                    # Pair info (K_max)
                    "person_idx": _to_numpy(batch["person_idx"][t]),
                    "object_idx": _to_numpy(batch["object_idx"][t]),
                    "pair_valid": _to_numpy(batch["pair_valid"][t]),
                    # GT labels (K_max) — same indexing as predictions
                    "gt_attention": _to_numpy(batch["gt_attention"][t]),
                    "gt_spatial": _to_numpy(batch["gt_spatial"][t]),
                    "gt_contacting": _to_numpy(batch["gt_contacting"][t]),
                    # Object info (N_max)
                    "object_classes": _to_numpy(batch["object_classes"][t]),
                    "bboxes_2d": _to_numpy(batch["bboxes_2d"][t]),
                    "valid_mask": _to_numpy(batch["valid_mask"][t]),
                }

                # SGDet-specific
                if mode == "sgdet" and "corners" in batch:
                    frame_data["bboxes_3d"] = _to_numpy(batch["corners"][t])

                frames_data[t] = frame_data

            if not frames_data:
                if n_debug < 3:
                    logger.warning(f"  [{video_id}] No frames with valid pairs (T={T})")
                    n_debug += 1
                continue

            # Debug logging
            if n_debug < 3:
                logger.info(
                    f"  [{video_id}] T={T}, frames_with_pairs={len(frames_data)}, "
                    f"total_valid_pairs={total_valid}"
                )
                for t, fd in list(frames_data.items())[:2]:
                    pv = fd["pair_valid"].astype(bool)
                    kv = pv.sum()
                    att_valid = fd["attention_distribution"][pv]
                    logger.info(
                        f"    frame[{t}]: K_valid={kv}, "
                        f"att range=[{att_valid.min():.4f}, {att_valid.max():.4f}], "
                        f"gt_att={fd['gt_attention'][pv].tolist()}"
                    )
                n_debug += 1

            result = {
                "video_id": video_id,
                "frames": frames_data,
            }

            with open(out_path, "wb") as f:
                pickle.dump(result, f)
            n_saved += 1

    logger.info(
        f"  Epoch {epoch}: saved {n_saved} PKLs, "
        f"skipped {n_skipped} existing → {epoch_dir}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dump WSGG predictions")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--min_epoch", type=int, default=4)
    parser.add_argument("--output_root", type=str,
                        default="/data/rohith/ag/wsgg_logits")
    args = parser.parse_args()

    conf = load_wsgg_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    experiment_name = getattr(conf, "experiment_name",
                              f"{conf.method_name}_{conf.mode}")
    conf.experiment_name = experiment_name
    experiment_dir = os.path.join(conf.save_path, experiment_name)
    output_dir = os.path.join(args.output_root, conf.mode, experiment_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Config:          {args.config}")
    logger.info(f"Method:          {conf.method_name}")
    logger.info(f"Mode:            {conf.mode}")
    logger.info(f"Experiment:      {experiment_name}")
    logger.info(f"Checkpoint dir:  {experiment_dir}")
    logger.info(f"Output dir:      {output_dir}")

    checkpoints = discover_checkpoints(experiment_dir, args.min_epoch)
    if not checkpoints:
        logger.error(f"No checkpoints found >= epoch {args.min_epoch}")
        return
    logger.info(f"Found {len(checkpoints)} checkpoints: {[e for e, _ in checkpoints]}")

    from test_wsgg_methods import METHOD_MAP
    if conf.method_name not in METHOD_MAP:
        raise ValueError(f"Unknown method: {conf.method_name}")

    conf.use_wandb = False
    tester = METHOD_MAP[conf.method_name](conf)
    tester._device = device
    tester._enable_wandb = False
    tester._experiment_name = experiment_name
    tester._experiment_dir = experiment_dir
    tester._init_config(is_train=False)
    tester._init_dataset()
    tester.init_model()

    logger.info(f"Model on {device}: {conf.method_name}")

    for epoch, ckpt_name in checkpoints:
        logger.info(f"Processing epoch {epoch}...")
        dump_one_epoch(tester, epoch, ckpt_name, experiment_dir,
                       output_dir, device, conf.mode)
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Done → {output_dir}")


if __name__ == "__main__":
    main()
