#!/usr/bin/env python3
"""
Dump WSGG Predictions from Checkpoints
========================================

Load each checkpoint (epoch >= min_epoch) for a given config, run inference
on the test set, and save per-video prediction PKLs with **per-frame** data.

Output structure::

    <output_root>/<mode>/<experiment_name>/epoch_<N>/<video_id>.pkl

Each PKL stores a dict with:
    - "video_id": str
    - "frame_names": list of frame file names
    - "frames": dict[frame_name → per-frame prediction dict]

Usage::

    python dump_predictions.py --config configs/methods/predcls/gl_stgn_predcls_dinov2b.yaml
    python dump_predictions.py --config configs/methods/sgdet/amwae_sgdet_dinov2b.yaml --min_epoch 5
"""

import gc
import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from wsgg_base import load_wsgg_config

logger = logging.getLogger(__name__)


# ---- helpers ---------------------------------------------------------------

def _to_numpy(t):
    """Convert tensor to numpy array (no-op if already numpy)."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ---- discovery -------------------------------------------------------------

def discover_checkpoints(experiment_dir: str, min_epoch: int):
    """Find all checkpoint_N directories with N >= min_epoch.

    Returns sorted list of (epoch, ckpt_dir_name) tuples.
    """
    pattern = re.compile(r"^checkpoint_(\d+)$")
    hits = []
    if not os.path.isdir(experiment_dir):
        logger.warning(f"Experiment dir not found: {experiment_dir}")
        return hits
    for name in os.listdir(experiment_dir):
        m = pattern.match(name)
        if m:
            epoch = int(m.group(1))
            ckpt_file = os.path.join(experiment_dir, name, "checkpoint_state.pth")
            if epoch >= min_epoch and os.path.isfile(ckpt_file):
                hits.append((epoch, name))
    return sorted(hits)


# ---- inference --------------------------------------------------------------

def dump_one_epoch(tester, epoch, ckpt_name, experiment_dir, output_dir, device, mode):
    """Load one checkpoint and dump per-video predictions."""
    # Load weights
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
    n_first_debug = 3  # Log detailed info for first 3 videos
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  Epoch {epoch}", leave=False):
            video_id = batch.get("video_id", f"unknown_{n_saved}")
            out_path = os.path.join(epoch_dir, f"{video_id}.pkl")

            if os.path.exists(out_path):
                n_skipped += 1
                continue

            # Run inference via tester's process_test_video
            prediction = tester.process_test_video(batch)

            if prediction is None:
                continue

            # The model's forward() returns predictions for ALL T frames.
            # process_test_video returns the LAST frame's predictions.
            # But we need per-frame data for evaluation.
            # Re-run the model to get full temporal predictions.
            #
            # Actually, let's extract per-frame data from the batch directly
            # and pair it with the single-frame prediction output.

            T = batch["visual_features"].shape[0]
            frame_names = batch.get("frame_names", [f"frame_{t}" for t in range(T)])

            # Debug logging
            if n_saved < n_first_debug:
                logger.info(f"  [{video_id}] T={T}, frame_names={frame_names[:3]}...")
                for t in range(T):
                    pv = _to_numpy(batch["pair_valid"][t])
                    n_valid = pv.astype(bool).sum()
                    logger.info(f"    frame[{t}]: pair_valid has {n_valid}/{len(pv)} valid pairs")

            # Build per-frame data from the batch
            frames_data = {}
            for t in range(T):
                frame_name = frame_names[t] if t < len(frame_names) else f"frame_{t}"

                pair_valid = _to_numpy(batch["pair_valid"][t])       # (K_max,)
                person_idx = _to_numpy(batch["person_idx"][t])       # (K_max,)
                object_idx = _to_numpy(batch["object_idx"][t])       # (K_max,)
                valid_mask = _to_numpy(batch["valid_mask"][t])        # (N_max,)
                object_classes = _to_numpy(batch["object_classes"][t])  # (N_max,)
                bboxes_2d = _to_numpy(batch["bboxes_2d"][t])          # (N_max, 4)

                valid_k = pair_valid.astype(bool)
                K_valid = valid_k.sum()

                if K_valid == 0:
                    continue  # Skip frames with no valid pairs

                frame_data = {
                    # Pair indices (valid only)
                    "person_idx": person_idx[valid_k],
                    "object_idx": object_idx[valid_k],
                    "pair_valid": pair_valid,
                    # Object info
                    "object_classes": object_classes,
                    "bboxes_2d": bboxes_2d,
                    "valid_mask": valid_mask,
                    "object_scores": np.ones(int(valid_mask.sum()), dtype=np.float32),
                }

                # SGDet-specific: 3D boxes
                if mode == "sgdet" and "corners" in batch:
                    frame_data["bboxes_3d"] = _to_numpy(batch["corners"][t])

                frames_data[frame_name] = frame_data

            if not frames_data:
                if n_saved < n_first_debug:
                    logger.warning(f"  [{video_id}] No frames with valid pairs!")
                continue

            # Attach prediction distributions to the LAST frame that has valid pairs
            # (process_test_video returns last-frame predictions)
            last_frame_with_pairs = list(frames_data.keys())[-1]  # last frame with pairs
            last_frame_data = frames_data[last_frame_with_pairs]

            att_dist = _to_numpy(prediction["attention_distribution"])
            spa_dist = _to_numpy(prediction["spatial_distribution"])
            con_dist = _to_numpy(prediction["contacting_distribution"])

            # The prediction is for the last frame (T-1), which may or may not
            # be the same as last_frame_with_pairs. We need to match dimensions.
            last_t = T - 1
            pair_valid_last_t = _to_numpy(batch["pair_valid"][last_t])
            valid_k_last_t = pair_valid_last_t.astype(bool)

            if att_dist.shape[0] == valid_k_last_t.shape[0]:
                # Filter prediction to valid pairs of the actual last frame
                att_dist_valid = att_dist[valid_k_last_t]
                spa_dist_valid = spa_dist[valid_k_last_t]
                con_dist_valid = con_dist[valid_k_last_t]
            else:
                att_dist_valid = att_dist
                spa_dist_valid = spa_dist
                con_dist_valid = con_dist

            if n_saved < n_first_debug:
                logger.info(
                    f"  [{video_id}] Pred shapes: "
                    f"att_dist={att_dist.shape} → filtered={att_dist_valid.shape}, "
                    f"last_t pair_valid has {valid_k_last_t.sum()} valid"
                )

            # Attach distributions to ALL frames with valid pairs
            # For simplicity, we attach the same prediction to each frame
            # (the model is temporal so it sees all frames, but outputs last-frame preds)
            for fname, fdata in frames_data.items():
                K_f = fdata["person_idx"].shape[0]
                # If prediction K matches this frame's K, use directly
                if att_dist_valid.shape[0] == K_f:
                    fdata["attention_distribution"] = att_dist_valid
                    fdata["spatial_distribution"] = spa_dist_valid
                    fdata["contacting_distribution"] = con_dist_valid
                else:
                    # Mismatch — use unfiltered prediction truncated to K_f
                    fdata["attention_distribution"] = att_dist[:K_f] if att_dist.shape[0] >= K_f else att_dist
                    fdata["spatial_distribution"] = spa_dist[:K_f] if spa_dist.shape[0] >= K_f else spa_dist
                    fdata["contacting_distribution"] = con_dist[:K_f] if con_dist.shape[0] >= K_f else con_dist

                if "pred_labels" in prediction:
                    fdata["pred_labels"] = _to_numpy(prediction["pred_labels"])
                if "pred_scores" in prediction:
                    fdata["pred_scores"] = _to_numpy(prediction["pred_scores"])

            result = {
                "video_id": video_id,
                "frame_names": frame_names,
                "frames": frames_data,
            }

            with open(out_path, "wb") as f:
                pickle.dump(result, f)
            n_saved += 1

    logger.info(
        f"  Epoch {epoch}: saved {n_saved} PKLs, skipped {n_skipped} existing → {epoch_dir}"
    )


# ---- main -------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dump WSGG predictions from checkpoints")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to method config YAML")
    parser.add_argument("--min_epoch", type=int, default=4,
                        help="Minimum epoch number to evaluate (default: 4)")
    parser.add_argument("--output_root", type=str,
                        default="/data/rohith/ag/wsgg_logits",
                        help="Root directory for output logit PKLs")
    args = parser.parse_args()

    # Load config
    conf = load_wsgg_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    experiment_name = getattr(
        conf, "experiment_name",
        f"{conf.method_name}_{conf.mode}",
    )
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
    logger.info(f"Min epoch:       {args.min_epoch}")

    # Discover checkpoints
    checkpoints = discover_checkpoints(experiment_dir, args.min_epoch)
    if not checkpoints:
        logger.error(f"No checkpoints found with epoch >= {args.min_epoch} in {experiment_dir}")
        return
    logger.info(f"Found {len(checkpoints)} checkpoints: {[e for e, _ in checkpoints]}")

    # Build tester — partial init (no checkpoint load, no test run)
    from test_wsgg_methods import METHOD_MAP

    method_name = conf.method_name
    if method_name not in METHOD_MAP:
        raise ValueError(
            f"Unknown method: {method_name}. Choose from: {list(METHOD_MAP.keys())}"
        )

    # Disable WandB for dumping
    conf.use_wandb = False

    tester = METHOD_MAP[method_name](conf)
    tester._device = device
    tester._enable_wandb = False
    tester._experiment_name = experiment_name
    tester._experiment_dir = experiment_dir

    # Init config (logging setup)
    tester._init_config(is_train=False)

    # Init dataset
    tester._init_dataset()

    # Init model (architecture only, no weights yet)
    tester.init_model()

    logger.info(f"Model initialized on {device}: {conf.method_name}")

    # Run inference for each checkpoint
    for epoch, ckpt_name in checkpoints:
        logger.info(f"Processing epoch {epoch}...")
        dump_one_epoch(tester, epoch, ckpt_name, experiment_dir, output_dir, device, conf.mode)
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Done! All predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
