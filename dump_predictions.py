#!/usr/bin/env python3
"""
Dump WSGG Predictions from Checkpoints
========================================

Load each checkpoint (epoch >= min_epoch) for a given config, run inference
on the test set, and save per-video prediction PKLs.

Output structure::

    <output_root>/<mode>/<experiment_name>/epoch_<N>/<video_id>.pkl

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

            # Extract metadata from batch (last frame: index [-1] of the
            # (T, N_max, ...) tensors, but batch_size=1 so no batch dim)
            T = batch["visual_features"].shape[0]
            last = T - 1

            pair_valid = _to_numpy(batch["pair_valid"][last])       # (K_max,)
            person_idx = _to_numpy(batch["person_idx"][last])       # (K_max,)
            object_idx = _to_numpy(batch["object_idx"][last])       # (K_max,)
            valid_mask = _to_numpy(batch["valid_mask"][last])        # (N_max,)
            object_classes = _to_numpy(batch["object_classes"][last])  # (N_max,)
            bboxes_2d = _to_numpy(batch["bboxes_2d"][last])          # (N_max, 4)

            # Filter predictions to valid pairs only
            valid_k = pair_valid.astype(bool)

            att_dist = _to_numpy(prediction["attention_distribution"])
            spa_dist = _to_numpy(prediction["spatial_distribution"])
            con_dist = _to_numpy(prediction["contacting_distribution"])

            result = {
                "video_id": video_id,
                # Relationship distributions (last-frame, valid pairs only)
                "attention_distribution": att_dist[valid_k] if att_dist.shape[0] == valid_k.shape[0] else att_dist,
                "spatial_distribution": spa_dist[valid_k] if spa_dist.shape[0] == valid_k.shape[0] else spa_dist,
                "contacting_distribution": con_dist[valid_k] if con_dist.shape[0] == valid_k.shape[0] else con_dist,
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

            # Raw logits (if model returns them)
            for logit_key in ("attention_logits", "spatial_logits", "contacting_logits"):
                if logit_key in prediction:
                    lk = _to_numpy(prediction[logit_key])
                    result[logit_key] = lk[valid_k] if lk.shape[0] == valid_k.shape[0] else lk

            # SGDet-specific: 3D boxes, predicted labels, detection scores
            if mode == "sgdet":
                if "corners" in batch:
                    corners = _to_numpy(batch["corners"][last])  # (N_max, 8, 3)
                    result["bboxes_3d"] = corners

                if "pred_labels" in prediction:
                    result["pred_labels"] = _to_numpy(prediction["pred_labels"])
                if "pred_scores" in prediction:
                    result["pred_scores"] = _to_numpy(prediction["pred_scores"])

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
