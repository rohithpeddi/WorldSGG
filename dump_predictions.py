#!/usr/bin/env python3
"""
Dump WSGG Predictions from Checkpoints
========================================

Load each checkpoint (epoch >= min_epoch) for a given config, run inference
on the test set, and save per-video prediction PKLs.

Each PKL stores the **last-frame** prediction alongside the batch's GT labels
(same N_max indexing), ensuring consistent evaluation.

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

import numpy as np
import torch
from tqdm import tqdm

from wsgg_base import load_wsgg_config

logger = logging.getLogger(__name__)


def _to_numpy(t):
    """Convert tensor to numpy array (no-op if already numpy)."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def discover_checkpoints(experiment_dir: str, min_epoch: int):
    """Find checkpoint_N dirs with N >= min_epoch. Returns sorted (epoch, name)."""
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
    """Load one checkpoint and dump per-video predictions (last frame only)."""
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

            # Run inference
            prediction = tester.process_test_video(batch)
            if prediction is None:
                continue

            T = batch["visual_features"].shape[0]
            last = T - 1

            # ---- Last-frame batch data (N_max indexing) ----
            pair_valid = _to_numpy(batch["pair_valid"][last])         # (K_max,)
            person_idx = _to_numpy(batch["person_idx"][last])         # (K_max,)
            object_idx = _to_numpy(batch["object_idx"][last])         # (K_max,)
            valid_mask = _to_numpy(batch["valid_mask"][last])          # (N_max,)
            object_classes = _to_numpy(batch["object_classes"][last])  # (N_max,)
            bboxes_2d = _to_numpy(batch["bboxes_2d"][last])            # (N_max, 4)

            # ---- Last-frame GT labels (same K_max pair indexing) ----
            gt_attention = _to_numpy(batch["gt_attention"][last])      # (K_max,)
            gt_spatial = _to_numpy(batch["gt_spatial"][last])          # (K_max, 6)
            gt_contacting = _to_numpy(batch["gt_contacting"][last])    # (K_max, 17)

            # ---- Prediction distributions (K_max from process_test_video) ----
            att_dist = _to_numpy(prediction["attention_distribution"])   # (K_max, 3)
            spa_dist = _to_numpy(prediction["spatial_distribution"])     # (K_max, 6)
            con_dist = _to_numpy(prediction["contacting_distribution"])  # (K_max, 17)

            valid_k = pair_valid.astype(bool)
            K_valid = valid_k.sum()

            # Debug logging for first 3 videos
            if n_debug < 3:
                logger.info(
                    f"  [{video_id}] T={T}, K_max={len(pair_valid)}, "
                    f"K_valid={K_valid}, N_valid={int(valid_mask.sum())}"
                )
                if K_valid > 0:
                    logger.info(
                        f"    att_dist[valid] range: "
                        f"[{att_dist[valid_k].min():.4f}, {att_dist[valid_k].max():.4f}]"
                    )
                    logger.info(
                        f"    person_idx[valid]: {person_idx[valid_k].tolist()}, "
                        f"object_idx[valid]: {object_idx[valid_k].tolist()}"
                    )
                    logger.info(
                        f"    gt_attention[valid]: {gt_attention[valid_k].tolist()}"
                    )
                n_debug += 1

            # Store everything with consistent K_max / N_max indexing
            # We do NOT filter by pair_valid here — the evaluator will do it
            result = {
                "video_id": video_id,
                # Prediction distributions (K_max, ...)
                "attention_distribution": att_dist,
                "spatial_distribution": spa_dist,
                "contacting_distribution": con_dist,
                # Pair info (K_max)
                "person_idx": person_idx,
                "object_idx": object_idx,
                "pair_valid": pair_valid,
                # GT labels (K_max) — same indexing as predictions
                "gt_attention": gt_attention,
                "gt_spatial": gt_spatial,
                "gt_contacting": gt_contacting,
                # Object info (N_max)
                "object_classes": object_classes,
                "bboxes_2d": bboxes_2d,
                "valid_mask": valid_mask,
            }

            # SGDet-specific
            if mode == "sgdet":
                if "corners" in batch:
                    result["bboxes_3d"] = _to_numpy(batch["corners"][last])
                for key in ("pred_labels", "pred_scores"):
                    if key in prediction:
                        result[key] = _to_numpy(prediction[key])

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
