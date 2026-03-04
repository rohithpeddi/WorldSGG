#!/usr/bin/env python3
"""
Evaluate WSGG Predictions
==========================

Load saved prediction PKLs (from ``dump_predictions.py``), compute Recall@K
and Mean Recall@K, log to WandB, and export Excel summaries.

The new format stores GT labels from the batch directly inside the PKL,
so no separate annotation directory is needed.

Output structure::

    <logit_root>/eval_results.xlsx    (per-epoch metrics)
    <logit_root>/eval_results.json    (machine-readable summary)

Usage::

    # Evaluate all epochs for one method:
    python evaluate_predictions.py \\
        --logit_root /data/rohith/ag/wsgg_logits/predcls/gl_stgn_predcls_dinov2b/ \\
        --mode predcls \\
        --experiment_name gl_stgn_predcls_dinov2b

    # Evaluate a single epoch:
    python evaluate_predictions.py \\
        --logit_dir /data/rohith/ag/wsgg_logits/predcls/gl_stgn_predcls_dinov2b/epoch_5/ \\
        --mode predcls
"""

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_pkl(path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_one_epoch(
    logit_dir: str,
    mode: str,
    epoch: int,
):
    """
    Evaluate all video predictions in one epoch directory.

    Returns dict with R@K and mR@K values.
    """
    from lib.supervised.evaluation_recall import (
        BasicSceneGraphEvaluator,
        evaluate_wsgg_video,
    )
    from dataloader.world_ag_dataset import (
        ATTENTION_RELATIONSHIPS,
        SPATIAL_RELATIONSHIPS,
        CONTACTING_RELATIONSHIPS,
        OBJECT_CLASSES,
    )

    all_predicates = (
        list(ATTENTION_RELATIONSHIPS)
        + list(SPATIAL_RELATIONSHIPS)
        + list(CONTACTING_RELATIONSHIPS)
    )

    evaluator = BasicSceneGraphEvaluator(
        mode=mode,
        AG_object_classes=OBJECT_CLASSES,
        AG_all_predicates=all_predicates,
        AG_attention_predicates=list(ATTENTION_RELATIONSHIPS),
        AG_spatial_predicates=list(SPATIAL_RELATIONSHIPS),
        AG_contacting_predicates=list(CONTACTING_RELATIONSHIPS),
        iou_threshold=0.5,
        save_file=os.path.join(logit_dir, "eval_stats.txt"),
        constraint="with",
    )

    # Discover prediction PKLs
    pred_paths = sorted(Path(logit_dir).glob("*.pkl"))
    if not pred_paths:
        logger.warning(f"  No prediction PKLs found in {logit_dir}")
        return None

    n_evaluated = 0
    n_skipped = 0

    for pred_path in pred_paths:
        video_id = pred_path.stem

        # Load prediction + embedded GT
        pred_pkl = load_pkl(pred_path)

        # Check that this PKL has the required GT labels
        if "gt_attention" not in pred_pkl:
            logger.debug(f"  {video_id}: missing gt_attention, skipping")
            n_skipped += 1
            continue

        try:
            evaluate_wsgg_video(
                pred_pkl=pred_pkl,
                evaluator=evaluator,
                mode=mode,
                verbose=(n_evaluated < 3),
            )
            n_evaluated += 1
        except Exception as e:
            logger.warning(f"  Error evaluating {video_id}: {e}", exc_info=(n_evaluated < 3))
            n_skipped += 1

    logger.info(
        f"  Epoch {epoch}: evaluated {n_evaluated} videos, skipped {n_skipped}"
    )

    if n_evaluated == 0:
        return None

    # Fetch results
    results = evaluator.fetch_stats_json()
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate WSGG predictions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--logit_root", type=str, default=None,
                       help="Root dir containing epoch_N/ subdirs (evaluates all)")
    group.add_argument("--logit_dir", type=str, default=None,
                       help="Single epoch directory to evaluate")
    parser.add_argument("--mode", required=True, type=str, choices=["predcls", "sgdet"],
                        help="Evaluation mode")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="WandB experiment name (for logging)")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Log results to WandB")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Discover epoch directories
    epoch_dirs = []
    if args.logit_dir:
        # Single epoch
        epoch_match = re.search(r"epoch_(\d+)", args.logit_dir)
        epoch_num = int(epoch_match.group(1)) if epoch_match else 0
        epoch_dirs = [(epoch_num, args.logit_dir)]
        output_root = str(Path(args.logit_dir).parent)
    else:
        # All epochs under logit_root
        output_root = args.logit_root
        pattern = re.compile(r"^epoch_(\d+)$")
        for name in sorted(os.listdir(args.logit_root)):
            m = pattern.match(name)
            if m:
                epoch_num = int(m.group(1))
                epoch_dirs.append((epoch_num, os.path.join(args.logit_root, name)))

    if not epoch_dirs:
        logger.error("No epoch directories found.")
        return

    epoch_dirs.sort()
    logger.info(f"Mode:            {args.mode}")
    logger.info(f"Epochs to eval:  {[e for e, _ in epoch_dirs]}")

    # WandB init
    wandb_run = None
    if args.use_wandb and args.experiment_name:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.experiment_name,
                name=f"eval_{args.mode}",
                config={"mode": args.mode, "experiment": args.experiment_name},
            )
            logger.info(f"WandB initialized: project={args.experiment_name}")
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # Evaluate each epoch
    all_results = {}
    for epoch_num, epoch_dir in epoch_dirs:
        logger.info(f"Evaluating epoch {epoch_num}...")
        results = evaluate_one_epoch(
            logit_dir=epoch_dir,
            mode=args.mode,
            epoch=epoch_num,
        )
        if results is None:
            continue

        all_results[epoch_num] = results

        # Print metrics
        logger.info(f"  Epoch {epoch_num} results:")
        for k, v in results.get("recall", {}).items():
            logger.info(f"    R@{k}: {v:.6f}")
        for k, v in results.get("mean_recall", {}).items():
            logger.info(f"    mR@{k}: {v:.6f}")
        for k, v in results.get("harmonic_mean_recall", {}).items():
            logger.info(f"    hR@{k}: {v:.6f}")

        # WandB logging
        if wandb_run is not None:
            log_dict = {"epoch": epoch_num}
            for k, v in results.get("recall", {}).items():
                log_dict[f"R@{k}"] = v
            for k, v in results.get("mean_recall", {}).items():
                log_dict[f"mR@{k}"] = v
            for k, v in results.get("harmonic_mean_recall", {}).items():
                log_dict[f"hR@{k}"] = v
            wandb_run.log(log_dict, step=epoch_num)

    # ---- Save JSON summary ----
    json_path = os.path.join(output_root, "eval_results.json")
    serializable = {}
    for epoch, res in all_results.items():
        serializable[str(epoch)] = {
            k1: {str(k2): float(v2) for k2, v2 in v1.items()}
            for k1, v1 in res.items()
        }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results JSON saved: {json_path}")

    # ---- Save Excel summary ----
    try:
        import openpyxl

        xlsx_path = os.path.join(output_root, "eval_results.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = args.mode

        # Header
        headers = ["Epoch",
                    "R@10", "R@20", "R@50", "R@100",
                    "mR@10", "mR@20", "mR@50", "mR@100",
                    "hR@10", "hR@20", "hR@50", "hR@100"]
        ws.append(headers)

        from openpyxl.styles import Font
        for col in range(1, len(headers) + 1):
            ws.cell(row=1, column=col).font = Font(bold=True)

        for epoch in sorted(all_results.keys()):
            res = all_results[epoch]
            row = [epoch]
            for k in [10, 20, 50, 100]:
                row.append(res.get("recall", {}).get(k, 0.0))
            for k in [10, 20, 50, 100]:
                row.append(res.get("mean_recall", {}).get(k, 0.0))
            for k in [10, 20, 50, 100]:
                row.append(res.get("harmonic_mean_recall", {}).get(k, 0.0))
            ws.append(row)

        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max_len + 2

        wb.save(xlsx_path)
        logger.info(f"Excel results saved: {xlsx_path}")
    except ImportError:
        logger.warning("openpyxl not installed — skipping Excel export.")

    # Finish WandB
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("WandB run finished.")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
