"""
WSGG Training Base
===================

Common training loop for all WSGG methods.
Analogous to train_sgg_base.py in the SGG pipeline.

Handles:
  - Dataset loading (WorldAG)
  - Loss function initialization
  - Training loop (temporal + stateless modes)
  - Monocular3D trained detector loading
  - End-of-epoch evaluation
  - init_method_training() orchestration
"""

import copy
import gc
import json
import logging
import os
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsgg_base import WSGGBase

logger = logging.getLogger(__name__)


class TrainWSGGBase(WSGGBase):
    """
    Common training loop for all WSGG methods.

    Subclasses override:
      - init_model()               → instantiate method-specific model
      - init_loss_fn()             → instantiate method-specific loss
      - is_temporal()              → True = sequential videos, False = frame-shuffled
      - process_train_video(batch) → forward + return loss dict
      - process_test_video(batch)  → inference for one video
    """

    def __init__(self, conf):
        super().__init__(conf)
        self._loss_fn = None
        self._dataloader_train = None
        self._dataloader_test = None
        self._object_classes = None
        self._scaler = None
        self._best_score = 0.0
        self._best_epoch = -1

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def init_dataset(self):
        """Initialize WorldAG train (and optionally test) datasets."""
        from dataloader.world_ag_dataset import WorldAG, world_collate_fn

        skip_test = getattr(self._conf, 'skip_test', False)
        logger.info("Initializing WorldAG datasets...")

        self._train_dataset = WorldAG(
            phase="train",
            data_path=self._conf.data_path,
            mode=self._conf.mode,
            feature_model=getattr(self._conf, 'feature_model', 'dinov2b'),
            include_invisible=getattr(self._conf, 'include_invisible', True),
            max_objects=getattr(self._conf, 'max_objects', 64),
        )

        self._object_classes = self._train_dataset.object_classes

        self._dataloader_train = DataLoader(
            self._train_dataset, batch_size=1, shuffle=True, num_workers=0,
            collate_fn=world_collate_fn,
        )

        if not skip_test:
            self._test_dataset = WorldAG(
                phase="test",
                data_path=self._conf.data_path,
                mode=self._conf.mode,
                feature_model=getattr(self._conf, 'feature_model', 'dinov2b'),
                include_invisible=getattr(self._conf, 'include_invisible', True),
                max_objects=getattr(self._conf, 'max_objects', 64),
            )
            self._dataloader_test = DataLoader(
                self._test_dataset, batch_size=1, shuffle=False, num_workers=0,
                collate_fn=world_collate_fn,
            )
            logger.info(f"  Train: {len(self._train_dataset)} items | Test: {len(self._test_dataset)} items")
        else:
            logger.info(f"  Train: {len(self._train_dataset)} items | Test: SKIPPED")

    # ------------------------------------------------------------------
    # Loss Functions
    # ------------------------------------------------------------------
    def _init_loss_functions(self):
        """Initialize standard CE and BCE losses."""
        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCELoss()



    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    def _train_model(self):
        """Main training loop over epochs."""
        use_amp = self._conf.use_amp and torch.cuda.is_available()
        if use_amp:
            # Prefer bfloat16 (same exponent range as fp32 — no overflow at 65504)
            # Falls back to float16 + GradScaler on older GPUs
            self._use_bf16 = torch.cuda.is_bf16_supported()
            if self._use_bf16:
                self._scaler = None  # bfloat16 doesn't need loss scaling
                logger.info("  AMP: using bfloat16 (no GradScaler needed)")
            else:
                self._scaler = torch.amp.GradScaler('cuda')
                logger.info("  AMP: using float16 + GradScaler")
        else:
            self._use_bf16 = False

        log_every = self._conf.log_every

        for epoch in range(self._starting_epoch, self._conf.nepoch):
            self._model.train()
            train_iter = iter(self._dataloader_train)
            tr = []
            start_time = time.time()

            for batch_idx in tqdm(range(len(self._dataloader_train)), desc=f"Epoch {epoch + 1}/{self._conf.nepoch}"):
                batch = next(train_iter)

                self._optimizer.zero_grad(set_to_none=True)

                # Forward + loss
                if use_amp:
                    amp_dtype = torch.bfloat16 if self._use_bf16 else torch.float16
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        losses = self.process_train_video(batch)
                else:
                    losses = self.process_train_video(batch)

                # Use pre-computed total (loss functions return a "total" key)
                loss = losses.get("total", sum(losses.values()))

                # Skip NaN/Inf or zero-grad losses (e.g., no valid pairs)
                if not torch.isfinite(loss) or loss.item() == 0.0:
                    if not torch.isfinite(loss):
                        logger.warning(f"  NaN/Inf loss at batch {batch_idx}, skipping")
                    continue

                # Backward + step
                if use_amp and self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_norm=self._conf.grad_clip,
                    )
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_norm=self._conf.grad_clip,
                    )
                    self._optimizer.step()

                # Logging
                if self._enable_wandb:
                    wandb.log({k: v.item() for k, v in losses.items()})
                    wandb.log({"lr": self._optimizer.param_groups[0]["lr"]})
                tr.append(pd.Series({k: v.item() for k, v in losses.items()}))

                # Step scheduler (per-iteration warmup → cosine)
                self._scheduler.step()

                if batch_idx % log_every == 0 and batch_idx > 0:
                    elapsed = time.time() - start_time
                    print(f"\n  e{epoch:2d} b{batch_idx:5d}/{len(self._dataloader_train):5d}"
                          f"  {elapsed / batch_idx:.3f}s/batch")
                    mn = pd.concat(tr[-log_every:], axis=1).mean(1)
                    logger.info(f"\n{mn}")

            # Save full-state checkpoint
            self._save_checkpoint(epoch)

            # End-of-epoch evaluation (skip if no test dataset)
            if self._dataloader_test is not None:
                score = self._evaluate_after_epoch(epoch)
            else:
                score = 0.0

            # Epoch-level WandB logging
            epoch_losses = pd.concat(tr, axis=1).mean(1) if tr else pd.Series()
            if self._enable_wandb:
                wandb_epoch = {"epoch": epoch + 1}
                for k, v in epoch_losses.items():
                    wandb_epoch[f"epoch/{k}"] = v
                wandb_epoch["epoch/recall@20"] = score
                wandb_epoch["epoch/best_score"] = self._best_score
                wandb.log(wandb_epoch)

            # Epoch summary
            logger.info(f"\n{'═' * 60}")
            logger.info(f"  EPOCH {epoch + 1}/{self._conf.nepoch} SUMMARY")
            logger.info(f"  Recall@20: {score:.4f}  |  Best: {self._best_score:.4f} (epoch {self._best_epoch + 1})")
            if len(epoch_losses) > 0:
                logger.info(f"  Avg losses: {dict(epoch_losses.round(4))}")
            logger.info(f"{'═' * 60}")

    def _evaluate_after_epoch(self, epoch: int) -> float:
        """Run test evaluation after each epoch. Returns score for scheduler."""
        from lib.supervised.evaluation_recall import evaluate_wsgg_video

        test_iter = iter(self._dataloader_test)
        self._model.eval()
        with torch.no_grad():
            for b in tqdm(range(len(self._dataloader_test)), desc="Evaluating"):
                batch = next(test_iter)
                pred = self.process_test_video(batch)
                if pred is not None and self._evaluator is not None:
                    # Build pred_pkl dict for the last frame (what the model
                    # predicted on) from batch metadata + model outputs.
                    T = batch["T"]
                    last = T - 1
                    pred_pkl = {
                        "video_id": batch["video_id"],
                        # Model predictions (last frame)
                        "attention_distribution": pred["attention_distribution"].cpu().numpy(),
                        "spatial_distribution": pred["spatial_distribution"].cpu().numpy(),
                        "contacting_distribution": pred["contacting_distribution"].cpu().numpy(),
                        # GT labels (last frame)
                        "gt_attention": batch["gt_attention"][last].numpy(),
                        "gt_spatial": batch["gt_spatial"][last].numpy(),
                        "gt_contacting": batch["gt_contacting"][last].numpy(),
                        # Pair metadata (last frame)
                        "pair_valid": batch["pair_valid"][last].numpy(),
                        "person_idx": batch["person_idx"][last].numpy(),
                        "object_idx": batch["object_idx"][last].numpy(),
                        # Object metadata (last frame)
                        "object_classes": batch["object_classes"][last].numpy(),
                        "bboxes_2d": batch["bboxes_2d"][last].numpy(),
                        "valid_mask": batch["valid_mask"][last].numpy(),
                    }
                    # SGDet: add detector-predicted labels and corners
                    if self._conf.mode == "sgdet":
                        pred_pkl["pred_labels"] = batch["object_classes"][last].numpy()
                        pred_pkl["pred_scores"] = np.ones(batch["object_classes"][last].shape[0],
                                                          dtype=np.float32)
                        # Real GT annotation boxes/corners for proper IoU evaluation
                        pred_pkl["gt_bboxes_2d"] = batch["gt_bboxes_2d"][last].numpy()
                        pred_pkl["gt_corners"] = batch["gt_corners"][last].numpy()

                        # Transform detector 3D corners: camera space → FINAL space
                        corners_raw = batch.get("corners")
                        cam_pose = batch.get("camera_poses")
                        if corners_raw is not None:
                            corners_cam = corners_raw[last].numpy()  # (N_max, 8, 3)
                            if cam_pose is not None:
                                T = cam_pose[last].numpy()  # (4, 4) cam-to-FINAL
                                R, t = T[:3, :3], T[:3, 3]
                                # corners_cam (N,8,3) → corners_final (N,8,3)
                                corners_cam = np.einsum('ij,nkj->nki', R, corners_cam) + t
                            pred_pkl["bboxes_3d"] = corners_cam

                    # Feed both evaluators with the same predictions
                    evaluate_wsgg_video(
                        pred_pkl, self._evaluator,
                        mode=self._conf.mode, verbose=False,
                    )
                    evaluate_wsgg_video(
                        pred_pkl, self._evaluator_nc,
                        mode=self._conf.mode, verbose=False,
                    )

        if self._evaluator is not None:
            # --- With-constraint metrics ---
            stats_wc = self._evaluator.fetch_stats_json()
            r_wc = stats_wc["recall"]
            mr_wc = stats_wc["mean_recall"]
            hr_wc = stats_wc["harmonic_mean_recall"]

            # --- No-constraint metrics ---
            stats_nc = self._evaluator_nc.fetch_stats_json()
            r_nc = stats_nc["recall"]
            mr_nc = stats_nc["mean_recall"]
            hr_nc = stats_nc["harmonic_mean_recall"]

            score = r_wc.get(20, 0.0)

            self._evaluator.print_stats()
            logger.info("--- No-Constraint ---")
            self._evaluator_nc.print_stats()

            # Log all metrics to WandB
            if self._enable_wandb:
                wandb_metrics = {"epoch": epoch + 1}
                for k in [10, 20, 50, 100]:
                    # With constraint
                    wandb_metrics[f"metrics/wc/R@{k}"] = r_wc.get(k, 0.0)
                    wandb_metrics[f"metrics/wc/mR@{k}"] = mr_wc.get(k, 0.0)
                    wandb_metrics[f"metrics/wc/hR@{k}"] = hr_wc.get(k, 0.0)
                    # No constraint
                    wandb_metrics[f"metrics/nc/R@{k}"] = r_nc.get(k, 0.0)
                    wandb_metrics[f"metrics/nc/mR@{k}"] = mr_nc.get(k, 0.0)
                    wandb_metrics[f"metrics/nc/hR@{k}"] = hr_nc.get(k, 0.0)
                wandb.log(wandb_metrics)

            # Save metrics to results log file
            results_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "results"
            )
            os.makedirs(results_dir, exist_ok=True)
            log_path = os.path.join(
                results_dir, f"{self._conf.experiment_name}_metrics.jsonl"
            )
            row = {"epoch": epoch + 1}
            for k in [10, 20, 50, 100]:
                row[f"wc/R@{k}"] = round(r_wc.get(k, 0.0), 6)
                row[f"wc/mR@{k}"] = round(mr_wc.get(k, 0.0), 6)
                row[f"wc/hR@{k}"] = round(hr_wc.get(k, 0.0), 6)
                row[f"nc/R@{k}"] = round(r_nc.get(k, 0.0), 6)
                row[f"nc/mR@{k}"] = round(mr_nc.get(k, 0.0), 6)
                row[f"nc/hR@{k}"] = round(hr_nc.get(k, 0.0), 6)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(f"📊 Metrics saved → {log_path}")

            self._evaluator.reset_result()
            self._evaluator_nc.reset_result()
        else:
            score = 0.0

        # Best model tracking
        if score > self._best_score:
            self._best_score = score
            self._best_epoch = epoch
            self._save_best_model(epoch, score)
            logger.info(f"🏆 New best model! Recall@20={score:.4f} at epoch {epoch + 1}")

        logger.info('─' * 60)
        return score

    def _save_best_model(self, epoch: int, score: float) -> None:
        """Save model weights as best_model.pth when recall improves."""
        import os
        best_path = os.path.join(self._experiment_dir, "best_model.pth")
        torch.save({
            "epoch": epoch,
            "score": score,
            "model_state_dict": self._model.state_dict(),
        }, best_path)
        logger.info(f"✓ Best model saved → {best_path}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def init_method_training(self):
        """Full initialization → training pipeline."""
        # 0. Config
        self._init_config()

        # 1. Dataset
        self.init_dataset()

        # 2. Evaluators (skip if no test dataset)
        if not getattr(self._conf, 'skip_test', False):
            self._init_evaluators()

        # 3. Model + Loss
        self.init_model()
        self.init_loss_fn()
        self._init_loss_functions()

        # 4. Optimizer + Scheduler
        self._init_optimizer()
        total_steps = self._conf.nepoch * len(self._dataloader_train)
        self._init_scheduler(total_steps)

        # 5. Resume from checkpoint (must come after optimizer/scheduler init)
        self._maybe_resume()

        # 6. Train
        logger.info("━" * 60)
        logger.info(f"  Method   : {self._conf.method_name}")
        logger.info(f"  Temporal : {self.is_temporal()}")
        logger.info(f"  Mode     : {self._conf.mode}")
        logger.info(f"  Features : {getattr(self._conf, 'feature_model', 'unknown')}")
        logger.info(f"  Epochs   : {self._starting_epoch} → {self._conf.nepoch}")
        logger.info("━" * 60)
        self._train_model()

    # ------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------
    @abstractmethod
    def is_temporal(self) -> bool:
        """True for sequential (GL-STGN, AMWAE, LKS)."""
        pass

    @abstractmethod
    def init_loss_fn(self):
        """Initialize method-specific loss module. Must set self._loss_fn."""
        pass

    @abstractmethod
    def process_train_video(self, batch) -> dict:
        """
        Method-specific forward pass for training.

        Args:
            batch: Data from dataloader (video or frame depending on is_temporal).

        Returns:
            dict of {loss_name: loss_tensor} — base handles backward + optimizer.
        """
        pass

    @abstractmethod
    def process_test_video(self, batch) -> dict:
        """
        Method-specific inference.

        Args:
            batch: Data from dataloader.

        Returns:
            dict of predictions for evaluator.
        """
        pass
