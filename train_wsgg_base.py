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

import json
import logging
import os
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import wandb
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
    def _make_dataset(self, phase: str):
        """Dataset for one split. Methods needing extra per-video inputs (e.g.
        WorldWise++ token grids) override this and return a WorldAG subclass."""
        from dataloader.world_ag_dataset import WorldAG
        from wsgg_base import annot_dir_for
        return WorldAG(
            phase=phase,
            data_path=self._conf.data_path,
            mode=self._conf.mode,
            feature_model=getattr(self._conf, 'feature_model', 'dinov2b'),
            include_invisible=getattr(self._conf, 'include_invisible', True),
            max_objects=getattr(self._conf, 'max_objects', 64),
            annot_dir_name=annot_dir_for(self._conf, phase),
        )

    def init_dataset(self):
        """Initialize WorldAG train (and optionally test) datasets."""
        from dataloader.world_ag_dataset import world_collate_fn

        skip_test = getattr(self._conf, 'skip_test', False)
        logger.info("Initializing WorldAG datasets...")

        self._train_dataset = self._make_dataset("train")

        self._object_classes = self._train_dataset.object_classes

        # Every item re-reads its feature + annotation PKLs (and, for
        # WorldWise++, its token grids) from disk, so with num_workers=0 the
        # GPU waits on each read. num_workers>0 overlaps them with compute.
        # Default 0 keeps the historical behaviour for already-running cells.
        n_workers = int(getattr(self._conf, 'num_workers', 0))
        loader_kw = {"num_workers": n_workers}
        if n_workers > 0:
            loader_kw.update(persistent_workers=True,
                             prefetch_factor=int(getattr(self._conf, 'prefetch_factor', 4)))
        logger.info(f"  DataLoader workers: {n_workers}")

        self._dataloader_train = DataLoader(
            self._train_dataset, batch_size=1, shuffle=True,
            collate_fn=world_collate_fn, **loader_kw,
        )

        if not skip_test:
            self._test_dataset = self._make_dataset("test")
            self._dataloader_test = DataLoader(
                self._test_dataset, batch_size=1, shuffle=False,
                collate_fn=world_collate_fn, **loader_kw,
            )
            logger.info(f"  Train: {len(self._train_dataset)} items | Test: {len(self._test_dataset)} items")
        else:
            logger.info(f"  Train: {len(self._train_dataset)} items | Test: SKIPPED")

    # ------------------------------------------------------------------
    # Run identity (decision-log artifact)
    # ------------------------------------------------------------------
    def _results_log_path(self):
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results"
        )
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(
            results_dir, f"{self._conf.experiment_name}_metrics.jsonl"
        )

    def _write_run_header(self):
        """One identity row per run in the metrics jsonl: git commit, config
        (incl. all plugin flags), param counts. Without this, results can't be
        traced to a code state when the next refinement round is decided."""
        import subprocess
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except Exception:
            commit = "unknown"

        n_total = sum(p.numel() for p in self._model.parameters())
        n_train = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        # JSON-safe copy of the config (drop non-serializable values)
        conf_args = {}
        for k, v in self._conf.args.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                conf_args[k] = v

        header = {
            "type": "header",
            "experiment": self._conf.experiment_name,
            "git_commit": commit,
            "method": self._conf.method_name,
            "mode": self._conf.mode,
            "backbone": getattr(self._conf, "feature_model", "unknown"),
            "seed": int(getattr(self._conf, "seed", 0)),
            "starting_epoch": self._starting_epoch,
            "params_total": n_total,
            "params_trainable": n_train,
            "config": conf_args,
        }
        with open(self._results_log_path(), "a") as f:
            f.write(json.dumps(header) + "\n")
        logger.info(f"Run header written (commit {commit}, {n_total:,} params)")

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
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

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

            # Epoch cost + loss aggregates (logged into the metrics row)
            epoch_time = time.time() - start_time
            peak_vram_gb = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() else 0.0
            )
            epoch_losses = pd.concat(tr, axis=1).mean(1) if tr else pd.Series()

            # End-of-epoch evaluation (skip if no test dataset)
            if self._dataloader_test is not None:
                score = self._evaluate_after_epoch(
                    epoch, epoch_losses=epoch_losses,
                    epoch_time=epoch_time, peak_vram_gb=peak_vram_gb,
                )
            else:
                score = 0.0
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

    def _evaluate_after_epoch(
        self, epoch: int, epoch_losses=None, epoch_time=0.0, peak_vram_gb=0.0,
    ) -> float:
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

                    # Occlusion-stratified split: pairs whose BOTH endpoints
                    # are visible vs pairs with an unseen endpoint. Filtering
                    # pair_valid is enough — GT and preds are both derived
                    # from it inside evaluate_wsgg_video.
                    if self._evaluator_vis is not None:
                        vis_last = batch["visibility_mask"][last].numpy().astype(bool)
                        n_max = len(vis_last)
                        p_idx = np.clip(pred_pkl["person_idx"], 0, n_max - 1)
                        o_idx = np.clip(pred_pkl["object_idx"], 0, n_max - 1)
                        pair_visible = vis_last[p_idx] & vis_last[o_idx]
                        pv = pred_pkl["pair_valid"].astype(bool)

                        pkl_vis = dict(pred_pkl, pair_valid=(pv & pair_visible))
                        pkl_occ = dict(pred_pkl, pair_valid=(pv & ~pair_visible))
                        evaluate_wsgg_video(
                            pkl_vis, self._evaluator_vis,
                            mode=self._conf.mode, verbose=False,
                        )
                        evaluate_wsgg_video(
                            pkl_occ, self._evaluator_occ,
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
            log_path = self._results_log_path()
            row = {"epoch": epoch + 1}
            for k in [10, 20, 50, 100]:
                row[f"wc/R@{k}"] = round(r_wc.get(k, 0.0), 6)
                row[f"wc/mR@{k}"] = round(mr_wc.get(k, 0.0), 6)
                row[f"wc/hR@{k}"] = round(hr_wc.get(k, 0.0), 6)
                row[f"nc/R@{k}"] = round(r_nc.get(k, 0.0), 6)
                row[f"nc/mR@{k}"] = round(mr_nc.get(k, 0.0), 6)
                row[f"nc/hR@{k}"] = round(hr_nc.get(k, 0.0), 6)

            # Per-predicate recall vector (wc, K=20) — tail-behavior heatmaps
            row["wc/per_predicate_R@20"] = {
                name: round(v, 6)
                for name, v in self._evaluator.fetch_per_predicate_recall(20).items()
            }

            # Occlusion-stratified metrics (wc) — the MWAE story, measured
            if self._evaluator_vis is not None:
                stats_vis = self._evaluator_vis.fetch_stats_json()
                stats_occ = self._evaluator_occ.fetch_stats_json()
                for k in [10, 20, 50]:
                    row[f"vispair/R@{k}"] = round(stats_vis["recall"].get(k, 0.0), 6)
                    row[f"vispair/mR@{k}"] = round(stats_vis["mean_recall"].get(k, 0.0), 6)
                    row[f"occpair/R@{k}"] = round(stats_occ["recall"].get(k, 0.0), 6)
                    row[f"occpair/mR@{k}"] = round(stats_occ["mean_recall"].get(k, 0.0), 6)
                logger.info(
                    f"  Occlusion split — visible pairs R@20: "
                    f"{row.get('vispair/R@20', 0.0):.4f} | masked pairs R@20: "
                    f"{row.get('occpair/R@20', 0.0):.4f}"
                )

            # Loss sub-terms + epoch cost — plugin effects show up here first
            if epoch_losses is not None and len(epoch_losses) > 0:
                for name, val in epoch_losses.items():
                    if np.isfinite(val):
                        row[f"loss/{name}"] = round(float(val), 6)
            row["epoch_time_s"] = round(float(epoch_time), 1)
            row["peak_vram_gb"] = round(float(peak_vram_gb), 2)

            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(f"📊 Metrics saved → {log_path}")

            self._evaluator.reset_result()
            self._evaluator_nc.reset_result()
            if self._evaluator_vis is not None:
                self._evaluator_vis.reset_result()
                self._evaluator_occ.reset_result()
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
    def _set_seed(self, seed: int):
        """Seed all RNGs for single-seed reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"  Seed: {seed}")

    def init_method_training(self):
        """Full initialization → training pipeline."""
        # 0. Reproducibility (single fixed seed per run)
        self._set_seed(int(getattr(self._conf, 'seed', 0)))

        # 0b. Config
        self._init_config()

        # 1. Dataset
        self.init_dataset()

        # 2. Evaluators (skip if no test dataset)
        if not getattr(self._conf, 'skip_test', False):
            self._init_evaluators()

        # 3. Model + Loss
        self.init_model()
        self.init_loss_fn()

        # 4. Optimizer + Scheduler
        self._init_optimizer()
        total_steps = self._conf.nepoch * len(self._dataloader_train)
        self._init_scheduler(total_steps)

        # 5. Resume from checkpoint (must come after optimizer/scheduler init)
        self._maybe_resume()

        # 5b. Run-identity header (git commit + config + param counts)
        self._write_run_header()

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
