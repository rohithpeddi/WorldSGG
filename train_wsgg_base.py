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

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def init_dataset(self):
        """Initialize WorldAG train/test datasets and dataloaders."""
        from dataloader.supervised.generation.world_ag.world_ag_dataset import WorldAG

        print("Initializing WorldAG datasets...")

        self._train_dataset = WorldAG(
            phase="train",
            data_path=self._conf.data_path,
            datasize=getattr(self._conf, 'datasize', 'large'),
            world_sg_dir=getattr(self._conf, 'world_sg_dir', ''),
            include_invisible=getattr(self._conf, 'include_invisible', True),
        )
        self._test_dataset = WorldAG(
            phase="test",
            data_path=self._conf.data_path,
            datasize=getattr(self._conf, 'datasize', 'large'),
            world_sg_dir=getattr(self._conf, 'world_sg_dir', ''),
            include_invisible=getattr(self._conf, 'include_invisible', True),
        )

        self._object_classes = self._train_dataset.object_classes

        # Dataloaders: batch_size=1 for temporal (per-video), configurable for stateless
        if self.is_temporal():
            self._dataloader_train = DataLoader(
                self._train_dataset, batch_size=1, shuffle=True, num_workers=0,
            )
            self._dataloader_test = DataLoader(
                self._test_dataset, batch_size=1, shuffle=False, num_workers=0,
            )
        else:
            bs = getattr(self._conf, 'batch_size', 16)
            self._dataloader_train = DataLoader(
                self._train_dataset, batch_size=bs, shuffle=True, num_workers=0,
            )
            self._dataloader_test = DataLoader(
                self._test_dataset, batch_size=1, shuffle=False, num_workers=0,
            )

        print(f"  Train: {len(self._train_dataset)} items | Test: {len(self._test_dataset)} items")

    # ------------------------------------------------------------------
    # Loss Functions
    # ------------------------------------------------------------------
    def _init_loss_functions(self):
        """Initialize standard CE and BCE losses."""
        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCELoss()

    # ------------------------------------------------------------------
    # Monocular3D Trained Detector Loader
    # ------------------------------------------------------------------
    def _load_trained_detector(self):
        """
        Load a trained monocular3d detector checkpoint.

        Follows the pattern from lib/detector/monocular3d/trainer.py:
          1. Create DinoV3Monocular3D(num_classes, model, head_3d_mode)
          2. Load state_dict from checkpoint
          3. Freeze all params + eval()
          4. Wrap in DINOFeatureExtractor for ROI extraction
        """
        from lib.supervised.worldsgg.worldsgg_base import DINOFeatureExtractor

        detector_ckpt = getattr(self._conf, 'detector_ckpt', '')
        if not detector_ckpt:
            print("[TrainWSGGBase] No detector checkpoint specified, skipping detector load.")
            return

        detector_type = getattr(self._conf, 'detector_type', 'none')
        if detector_type != 'dino_mono3d':
            return

        from lib.detector.monocular3d.models.dino_mono_3d import DinoV3Monocular3D

        print(f"[TrainWSGGBase] Loading trained detector from: {detector_ckpt}")

        detector = DinoV3Monocular3D(
            num_classes=getattr(self._conf, 'num_detector_classes', 37),
            pretrained=False,
            model=getattr(self._conf, 'detector_model', 'v3l'),
            head_3d_mode="unified",
        )

        # Load checkpoint
        ckpt = torch.load(detector_ckpt, map_location="cpu")
        if "model_state_dict" in ckpt:
            detector.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "state_dict" in ckpt:
            detector.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            detector.load_state_dict(ckpt, strict=False)
        del ckpt
        gc.collect()

        # Freeze + eval
        detector.to(self._device)
        detector.eval()
        for p in detector.parameters():
            p.requires_grad = False

        # Wrap in DINOFeatureExtractor
        self._detector = DINOFeatureExtractor(
            detector_ckpt="",  # Already loaded
            detector_model=getattr(self._conf, 'detector_model', 'v3l'),
            num_classes=getattr(self._conf, 'num_detector_classes', 37),
            device=str(self._device),
        )
        self._detector._detector = detector  # Inject pre-loaded model
        print(f"[TrainWSGGBase] Detector loaded and frozen ({sum(p.numel() for p in detector.parameters()):,} params)")

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    def _train_model(self):
        """Main training loop over epochs."""
        use_amp = getattr(self._conf, 'use_amp', True) and torch.cuda.is_available()
        if use_amp:
            self._scaler = torch.amp.GradScaler()

        log_every = getattr(self._conf, 'log_every', 100)

        for epoch in range(self._conf.nepoch):
            self._model.train()
            train_iter = iter(self._dataloader_train)
            tr = []
            start_time = time.time()

            for batch_idx in tqdm(range(len(self._dataloader_train)), desc=f"Epoch {epoch + 1}/{self._conf.nepoch}"):
                batch = next(train_iter)

                # ---------- Method-specific forward pass ----------
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        losses = self.process_train_video(batch)
                else:
                    losses = self.process_train_video(batch)
                # --------------------------------------------------

                # Compute total loss
                loss = sum(losses.values())

                # Skip NaN/Inf
                if not torch.isfinite(loss):
                    self._optimizer.zero_grad(set_to_none=True)
                    print(f"  Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                    continue

                # Backward
                self._optimizer.zero_grad()
                if use_amp and self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_norm=getattr(self._conf, 'grad_clip', 5.0),
                    )
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_norm=getattr(self._conf, 'grad_clip', 5.0),
                    )
                    self._optimizer.step()

                # Logging
                if self._enable_wandb:
                    wandb.log({k: v.item() for k, v in losses.items()})
                tr.append(pd.Series({k: v.item() for k, v in losses.items()}))

                if batch_idx % log_every == 0 and batch_idx > 0:
                    elapsed = time.time() - start_time
                    print(f"\n  e{epoch:2d} b{batch_idx:5d}/{len(self._dataloader_train):5d}"
                          f"  {elapsed / batch_idx:.3f}s/batch")
                    mn = pd.concat(tr[-log_every:], axis=1).mean(1)
                    print(mn)

            # Save checkpoint
            self._save_model(
                model=self._model,
                epoch=epoch,
                checkpoint_save_file_path=self._checkpoint_save_dir_path,
                checkpoint_name=self._checkpoint_name,
                method_name=self._conf.method_name,
            )

            # End-of-epoch evaluation
            score = self._evaluate_after_epoch()
            self._scheduler.step(score)

    def _evaluate_after_epoch(self) -> float:
        """Run test evaluation after each epoch. Returns score for scheduler."""
        test_iter = iter(self._dataloader_test)
        self._model.eval()
        with torch.no_grad():
            for b in tqdm(range(len(self._dataloader_test)), desc="Evaluating"):
                batch = next(test_iter)
                pred = self.process_test_video(batch)
                if pred is not None and self._evaluator is not None:
                    self._evaluator.evaluate_scene_graph(batch, pred)

        if self._evaluator is not None:
            score = np.mean(self._evaluator.result_dict.get(
                self._conf.mode + "_recall", {}).get(20, [0.0]))
            self._evaluator.print_stats()
            self._evaluator.reset_result()
        else:
            score = 0.0

        print('-------------------------------------------------------------------')
        return score

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def init_method_training(self):
        """Full initialization → training pipeline."""
        # 0. Config
        self._init_config()

        # 1. Dataset
        self.init_dataset()

        # 2. Evaluators
        self._init_evaluators()

        # 3. Model + Loss
        self.init_model()
        self.init_loss_fn()
        self._init_loss_functions()
        self._load_checkpoint()

        # 4. Detector
        self._init_detector()
        self._load_trained_detector()

        # 5. Optimizer + Scheduler
        self._init_optimizer()
        self._init_scheduler()

        # 6. Train
        print("-----------------------------------------------------")
        print(f"Initialized training for: {self._conf.method_name}")
        print(f"  Temporal: {self.is_temporal()}")
        print(f"  Detector: {getattr(self._conf, 'detector_type', 'none')}")
        print("-----------------------------------------------------")
        self._train_model()

    # ------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------
    @abstractmethod
    def is_temporal(self) -> bool:
        """True for sequential (GL-STGN, AMWAE, LKS), False for frame-shuffled (Amnesic)."""
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
