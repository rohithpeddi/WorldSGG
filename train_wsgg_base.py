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
        """Initialize WorldAG train (and optionally test) datasets."""
        from dataloader.world_ag_dataset import WorldAG, world_collate_fn

        skip_test = getattr(self._conf, 'skip_test', False)
        print("Initializing WorldAG datasets...")

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
            print(f"  Train: {len(self._train_dataset)} items | Test: {len(self._test_dataset)} items")
        else:
            print(f"  Train: {len(self._train_dataset)} items | Test: SKIPPED")

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
            self._scaler = torch.amp.GradScaler()

        log_every = self._conf.log_every

        for epoch in range(self._starting_epoch, self._conf.nepoch):
            self._model.train()
            train_iter = iter(self._dataloader_train)
            tr = []
            start_time = time.time()

            for batch_idx in tqdm(range(len(self._dataloader_train)), desc=f"Epoch {epoch + 1}/{self._conf.nepoch}"):
                batch = next(train_iter)

                # Method-specific forward pass
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        losses = self.process_train_video(batch)
                else:
                    losses = self.process_train_video(batch)

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
                    print(mn)

            # Save full-state checkpoint
            self._save_checkpoint(epoch)

            # End-of-epoch evaluation (skip if no test dataset)
            if self._dataloader_test is not None:
                self._evaluate_after_epoch()

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
        print("━" * 60)
        print(f"  Method   : {self._conf.method_name}")
        print(f"  Temporal : {self.is_temporal()}")
        print(f"  Mode     : {self._conf.mode}")
        print(f"  Features : {getattr(self._conf, 'feature_model', 'unknown')}")
        print(f"  Epochs   : {self._starting_epoch} → {self._conf.nepoch}")
        print("━" * 60)
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
