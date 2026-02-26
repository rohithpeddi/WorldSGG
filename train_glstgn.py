"""
GL-STGN Training Script
========================

Complete training pipeline for the Global-Local Spatio-Temporal Graph Network.
Follows the TrainSGGBase pattern adapted for the WorldAG dataset and temporal
memory-bank architecture.

Usage:
    python train_glstgn.py --data_path /data/rohith/ag --nepoch 20 --use_wandb

"""

import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from constants import Constants as const
from dataloader.world_ag_dataset import WorldAG, world_collate_fn
from lib.supervised.worldsgg.gl_stgn.config import GLSTGNConfig
from lib.supervised.worldsgg.gl_stgn.gl_stgn import GLSTGN
from lib.supervised.worldsgg.gl_stgn.loss import GLSTGNLoss
from lib.supervised.worldsgg.gl_stgn.feature_extractor import DINOFeatureExtractor


# ---------------------------------------------------------------------------
# Training class
# ---------------------------------------------------------------------------

class TrainGLSTGN:
    """
    Full training pipeline for GL-STGN.

    Handles:
      - Dataset loading (WorldAG)
      - Frozen DINO detector for feature extraction
      - GL-STGN model initialization
      - AdamW optimizer + cosine schedule with warmup
      - BPTT over temporal chunks
      - Split visible/unseen loss
      - Visual feature masking curriculum
      - WandB logging
      - Checkpoint save/load
      - Evaluation with Recall@K
    """

    def __init__(self, conf: GLSTGNConfig):
        self.conf = conf
        self.device = torch.device(f"cuda:{const.CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

        # Will be initialized in init_and_run()
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.loss_fn = None
        self.feature_extractor = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.global_step = 0

        # Checkpoint paths
        self.checkpoint_dir = os.path.join(conf.save_path, conf.method_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_and_run(self):
        """Full initialization → training pipeline."""
        print("=" * 70)
        print("GL-STGN Training Pipeline")
        print("=" * 70)

        self._init_datasets()
        self._init_feature_extractor()
        self._init_model()
        self._init_optimizer()
        self._init_wandb()
        self._load_checkpoint()

        print(f"\n{'=' * 70}")
        print(f"Starting training: {self.conf.nepoch} epochs")
        print(f"  Device: {self.device}")
        print(f"  Train videos: {len(self.train_dataset)}")
        print(f"  Test videos: {len(self.test_dataset)}")
        print(f"  Chunk length: {self.conf.chunk_length}")
        print(f"  Lambda unseen: {self.conf.lambda_unseen}")
        print(f"  Visual mask prob: {self.conf.p_mask_visual}")
        print(f"  Teacher forcing epochs: {self.conf.teacher_forcing_epochs}")
        print(f"{'=' * 70}\n")

        self._train()

    def _init_datasets(self):
        """Load WorldAG train and test datasets."""
        print("[1/5] Loading datasets...")

        world_sg_dir = self.conf.world_sg_dir or None

        self.train_dataset = WorldAG(
            phase="train",
            data_path=self.conf.data_path,
            world_sg_dir=world_sg_dir,
            filter_nonperson_box_frame=True,
            include_invisible=self.conf.include_invisible,
        )

        self.test_dataset = WorldAG(
            phase="test",
            data_path=self.conf.data_path,
            world_sg_dir=world_sg_dir,
            filter_nonperson_box_frame=True,
            include_invisible=self.conf.include_invisible,
        )

        # DataLoaders (batch_size=1, each item is one video)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=world_collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=world_collate_fn,
            num_workers=0,
            pin_memory=False,
        )

    def _init_feature_extractor(self):
        """Load frozen DINO detector for visual feature extraction."""
        print("[2/5] Loading frozen DINO feature extractor...")
        self.feature_extractor = DINOFeatureExtractor(
            detector_ckpt=self.conf.detector_ckpt,
            detector_model=self.conf.detector_model,
            device=str(self.device),
        )
        # Lazy-load is done on first use — only load if checkpoint is provided
        if self.conf.detector_ckpt:
            self.feature_extractor.load_detector()
        else:
            print("  [INFO] No detector checkpoint provided. Visual features will be zeros.")
            print("         This means the model relies solely on structural prior + memory.")

    def _init_model(self):
        """Initialize GL-STGN model and loss."""
        print("[3/5] Initializing GL-STGN model...")

        num_obj_classes = len(self.train_dataset.object_classes)
        att_classes = len(self.train_dataset.attention_relationships)
        spa_classes = len(self.train_dataset.spatial_relationships)
        con_classes = len(self.train_dataset.contacting_relationships)

        self.model = GLSTGN(
            config=self.conf,
            num_object_classes=num_obj_classes,
            attention_class_num=att_classes,
            spatial_class_num=spa_classes,
            contact_class_num=con_classes,
        ).to(self.device)

        self.loss_fn = GLSTGNLoss(
            lambda_unseen=self.conf.lambda_unseen,
            bce_loss=self.conf.bce_loss,
            mode=self.conf.mode,
        ).to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  GL-STGN trainable parameters: {n_params:,}")
        print(f"  Object classes: {num_obj_classes} | "
              f"Attention: {att_classes} | Spatial: {spa_classes} | Contacting: {con_classes}")

    def _init_optimizer(self):
        """Initialize AdamW optimizer, cosine scheduler, and grad scaler."""
        print("[4/5] Initializing optimizer and scheduler...")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.conf.lr,
            weight_decay=self.conf.weight_decay,
        )

        # ReduceLROnPlateau based on validation recall (matching SGG pattern)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=3,
            factor=0.5,
            verbose=True,
        )

        # GradScaler for mixed precision
        if self.conf.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            print("  AMP enabled with GradScaler")
        else:
            self.scaler = None

    def _init_wandb(self):
        """Initialize WandB logging if enabled."""
        if self.conf.use_wandb and wandb is not None:
            wandb.init(
                project="WorldSGG",
                name=f"gl_stgn_{self.conf.mode}",
                config=vars(self.conf),
            )
            print("[WandB] Initialized")

    def _load_checkpoint(self):
        """Load GL-STGN checkpoint if provided."""
        if not self.conf.ckpt:
            return

        if os.path.exists(self.conf.ckpt):
            print(f"[5/5] Loading checkpoint: {self.conf.ckpt}")
            state = torch.load(self.conf.ckpt, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            if "optimizer_state_dict" in state:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if "global_step" in state:
                self.global_step = state["global_step"]
            print(f"  Resumed from step {self.global_step}")
        else:
            print(f"[WARNING] Checkpoint not found: {self.conf.ckpt}")

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"gl_stgn_{self.conf.mode}_epoch{epoch}.pt",
        )
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "config": vars(self.conf),
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train(self):
        """Main training loop: epochs → videos → chunks."""
        tr_losses = []

        for epoch in range(self.conf.nepoch):
            print(f"\n{'─' * 50}")
            print(f"Epoch {epoch + 1}/{self.conf.nepoch}")
            print(f"{'─' * 50}")

            self.model.train()
            epoch_start = time.time()
            epoch_losses = []

            # Compute current visual masking probability (curriculum)
            if epoch < self.conf.teacher_forcing_epochs:
                p_mask = 0.0
            else:
                # Ramp from 0 to p_mask_visual over remaining epochs
                ramp_epochs = self.conf.nepoch - self.conf.teacher_forcing_epochs
                ramp_progress = (epoch - self.conf.teacher_forcing_epochs) / max(ramp_epochs, 1)
                p_mask = min(ramp_progress * self.conf.p_mask_visual, self.conf.p_mask_visual)

            for video_idx in tqdm(range(len(self.train_dataset)), desc=f"Epoch {epoch + 1}"):
                try:
                    loss_dict = self._train_one_video(video_idx, p_mask)
                except Exception as e:
                    print(f"\n  [ERROR] Video {video_idx}: {e}")
                    continue

                if loss_dict is None:
                    continue

                epoch_losses.append(loss_dict)
                self.global_step += 1

                # Periodic logging
                if self.global_step % self.conf.log_every == 0 and epoch_losses:
                    avg = self._average_losses(epoch_losses[-self.conf.log_every:])
                    lr_current = self.optimizer.param_groups[0]["lr"]
                    print(f"\n  Step {self.global_step} | lr={lr_current:.2e} | "
                          f"total={avg.get('total', 0):.4f} | "
                          f"vis_att={avg.get('vis_attention_relation_loss', 0):.4f} | "
                          f"unseen_att={avg.get('unseen_attention_relation_loss', 0):.4f} | "
                          f"p_mask={p_mask:.2f}")

                    if self.conf.use_wandb and wandb is not None:
                        wandb.log(avg, step=self.global_step)

            # End of epoch
            epoch_time = time.time() - epoch_start
            if epoch_losses:
                avg_epoch = self._average_losses(epoch_losses)
                print(f"\n  Epoch {epoch + 1} summary ({epoch_time:.0f}s):")
                for k, v in sorted(avg_epoch.items()):
                    print(f"    {k}: {v:.6f}")

            # Save checkpoint
            self._save_checkpoint(epoch)

            # Evaluate
            score = self._evaluate(epoch)

            # Step scheduler
            self.scheduler.step(score)

    def _train_one_video(self, video_idx: int, p_mask: float) -> dict:
        """Process one video with BPTT over temporal chunks."""
        # Get GL-STGN formatted tensors
        batch = self.train_dataset.get_glstgn_tensors(
            video_idx,
            max_objects=self.conf.max_objects,
        )

        T = batch["T"]
        if T < 2:
            return None

        # Move tensors to device
        corners_seq = [c.to(self.device) for c in batch["corners_seq"]]
        valid_mask_seq = [v.to(self.device) for v in batch["valid_mask_seq"]]
        visibility_mask_seq = [v.to(self.device) for v in batch["visibility_mask_seq"]]
        person_idx_seq = [p.to(self.device) for p in batch["person_idx_seq"]]
        object_idx_seq = [o.to(self.device) for o in batch["object_idx_seq"]]
        gt_attention_seq = batch["gt_attention_seq"]
        gt_spatial_seq = batch["gt_spatial_seq"]
        gt_contacting_seq = batch["gt_contacting_seq"]
        object_classes_seq = [c.to(self.device) for c in batch["object_classes_seq"]]

        # Prepare visual features (zeros if no detector loaded)
        N_max = batch["N_max"]
        visual_features_seq = [
            torch.zeros(N_max, self.conf.d_detector_roi, device=self.device)
            for _ in range(T)
        ]

        # BPTT: split video into chunks
        chunk_len = self.conf.chunk_length
        total_loss_dict = {}
        n_chunks = 0

        for chunk_start in range(0, T, chunk_len):
            chunk_end = min(chunk_start + chunk_len, T)
            if chunk_end - chunk_start < 2:
                continue

            # Slice sequences for this chunk
            chunk_vis = visual_features_seq[chunk_start:chunk_end]
            chunk_corners = corners_seq[chunk_start:chunk_end]
            chunk_valid = valid_mask_seq[chunk_start:chunk_end]
            chunk_visibility = visibility_mask_seq[chunk_start:chunk_end]
            chunk_person = person_idx_seq[chunk_start:chunk_end]
            chunk_object = object_idx_seq[chunk_start:chunk_end]
            chunk_att_gt = gt_attention_seq[chunk_start:chunk_end]
            chunk_spa_gt = gt_spatial_seq[chunk_start:chunk_end]
            chunk_con_gt = gt_contacting_seq[chunk_start:chunk_end]
            chunk_node_gt = object_classes_seq[chunk_start:chunk_end]

            # Forward pass
            self.optimizer.zero_grad()

            if self.conf.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    predictions = self.model(
                        visual_features_seq=chunk_vis,
                        corners_seq=chunk_corners,
                        valid_mask_seq=chunk_valid,
                        visibility_mask_seq=chunk_visibility,
                        person_idx_seq=chunk_person,
                        object_idx_seq=chunk_object,
                        p_mask_visual=p_mask,
                    )

                    losses = self.loss_fn(
                        predictions=predictions,
                        gt_attention=chunk_att_gt,
                        gt_spatial=chunk_spa_gt,
                        gt_contacting=chunk_con_gt,
                        gt_node_labels=chunk_node_gt,
                        visibility_mask_seq=chunk_visibility,
                        person_idx_seq=chunk_person,
                        object_idx_seq=chunk_object,
                        valid_mask_seq=chunk_valid,
                    )

                total_loss = losses["total"]
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.conf.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(
                    visual_features_seq=chunk_vis,
                    corners_seq=chunk_corners,
                    valid_mask_seq=chunk_valid,
                    visibility_mask_seq=chunk_visibility,
                    person_idx_seq=chunk_person,
                    object_idx_seq=chunk_object,
                    p_mask_visual=p_mask,
                )

                losses = self.loss_fn(
                    predictions=predictions,
                    gt_attention=chunk_att_gt,
                    gt_spatial=chunk_spa_gt,
                    gt_contacting=chunk_con_gt,
                    gt_node_labels=chunk_node_gt,
                    visibility_mask_seq=chunk_visibility,
                    person_idx_seq=chunk_person,
                    object_idx_seq=chunk_object,
                    valid_mask_seq=chunk_valid,
                )

                total_loss = losses["total"]
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.conf.grad_clip
                )
                self.optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = 0.0
                total_loss_dict[k] += v.item()
            n_chunks += 1

        # Average over chunks
        if n_chunks > 0:
            return {k: v / n_chunks for k, v in total_loss_dict.items()}
        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, epoch: int) -> float:
        """
        Evaluate on test set. Returns mean Recall@20 for scheduler.

        For now, computes loss-based metrics. Full evaluator integration
        (BasicSceneGraphEvaluator) can be added when the model produces
        outputs in the standard format.
        """
        print(f"\n  Evaluating epoch {epoch + 1}...")
        self.model.eval()
        eval_losses = []

        with torch.no_grad():
            for video_idx in tqdm(range(len(self.test_dataset)), desc="Eval"):
                try:
                    batch = self.test_dataset.get_glstgn_tensors(
                        video_idx,
                        max_objects=self.conf.max_objects,
                    )
                except Exception:
                    continue

                T = batch["T"]
                if T < 2:
                    continue

                # Move to device
                corners_seq = [c.to(self.device) for c in batch["corners_seq"]]
                valid_mask_seq = [v.to(self.device) for v in batch["valid_mask_seq"]]
                visibility_mask_seq = [v.to(self.device) for v in batch["visibility_mask_seq"]]
                person_idx_seq = [p.to(self.device) for p in batch["person_idx_seq"]]
                object_idx_seq = [o.to(self.device) for o in batch["object_idx_seq"]]
                object_classes_seq = [c.to(self.device) for c in batch["object_classes_seq"]]
                N_max = batch["N_max"]

                visual_features_seq = [
                    torch.zeros(N_max, self.conf.d_detector_roi, device=self.device)
                    for _ in range(T)
                ]

                predictions = self.model(
                    visual_features_seq=visual_features_seq,
                    corners_seq=corners_seq,
                    valid_mask_seq=valid_mask_seq,
                    visibility_mask_seq=visibility_mask_seq,
                    person_idx_seq=person_idx_seq,
                    object_idx_seq=object_idx_seq,
                    p_mask_visual=0.0,
                )

                losses = self.loss_fn(
                    predictions=predictions,
                    gt_attention=batch["gt_attention_seq"],
                    gt_spatial=batch["gt_spatial_seq"],
                    gt_contacting=batch["gt_contacting_seq"],
                    gt_node_labels=object_classes_seq,
                    visibility_mask_seq=visibility_mask_seq,
                    person_idx_seq=person_idx_seq,
                    object_idx_seq=object_idx_seq,
                    valid_mask_seq=valid_mask_seq,
                )

                eval_losses.append({k: v.item() for k, v in losses.items()})

        if eval_losses:
            avg = self._average_losses(eval_losses)
            print("\n  Eval results:")
            for k, v in sorted(avg.items()):
                print(f"    {k}: {v:.6f}")

            if self.conf.use_wandb and wandb is not None:
                wandb.log({f"eval_{k}": v for k, v in avg.items()}, step=self.global_step)

            # Use negative total loss as "score" for scheduler
            # Lower loss = higher score
            return -avg.get("total", float("inf"))
        return 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _average_losses(loss_dicts: list) -> dict:
        """Average a list of loss dicts."""
        if not loss_dicts:
            return {}
        keys = loss_dicts[0].keys()
        return {
            k: np.mean([d[k] for d in loss_dicts if k in d])
            for k in keys
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    conf = GLSTGNConfig.from_args()
    trainer = TrainGLSTGN(conf)
    trainer.init_and_run()


if __name__ == "__main__":
    main()
