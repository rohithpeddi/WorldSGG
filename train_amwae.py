"""
AMWAE Training Script
======================

Complete training pipeline for the Associative Masked World Auto-Encoder.
Unlike GL-STGN, each frame is processed as an independent masked auto-encoding
step — no BPTT over temporal chunks needed. The episodic memory bank provides
temporal context through cross-attention retrieval.

Usage:
    python train_amwae.py --data_path /data/rohith/ag --nepoch 20 --use_wandb
"""

import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from constants import Constants as const
from dataloader.world_ag_dataset import WorldAG, world_collate_fn
from lib.supervised.worldsgg.amwae.config import AMWAEConfig
from lib.supervised.worldsgg.amwae.amwae import AMWAE
from lib.supervised.worldsgg.amwae.loss import AMWAELoss
from lib.supervised.worldsgg.gl_stgn.feature_extractor import DINOFeatureExtractor


class TrainAMWAE:
    """
    Full training pipeline for AMWAE.

    Key differences from GL-STGN training:
      - Memory bank reset per video (no BPTT across chunks)
      - Each frame = independent masked auto-encoding step
      - Three loss terms: reconstruction + split SG + contrastive InfoNCE
      - Loss backward per-frame (memory stores detached tokens)
    """

    def __init__(self, conf: AMWAEConfig):
        self.conf = conf
        self.device = torch.device(f"cuda:{const.CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

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

        self.checkpoint_dir = os.path.join(conf.save_path, conf.method_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_and_run(self):
        """Full initialization → training pipeline."""
        print("=" * 70)
        print("AMWAE Training Pipeline")
        print("  Associative Masked World Auto-Encoder")
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
        print(f"  Memory bank size: {self.conf.memory_bank_size} frames")
        print(f"  Masking prob (visible): {self.conf.p_mask_visible}")
        print(f"  Lambda masked: {self.conf.lambda_masked}")
        print(f"  Lambda recon: {self.conf.lambda_recon}")
        print(f"  Lambda contrastive: {self.conf.lambda_contrastive}")
        print(f"  InfoNCE temperature: {self.conf.temperature}")
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

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=True,
            collate_fn=world_collate_fn, num_workers=0, pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False,
            collate_fn=world_collate_fn, num_workers=0, pin_memory=False,
        )

    def _init_feature_extractor(self):
        """Load frozen DINO detector."""
        print("[2/5] Loading frozen DINO feature extractor...")
        self.feature_extractor = DINOFeatureExtractor(
            detector_ckpt=self.conf.detector_ckpt,
            detector_model=self.conf.detector_model,
            device=str(self.device),
        )
        if self.conf.detector_ckpt:
            self.feature_extractor.load_detector()
        else:
            print("  [INFO] No detector checkpoint. Visual features will be zeros.")

    def _init_model(self):
        """Initialize AMWAE model and loss."""
        print("[3/5] Initializing AMWAE model...")

        num_obj_classes = len(self.train_dataset.object_classes)
        att_classes = len(self.train_dataset.attention_relationships)
        spa_classes = len(self.train_dataset.spatial_relationships)
        con_classes = len(self.train_dataset.contacting_relationships)

        self.model = AMWAE(
            config=self.conf,
            num_object_classes=num_obj_classes,
            attention_class_num=att_classes,
            spatial_class_num=spa_classes,
            contact_class_num=con_classes,
        ).to(self.device)

        self.loss_fn = AMWAELoss(
            lambda_masked=self.conf.lambda_masked,
            lambda_recon=self.conf.lambda_recon,
            lambda_contrastive=self.conf.lambda_contrastive,
            temperature=self.conf.temperature,
            bce_loss=self.conf.bce_loss,
            mode=self.conf.mode,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  AMWAE trainable parameters: {n_params:,}")

    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        print("[4/5] Initializing optimizer and scheduler...")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.conf.lr,
            weight_decay=self.conf.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=3, factor=0.5, verbose=True,
        )
        if self.conf.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            print("  AMP enabled")

    def _init_wandb(self):
        if self.conf.use_wandb and wandb is not None:
            wandb.init(project="WorldSGG", name=f"amwae_{self.conf.mode}", config=vars(self.conf))

    def _load_checkpoint(self):
        if not self.conf.ckpt or not os.path.exists(self.conf.ckpt):
            return
        print(f"[5/5] Loading checkpoint: {self.conf.ckpt}")
        state = torch.load(self.conf.ckpt, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "global_step" in state:
            self.global_step = state["global_step"]

    def _save_checkpoint(self, epoch: int):
        ckpt_path = os.path.join(self.checkpoint_dir, f"amwae_{self.conf.mode}_epoch{epoch}.pt")
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
        """Main training loop."""
        for epoch in range(self.conf.nepoch):
            print(f"\n{'─' * 50}")
            print(f"Epoch {epoch + 1}/{self.conf.nepoch}")
            print(f"{'─' * 50}")

            self.model.train()
            epoch_start = time.time()
            epoch_losses = []

            for video_idx in tqdm(range(len(self.train_dataset)), desc=f"Epoch {epoch + 1}"):
                try:
                    loss_dict = self._train_one_video(video_idx)
                except Exception as e:
                    print(f"\n  [ERROR] Video {video_idx}: {e}")
                    continue

                if loss_dict is None:
                    continue

                epoch_losses.append(loss_dict)
                self.global_step += 1

                if self.global_step % self.conf.log_every == 0 and epoch_losses:
                    avg = self._average_losses(epoch_losses[-self.conf.log_every:])
                    lr_cur = self.optimizer.param_groups[0]["lr"]
                    print(f"\n  Step {self.global_step} | lr={lr_cur:.2e} | "
                          f"total={avg.get('total', 0):.4f} | "
                          f"recon={avg.get('recon_loss', 0):.4f} | "
                          f"contra={avg.get('contrastive_loss', 0):.4f} | "
                          f"vis_att={avg.get('vis_attention_relation_loss', 0):.4f} | "
                          f"mask_att={avg.get('masked_attention_relation_loss', 0):.4f}")

                    if self.conf.use_wandb and wandb is not None:
                        wandb.log(avg, step=self.global_step)

            epoch_time = time.time() - epoch_start
            if epoch_losses:
                avg_epoch = self._average_losses(epoch_losses)
                print(f"\n  Epoch {epoch + 1} summary ({epoch_time:.0f}s):")
                for k, v in sorted(avg_epoch.items()):
                    print(f"    {k}: {v:.6f}")

            self._save_checkpoint(epoch)
            score = self._evaluate(epoch)
            self.scheduler.step(score)

    def _train_one_video(self, video_idx: int) -> dict:
        """Process one video: reset memory → process all frames → accumulate loss."""
        batch = self.train_dataset.get_glstgn_tensors(
            video_idx, max_objects=self.conf.max_objects,
        )
        T = batch["T"]
        if T < 2:
            return None

        # Reset episodic memory for this video
        self.model.reset_memory()

        # Move tensors to device
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

        # Forward all frames (memory bank accumulates internally)
        self.optimizer.zero_grad()

        if self.conf.use_amp and self.scaler is not None:
            with torch.amp.autocast("cuda"):
                predictions = self.model(
                    visual_features_seq=visual_features_seq,
                    corners_seq=corners_seq,
                    valid_mask_seq=valid_mask_seq,
                    visibility_mask_seq=visibility_mask_seq,
                    person_idx_seq=person_idx_seq,
                    object_idx_seq=object_idx_seq,
                    p_mask_visible=self.conf.p_mask_visible,
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

            total_loss = losses["total"]
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.conf.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.model(
                visual_features_seq=visual_features_seq,
                corners_seq=corners_seq,
                valid_mask_seq=valid_mask_seq,
                visibility_mask_seq=visibility_mask_seq,
                person_idx_seq=person_idx_seq,
                object_idx_seq=object_idx_seq,
                p_mask_visible=self.conf.p_mask_visible,
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
            total_loss = losses["total"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.conf.grad_clip)
            self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, epoch: int) -> float:
        """Evaluate on test set. Returns metric for scheduler."""
        print(f"\n  Evaluating epoch {epoch + 1}...")
        self.model.eval()
        eval_losses = []

        with torch.no_grad():
            for video_idx in tqdm(range(len(self.test_dataset)), desc="Eval"):
                try:
                    batch = self.test_dataset.get_glstgn_tensors(
                        video_idx, max_objects=self.conf.max_objects,
                    )
                except Exception:
                    continue

                T = batch["T"]
                if T < 2:
                    continue

                self.model.reset_memory()

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
                    p_mask_visible=0.0,  # No masking during eval
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
            return -avg.get("total", float("inf"))
        return 0.0

    @staticmethod
    def _average_losses(loss_dicts):
        if not loss_dicts:
            return {}
        keys = loss_dicts[0].keys()
        return {k: np.mean([d[k] for d in loss_dicts if k in d]) for k in keys}


def main():
    conf = AMWAEConfig.from_args()
    trainer = TrainAMWAE(conf)
    trainer.init_and_run()


if __name__ == "__main__":
    main()
