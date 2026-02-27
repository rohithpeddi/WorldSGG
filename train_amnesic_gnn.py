"""
Amnesic GNN Training Script
=============================

Frame-shuffled training for the stateless Amnesic Geometric GNN baseline.

Key design: frames from ALL videos are exploded into individual samples
and fully shuffled, preventing any implicit temporal leakage. Each sample
is a single frame processed in a pure feed-forward manner.

Usage:
    python train_amnesic_gnn.py --data_path /data/rohith/ag --nepoch 20 --use_wandb
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from constants import Constants as const
from dataloader.world_ag_dataset import WorldAG, world_collate_fn
from lib.supervised.worldsgg.amnesic_gnn.config import AmnesicGNNConfig
from lib.supervised.worldsgg.amnesic_gnn.amnesic_gnn import AmnesicGNN
from lib.supervised.worldsgg.amnesic_gnn.loss import AmnesicGNNLoss


# ---------------------------------------------------------------------------
# Frame-Shuffled Dataset
# ---------------------------------------------------------------------------

class FrameShuffledDataset(Dataset):
    """
    Explodes WorldAG into individual frames and shuffles them.

    Each __getitem__ returns a single frame's tensors, preventing any
    temporal leakage during training. This is CRUCIAL for the scientific
    validity of the amnesic baseline — the model must not implicitly
    learn sequence patterns.

    Args:
        world_ag: WorldAG dataset instance.
        max_objects: Max objects per frame (for padding).
    """

    def __init__(self, world_ag: WorldAG, max_objects: int = 64):
        super().__init__()
        self.world_ag = world_ag
        self.max_objects = max_objects

        # Build flat index: (video_idx, frame_idx) for every frame
        self.frame_index = []
        for video_idx in range(len(world_ag)):
            gt_ann = world_ag.gt_annotations[video_idx]
            T = len(gt_ann)
            for frame_idx in range(T):
                self.frame_index.append((video_idx, frame_idx))

        print(f"  FrameShuffledDataset: {len(self.frame_index)} total frames "
              f"from {len(world_ag)} videos")

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx):
        video_idx, frame_idx = self.frame_index[idx]

        # Get full video tensors (cached per video)
        batch = self.world_ag.get_glstgn_tensors(
            video_idx, max_objects=self.max_objects,
        )

        # Extract single frame
        return {
            "corners": batch["corners_seq"][frame_idx],
            "valid_mask": batch["valid_mask_seq"][frame_idx],
            "visibility_mask": batch["visibility_mask_seq"][frame_idx],
            "object_classes": batch["object_classes_seq"][frame_idx],
            "person_idx": batch["person_idx_seq"][frame_idx],
            "object_idx": batch["object_idx_seq"][frame_idx],
            "gt_attention": batch["gt_attention_seq"][frame_idx],
            "gt_spatial": batch["gt_spatial_seq"][frame_idx],
            "gt_contacting": batch["gt_contacting_seq"][frame_idx],
            "N_max": batch["N_max"],
        }


def frame_collate_fn(batch):
    """Collate function for single-frame samples (no stacking — variable sizes)."""
    return batch[0]  # batch_size=1, return the single frame dict


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TrainAmnesicGNN:
    """
    Frame-shuffled training pipeline for the Amnesic GNN baseline.

    Key differences from GL-STGN/AMWAE:
      - No temporal processing
      - Frames shuffled across all videos
      - Standard batched training (no BPTT, no memory resets)
      - Stratified loss logging (Vis-Vis / Vis-Unseen / Unseen-Unseen)
      - Should be 3-5× faster to train
    """

    def __init__(self, conf: AmnesicGNNConfig):
        self.conf = conf
        self.device = torch.device(
            f"cuda:{const.CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.global_step = 0

        self.checkpoint_dir = os.path.join(conf.save_path, conf.method_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def init_and_run(self):
        print("=" * 70)
        print("Amnesic Geometric GNN — Baseline 2 (Zero Memory)")
        print("=" * 70)

        self._init_datasets()
        self._init_model()
        self._init_optimizer()
        self._init_wandb()
        self._load_checkpoint()

        print(f"\n{'=' * 70}")
        print(f"Starting training: {self.conf.nepoch} epochs")
        print(f"  Device: {self.device}")
        print(f"  Train frames: {len(self.train_dataset)}")
        print(f"  Test frames: {len(self.test_dataset)}")
        print(f"  Frame-shuffled: YES (no temporal leakage)")
        print(f"  Memory: NONE (amnesic)")
        print(f"{'=' * 70}\n")

        self._train()

    def _init_datasets(self):
        print("[1/4] Loading datasets (frame-shuffled)...")
        world_sg_dir = self.conf.world_sg_dir or None

        train_world_ag = WorldAG(
            phase="train",
            data_path=self.conf.data_path,
            world_sg_dir=world_sg_dir,
            filter_nonperson_box_frame=True,
            include_invisible=self.conf.include_invisible,
        )
        test_world_ag = WorldAG(
            phase="test",
            data_path=self.conf.data_path,
            world_sg_dir=world_sg_dir,
            filter_nonperson_box_frame=True,
            include_invisible=self.conf.include_invisible,
        )

        self.train_dataset = FrameShuffledDataset(train_world_ag, self.conf.max_objects)
        self.test_dataset = FrameShuffledDataset(test_world_ag, self.conf.max_objects)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # Each frame is already a full graph
            shuffle=True,  # CRUCIAL: complete frame shuffling
            collate_fn=frame_collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=frame_collate_fn,
            num_workers=0,
            pin_memory=False,
        )

    def _init_model(self):
        print("[2/4] Initializing Amnesic GNN model...")

        train_world_ag = self.train_dataset.world_ag
        num_obj = len(train_world_ag.object_classes)
        att_cls = len(train_world_ag.attention_relationships)
        spa_cls = len(train_world_ag.spatial_relationships)
        con_cls = len(train_world_ag.contacting_relationships)

        self.model = AmnesicGNN(
            config=self.conf,
            num_object_classes=num_obj,
            attention_class_num=att_cls,
            spatial_class_num=spa_cls,
            contact_class_num=con_cls,
        ).to(self.device)

        self.loss_fn = AmnesicGNNLoss(
            bce_loss=self.conf.bce_loss,
            mode=self.conf.mode,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Amnesic GNN trainable parameters: {n_params:,}")

    def _init_optimizer(self):
        print("[3/4] Initializing optimizer...")
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

    def _init_wandb(self):
        if self.conf.use_wandb and wandb is not None:
            wandb.init(
                project="WorldSGG",
                name=f"amnesic_gnn_{self.conf.mode}",
                config=vars(self.conf),
            )

    def _load_checkpoint(self):
        if not self.conf.ckpt or not os.path.exists(self.conf.ckpt):
            return
        print(f"[4/4] Loading checkpoint: {self.conf.ckpt}")
        state = torch.load(self.conf.ckpt, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "global_step" in state:
            self.global_step = state["global_step"]

    def _save_checkpoint(self, epoch: int):
        ckpt_path = os.path.join(
            self.checkpoint_dir, f"amnesic_gnn_{self.conf.mode}_epoch{epoch}.pt",
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
        for epoch in range(self.conf.nepoch):
            print(f"\n{'─' * 50}")
            print(f"Epoch {epoch + 1}/{self.conf.nepoch}")
            print(f"{'─' * 50}")

            self.model.train()
            epoch_start = time.time()
            epoch_losses = []

            for frame_data in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                try:
                    loss_dict = self._train_one_frame(frame_data)
                except Exception as e:
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
                          f"vv_att={avg.get('vis_vis_att', 0):.4f} | "
                          f"vu_att={avg.get('vis_unseen_att', 0):.4f} | "
                          f"uu_att={avg.get('unseen_unseen_att', 0):.4f}")

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

    def _train_one_frame(self, frame_data: dict) -> dict:
        """Process a single frame (feed-forward, no temporal state)."""
        corners = frame_data["corners"].to(self.device)
        valid_mask = frame_data["valid_mask"].to(self.device)
        vis_mask = frame_data["visibility_mask"].to(self.device)
        person_idx = frame_data["person_idx"].to(self.device)
        object_idx = frame_data["object_idx"].to(self.device)
        obj_classes = frame_data["object_classes"].to(self.device)
        gt_attention = frame_data["gt_attention"]
        gt_spatial = frame_data["gt_spatial"]
        gt_contacting = frame_data["gt_contacting"]
        N_max = frame_data["N_max"]

        if person_idx.shape[0] == 0:
            return None

        # Zero visual features (or use detector if available)
        visual_features = torch.zeros(
            N_max, self.conf.d_detector_roi, device=self.device,
        )

        self.optimizer.zero_grad()

        if self.conf.use_amp and self.scaler is not None:
            with torch.amp.autocast("cuda"):
                predictions = self.model(
                    visual_features=visual_features,
                    corners=corners,
                    valid_mask=valid_mask,
                    visibility_mask=vis_mask,
                    person_idx=person_idx,
                    object_idx=object_idx,
                )
                losses = self.loss_fn(
                    predictions=predictions,
                    gt_attention=gt_attention,
                    gt_spatial=gt_spatial,
                    gt_contacting=gt_contacting,
                    visibility_mask=vis_mask,
                    person_idx=person_idx,
                    object_idx=object_idx,
                    valid_mask=valid_mask,
                    gt_node_labels=obj_classes,
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.model(
                visual_features=visual_features,
                corners=corners,
                valid_mask=valid_mask,
                visibility_mask=vis_mask,
                person_idx=person_idx,
                object_idx=object_idx,
            )
            losses = self.loss_fn(
                predictions=predictions,
                gt_attention=gt_attention,
                gt_spatial=gt_spatial,
                gt_contacting=gt_contacting,
                visibility_mask=vis_mask,
                person_idx=person_idx,
                object_idx=object_idx,
                valid_mask=valid_mask,
                gt_node_labels=obj_classes,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.grad_clip)
            self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, epoch: int) -> float:
        print(f"\n  Evaluating epoch {epoch + 1}...")
        self.model.eval()
        eval_losses = []

        with torch.no_grad():
            for frame_data in tqdm(self.test_loader, desc="Eval"):
                try:
                    corners = frame_data["corners"].to(self.device)
                    valid_mask = frame_data["valid_mask"].to(self.device)
                    vis_mask = frame_data["visibility_mask"].to(self.device)
                    person_idx = frame_data["person_idx"].to(self.device)
                    object_idx = frame_data["object_idx"].to(self.device)
                    obj_classes = frame_data["object_classes"].to(self.device)
                    N_max = frame_data["N_max"]

                    if person_idx.shape[0] == 0:
                        continue

                    visual_features = torch.zeros(
                        N_max, self.conf.d_detector_roi, device=self.device,
                    )

                    predictions = self.model(
                        visual_features=visual_features,
                        corners=corners,
                        valid_mask=valid_mask,
                        visibility_mask=vis_mask,
                        person_idx=person_idx,
                        object_idx=object_idx,
                    )
                    losses = self.loss_fn(
                        predictions=predictions,
                        gt_attention=frame_data["gt_attention"],
                        gt_spatial=frame_data["gt_spatial"],
                        gt_contacting=frame_data["gt_contacting"],
                        visibility_mask=vis_mask,
                        person_idx=person_idx,
                        object_idx=object_idx,
                        valid_mask=valid_mask,
                        gt_node_labels=obj_classes,
                    )
                    eval_losses.append({k: v.item() for k, v in losses.items()})
                except Exception:
                    continue

        if eval_losses:
            avg = self._average_losses(eval_losses)
            print("\n  Eval results (stratified):")
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
    conf = AmnesicGNNConfig.from_args()
    trainer = TrainAmnesicGNN(conf)
    trainer.init_and_run()


if __name__ == "__main__":
    main()
