"""
LKS GNN Training Script
=========================

Sequential rollouts with detached memory for the LKS Buffer baseline.

Key: Frames processed in temporal order within each video, but gradients
do NOT flow through time (memory buffer uses .detach()). No BPTT needed.

Usage:
    python train_lks_gnn.py --data_path /data/rohith/ag --nepoch 20 --use_wandb
"""

import os
import time

import numpy as np
import torch
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
from lib.supervised.worldsgg.lks_buffer.config import LKSConfig
from lib.supervised.worldsgg.lks_buffer.lks_gnn import LKSGNN
from lib.supervised.worldsgg.lks_buffer.loss import LKSLoss


class TrainLKSGNN:
    """
    Sequential training pipeline for the LKS Buffer baseline.

    Key differences from other methods:
      - Sequential frame processing (NOT shuffled like Amnesic GNN)
      - Memory buffer resets per video, fills up as camera explores
      - Per-frame forward + backward (no BPTT — buffer is detached)
      - Faster than GL-STGN (no through-time gradients)
    """

    def __init__(self, conf: LKSConfig):
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
        self.global_step = 0

        self.checkpoint_dir = os.path.join(conf.save_path, conf.method_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def init_and_run(self):
        print("=" * 70)
        print("LKS Buffer — Baseline 1 (Passive Memory)")
        print("  'Objects freeze when the camera looks away'")
        print("=" * 70)

        self._init_datasets()
        self._init_model()
        self._init_optimizer()
        self._init_wandb()
        self._load_checkpoint()

        print(f"\n{'=' * 70}")
        print(f"Starting training: {self.conf.nepoch} epochs")
        print(f"  Device: {self.device}")
        print(f"  Train videos: {len(self.train_dataset)}")
        print(f"  Test videos: {len(self.test_dataset)}")
        print(f"  Sequential: YES (memory builds up per video)")
        print(f"  BPTT: NO (buffer is detached)")
        print(f"{'=' * 70}\n")

        self._train()

    def _init_datasets(self):
        print("[1/4] Loading datasets...")
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

    def _init_model(self):
        print("[2/4] Initializing LKS GNN model...")
        num_obj = len(self.train_dataset.object_classes)
        att_cls = len(self.train_dataset.attention_relationships)
        spa_cls = len(self.train_dataset.spatial_relationships)
        con_cls = len(self.train_dataset.contacting_relationships)

        self.model = LKSGNN(
            config=self.conf,
            num_object_classes=num_obj,
            attention_class_num=att_cls,
            spatial_class_num=spa_cls,
            contact_class_num=con_cls,
        ).to(self.device)

        self.loss_fn = LKSLoss(
            bce_loss=self.conf.bce_loss,
            mode=self.conf.mode,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  LKS GNN trainable parameters: {n_params:,}")

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
            wandb.init(project="WorldSGG", name=f"lks_{self.conf.mode}", config=vars(self.conf))

    def _load_checkpoint(self):
        if not self.conf.ckpt or not os.path.exists(self.conf.ckpt):
            return
        state = torch.load(self.conf.ckpt, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "global_step" in state:
            self.global_step = state["global_step"]

    def _save_checkpoint(self, epoch: int):
        ckpt_path = os.path.join(
            self.checkpoint_dir, f"lks_{self.conf.mode}_epoch{epoch}.pt",
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
    # Training
    # ------------------------------------------------------------------

    def _train(self):
        for epoch in range(self.conf.nepoch):
            print(f"\n{'─' * 50}")
            print(f"Epoch {epoch + 1}/{self.conf.nepoch}")
            print(f"{'─' * 50}")

            self.model.train()
            epoch_start = time.time()
            epoch_losses = []

            for video_idx in tqdm(range(len(self.train_dataset)), desc=f"Epoch {epoch + 1}"):
                try:
                    video_losses = self._train_one_video(video_idx)
                except Exception as e:
                    print(f"\n  [ERROR] Video {video_idx}: {e}")
                    continue

                epoch_losses.extend(video_losses)
                self.global_step += 1

                if self.global_step % self.conf.log_every == 0 and epoch_losses:
                    avg = self._average_losses(epoch_losses[-500:])
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

    def _train_one_video(self, video_idx: int) -> list:
        """
        Process one video sequentially. Per-frame forward + backward.

        Memory buffer resets at start, fills up as camera explores.
        Gradients only flow through current frame (buffer is detached).
        """
        batch = self.train_dataset.get_glstgn_tensors(
            video_idx, max_objects=self.conf.max_objects,
        )
        T = batch["T"]
        if T < 2:
            return []

        # Reset memory for this video
        self.model.reset_memory(self.device)

        N_max = batch["N_max"]
        frame_losses = []

        for t in range(T):
            corners_t = batch["corners_seq"][t].to(self.device)
            valid_t = batch["valid_mask_seq"][t].to(self.device)
            vis_t = batch["visibility_mask_seq"][t].to(self.device)
            person_idx_t = batch["person_idx_seq"][t].to(self.device)
            object_idx_t = batch["object_idx_seq"][t].to(self.device)
            obj_classes_t = batch["object_classes_seq"][t].to(self.device)
            gt_att_t = batch["gt_attention_seq"][t]
            gt_spa_t = batch["gt_spatial_seq"][t]
            gt_con_t = batch["gt_contacting_seq"][t]

            if person_idx_t.shape[0] == 0:
                continue

            visual_t = torch.zeros(N_max, self.conf.d_detector_roi, device=self.device)

            self.optimizer.zero_grad()

            if self.conf.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    predictions = self.model.forward_frame(
                        visual_features=visual_t,
                        corners=corners_t,
                        valid_mask=valid_t,
                        visibility_mask=vis_t,
                        person_idx=person_idx_t,
                        object_idx=object_idx_t,
                    )
                    losses = self.loss_fn(
                        predictions=predictions,
                        gt_attention=gt_att_t,
                        gt_spatial=gt_spa_t,
                        gt_contacting=gt_con_t,
                        visibility_mask=vis_t,
                        person_idx=person_idx_t,
                        object_idx=object_idx_t,
                        valid_mask=valid_t,
                        gt_node_labels=obj_classes_t,
                    )
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model.forward_frame(
                    visual_features=visual_t,
                    corners=corners_t,
                    valid_mask=valid_t,
                    visibility_mask=vis_t,
                    person_idx=person_idx_t,
                    object_idx=object_idx_t,
                )
                losses = self.loss_fn(
                    predictions=predictions,
                    gt_attention=gt_att_t,
                    gt_spatial=gt_spa_t,
                    gt_contacting=gt_con_t,
                    visibility_mask=vis_t,
                    person_idx=person_idx_t,
                    object_idx=object_idx_t,
                    valid_mask=valid_t,
                    gt_node_labels=obj_classes_t,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.grad_clip)
                self.optimizer.step()

            frame_losses.append({k: v.item() for k, v in losses.items()})

        return frame_losses

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, epoch: int) -> float:
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

                self.model.reset_memory(self.device)
                N_max = batch["N_max"]

                for t in range(T):
                    corners_t = batch["corners_seq"][t].to(self.device)
                    valid_t = batch["valid_mask_seq"][t].to(self.device)
                    vis_t = batch["visibility_mask_seq"][t].to(self.device)
                    person_idx_t = batch["person_idx_seq"][t].to(self.device)
                    object_idx_t = batch["object_idx_seq"][t].to(self.device)
                    obj_classes_t = batch["object_classes_seq"][t].to(self.device)

                    if person_idx_t.shape[0] == 0:
                        continue

                    visual_t = torch.zeros(N_max, self.conf.d_detector_roi, device=self.device)

                    predictions = self.model.forward_frame(
                        visual_features=visual_t,
                        corners=corners_t,
                        valid_mask=valid_t,
                        visibility_mask=vis_t,
                        person_idx=person_idx_t,
                        object_idx=object_idx_t,
                    )
                    losses = self.loss_fn(
                        predictions=predictions,
                        gt_attention=batch["gt_attention_seq"][t],
                        gt_spatial=batch["gt_spatial_seq"][t],
                        gt_contacting=batch["gt_contacting_seq"][t],
                        visibility_mask=vis_t,
                        person_idx=person_idx_t,
                        object_idx=object_idx_t,
                        valid_mask=valid_t,
                        gt_node_labels=obj_classes_t,
                    )
                    eval_losses.append({k: v.item() for k, v in losses.items()})

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
    conf = LKSConfig.from_args()
    trainer = TrainLKSGNN(conf)
    trainer.init_and_run()


if __name__ == "__main__":
    main()
