#!/usr/bin/env python3
"""
Train DINOv2 AG object detector (class-based) + COCO-style evaluation via DetectionEvaluator.

Key change vs your original:
- Replaces evaluate_MAP_full(...) with DetectionEvaluator.evaluate_2d_map_coco(...)
- Logs COCO-style: map, map_50, map_75 (+ raw torchmetrics dict)
"""

import argparse
import gc
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from ag_dataset_resize import ActionGenomeDatasetResize as ActionGenomeDataset, collate_fn
from dinov2_torch import create_model
from utils import LocalLogger
from lib.detector.evaluate import DetectionEvaluator


def clear_cuda_cache_for_current_process(sync: bool = True) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    if sync:
        torch.cuda.synchronize()
    for dev in range(torch.cuda.device_count()):
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()


# ============================
# Helpers (kept as functions)
# ============================
def configure_rpn_and_roi(model):
    rpn = getattr(model, "rpn", None)
    if rpn is not None:
        if hasattr(rpn, "pre_nms_top_n_train") and hasattr(rpn, "post_nms_top_n_train"):
            rpn.pre_nms_top_n_train = 2000
            rpn.post_nms_top_n_train = 1000
            rpn.pre_nms_top_n_test = 2000
            rpn.post_nms_top_n_test = 500
        elif hasattr(rpn, "pre_nms_top_n") and isinstance(rpn.pre_nms_top_n, dict):
            rpn.pre_nms_top_n = {"train": 2000, "test": 2000}
            rpn.post_nms_top_n = {"train": 1000, "test": 500}

        if hasattr(rpn, "batch_size_per_image"):
            rpn.batch_size_per_image = 128
        if hasattr(rpn, "positive_fraction"):
            rpn.positive_fraction = 0.5

    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is not None:
        if hasattr(roi_heads, "batch_size_per_image"):
            roi_heads.batch_size_per_image = 256
        if hasattr(roi_heads, "detections_per_img"):
            roi_heads.detections_per_img = 200

    if hasattr(model, "transform"):
        model.transform.min_size = (512,)
        model.transform.max_size = 1024


def tensor_to_image_np(img_tensor, mean, std):
    img = img_tensor.detach().cpu().float().clone()
    mean_t = torch.tensor(mean, dtype=img.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(3, 1, 1)
    img = img * std_t + mean_t
    img = img.clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields.
    """
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict


# ============================
# Config
# ============================
@dataclass
class TrainConfig:
    experiment_name: str
    # working_dir: str = "/home/cse/msr/csy227518/scratch/Projects/Active/Practice/ag_object_detection/train_data"
    # data_path: str = "/home/cse/msr/csy227518/scratch/Datasets/action_genome"
    # save_path: str = "/home/cse/msr/csy227518/scratch/Projects/Active/Practice/ag_object_detection/save_models"
    working_dir: str = "/data/rohith/ag/"
    data_path: str = "/data/rohith/ag/"
    save_path: str = "/data/rohith/ag/detector/"
    ckpt: Optional[str] = None

    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 70

    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    use_collate: bool = True
    use_wandb: bool = True

    num_workers_train: int = 16
    num_workers_test: int = 16
    num_workers_test_subset: int = 32

    train_subset_size: int = 180000
    test_subset_size: int = 2000

    target_size: int = 1024

    # iteration logging & eval
    iter_log_every: int = 10000
    eval_every_iters: int = 10_000_000  # keep your "effectively disabled" default

    # plotting
    plot_each_epoch: bool = True
    plot_sample_idx: int = 120
    plot_score_thresh: float = 0.1


# ============================
# Trainer Class
# ============================
class DinoAGTrainer:

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.path_to_experiment = os.path.join(cfg.working_dir, cfg.experiment_name)

        self.accelerator = Accelerator(
            project_dir=self.path_to_experiment,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            log_with="wandb",
        )
        self.local_logger = LocalLogger(self.path_to_experiment)

        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.test_loader_subset = None

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.model_device = None

        self.evaluator: Optional[DetectionEvaluator] = None

        self.starting_epoch = 0
        self.global_iteration = 0

    # ----------------------------
    # Setup
    # ----------------------------
    def init_trackers(self) -> None:
        experiment_config = {
            "epochs": self.cfg.epochs,
            "Effective_batch_size": self.cfg.batch_size * self.accelerator.num_processes,
            "learning_rate": self.cfg.lr,
        }
        self.accelerator.init_trackers(self.cfg.experiment_name, config=experiment_config)

        if self.cfg.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project="DINOv2-Object-Detector-AG",
                config={
                    "learning_rate": self.cfg.lr,
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "dataset": "Action_Genome",
                },
            )

    def build_datasets(self) -> None:
        self.train_dataset = ActionGenomeDataset(self.cfg.data_path, phase="train", target_size=self.cfg.target_size)
        self.test_dataset = ActionGenomeDataset(self.cfg.data_path, phase="test", target_size=self.cfg.target_size)

    def build_dataloaders(self) -> None:
        train_subset = Subset(self.train_dataset, list(range(self.cfg.train_subset_size)))
        test_subset = Subset(self.test_dataset, list(range(self.cfg.test_subset_size)))

        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers_train,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers_test,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )
        self.test_loader_subset = DataLoader(
            test_subset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers_test_subset,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def build_model(self) -> None:
        num_classes = len(self.train_dataset.object_classes)
        model = create_model(num_classes=num_classes, pretrained=True, use_fpn=True, model="v3l")
        configure_rpn_and_roi(model)
        self.model = model

    def build_optimizer_and_scheduler(self) -> None:
        total_steps = self.cfg.epochs * len(self.train_loader)
        warmup_steps = int(0.01 * total_steps)

        params = [p for _, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=0.001)

        warmup = LinearLR(self.optimizer, start_factor=1e-1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=(total_steps - warmup_steps), eta_min=self.cfg.lr * 0.1)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    def prepare_with_accelerator(self) -> None:
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.test_loader,
            self.test_loader_subset,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.test_loader,
            self.test_loader_subset,
            self.scheduler,
        )

        self.accelerator.register_for_checkpointing(self.scheduler)

        self.model_device = next(self.model.parameters()).device
        self.model = self.model.to(self.model_device, memory_format=torch.channels_last)

        # speed toggles
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

        # NEW: DetectionEvaluator (COCO-style 2D)
        self.evaluator = DetectionEvaluator(
            device=self.model_device,
            accelerator=self.accelerator,
            frame_batch_size=10,
            iou_thresholds_2d=None,  # None => COCO default (0.5:0.95:0.05)
        )

    # ----------------------------
    # Checkpointing
    # ----------------------------
    def maybe_resume(self) -> None:
        if self.cfg.ckpt is None:
            self.starting_epoch = 0
            return

        self.accelerator.print(f"Resuming from checkpoint : {self.cfg.ckpt}")
        path_to_checkpoint = os.path.join(self.path_to_experiment, self.cfg.ckpt)
        checkpoint_file = os.path.join(path_to_checkpoint, "checkpoint_state.pth")

        if not os.path.exists(checkpoint_file):
            self.accelerator.print(f"⚠️  Checkpoint file not found at {checkpoint_file}")
            self.starting_epoch = int(self.cfg.ckpt.split("_")[-1]) if "_" in self.cfg.ckpt else 0
            return

        checkpoint_state = torch.load(checkpoint_file, map_location="cpu")
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint_state["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])
        self.starting_epoch = checkpoint_state.get("epoch", 0) + 1

        self.accelerator.print(f"✓ Loaded checkpoint from epoch {self.starting_epoch}")
        self.accelerator.print(f"  Checkpoint LR: {checkpoint_state.get('current_lr', 'N/A')}")

    def save_checkpoint(self, epoch: int) -> None:
        path_to_checkpoint = os.path.join(self.path_to_experiment, f"checkpoint_{epoch}")
        if not self.accelerator.is_main_process:
            self.accelerator.print(f"Checkpoint saved at epoch {epoch + 1}")
            return

        os.makedirs(path_to_checkpoint, exist_ok=True)
        checkpoint_file = os.path.join(path_to_checkpoint, "checkpoint_state.pth")

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_lr": self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, "get_last_lr") else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint_dict, checkpoint_file)
        self.accelerator.print(f"✓ Checkpoint saved at epoch {epoch + 1} to {checkpoint_file}")

    # ----------------------------
    # Visualization
    # ----------------------------
    def plot_predictions_after_epoch(self, epoch: int, sample_idx: int, score_threshold: float) -> Optional[str]:
        if not self.accelerator.is_main_process or not self.cfg.plot_each_epoch:
            return None

        save_dir = os.path.join(self.path_to_experiment, "visualizations")
        os.makedirs(save_dir, exist_ok=True)

        self.model.eval()
        if sample_idx >= len(self.test_dataset):
            sample_idx = 0

        image_tensor, target = self.test_dataset[sample_idx]
        sample = self.test_dataset.samples[sample_idx]

        mean = tuple(self.test_dataset.image_mean)
        std = tuple(self.test_dataset.image_std)

        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.model_device)
            predictions = self.model(image_batch)

        pred = predictions[0]
        pred_boxes = pred["boxes"].detach().cpu().numpy()
        pred_labels = pred["labels"].detach().cpu().numpy()
        pred_scores = pred["scores"].detach().cpu().numpy()

        keep = pred_scores >= score_threshold
        pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]

        gt_boxes = target["boxes"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()

        image_np = tensor_to_image_np(image_tensor, mean, std)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image_np)
        ax.set_title(
            f"Epoch {epoch + 1} | Sample: {sample['filename']}\n"
            f"GT: {len(gt_boxes)} boxes | Pred: {len(pred_boxes)} boxes (score>={score_threshold})"
        )
        ax.axis("off")

        for box, lab in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            color = "cyan"
            rect = patches.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1),
                                     linewidth=2.5, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            class_name = self.test_dataset.object_classes[int(lab)] if 0 <= int(lab) < len(
                self.test_dataset.object_classes) else str(lab)
            ax.text(
                max(0, x1), max(10, y1 - 2), f"{class_name}",
                fontsize=8, color="black", verticalalignment="top",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2")
            )

        for box, lab, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.tolist()
            color = "lime"
            rect = patches.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1),
                                     linewidth=2.0, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            class_name = self.test_dataset.object_classes[int(lab)] if 0 <= int(lab) < len(
                self.test_dataset.object_classes) else str(lab)
            ax.text(
                max(0, x1), max(10, y1 - 2), f"{class_name} ({score:.2f})",
                fontsize=8, color="black", verticalalignment="top",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2")
            )

        legend_elements = [
            Line2D([0], [0], color="cyan", lw=3, label="Ground Truth"),
            Line2D([0], [0], color="lime", lw=3, label=f"Predictions (score>={score_threshold})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        save_path = os.path.join(save_dir, f"predictions_epoch_resize_{epoch + 1:03d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # ----------------------------
    # Evaluation (COCO-style via DetectionEvaluator)
    # ----------------------------
    @staticmethod
    def _tensor_dict_to_python(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = float(v.item()) if v.numel() == 1 else float(v.float().mean().item())
            else:
                out[k] = v
        return out

    def evaluate_map_coco(self) -> Dict[str, Any]:
        """
        Replaces evaluate_MAP_full. COCO-style:
          - map (0.5:0.95)
          - map_50
          - map_75
        """
        assert self.evaluator is not None, "Evaluator not initialized."
        self.model.eval()
        with torch.no_grad():
            coco = self.evaluator.evaluate_2d_map_coco(self.model, self.test_loader)

        if not coco:
            return {}

        raw_py = self._tensor_dict_to_python(coco.get("raw", {}))
        return {
            "map": float(coco["map"]),
            "map_50": float(coco["map_50"]),
            "map_75": float(coco["map_75"]),
            "raw": raw_py,
            "map_per_class": coco.get("map_per_class", None),
        }

    # ----------------------------
    # Training
    # ----------------------------
    def _move_targets(self, targets: Any) -> Any:
        if self.cfg.use_collate:
            return [
                {k: v.to(self.model_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
        # for non-collate (rare)
        return [{k: v.to(self.model_device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}]

    def _log_iteration_losses(
            self,
            running_total_loss: float,
            running_cls_loss: float,
            running_box_loss: float,
            running_object_loss: float,
            running_rpn_loss: float,
            running_count: int,
    ) -> None:
        if running_count == 0:
            return
        avg_total_loss = running_total_loss / running_count
        avg_cls_loss = running_cls_loss / running_count
        avg_box_loss = running_box_loss / running_count
        avg_object_loss = running_object_loss / running_count
        avg_rpn_loss = running_rpn_loss / running_count
        lr = self.scheduler.get_last_lr()[0]

        if self.cfg.use_wandb and self.accelerator.is_main_process:
            wandb.log({
                "iteration": self.global_iteration,
                "iter/total_loss": avg_total_loss,
                "iter/cls_loss": avg_cls_loss,
                "iter/box_loss": avg_box_loss,
                "iter/object_loss": avg_object_loss,
                "iter/rpn_loss": avg_rpn_loss,
                "learning_rate": lr,
            })

        self.local_logger.log(
            log_type="iteration",
            iteration=self.global_iteration,
            iter_total_loss=avg_total_loss,
            iter_cls_loss=avg_cls_loss,
            iter_box_loss=avg_box_loss,
            iter_object_loss=avg_object_loss,
            iter_rpn_loss=avg_rpn_loss,
            learning_rate=lr,
        )

        self.accelerator.print(
            f"Iteration {self.global_iteration}: Loss={avg_total_loss:.4f} "
            f"(Cls:{avg_cls_loss:.4f}, Box:{avg_box_loss:.4f}, "
            f"Object:{avg_object_loss:.4f}, RPN:{avg_rpn_loss:.4f})"
        )

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        batch_loss_list = []
        batch_loss_cls_list = []
        batch_loss_box_reg_list = []
        batch_loss_objectness_list = []
        batch_loss_rpn_list = []

        running_total_loss = running_cls_loss = running_box_loss = 0.0
        running_object_loss = running_rpn_loss = 0.0
        running_count = 0

        for images, targets in tqdm(self.train_loader, ascii=True):
            images = torch.stack([img for img in images]).to(self.model_device, non_blocking=True)
            images = images.contiguous(memory_format=torch.channels_last)
            targets = self._move_targets(targets)

            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    loss_dict_original = self.model(images, targets)

                # keep your weighting hook
                box_weight = 1.0
                loss_dict: Dict[str, torch.Tensor] = {}
                for k, v in loss_dict_original.items():
                    loss_dict[k] = box_weight * v if k in ("loss_box_reg", "loss_rpn_box_reg") else v

                losses = sum(loss / self.cfg.gradient_accumulation_steps for loss in loss_dict.values())

                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss / self.cfg.gradient_accumulation_steps for loss in loss_dict_reduced.values())

                self.accelerator.backward(losses)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

                self.global_iteration += 1

                if self.accelerator.is_main_process:
                    batch_loss_list.append(losses_reduced.item())
                    batch_loss_cls_list.append(loss_dict_reduced.get("loss_classifier", torch.tensor(0.0)).item())
                    batch_loss_box_reg_list.append(loss_dict_reduced.get("loss_box_reg", torch.tensor(0.0)).item())
                    batch_loss_objectness_list.append(
                        loss_dict_reduced.get("loss_objectness", torch.tensor(0.0)).item())
                    batch_loss_rpn_list.append(loss_dict_reduced.get("loss_rpn_box_reg", torch.tensor(0.0)).item())

                    running_total_loss += losses_reduced.item()
                    running_cls_loss += loss_dict_reduced.get("loss_classifier", torch.tensor(0.0)).item()
                    running_box_loss += loss_dict_reduced.get("loss_box_reg", torch.tensor(0.0)).item()
                    running_object_loss += loss_dict_reduced.get("loss_objectness", torch.tensor(0.0)).item()
                    running_rpn_loss += loss_dict_reduced.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
                    running_count += 1

                if self.global_iteration % self.cfg.iter_log_every == 0 and self.accelerator.is_main_process:
                    self._log_iteration_losses(
                        running_total_loss, running_cls_loss, running_box_loss,
                        running_object_loss, running_rpn_loss, running_count
                    )
                    running_total_loss = running_cls_loss = running_box_loss = 0.0
                    running_object_loss = running_rpn_loss = 0.0
                    running_count = 0

                # Optional mid-epoch COCO eval (your default disables it)
                if self.global_iteration % self.cfg.eval_every_iters == 0:
                    self._maybe_eval_mid_epoch()

        return {
            "train_total_loss": float(np.mean(batch_loss_list)) if batch_loss_list else 0.0,
            "train_cls_loss": float(np.mean(batch_loss_cls_list)) if batch_loss_cls_list else 0.0,
            "train_box_loss": float(np.mean(batch_loss_box_reg_list)) if batch_loss_box_reg_list else 0.0,
            "train_object_loss": float(np.mean(batch_loss_objectness_list)) if batch_loss_objectness_list else 0.0,
            "train_rpn_loss": float(np.mean(batch_loss_rpn_list)) if batch_loss_rpn_list else 0.0,
        }

    def _maybe_eval_mid_epoch(self) -> None:
        self.accelerator.print(f"Evaluating COCO mAP at iteration {self.global_iteration}...")
        metrics = self.evaluate_map_coco()

        if self.cfg.use_wandb and self.accelerator.is_main_process:
            wandb.log({
                "iteration": self.global_iteration,
                "iter/map": metrics.get("map", 0.0),
                "iter/map_50": metrics.get("map_50", 0.0),
                "iter/map_75": metrics.get("map_75", 0.0),
            })

        self.local_logger.log(
            log_type="mAP_evaluation",
            iteration=self.global_iteration,
            mAP=metrics,
        )

        self.accelerator.print(f"Iteration {self.global_iteration}: COCO mAP = {metrics}")
        self.accelerator.wait_for_everyone()
        self.model.train()

    # ----------------------------
    # Main run
    # ----------------------------
    def run(self) -> None:
        os.makedirs(self.path_to_experiment, exist_ok=True)

        # self.init_trackers()
        self.build_datasets()
        self.build_dataloaders()
        self.build_model()
        self.build_optimizer_and_scheduler()
        self.prepare_with_accelerator()
        self.maybe_resume()

        for epoch in range(self.starting_epoch, self.cfg.epochs):
            train_stats = self.train_one_epoch(epoch)

            # Epoch-end COCO eval
            self.accelerator.print(f"Evaluating COCO mAP at end of epoch {epoch + 1}...")
            epoch_metrics = self.evaluate_map_coco()

            # Plot predictions
            if self.cfg.plot_each_epoch and self.accelerator.is_main_process:
                try:
                    self.plot_predictions_after_epoch(
                        epoch=epoch,
                        sample_idx=self.cfg.plot_sample_idx,
                        score_threshold=self.cfg.plot_score_thresh,
                    )
                    self.accelerator.print(f"✓ Prediction visualization saved for epoch {epoch + 1}")
                except Exception as e:
                    self.accelerator.print(f"⚠️  Failed to save visualization: {e}")

            # Log
            lr = self.scheduler.get_last_lr()[0]
            if self.cfg.use_wandb and self.accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch,
                    "train/total_loss": train_stats["train_total_loss"],
                    "train/cls_loss": train_stats["train_cls_loss"],
                    "train/box_loss": train_stats["train_box_loss"],
                    "train/object_loss": train_stats["train_object_loss"],
                    "train/rpn_loss": train_stats["train_rpn_loss"],
                    "epoch/map": epoch_metrics.get("map", 0.0),
                    "epoch/map_50": epoch_metrics.get("map_50", 0.0),
                    "epoch/map_75": epoch_metrics.get("map_75", 0.0),
                    "learning_rate": lr,
                })

            self.local_logger.log(
                log_type="epoch",
                epoch=epoch,
                learning_rate=lr,
                mAP=epoch_metrics,
                **train_stats,
            )

            # Print summary
            self.accelerator.print(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")
            self.accelerator.print(f"Train Loss: {train_stats['train_total_loss']:.4f}")
            self.accelerator.print(f"  Cls: {train_stats['train_cls_loss']:.4f}")
            self.accelerator.print(f"  Box: {train_stats['train_box_loss']:.4f}")
            self.accelerator.print(f"  Object: {train_stats['train_object_loss']:.4f}")
            self.accelerator.print(f"  RPN: {train_stats['train_rpn_loss']:.4f}")
            self.accelerator.print(f"COCO mAP: {epoch_metrics}")
            self.accelerator.print(f"Learning Rate: {lr:.6f}")
            self.accelerator.print("-" * 80)

            # Save checkpoint
            self.save_checkpoint(epoch)

        self.accelerator.end_training()


# ============================
# CLI
# ============================
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train DINOv2 AG object detector (class-based + COCO eval)")
    parser.add_argument("--experiment_name", required=True, type=str)
    parser.add_argument("--working_dir", default=TrainConfig.working_dir, type=str)
    parser.add_argument("--use_collate", action="store_true")
    parser.add_argument("--no_collate", action="store_false", dest="use_collate")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--data_path", type=str, default=TrainConfig.data_path)
    parser.add_argument("--save_path", type=str, default=TrainConfig.save_path)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--max_grad_norm", type=float, default=TrainConfig.max_grad_norm)

    # Keep your defaults
    parser.set_defaults(use_wandb=True)
    parser.set_defaults(use_collate=True)

    args = parser.parse_args()

    return TrainConfig(
        experiment_name=args.experiment_name,
        working_dir=args.working_dir,
        data_path=args.data_path,
        save_path=args.save_path,
        ckpt=args.ckpt,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_wandb=args.use_wandb,
        use_collate=args.use_collate,
    )


def main():
    cfg = parse_args()
    trainer = DinoAGTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
