#!/usr/bin/env python3
"""
DinoAGTrainer3D: Trainer class for DINOv2 AG Monocular 3D object detection.

This module contains the TrainConfig dataclass, helper utilities, and the
DinoAGTrainer3D class. It is intended to be driven by train.py which handles
YAML config loading and CLI override merging.
"""

import contextlib
import gc
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # Avoid /dev/shm open file limit
import wandb
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from .datasets.ag_dataset_3d import ActionGenomeDataset3D, ResolutionBucketBatchSampler, collate_fn
from .models.dino_mono_3d import DinoV3Monocular3D
from .utils.json_logger import LocalLogger
from .evaluation.evaluate_3d import evaluate_3d_metrics
from .evaluation.evaluate_2d import evaluate_2d_coco_map, evaluate_2d_and_3d_fused


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
# DummyAccelerator (non-dist)
# ============================
class DummyAccelerator:
    """
    Minimal drop-in replacement for Accelerate's Accelerator when running
    single-process / non-distributed training.
    """

    def __init__(self):
        self.num_processes = 1
        self.is_main_process = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare(self, *args):
        """Move models to GPU, pass other objects through unchanged."""
        out = []
        for obj in args:
            if isinstance(obj, torch.nn.Module):
                out.append(obj.to(self.device))
            else:
                out.append(obj)
        return tuple(out)

    def accumulate(self, model):
        return contextlib.nullcontext()

    def autocast(self):
        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    @property
    def sync_gradients(self):
        return True

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model

    def register_for_checkpointing(self, *args, **kwargs):
        pass

    def end_training(self):
        pass

    def init_trackers(self, *args, **kwargs):
        pass


# ============================
# Helpers
# ============================
def tensor_to_image_np(img_tensor, mean, std):
    img = img_tensor.detach().cpu().float().clone()
    mean_t = torch.tensor(mean, dtype=img.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(3, 1, 1)
    img = img * std_t + mean_t
    img = img.clamp(0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def reduce_dict(input_dict, average=True):
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
    experiment_name: str = "mono3d_default"
    working_dir: str = "/home/cse/msr/csy227518/scratch/Projects/Active/Scene4Cast/lib/detector/monocular3d"
    data_path: str = "/home/cse/msr/csy227518/scratch/Datasets/action_genome"
    save_path: str = "/home/cse/msr/csy227518/scratch/Projects/Active/Scene4Cast/save_path"
    # 3D loss GT from pkl: folder of per-video .pkl. If None, uses data_path/world_annotations/bbox_annotations_3d_final
    world_3d_annotations_path: Optional[str] = None
    ckpt: Optional[str] = None

    # Model
    model: str = "v3l"
    num_classes: Optional[int] = None  # None = auto-detect from dataset
    pretrained: bool = True
    use_compile: bool = False

    # Training
    lr: float = 1e-4
    weight_decay: float = 0.001
    batch_size: int = 128
    epochs: int = 70
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.01

    # DataLoader
    num_workers_train: int = 8
    num_workers_test: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True

    # Image
    target_size: Optional[int] = None  # None = use pixel_limit (Pi3-compatible); int = force square resize
    pixel_limit: int = 255000          # Pi3's PIXEL_LIMIT for aspect-ratio-preserving resize
    patch_size: int = 14               # Must match backbone patch_size (14 for DINOv2, 16 for some ViT-L)

    # Logging
    use_wandb: bool = True
    wandb_project: str = "DINOv2-Object-Detector-AG-3D"
    iter_log_every: int = 10000
    eval_every_iters: int = 10_000_000

    # Visualization
    plot_each_epoch: bool = True
    plot_sample_idx: int = 120
    plot_score_thresh: float = 0.1

    # Misc
    use_collate: bool = True
    use_accelerator: bool = False


# ============================
# Trainer Class
# ============================
class DinoAGTrainer3D:

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.path_to_experiment = os.path.join(cfg.working_dir, cfg.experiment_name)

        if cfg.use_accelerator:
            self.accelerator = Accelerator(
                project_dir=self.path_to_experiment,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                log_with="wandb" if cfg.use_wandb else None,
            )
        else:
            self.accelerator = DummyAccelerator()

        self.local_logger = LocalLogger(self.path_to_experiment)

        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # GradScaler for AMP
        self.model_device = None

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
                project=self.cfg.wandb_project,
                config={
                    "learning_rate": self.cfg.lr,
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "dataset": "Action_Genome_3D",
                    "model": self.cfg.model,
                    "target_size": self.cfg.target_size,
                },
            )

    def build_datasets(self) -> None:
        """Load and build train + test datasets from Action Genome annotations."""
        self.accelerator.print("━" * 60)
        self.accelerator.print("  [1/5] Building datasets...")
        self.accelerator.print("━" * 60)
        kwargs = {
            "phase": "train",
            "target_size": self.cfg.target_size,
            "pixel_limit": self.cfg.pixel_limit,
            "patch_size": self.cfg.patch_size,
        }
        if self.cfg.world_3d_annotations_path is not None:
            kwargs["world_3d_annotations_path"] = self.cfg.world_3d_annotations_path
        self.train_dataset = ActionGenomeDataset3D(self.cfg.data_path, **kwargs)
        kwargs["phase"] = "test"
        self.test_dataset = ActionGenomeDataset3D(self.cfg.data_path, **kwargs)
        self.accelerator.print(f"  ✓ Train: {len(self.train_dataset)} frames  |  Test: {len(self.test_dataset)} frames")

    def build_dataloaders(self) -> None:
        """Create DataLoaders with resolution-bucketed batch samplers."""
        self.accelerator.print("  [2/5] Building dataloaders (resolution-bucketed)...")

        # Build resolution-bucketed batch samplers
        # Frames sharing the same (target_w, target_h) go in the same batch
        train_batch_sampler = ResolutionBucketBatchSampler(
            self.train_dataset.resolution_buckets, self.cfg.batch_size, drop_last=False)

        test_batch_sampler = ResolutionBucketBatchSampler(
            self.test_dataset.resolution_buckets, self.cfg.batch_size, drop_last=True)



        _persistent = self.cfg.num_workers_train > 0 and self.cfg.persistent_workers
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=self.cfg.num_workers_train,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=_persistent,
            prefetch_factor=self.cfg.prefetch_factor if _persistent else None,
        )
        _persistent_test = self.cfg.num_workers_test > 0 and self.cfg.persistent_workers
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_sampler=test_batch_sampler,
            num_workers=self.cfg.num_workers_test,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=_persistent_test,
            prefetch_factor=self.cfg.prefetch_factor if _persistent_test else None,
        )


    def build_model(self) -> None:
        """Initialize the DINOv2 + Faster R-CNN + 3D head model."""
        self.accelerator.print("  [3/5] Building model...")
        num_classes = self.cfg.num_classes or len(self.train_dataset.object_classes)
        self.accelerator.print(f"    Model variant: {self.cfg.model}  |  Classes: {num_classes}")
        self.model = DinoV3Monocular3D(
            num_classes=num_classes,
            pretrained=self.cfg.pretrained,
            model=self.cfg.model,
        )
        # Count trainable vs frozen parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.accelerator.print(f"    ✓ Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}% trainable)")
        if self.cfg.use_compile:
            self.accelerator.print("    Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def build_optimizer_and_scheduler(self) -> None:
        """Create AdamW optimizer + warmup→cosine LR schedule + GradScaler for AMP."""
        self.accelerator.print("  [4/5] Building optimizer & scheduler...")
        total_steps = self.cfg.epochs * len(self.train_loader)
        warmup_steps = int(self.cfg.warmup_fraction * total_steps)

        # Only optimize trainable parameters (backbone is frozen)
        params = [p for _, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Schedule: linear warmup → cosine annealing
        warmup = LinearLR(self.optimizer, start_factor=1e-1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=(total_steps - warmup_steps), eta_min=self.cfg.lr * 0.1)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        self.accelerator.print(f"    ✓ Total steps: {total_steps:,}  |  Warmup: {warmup_steps:,}  |  LR: {self.cfg.lr}")

        # GradScaler for mixed precision (only used with DummyAccelerator / single-GPU)
        if not self.cfg.use_accelerator and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler()
            self.accelerator.print("    ✓ AMP GradScaler enabled")

    def prepare_with_accelerator(self) -> None:
        """Wrap model/optimizer/loaders with Accelerator and enable speed toggles."""
        self.accelerator.print("  [5/5] Preparing with accelerator + speed toggles...")
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.test_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.test_loader,
            self.scheduler,
        )

        self.accelerator.register_for_checkpointing(self.scheduler)

        self.model_device = next(self.model.parameters()).device

        # Speed toggles: enable cuDNN autotuner, TF32, and Flash Attention
        torch.backends.cudnn.benchmark = True        # Auto-tune convolution algorithms
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matrix multiplications
        torch.backends.cudnn.allow_tf32 = True         # TF32 for cuDNN operations
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

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

    def save_checkpoint(self, epoch: int) -> None:
        path_to_checkpoint = os.path.join(self.path_to_experiment, f"checkpoint_{epoch}")
        if not self.accelerator.is_main_process:
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

        # Access the raw dataset (unwrap Accelerator/Subset wrappers)
        ds = self.test_dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset

        if sample_idx >= len(ds):
            sample_idx = 0

        image_tensor, target = ds[sample_idx]

        # ds.samples may be a dict or a list
        if hasattr(ds, 'samples'):
            if isinstance(ds.samples, dict):
                keys = list(ds.samples.keys())
                sample = ds.samples[keys[sample_idx]] if sample_idx < len(keys) else {"filename": f"sample_{sample_idx}"}
            else:
                sample = ds.samples[sample_idx]
        else:
            sample = {"filename": f"sample_{sample_idx}"}

        mean = tuple(ds.image_mean) if hasattr(ds, 'image_mean') else (0.485, 0.456, 0.406)
        std = tuple(ds.image_std) if hasattr(ds, 'image_std') else (0.229, 0.224, 0.225)

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

        # Plot GT
        for box, lab in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            color = "cyan"
            rect = patches.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1),
                                     linewidth=2.5, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            class_name = ds.object_classes[int(lab)] if hasattr(ds, 'object_classes') and 0 <= int(lab) < len(
                ds.object_classes) else str(lab)
            ax.text(
                max(0, x1), max(10, y1 - 2), f"{class_name}",
                fontsize=8, color="black", verticalalignment="top",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2")
            )

        # Plot Pred
        for box, lab, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.tolist()
            color = "lime"
            rect = patches.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1),
                                     linewidth=2.0, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            class_name = ds.object_classes[int(lab)] if hasattr(ds, 'object_classes') and 0 <= int(lab) < len(
                ds.object_classes) else str(lab)
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

        save_path = os.path.join(save_dir, f"predictions_epoch_3d_{epoch + 1:03d}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # ----------------------------
    # Evaluation
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
        """COCO-style 2D mAP using the self-contained evaluate_2d_coco_map.
        Uses the full test set for accurate end-of-epoch evaluation."""
        self.model.eval()
        accelerator_arg = self.accelerator if self.cfg.use_accelerator else None
        with torch.no_grad():
            coco = evaluate_2d_coco_map(
                self.model, self.test_loader, self.model_device,
                accelerator=accelerator_arg,
            )

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



    def evaluate_all(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fused 2D + 3D evaluation in a single forward pass over the full test set.
        Returns (metrics_2d, metrics_3d)."""
        self.model.eval()
        accelerator_arg = self.accelerator if self.cfg.use_accelerator else None
        with torch.no_grad():
            results = evaluate_2d_and_3d_fused(
                self.model, self.test_loader, self.model_device,
                accelerator=accelerator_arg,
            )

        metrics_2d = results.get("metrics_2d", {})
        if metrics_2d:
            raw_py = self._tensor_dict_to_python(metrics_2d.get("raw", {}))
            metrics_2d["raw"] = raw_py

        return metrics_2d, results.get("metrics_3d", {})

    # ----------------------------
    # Training
    # ----------------------------
    def _move_targets(self, targets: Any) -> Any:
        if isinstance(targets, list):
            return [
                {k: v.to(self.model_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
        return [{k: v.to(self.model_device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}]

    def _log_iteration_losses(
            self,
            running_total_loss: float,
            running_cls_loss: float,
            running_box_loss: float,
            running_object_loss: float,
            running_rpn_loss: float,
            running_3d_loss: float,
            running_count: int,
    ) -> None:
        if running_count == 0:
            return
        avg_total_loss = running_total_loss / running_count
        avg_cls_loss = running_cls_loss / running_count
        avg_box_loss = running_box_loss / running_count
        avg_object_loss = running_object_loss / running_count
        avg_rpn_loss = running_rpn_loss / running_count
        avg_3d_loss = running_3d_loss / running_count
        lr = self.scheduler.get_last_lr()[0]

        if self.cfg.use_wandb and self.accelerator.is_main_process:
            wandb.log({
                "iteration": self.global_iteration,
                "iter/total_loss": avg_total_loss,
                "iter/cls_loss": avg_cls_loss,
                "iter/box_loss": avg_box_loss,
                "iter/object_loss": avg_object_loss,
                "iter/rpn_loss": avg_rpn_loss,
                "iter/3d_loss": avg_3d_loss,
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
            iter_3d_loss=avg_3d_loss,
            learning_rate=lr,
        )

        self.accelerator.print(
            f"Iteration {self.global_iteration}: Loss={avg_total_loss:.4f} "
            f"(Cls:{avg_cls_loss:.4f}, Box:{avg_box_loss:.4f}, "
            f"Object:{avg_object_loss:.4f}, RPN:{avg_rpn_loss:.4f}, 3D:{avg_3d_loss:.4f})"
        )

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one full training epoch. Returns dict of average loss components."""
        self.model.train()

        # Per-epoch loss accumulators
        batch_loss_list = []
        batch_loss_cls_list = []
        batch_loss_box_reg_list = []
        batch_loss_objectness_list = []
        batch_loss_rpn_list = []
        batch_loss_3d_list = []

        # Running accumulators for periodic logging within the epoch
        running_total_loss = running_cls_loss = running_box_loss = 0.0
        running_object_loss = running_rpn_loss = running_3d_loss = 0.0
        running_count = 0

        _zero = 0.0
        first_iter = True
        for images, targets in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.cfg.epochs} [Train]",
            ascii=True,
            dynamic_ncols=True,
        ):
            if first_iter:
                self.accelerator.print("First batch: loading data and first CUDA run (can take 2–5 min)...")
                first_iter = False
            # Move batch to GPU (non_blocking=True leverages pin_memory)
            images = torch.stack(images).to(self.model_device, non_blocking=True)
            targets = self._move_targets(targets)

            with self.accelerator.accumulate(self.model):  # handles gradient accumulation
                with self.accelerator.autocast():  # mixed precision forward pass
                    loss_dict_original = self.model(images, targets)

                box_weight = 1.0
                loss_dict: Dict[str, torch.Tensor] = {}
                for k, v in loss_dict_original.items():
                    loss_dict[k] = box_weight * v if k in ("loss_box_reg", "loss_rpn_box_reg") else v

                # Scale loss by gradient accumulation steps for correct effective batch size
                losses = sum(loss / self.cfg.gradient_accumulation_steps for loss in loss_dict.values())

                # Reduce across processes (no-op for single GPU)
                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss / self.cfg.gradient_accumulation_steps for loss in loss_dict_reduced.values())

                # Backward with GradScaler if available, otherwise standard
                if self.scaler is not None:
                    self.scaler.scale(losses).backward()
                else:
                    self.accelerator.backward(losses)

                if self.accelerator.sync_gradients:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

                self.global_iteration += 1

                if self.accelerator.is_main_process:
                    _total = losses_reduced.item()
                    _cls = loss_dict_reduced["loss_classifier"].item() if "loss_classifier" in loss_dict_reduced else _zero
                    _box = loss_dict_reduced["loss_box_reg"].item() if "loss_box_reg" in loss_dict_reduced else _zero
                    _obj = loss_dict_reduced["loss_objectness"].item() if "loss_objectness" in loss_dict_reduced else _zero
                    _rpn = loss_dict_reduced["loss_rpn_box_reg"].item() if "loss_rpn_box_reg" in loss_dict_reduced else _zero
                    _l3d = loss_dict_reduced["loss_3d"].item() if "loss_3d" in loss_dict_reduced else _zero

                    batch_loss_list.append(_total)
                    batch_loss_cls_list.append(_cls)
                    batch_loss_box_reg_list.append(_box)
                    batch_loss_objectness_list.append(_obj)
                    batch_loss_rpn_list.append(_rpn)
                    batch_loss_3d_list.append(_l3d)

                    running_total_loss += _total
                    running_cls_loss += _cls
                    running_box_loss += _box
                    running_object_loss += _obj
                    running_rpn_loss += _rpn
                    running_3d_loss += _l3d
                    running_count += 1

                if self.global_iteration % self.cfg.iter_log_every == 0 and self.accelerator.is_main_process:
                    self._log_iteration_losses(
                        running_total_loss, running_cls_loss, running_box_loss,
                        running_object_loss, running_rpn_loss, running_3d_loss, running_count
                    )
                    running_total_loss = running_cls_loss = running_box_loss = 0.0
                    running_object_loss = running_rpn_loss = running_3d_loss = 0.0
                    running_count = 0

                if self.global_iteration % self.cfg.eval_every_iters == 0:
                    self._maybe_eval_mid_epoch()

        return {
            "train_total_loss": float(np.mean(batch_loss_list)) if batch_loss_list else 0.0,
            "train_cls_loss": float(np.mean(batch_loss_cls_list)) if batch_loss_cls_list else 0.0,
            "train_box_loss": float(np.mean(batch_loss_box_reg_list)) if batch_loss_box_reg_list else 0.0,
            "train_object_loss": float(np.mean(batch_loss_objectness_list)) if batch_loss_objectness_list else 0.0,
            "train_rpn_loss": float(np.mean(batch_loss_rpn_list)) if batch_loss_rpn_list else 0.0,
            "train_3d_loss": float(np.mean(batch_loss_3d_list)) if batch_loss_3d_list else 0.0,
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
        """Main entry point: setup → train → evaluate → checkpoint, for all epochs."""
        os.makedirs(self.path_to_experiment, exist_ok=True)
        self.accelerator.print("\n" + "═" * 60)
        self.accelerator.print("  Monocular3D Training Pipeline")
        self.accelerator.print("═" * 60)

        # ---- Setup Phase ----
        self.build_datasets()          # [1/5] Load AG annotations + build frame index
        self.build_dataloaders()       # [2/5] Resolution-bucketed batch samplers
        self.build_model()             # [3/5] DINOv2 backbone + FPN + Faster R-CNN + 3D head
        self.build_optimizer_and_scheduler()  # [4/5] AdamW + warmup→cosine schedule
        self.prepare_with_accelerator()       # [5/5] Device placement + speed toggles
        self.maybe_resume()            # Resume from checkpoint if --ckpt specified

        self.accelerator.print("\n" + "═" * 60)
        self.accelerator.print(f"  Starting training: epochs {self.starting_epoch+1}→{self.cfg.epochs}")
        self.accelerator.print("═" * 60 + "\n")

        # ---- Training Loop ----
        for epoch in range(self.starting_epoch, self.cfg.epochs):
            train_stats = self.train_one_epoch(epoch)

            # End-of-epoch evaluation: fused 2D + 3D in a single forward pass
            self.accelerator.print(f"📊 Evaluating 2D COCO mAP + 3D metrics at end of epoch {epoch + 1}...")
            epoch_metrics, epoch_metrics_3d = self.evaluate_all()

            if self.cfg.plot_each_epoch and self.accelerator.is_main_process:
                try:
                    self.plot_predictions_after_epoch(
                        epoch=epoch,
                        sample_idx=self.cfg.plot_sample_idx,
                        score_threshold=self.cfg.plot_score_thresh,
                    )
                    self.accelerator.print(f"✓ Prediction visualization saved for epoch {epoch + 1}")
                except Exception as e:
                    import traceback
                    self.accelerator.print(f"⚠️  Failed to save visualization: {type(e).__name__}: {e}")
                    self.accelerator.print(traceback.format_exc())

            lr = self.scheduler.get_last_lr()[0]
            if self.cfg.use_wandb and self.accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch,
                    "train/total_loss": train_stats["train_total_loss"],
                    "train/cls_loss": train_stats["train_cls_loss"],
                    "train/box_loss": train_stats["train_box_loss"],
                    "train/object_loss": train_stats["train_object_loss"],
                    "train/rpn_loss": train_stats["train_rpn_loss"],
                    "train/3d_loss": train_stats["train_3d_loss"],
                    # 2D metrics
                    "epoch/map": epoch_metrics.get("map", 0.0),
                    "epoch/map_50": epoch_metrics.get("map_50", 0.0),
                    "epoch/map_75": epoch_metrics.get("map_75", 0.0),
                    # 3D metrics
                    "epoch/chamfer_mean": epoch_metrics_3d.get("chamfer_mean", 0.0),
                    "epoch/corner_l2_mean": epoch_metrics_3d.get("corner_l2_mean", 0.0),
                    "epoch/mAP_3d_50": epoch_metrics_3d.get("mAP_3d_50", 0.0),
                    "epoch/mAP_3d_75": epoch_metrics_3d.get("mAP_3d_75", 0.0),
                    "epoch/mean_iou_3d": epoch_metrics_3d.get("mean_iou_3d", 0.0),
                    "epoch/center_l2_mean": epoch_metrics_3d.get("center_l2_mean", 0.0),
                    "epoch/dims_l1_mean": epoch_metrics_3d.get("dims_l1_mean", 0.0),
                    "epoch/rotation_deg_mean": epoch_metrics_3d.get("rotation_deg_mean", 0.0),
                    "learning_rate": lr,
                })

            self.local_logger.log(
                log_type="epoch",
                epoch=epoch,
                learning_rate=lr,
                mAP_2d=epoch_metrics,
                metrics_3d=epoch_metrics_3d,
                **train_stats,
            )

            self.accelerator.print(f"\n{'═' * 80}")
            self.accelerator.print(f"  EPOCH {epoch + 1}/{self.cfg.epochs}  SUMMARY")
            self.accelerator.print(f"{'═' * 80}")

            # ── Training Losses ──
            self.accelerator.print(f"\n  ── Training Losses ──")
            self.accelerator.print(f"  Total Loss:      {train_stats['train_total_loss']:.4f}")
            self.accelerator.print(f"    Classifier:    {train_stats['train_cls_loss']:.4f}")
            self.accelerator.print(f"    Box Reg:       {train_stats['train_box_loss']:.4f}")
            self.accelerator.print(f"    Objectness:    {train_stats['train_object_loss']:.4f}")
            self.accelerator.print(f"    RPN Box Reg:   {train_stats['train_rpn_loss']:.4f}")
            self.accelerator.print(f"    3D Loss:       {train_stats['train_3d_loss']:.4f}")

            # ── 2D Evaluation Results ──
            self.accelerator.print(f"\n  ── 2D Evaluation (COCO-style) ──")
            if epoch_metrics:
                self.accelerator.print(f"  mAP (0.50:0.95): {epoch_metrics.get('map', 0.0):.4f}")
                self.accelerator.print(f"  mAP@50:           {epoch_metrics.get('map_50', 0.0):.4f}")
                self.accelerator.print(f"  mAP@75:           {epoch_metrics.get('map_75', 0.0):.4f}")
                if epoch_metrics.get("map_per_class") is not None:
                    self.accelerator.print(f"  Per-class AP:     (available in logs)")
            else:
                self.accelerator.print(f"  ⚠️  No 2D metrics available")

            # ── 3D Evaluation Results ──
            self.accelerator.print(f"\n  ── 3D Evaluation (matched by 2D IoU ≥ 0.5) ──")
            if epoch_metrics_3d:
                n_matched = epoch_metrics_3d.get('n_matched', 0)
                n_gt = epoch_metrics_3d.get('n_gt_3d', 0)
                match_rate = (n_matched / n_gt * 100) if n_gt > 0 else 0.0
                self.accelerator.print(f"  Matched Pairs:    {n_matched}/{n_gt} GT boxes ({match_rate:.1f}%)")
                self.accelerator.print(f"  ┌─ Box Quality ─────────────────────────────────┐")
                self.accelerator.print(f"  │  Chamfer Distance:   {epoch_metrics_3d.get('chamfer_mean', 0.0):10.4f}          │")
                self.accelerator.print(f"  │  Corner L2 (mean):   {epoch_metrics_3d.get('corner_l2_mean', 0.0):10.4f}          │")
                self.accelerator.print(f"  │  Mean IoU 3D:        {epoch_metrics_3d.get('mean_iou_3d', 0.0):10.4f}          │")
                self.accelerator.print(f"  └────────────────────────────────────────────────┘")
                self.accelerator.print(f"  ┌─ 3D mAP ────────────────────────────────────────┐")
                self.accelerator.print(f"  │  mAP_3d@50:          {epoch_metrics_3d.get('mAP_3d_50', 0.0):10.4f}          │")
                self.accelerator.print(f"  │  mAP_3d@75:          {epoch_metrics_3d.get('mAP_3d_75', 0.0):10.4f}          │")
                self.accelerator.print(f"  └────────────────────────────────────────────────┘")
                self.accelerator.print(f"  ┌─ Per-Attribute Errors ─────────────────────────┐")
                self.accelerator.print(f"  │  Center L2:          {epoch_metrics_3d.get('center_l2_mean', 0.0):10.4f}          │")
                self.accelerator.print(f"  │  Dimensions L1:      {epoch_metrics_3d.get('dims_l1_mean', 0.0):10.4f}          │")
                self.accelerator.print(f"  │  Rotation Error:     {epoch_metrics_3d.get('rotation_deg_mean', 0.0):9.1f}°         │")
                self.accelerator.print(f"  └────────────────────────────────────────────────┘")
            else:
                self.accelerator.print(f"  ⚠️  No 3D metrics available")

            self.accelerator.print(f"\n  Learning Rate: {lr:.6f}")
            self.accelerator.print(f"{'─' * 80}")

            self.save_checkpoint(epoch)

        self.accelerator.end_training()
