"""
WSGG Base Class
================

Shared infrastructure for all WSGG train/test classes.

Provides:
  - Config loading (YAML + CLI override via load_wsgg_config)
  - Config initialization (device, experiment directory, WandB)
  - Optimizer / scheduler
  - Full-state checkpoint save / resume (model + optimizer + scheduler + scaler)
  - Model-only checkpoint load (for testing)
  - Evaluator initialization
  - Detector initialization (config-driven: dino_mono3d / frcnn / none)
"""

import gc
import logging
import os
from abc import abstractmethod
from argparse import ArgumentParser
from types import SimpleNamespace

import yaml
import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator

logger = logging.getLogger(__name__)


# ============================================================================
# YAML Config Loader
# ============================================================================

def load_wsgg_config(yaml_path: str = None) -> SimpleNamespace:
    """
    Load WSGG config from YAML, merge with CLI overrides.

    Usage:
        conf = load_wsgg_config()                                         # auto-detect --config
        conf = load_wsgg_config("configs/methods/predcls/gl_stgn_predcls.yaml")  # explicit path

    CLI args override YAML values:
        python script.py --config configs/methods/predcls/amwae_predcls.yaml --lr 5e-5
    """
    parser = ArgumentParser(description="WSGG")
    parser.add_argument("--config", default="configs/methods/predcls/gl_stgn_predcls.yaml", type=str)
    # Add all possible overrides (non-None default = "not specified")
    parser.add_argument("--method_name", default=None, type=str)
    parser.add_argument("--mode", default=None, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument("--nepoch", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--grad_clip", default=None, type=float)
    parser.add_argument("--d_model", default=None, type=int)
    parser.add_argument("--d_struct", default=None, type=int)
    parser.add_argument("--d_visual", default=None, type=int)
    parser.add_argument("--d_memory", default=None, type=int)
    parser.add_argument("--d_camera", default=None, type=int)
    parser.add_argument("--d_motion", default=None, type=int)
    parser.add_argument("--n_heads", default=None, type=int)
    parser.add_argument("--dropout", default=None, type=float)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--lambda_vlm", default=None, type=float)
    parser.add_argument("--lambda_smooth", default=None, type=float)
    parser.add_argument("--label_smoothing_vlm", default=None, type=float)
    parser.add_argument("--use_wandb", action="store_true", default=None)
    parser.add_argument("--use_amp", action="store_true", default=None)
    parser.add_argument("--datasize", default=None, type=str)

    args, _ = parser.parse_known_args()
    cli_args = {k: v for k, v in vars(args).items() if v is not None and k != "config"}

    # Load YAML
    config_path = yaml_path or args.config
    yaml_cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    # Merge: YAML first, CLI overrides
    merged = dict(yaml_cfg)
    merged.update(cli_args)
    merged["args"] = merged.copy()  # for WandB logging

    return SimpleNamespace(**merged)



class WSGGBase:
    """
    Root base class for all WSGG methods.

    Checkpoint layout (matches DinoAGTrainer3D):
        save_path/
        └── experiment_name/
            ├── checkpoint_0/checkpoint_state.pth
            ├── checkpoint_1/checkpoint_state.pth
            └── ...

    Resume: set config.ckpt = "checkpoint_5" to resume from epoch 5.
    """

    def __init__(self, conf):
        self._conf = conf
        self._model = None
        self._device = None
        self._evaluator = None
        self._optimizer = None
        self._scheduler = None
        self._scaler = None

        self._train_dataset = None
        self._test_dataset = None

        self._experiment_name = None
        self._experiment_dir = None
        self._starting_epoch = 0

        self._enable_wandb = getattr(conf, 'use_wandb', False)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _init_config(self, is_train=True):
        """Initialize device, experiment directory, logging, and WandB."""
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Experiment directory: save_path / experiment_name
        self._experiment_name = getattr(
            self._conf, 'experiment_name',
            f"{self._conf.method_name}_{self._conf.mode}",
        )
        self._experiment_dir = os.path.join(self._conf.save_path, self._experiment_name)
        os.makedirs(self._experiment_dir, exist_ok=True)

        # --- Logging setup: file + console ---
        self._setup_logging()

        ckpt = getattr(self._conf, 'ckpt', None)
        if ckpt is not None and ckpt != "null" and ckpt != "":
            logger.info("━" * 60)
            logger.info(f"  Experiment : {self._experiment_name}")
            logger.info(f"  Resume from: {ckpt}")
            logger.info(f"  Mode       : {self._conf.mode}")
            logger.info("━" * 60)
        else:
            action = "Training" if is_train else "Testing"
            logger.info("━" * 60)
            logger.info(f"  Experiment: {self._experiment_name}")
            logger.info(f"  {action} from scratch")
            logger.info(f"  Mode      : {self._conf.mode}")
            logger.info("━" * 60)

        # WandB
        if self._enable_wandb:
            wandb.init(project=self._experiment_name, config=self._conf.args)

        logger.info("─── Configuration ───")
        for k, v in sorted(self._conf.args.items()):
            logger.info(f"  {k}: {v}")
        logger.info("─" * 40)

    def _setup_logging(self):
        """Configure root logger with file and console handlers.

        Log file: <project_root>/logs/<experiment_name>.log
        """
        # Determine project root (directory containing wsgg_base.py)
        project_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{self._experiment_name}.log")

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Avoid duplicate handlers on re-init
        if not root_logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # File handler
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            root_logger.addHandler(ch)

        logger.info(f"Logging to: {log_file}")

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def _init_optimizer(self):
        """Initialize optimizer from config."""
        opt_name = self._conf.optimizer.lower()
        lr = float(self._conf.lr)
        wd = float(self._conf.weight_decay)

        if opt_name == "adamw":
            self._optimizer = optim.AdamW(self._model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        elif opt_name == "sgd":
            self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise NotImplementedError(f"Unknown optimizer: {opt_name}")

    def _init_scheduler(self, total_steps: int):
        """Initialize warmup → cosine annealing LR scheduler.

        Args:
            total_steps: total training iterations (epochs × batches_per_epoch).
        """
        warmup_fraction = getattr(self._conf, 'warmup_fraction', 0.1)
        warmup_steps = int(warmup_fraction * total_steps)
        lr = self._conf.lr

        warmup = LinearLR(
            self._optimizer,
            start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self._optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=lr * 0.01,
        )
        self._scheduler = SequentialLR(
            self._optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        logger.info(f"  Scheduler: warmup({warmup_steps}) → cosine({total_steps - warmup_steps}) | eta_min={lr * 0.01:.2e}")

    # ------------------------------------------------------------------
    # Checkpoint — Full-State Save (train)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save full training state to experiment_dir/checkpoint_{epoch}/.

        Saved keys:
          - epoch
          - model_state_dict
          - optimizer_state_dict
          - scheduler_state_dict
          - scaler_state_dict (None if AMP disabled)
        """
        ckpt_dir = os.path.join(self._experiment_dir, f"checkpoint_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_file = os.path.join(ckpt_dir, "checkpoint_state.pth")

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict(),
            "scaler_state_dict": self._scaler.state_dict() if self._scaler else None,
        }
        torch.save(checkpoint_dict, ckpt_file)
        logger.info(f"✓ Checkpoint saved: epoch {epoch + 1} → {ckpt_file}")

    # ------------------------------------------------------------------
    # Checkpoint — Full-State Resume (train)
    # ------------------------------------------------------------------
    def _maybe_resume(self) -> None:
        """
        Resume full training state from a checkpoint directory.

        Expects config.ckpt = "checkpoint_5" (directory name under experiment_dir).
        Restores model, optimizer, scheduler, and scaler state dicts sequentially
        with gc.collect() calls to avoid GPU memory spikes.
        """
        ckpt = getattr(self._conf, 'ckpt', None)
        if ckpt is None or ckpt == "null" or ckpt == "":
            self._starting_epoch = 0
            return

        ckpt_path = os.path.join(self._experiment_dir, ckpt, "checkpoint_state.pth")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            # Try to infer epoch from directory name (e.g., checkpoint_5 → epoch 5)
            try:
                self._starting_epoch = int(ckpt.split("_")[-1]) + 1
            except (ValueError, IndexError):
                self._starting_epoch = 0
            logger.info(f"  Starting from epoch {self._starting_epoch}")
            return

        logger.info(f"Resuming from: {ckpt_path}")

        # Load to CPU to avoid GPU memory spike
        checkpoint_state = torch.load(ckpt_path, map_location="cpu")

        # 1. Restore model weights, then free CPU copy
        self._model.load_state_dict(checkpoint_state["model_state_dict"])
        del checkpoint_state["model_state_dict"]
        gc.collect()
        logger.info("  ✓ Model state restored")

        # 2. Restore scheduler (small — just scalar state)
        if "scheduler_state_dict" in checkpoint_state and self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])
        del checkpoint_state["scheduler_state_dict"]
        logger.info("  ✓ Scheduler state restored")

        # 3. Restore optimizer (largest: 2× param memory for Adam momentum buffers)
        optimizer_state = checkpoint_state.pop("optimizer_state_dict")
        self._starting_epoch = checkpoint_state.get("epoch", 0) + 1

        # 4. Restore scaler if present
        if checkpoint_state.get("scaler_state_dict") is not None and self._scaler is not None:
            self._scaler.load_state_dict(checkpoint_state["scaler_state_dict"])
            logger.info("  ✓ AMP scaler state restored")

        del checkpoint_state
        gc.collect()

        self._optimizer.load_state_dict(optimizer_state)
        del optimizer_state
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("  ✓ Optimizer state restored")

        logger.info(f"✓ Resumed from epoch {self._starting_epoch} (GPU cache cleared)")

    # ------------------------------------------------------------------
    # Checkpoint — Model-Only Load (test)
    # ------------------------------------------------------------------
    def _load_checkpoint(self) -> None:
        """
        Load model weights only from a checkpoint (for testing/inference).

        Expects config.ckpt = "checkpoint_5" (directory name under experiment_dir).
        Only restores model_state_dict — no optimizer/scheduler.
        """
        if self._model is None:
            raise ValueError("Model is not initialized")

        ckpt = getattr(self._conf, 'ckpt', None)
        if ckpt is None or ckpt == "null" or ckpt == "":
            return

        ckpt_path = os.path.join(self._experiment_dir, ckpt, "checkpoint_state.pth")
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Loading model weights from: {ckpt_path}")
        checkpoint_state = torch.load(ckpt_path, map_location=self._device)
        self._model.load_state_dict(checkpoint_state["model_state_dict"], strict=False)
        epoch = checkpoint_state.get("epoch", "?")
        del checkpoint_state
        gc.collect()
        logger.info(f"  ✓ Model loaded (epoch {epoch})")

    # ------------------------------------------------------------------
    # Evaluators
    # ------------------------------------------------------------------
    def _init_evaluators(self):
        """Initialize scene graph evaluators."""
        self._evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._test_dataset.object_classes
            if self._test_dataset else None,
            iou_threshold=0.5,
            constraint="with",
        )



    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    @abstractmethod
    def init_model(self):
        """Initialize the method-specific model. Must set self._model."""
        pass
