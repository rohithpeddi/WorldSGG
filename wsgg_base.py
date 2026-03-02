"""
WSGG Base Class
================

Shared infrastructure for all WSGG train/test classes.
Analogous to stsg_base.py in the SGG pipeline.

Provides:
  - Config loading (YAML + CLI override via load_wsgg_config)
  - Config initialization (device, paths, WandB)
  - Optimizer / scheduler
  - Checkpoint load / save
  - Evaluator initialization
  - Detector initialization (config-driven: dino_mono3d / frcnn / none)
"""

import os
import sys
from abc import abstractmethod
from argparse import ArgumentParser
from types import SimpleNamespace

import yaml
import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.supervised import BasicSceneGraphEvaluator
from lib.supervised.worldsgg.worldsgg_base import DINOFeatureExtractor


# ============================================================================
# YAML Config Loader (replaces wsgg_config.py)
# ============================================================================

def load_wsgg_config(yaml_path: str = None) -> SimpleNamespace:
    """
    Load WSGG config from YAML, merge with CLI overrides.

    Usage:
        conf = load_wsgg_config()                       # auto-detect --config
        conf = load_wsgg_config("configs/wsgg.yaml")    # explicit path

    CLI args override YAML values:
        python script.py --config configs/wsgg.yaml --method_name amwae --lr 5e-5
    """
    parser = ArgumentParser(description="WSGG")
    parser.add_argument("--config", default="configs/wsgg.yaml", type=str)
    # Add all possible overrides (non-None default = "not specified")
    parser.add_argument("--method_name", default=None, type=str)
    parser.add_argument("--mode", default=None, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--detector_ckpt", default=None, type=str)
    parser.add_argument("--detector_model", default=None, type=str)
    parser.add_argument("--detector_type", default=None, type=str)
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
    Mirrors STSGBase from stsg_base.py with WSGG-specific adaptations.
    """

    def __init__(self, conf):
        self._conf = conf
        self._model = None
        self._detector = None
        self._device = None
        self._evaluator = None
        self._optimizer = None
        self._scheduler = None

        self._train_dataset = None
        self._test_dataset = None

        self._checkpoint_name = None
        self._checkpoint_save_dir_path = None

        self._enable_wandb = getattr(conf, 'use_wandb', False)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _init_config(self, is_train=True):
        """Initialize device, checkpoint paths, and WandB."""
        print("The CKPT saved here:", self._conf.save_path)
        os.makedirs(self._conf.save_path, exist_ok=True)

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self._conf.ckpt is not None and self._conf.ckpt != "null":
            self._checkpoint_name_with_epoch = os.path.basename(self._conf.ckpt).split('.')[0]
            self._checkpoint_name = "_".join(self._checkpoint_name_with_epoch.split('_')[:-2])
            print("--------------------------------------------------------")
            print(f"Loading checkpoint with name: {self._checkpoint_name}")
            print(f"Mode: {self._conf.mode}")
            print("--------------------------------------------------------")
        else:
            self._checkpoint_name = f"{self._conf.method_name}_{self._conf.mode}"
            print("--------------------------------------------------------")
            print(f"Training model with name: {self._checkpoint_name}")
            print("--------------------------------------------------------")

        self._checkpoint_save_dir_path = os.path.join(
            self._conf.save_path, self._conf.task_name, self._conf.method_name
        )
        os.makedirs(self._checkpoint_save_dir_path, exist_ok=True)

        # WandB
        if self._enable_wandb:
            wandb.init(project=self._checkpoint_name, config=self._conf.args)

        print("-------------------- CONFIGURATION DETAILS ------------------------")
        for k, v in sorted(self._conf.args.items()):
            print(f"  {k}: {v}")
        print("-------------------------------------------------------------------")

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def _init_optimizer(self):
        """Initialize optimizer from config."""
        opt_name = getattr(self._conf, 'optimizer', 'adamw').lower()
        lr = getattr(self._conf, 'lr', 1e-4)
        wd = getattr(self._conf, 'weight_decay', 1e-4)

        if opt_name == "adamw":
            self._optimizer = optim.AdamW(self._model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        elif opt_name == "sgd":
            self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise NotImplementedError(f"Unknown optimizer: {opt_name}")

    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        self._scheduler = ReduceLROnPlateau(self._optimizer, "max", patience=1, factor=0.5, verbose=True)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _load_checkpoint(self):
        """Load model weights from checkpoint if specified."""
        if self._model is None:
            raise ValueError("Model is not initialized")

        ckpt_path = getattr(self._conf, 'ckpt', None)
        if ckpt_path is None or ckpt_path == "null" or ckpt_path == "":
            return

        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint file {ckpt_path} does not exist")

        try:
            ckpt = torch.load(ckpt_path, map_location=self._device)
            state_dict_key = 'state_dict' if 'state_dict' in ckpt else f'{self._conf.method_name}_state_dict'
            self._model.load_state_dict(ckpt[state_dict_key], strict=False)
            print(f"Loaded model from checkpoint {ckpt_path}")
        except FileNotFoundError:
            print(f"Error: Checkpoint file {ckpt_path} not found.")
        except KeyError:
            print(f"Error: Appropriate state_dict not found in the checkpoint.")
        except Exception as e:
            print(f"An error occurred loading checkpoint: {str(e)}")

    @staticmethod
    def _save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, method_name):
        """Save model checkpoint."""
        save_path = os.path.join(checkpoint_save_file_path, f"{checkpoint_name}_epoch_{epoch}.tar")
        torch.save({
            f"{method_name}_state_dict": model.state_dict(),
            "epoch": epoch,
        }, save_path)
        print(f"Model saved: {save_path}")

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
    # Detector Initialization (config-driven)
    # ------------------------------------------------------------------
    def _init_detector(self):
        """
        Initialize detector based on config.detector_type.

        Supported:
          - "dino_mono3d": DinoV3Monocular3D via DINOFeatureExtractor
          - "frcnn": Standard FasterRCNN from lib_b.Detector
          - "none": No detector (precomputed features / zeros)
        """
        detector_type = getattr(self._conf, 'detector_type', 'none')

        if detector_type == "dino_mono3d":
            self._detector = DINOFeatureExtractor(
                detector_ckpt=getattr(self._conf, 'detector_ckpt', ''),
                detector_model=getattr(self._conf, 'detector_model', 'v3l'),
                num_classes=getattr(self._conf, 'num_detector_classes', 37),
                device=str(self._device),
            )
            if getattr(self._conf, 'detector_frozen', True):
                self._detector.load_detector()
            print(f"[WSGGBase] Initialized DINO detector (model={self._conf.detector_model})")

        elif detector_type == "frcnn":
            from lib_b.object_detector import Detector
            object_classes = (
                self._train_dataset.object_classes
                if self._train_dataset else self._test_dataset.object_classes
            )
            self._detector = Detector(
                train=False,
                object_classes=object_classes,
                use_SUPPLY=True,
                mode=self._conf.mode,
            ).to(self._device)
            self._detector.eval()
            print("[WSGGBase] Initialized FRCNN detector")

        elif detector_type == "none":
            self._detector = None
            print("[WSGGBase] No detector initialized (using precomputed features)")

        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    @abstractmethod
    def init_model(self):
        """Initialize the method-specific model. Must set self._model."""
        pass
