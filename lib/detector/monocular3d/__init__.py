"""
Monocular3D — DINOv2-based 3D Object Detection Package

Sub-packages:
    datasets    - ActionGenomeDataset3D
    models      - DinoV2Monocular3D, DINOv2 backbone + FPN
    losses      - OVMono3D loss
    evaluation  - 2D COCO + 3D metrics
    utils       - LocalLogger

Entry points:
    python -m lib.detector.monocular3d.train --config configs/dinov2b_saurabh_separate.yaml
    python -m lib.detector.monocular3d.evaluate --checkpoint ...
"""

from .trainer import TrainConfig, DinoAGTrainer3D
from .datasets import ActionGenomeDataset3D, collate_fn
from .models import DinoV3Monocular3D, create_model
from .losses import ovmono3d_loss
from .constants import Constants

__all__ = [
    "TrainConfig",
    "DinoAGTrainer3D",
    "ActionGenomeDataset3D",
    "collate_fn",
    "DinoV3Monocular3D",
    "create_model",
    "ovmono3d_loss",
    "Constants",
]
