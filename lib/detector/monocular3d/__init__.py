"""
Monocular3D — DINOv2/v3/ResNet50 Monocular 3D Object Detection Package

Sub-packages:
    datasets    - ActionGenomeDataset3D
    models      - DinoV3Monocular3D, ResNetMonocular3D, DINOv2/v3 backbone + FPN
    losses      - OVMono3D loss
    evaluation  - 2D COCO + 3D metrics
    utils       - LocalLogger

Entry points:
    python -m lib.detector.monocular3d.train --config configs/detector/dinov2_saurabh_v1.yaml
    python -m lib.detector.monocular3d.train --config configs/detector/resnet50_unified_v1.yaml
    python -m lib.detector.monocular3d.evaluate --checkpoint ...
"""

from .trainer import TrainConfig, DinoAGTrainer3D
from .datasets import ActionGenomeDataset3D, collate_fn
from .models import DinoV3Monocular3D, create_model, ResNetMonocular3D
from .losses import ovmono3d_loss
from .constants import Constants

__all__ = [
    "TrainConfig",
    "DinoAGTrainer3D",
    "ActionGenomeDataset3D",
    "collate_fn",
    "DinoV3Monocular3D",
    "ResNetMonocular3D",
    "create_model",
    "ovmono3d_loss",
    "Constants",
]
