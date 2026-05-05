# Monocular 3D Object Detection

Multi-backbone monocular 3D object detector trained on Action Genome. Supports three configurable backbone architectures (DINOv2, DINOv3, ResNet50-FPN), each combined with Faster R-CNN (2D detection) and a factorized 3D head for predicting 8-corner 3D bounding boxes from monocular images.

## Directory Structure

```
monocular3d/
├── datasets/                   # ActionGenomeDataset3D + collate_fn
├── evaluation/                 # 2D COCO mAP + 3D metrics (fused single-pass)
├── losses/                     # OVMono3D disentangled 3D loss
├── models/
│   ├── dino_mono_3d.py         # DINOv2/v3 backbone + SimpleFPN + RCNN + 3D head
│   └── resnet_mono_3d.py       # ResNet50-FPN backbone + RCNN + 3D head
├── utils/                      # JSON logger
├── train.py                    # Entry point (YAML config + CLI overrides)
└── trainer.py                  # DinoAGTrainer3D (training loop, evaluation, checkpointing)

# Training configs are in the project root configs/detector/ folder
```

## Prerequisites

### 1. Environment Setup

```bash
conda create -n scene4cast python=3.11
conda activate scene4cast
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate wandb torchmetrics tqdm pyyaml matplotlib numpy
```

### 2. W&B Login

[Weights & Biases](https://wandb.ai/) is used for experiment tracking. Login once per server:

```bash
# Option A: Interactive login (prompts for API key)
wandb login

# Option B: Direct API key (get it from https://wandb.ai/authorize)
wandb login YOUR_API_KEY

# Option C: Environment variable
export WANDB_API_KEY="YOUR_API_KEY"
```

Your API key is saved to `~/.netrc` — you only need to do this once.

### 3. Data

The Action Genome dataset should be available at the path specified by `data_path` in your config:

```
<data_path>/
├── frames/                     # Video frames
└── world_annotations/
    └── bbox_annotations_3d_final/  # 3D bounding box annotations (pkl files)
```

### 4. Shared Memory Limit (Linux)

To avoid `Too many open files` errors with DataLoader workers:

```bash
# Add to ~/.bashrc or run before training
ulimit -n 65536
```

---

## Backbone Options

All three backbones share the same 3D head architecture and can be selected via the `backbone` config field:

| Config Key | Backbone | Params | FPN | Pretrained On | Fine-tuned | `backbone` |
|-----------|----------|--------|-----|---------------|------------|------------|
| DINOv2 ViT-B | `v2` | 86M | Custom SimpleFPN | ImageNet | FPN+RPN+ROI+3D only | `dino_v2` |
| DINOv2 ViT-L | `v2l` | 304M | Custom SimpleFPN | ImageNet | FPN+RPN+ROI+3D only | `dino_v2` |
| DINOv3 ViT-L | `v3l` | 304M | Custom SimpleFPN | LVD-1689M | FPN+RPN+ROI+3D only | `dino_v3` |
| ResNet50-FPN-V2 | — | 26M | torchvision FPN | COCO (detection) | **Full model** | `resnet50` |

### Architecture Diagram

```
DINOv2/v3 path:  Image → Dataset(resize+norm) → _NoOpTransform → ViT(frozen) → SimpleFPN → RPN → ROI+3D → Detections
ResNet50 path:   Image → Dataset(resize+norm) → _NoOpTransform → ResNet+FPN(trainable) → RPN → ROI+3D → Detections
```

The 3D head (Mono3DRoIHeads / SeparateMono3DHead) and loss function (OVMono3D) are identical across all backbones.

---

## Training

### Basic Usage

```bash
cd <project_root>

# ResNet50-FPN training (COCO-pretrained, fully fine-tuned)
python -m lib.detector.monocular3d.train --config configs/detector/resnet50_unified_v1.yaml

# DINOv2-Base training
python -m lib.detector.monocular3d.train --config configs/detector/dinov2_saurabh_v1.yaml

# DINOv3-Large training
python -m lib.detector.monocular3d.train --config configs/detector/dinov3l_saurabh_v1.yaml
```

### Available Configs

| Config | Backbone | 3D Head | W&B Project |
|--------|----------|---------|-------------|
| `dinov2_saurabh_v1.yaml` | DINOv2 ViT-B | Unified | `DINOv2-Object-Detector-AG-3D` |
| `dinov2_saurabh_separate_v1.yaml` | DINOv2 ViT-B | Separate | `DINOv2-separate-Object-Detector-AG-3D` |
| `dinov2l_saurabh_v1.yaml` | DINOv2 ViT-L | Unified | `DINOv2-large-Object-Detector-AG-3D` |
| `dinov2l_saurabh_separate_v1.yaml` | DINOv2 ViT-L | Separate | `DINOv2-large-Object-Detector-AG-3D` |
| `dinov3l_saurabh_v1.yaml` | DINOv3 ViT-L | Unified | `DINOv3-large-Object-Detector-AG-3D` |
| `dinov3l_saurabh_separate_v1.yaml` | DINOv3 ViT-L | Separate | `DINOv3-large-Object-Detector-AG-3D` |
| `resnet50_unified_v1.yaml` | ResNet50-FPN | Unified | `ResNet50-Object-Detector-AG-3D` |
| `resnet50_separate_v1.yaml` | ResNet50-FPN | Separate | `ResNet50-separate-Object-Detector-AG-3D` |

### CLI Overrides

Any YAML config field can be overridden via CLI:

```bash
# Change learning rate and batch size
python -m lib.detector.monocular3d.train \
    --config configs/detector/resnet50_unified_v1.yaml \
    --lr 5e-5 \
    --batch_size 64

# Switch backbone via CLI override
python -m lib.detector.monocular3d.train \
    --config configs/detector/dinov2_saurabh_v1.yaml \
    --backbone resnet50 \
    --lr 5e-5

# Disable W&B for a quick test
python -m lib.detector.monocular3d.train \
    --config configs/detector/resnet50_unified_v1.yaml \
    --use_wandb false \
    --epochs 2
```

### Resume from Checkpoint

```bash
python -m lib.detector.monocular3d.train \
    --config configs/detector/resnet50_unified_v1.yaml \
    --ckpt checkpoint_10
```

The checkpoint folder is located at `<save_path>/<experiment_name>/checkpoint_<epoch>/`.

---

## Training Methodology

### DINOv2/v3 Training

- Backbone is **frozen** (ViT weights from HuggingFace). Only the custom SimpleFPN, RPN, ROI heads, and 3D head are trained.
- ~9M trainable parameters.
- Learning rate: `1e-4`.
- Images resized to `pixel_limit` with patch-size-aligned dimensions (multiples of 14 or 16).

### ResNet50-FPN Training

- Full model is **trainable** (COCO-pretrained ResNet50 body + FPN + RPN + ROI heads + 3D head).
- ~26M trainable parameters.
- Learning rate: `5e-5` (lower for fine-tuning COCO-pretrained weights).
- Images resized to `pixel_limit` with div-32-aligned dimensions.

### Training Schedule (Staged 3D Ramp)

With `weight_3d_ramp_epochs=5`:

1. **Epochs 1–5**: 2D-only training (3D loss weight = 0). Stabilizes 2D detection.
2. **Epochs 6–10**: 3D loss ramps linearly from 0 → `weight_3d`. Joint 2D + 3D training begins.
3. **Epochs 11+**: Full joint training with stable 3D loss weight.

---

## Multi-GPU Training (Accelerate)

When `use_accelerator: true` is set in the config:

```bash
# Single GPU (default)
python -m lib.detector.monocular3d.train --config configs/detector/resnet50_unified_v1.yaml

# Multi-GPU with accelerate
accelerate launch --num_processes 4 \
    -m lib.detector.monocular3d.train \
    --config configs/detector/resnet50_unified_v1.yaml
```

---

## Evaluation

### 2D COCO mAP

```bash
# ResNet50
python -m lib.detector.monocular3d.evaluation.evaluate_2d \
    --checkpoint /path/to/checkpoint_XX \
    --data_path /path/to/action_genome \
    --backbone resnet50

# DINOv3
python -m lib.detector.monocular3d.evaluation.evaluate_2d \
    --checkpoint /path/to/checkpoint_XX \
    --backbone dino_v3 --model v3l
```

Reports: `mAP`, `mAP@50`, `mAP@75`, per-class AP, precision/recall.

### 3D Metrics

```bash
python -m lib.detector.monocular3d.evaluation.evaluate_3d \
    --checkpoint /path/to/checkpoint_XX \
    --data_path /path/to/action_genome \
    --backbone resnet50
```

Reports: Chamfer distance, corner L2, oriented 3D IoU hit rates (@50, @75), center L2, dimension L1, rotation error.

### Fused 2D + 3D (End-of-Epoch)

During training, `evaluate_all()` runs a fused 2D + 3D evaluator in a single forward pass at the end of every epoch. This is backbone-agnostic.

---

## What Gets Logged

### W&B Metrics

**Per iteration** (every `iter_log_every` steps):
- `iter/total_loss`, `iter/cls_loss`, `iter/box_loss`, `iter/object_loss`, `iter/rpn_loss`, `iter/3d_loss`
- `learning_rate`

**Per epoch** (end-of-epoch evaluation):
- Training losses: `train/total_loss`, `train/cls_loss`, `train/box_loss`, `train/object_loss`, `train/rpn_loss`, `train/3d_loss`
- 2D COCO mAP: `epoch/map`, `epoch/map_50`, `epoch/map_75`
- 3D metrics: `epoch/chamfer_mean`, `epoch/corner_l2_mean`, `epoch/iou3d_hit_50`, `epoch/iou3d_hit_75`, `epoch/mean_iou_3d`
- 3D attribute errors: `epoch/center_l2_mean`, `epoch/dims_l1_mean`, `epoch/rotation_deg_mean`

### Local Logging

All metrics are also saved to `<working_dir>/<experiment_name>/train_log.json`.

### Visualizations

When `plot_each_epoch: true`, prediction visualizations (GT vs predicted boxes) are saved to:
```
<working_dir>/<experiment_name>/visualizations/predictions_epoch_3d_001.png
```

---

## Config Reference

```yaml
# --- Experiment ---
experiment_name: "mono3d_default"   # Unique name for this run
working_dir: "/path/to/monocular3d" # Where to save logs, visualizations
save_path: "/path/to/save"          # Where to save checkpoints

# --- Data ---
data_path: "/path/to/action_genome" # Action Genome root directory
world_3d_annotations_path: null     # null = auto (data_path/world_annotations/...)

# --- Model ---
backbone: "dino_v2"                 # "dino_v2" | "dino_v3" | "resnet50"
model: "v2"                         # v2 | v2l | v3l (used for DINOv2/v3 only)
num_classes: null                   # null = auto-detect from dataset
pretrained: true                    # Use pretrained weights
use_compile: false                  # torch.compile() (PyTorch 2.0+)

# --- Training ---
lr: 1.0e-4                          # Learning rate (5e-5 recommended for ResNet50)
weight_decay: 0.001                 # AdamW weight decay
batch_size: 128                     # Per-GPU batch size
epochs: 70                          # Total training epochs
gradient_accumulation_steps: 1      # Effective batch = batch_size × this
max_grad_norm: 1.0                  # Gradient clipping
warmup_fraction: 0.01               # Fraction of steps for LR warmup

# --- 3D Head ---
head_3d_mode: "unified"             # "unified" | "separate"
head_3d_version: "v1"               # "v1" | "v2" (v2 adds depth stats)
max_3d_proposals: 64                # Max proposals for 3D loss
depth_maps_dir: null                # Path to depth maps (required for v2)

# --- Loss Weights ---
weight_cls: 1.0                     # Classification loss
weight_box: 1.0                     # Box regression loss
weight_obj: 1.0                     # RPN objectness loss
weight_rpn: 1.0                     # RPN box regression loss
weight_3d: 1.0                      # 3D head loss
weight_3d_ramp_epochs: 5            # Staged ramp (0 = disabled)

# --- Image ---
target_size: null                   # Fixed resize (null = dynamic pixel_limit resize)
pixel_limit: 255000                 # Max pixels for dynamic resize
patch_size: 14                      # 14 for DINOv2, 16 for DINOv3, 32 for ResNet50

# --- DataLoader ---
num_workers_train: 8                # DataLoader workers for training
num_workers_test: 8                 # DataLoader workers for evaluation
prefetch_factor: 4                  # Batches to prefetch per worker
persistent_workers: true            # Keep workers alive between epochs

# --- Logging ---
use_wandb: true                     # Enable W&B logging
wandb_project: "project-name"       # W&B project name
iter_log_every: 10000               # Log losses every N iterations
eval_every_iters: 10000000          # Mid-epoch eval (set high to disable)

# --- Visualization ---
plot_each_epoch: true               # Save prediction plots after each epoch
plot_sample_idx: 120                # Test sample index to visualize
plot_score_thresh: 0.1              # Min score for predicted boxes

# --- Misc ---
use_collate: true                   # Use custom collate_fn
use_accelerator: true               # Use HF Accelerate (multi-GPU)
ckpt: null                          # Checkpoint folder to resume from
```

---

## Creating a New Config

1. Copy an existing config:
   ```bash
   cp configs/detector/resnet50_unified_v1.yaml configs/detector/my_experiment.yaml
   ```

2. Update the server-specific paths:
   - `working_dir` — where logs and visualizations go
   - `save_path` — where checkpoints are saved
   - `data_path` — Action Genome dataset location

3. Choose your backbone:
   - `backbone: "dino_v2"` + `model: "v2"` — DINOv2 ViT-Base
   - `backbone: "dino_v2"` + `model: "v2l"` — DINOv2 ViT-Large
   - `backbone: "dino_v3"` + `model: "v3l"` — DINOv3 ViT-Large
   - `backbone: "resnet50"` — ResNet50-FPN-V2 (COCO-pretrained)

4. Set a unique `experiment_name` and `wandb_project`

5. Run:
   ```bash
   python -m lib.detector.monocular3d.train --config configs/detector/my_experiment.yaml
   ```
