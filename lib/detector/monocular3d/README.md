# Monocular 3D Object Detection

DINOv2/DINOv3-based monocular 3D object detector trained on Action Genome. Combines a frozen ViT backbone with a SimpleFPN, Faster R-CNN (2D detection), and a factorized 3D head for predicting 8-corner 3D bounding boxes from monocular images.

## Directory Structure

```
monocular3d/
├── configs/                    # YAML training configs (one per experiment)
│   ├── default_rohith.yaml     # DINOv2 ViT-B on Rohith's server
│   ├── dinov2b_saurabh.yaml    # DINOv2 ViT-B on Saurabh's server
│   ├── dinov2l_saurabh.yaml    # DINOv2 ViT-L on Saurabh's server
│   └── dinov3l_saurabh.yaml    # DINOv3 ViT-L on Saurabh's server
├── datasets/                   # ActionGenomeDataset3D + collate_fn
├── evaluation/                 # 2D COCO mAP + 3D metrics (fused single-pass)
├── losses/                     # OVMono3D disentangled 3D loss
├── models/                     # DinoV3Monocular3D model (backbone + FPN + RCNN + 3D head)
├── utils/                      # JSON logger
├── train.py                    # Entry point (YAML config + CLI overrides)
└── trainer.py                  # DinoAGTrainer3D (training loop, evaluation, checkpointing)
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

## Training

### Basic Usage

```bash
cd <project_root>

# Train with a specific config
python -m lib.detector.monocular3d.train --config configs/default_rohith.yaml
```

### Available Configs

| Config | Backbone | Params | W&B Project | Server |
|--------|----------|--------|-------------|--------|
| `default_rohith.yaml` | DINOv2 ViT-B | 86M | `DINOv2-Object-Detector-AG-3D` | Rohith |
| `dinov2b_saurabh.yaml` | DINOv2 ViT-B | 86M | `DINOv2-base-Object-Detector-AG-3D` | Saurabh |
| `dinov2l_saurabh.yaml` | DINOv2 ViT-L | 304M | `DINOv2-large-Object-Detector-AG-3D` | Saurabh |
| `dinov3l_saurabh.yaml` | DINOv3 ViT-L | 304M | `DINOv3-large-Object-Detector-AG-3D` | Saurabh |

### CLI Overrides

Any YAML config field can be overridden via CLI:

```bash
# Change learning rate and batch size
python -m lib.detector.monocular3d.train \
    --config configs/default_rohith.yaml \
    --lr 5e-5 \
    --batch_size 64

# Disable W&B for a quick test
python -m lib.detector.monocular3d.train \
    --config configs/default_rohith.yaml \
    --use_wandb false \
    --epochs 2

# Use a different backbone
python -m lib.detector.monocular3d.train \
    --config configs/default_rohith.yaml \
    --model v2l
```

### Resume from Checkpoint

```bash
python -m lib.detector.monocular3d.train \
    --config configs/default_rohith.yaml \
    --ckpt checkpoint_10
```

The checkpoint folder is located at `<save_path>/<experiment_name>/checkpoint_<epoch>/`.

---

## Multi-GPU Training (Accelerate)

When `use_accelerator: true` is set in the config:

```bash
# Single GPU (default)
python -m lib.detector.monocular3d.train --config configs/default_rohith.yaml

# Multi-GPU with accelerate
accelerate launch --num_processes 4 \
    -m lib.detector.monocular3d.train \
    --config configs/default_rohith.yaml
```

---

## Model Variants

| Key | Model | Hidden Size | Patch Size | Notes |
|-----|-------|-------------|------------|-------|
| `v2` | DINOv2 ViT-B | 768 | 14 | Fast, 86M params — default |
| `v2s` | DINOv2 ViT-S | 384 | 14 | Smallest, 22M params |
| `v2l` | DINOv2 ViT-L | 1024 | 14 | Large, 304M params |
| `v3l` | DINOv3 ViT-L | 1024 | 16 | DINOv3 variant |

---

## What Gets Logged

### W&B Metrics

**Per iteration** (every `iter_log_every` steps):
- `iter/total_loss`, `iter/cls_loss`, `iter/box_loss`, `iter/object_loss`, `iter/rpn_loss`, `iter/3d_loss`
- `learning_rate`

**Per epoch** (end-of-epoch evaluation):
- Training losses: `train/total_loss`, `train/cls_loss`, `train/box_loss`, `train/object_loss`, `train/rpn_loss`, `train/3d_loss`
- 2D COCO mAP: `epoch/map`, `epoch/map_50`, `epoch/map_75`
- 3D metrics: `epoch/chamfer_mean`, `epoch/corner_l2_mean`, `epoch/mAP_3d_50`, `epoch/mAP_3d_75`, `epoch/mean_iou_3d`
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
model: "v2"                         # v2 | v2s | v2l | v3l
num_classes: null                   # null = auto-detect from dataset
pretrained: true                    # Use HuggingFace pretrained weights
use_compile: false                  # torch.compile() (PyTorch 2.0+)

# --- Training ---
lr: 1.0e-4                          # Learning rate
weight_decay: 0.001                 # AdamW weight decay
batch_size: 128                     # Per-GPU batch size
epochs: 70                          # Total training epochs
gradient_accumulation_steps: 1      # Effective batch = batch_size × this
max_grad_norm: 1.0                  # Gradient clipping
warmup_fraction: 0.01               # Fraction of steps for LR warmup

# --- Image ---
target_size: null                   # Fixed resize (null = dynamic Pi3 resize)
pixel_limit: 255000                 # Max pixels for dynamic resize
patch_size: 14                      # Must match backbone (14 for DINOv2)

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
   cp configs/default_rohith.yaml configs/my_experiment.yaml
   ```

2. Update the server-specific paths:
   - `working_dir` — where logs and visualizations go
   - `save_path` — where checkpoints are saved
   - `data_path` — Action Genome dataset location

3. Choose your backbone (`model: "v2"` / `"v2l"` / `"v3l"`)

4. Set a unique `experiment_name` and `wandb_project`

5. Run:
   ```bash
   python -m lib.detector.monocular3d.train --config configs/my_experiment.yaml
   ```
