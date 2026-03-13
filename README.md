<h1 align=center>
  Spatio Temporal Scene Graph Generation and Anticipation in Dynamic 4D Scenes
</h1>

<p align=center>  
  Rohith Peddi, Shravan Shanmugam, Saurabh, Likhitha Pallapothula, Yu Xiang, Parag Singla, Vibhav Gogate
</p>

<p align="center">
    Under Review
</p>

<div align=center>
  <a src="https://img.shields.io/badge/project-website-green" href="https://rohithpeddi.github.io/#/impartail">
    <img src="https://img.shields.io/badge/project-website-green">
  </a>
  <a src="https://img.shields.io/badge/paper-arxiv-red" href="https://arxiv.org/pdf/2403.04899v1.pdf">
    <img src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <a src="https://img.shields.io/badge/bibtex-citation-blue" href="">
    <img src="https://img.shields.io/badge/bibtex-citation-blue">
  </a> 
</div>

<p align="center">
  (This page is under continuous update)
</p>

----

## UPDATE


-------
### ACKNOWLEDGEMENTS

This code is based on the following awesome repositories. 
We thank all the authors for releasing their code. 


-------
# SETUP

## Environment Setup

```bash
conda create -n scene4cast python=3.12 pip

conda activate scene4cast


pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```




## Dataset Preparation 

------

## Monocular 3D Object Detector

DINOv2/DINOv3-based monocular 3D detector trained on Action Genome. Combines a frozen ViT backbone with SimpleFPN, Faster R-CNN (2D detection), and a factorized 3D head for 8-corner 3D bounding boxes.

### Backbone Variants

| Config Key | Model | Hidden Size | Patch Size | Params |
|------------|-------|-------------|------------|--------|
| `v2` | DINOv2 ViT-B | 768 | 14 | 86M |
| `v2l` | DINOv2 ViT-L | 1024 | 14 | 304M |
| `v3l` | DINOv3 ViT-L | 1024 | 16 | 304M |

### Training

Training configs are in `configs/detector/`. Each YAML specifies the backbone, paths, hyperparameters, and 3D head mode (`unified` or `separate`).

```bash
# Train with a specific config
python -m lib.detector.monocular3d.train --config configs/detector/default_rohith.yaml

# Override any field via CLI
python -m lib.detector.monocular3d.train \
    --config configs/detector/dinov2_saurabh_separate_v1.yaml \
    --lr 5e-5 --batch_size 64

# Resume from checkpoint
python -m lib.detector.monocular3d.train \
    --config configs/detector/default_rohith.yaml \
    --ckpt checkpoint_10
```

Checkpoints are saved to `{working_dir}/{experiment_name}/checkpoint_{epoch}/checkpoint_state.pth`.

------

## ROI Feature Extraction

Extracts pre-computed 1024-d ROI features from trained detectors for all annotated objects per frame. Used downstream by WorldSGG methods instead of running the detector on-the-fly.

### How It Works

1. Loads a trained `DinoV3Monocular3D` checkpoint
2. For each frame: collects GT bboxes + GDino detections for missing object labels
3. Preprocesses images identically to training (`_compute_target_size` + DINOv2 normalization)
4. Runs frozen backbone → FPN → ROI pooling → `box_head` → 1024-d features
5. Saves per-video PKL files

### Extraction Configs

Feature extraction configs are in `configs/features/predcls/`, organized by architecture and server:

| Config | Backbone | Server |
|--------|----------|--------|
| `ex_roi_feat_v1_dinov2b_rohith.yaml` | DINOv2-Base | Rohith |
| `ex_roi_feat_v1_dinov2b_saurabh.yaml` | DINOv2-Base | Saurabh |
| `ex_roi_feat_v1_dinov2_large_rohith.yaml` | DINOv2-Large | Rohith |
| `ex_roi_feat_v1_dinov2_large_saurabh.yaml` | DINOv2-Large | Saurabh |
| `ex_roi_feat_v1_dinov3_large_rohith.yaml` | DINOv3-Large | Rohith |
| `ex_roi_feat_v1_dinov3_large_saurabh.yaml` | DINOv3-Large | Saurabh |

### Running

```bash
# DINOv2-Base features
python datasets/preprocess/features/extract_roi_features.py \
    --config configs/features/predcls/ex_roi_feat_v1_dinov2b_saurabh.yaml

# DINOv2-Large features
python datasets/preprocess/features/extract_roi_features.py \
    --config configs/features/predcls/ex_roi_feat_v1_dinov2_large_saurabh.yaml

# DINOv3-Large features
python datasets/preprocess/features/extract_roi_features.py \
    --config configs/features/predcls/ex_roi_feat_v1_dinov3_large_saurabh.yaml

# Single video (for testing)
python datasets/preprocess/features/extract_roi_features.py \
    --config configs/features/predcls/ex_roi_feat_v1_dinov2b_saurabh.yaml \
    --video 001YG.mp4
```

### Output Format

Features are saved to `{data_path}/features/roi_features/predcls/{dinov2b|dinov2l|dinov3l}/{video_stem}.pkl`:

```python
{
    "video_id": "001YG.mp4",
    "model": "v2",
    "feature_dim": 1024,
    "checkpoint": "...",
    "frames": {
        "000001.png": {
            "roi_features": np.ndarray,   # (N, 1024) float16
            "bboxes_xyxy": np.ndarray,    # (N, 4) float32
            "labels": ["person", "chair", ...],
            "sources": ["gt", "gt", "gdino", ...],
            "gdino_scores": [-1.0, -1.0, 0.85, ...],
        },
    }
}
```

------

# Instructions to run

Please see the scripts/training for Python modules.

Please see the scripts/tests for testing Python modules.

------


### Notes

To remove .DS_Store files from a directory, run the following command in the terminal:

```bash
find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch
``` 
