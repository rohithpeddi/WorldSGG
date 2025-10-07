# Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video

David Yifan Yao, Albert J. Zhai, Shenlong Wang  
University of Illinois at Urbana-Champaign

CVPR 2025 (Highlight)

[Paper](https://arxiv.org/abs/2503.21761) | [Project Page](https://davidyao99.github.io/uni4d/)

**4D Reconstruction from Single Casual Video**  
![Teaser](assets/teaser.png)

## Setup

Uni4D relies on several visual foundation model repositories for preprocessing, included as submodules:

```bash
git clone --recursive https://github.com/Davidyao99/uni4d_dev.git
```

An installation script is provided and tested with CUDA Toolkit 12.1:

```bash
conda create -n uni4d python=3.10
conda activate uni4d
bash scripts/install.sh
```

Download model weights for the visual foundation models:
```bash
bash scripts/download_weights.sh
```

Our preprocessing involves calling GPT via the [OpenAI API](https://platform.openai.com/). This requires an account with credits (usage will be minimal). Find your API key [here](https://platform.openai.com/api-keys) and set it as an environment variable:

```bash
echo "OPENAI_API_KEY=sk-your_api_key_here" > .env
```

## Demo

We've included the lady-running sequence from DAVIS as a demonstration:

```bash
bash scripts/demo.sh
```

The output directory follows this structure:

```
data/demo/
└── lady-running/                # Demo sequence name
    ├── rgb/                     # Original RGB images
    │   ├── 00000.jpg            # Sequential RGB frames
    │   └── ...
    │
    ├── deva/                    # DEVA model outputs
    │   ├── pred.json            # Prediction data
    │   ├── args.txt             # Arguments/parameters
    │   ├── Annotations/         # Annotation masks
    │   │   ├── 00000.png
    │   │   └── ...
    │   └── Visualizations/      # Visual results
    │       ├── 00000.jpg
    │       └── ...
    │
    ├── ram/                     # RAM model detection results
    │   └── tags.json            # Contains RAM detected classes, GPT output, and filtered classes
    │
    ├── unidepth/                # Depth estimation results
    │   ├── depth_vis/           # Visualizations of depth maps
    │   │   ├── 00000.png
    │   │   └── ...
    │   ├── depth.npy            # Raw depth data
    │   └── intrinsics.npy       # Camera intrinsic parameters
    │
    ├── gsm2/                    # GSM2 model outputs
    │   ├── mask/                # Segmentation masks
    │   │   ├── 00000.png        # Binary mask image
    │   │   ├── 00000.json       # Metadata/parameters
    │   │   └── ...
    │   └── vis/                 # Mask visualizations
    │       ├── 00000.jpg
    │       └── ...
    │
    ├── cotrackerv3_F_G/        # Cotracker outputs (F=frames, G=grid size)
    │   ├── results.npz          # Tracklet data
    │   └── vis/                 # Cotracker visualizations
    │       ├── 00000.png
    │       └── ...
    │
    └── uni4d/                   # Uni4D model outputs
        └── experiment_name/     # Experiment Name
            ├── fused_4d.npz     # Fused 4D representation data
            ├── timer.txt        # Runtime for each stage
            ├── training_info.log # Training logs
            └── *.npz            # Raw results
```

## Customization

Uni4D is modular and any component can be swapped for other visual foundation model outputs. For custom vdeo depth estimation and dynamic masks, save them in the following format:
```
data/demo/
└── lady-running/                # Demo sequence name
    ├── rgb/                     # Original RGB images
    │   ├── 00000.jpg            # Sequential RGB frames
    │   └── ...
    ├── custom_depth/            # Custom Depth estimation results
    │   ├── depth.npy            # Raw depth data saved as F x 1 x H x W
    │   └── intrinsics.npy       # Camera intrinsic parameters (intrinsics to initialize Uni4D with stored as 3x3 matrix)
    ├── custom_segmentation/     # Custom Segmentation results
    │   ├── Annotations/         # Annotation masks
    │   │   ├── 00000.png        # tracklet id if 1 channel, unique RGB for each tracklet if 3 channel
    ...
```

When running Uni4d optimization, use the following arguments to use your custom preprocessed estimations:
```
python ./uni4d/run.py --config ./uni4d/config/config_demo.yaml --depth_net <custom_depth> --dyn_mask_dir <custom_segmentation>
```

## Evaluation

We provide scripts to prepare all datasets, preprocessing, and run Uni4D to reproduce Table 1 and 2 in our main paper.

### Datasets

We provide scripts for downloading and preparing datasets (Bonn, TUM-Dynamics, Sintel, Kitti) used in our paper. Run:

```bash
bash ./scripts/prepare_datasets.sh
```

### Preprocessing

You can either preprocess all datasets from scratch or download our preprocessed data that we used for the quantitative results in our paper:

Option 1: Preprocess from scratch
```bash
bash ./scripts/prepare_datasets.sh
```

Option 2: Download our preprocessed data
```bash
cd data
wget https://uofi.box.com/shared/static/jkbbb0u2oacubd2qhquoweqx9xecydmi.zip -O preprocessed.zip
unzip preprocessed.zip
```

### Running Uni4D

```bash
bash ./scripts/run_experiment.sh
```

### Metrics Evaluation

We provide evaluation scripts for pose and depth metrics:

```bash
bash ./scripts/eval.sh
```

## Acknowledgements

Our codebase is based on [CasualSAM](https://github.com/ztzhang/casualSAM). Our evaluation and dataset preparation is based on [MonST3R](https://github.com/Junyi42/monst3r). Our preprocessing relies on [Tracking-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA), [Grounded-Sam-2](https://github.com/IDEA-Research/Grounded-SAM-2), [CotrackerV3](https://github.com/facebookresearch/co-tracker), [Unidepth](https://github.com/lpiccinelli-eth/UniDepth), and [Recognize-Anything](https://github.com/xinyu1205/recognize-anything). We thank the authors for their excellent work!
