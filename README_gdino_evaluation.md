# GroundingDINO Zero-Shot Evaluation on Action Genome Dataset

This script evaluates the zero-shot object detection performance of GroundingDINO on the Action Genome dataset using mAP (mean Average Precision) scores calculated with pycocotools.

## Features

- **Complete mAP Evaluation**: Computes standard COCO evaluation metrics including mAP@0.5, mAP@0.75, and size-specific metrics
- **Class Mapping**: Handles mapping between Action Genome object classes and GroundingDINO object labels
- **COCO Format Conversion**: Converts Action Genome annotations and GroundingDINO predictions to COCO format for evaluation
- **Comprehensive Results**: Provides detailed evaluation results with JSON output for further analysis

## Requirements

```bash
pip install pycocotools torch numpy tqdm
```

## Usage

### Basic Usage

```bash
python test_gdino_zero_shot.py \
    --ag_root_directory /path/to/action_genome_dataset/ \
    --gdino_predictions_path /path/to/gdino/predictions/ \
    --output_file evaluation_results.json
```

### Parameters

- `--ag_root_directory`: Root directory of the Action Genome dataset (contains frames/, annotations/, etc.)
- `--gdino_predictions_path`: Path to GroundingDINO predictions (default: ag_root_directory/detection/gdino)
- `--output_file`: Output JSON file for evaluation results (default: gdino_evaluation_results.json)

## Expected Directory Structure

```
action_genome_dataset/
├── frames/
│   ├── video1.avi/
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   └── ...
├── annotations/
│   ├── object_classes.txt
│   ├── person_bbox.pkl
│   └── object_bbox_and_relationship.pkl
└── detection/
    └── gdino/
        ├── video1.pkl
        ├── video2.pkl
        └── ...
```

## Output

The script generates:

1. **Console Output**: Real-time progress and final mAP scores
2. **JSON Results File**: Detailed metrics and metadata including:
   - Overall mAP scores (mAP, mAP@0.5, mAP@0.75, etc.)
   - Dataset information (number of videos, classes, etc.)
   - Class mapping between Action Genome and GroundingDINO

### Example Output

```
======================================================
GROUNDING DINO ZERO-SHOT EVALUATION RESULTS
======================================================
Overall mAP: 0.2845
mAP@0.5: 0.4521
mAP@0.75: 0.2987
mAP (small): 0.1234
mAP (medium): 0.2876
mAP (large): 0.3654
======================================================
```

## Implementation Details

### Class Mapping

The script automatically maps Action Genome object classes to GroundingDINO labels:
- Direct matches (e.g., "chair" → "chair")
- Special cases (e.g., "cup/glass/bottle" → "cup")
- Approximate mappings (e.g., "sofa/couch" → "chair")

### COCO Format Conversion

- Ground truth annotations are converted from Action Genome's xyxy format to COCO's xywh format
- GroundingDINO predictions are mapped to corresponding Action Genome categories
- Image metadata is preserved for proper evaluation

### Evaluation Metrics

Uses standard COCO evaluation metrics:
- **mAP**: Mean Average Precision over IoU thresholds 0.5:0.05:0.95
- **mAP@0.5**: Average Precision at IoU threshold 0.5
- **mAP@0.75**: Average Precision at IoU threshold 0.75
- **Size-specific metrics**: mAP for small, medium, and large objects

## Error Handling

- Graceful handling of missing prediction files
- Automatic class mapping with fallbacks
- Temporary file cleanup for COCO evaluation
- Clear error messages for missing dependencies

## Notes

- Requires pre-generated GroundingDINO predictions (use `ag_gdino.py` to generate)
- Evaluation is performed on the test split of Action Genome dataset
- Results are compatible with standard COCO evaluation protocols
