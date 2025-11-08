import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

# COCO evaluation imports
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_util
except ImportError:
    print("Warning: pycocotools not installed. Please install with: pip install pycocotools")
    COCO = None
    COCOeval = None

from dataloader.base_ag_dataset import BaseAG
from constants import Constants as const


class ActionGenomeEvaluationDataset(BaseAG):
    def __init__(self, phase="test", mode="sgdet", datasize="full", data_path=None, 
                 filter_nonperson_box_frame=True, filter_small_box=False):
        super().__init__(phase, mode, datasize, data_path, filter_nonperson_box_frame, filter_small_box)
        
    def __getitem__(self, index):
        frame_names = self.video_list[index]
        gt_annotations = self.gt_annotations[index]
        video_size = self.video_size[index]
        
        return {
            'frame_names': frame_names,
            'gt_annotations': gt_annotations,
            'video_size': video_size,
            'index': index
        }


class GDinoZeroShotEvaluator:
    
    def __init__(self, ag_root_directory: str, gdino_predictions_path: str = None):
        self.ag_root_directory = ag_root_directory
        self.gdino_predictions_path = gdino_predictions_path or os.path.join(ag_root_directory, "detection", "gdino")
        
        # Initialize dataset
        self.dataset = ActionGenomeEvaluationDataset(
            phase="test",
            mode="sgdet", 
            datasize="full",
            data_path=ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=False
        )
        
        # Object class mapping from Action Genome to GroundingDINO format
        self.ag_to_gdino_mapping = self._create_class_mapping()
        
        # Store evaluation results
        self.coco_gt = None
        self.coco_dt = None
        self.predictions = []
        self.ground_truths = []
        
    def _create_class_mapping(self) -> Dict[str, str]:
        # GroundingDINO object labels from ag_gdino.py
        gdino_labels = [
            "person", "bag", "blanket", "book", "box", "broom", "chair", "clothes", "cup", "dish", 
            "food", "laptop", "paper", "phone", "picture", "pillow", "sandwich", "shoe", "towel", 
            "vacuum", "glass", "bottle", "notebook", "camera"
        ]
        
        # Action Genome object classes (from the dataset)
        ag_classes = self.dataset.object_classes[1:]  # Skip background class
        
        # Create mapping - this is a simplified mapping, may need refinement
        mapping = {}
        for ag_class in ag_classes:
            # Direct matches
            if ag_class.lower() in gdino_labels:
                mapping[ag_class] = ag_class.lower()
            # Handle special cases
            elif ag_class == "closet/cabinet":
                mapping[ag_class] = "box"  # Approximate mapping
            elif ag_class == "cup/glass/bottle":
                mapping[ag_class] = "cup"
            elif ag_class == "paper/notebook":
                mapping[ag_class] = "paper"
            elif ag_class == "phone/camera":
                mapping[ag_class] = "phone"
            elif ag_class == "sofa/couch":
                mapping[ag_class] = "chair"  # Approximate mapping
            else:
                # Try to find the closest match
                ag_lower = ag_class.lower()
                for gdino_label in gdino_labels:
                    if gdino_label in ag_lower or ag_lower in gdino_label:
                        mapping[ag_class] = gdino_label
                        break
                else:
                    # Default fallback
                    mapping[ag_class] = "person"  # Default to person if no match
                    
        return mapping
    
    def load_gdino_predictions(self, video_id: str) -> Dict[int, Dict]:
        prediction_file = os.path.join(self.gdino_predictions_path, f"{video_id[:-4]}.pkl")
        
        if not os.path.exists(prediction_file):
            print(f"Warning: No predictions found for video {video_id}")
            return {}
            
        with open(prediction_file, 'rb') as f:
            predictions = torch.load(f)
            
        return predictions
    
    def convert_to_coco_format(self) -> tuple:
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        coco_predictions = []
        
        # Create categories
        for idx, class_name in enumerate(self.dataset.object_classes[1:], 1):  # Skip background
            coco_gt["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "object"
            })
        
        annotation_id = 1
        image_id = 1
        
        print("Converting annotations to COCO format...")
        
        for video_idx in tqdm(range(len(self.dataset))):
            video_data = self.dataset[video_idx]
            frame_names = video_data['frame_names']
            gt_annotations = video_data['gt_annotations']
            video_size = video_data['video_size']
            
            # Extract video ID from frame names
            video_id = frame_names[0].split('/')[0] + '.avi'
            
            # Load GroundingDINO predictions for this video
            gdino_predictions = self.load_gdino_predictions(video_id)
            
            for frame_idx, (frame_name, gt_frame) in enumerate(zip(frame_names, gt_annotations)):
                frame_number = int(frame_name.split('/')[-1].replace('.png', ''))
                
                # Add image info
                coco_gt["images"].append({
                    "id": image_id,
                    "file_name": frame_name,
                    "height": video_size[1],
                    "width": video_size[0]
                })
                
                # Add ground truth annotations
                for gt_obj in gt_frame[1:]:  # Skip person bbox (first element)
                    if const.VISIBLE in gt_obj and gt_obj[const.VISIBLE]:
                        bbox = gt_obj[const.BOUNDING_BOX]
                        # Convert from xyxy to xywh format
                        coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        area = coco_bbox[2] * coco_bbox[3]
                        
                        coco_gt["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": gt_obj[const.CLASS],
                            "bbox": coco_bbox,
                            "area": area,
                            "iscrowd": 0
                        })
                        annotation_id += 1
                
                # Add predictions for this frame
                if frame_number in gdino_predictions:
                    pred_frame = gdino_predictions[frame_number]
                    
                    for box, score, label in zip(pred_frame["boxes"], pred_frame["scores"], pred_frame["text_labels"]):
                        # Map GroundingDINO label to Action Genome class ID
                        ag_class_name = None
                        for ag_name, gdino_name in self.ag_to_gdino_mapping.items():
                            if gdino_name == label:
                                ag_class_name = ag_name
                                break
                        
                        if ag_class_name and ag_class_name in self.dataset.object_classes:
                            category_id = self.dataset.object_classes.index(ag_class_name)
                            
                            # Convert box format if needed
                            if isinstance(box, torch.Tensor):
                                box = box.cpu().numpy()
                            
                            # Convert from xyxy to xywh
                            coco_bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                            
                            coco_predictions.append({
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": coco_bbox,
                                "score": float(score),
                            })
                
                image_id += 1
        
        return coco_gt, coco_predictions
    
    def evaluate_map(self, iou_thresholds: List[float] = None) -> Dict[str, float]:
        print("Converting data to COCO format...")
        coco_gt_data, coco_predictions = self.convert_to_coco_format()
        
        # Create temporary files for COCO API
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_gt_data, f)
            gt_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_predictions, f)
            pred_file = f.name
        
        try:
            # Load COCO ground truth
            coco_gt = COCO(gt_file)
            
            # Load predictions
            coco_dt = coco_gt.loadRes(pred_file) if coco_predictions else COCO()
            
            # Initialize evaluator
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            
            # Set evaluation parameters
            if iou_thresholds:
                coco_eval.params.iouThrs = np.array(iou_thresholds)
            
            # Run evaluation
            print("Running COCO evaluation...")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract results
            results = {
                'mAP': coco_eval.stats[0],
                'mAP_50': coco_eval.stats[1],
                'mAP_75': coco_eval.stats[2],
                'mAP_small': coco_eval.stats[3],
                'mAP_medium': coco_eval.stats[4],
                'mAP_large': coco_eval.stats[5]
            }
            
            return results
            
        finally:
            # Clean up temporary files
            os.unlink(gt_file)
            os.unlink(pred_file)
    
    def print_evaluation_results(self, results: Dict[str, float]):
        """
        Print formatted evaluation results
        """
        print("\n" + "="*60)
        print("GROUNDING DINO ZERO-SHOT EVALUATION RESULTS")
        print("="*60)
        
        print(f"Overall mAP: {results['mAP']:.4f}")
        print(f"mAP@0.5: {results['mAP_50']:.4f}")
        print(f"mAP@0.75: {results['mAP_75']:.4f}")
        print(f"mAP (small): {results['mAP_small']:.4f}")
        print(f"mAP (medium): {results['mAP_medium']:.4f}")
        print(f"mAP (large): {results['mAP_large']:.4f}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GroundingDINO zero-shot performance on Action Genome")
    parser.add_argument(
        "--ag_root_directory",
        type=str,
        default="/data/rohith/ag/",
        help="Root directory of Action Genome dataset"
    )
    parser.add_argument(
        "--gdino_predictions_path", 
        type=str,
        default=None,
        help="Path to GroundingDINO predictions (default: ag_root_directory/detection/gdino)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gdino_evaluation_results.json",
        help="Output file for evaluation results"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    print("Initializing GroundingDINO zero-shot evaluator...")
    evaluator = GDinoZeroShotEvaluator(
        ag_root_directory=args.ag_root_directory,
        gdino_predictions_path=args.gdino_predictions_path
    )
    
    print(f"Loaded dataset with {len(evaluator.dataset)} videos")
    print(f"Object classes: {len(evaluator.dataset.object_classes)} classes")
    
    # Run evaluation
    print("Computing mAP scores...")
    results = evaluator.evaluate_map()
    
    # Print results
    evaluator.print_evaluation_results(results)
    
    # Save results
    output_data = {
        "overall_metrics": results,
        "dataset_info": {
            "num_videos": len(evaluator.dataset),
            "num_classes": len(evaluator.dataset.object_classes),
            "class_mapping": evaluator.ag_to_gdino_mapping
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()