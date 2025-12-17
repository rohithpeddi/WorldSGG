import torch
from torch.utils.data import DataLoader
from constant import const
from dataloader.ag_dataset import ActionGenomeDataset,collate_fn
from model.dinov2 import DINOv2Detector
import torch.nn as nn
import torch.nn.functional as F
import pdb
from tqdm import tqdm
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np


import gc, torch

def clear_cuda_cache_for_current_process(sync=True):
    gc.collect()
    if not torch.cuda.is_available():
        return
    # synchronize to ensure kernels finished
    if sync:
        torch.cuda.synchronize()
    # clear cache on visible devices for this process (often 1 device)
    for dev in range(torch.cuda.device_count()):
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    # optional: torch.cuda.synchronize()  # re-sync if needed afterwards


def compute_iou(box1, box2):
    # Convert to numpy floats for safe scalar arithmetic
    if not isinstance(box1, np.ndarray):
        box1 = box1.detach().cpu().numpy()
    if not isinstance(box2, np.ndarray):
        box2 = box2.detach().cpu().numpy()

    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    b1_area = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    b2_area = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

    union_area =  b1_area + b2_area - inter_area

    return inter_area/union_area if union_area>0 else 0.0

def evaluate_predictions(model, dataloader, num_batches, device, iou_threshold = 0.5):
    model.eval()
    with torch.no_grad():
        total_precision, total_recall = 0.0, 0.0
        batch_count = 0
        for images, gt in tqdm(dataloader, ascii=True):
            images = torch.stack([img for img in images]).to(device)
            pred, _ = model(images, targets=None, mode="test")
    
            TP, FP, FN = 0, 0, 0

            for i in range(len(pred["boxes"])):
                pred_boxes = pred["boxes"][i].detach().cpu()
                pred_labels = pred["labels"][i].detach().cpu()
                pred_scores = pred["scores"][i].detach().cpu()

                gt_boxes = gt[i]["boxes"].cpu()
                gt_labels = gt[i]["labels"].cpu()

                matched_gt = set()

                for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
                    ious = [compute_iou(pb,gb) for gb in gt_boxes]
                    max_iou_idx = np.argmax(ious) if len(ious)>0 else -1

                    if max_iou_idx >= 0 and ious[max_iou_idx] >= iou_threshold and pl == gt_labels[max_iou_idx]:
                        if max_iou_idx not in matched_gt:
                            TP+=1
                            matched_gt.add(max_iou_idx)
                        else:
                            FP += 1
                    else:
                        FP+=1

                FN += len(gt_boxes) - len(matched_gt)

            precision = TP/(TP+FP+1e-7)
            recall = TP/(TP+FN+1e-7)
            total_precision+=precision
            total_recall+=recall
            batch_count += 1
        denom = batch_count if batch_count > 0 else num_batches
        return total_precision/denom, total_recall/denom


# def evaluate_MAP_full(model, dataloader, device, accelerator):
#     model.eval()
#     metric = MeanAveragePrecision(iou_type="bbox")

#     with torch.no_grad():
#         for images, targets in tqdm(dataloader, ascii=True):
#             images = torch.stack([img for img in images]).to(device)
            
#             # For Faster R-CNN, call model in eval mode (no targets needed)
#             outputs = model(images)
            
#             preds, gts = [], []
            
#             # Process predictions - Faster R-CNN returns list of dicts
#             for i, output in enumerate(outputs):
#                 # Filter predictions by confidence threshold (optional)
#                 scores = output['scores']
#                 keep = scores > 0.5  # Adjust threshold as needed
                
#                 preds.append({
#                     "boxes": output['boxes'][keep].detach().cpu(),
#                     "scores": output['scores'][keep].detach().cpu(),
#                     "labels": output['labels'][keep].detach().cpu(),
#                 })

#             # Process ground truth targets
#             for t in targets:
#                 gts.append({
#                     "boxes": t["boxes"].cpu(),
#                     "labels": t["labels"].cpu()
#                 })

#             metric.update(preds, gts)
#             # torch.cuda.empty_cache()
#             # for gpu_id in range(torch.cuda.device_count()):
#             #     with torch.cuda.device(gpu_id):
#             #         torch.cuda.empty_cache()
#             #         torch.cuda.synchronize() 
#     # Return just the mAP value
#     result = metric.compute()
#     return result


# def evaluate_MAP_full(model, dataloader, device, accelerator):
#     model.eval()
#     metric = MeanAveragePrecision(iou_type="bbox",sync_on_compute=False)
#     frame_batch_size = 10  # Process 10 frames at a time

#     with torch.no_grad():
#         for images, targets in tqdm(dataloader, ascii=True):
#             # images is a list of frames for one video
#             total_frames = len(images)
            
#             # Process frames in batches of 10
#             for start_idx in range(0, total_frames, frame_batch_size):
#                 end_idx = min(start_idx + frame_batch_size, total_frames)
#                 batch_frames = images[start_idx:end_idx]
                
#                 # Convert to tensor and move to device
#                 batch_images = torch.stack(batch_frames).to(device)
#                 # Get predictions for this batch of frames
#                 outputs = model(batch_images)
                
#                 preds, gts = [], []
                
#                 # Process predictions for this batch
#                 for i, output in enumerate(outputs):
#                     # Filter predictions by confidence threshold
#                     scores = output['scores']
#                     keep = scores > 0.0  # Adjust threshold as needed
                    
#                     preds.append({
#                         "boxes": output['boxes'][keep].detach().cpu(),
#                         "scores": output['scores'][keep].detach().cpu(),
#                         "labels": output['labels'][keep].detach().cpu() ,
#                     })

#                 # Process ground truth targets for this batch
#                 # Create one target per frame in the batch
#                 for i in range(start_idx,end_idx):
#                     # Use the first target for all frames (assuming targets are per-video)
#                     gts.append({
#                         "boxes": targets[i]["boxes"].detach().cpu(),
#                         "labels": targets[i]["labels"].detach().cpu()
#                     })

#                 metric.update(preds, gts)
                
#                 # Clear cache after each batch to prevent OOM
#                 torch.cuda.empty_cache()
    
#     # Return the final mAP value
#     result = metric.compute()
#     return result


def evaluate_MAP_full(model, dataloader, device, accelerator):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", sync_on_compute=False)
    frame_batch_size = 10
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, ascii=True):
            total_frames = len(images)
            
            for start_idx in range(0, total_frames, frame_batch_size):
                end_idx = min(start_idx + frame_batch_size, total_frames)
                batch_frames = images[start_idx:end_idx]
                
                batch_images = torch.stack(batch_frames).to(device)
                outputs = model(batch_images)
                
                preds, gts = [], []
                
                # Process predictions for this batch
                for i, output in enumerate(outputs):
                    scores = output['scores']
                    keep = scores > 0.0
                    
                    # Always add a prediction dict, even if empty
                    preds.append({
                        "boxes": output['boxes'].detach().cpu(),
                        "scores": output['scores'].detach().cpu(),
                        "labels": output['labels'].detach().cpu(),
                    })

                # Process ground truth targets for this batch
                for i in range(start_idx, end_idx):
                    # Always add a ground truth dict, even if empty
                    gts.append({
                        "boxes": targets[i]["boxes"].detach().cpu(),
                        "labels": targets[i]["labels"].detach().cpu()
                    })
                
                # CRITICAL: Ensure equal lengths
                min_len = min(len(preds), len(gts))
                if min_len > 0:
                    metric.update(preds[:min_len], gts[:min_len])
                
                torch.cuda.empty_cache()
    
    if accelerator.is_main_process:
        result = metric.compute()
        return result
    else:
        return {}
