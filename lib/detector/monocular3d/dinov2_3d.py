import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from dinov2_torch import create_model

class Monocular3DHead(nn.Module):
    def __init__(self, in_channels, representation_size=1024, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        # Output: 8 corners * 3 coordinates = 24 values
        # We predict this for each class or class-agnostic. 
        # The prompt implies "predicts the 3D bbox information".
        # Let's do class-agnostic for simplicity as per "lifting 2D bounding boxes to 3D cuboids in a class-agnostic manner" from the snippet.
        self.bbox_pred = nn.Linear(representation_size, 24) 
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        bbox_3d = self.bbox_pred(x)
        return bbox_3d

class DinoV2Monocular3D(nn.Module):
    def __init__(self, num_classes=37, pretrained=True, model="v3l"):
        super().__init__()
        # Create the base 2D detector
        self.base_detector = create_model(num_classes=num_classes, pretrained=pretrained, use_fpn=True, model=model)
        
        # Extract components
        self.backbone = self.base_detector.backbone
        self.rpn = self.base_detector.rpn
        self.roi_heads = self.base_detector.roi_heads
        self.transform = self.base_detector.transform
        
        # 3D Head components
        # We use the same ROI align resolution as the 2D head usually, or larger.
        # 2D head uses 7x7. Let's use 7x7.
        self.roi_pooler_3d = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['p2', 'p3', 'p4', 'p5'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Input channels to head: 256 (FPN output) * 7 * 7
        in_channels = 256 * 7 * 7
        self.head_3d = Monocular3DHead(in_channels)
        
    def forward(self, images, targets=None):
        # 1. Transform images (normalization, resizing if needed - though we resize in dataset)
        # GeneralizedRCNNTransform expects a list of tensors
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                images = [img for img in images]
        
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        
        # 2. Backbone features
        features = self.backbone(images.tensors)
        
        # 3. RPN
        if isinstance(features, torch.Tensor):
            features = {"0": features}
            
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        # 4. ROI Heads (2D Detection)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        # 5. 3D Head
        losses_3d = {}
        detections_3d = []
        
        if self.training:
            # During training, we use Ground Truth 2D boxes to train the 3D head
            # targets contains 'boxes' and 'boxes_3d'
            
            # We need to filter out targets that don't have valid 3D boxes if any?
            # Our dataset puts zeros if missing. We should mask them in loss.
            
            gt_boxes = [t['boxes'] for t in targets]
            gt_boxes_3d = [t['boxes_3d'] for t in targets]
            
            # Extract features for GT boxes
            # Note: roi_align expects boxes to be in the image coordinate system of 'images.tensors'
            # The 'targets' passed to self.transform are already transformed (resized) by GeneralizedRCNNTransform?
            # Yes, self.transform updates targets['boxes'].
            
            box_features = self.roi_pooler_3d(features, gt_boxes, images.image_sizes)
            pred_3d = self.head_3d(box_features)
            
            # Calculate Loss
            # Flatten GT 3D boxes
            gt_3d_flat = torch.cat(gt_boxes_3d, dim=0).view(-1, 24)
            
            # Simple L1 loss for now
            # Mask out invalid boxes (all zeros)
            mask = (gt_3d_flat.abs().sum(dim=1) > 0)
            if mask.sum() > 0:
                loss_3d = F.l1_loss(pred_3d[mask], gt_3d_flat[mask])
            else:
                loss_3d = torch.tensor(0.0, device=pred_3d.device, requires_grad=True)
                
            losses_3d = {'loss_3d': loss_3d}
            
        else:
            # Inference
            # Use detected boxes
            pred_boxes = [d['boxes'] for d in detections]
            
            # If no boxes detected, handle gracefully
            if all(len(b) == 0 for b in pred_boxes):
                for d in detections:
                    d['boxes_3d'] = torch.empty((0, 8, 3), device=d['boxes'].device)
            else:
                box_features = self.roi_pooler_3d(features, pred_boxes, images.image_sizes)
                pred_3d = self.head_3d(box_features)
                
                # Split predictions back to per-image
                boxes_per_image = [len(b) for b in pred_boxes]
                pred_3d_split = pred_3d.split(boxes_per_image)
                
                for i, d in enumerate(detections):
                    d['boxes_3d'] = pred_3d_split[i].view(-1, 8, 3)
                    
        # Combine losses
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        losses.update(losses_3d)
        
        if self.training:
            return losses
        else:
            return detections
