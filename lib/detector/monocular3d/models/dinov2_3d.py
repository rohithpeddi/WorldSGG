import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .dinov2_torch import create_model
from ..losses.ovmono3d_loss import ovmono3d_loss


class Monocular3DHead(nn.Module):
    """Lifting head: 8 corners (24) + uncertainty µ (1) for OVMono3D-style loss."""
    def __init__(self, in_channels, representation_size=1024, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.bbox_pred = nn.Linear(representation_size, 24)   # 8 corners * 3
        self.mu_pred = nn.Linear(representation_size, 1)      # uncertainty for L = sqrt(2)*exp(-µ)*L3D + µ

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        bbox_3d = self.bbox_pred(x)   # (N, 24)
        mu = self.mu_pred(x).squeeze(-1)  # (N,)
        return bbox_3d, mu

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
        self.roi_pooler_3d = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['p2', 'p3', 'p4', 'p5'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Input channels to head: 256 (FPN output) * 7 * 7
        in_channels = 256 * 7 * 7
        self.head_3d = Monocular3DHead(in_channels)
        
    def forward(self, images, targets=None):
        # 1. Transform images — GeneralizedRCNNTransform expects a list of tensors
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = list(images.unbind(0))  # unbind is faster than list comprehension

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

        if self.training:
            # Use GT 2D boxes to train the 3D head
            gt_boxes = [t['boxes'] for t in targets]
            gt_boxes_3d = [t['boxes_3d'] for t in targets]

            box_features = self.roi_pooler_3d(features, gt_boxes, images.image_sizes)
            pred_3d, pred_mu = self.head_3d(box_features)
            gt_corners = torch.cat(gt_boxes_3d, dim=0)  # (N, 8, 3)

            loss_3d, _l3d, _chamfer = ovmono3d_loss(
                pred_3d, gt_corners, pred_mu, use_smooth_l1=True
            )
            losses_3d = {'loss_3d': loss_3d}

        else:
            # Inference: use detected boxes from 2D head
            pred_boxes = [d['boxes'] for d in detections]

            if all(len(b) == 0 for b in pred_boxes):
                for d in detections:
                    d['boxes_3d'] = torch.empty((0, 8, 3), device=d['boxes'].device)
            else:
                with torch.no_grad():
                    box_features = self.roi_pooler_3d(features, pred_boxes, images.image_sizes)
                    pred_3d, _ = self.head_3d(box_features)
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
