
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from huggingface_hub import login
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from transformers import AutoModel

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(_hf_token)

# Import loss
try:
    from ..losses.ovmono3d_loss import ovmono3d_loss
except ImportError:
    # For standalone testing
    from ovmono3d_loss import ovmono3d_loss

class FactorizedMonocular3DHead(nn.Module):
    """
    Factorized 3D Head: Disentangled regression for 3D box parameters.
    Predicts: Dimensions (3), Rotation (sin, cos), Depth (1), Center Offset (2), Uncertainty (1).
    Constructs 8 corners from these parameters.
    """
    def __init__(self, in_channels, representation_size=1024):
        super().__init__()
        
        # 1. Conv Reduction: 256x14x14 -> 128x7x7
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) 
        )
        
        reduced_channels = 128 * 7 * 7  # 6272
        
        # 2. FC Layers
        # Input: Features + Camera Intrinsics (4: fx, fy, cx, cy) + 2D BBox (4: x1, y1, x2, y2)
        self.fc1 = nn.Linear(reduced_channels + 8, representation_size) 
        self.fc2 = nn.Linear(representation_size, representation_size)
        
        # 3. Disentangled Heads
        self.dim_pred = nn.Linear(representation_size, 3)          # l, w, h
        self.rot_pred = nn.Linear(representation_size, 2)          # sin, cos
        self.depth_pred = nn.Linear(representation_size, 1)        # z
        self.center_offset_pred = nn.Linear(representation_size, 2) # du, dv
        self.mu_pred = nn.Linear(representation_size, 1)           # uncertainty

    def forward(self, x, bbox_2d, camera_intrinsics):
        """
        Args:
            x: (N, C, 14, 14) ROI features
            bbox_2d: (N, 4) - x1, y1, x2, y2
            camera_intrinsics: (N, 4) - fx, fy, cx, cy
        """
        # x: (N, C, 14, 14)
        x = self.conv_reduce(x)
        x = x.flatten(start_dim=1) # (N, 6272)
        
        # Concatenate geometric context
        # bbox_2d: (N, 4), camera_intrinsics: (N, 4) = [fx, fy, cx, cy]
        x = torch.cat([x, bbox_2d, camera_intrinsics], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Predict parameters
        dims = F.softplus(self.dim_pred(x)) # Ensure positive dimensions
        rot_sin_cos = self.rot_pred(x)
        depth = self.depth_pred(x)
        center_offset = self.center_offset_pred(x)
        mu = self.mu_pred(x).squeeze(-1)
        
        # Normalize rotation
        rot_sin_cos = F.normalize(rot_sin_cos, p=2, dim=1)
        
        # Compute corners using real intrinsics
        focal_lengths = camera_intrinsics[:, :2]    # (N, 2) -> fx, fy
        principal_point = camera_intrinsics[:, 2:]   # (N, 2) -> cx, cy
        bbox_3d = self.compute_3d_corners(dims, rot_sin_cos, depth, center_offset, bbox_2d, focal_lengths, principal_point)
        
        return bbox_3d, mu

    def compute_3d_corners(self, dims, rot_sin_cos, depth, center_offset, bbox_2d, focal_lengths, principal_point):
        """
        Reconstruct 8 corners from factorized parameters using proper pinhole back-projection.

        Args:
            dims: (N, 3) - l, w, h
            rot_sin_cos: (N, 2) - sin(theta), cos(theta)
            depth: (N, 1) - z
            center_offset: (N, 2) - du, dv (pixels)
            bbox_2d: (N, 4) - x1, y1, x2, y2
            focal_lengths: (N, 2) - fx, fy
            principal_point: (N, 2) - cx, cy (from camera calibration)
        """
        N = dims.shape[0]
        
        # 1. 2D Center + Offset
        cx_2d = (bbox_2d[:, 0] + bbox_2d[:, 2]) / 2.0
        cy_2d = (bbox_2d[:, 1] + bbox_2d[:, 3]) / 2.0
        
        u_final = cx_2d + center_offset[:, 0]
        v_final = cy_2d + center_offset[:, 1]
        
        # 2. Local corners (N, 8, 3)
        l, w, h = dims[:, 0], dims[:, 1], dims[:, 2]
        x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1)
        y_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1)
        z_corners = torch.stack([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=1)
        
        # 3. Rotate (yaw in xy-plane, consistent with OVMono3D loss)
        sin, cos = rot_sin_cos[:, 0:1], rot_sin_cos[:, 1:2]
        x_rot = x_corners * cos - y_corners * sin
        y_rot = x_corners * sin + y_corners * cos
        z_rot = z_corners
        
        # 4. Back-project 2D center to 3D using real intrinsics
        #    Standard pinhole: X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy
        z_c = depth  # (N, 1)
        px = principal_point[:, 0:1]  # (N, 1)
        py = principal_point[:, 1:2]  # (N, 1)
        fx = focal_lengths[:, 0:1]    # (N, 1)
        fy = focal_lengths[:, 1:2]    # (N, 1)
        
        x_c = (u_final.unsqueeze(1) - px) * z_c / fx
        y_c = (v_final.unsqueeze(1) - py) * z_c / fy
        
        # 5. Translate: add 3D center to rotated local corners
        x_world = x_rot + x_c
        y_world = y_rot + y_c
        z_world = z_rot + z_c
        
        corners_3d = torch.stack([x_world, y_world, z_world], dim=-1)  # (N, 8, 3)
        return corners_3d


class ConvAdapter(nn.Module):
    """
    Trainable convolutional adapter layers placed after frozen backbone.
    Adapts features for better object detection before RPN.
    """

    def __init__(self, in_channels=768, num_levels=4, num_layers=2):
        super().__init__()
        self.in_channels = in_channels

        # Create adapter layers for each feature level
        self.adapters = nn.ModuleDict()
        for i in range(num_levels):
            layers = []
            for layer_idx in range(num_layers):
                if layer_idx == 0:
                    # First layer: 3x3 conv for spatial adaptation
                    layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
                else:
                    # Subsequent layers: 1x1 conv for channel mixing
                    layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False))

                layers.append(nn.BatchNorm2d(in_channels))
                layers.append(nn.ReLU(inplace=True))

            self.adapters[f'level_{i}'] = nn.Sequential(*layers)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply conv adapters to backbone features with residual connection."""
        adapted_features = {}
        for level, feat in features.items():
            level_idx = int(level)
            adapter = self.adapters[f'level_{level_idx}']
            # Residual connection: helps preserve frozen backbone features
            adapted_features[level] = feat + 0.5 * adapter(feat)
        return adapted_features


class LastLevelMaxPool(nn.Module):
    """Pooling to create p6 feature map (for larger object detection)."""

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [nn.functional.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class SimpleFeaturePyramid(nn.Module):
    """
    Simple Feature Pyramid Network (SimpleFPN) adapted from ViTDet.
    Creates multiscale pyramid features from single-scale backbone output.
    """

    def __init__(
            self,
            in_channels=768,
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            top_block=None,
            norm="BN",
            model="v2"
    ):
        """
        Args:
            in_channels: Input feature channels from backbone (768 for DINOv2-base)
            out_channels: Output feature channels (typically 256)
            scale_factors: List of scaling factors for pyramid levels
            top_block: Optional top block to add p6 feature
            norm: Normalization type ('BN' or 'LN')
        """
        super().__init__()

        if model == "v3_7b":
            in_channels = 4096
        if model == "v3l":
            in_channels = 1024
        if model == "v3h":
            in_channels = 1280
        self.scale_factors = scale_factors
        self.top_block = top_block

        # Calculate strides (assuming patch_size=16, base stride=16)
        base_stride = 16
        strides = [int(base_stride / scale) for scale in scale_factors]

        # Create pyramid stages
        self.stages = nn.ModuleList()
        self._out_feature_strides = {}
        self._out_features = []

        for idx, scale in enumerate(scale_factors):
            out_dim = in_channels
            layers = []

            # Upsample/downsample layers
            if scale == 4.0:
                # 4x upsampling: 2x transpose convs
                layers.extend([
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(in_channels // 2) if norm == "BN" else nn.GroupNorm(1, in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2, bias=False),
                ])
                out_dim = in_channels // 4
            elif scale == 2.0:
                # 2x upsampling
                layers.extend(
                    [nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False), ])
                out_dim = in_channels // 2
            elif scale == 1.0:
                # No scaling
                pass
            elif scale == 0.5:
                # 2x downsampling
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported")

            # Channel reduction and refinement
            use_bias = norm == ""
            layers.extend([
                nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=use_bias),
                nn.BatchNorm2d(out_channels) if norm == "BN" else nn.GroupNorm(1, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
                nn.BatchNorm2d(out_channels) if norm == "BN" else nn.GroupNorm(1, out_channels),
            ])

            stage = nn.Sequential(*layers)
            self.stages.append(stage)

            # Feature map naming: p2, p3, p4, p5 (stride = 2^stage)
            stage_num = int(torch.log2(torch.tensor(strides[idx])).item())
            feat_name = f"p{stage_num}"
            self._out_feature_strides[feat_name] = strides[idx]
            self._out_features.append(feat_name)

        # Add top block features (p6)
        if self.top_block is not None:
            last_stage = int(torch.log2(torch.tensor(strides[-1])).item())
            for s in range(last_stage, last_stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)
                self._out_features.append(f"p{s + 1}")

        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self.out_channels = out_channels

    def forward(self, features) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Dict of backbone features {'0': tensor, '1': tensor, ...}

        Returns:
            Dict of pyramid features {'p2': tensor, 'p3': tensor, ...}
        """
        # Use the finest feature map (typically '0' or '1') as base
        # For DINOv2, we'll use the first feature map
        base_feature = features  # [B, C, H, W]

        results = []
        for stage in self.stages:
            results.append(stage(base_feature))

        # Add top block features (p6)
        if self.top_block is not None:
            top_block_in = results[-1]  # Use last pyramid feature
            top_results = self.top_block(top_block_in)
            results.extend(top_results)

        # Create output dict
        output_features = {f: res for f, res in zip(self._out_features, results)}
        return output_features


class Dinov3ModelBackbone(nn.Module):

    def __init__(self,
                 model_name='facebook/dinov3-vitl16-pretrain-lvd1689m'):  # facebook/dinov3-vitb16-pretrain-lvd1689m , facebook/dinov2-base
        super().__init__()  # facebook/dinov3-vith16plus-pretrain-lvd1689m
        self.model_name = model_name
        self.bck_model = AutoModel.from_pretrained(self.model_name)
        self.bck_model.eval()
        self.out_channels = self.bck_model.config.hidden_size
        self.patch_size = 14 # Default for DINOv2/v3

    def forward(self, x):
        # x shape is (B, 3, Img_H, Img_W)
        B, _, Img_H, Img_W = x.shape
        H, W = Img_H // self.patch_size, Img_W // self.patch_size
        
        outputs = self.bck_model(x)
        features = outputs.last_hidden_state
        
        # Remove the CLS token
        features = features[:, 1:, :] # (B, N, C)
        
        # Safely reshape using exact spatial dimensions
        # Note: If N != H*W due to interpolation or other factors, we might need logic.
        # But normally H*W = N for ViT if image size is multiple of patch size.
        # For DINOv2 with register tokens (optional), need to be careful. 
        # But standard DINOv2 usually has just CLS + patches.
        # In case of mismatch (e.g. registers), truncate/pad
        
        if features.shape[1] != H * W:
             # Fallback to nearest square or original logic if completely mismatched?
             # But prompt says: "Calculate H and W dynamically based on the input image shape"
             target_len = H * W
             if features.shape[1] > target_len:
                 features = features[:, :target_len, :]
             elif features.shape[1] < target_len:
                 # Should rare happen if input is divisible
                 pass
        
        features = features.permute(0, 2, 1).contiguous().view(B, self.out_channels, H, W)
        return features

class _NoResizeRCNNTransform(GeneralizedRCNNTransform):
    """GeneralizedRCNNTransform that normalizes and batches but does NOT resize.
    The dataset already produces images at the correct Pi3-compatible resolution
    (multiples of 14), so any further resizing would break the ViT patch grid.
    """

    def resize(self, image, target):
        # Skip resize entirely — return image and target unchanged
        return image, target


class DinoV3Monocular3D(nn.Module):

    def __init__(self, num_classes=37, pretrained=True, model="v3l"):
        super().__init__()
        # Create the base 2D detector
        self.base_detector = create_model(num_classes=num_classes, pretrained=pretrained, use_fpn=True, model=model)

        # Extract components
        self.backbone = self.base_detector.backbone
        self.rpn = self.base_detector.rpn
        self.roi_heads = self.base_detector.roi_heads
        
        # Image transform: normalize ONLY — no resizing.
        # The dataset already resizes to Pi3-compatible dims (multiples of 14).
        # GeneralizedRCNNTransform.resize would shrink/grow images based on min_size,
        # breaking the ViT patch grid. We override resize to be a no-op.
        # size_divisible=14 ensures batch padding stays on the DINOv2 patch grid.
        self.transform = _NoResizeRCNNTransform(
            min_size=800,   # ignored (resize is a no-op)
            max_size=1333,  # ignored (resize is a no-op)
            image_mean=[0.485, 0.456, 0.406],  # DINOv2 mean
            image_std=[0.229, 0.224, 0.225],   # DINOv2 std
            size_divisible=14,
        )
        
        # 3D Head components
        self.roi_pooler_3d = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['p2', 'p3', 'p4', 'p5'],
            output_size=14,
            sampling_ratio=2
        )

        # Input channels to head: 256 (FPN output)
        in_channels = 256
        self.head_3d = FactorizedMonocular3DHead(in_channels)

    def forward(self, images, targets=None):
        # 1. Transform images
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = list(images.unbind(0))

        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        # 2. Backbone features
        features = self.backbone(images.tensors)

        # 3. RPN
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, proposal_losses = self.rpn(images, features, targets)

        # 4. ROI Heads (2D Detection)
        # We need the 2D detections/proposals for the 3D head
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 5. 3D Head
        losses_3d = {}

        if self.training:
            # Proposal-based training: match RPN proposals to GT
            # targets contains 'boxes' (GT 2D), 'boxes_3d' (GT 3D),
            #   'focal_lengths' (2,), 'principal_point' (2,)
            
            final_proposals_3d = []
            final_gt_3d = []
            final_gt_2d_for_context = []
            final_intrinsics = []  # (N, 4) -> [fx, fy, cx, cy]
            
            for i in range(len(proposals)):
                prop = proposals[i]  # (N_p, 4)
                gt_box = targets[i]['boxes']  # (N_g, 4)
                gt_3d = targets[i]['boxes_3d']  # (N_g, 8, 3)
                
                # Match proposals to GT
                if len(gt_box) == 0 or len(prop) == 0:
                    continue
                    
                ious = torchvision.ops.box_iou(prop, gt_box)  # (N_p, N_g)
                val, idx = ious.max(dim=1)
                
                mask = val >= 0.5
                valid_props = prop[mask]
                matched_gt_indices = idx[mask]
                
                if len(valid_props) == 0:
                    continue
                
                matched_gt_3d = gt_3d[matched_gt_indices]  # (K, 8, 3)
                
                final_proposals_3d.append(valid_props)
                final_gt_3d.append(matched_gt_3d)
                final_gt_2d_for_context.append(valid_props)
                
                # Per-image intrinsics: expand to match number of valid proposals
                fl = targets[i].get('focal_lengths', torch.tensor([500.0, 500.0], device=prop.device))
                pp = targets[i].get('principal_point', torch.tensor([0.0, 0.0], device=prop.device))
                intr = torch.cat([fl, pp], dim=0)  # (4,)
                final_intrinsics.append(intr.unsqueeze(0).expand(len(valid_props), -1))

            if len(final_proposals_3d) > 0:
                box_features = self.roi_pooler_3d(features, final_proposals_3d, images.image_sizes)
                
                cat_bboxes = torch.cat(final_gt_2d_for_context, dim=0)  # (Total_N, 4)
                cat_gt_3d = torch.cat(final_gt_3d, dim=0)
                cat_intrinsics = torch.cat(final_intrinsics, dim=0)  # (Total_N, 4)
                
                pred_3d, pred_mu = self.head_3d(box_features, cat_bboxes, cat_intrinsics)
                
                loss_3d, _l3d, _chamfer = ovmono3d_loss(
                    pred_3d.view(-1, 24), cat_gt_3d, pred_mu, use_smooth_l1=True
                )
                losses_3d = {'loss_3d': loss_3d}
            else:
                losses_3d = {'loss_3d': torch.tensor(0.0, device=features['p2'].device, requires_grad=True)}

        else:
            # Inference
            pred_boxes = [d['boxes'] for d in detections]
            
            if all(len(b) == 0 for b in pred_boxes):
                for d in detections:
                    d['boxes_3d'] = torch.empty((0, 8, 3), device=d['boxes'].device)
            else:
                intrinsics_list = []
                for i in range(len(pred_boxes)):
                    N_det = len(pred_boxes[i])
                    if targets is not None and i < len(targets):
                        fl = targets[i].get('focal_lengths', torch.tensor([500.0, 500.0], device=pred_boxes[i].device))
                        pp = targets[i].get('principal_point', torch.tensor([0.0, 0.0], device=pred_boxes[i].device))
                    else:
                        fl = torch.tensor([500.0, 500.0], device=pred_boxes[i].device)
                        pp = torch.tensor([0.0, 0.0], device=pred_boxes[i].device)
                    intr = torch.cat([fl, pp], dim=0)  # (4,)
                    intrinsics_list.append(intr.unsqueeze(0).expand(N_det, -1))
                     
                with torch.no_grad():
                    box_features = self.roi_pooler_3d(features, pred_boxes, images.image_sizes)
                    
                    cat_boxes = torch.cat(pred_boxes, dim=0)
                    cat_intrinsics = torch.cat(intrinsics_list, dim=0)
                    
                    pred_3d, _ = self.head_3d(box_features, cat_boxes, cat_intrinsics)
                    
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


def create_model(num_classes=37, pretrained=True, coco_model=False, use_fpn=True, model="v2"):
    # Create base backbone
    base_backbone = Dinov3ModelBackbone()
    base_backbone.out_channels = 768

    # Freeze backbone (keep adapter trainable)
    for name, params in base_backbone.named_parameters():
        if 'adapter' not in name:
            params.requires_grad_(False)

    # Wrap with SimpleFeaturePyramid
    if use_fpn:
        backbone = SimpleFeaturePyramid(
            in_channels=768,
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),  # Creates p2, p3, p4, p5
            top_block=LastLevelMaxPool(),  # Adds p6
            norm="BN",
            model=model,
        )

        class BackboneWithFPN(nn.Module):
            def __init__(self, base_backbone, fpn):
                super().__init__()
                self.base_backbone = base_backbone
                self.fpn = fpn
                self.out_channels = fpn.out_channels

            def forward(self, x):
                with torch.no_grad():
                    features = self.base_backbone(x)
                return self.fpn(features)

        backbone = BackboneWithFPN(base_backbone, backbone)
        print("✅ Using SimpleFeaturePyramid (FPN)")
    else:
        backbone = base_backbone
        backbone.out_channels = 768
        print("Using backbone without FPN")

    # Anchor generator
    if use_fpn:
        featmap_names = backbone.fpn._out_features
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128, 256, 512, 1024),) * len(featmap_names),
            aspect_ratios=((0.5, 1.0, 2.0),) * len(featmap_names)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=7,
            sampling_ratio=2
        )
    else:
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),) * 4,
            aspect_ratios=((0.5, 1.0, 2.0),) * 4
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model
