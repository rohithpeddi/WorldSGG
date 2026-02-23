
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

# ---------------------------------------------------------------------------
# Model registry: config key → HuggingFace model ID
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    "v2":    "facebook/dinov2-base",       # ViT-B/14  86M   hidden_size=768
    "v2l":   "facebook/dinov2-large",      # ViT-L/14  304M  hidden_size=1024
    "v3l":   "facebook/dinov3-vitl16-pretrain-lvd1689m",   # hidden_size=1024
}

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
    ):
        """
        Args:
            in_channels: Input feature channels from backbone (auto-detected)
            out_channels: Output feature channels (typically 256)
            scale_factors: List of scaling factors for pyramid levels
            top_block: Optional top block to add p6 feature
            norm: Normalization type ('BN' or 'LN')
        """
        super().__init__()

        # in_channels is now passed directly from base_backbone.out_channels
        # (no more hardcoded overrides per model variant)
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

    def __init__(self, model: str = "v2"):
        """
        Args:
            model: Key into MODEL_REGISTRY (e.g. 'v2', 'v2s', 'v2l', 'v3l').
                   Defaults to 'v2' (DINOv2-Base, 86M params — fastest option).
        """
        super().__init__()
        if model in MODEL_REGISTRY:
            model_name = MODEL_REGISTRY[model]
        else:
            # Allow direct HuggingFace model IDs for flexibility
            model_name = model
        self.model_name = model_name
        self.bck_model = AutoModel.from_pretrained(self.model_name)
        self.bck_model.eval()
        # Read actual hidden_size and patch_size from model config
        self.out_channels = self.bck_model.config.hidden_size
        self.patch_size = getattr(self.bck_model.config, 'patch_size', 14)
        print(f"  Backbone: {model_name}  hidden_size={self.out_channels}  patch_size={self.patch_size}")

    def forward(self, x):
        # x shape is (B, 3, Img_H, Img_W)
        B, _, Img_H, Img_W = x.shape
        H = Img_H // self.patch_size
        W = Img_W // self.patch_size
        n_patches = H * W

        # Use autocast for aggressive AMP — backbone is frozen, so float16/bfloat16 is safe
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = self.bck_model(x)
        features = outputs.last_hidden_state  # (B, N_total, C)
        C = features.shape[-1]

        # Strip CLS + any register tokens: keep only the last n_patches spatial tokens.
        features = features[:, -n_patches:, :]  # (B, H*W, C)

        features = features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return features

class _NoOpRCNNTransform(GeneralizedRCNNTransform):
    """GeneralizedRCNNTransform that ONLY batches — no resize, no normalization.
    The dataset already produces normalized images at the correct Pi3-compatible
    resolution (multiples of patch_size), so any further transforms are pure overhead.
    """

    def resize(self, image, target):
        return image, target

    def normalize(self, image):
        # Skip normalization — already done in dataset __getitem__
        return image


class DinoV3Monocular3D(nn.Module):
    """
    Full Monocular 3D Object Detector combining:
      - DINOv2 frozen backbone (ViT) for feature extraction
      - SimpleFPN for multi-scale feature pyramid
      - Faster R-CNN (RPN + ROI heads) for 2D detection
      - FactorizedMonocular3DHead for 3D bounding box regression

    Forward pass pipeline:
      images → Transform → Backbone → FPN → RPN → ROI Heads → 3D Head
               (batch)    (frozen)   (p2-p6)  (proposals)  (2D det)  (3D corners)
    """

    def __init__(self, num_classes=37, pretrained=True, model="v3l"):
        super().__init__()
        # Create the base Faster R-CNN detector with DINOv2 backbone + FPN
        self.base_detector = create_model(num_classes=num_classes, pretrained=pretrained, use_fpn=True, model=model)

        # Extract Faster R-CNN components for explicit forward control
        self.backbone = self.base_detector.backbone  # DINOv2 + SimpleFPN
        self.rpn = self.base_detector.rpn             # Region Proposal Network
        self.roi_heads = self.base_detector.roi_heads  # Box classification + regression
        
        # Transform: batching ONLY — no resize, no normalization.
        # The dataset already produces normalized images at Pi3-compatible dims.
        _ps = self.backbone.base_backbone.patch_size if hasattr(self.backbone, 'base_backbone') else 14
        self.transform = _NoOpRCNNTransform(
            min_size=800,   # ignored (resize is a no-op)
            max_size=1333,  # ignored (resize is a no-op)
            image_mean=[0.0, 0.0, 0.0],  # no-op (normalization done in dataset)
            image_std=[1.0, 1.0, 1.0],   # no-op (normalization done in dataset)
            size_divisible=_ps,
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
        """
        Full forward pass: 2D detection + 3D box regression.

        Training: returns dict of losses (cls, box, rpn, objectness, 3d)
        Inference: returns list of detection dicts with boxes, labels, scores, boxes_3d
        """
        # ---- Stage 1: Transform (batching only — normalize + resize are no-ops) ----
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = list(images.unbind(0))  # GeneralizedRCNNTransform expects a list

        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)  # Pads + creates ImageList

        # ---- Stage 2: Backbone (frozen DINOv2 ViT → FPN multi-scale features) ----
        features = self.backbone(images.tensors)  # Returns dict {p2, p3, p4, p5, p6}

        # ---- Stage 3: RPN (generate ~1000 object proposals per image) ----
        if isinstance(features, torch.Tensor):
            features = {"0": features}  # Wrap for non-FPN case

        proposals, proposal_losses = self.rpn(images, features, targets)

        # ---- Stage 4: ROI Heads (2D classification + box regression) ----
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # ---- Stage 5: 3D Head (monocular 3D bounding box regression) ----
        losses_3d = {}

        if self.training:
            # --- Training: match RPN proposals to GT for 3D supervision ---
            # Each target dict contains:
            #   'boxes' (N_g, 4)          - GT 2D bounding boxes
            #   'boxes_3d' (N_g, 8, 3)    - GT 3D corners in world coordinates
            #   'focal_lengths' (2,)       - camera fx, fy
            #   'principal_point' (2,)     - camera cx, cy
            
            final_proposals_3d = []
            final_gt_3d = []
            final_gt_2d_for_context = []
            final_intrinsics = []  # (N, 4) -> [fx, fy, cx, cy]
            
            for i in range(len(proposals)):
                prop = proposals[i]  # (N_p, 4)
                gt_box = targets[i]['boxes']  # (N_g, 4)
                gt_3d = targets[i]['boxes_3d']  # (N_g, 8, 3)
                
                # Compute IoU between all proposals and GT boxes for matching
                ious = torchvision.ops.box_iou(prop, gt_box)  # (N_p, N_g) IoU matrix
                val, idx = ious.max(dim=1)  # Best GT match for each proposal
                
                # Keep proposals with IoU >= 0.3 (lower threshold so 3D loss kicks in early)
                mask = val >= 0.3
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
                # Pool FPN features at 14×14 for matched proposals
                box_features = self.roi_pooler_3d(features, final_proposals_3d, images.image_sizes)
                
                # Concatenate across all images in the batch
                cat_bboxes = torch.cat(final_gt_2d_for_context, dim=0)     # (Total_K, 4)
                cat_gt_3d = torch.cat(final_gt_3d, dim=0)                  # (Total_K, 8, 3)
                cat_intrinsics = torch.cat(final_intrinsics, dim=0)        # (Total_K, 4)
                
                # Predict 3D corners + uncertainty from ROI features
                pred_3d, pred_mu = self.head_3d(box_features, cat_bboxes, cat_intrinsics)
                
                # OVMono3D disentangled loss (geometry-level attribute supervision)
                loss_3d, _l3d, _chamfer = ovmono3d_loss(
                    pred_3d.view(-1, 24), cat_gt_3d, pred_mu, use_smooth_l1=True
                )
                losses_3d = {'loss_3d': loss_3d}
            else:
                # No valid matches — return zero loss (still requires grad for backward)
                losses_3d = {'loss_3d': torch.tensor(0.0, device=features['p2'].device, requires_grad=True)}

        else:
            # --- Inference: run 3D head on detected boxes ---
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

        # ---- Combine all losses (RPN + 2D detector + 3D head) ----
        losses = {}
        losses.update(proposal_losses)   # loss_objectness + loss_rpn_box_reg
        losses.update(detector_losses)   # loss_classifier + loss_box_reg
        losses.update(losses_3d)         # loss_3d

        if self.training:
            return losses
        else:
            return detections


def create_model(num_classes=37, pretrained=True, coco_model=False, use_fpn=True, model="v2"):
    """
    Factory function to build the full Faster R-CNN detector.

    Args:
        num_classes: Number of object classes (including background)
        pretrained: Load pretrained DINOv2 weights from HuggingFace
        use_fpn: Wrap backbone with SimpleFPN (recommended)
        model: Key from MODEL_REGISTRY ('v2', 'v2s', 'v2l', 'v3l', etc.)

    Returns:
        FasterRCNN model with DINOv2 backbone
    """
    # Create base backbone — model key selects from MODEL_REGISTRY
    print(f"  Creating backbone: model={model}")
    base_backbone = Dinov3ModelBackbone(model=model)

    # Freeze backbone (keep adapter trainable)
    for name, params in base_backbone.named_parameters():
        if 'adapter' not in name:
            params.requires_grad_(False)

    # Wrap with SimpleFeaturePyramid
    if use_fpn:
        backbone = SimpleFeaturePyramid(
            in_channels=base_backbone.out_channels,  # auto-detected from backbone config
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),  # Creates p2, p3, p4, p5
            top_block=LastLevelMaxPool(),  # Adds p6
            norm="BN",
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
