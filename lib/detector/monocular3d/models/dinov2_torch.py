from typing import Dict, Union, List
import sys
import os
import torch
import torchvision
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from huggingface_hub import login

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(_hf_token)


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
                layers.extend([nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False),])
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

    def __init__(self, model_name='facebook/dinov3-vitl16-pretrain-lvd1689m'):  # facebook/dinov3-vitb16-pretrain-lvd1689m , facebook/dinov2-base
        super().__init__()  #facebook/dinov3-vith16plus-pretrain-lvd1689m
        self.model_name = model_name
        self.bck_model = AutoModel.from_pretrained(self.model_name)
        self.bck_model.eval()
        self.out_channels = self.bck_model.config.hidden_size

    def forward(self, x):
        outputs = self.bck_model(x)
        features = outputs.last_hidden_state

        # Remove the CLS token (first token) and reshape to spatial dimensions
        features = features[:, 1:, :]
        B, N, C = features.shape

        H = W = int(N ** 0.5)

        if H * W != N:
            target_size = H * W
            if N > target_size:
                features = features[:, :target_size, :]
            else:
                padding = target_size - N
                features = torch.cat([features, torch.zeros(B, padding, C, device=features.device)], dim=1)
        features = features.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return features


def create_model(num_classes=37, pretrained=True, coco_model=False, use_fpn=True, model="v2"):
    # Create base backbone
    base_backbone = Dinov3ModelBackbone()
    base_backbone.out_channels = 768

    #Freeze backbone (keep adapter trainable)
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

        # Create a wrapper that combines backbone + FPN
        class BackboneWithFPN(nn.Module):
            def __init__(self, base_backbone, fpn):
                super().__init__()
                self.base_backbone = base_backbone
                self.fpn = fpn
                self.out_channels = fpn.out_channels

            def forward(self, x):
                # Get features from frozen backbone
                with torch.no_grad():
                    features = self.base_backbone(x)
                # Apply FPN
                return self.fpn(features)

        backbone = BackboneWithFPN(base_backbone, backbone)
        print("✅ Using SimpleFeaturePyramid (FPN)")
    else:
        backbone = base_backbone
        backbone.out_channels = 768
        print("Using backbone without FPN")

    # Anchor generator for FPN features (p2-p6)
    if use_fpn:
        featmap_names = backbone.fpn._out_features  # ['p2', 'p3', 'p4', 'p5', 'p6']
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
