"""Feature Pyramid Network (FPN) for multi-scale feature fusion.

FPN builds a feature pyramid with top-down pathway and lateral connections
to combine features from multiple scales effectively.

Reference: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FeaturePyramidFusion(nn.Module):
    """Feature Pyramid Network for multi-scale fusion.
    
    Combines features from multiple scales using a top-down pathway with
    lateral connections. Each scale is processed through:
    1. Lateral connection: 1x1 conv to match channel dimensions
    2. Top-down pathway: Upsample higher-level features and add
    3. Output convolution: 3x3 conv to reduce aliasing
    
    The final output concatenates all scales and projects to output channels.
    
    Args:
        in_channels_list: List of input channel dimensions for each scale
                         (from low-resolution to high-resolution)
        out_channels: Output channel dimension (default: 256)
        
    Example:
        >>> # Features from 3 scales: [low-res, mid-res, high-res]
        >>> fpn = FeaturePyramidFusion(
        ...     in_channels_list=[512, 256, 128],
        ...     out_channels=256
        ... )
        >>> features = [
        ...     torch.randn(4, 512, 8, 8),   # Low resolution
        ...     torch.randn(4, 256, 16, 16), # Mid resolution
        ...     torch.randn(4, 128, 32, 32)  # High resolution
        ... ]
        >>> fused = fpn(features)
        >>> fused.shape
        torch.Size([4, 256, 32, 32])
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_scales = len(in_channels_list)
        
        # Lateral connections: 1x1 convs to match channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Output convolutions: 3x3 convs to reduce aliasing
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        # Final fusion layer to combine all scales
        self.fusion_conv = nn.Conv2d(
            out_channels * self.num_scales,
            out_channels,
            kernel_size=1
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Apply FPN fusion to multi-scale features.
        
        Args:
            features: List of feature maps at different scales
                     [(B, C1, H1, W1), (B, C2, H2, W2), ...]
                     Ordered from low-resolution to high-resolution
                     
        Returns:
            fused: Fused multi-scale features (B, out_channels, H_max, W_max)
                  where H_max, W_max are the dimensions of the highest resolution
        """
        if len(features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature maps, got {len(features)}"
            )
        
        # Apply lateral connections
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]
        
        # Build top-down pathway
        # Start from the lowest resolution (highest semantic level)
        for i in range(self.num_scales - 1, 0, -1):
            # Upsample higher-level features to match lower-level spatial dimensions
            h, w = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[i],
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            # Add to lower-level features
            laterals[i - 1] = laterals[i - 1] + upsampled
        
        # Apply output convolutions to reduce aliasing
        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]
        
        # Upsample all scales to the highest resolution
        target_h, target_w = outputs[-1].shape[2:]  # Highest resolution
        upsampled_outputs = []
        
        for output in outputs:
            if output.shape[2:] != (target_h, target_w):
                upsampled = F.interpolate(
                    output,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                upsampled_outputs.append(upsampled)
            else:
                upsampled_outputs.append(output)
        
        # Concatenate all scales
        concatenated = torch.cat(upsampled_outputs, dim=1)
        
        # Final fusion
        fused = self.fusion_conv(concatenated)
        
        return fused
