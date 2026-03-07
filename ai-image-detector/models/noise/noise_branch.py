"""
Noise imprint branch for feature extraction from noise residuals.

This module processes noise residuals through a CNN to extract generator-specific
features that can be used for detection and optional attribution.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class NoiseImprintBranch(nn.Module):
    """
    CNN branch for noise imprint feature extraction.
    
    This module processes noise residuals through a 4-layer CNN to extract
    features that capture generator-specific noise patterns. It can optionally
    include an attribution head that predicts which generator created the image.
    
    Architecture:
        - Conv1: 3 → 64 channels, 3×3 kernel, stride 1, padding 1
        - Conv2: 64 → 128 channels, 3×3 kernel, stride 2, padding 1
        - Conv3: 128 → 256 channels, 3×3 kernel, stride 2, padding 1
        - Conv4: 256 → 256 channels, 3×3 kernel, stride 2, padding 1
        - Global average pooling
        - FC layer to feature_dim
        - Optional attribution head: feature_dim → num_generators
    
    Args:
        input_channels: Number of input channels (default: 3 for RGB residuals)
        feature_dim: Output feature dimension (default: 256)
        enable_attribution: Whether to predict generator type (default: False)
        num_generators: Number of generator classes for attribution (default: 10)
        
    Example:
        >>> branch = NoiseImprintBranch(feature_dim=256, enable_attribution=True, num_generators=5)
        >>> residual = torch.randn(2, 3, 256, 256)
        >>> features, attribution = branch(residual)
        >>> features.shape
        torch.Size([2, 256])
        >>> attribution.shape
        torch.Size([2, 5])
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 256,
        enable_attribution: bool = False,
        num_generators: int = 10
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.enable_attribution = enable_attribution
        self.num_generators = num_generators
        
        # 4-layer CNN with increasing channels
        # Conv1: 3 → 64, stride 1 (maintains spatial resolution)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Conv2: 64 → 128, stride 2 (downsamples by 2x)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Conv3: 128 → 256, stride 2 (downsamples by 2x)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Conv4: 256 → 256, stride 2 (downsamples by 2x)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # FC layer to project to feature_dim
        self.fc = nn.Linear(256, feature_dim)
        
        # Optional attribution head
        if enable_attribution:
            self.attribution_head = nn.Sequential(
                nn.Linear(feature_dim, num_generators),
                nn.Softmax(dim=1)
            )
        else:
            self.attribution_head = None
    
    def forward(
        self, 
        residual: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through noise imprint branch.
        
        Args:
            residual: Noise residual tensor (B, C, H, W) in range typically [-1, 1]
        
        Returns:
            If enable_attribution is False:
                features: Noise imprint features (B, feature_dim)
            If enable_attribution is True:
                features: Noise imprint features (B, feature_dim)
                attribution: Generator probabilities (B, num_generators)
        """
        # Pass through CNN layers
        x = self.conv1(residual)  # (B, 64, H, W)
        x = self.conv2(x)          # (B, 128, H/2, W/2)
        x = self.conv3(x)          # (B, 256, H/4, W/4)
        x = self.conv4(x)          # (B, 256, H/8, W/8)
        
        # Global average pooling
        x = self.global_pool(x)    # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        # Project to feature_dim
        features = self.fc(x)      # (B, feature_dim)
        
        # Optional attribution
        if self.enable_attribution:
            attribution = self.attribution_head(features)  # (B, num_generators)
            return features, attribution
        
        return features
