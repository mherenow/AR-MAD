"""
Backbone architectures for AI-generated image detection.

This module provides various CNN architectures that can be used as feature extractors
for detecting AI-generated images.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class SimpleCNN(nn.Module):
    """
    Simple CNN with 4 convolutional layers for baseline feature extraction.
    
    Architecture:
        - Conv1: 3 -> 64 channels
        - Conv2: 64 -> 128 channels
        - Conv3: 128 -> 256 channels
        - Conv4: 256 -> 512 channels
    
    Each conv block includes: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    """
    
    def __init__(self, input_channels: int = 3):
        """
        Initialize SimpleCNN.
        
        Args:
            input_channels: Number of input channels (default: 3 for RGB images)
        """
        super(SimpleCNN, self).__init__()
        
        # Conv Block 1: 3 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 2: 64 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 3: 128 -> 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 4: 256 -> 512
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # TODO: Add spectral feature extraction layers
        # Consider adding FFT-based spectral analysis to detect frequency domain artifacts
        # that are common in AI-generated images (e.g., checkerboard patterns, upsampling artifacts)
        
        # TODO: Add noise-based imprint detection
        # Implement noise residual extraction (e.g., using high-pass filters or denoising)
        # to capture generator-specific noise patterns and fingerprints
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, 512, H/16, W/16)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def get_resnet18(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Get ResNet18 backbone from torchvision.
    
    Args:
        pretrained: Whether to load ImageNet pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone parameters (default: False)
        
    Returns:
        ResNet18 model without the final classification layer
    """
    # Load ResNet18 with optional pretrained weights
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Remove the final fully connected layer
    model = nn.Sequential(*list(model.children())[:-1])
    
    # Optionally freeze backbone parameters
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # TODO: Consider adding spectral feature branches
    # ResNet features could be augmented with frequency domain analysis
    # to better capture AI generation artifacts
    
    return model


def get_resnet50(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Get ResNet50 backbone from torchvision.
    
    Args:
        pretrained: Whether to load ImageNet pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone parameters (default: False)
        
    Returns:
        ResNet50 model without the final classification layer
    """
    # Load ResNet50 with optional pretrained weights
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Remove the final fully connected layer
    model = nn.Sequential(*list(model.children())[:-1])
    
    # Optionally freeze backbone parameters
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # TODO: Add noise pattern extraction module
    # Implement a parallel branch that extracts and analyzes noise residuals
    # to detect generator-specific fingerprints and artifacts
    
    return model
