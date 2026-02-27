"""
Binary classifier model for AI-generated image detection.

This module provides the complete binary classification architecture combining
backbone feature extractors with a classification head for detecting AI-generated images.
"""

import torch
import torch.nn as nn
from typing import Literal
from .backbones import SimpleCNN, get_resnet18, get_resnet50


class ClassificationHead(nn.Module):
    """
    Classification head for binary AI-generated image detection.
    
    Architecture:
        Linear(feature_dim, 256) → ReLU → Dropout(0.5) → Linear(256, 1) → Sigmoid
    
    Attributes:
        classifier: Sequential neural network module containing the classification layers
    """
    
    def __init__(self, feature_dim: int):
        """
        Initialize the classification head.
        
        Args:
            feature_dim: Dimension of input features from the backbone
        """
        super(ClassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, feature_dim)
        
        Returns:
            torch.Tensor: Classification probabilities of shape (batch_size, 1)
        """
        return self.classifier(x)


class BinaryClassifier(nn.Module):
    """
    Complete binary classifier for AI-generated image detection.
    
    Combines a backbone feature extractor with a classification head to produce
    binary predictions (real vs AI-generated).
    
    Supported backbones:
        - 'simple_cnn': Custom 4-layer CNN (SimpleCNN)
        - 'resnet18': ResNet-18 architecture
        - 'resnet50': ResNet-50 architecture
    
    Attributes:
        backbone_type: Type of backbone architecture being used
        backbone: Feature extraction backbone network
        global_pool: Adaptive average pooling layer
        classifier: Classification head for binary prediction
    """
    
    def __init__(
        self,
        backbone_type: Literal['simple_cnn', 'resnet18', 'resnet50'] = 'simple_cnn',
        pretrained: bool = True
    ):
        """
        Initialize the binary classifier.
        
        Args:
            backbone_type: Type of backbone architecture to use ('simple_cnn', 'resnet18', or 'resnet50')
            pretrained: Whether to use pretrained weights (only applies to ResNet models)
            
        Raises:
            ValueError: If backbone_type is not one of the supported architectures
        """
        super(BinaryClassifier, self).__init__()
        
        self.backbone_type = backbone_type
        
        # Initialize backbone based on type
        if backbone_type == 'simple_cnn':
            self.backbone = SimpleCNN(input_channels=3)
            feature_dim = 512  # SimpleCNN outputs 512 channels
        elif backbone_type == 'resnet18':
            self.backbone = get_resnet18(pretrained=pretrained, freeze_backbone=False)
            feature_dim = 512  # ResNet18 outputs 512 channels
        elif backbone_type == 'resnet50':
            self.backbone = get_resnet50(pretrained=pretrained, freeze_backbone=False)
            feature_dim = 2048  # ResNet50 outputs 2048 channels
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Global average pooling to convert spatial features to vectors
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = ClassificationHead(feature_dim)
        
        # TODO: Add spectral/frequency-domain feature extraction
        # Implement FFT-based spectral analysis to detect frequency domain artifacts
        # common in AI-generated images (e.g., checkerboard patterns from upsampling,
        # periodic artifacts from convolution operations). Consider adding a parallel
        # branch that processes DCT/FFT coefficients alongside spatial features.
        
        # TODO: Add noise-based imprint detection
        # Implement noise residual extraction (e.g., using high-pass filters or
        # denoising networks) to capture generator-specific noise patterns and
        # fingerprints. Different GANs/diffusion models leave distinct noise signatures
        # that can be used for detection and attribution.
        
        # TODO: Add multi-dataset support
        # Implement domain adaptation or multi-task learning to handle images from
        # different datasets with varying characteristics (resolution, compression,
        # color spaces). Consider adding dataset-specific normalization or
        # domain-adversarial training to improve generalization across datasets.
        
        # TODO: Add any-resolution processing capability
        # Current architecture assumes fixed 256x256 input. Implement adaptive pooling
        # or fully convolutional approach to handle arbitrary input resolutions without
        # resizing. This preserves original image details and avoids artifacts from
        # interpolation that could affect detection accuracy.
        
        # TODO: Add attention mechanism
        # Consider implementing spatial attention (e.g., CBAM, SE-Net) to focus on
        # discriminative regions that contain AI generation artifacts. Attention can
        # help the model focus on subtle inconsistencies in textures, edges, or patterns.
        
        # TODO: Add multi-scale feature fusion
        # Implement feature pyramid or multi-scale processing to capture artifacts at
        # different resolutions. AI-generated images may have inconsistencies across
        # scales (e.g., fine details vs. global structure). Consider FPN-style fusion
        # or extracting features from multiple backbone layers.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete classifier.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 256, 256)
        
        Returns:
            Probabilities of shape (batch_size, 1) where values close to 1
            indicate AI-generated and values close to 0 indicate real images
        """
        # Extract features using backbone
        features = self.backbone(x)  # (B, C, H, W)
        
        # Global average pooling to get fixed-size feature vectors
        pooled = self.global_pool(features)  # (B, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        
        # Classification
        output = self.classifier(pooled)  # (B, 1)
        
        return output
