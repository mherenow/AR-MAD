"""
Binary classifier model for AI-generated image detection.

This module provides the complete binary classification architecture combining
backbone feature extractors with a classification head for detecting AI-generated images.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Union, Tuple, List
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
        pretrained: bool = True,
        use_spectral: bool = False,
        use_noise_imprint: bool = False,
        use_color_features: bool = False,
        use_local_patches: bool = False,
        use_fpn: bool = False,
        use_attention: Literal['cbam', 'se', None] = None,
        enable_attribution: bool = False
    ):
        """
        Initialize the binary classifier.
        
        Args:
            backbone_type: Type of backbone architecture to use ('simple_cnn', 'resnet18', or 'resnet50')
            pretrained: Whether to use pretrained weights (only applies to ResNet models)
            use_spectral: Enable spectral branch for frequency domain analysis (default: False)
            use_noise_imprint: Enable noise imprint branch for generator fingerprinting (default: False)
            use_color_features: Enable chrominance feature extraction (default: False)
            use_local_patches: Enable local patch classifier for fine-grained detection (default: False)
            use_fpn: Enable feature pyramid fusion for multi-scale features (default: False)
            use_attention: Attention mechanism type ('cbam', 'se', or None) (default: None)
            enable_attribution: Enable generator attribution head (requires use_noise_imprint=True) (default: False)
            
        Raises:
            ValueError: If backbone_type is not one of the supported architectures
        """
        super(BinaryClassifier, self).__init__()
        
        self.backbone_type = backbone_type
        
        # Store feature flags as instance variables
        self.use_spectral = use_spectral
        self.use_noise_imprint = use_noise_imprint
        self.use_color_features = use_color_features
        self.use_local_patches = use_local_patches
        self.use_fpn = use_fpn
        self.use_attention = use_attention
        self.enable_attribution = enable_attribution
        
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
        
        # Conditional module instantiation based on feature flags
        # These modules are only created when their corresponding flags are True
        
        # Spectral branch for frequency domain analysis (optional)
        self.spectral_branch = None
        spectral_feature_dim = 0
        if use_spectral:
            from .spectral.spectral_branch import SpectralBranch
            self.spectral_branch = SpectralBranch(
                patch_size=16,
                embed_dim=256,
                depth=4,
                num_heads=8,
                mask_type='high_pass',
                cutoff_freq=0.3,
                num_bands=4,
                consistency_dim=128
            )
            # SpectralBranch outputs SRS (256) + SCV (128) = 384 dimensions
            spectral_feature_dim = 256 + 128
        
        # Noise imprint branch for generator fingerprinting (optional)
        self.noise_extractor = None
        self.noise_branch = None
        noise_feature_dim = 0
        if use_noise_imprint:
            from .noise.residual_extractor import NoiseResidualExtractor
            from .noise.noise_branch import NoiseImprintBranch
            self.noise_extractor = NoiseResidualExtractor(
                method='diffusion',
                diffusion_steps=50,
                gaussian_sigma=2.0
            )
            self.noise_branch = NoiseImprintBranch(
                input_channels=3,
                feature_dim=256,
                enable_attribution=enable_attribution,
                num_generators=10
            )
            noise_feature_dim = 256
        
        # Chrominance branch for color feature extraction (optional)
        self.rgb_to_ycbcr = None
        self.chrominance_branch = None
        color_feature_dim = 0
        if use_color_features:
            from .color.color_space import RGBtoYCbCr
            from .color.chrominance_branch import ChrominanceBranch
            self.rgb_to_ycbcr = RGBtoYCbCr()
            self.chrominance_branch = ChrominanceBranch(
                num_bins=64,
                feature_dim=256
            )
            color_feature_dim = 256
        
        # Attention mechanism (optional)
        self.attention_module = None
        if use_attention is not None:
            if use_attention == 'cbam':
                from .attention.cbam import CBAM
                self.attention_module = CBAM(
                    channels=feature_dim,
                    reduction_ratio=16,
                    kernel_size=7
                )
            elif use_attention == 'se':
                from .attention.se_block import SEBlock
                self.attention_module = SEBlock(
                    channels=feature_dim,
                    reduction=16
                )
        
        # Feature Pyramid Network for multi-scale fusion (optional)
        self.fpn = None
        fpn_feature_dim = 0
        if use_fpn:
            from .fusion.fpn import FeaturePyramidFusion
            # Extract multi-scale features from backbone
            # For ResNet: we'll use features from different stages
            # For SimpleCNN: we'll extract from intermediate layers
            if backbone_type in ['resnet18', 'resnet50']:
                # ResNet has 4 stages with channels [64, 128, 256, 512] for ResNet18
                # or [256, 512, 1024, 2048] for ResNet50
                if backbone_type == 'resnet18':
                    in_channels_list = [128, 256, 512]  # Use last 3 stages
                else:  # resnet50
                    in_channels_list = [512, 1024, 2048]  # Use last 3 stages
            else:  # simple_cnn
                # SimpleCNN has 4 conv layers: 64, 128, 256, 512
                in_channels_list = [128, 256, 512]  # Use last 3 layers
            
            self.fpn = FeaturePyramidFusion(
                in_channels_list=in_channels_list,
                out_channels=256
            )
            fpn_feature_dim = 256
        
        # Local patch classifier for fine-grained detection (optional)
        self.local_patch_classifier = None
        if use_local_patches:
            from .attention.local_patch_classifier import LocalPatchClassifier
            self.local_patch_classifier = LocalPatchClassifier(
                feature_dim=feature_dim,
                patch_size=8,
                num_classes=1
            )
        
        # Calculate total feature dimension for fusion
        # Start with backbone feature dimension
        total_feature_dim = feature_dim
        
        # Add dimensions from optional branches
        total_feature_dim += spectral_feature_dim + noise_feature_dim + color_feature_dim + fpn_feature_dim
        
        # Fusion layer for combining multiple feature sources (only needed if multiple features enabled)
        self.fusion_layer = None
        if sum([use_spectral, use_noise_imprint, use_color_features, use_fpn]) > 0:
            # This layer will project concatenated features to a unified dimension
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        
        # Classification head
        self.classifier = ClassificationHead(feature_dim)
    
    def _extract_multi_scale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from backbone for FPN.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            List of feature maps at different scales (low-res to high-res)
        """
        features = []
        
        if self.backbone_type in ['resnet18', 'resnet50']:
            # For ResNet, the backbone is a Sequential container
            # We need to access the internal layers through indexing
            # Sequential structure: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
            
            # Initial layers
            x = self.backbone[0](x)  # conv1
            x = self.backbone[1](x)  # bn1
            x = self.backbone[2](x)  # relu
            x = self.backbone[3](x)  # maxpool
            
            # Layer 1
            x = self.backbone[4](x)  # layer1
            
            # Layer 2
            x = self.backbone[5](x)  # layer2
            features.append(x)  # First scale
            
            # Layer 3
            x = self.backbone[6](x)  # layer3
            features.append(x)  # Second scale
            
            # Layer 4
            x = self.backbone[7](x)  # layer4
            features.append(x)  # Third scale
        else:
            # For SimpleCNN, extract from intermediate conv layers
            # SimpleCNN has 4 conv blocks
            x = self.backbone.conv1(x)
            
            x = self.backbone.conv2(x)
            features.append(x)  # First scale (128 channels)
            
            x = self.backbone.conv3(x)
            features.append(x)  # Second scale (256 channels)
            
            x = self.backbone.conv4(x)
            features.append(x)  # Third scale (512 channels)
        
        return features
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the complete classifier.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Probabilities of shape (batch_size, 1) where values close to 1
            indicate AI-generated and values close to 0 indicate real images.
            If enable_attribution is True and use_noise_imprint is True,
            returns a tuple of (prediction, attribution).
        """
        # List to collect features from all enabled branches
        features_list = []
        attribution = None
        
        # 1. Spatial features from backbone (always enabled)
        spatial_features = self.backbone(x)  # (B, C, H, W)
        
        # Apply attention to spatial features if enabled
        if self.attention_module is not None:
            spatial_features = self.attention_module(spatial_features)
        
        # 2. Spectral features (optional)
        if self.spectral_branch is not None:
            srs, scv = self.spectral_branch(x)  # (B, 256), (B, 128)
            features_list.extend([srs, scv])
        
        # 3. Noise imprint features (optional)
        if self.noise_branch is not None:
            residual = self.noise_extractor(x)  # (B, 3, H, W)
            noise_output = self.noise_branch(residual)
            if self.enable_attribution:
                noise_features, attribution = noise_output  # (B, 256), (B, num_generators)
            else:
                noise_features = noise_output  # (B, 256)
            features_list.append(noise_features)
        
        # 4. Chrominance features (optional)
        if self.chrominance_branch is not None:
            ycbcr = self.rgb_to_ycbcr(x * 255.0)  # Convert to [0, 255] range for YCbCr
            color_features = self.chrominance_branch(ycbcr)  # (B, 256)
            features_list.append(color_features)
        
        # 5. Multi-scale features from FPN (optional)
        if self.fpn is not None:
            multi_scale_features = self._extract_multi_scale_features(x)
            fused_fpn_features = self.fpn(multi_scale_features)  # (B, 256, H', W')
            # Global pool FPN features to get a vector
            fpn_pooled = self.global_pool(fused_fpn_features)  # (B, 256, 1, 1)
            fpn_pooled = fpn_pooled.view(fpn_pooled.size(0), -1)  # (B, 256)
            features_list.append(fpn_pooled)
        
        # 6. Global pooling of spatial features
        pooled_spatial = self.global_pool(spatial_features)  # (B, C, 1, 1)
        pooled_spatial = pooled_spatial.view(pooled_spatial.size(0), -1)  # (B, C)
        
        # 7. Feature fusion (if multiple features are enabled)
        if len(features_list) > 0 and self.fusion_layer is not None:
            # Concatenate spatial features with all optional features
            combined_features = torch.cat([pooled_spatial] + features_list, dim=1)
            # Apply fusion layer to project to unified dimension
            features = self.fusion_layer(combined_features)
        else:
            # Backward compatibility: when all flags are False, use only spatial features
            features = pooled_spatial
        
        # 8. Classification
        prediction = self.classifier(features)  # (B, 1)
        
        # Return prediction with optional attribution
        if self.enable_attribution and attribution is not None:
            return prediction, attribution
        return prediction
    
    def get_feature_flags(self) -> dict:
        """
        Get the current feature flag configuration.
        
        This method is used for checkpoint compatibility to ensure that
        saved models can be loaded with the correct feature configuration.
        
        Returns:
            Dictionary containing all feature flags and their current values
        """
        return {
            'use_spectral': self.use_spectral,
            'use_noise_imprint': self.use_noise_imprint,
            'use_color_features': self.use_color_features,
            'use_local_patches': self.use_local_patches,
            'use_fpn': self.use_fpn,
            'use_attention': self.use_attention,
            'enable_attribution': self.enable_attribution
        }
