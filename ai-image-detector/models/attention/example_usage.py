"""Example usage of attention mechanisms and multi-scale fusion.

This script demonstrates how to use CBAM, SEBlock, LocalPatchClassifier,
and FeaturePyramidFusion modules in a typical workflow.
"""

import torch
import torch.nn as nn

# Handle both direct execution and module import
try:
    from .cbam import CBAM
    from .se_block import SEBlock
    from .local_patch_classifier import LocalPatchClassifier
    from ..fusion.fpn import FeaturePyramidFusion
except ImportError:
    from cbam import CBAM
    from se_block import SEBlock
    from local_patch_classifier import LocalPatchClassifier
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fusion.fpn import FeaturePyramidFusion


def example_cbam():
    """Example: Using CBAM attention on feature maps."""
    print("=" * 60)
    print("Example 1: CBAM Attention")
    print("=" * 60)
    
    # Create CBAM module
    cbam = CBAM(channels=256, reduction_ratio=16, kernel_size=7)
    
    # Input feature map
    features = torch.randn(4, 256, 32, 32)
    print(f"Input shape: {features.shape}")
    
    # Apply CBAM
    attended_features = cbam(features)
    print(f"Output shape: {attended_features.shape}")
    print(f"CBAM applies both channel and spatial attention\n")


def example_se_block():
    """Example: Using SEBlock for channel recalibration."""
    print("=" * 60)
    print("Example 2: SEBlock Channel Recalibration")
    print("=" * 60)
    
    # Create SEBlock module
    se = SEBlock(channels=256, reduction=16)
    
    # Input feature map
    features = torch.randn(4, 256, 32, 32)
    print(f"Input shape: {features.shape}")
    
    # Apply SEBlock
    recalibrated_features = se(features)
    print(f"Output shape: {recalibrated_features.shape}")
    print(f"SEBlock recalibrates channel-wise responses\n")


def example_local_patch_classifier():
    """Example: Using LocalPatchClassifier for fine-grained detection."""
    print("=" * 60)
    print("Example 3: Local Patch Classification")
    print("=" * 60)
    
    # Create LocalPatchClassifier
    classifier = LocalPatchClassifier(
        feature_dim=256,
        patch_size=8,
        aggregation='average'
    )
    
    # Input feature map
    features = torch.randn(4, 256, 32, 32)
    print(f"Input shape: {features.shape}")
    
    # Get prediction without heatmap
    prediction = classifier(features)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction values: {prediction[0].item():.4f}")
    
    # Get prediction with heatmap
    prediction, heatmap = classifier(features, return_heatmap=True)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap shows patch-level predictions (4x4 patches)\n")


def example_fpn():
    """Example: Using FeaturePyramidFusion for multi-scale fusion."""
    print("=" * 60)
    print("Example 4: Feature Pyramid Fusion")
    print("=" * 60)
    
    # Create FPN module
    fpn = FeaturePyramidFusion(
        in_channels_list=[512, 256, 128],
        out_channels=256
    )
    
    # Multi-scale features (from low-res to high-res)
    features = [
        torch.randn(4, 512, 8, 8),   # Low resolution
        torch.randn(4, 256, 16, 16), # Mid resolution
        torch.randn(4, 128, 32, 32)  # High resolution
    ]
    
    print("Input features:")
    for i, feat in enumerate(features):
        print(f"  Scale {i}: {feat.shape}")
    
    # Apply FPN fusion
    fused = fpn(features)
    print(f"\nFused output shape: {fused.shape}")
    print(f"FPN combines multi-scale features into unified representation\n")


def example_combined_pipeline():
    """Example: Combining all modules in a detection pipeline."""
    print("=" * 60)
    print("Example 5: Combined Pipeline")
    print("=" * 60)
    
    # Simulate a backbone that produces multi-scale features
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
            
        def forward(self, x):
            # Extract multi-scale features
            f1 = self.conv1(x)  # High-res
            f2 = self.conv2(f1)  # Mid-res
            f3 = self.conv3(f2)  # Low-res
            return [f3, f2, f1]  # Return in FPN order (low to high)
    
    # Create pipeline components
    backbone = SimpleBackbone()
    fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
    attention = CBAM(channels=256, reduction_ratio=16, kernel_size=7)
    classifier = LocalPatchClassifier(feature_dim=256, patch_size=8)
    
    # Input image
    image = torch.randn(2, 3, 32, 32)
    print(f"Input image shape: {image.shape}")
    
    # Forward pass
    multi_scale_features = backbone(image)
    print(f"\nMulti-scale features extracted:")
    for i, feat in enumerate(multi_scale_features):
        print(f"  Scale {i}: {feat.shape}")
    
    fused_features = fpn(multi_scale_features)
    print(f"\nFused features: {fused_features.shape}")
    
    attended_features = attention(fused_features)
    print(f"After attention: {attended_features.shape}")
    
    prediction, heatmap = classifier(attended_features, return_heatmap=True)
    print(f"\nFinal prediction: {prediction.shape}")
    print(f"Spatial heatmap: {heatmap.shape}")
    print(f"Prediction values: {prediction[0].item():.4f}, {prediction[1].item():.4f}")
    print("\nPipeline: Backbone -> FPN -> Attention -> Classifier")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Attention Mechanisms and Multi-Scale Fusion Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    example_cbam()
    example_se_block()
    example_local_patch_classifier()
    example_fpn()
    example_combined_pipeline()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
