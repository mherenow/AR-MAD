"""Local Patch Classifier for fine-grained detection.

This module performs patch-level classification to identify which regions of an
image are likely to be ML-generated, enabling fine-grained spatial analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Literal


class LocalPatchClassifier(nn.Module):
    """Patch-level classifier for fine-grained detection.
    
    Divides feature maps into patches and classifies each patch independently,
    then aggregates predictions for a final decision. Can optionally return
    a spatial heatmap showing which patches are classified as ML-generated.
    
    Args:
        feature_dim: Input feature dimension per spatial location
        patch_size: Size of patches in feature map (default: 8)
        num_classes: Number of output classes (default: 1 for binary)
        aggregation: Aggregation method ('average' or 'max', default: 'average')
        hidden_dim: Hidden dimension for classification head (default: 128)
        
    Example:
        >>> classifier = LocalPatchClassifier(feature_dim=256, patch_size=8)
        >>> features = torch.randn(4, 256, 32, 32)
        >>> prediction = classifier(features)
        >>> prediction.shape
        torch.Size([4, 1])
        
        >>> # With heatmap
        >>> prediction, heatmap = classifier(features, return_heatmap=True)
        >>> heatmap.shape
        torch.Size([4, 4, 4])  # 32/8 = 4 patches per dimension
    """
    
    def __init__(
        self,
        feature_dim: int,
        patch_size: int = 8,
        num_classes: int = 1,
        aggregation: Literal['average', 'max'] = 'average',
        hidden_dim: int = 128
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.aggregation = aggregation
        self.hidden_dim = hidden_dim
        
        # Per-patch classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        return_heatmap: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Classify patches and aggregate predictions.
        
        Args:
            features: Feature map (B, C, H, W)
            return_heatmap: Whether to return patch-level predictions
            
        Returns:
            prediction: Aggregated prediction (B, num_classes)
            heatmap: Patch-level predictions (B, H_patches, W_patches) if return_heatmap=True
        """
        batch_size, channels, height, width = features.size()
        
        # Apply per-patch classification
        # (B, C, H, W) -> (B, num_classes, H, W)
        patch_logits = self.classifier(features)
        
        # Divide into patches using average pooling
        # (B, num_classes, H, W) -> (B, num_classes, H_patches, W_patches)
        patch_predictions = F.avg_pool2d(
            patch_logits,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Apply sigmoid for binary classification
        patch_predictions = torch.sigmoid(patch_predictions)
        
        # Aggregate patch predictions
        if self.aggregation == 'average':
            # Average pooling across spatial dimensions
            aggregated = F.adaptive_avg_pool2d(patch_predictions, 1)
        elif self.aggregation == 'max':
            # Max pooling across spatial dimensions
            aggregated = F.adaptive_max_pool2d(patch_predictions, 1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Reshape to (B, num_classes)
        prediction = aggregated.view(batch_size, self.num_classes)
        
        if return_heatmap:
            # Return both aggregated prediction and spatial heatmap
            # Squeeze class dimension for binary classification
            heatmap = patch_predictions.squeeze(1)  # (B, H_patches, W_patches)
            return prediction, heatmap
        else:
            return prediction
    
    def get_patch_grid_size(self, feature_height: int, feature_width: int) -> Tuple[int, int]:
        """Calculate the number of patches in each dimension.
        
        Args:
            feature_height: Height of feature map
            feature_width: Width of feature map
            
        Returns:
            (num_patches_h, num_patches_w): Number of patches in each dimension
        """
        num_patches_h = feature_height // self.patch_size
        num_patches_w = feature_width // self.patch_size
        return num_patches_h, num_patches_w
