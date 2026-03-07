"""Squeeze-and-Excitation (SE) Block implementation.

SE blocks recalibrate channel-wise feature responses by explicitly modeling
interdependencies between channels.

Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration.
    
    The SE block adaptively recalibrates channel-wise feature responses by
    explicitly modeling channel interdependencies through a squeeze-and-excitation
    mechanism:
    1. Squeeze: Global average pooling to aggregate spatial information
    2. Excitation: Two FC layers to capture channel-wise dependencies
    3. Scale: Multiply input by learned channel weights
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for the bottleneck (default: 16)
        
    Example:
        >>> se = SEBlock(channels=256, reduction=16)
        >>> x = torch.randn(4, 256, 32, 32)
        >>> out = se(x)
        >>> out.shape
        torch.Size([4, 256, 32, 32])
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Ensure reduction doesn't make bottleneck too small
        reduced_channels = max(channels // reduction, 1)
        
        # Excitation network: squeeze -> FC -> ReLU -> FC -> Sigmoid
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Channel-recalibrated features (B, C, H, W)
        """
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global average pooling
        # (B, C, H, W) -> (B, C)
        squeezed = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excited = self.fc1(squeezed)
        excited = F.relu(excited, inplace=True)
        excited = self.fc2(excited)
        excited = torch.sigmoid(excited)
        
        # Reshape to (B, C, 1, 1) for broadcasting
        excited = excited.view(batch_size, channels, 1, 1)
        
        # Scale: Multiply input by channel weights
        return x * excited
