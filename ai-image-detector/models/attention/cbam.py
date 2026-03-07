"""Convolutional Block Attention Module (CBAM) implementation.

CBAM applies both channel attention and spatial attention sequentially to
refine feature maps for improved discriminative power.

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module using both average and max pooling.
    
    Args:
        channels: Number of input channels
        reduction_ratio: Channel reduction ratio for the MLP (default: 16)
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Shared MLP: channels -> channels/r -> channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Channel attention weights (B, C, 1, 1)
        """
        batch_size, channels, _, _ = x.size()
        
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        avg_out = self.mlp(avg_pool)
        
        # Global max pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, channels)
        max_out = self.mlp(max_pool)
        
        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out)
        
        return attention.view(batch_size, channels, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module using channel-wise pooling.
    
    Args:
        kernel_size: Convolution kernel size (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Convolution to generate spatial attention
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Spatial attention weights (B, 1, H, W)
        """
        # Channel-wise average pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Channel-wise max pooling
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution and sigmoid
        attention = torch.sigmoid(self.conv(pooled))
        
        return attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Applies channel attention followed by spatial attention to refine feature maps.
    
    Args:
        channels: Number of input channels
        reduction_ratio: Channel reduction ratio (default: 16)
        kernel_size: Spatial attention kernel size (default: 7)
        
    Example:
        >>> cbam = CBAM(channels=256, reduction_ratio=16, kernel_size=7)
        >>> x = torch.randn(4, 256, 32, 32)
        >>> out = cbam(x)
        >>> out.shape
        torch.Size([4, 256, 32, 32])
    """
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM attention.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Attention-weighted features (B, C, H, W)
        """
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights
        
        return x
