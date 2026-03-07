"""Color space conversion utilities for RGB to YCbCr and vice versa.

This module implements ITU-R BT.601 standard color space conversions.
"""

import torch
import torch.nn as nn
from typing import Optional


class RGBtoYCbCr(nn.Module):
    """Convert RGB images to YCbCr color space using ITU-R BT.601 standard.
    
    The conversion uses the following transformation:
        Y  =  0.299*R + 0.587*G + 0.114*B
        Cb = -0.169*R - 0.331*G + 0.500*B + 128
        Cr =  0.500*R - 0.419*G - 0.081*B + 128
    
    Args:
        None
    
    Example:
        >>> converter = RGBtoYCbCr()
        >>> rgb = torch.rand(1, 3, 224, 224) * 255  # RGB in [0, 255]
        >>> ycbcr = converter(rgb)
        >>> print(ycbcr.shape)  # (1, 3, 224, 224)
    """
    
    def __init__(self):
        super().__init__()
        
        # ITU-R BT.601 conversion matrix
        # Shape: (3, 3) for matrix multiplication
        self.register_buffer(
            'transform_matrix',
            torch.tensor([
                [ 0.299,  0.587,  0.114],
                [-0.169, -0.331,  0.500],
                [ 0.500, -0.419, -0.081]
            ], dtype=torch.float32)
        )
        
        # Offset for Cb and Cr channels
        self.register_buffer(
            'offset',
            torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32)
        )
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to YCbCr color space.
        
        Args:
            rgb: RGB image tensor of shape (B, 3, H, W) in range [0, 255]
        
        Returns:
            ycbcr: YCbCr image tensor of shape (B, 3, H, W)
                   Y in range [0, 255], Cb and Cr in range [0, 256]
        
        Raises:
            ValueError: If input tensor doesn't have 3 channels
        """
        if rgb.shape[1] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {rgb.shape[1]}")
        
        # Reshape for matrix multiplication: (B, 3, H, W) -> (B, H, W, 3)
        rgb_permuted = rgb.permute(0, 2, 3, 1)
        
        # Apply transformation: (B, H, W, 3) @ (3, 3)^T -> (B, H, W, 3)
        ycbcr = torch.matmul(rgb_permuted, self.transform_matrix.t())
        
        # Add offset for Cb and Cr channels
        ycbcr = ycbcr + self.offset
        
        # Reshape back: (B, H, W, 3) -> (B, 3, H, W)
        ycbcr = ycbcr.permute(0, 3, 1, 2)
        
        return ycbcr


class YCbCrtoRGB(nn.Module):
    """Convert YCbCr images to RGB color space using ITU-R BT.601 standard.
    
    The conversion uses the inverse transformation:
        R = Y + 1.402*(Cr - 128)
        G = Y - 0.344*(Cb - 128) - 0.714*(Cr - 128)
        B = Y + 1.772*(Cb - 128)
    
    Args:
        None
    
    Example:
        >>> converter = YCbCrtoRGB()
        >>> ycbcr = torch.rand(1, 3, 224, 224) * 255
        >>> rgb = converter(ycbcr)
        >>> print(rgb.shape)  # (1, 3, 224, 224)
    """
    
    def __init__(self):
        super().__init__()
        
        # Inverse transformation matrix (derived from ITU-R BT.601)
        self.register_buffer(
            'transform_matrix',
            torch.tensor([
                [1.000,  0.000,  1.402],
                [1.000, -0.344, -0.714],
                [1.000,  1.772,  0.000]
            ], dtype=torch.float32)
        )
        
        # Offset to subtract from Cb and Cr before transformation
        self.register_buffer(
            'offset',
            torch.tensor([0.0, -128.0, -128.0], dtype=torch.float32)
        )
    
    def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
        """Convert YCbCr image to RGB color space.
        
        Args:
            ycbcr: YCbCr image tensor of shape (B, 3, H, W)
        
        Returns:
            rgb: RGB image tensor of shape (B, 3, H, W) in range [0, 255]
        
        Raises:
            ValueError: If input tensor doesn't have 3 channels
        """
        if ycbcr.shape[1] != 3:
            raise ValueError(f"Expected 3 channels (YCbCr), got {ycbcr.shape[1]}")
        
        # Reshape for matrix multiplication: (B, 3, H, W) -> (B, H, W, 3)
        ycbcr_permuted = ycbcr.permute(0, 2, 3, 1)
        
        # Subtract offset from Cb and Cr channels
        ycbcr_adjusted = ycbcr_permuted + self.offset
        
        # Apply transformation: (B, H, W, 3) @ (3, 3)^T -> (B, H, W, 3)
        rgb = torch.matmul(ycbcr_adjusted, self.transform_matrix.t())
        
        # Clamp to valid range [0, 255]
        rgb = torch.clamp(rgb, 0, 255)
        
        # Reshape back: (B, H, W, 3) -> (B, 3, H, W)
        rgb = rgb.permute(0, 3, 1, 2)
        
        return rgb
