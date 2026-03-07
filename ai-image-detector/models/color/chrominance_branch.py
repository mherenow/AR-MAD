"""Chrominance feature extraction branch for color-based detection.

This module extracts features from chrominance channels (Cb, Cr) including
histogram and variance statistics.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ChrominanceBranch(nn.Module):
    """Extract features from chrominance channels (Cb, Cr).
    
    This module computes histogram and variance features from the Cb and Cr
    channels of YCbCr images, then projects them to a fixed feature dimension.
    
    Args:
        num_bins: Number of histogram bins per channel (default: 64)
        feature_dim: Output feature dimension (default: 256)
    
    Example:
        >>> branch = ChrominanceBranch(num_bins=64, feature_dim=256)
        >>> ycbcr = torch.rand(4, 3, 224, 224) * 255
        >>> features = branch(ycbcr)
        >>> print(features.shape)  # (4, 256)
    """
    
    def __init__(self, num_bins: int = 64, feature_dim: int = 256):
        super().__init__()
        
        self.num_bins = num_bins
        self.feature_dim = feature_dim
        
        # Calculate total feature size before projection
        # Histogram features: 2 channels * num_bins
        # Variance features: 2 channels (global variance) + 2 channels (local variance stats)
        histogram_features = 2 * num_bins
        variance_features = 4  # 2 global + 2 local variance means
        total_features = histogram_features + variance_features
        
        # Feature projection layer
        # Note: Using track_running_stats=True allows batch size of 1 during inference
        self.projection = nn.Sequential(
            nn.Linear(total_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim, track_running_stats=True)
        )
    
    def _compute_histogram(self, channel: torch.Tensor) -> torch.Tensor:
        """Compute normalized histogram for a single channel.
        
        Args:
            channel: Single channel tensor of shape (B, H, W)
        
        Returns:
            histogram: Normalized histogram of shape (B, num_bins)
        """
        batch_size = channel.shape[0]
        histograms = []
        
        for i in range(batch_size):
            # Flatten the channel
            values = channel[i].flatten()
            
            # Compute histogram with bins in range [0, 256]
            hist = torch.histc(values, bins=self.num_bins, min=0, max=256)
            
            # Normalize to probability distribution
            hist = hist / (hist.sum() + 1e-8)
            
            histograms.append(hist)
        
        return torch.stack(histograms, dim=0)
    
    def _compute_variance(self, channel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute global and local variance for a single channel.
        
        Args:
            channel: Single channel tensor of shape (B, H, W)
        
        Returns:
            global_var: Global variance of shape (B,)
            local_var_mean: Mean of local variances of shape (B,)
        """
        batch_size, height, width = channel.shape
        
        # Global variance
        global_var = torch.var(channel.view(batch_size, -1), dim=1)
        
        # Local variance (8x8 patches)
        patch_size = 8
        h_patches = height // patch_size
        w_patches = width // patch_size
        
        if h_patches > 0 and w_patches > 0:
            # Reshape into patches
            patches = channel[:, :h_patches*patch_size, :w_patches*patch_size]
            patches = patches.reshape(
                batch_size, 
                h_patches, patch_size,
                w_patches, patch_size
            )
            patches = patches.permute(0, 1, 3, 2, 4).reshape(
                batch_size, h_patches * w_patches, patch_size * patch_size
            )
            
            # Compute variance for each patch
            local_vars = torch.var(patches, dim=2)
            
            # Mean of local variances
            local_var_mean = torch.mean(local_vars, dim=1)
        else:
            # If image is too small for patches, use global variance
            local_var_mean = global_var
        
        return global_var, local_var_mean
    
    def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
        """Extract chrominance features from YCbCr image.
        
        Args:
            ycbcr: YCbCr image tensor of shape (B, 3, H, W)
        
        Returns:
            features: Chrominance features of shape (B, feature_dim)
        
        Raises:
            ValueError: If input doesn't have 3 channels
        """
        if ycbcr.shape[1] != 3:
            raise ValueError(f"Expected 3 channels (YCbCr), got {ycbcr.shape[1]}")
        
        # Extract Cb and Cr channels
        cb_channel = ycbcr[:, 1, :, :]  # (B, H, W)
        cr_channel = ycbcr[:, 2, :, :]  # (B, H, W)
        
        # Compute histogram features
        cb_hist = self._compute_histogram(cb_channel)  # (B, num_bins)
        cr_hist = self._compute_histogram(cr_channel)  # (B, num_bins)
        
        # Compute variance features
        cb_global_var, cb_local_var = self._compute_variance(cb_channel)  # (B,), (B,)
        cr_global_var, cr_local_var = self._compute_variance(cr_channel)  # (B,), (B,)
        
        # Concatenate all features
        histogram_features = torch.cat([cb_hist, cr_hist], dim=1)  # (B, 2*num_bins)
        variance_features = torch.stack([
            cb_global_var, cb_local_var,
            cr_global_var, cr_local_var
        ], dim=1)  # (B, 4)
        
        all_features = torch.cat([histogram_features, variance_features], dim=1)
        
        # Project to feature_dim
        features = self.projection(all_features)
        
        return features
