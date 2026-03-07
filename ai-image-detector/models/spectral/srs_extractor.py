"""Spectral Response Signature (SRS) extractor for frequency domain features."""

import torch
import torch.nn as nn
from typing import Optional


class SRSExtractor(nn.Module):
    """
    Extracts Spectral Response Signatures (SRS) from tokenized spectral patches.
    
    SRS vectors are fixed-size feature representations that aggregate information
    across frequency bands. This module computes signatures by aggregating patch
    embeddings from the spectral tokenizer, producing a compact representation
    of the frequency domain characteristics.
    
    The aggregation is performed across frequency bands to capture both low and
    high frequency patterns that are indicative of ML-generated images.
    
    Args:
        embed_dim: Embedding dimension from patch tokenizer (default: 256)
        num_bands: Number of frequency bands to aggregate (default: 4)
        aggregation_method: Method for aggregation ('mean', 'max', or 'attention')
                           (default: 'mean')
    
    Example:
        >>> extractor = SRSExtractor(embed_dim=256)
        >>> tokens = torch.randn(2, 256, 256)  # (batch, num_patches, embed_dim)
        >>> srs = extractor(tokens)
        >>> srs.shape
        torch.Size([2, 256])  # (batch_size, embed_dim)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_bands: int = 4,
        aggregation_method: str = 'mean'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.aggregation_method = aggregation_method
        
        # Validate aggregation method
        valid_methods = ['mean', 'max', 'attention']
        if aggregation_method not in valid_methods:
            raise ValueError(
                f"aggregation_method must be one of {valid_methods}, "
                f"got '{aggregation_method}'"
            )
        
        # Attention-based aggregation requires learnable parameters
        if aggregation_method == 'attention':
            self.attention_weights = nn.Linear(embed_dim, 1)
    
    def _split_into_bands(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Split tokenized patches into frequency bands.
        
        Assumes patches are arranged in row-major order corresponding to
        frequency domain layout (after fftshift, DC is at center).
        
        Args:
            tokens: Tokenized patches (B, num_patches, embed_dim)
            height: Number of patches along height
            width: Number of patches along width
        
        Returns:
            bands: Tokens organized by frequency bands (B, num_bands, patches_per_band, embed_dim)
        """
        B, num_patches, embed_dim = tokens.shape
        
        # Reshape to spatial layout
        # (B, num_patches, embed_dim) -> (B, height, width, embed_dim)
        tokens_spatial = tokens.view(B, height, width, embed_dim)
        
        # Calculate center (DC component location after fftshift)
        center_h = height // 2
        center_w = width // 2
        
        # Create distance map from center (frequency magnitude)
        h_coords = torch.arange(height, device=tokens.device).view(-1, 1).float()
        w_coords = torch.arange(width, device=tokens.device).view(1, -1).float()
        
        # Euclidean distance from center
        dist_map = torch.sqrt(
            (h_coords - center_h) ** 2 + (w_coords - center_w) ** 2
        )
        
        # Normalize distances to [0, 1]
        max_dist = torch.sqrt(torch.tensor(center_h ** 2 + center_w ** 2, dtype=torch.float32))
        dist_map = dist_map / max_dist
        
        # Assign patches to bands based on distance
        # Band 0: lowest frequencies (near DC)
        # Band num_bands-1: highest frequencies (far from DC)
        band_assignments = (dist_map * self.num_bands).long()
        band_assignments = torch.clamp(band_assignments, 0, self.num_bands - 1)
        
        # Collect patches for each band
        bands_list = []
        for band_idx in range(self.num_bands):
            # Get mask for current band
            band_mask = (band_assignments == band_idx)
            
            # Extract patches belonging to this band
            # band_mask shape: (height, width)
            # tokens_spatial shape: (B, height, width, embed_dim)
            band_patches = tokens_spatial[:, band_mask, :]  # (B, patches_in_band, embed_dim)
            bands_list.append(band_patches)
        
        return bands_list
    
    def _aggregate_mean(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Aggregate tokens using mean pooling.
        
        Args:
            tokens: Input tokens (B, num_patches, embed_dim)
        
        Returns:
            aggregated: Mean-pooled features (B, embed_dim)
        """
        return tokens.mean(dim=1)
    
    def _aggregate_max(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Aggregate tokens using max pooling.
        
        Args:
            tokens: Input tokens (B, num_patches, embed_dim)
        
        Returns:
            aggregated: Max-pooled features (B, embed_dim)
        """
        return tokens.max(dim=1)[0]
    
    def _aggregate_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Aggregate tokens using learned attention weights.
        
        Args:
            tokens: Input tokens (B, num_patches, embed_dim)
        
        Returns:
            aggregated: Attention-weighted features (B, embed_dim)
        """
        # Compute attention scores
        # (B, num_patches, embed_dim) -> (B, num_patches, 1)
        attention_scores = self.attention_weights(tokens)
        
        # Apply softmax to get attention weights
        # (B, num_patches, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum
        # (B, num_patches, embed_dim) * (B, num_patches, 1) -> (B, num_patches, embed_dim)
        weighted_tokens = tokens * attention_weights
        
        # Sum across patches
        # (B, num_patches, embed_dim) -> (B, embed_dim)
        aggregated = weighted_tokens.sum(dim=1)
        
        return aggregated
    
    def forward(
        self,
        tokens: torch.Tensor,
        spatial_dims: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Extract Spectral Response Signatures from tokenized patches.
        
        Args:
            tokens: Tokenized spectral patches of shape (B, num_patches, embed_dim)
            spatial_dims: Optional tuple (height, width) specifying the spatial
                         arrangement of patches. If None, assumes square layout.
        
        Returns:
            srs: Spectral Response Signatures of shape (B, embed_dim)
        
        Raises:
            ValueError: If spatial_dims is provided but doesn't match num_patches
        """
        B, num_patches, embed_dim = tokens.shape
        
        # Determine spatial dimensions
        if spatial_dims is None:
            # Assume square layout
            height = width = int(num_patches ** 0.5)
            if height * width != num_patches:
                raise ValueError(
                    f"Cannot infer square spatial layout from {num_patches} patches. "
                    "Please provide spatial_dims explicitly."
                )
        else:
            height, width = spatial_dims
            if height * width != num_patches:
                raise ValueError(
                    f"spatial_dims ({height}, {width}) doesn't match "
                    f"num_patches ({num_patches})"
                )
        
        # Split tokens into frequency bands
        bands_list = self._split_into_bands(tokens, height, width)
        
        # Aggregate each band
        band_features = []
        for band_patches in bands_list:
            if band_patches.size(1) == 0:
                # Empty band (can happen with small images)
                # Use zeros
                band_feat = torch.zeros(B, embed_dim, device=tokens.device)
            else:
                # Apply aggregation method
                if self.aggregation_method == 'mean':
                    band_feat = self._aggregate_mean(band_patches)
                elif self.aggregation_method == 'max':
                    band_feat = self._aggregate_max(band_patches)
                elif self.aggregation_method == 'attention':
                    band_feat = self._aggregate_attention(band_patches)
            
            band_features.append(band_feat)
        
        # Stack band features and aggregate across bands
        # (num_bands, B, embed_dim) -> (B, num_bands, embed_dim)
        band_features = torch.stack(band_features, dim=1)
        
        # Final aggregation across bands to produce fixed-size SRS
        # (B, num_bands, embed_dim) -> (B, embed_dim)
        if self.aggregation_method == 'mean':
            srs = band_features.mean(dim=1)
        elif self.aggregation_method == 'max':
            srs = band_features.max(dim=1)[0]
        elif self.aggregation_method == 'attention':
            # Use attention across bands
            attention_scores = self.attention_weights(band_features)
            attention_weights = torch.softmax(attention_scores, dim=1)
            srs = (band_features * attention_weights).sum(dim=1)
        
        return srs
