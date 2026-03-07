"""Spectral Consistency Vector (SCV) computer for frequency domain analysis."""

import torch
import torch.nn as nn
from typing import Optional


class SCVComputer(nn.Module):
    """
    Computes Spectral Consistency Vectors (SCV) across frequency bands.
    
    SCV measures the consistency and statistical properties of spectral features
    across different frequency bands. This helps identify anomalies and patterns
    that are characteristic of ML-generated images, which often exhibit unusual
    consistency or inconsistency patterns in the frequency domain.
    
    The SCV includes:
    - Variance statistics across frequency bands
    - Correlation between adjacent bands
    - Energy distribution across bands
    
    Args:
        embed_dim: Embedding dimension from patch tokenizer (default: 256)
        num_bands: Number of frequency bands to analyze (default: 4)
        consistency_dim: Output dimension for consistency vector (default: 128)
    
    Example:
        >>> computer = SCVComputer(embed_dim=256, consistency_dim=128)
        >>> tokens = torch.randn(2, 256, 256)  # (batch, num_patches, embed_dim)
        >>> scv = computer(tokens)
        >>> scv.shape
        torch.Size([2, 128])  # (batch_size, consistency_dim)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_bands: int = 4,
        consistency_dim: int = 128
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.consistency_dim = consistency_dim
        
        # Calculate feature dimension before projection
        # Features include: variance per band, correlation between bands, energy per band
        # variance: num_bands features
        # correlation: (num_bands - 1) features (adjacent pairs)
        # energy: num_bands features
        self.raw_feature_dim = num_bands + (num_bands - 1) + num_bands
        
        # Projection layer to map raw features to consistency_dim
        self.projection = nn.Sequential(
            nn.Linear(self.raw_feature_dim, consistency_dim * 2),
            nn.ReLU(),
            nn.Linear(consistency_dim * 2, consistency_dim)
        )
    
    def _split_into_bands(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int
    ) -> list:
        """
        Split tokenized patches into frequency bands.
        
        Assumes patches are arranged in row-major order corresponding to
        frequency domain layout (after fftshift, DC is at center).
        
        Args:
            tokens: Tokenized patches (B, num_patches, embed_dim)
            height: Number of patches along height
            width: Number of patches along width
        
        Returns:
            bands: List of tensors, one per band (B, patches_per_band, embed_dim)
        """
        B, num_patches, embed_dim = tokens.shape
        
        # Reshape to spatial layout
        tokens_spatial = tokens.view(B, height, width, embed_dim)
        
        # Calculate center (DC component location after fftshift)
        center_h = height // 2
        center_w = width // 2
        
        # Create distance map from center
        h_coords = torch.arange(height, device=tokens.device).view(-1, 1).float()
        w_coords = torch.arange(width, device=tokens.device).view(1, -1).float()
        
        dist_map = torch.sqrt(
            (h_coords - center_h) ** 2 + (w_coords - center_w) ** 2
        )
        
        # Normalize distances to [0, 1]
        max_dist = torch.sqrt(torch.tensor(center_h ** 2 + center_w ** 2, dtype=torch.float32))
        dist_map = dist_map / max_dist
        
        # Assign patches to bands
        band_assignments = (dist_map * self.num_bands).long()
        band_assignments = torch.clamp(band_assignments, 0, self.num_bands - 1)
        
        # Collect patches for each band
        bands_list = []
        for band_idx in range(self.num_bands):
            band_mask = (band_assignments == band_idx)
            band_patches = tokens_spatial[:, band_mask, :]
            bands_list.append(band_patches)
        
        return bands_list
    
    def _compute_variance(self, band_patches: torch.Tensor) -> torch.Tensor:
        """
        Compute variance of features within a frequency band.
        
        Args:
            band_patches: Patches in a band (B, patches_in_band, embed_dim)
        
        Returns:
            variance: Variance across patches (B,)
        """
        if band_patches.size(1) <= 1:
            # Empty band or single patch (variance undefined)
            return torch.zeros(band_patches.size(0), device=band_patches.device)
        
        # Compute variance across patches (dim=1) and average across embed_dim (dim=2)
        variance = band_patches.var(dim=1, unbiased=False).mean(dim=1)
        return variance
    
    def _compute_energy(self, band_patches: torch.Tensor) -> torch.Tensor:
        """
        Compute energy (mean squared magnitude) of features in a frequency band.
        
        Args:
            band_patches: Patches in a band (B, patches_in_band, embed_dim)
        
        Returns:
            energy: Energy of the band (B,)
        """
        if band_patches.size(1) == 0:
            # Empty band
            return torch.zeros(band_patches.size(0), device=band_patches.device)
        
        # Compute mean squared magnitude
        energy = (band_patches ** 2).mean(dim=(1, 2))
        return energy
    
    def _compute_correlation(
        self,
        band1_patches: torch.Tensor,
        band2_patches: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation between two adjacent frequency bands.
        
        Args:
            band1_patches: Patches in first band (B, patches1, embed_dim)
            band2_patches: Patches in second band (B, patches2, embed_dim)
        
        Returns:
            correlation: Correlation coefficient (B,)
        """
        B = band1_patches.size(0)
        
        if band1_patches.size(1) == 0 or band2_patches.size(1) == 0:
            # Empty band
            return torch.zeros(B, device=band1_patches.device)
        
        # Aggregate each band to a single vector per batch
        band1_agg = band1_patches.mean(dim=1)  # (B, embed_dim)
        band2_agg = band2_patches.mean(dim=1)  # (B, embed_dim)
        
        # Compute correlation coefficient
        # Normalize vectors
        band1_norm = band1_agg - band1_agg.mean(dim=1, keepdim=True)
        band2_norm = band2_agg - band2_agg.mean(dim=1, keepdim=True)
        
        # Compute correlation
        numerator = (band1_norm * band2_norm).sum(dim=1)
        denominator = (
            torch.sqrt((band1_norm ** 2).sum(dim=1)) *
            torch.sqrt((band2_norm ** 2).sum(dim=1))
        )
        
        # Avoid division by zero
        correlation = numerator / (denominator + 1e-8)
        
        # Clamp to valid correlation range [-1, 1]
        correlation = torch.clamp(correlation, -1.0, 1.0)
        
        return correlation
    
    def forward(
        self,
        tokens: torch.Tensor,
        spatial_dims: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Compute Spectral Consistency Vectors from tokenized patches.
        
        Args:
            tokens: Tokenized spectral patches of shape (B, num_patches, embed_dim)
            spatial_dims: Optional tuple (height, width) specifying the spatial
                         arrangement of patches. If None, assumes square layout.
        
        Returns:
            scv: Spectral Consistency Vectors of shape (B, consistency_dim)
        
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
        
        # Compute variance for each band
        variances = []
        for band_patches in bands_list:
            variance = self._compute_variance(band_patches)
            variances.append(variance)
        
        # Stack variances: (num_bands, B) -> (B, num_bands)
        variances = torch.stack(variances, dim=1)
        
        # Compute energy for each band
        energies = []
        for band_patches in bands_list:
            energy = self._compute_energy(band_patches)
            energies.append(energy)
        
        # Stack energies: (num_bands, B) -> (B, num_bands)
        energies = torch.stack(energies, dim=1)
        
        # Compute correlation between adjacent bands
        correlations = []
        for i in range(self.num_bands - 1):
            correlation = self._compute_correlation(bands_list[i], bands_list[i + 1])
            correlations.append(correlation)
        
        # Stack correlations: (num_bands-1, B) -> (B, num_bands-1)
        if correlations:
            correlations = torch.stack(correlations, dim=1)
        else:
            # Single band case
            correlations = torch.zeros(B, 0, device=tokens.device)
        
        # Concatenate all features
        # (B, num_bands) + (B, num_bands-1) + (B, num_bands) -> (B, raw_feature_dim)
        raw_features = torch.cat([variances, correlations, energies], dim=1)
        
        # Project to consistency_dim
        scv = self.projection(raw_features)
        
        return scv
