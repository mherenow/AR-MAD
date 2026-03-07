"""Frequency masking module for filtering frequency domain features."""

import torch
import torch.nn as nn
from typing import Literal, Tuple


class FrequencyMasking(nn.Module):
    """
    Applies frequency domain filtering (low-pass, high-pass, or band-pass).
    
    This module creates frequency masks to filter specific frequency bands
    in the frequency domain. It supports three types of filters:
    - Low-pass: Keeps low frequencies, removes high frequencies
    - High-pass: Keeps high frequencies, removes low frequencies
    - Band-pass: Keeps a specific frequency band
    
    Args:
        mask_type: Type of frequency filter ('low_pass', 'high_pass', 'band_pass')
        cutoff_freq: Cutoff frequency as a fraction of max frequency (0.0 to 1.0)
                     For band-pass, this is the center frequency
        bandwidth: Bandwidth for band-pass filter (default: 0.1)
        preserve_dc: Whether to preserve DC component (zero frequency) (default: True)
    
    Example:
        >>> masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3)
        >>> spectrum = torch.randn(2, 3, 256, 256)
        >>> filtered = masking(spectrum)
        >>> filtered.shape
        torch.Size([2, 3, 256, 256])
    """
    
    def __init__(
        self,
        mask_type: Literal['low_pass', 'high_pass', 'band_pass'] = 'high_pass',
        cutoff_freq: float = 0.3,
        bandwidth: float = 0.1,
        preserve_dc: bool = True
    ):
        super().__init__()
        self.mask_type = mask_type
        self.cutoff_freq = cutoff_freq
        self.bandwidth = bandwidth
        self.preserve_dc = preserve_dc
        
        # Validate parameters
        if not 0.0 <= cutoff_freq <= 1.0:
            raise ValueError(f"cutoff_freq must be in [0, 1], got {cutoff_freq}")
        if mask_type not in ['low_pass', 'high_pass', 'band_pass']:
            raise ValueError(f"mask_type must be 'low_pass', 'high_pass', or 'band_pass', got {mask_type}")
    
    def _create_frequency_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Create a frequency mask based on the specified filter type.
        
        Args:
            height: Height of the frequency domain
            width: Width of the frequency domain
            device: Device to create the mask on
        
        Returns:
            mask: Binary mask of shape (1, 1, H, W)
        """
        # Create coordinate grids centered at (0, 0)
        # For fftshift output, center is at (H//2, W//2)
        y = torch.arange(height, device=device) - height // 2
        x = torch.arange(width, device=device) - width // 2
        
        # Create meshgrid
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Compute normalized distance from center
        # Max distance is sqrt((H/2)^2 + (W/2)^2)
        max_dist = torch.sqrt(torch.tensor((height / 2) ** 2 + (width / 2) ** 2, device=device))
        distance = torch.sqrt(yy.float() ** 2 + xx.float() ** 2) / max_dist
        
        # Create mask based on filter type
        if self.mask_type == 'low_pass':
            # Keep frequencies below cutoff
            mask = (distance <= self.cutoff_freq).float()
        
        elif self.mask_type == 'high_pass':
            # Keep frequencies above cutoff
            mask = (distance >= self.cutoff_freq).float()
        
        elif self.mask_type == 'band_pass':
            # Keep frequencies in band [cutoff - bandwidth/2, cutoff + bandwidth/2]
            lower = self.cutoff_freq - self.bandwidth / 2
            upper = self.cutoff_freq + self.bandwidth / 2
            mask = ((distance >= lower) & (distance <= upper)).float()
        
        # Preserve DC component if requested
        if self.preserve_dc:
            center_y, center_x = height // 2, width // 2
            mask[center_y, center_x] = 1.0
        
        # Add batch and channel dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return mask
    
    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to the input spectrum.
        
        Args:
            spectrum: Frequency domain tensor of shape (B, C, H, W)
                     Should be the output of FFTProcessor (magnitude spectrum)
        
        Returns:
            masked_spectrum: Filtered spectrum of shape (B, C, H, W)
        
        Note:
            The mask is created once per forward pass and broadcasted across
            batch and channel dimensions for efficiency.
        """
        B, C, H, W = spectrum.shape
        
        # Create frequency mask
        mask = self._create_frequency_mask(H, W, spectrum.device)
        
        # Apply mask (broadcast across batch and channels)
        masked_spectrum = spectrum * mask
        
        return masked_spectrum
    
    def get_mask(self, height: int, width: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Get the frequency mask for visualization purposes.
        
        Args:
            height: Height of the mask
            width: Width of the mask
            device: Device to create the mask on
        
        Returns:
            mask: Frequency mask of shape (1, 1, H, W)
        """
        return self._create_frequency_mask(height, width, device)
