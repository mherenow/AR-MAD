"""FFT processor for converting spatial domain images to frequency domain."""

import torch
import torch.nn as nn
from typing import Tuple


class FFTProcessor(nn.Module):
    """
    Converts spatial domain images to frequency domain using 2D FFT.
    
    This module applies 2D Fast Fourier Transform to RGB images, shifts the
    zero-frequency component to the center, and computes the magnitude spectrum
    with log scaling for better visualization and feature extraction.
    
    Args:
        log_scale: Whether to apply log scaling to magnitude spectrum (default: True)
        eps: Small constant for numerical stability in log scaling (default: 1e-8)
    
    Example:
        >>> fft_processor = FFTProcessor()
        >>> image = torch.randn(2, 3, 256, 256)  # Batch of 2 RGB images
        >>> magnitude = fft_processor(image)
        >>> magnitude.shape
        torch.Size([2, 3, 256, 256])
    """
    
    def __init__(self, log_scale: bool = True, eps: float = 1e-8):
        super().__init__()
        self.log_scale = log_scale
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert spatial domain image to frequency domain magnitude spectrum.
        
        Args:
            x: Input image tensor of shape (B, C, H, W) where:
               - B is batch size
               - C is number of channels (typically 3 for RGB)
               - H is height
               - W is width
        
        Returns:
            magnitude: Magnitude spectrum of shape (B, C, H, W)
                      If log_scale=True, returns log(1 + magnitude)
        
        Note:
            The output is shifted so that the DC component (zero frequency)
            is at the center of the spectrum.
        """
        # Apply 2D FFT to each channel
        # torch.fft.fft2 computes 2D FFT along the last two dimensions (H, W)
        fft_result = torch.fft.fft2(x, dim=(-2, -1))
        
        # Shift zero-frequency component to center
        fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        
        # Compute magnitude spectrum
        magnitude = torch.abs(fft_shifted)
        
        # Apply log scaling for better dynamic range
        if self.log_scale:
            magnitude = torch.log(1 + magnitude + self.eps)
        
        return magnitude
    
    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Convert frequency domain back to spatial domain (optional utility).
        
        Args:
            magnitude: Magnitude spectrum (B, C, H, W)
            phase: Phase spectrum (B, C, H, W)
        
        Returns:
            reconstructed: Spatial domain image (B, C, H, W)
        
        Note:
            This is provided for completeness but is not typically used
            in the forward pass of the spectral branch.
        """
        # Reconstruct complex spectrum from magnitude and phase
        complex_spectrum = magnitude * torch.exp(1j * phase)
        
        # Inverse shift
        complex_spectrum = torch.fft.ifftshift(complex_spectrum, dim=(-2, -1))
        
        # Inverse FFT
        reconstructed = torch.fft.ifft2(complex_spectrum, dim=(-2, -1))
        
        # Take real part (imaginary part should be negligible)
        return reconstructed.real
