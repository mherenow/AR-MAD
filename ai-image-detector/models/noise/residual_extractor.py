"""
Noise residual extraction module.

This module extracts noise residuals from images using two methods:
1. Diffusion-based: Uses a pretrained diffusion model to estimate and subtract the clean image
2. Gaussian fallback: Uses Gaussian blur subtraction when diffusers is not available
"""

import warnings
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import diffusers, but don't fail if it's not available
try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    AutoencoderKL = None


class NoiseResidualExtractor(nn.Module):
    """
    Extracts noise residuals using diffusion-based denoising or Gaussian fallback.
    
    The noise residual reveals generator-specific patterns that can be used for
    detection and attribution. This module supports two extraction methods:
    
    1. Diffusion-based (requires diffusers library):
       - Uses a pretrained diffusion model VAE to encode and decode the image
       - The reconstruction removes high-frequency noise patterns
       - Residual = original - reconstructed
       
    2. Gaussian fallback (always available):
       - Applies Gaussian blur to remove high-frequency content
       - Residual = original - blurred
       - Less effective but ensures functionality when diffusers is unavailable
    
    Args:
        method: Denoising method ('diffusion' or 'gaussian')
        diffusion_steps: Number of diffusion steps (default: 50, not used in current implementation)
        gaussian_sigma: Gaussian filter sigma for fallback (default: 2.0)
        
    Raises:
        ValueError: If method='diffusion' but diffusers library is not available
    """
    
    def __init__(
        self,
        method: Literal['diffusion', 'gaussian'] = 'diffusion',
        diffusion_steps: int = 50,
        gaussian_sigma: float = 2.0
    ):
        super().__init__()
        
        self.method = method
        self.diffusion_steps = diffusion_steps
        self.gaussian_sigma = gaussian_sigma
        
        # Check if diffusion method is available
        if method == 'diffusion':
            if not DIFFUSERS_AVAILABLE:
                warnings.warn(
                    "diffusers library not available. Falling back to 'gaussian' method. "
                    "Install diffusers with: pip install diffusers",
                    UserWarning
                )
                self.method = 'gaussian'
            else:
                # Initialize the diffusion model VAE
                # Using Stable Diffusion VAE for image reconstruction
                try:
                    self.vae = AutoencoderKL.from_pretrained(
                        "stabilityai/sd-vae-ft-mse",
                        torch_dtype=torch.float32
                    )
                    self.vae.eval()
                    # Freeze VAE parameters
                    for param in self.vae.parameters():
                        param.requires_grad = False
                except Exception as e:
                    warnings.warn(
                        f"Failed to load diffusion model: {e}. Falling back to 'gaussian' method.",
                        UserWarning
                    )
                    self.method = 'gaussian'
        
        # Compute Gaussian kernel for fallback method
        if self.method == 'gaussian':
            self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())
    
    def _create_gaussian_kernel(self) -> torch.Tensor:
        """
        Create a 2D Gaussian kernel for blur operation.
        
        Returns:
            kernel: Gaussian kernel tensor of shape (1, 1, kernel_size, kernel_size)
        """
        # Kernel size should be odd and large enough to capture the blur
        kernel_size = int(2 * round(3 * self.gaussian_sigma) + 1)
        
        # Create 1D Gaussian
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * self.gaussian_sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Create 2D Gaussian by outer product
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        
        # Reshape for conv2d: (1, 1, H, W)
        return gauss_2d.unsqueeze(0).unsqueeze(0)
    
    def _apply_gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to an image.
        
        Args:
            x: Input image tensor (B, 3, H, W) in range [0, 1]
        
        Returns:
            blurred: Blurred image tensor (B, 3, H, W)
        """
        B, C, H, W = x.shape
        
        # Apply Gaussian blur to each channel separately
        # Pad to maintain spatial dimensions
        kernel_size = self.gaussian_kernel.shape[-1]
        padding = kernel_size // 2
        
        # Reshape to (B*C, 1, H, W) for grouped convolution
        x_reshaped = x.view(B * C, 1, H, W)
        
        # Apply convolution with padding
        blurred = F.conv2d(
            x_reshaped,
            self.gaussian_kernel.to(x.device),
            padding=padding
        )
        
        # Reshape back to (B, C, H, W)
        blurred = blurred.view(B, C, H, W)
        
        return blurred
    
    def _extract_diffusion_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract noise residual using diffusion model VAE.
        
        Args:
            x: Input image tensor (B, 3, H, W) in range [0, 1]
        
        Returns:
            residual: Noise residual (B, 3, H, W)
        """
        with torch.no_grad():
            # Normalize to [-1, 1] for VAE
            x_normalized = x * 2.0 - 1.0
            
            # Encode to latent space
            latent = self.vae.encode(x_normalized).latent_dist.sample()
            
            # Decode back to image space
            reconstructed = self.vae.decode(latent).sample
            
            # Denormalize back to [0, 1]
            reconstructed = (reconstructed + 1.0) / 2.0
            
            # Compute residual
            residual = x - reconstructed
        
        return residual
    
    def _extract_gaussian_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract noise residual using Gaussian blur.
        
        Args:
            x: Input image tensor (B, 3, H, W) in range [0, 1]
        
        Returns:
            residual: Noise residual (B, 3, H, W)
        """
        # Apply Gaussian blur
        blurred = self._apply_gaussian_blur(x)
        
        # Compute residual
        residual = x - blurred
        
        return residual
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract noise residual from input image.
        
        Args:
            x: Input image tensor (B, 3, H, W) in range [0, 1]
        
        Returns:
            residual: Noise residual (B, 3, H, W)
        """
        if self.method == 'diffusion':
            return self._extract_diffusion_residual(x)
        else:
            return self._extract_gaussian_residual(x)
