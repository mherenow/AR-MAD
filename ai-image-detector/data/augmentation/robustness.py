"""
Robustness augmentation module for applying realistic perturbations.

This module implements JPEG compression, Gaussian blur, and Gaussian noise
augmentations with configurable severity levels (1-5) to improve detector
robustness against common image transformations.
"""

import io
from typing import Tuple, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class RobustnessAugmentation:
    """
    Applies JPEG compression, blur, and noise augmentations with severity levels.
    
    This augmentation module applies realistic perturbations to images to improve
    model robustness. Each augmentation type has 5 severity levels (1=mild, 5=severe)
    and can be applied with configurable probability.
    
    Args:
        jpeg_prob: Probability of applying JPEG compression (default: 0.3)
        blur_prob: Probability of applying Gaussian blur (default: 0.3)
        noise_prob: Probability of applying Gaussian noise (default: 0.3)
        severity_range: Tuple of (min, max) severity levels (default: (1, 5))
    
    Example:
        >>> aug = RobustnessAugmentation(jpeg_prob=0.5, blur_prob=0.3, noise_prob=0.3)
        >>> augmented_image = aug(image)
    """
    
    # JPEG compression quality levels by severity
    JPEG_QUALITY = {
        1: 95,  # Mild compression
        2: 85,
        3: 75,
        4: 65,
        5: 50   # Severe compression
    }
    
    # Gaussian blur sigma levels by severity
    BLUR_SIGMA = {
        1: 0.5,  # Mild blur
        2: 1.0,
        3: 1.5,
        4: 2.0,
        5: 2.5   # Severe blur
    }
    
    # Gaussian noise standard deviation by severity
    NOISE_STD = {
        1: 0.01,  # Mild noise
        2: 0.02,
        3: 0.03,
        4: 0.04,
        5: 0.05   # Severe noise
    }
    
    def __init__(
        self,
        jpeg_prob: float = 0.3,
        blur_prob: float = 0.3,
        noise_prob: float = 0.3,
        severity_range: Tuple[int, int] = (1, 5)
    ):
        """Initialize RobustnessAugmentation with configurable probabilities."""
        if not 0 <= jpeg_prob <= 1:
            raise ValueError(f"jpeg_prob must be in [0, 1], got {jpeg_prob}")
        if not 0 <= blur_prob <= 1:
            raise ValueError(f"blur_prob must be in [0, 1], got {blur_prob}")
        if not 0 <= noise_prob <= 1:
            raise ValueError(f"noise_prob must be in [0, 1], got {noise_prob}")
        if not (1 <= severity_range[0] <= severity_range[1] <= 5):
            raise ValueError(f"severity_range must be in [1, 5], got {severity_range}")
        
        self.jpeg_prob = jpeg_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.severity_range = severity_range
    
    def _apply_jpeg_compression(
        self,
        image: torch.Tensor,
        severity: int
    ) -> torch.Tensor:
        """
        Apply JPEG compression at specified severity level.
        
        Args:
            image: Input image tensor (C, H, W) in range [0, 1]
            severity: Severity level (1-5)
        
        Returns:
            Compressed image tensor (C, H, W)
        """
        quality = self.JPEG_QUALITY[severity]
        
        # Convert tensor to PIL Image
        pil_image = TF.to_pil_image(image)
        
        # Apply JPEG compression by encoding and decoding
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to tensor
        return TF.to_tensor(compressed_image)
    
    def _apply_gaussian_blur(
        self,
        image: torch.Tensor,
        severity: int
    ) -> torch.Tensor:
        """
        Apply Gaussian blur at specified severity level.
        
        Args:
            image: Input image tensor (C, H, W)
            severity: Severity level (1-5)
        
        Returns:
            Blurred image tensor (C, H, W)
        """
        sigma = self.BLUR_SIGMA[severity]
        
        # Calculate kernel size (must be odd and large enough for sigma)
        # Rule of thumb: kernel_size = 2 * ceil(3 * sigma) + 1
        kernel_size = int(2 * (3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian blur
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    
    def _apply_gaussian_noise(
        self,
        image: torch.Tensor,
        severity: int
    ) -> torch.Tensor:
        """
        Apply Gaussian noise at specified severity level.
        
        Args:
            image: Input image tensor (C, H, W)
            severity: Severity level (1-5)
        
        Returns:
            Noisy image tensor (C, H, W)
        """
        std = self.NOISE_STD[severity]
        
        # Generate Gaussian noise
        noise = torch.randn_like(image) * std
        
        # Add noise and clamp to valid range [0, 1]
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0.0, 1.0)
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply robustness augmentations to the input image.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W) in range [0, 1]
        
        Returns:
            Augmented image with same shape as input
        """
        # Handle batch dimension
        if image.dim() == 4:
            # Process each image in batch
            return torch.stack([self(img) for img in image])
        
        if image.dim() != 3:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")
        
        # Sample severity level using torch random
        severity = torch.randint(
            self.severity_range[0],
            self.severity_range[1] + 1,
            (1,)
        ).item()
        
        # Apply augmentations with configured probabilities
        augmented = image
        
        if torch.rand(1).item() < self.jpeg_prob:
            augmented = self._apply_jpeg_compression(augmented, severity)
        
        if torch.rand(1).item() < self.blur_prob:
            augmented = self._apply_gaussian_blur(augmented, severity)
        
        if torch.rand(1).item() < self.noise_prob:
            augmented = self._apply_gaussian_noise(augmented, severity)
        
        return augmented
