"""
MixUp augmentation module for training robustness.

This module implements MixUp augmentation, which blends two images together
using a weighted average. Labels are also mixed using the same weight. This
creates smooth interpolations between training examples and improves model
generalization.
"""

from typing import Tuple

import torch
import numpy as np


class MixUpAugmentation:
    """
    MixUp augmentation for training robustness.
    
    MixUp blends two images together using a weighted average:
        mixed_image = lambda * image1 + (1 - lambda) * image2
    
    Labels are also mixed using the same weight:
        mixed_label = lambda * label1 + (1 - lambda) * label2
    
    This creates smooth interpolations between training examples and improves
    model generalization by encouraging the model to behave linearly between
    training examples.
    
    Args:
        alpha: Beta distribution parameter for sampling lambda (default: 0.2)
        prob: Probability of applying MixUp (default: 0.5)
    
    Example:
        >>> mixup = MixUpAugmentation(alpha=0.2, prob=0.5)
        >>> mixed_image, mixed_label = mixup(image1, label1, image2, label2)
    
    Reference:
        mixup: Beyond Empirical Risk Minimization
        (https://arxiv.org/abs/1710.09412)
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """Initialize MixUpAugmentation with Beta distribution parameter."""
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0 <= prob <= 1:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        
        self.alpha = alpha
        self.prob = prob
    
    def _sample_lambda(self) -> float:
        """
        Sample lambda from Beta(alpha, alpha) distribution.
        
        Returns:
            Lambda value in range [0, 1]
        """
        # Sample from Beta distribution using numpy
        lam = np.random.beta(self.alpha, self.alpha)
        return float(lam)
    
    def __call__(
        self,
        image1: torch.Tensor,
        label1: torch.Tensor,
        image2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation to two images.
        
        Args:
            image1: First image tensor (B, C, H, W)
            label1: First label tensor (B,) or (B, 1)
            image2: Second image tensor (B, C, H, W)
            label2: Second label tensor (B,) or (B, 1)
        
        Returns:
            mixed_image: MixUp result (B, C, H, W)
            mixed_label: Interpolated labels (B, 1)
        """
        # Validate inputs
        if image1.shape != image2.shape:
            raise ValueError(
                f"Images must have same shape, got {image1.shape} and {image2.shape}"
            )
        
        if image1.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {image1.dim()}D")
        
        # Ensure labels are 2D (B, 1)
        if label1.dim() == 1:
            label1 = label1.unsqueeze(1)
        if label2.dim() == 1:
            label2 = label2.unsqueeze(1)
        
        # Apply MixUp with configured probability
        if torch.rand(1).item() >= self.prob:
            # Don't apply MixUp, return original
            return image1, label1
        
        # Sample lambda from Beta distribution
        lam = self._sample_lambda()
        
        # Blend images: mixed = lambda * image1 + (1 - lambda) * image2
        mixed_image = lam * image1 + (1.0 - lam) * image2
        
        # Mix labels: mixed_label = lambda * label1 + (1 - lambda) * label2
        mixed_label = lam * label1 + (1.0 - lam) * label2
        
        return mixed_image, mixed_label
