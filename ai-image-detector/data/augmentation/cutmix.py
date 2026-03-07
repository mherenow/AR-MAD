"""
CutMix augmentation module for training robustness.

This module implements CutMix augmentation, which cuts a random rectangular
region from one image and pastes it onto another image. Labels are mixed
proportionally to the area of the pasted region.
"""

from typing import Tuple

import torch
import numpy as np


class CutMixAugmentation:
    """
    CutMix augmentation for training robustness.
    
    CutMix cuts a random rectangular region from one image and pastes it onto
    another image. The labels are mixed proportionally to the area ratio of the
    pasted region. This helps the model learn from multiple images simultaneously
    and improves robustness.
    
    Args:
        alpha: Beta distribution parameter for sampling lambda (default: 1.0)
        prob: Probability of applying CutMix (default: 0.5)
    
    Example:
        >>> cutmix = CutMixAugmentation(alpha=1.0, prob=0.5)
        >>> mixed_image, mixed_label = cutmix(image1, label1, image2, label2)
    
    Reference:
        CutMix: Regularization Strategy to Train Strong Classifiers with
        Localizable Features (https://arxiv.org/abs/1905.04899)
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """Initialize CutMixAugmentation with Beta distribution parameter."""
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
    
    def _get_bbox(
        self,
        width: int,
        height: int,
        lam: float
    ) -> Tuple[int, int, int, int]:
        """
        Calculate random bounding box based on lambda.
        
        The bounding box area is proportional to (1 - lambda) of the total area,
        since lambda represents the proportion of the original image to keep.
        
        Args:
            width: Image width
            height: Image height
            lam: Lambda value from Beta distribution
        
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates
        """
        # Calculate cut ratio (area to cut out)
        cut_ratio = 1.0 - lam
        cut_w = int(width * np.sqrt(cut_ratio))
        cut_h = int(height * np.sqrt(cut_ratio))
        
        # Sample random center point
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        # Calculate bounding box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        
        return int(x1), int(y1), int(x2), int(y2)
    
    def __call__(
        self,
        image1: torch.Tensor,
        label1: torch.Tensor,
        image2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix augmentation to two images.
        
        Args:
            image1: First image tensor (B, C, H, W)
            label1: First label tensor (B,) or (B, 1)
            image2: Second image tensor (B, C, H, W)
            label2: Second label tensor (B,) or (B, 1)
        
        Returns:
            mixed_image: CutMix result (B, C, H, W)
            mixed_label: Interpolated labels (B, 1)
        """
        # Validate inputs
        if image1.shape != image2.shape:
            raise ValueError(
                f"Images must have same shape, got {image1.shape} and {image2.shape}"
            )
        
        if image1.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {image1.dim()}D")
        
        batch_size, channels, height, width = image1.shape
        
        # Ensure labels are 2D (B, 1)
        if label1.dim() == 1:
            label1 = label1.unsqueeze(1)
        if label2.dim() == 1:
            label2 = label2.unsqueeze(1)
        
        # Apply CutMix with configured probability
        if torch.rand(1).item() >= self.prob:
            # Don't apply CutMix, return original
            return image1, label1
        
        # Sample lambda from Beta distribution
        lam = self._sample_lambda()
        
        # Get random bounding box
        x1, y1, x2, y2 = self._get_bbox(width, height, lam)
        
        # Create mixed image by copying image1 and pasting region from image2
        mixed_image = image1.clone()
        mixed_image[:, :, y1:y2, x1:x2] = image2[:, :, y1:y2, x1:x2]
        
        # Calculate actual area ratio (may differ from lambda due to clipping)
        bbox_area = (x2 - x1) * (y2 - y1)
        total_area = width * height
        area_ratio = bbox_area / total_area
        
        # Mix labels based on actual area ratio
        # mixed_label = (1 - area_ratio) * label1 + area_ratio * label2
        mixed_label = (1.0 - area_ratio) * label1 + area_ratio * label2
        
        return mixed_image, mixed_label
