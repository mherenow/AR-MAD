"""
Augmented Dataset Wrapper

This module provides a wrapper dataset that applies augmentations to base datasets.
It integrates RobustnessAugmentation for per-image augmentations. CutMix and MixUp
augmentations require pairs of images and should be applied at the batch level in
the training loop, not in the dataset wrapper.

The wrapper maintains compatibility with both 2-tuple (image, label) and 3-tuple
(image, label, metadata) dataset formats.
"""

from typing import Optional, Union, Tuple

import torch
from torch.utils.data import Dataset

from data.augmentation.robustness import RobustnessAugmentation


class AugmentedDataset(Dataset):
    """
    Wrapper dataset that applies augmentations to a base dataset.
    
    This wrapper applies per-image augmentations (robustness augmentations like JPEG
    compression, blur, and noise) to images from a base dataset. It maintains
    compatibility with both 2-tuple and 3-tuple dataset formats.
    
    Note: CutMix and MixUp augmentations require pairs of images and should be
    applied at the batch level in the training loop, not in this dataset wrapper.
    
    Args:
        base_dataset: Base dataset to wrap (must implement __getitem__ and __len__)
        robustness_aug: RobustnessAugmentation instance (optional)
        aug_prob: Probability of applying augmentation (default: 1.0)
        
    Example:
        >>> from ai-image-detector.data.synthbuster_loader import SynthBusterDataset
        >>> from ai-image-detector.data.augmentation.robustness import RobustnessAugmentation
        >>> 
        >>> base_dataset = SynthBusterDataset('data/synthbuster')
        >>> robustness_aug = RobustnessAugmentation(jpeg_prob=0.3, blur_prob=0.3, noise_prob=0.3)
        >>> augmented_dataset = AugmentedDataset(base_dataset, robustness_aug=robustness_aug, aug_prob=0.5)
        >>> 
        >>> image, label, generator = augmented_dataset[0]
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        robustness_aug: Optional[RobustnessAugmentation] = None,
        aug_prob: float = 1.0
    ):
        """
        Initialize AugmentedDataset wrapper.
        
        Args:
            base_dataset: Base dataset to wrap
            robustness_aug: RobustnessAugmentation instance (optional)
            aug_prob: Probability of applying augmentation (default: 1.0)
        """
        if not hasattr(base_dataset, '__getitem__') or not hasattr(base_dataset, '__len__'):
            raise ValueError("base_dataset must implement __getitem__ and __len__")
        
        if not 0 <= aug_prob <= 1:
            raise ValueError(f"aug_prob must be in [0, 1], got {aug_prob}")
        
        self.base_dataset = base_dataset
        self.robustness_aug = robustness_aug
        self.aug_prob = aug_prob
    
    def __len__(self) -> int:
        """Return the total number of samples in the base dataset."""
        return len(self.base_dataset)
    
    def __getitem__(
        self,
        idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str]]:
        """
        Load and return a sample from the base dataset with optional augmentation.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            If base dataset returns 2-tuple: (image_tensor, label)
            If base dataset returns 3-tuple: (image_tensor, label, metadata)
            
            The image_tensor may be augmented based on aug_prob and robustness_aug.
        """
        # Get sample from base dataset
        sample = self.base_dataset[idx]
        
        # Handle both 2-tuple and 3-tuple formats
        if len(sample) == 2:
            image, label = sample
            metadata = None
        elif len(sample) == 3:
            image, label, metadata = sample
        else:
            raise ValueError(
                f"Expected base dataset to return 2 or 3 elements, got {len(sample)}"
            )
        
        # Apply augmentation with configured probability
        if self.robustness_aug is not None and torch.rand(1).item() < self.aug_prob:
            image = self.robustness_aug(image)
        
        # Return in the same format as the base dataset
        if metadata is None:
            return image, label
        else:
            return image, label, metadata
