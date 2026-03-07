"""
COCO 2017 Dataset Loader for PyTorch

This module provides a PyTorch Dataset class for loading the COCO 2017 dataset
as real images (label=0) for training the AI image detector.

The COCO dataset provides diverse real-world images that complement the RAISE
dataset in SynthBuster, offering more variety in real image training data.
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class COCO2017Dataset(Dataset):
    """
    PyTorch Dataset for COCO 2017 train images.
    
    All images are labeled as real (label=0) for binary classification.
    Images are loaded as RGB tensors normalized to [0, 1].
    When native_resolution=False (default), images are resized to 256x256.
    When native_resolution=True, images preserve their original dimensions.
    
    Args:
        root_dir: Root directory containing train2017 folder
        transform: Optional transform to apply to images
        max_samples: Optional limit on number of samples to load
        native_resolution: If True, preserve original image dimensions without resizing (default: False)
        
    Returns:
        Tuple of (image_tensor, label) where:
        - image_tensor: torch.Tensor of shape (3, H, W) normalized to [0, 1]
        - label: int (always 0 for real images)
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        native_resolution: bool = False
    ):
        """
        Initialize the COCO 2017 dataset.
        
        Args:
            root_dir: Path to the root directory (should contain train2017 folder)
            transform: Optional custom transform (if None, default transform is used)
            max_samples: Optional limit on number of samples (useful for balancing datasets)
            native_resolution: If True, preserve original image dimensions without resizing (default: False)
        """
        self.root_dir = Path(root_dir)
        self.train_dir = self.root_dir / "train2017"
        self.transform = transform
        self.max_samples = max_samples
        self.native_resolution = native_resolution
        
        # Default transform: conditionally resize based on native_resolution flag
        if self.transform is None:
            if self.native_resolution:
                # Native resolution mode: only convert to tensor, no resizing
                self.transform = transforms.Compose([
                    transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
                ])
            else:
                # Standard mode: resize to 256x256 for backward compatibility
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
                ])
        
        # Build dataset index
        self.samples = []
        self._build_dataset_index()
        
    def _build_dataset_index(self):
        """
        Scan the train2017 directory and build an index of all valid image files.
        """
        if not self.train_dir.exists():
            raise ValueError(f"Train directory does not exist: {self.train_dir}")
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Scan for image files
        for img_path in self.train_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                self.samples.append(img_path)
                
                # Stop if we've reached max_samples
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
        
        if len(self.samples) == 0:
            warnings.warn(f"No valid images found in {self.train_dir}")
        else:
            print(f"Loaded {len(self.samples)} COCO 2017 images (real, label=0)")
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a sample from the dataset.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Tuple of (image_tensor, label) where label is always 0 (real)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        # Try to load the image, skip corrupted ones
        max_attempts = min(10, len(self.samples))
        attempts = 0
        
        while attempts < max_attempts:
            try:
                img_path = self.samples[idx]
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Apply transform
                image_tensor = self.transform(image)
                
                # Label is always 0 (real)
                return image_tensor, 0
                
            except Exception as e:
                warnings.warn(
                    f"Failed to load image {img_path}: {str(e)}. "
                    f"Skipping to next image."
                )
                # Move to next index (wrap around if necessary)
                idx = (idx + 1) % len(self.samples)
                attempts += 1
        
        # If all attempts failed, raise an error
        raise RuntimeError(
            f"Failed to load any valid images after {max_attempts} attempts. "
            "Dataset may be corrupted."
        )
