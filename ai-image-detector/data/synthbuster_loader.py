"""
SynthBuster Dataset Loader for PyTorch

This module provides a PyTorch Dataset class for loading the SynthBuster dataset,
which contains real images from RAISE and synthetic images from various generators.

Workspace Dependencies:
    This module is self-contained and has NO dependencies on other workspace modules.
    It only depends on standard PyTorch and PIL libraries for maximum portability.
    
    - Does NOT use ai-image-detector.utils.config_loader
    - Does NOT use ai-image-detector.models
    - Does NOT use ai-image-detector.training or evaluation modules
    
    This design allows the data loader to be:
    - Used independently in other projects
    - Tested in isolation without workspace dependencies
    - Easily integrated into different training pipelines

Workspace Conventions:
    - Module Structure: Follows the ai-image-detector package structure with separate
      data/, models/, training/, evaluation/, and utils/ modules
    - Import Style: Uses absolute imports (e.g., from ai-image-detector.data import ...)
    - Configuration: While this module doesn't use config_loader, other workspace modules
      use ai-image-detector.utils.config_loader for YAML-based configuration
    - Testing: All modules include corresponding test_*.py files with pytest tests
    - Type Hints: Uses Python type hints for function signatures and return types
    - Docstrings: Follows Google-style docstrings with Args, Returns, and Examples
    
Integration with Workspace:
    - Imported by training module: ai-image-detector.training.train uses this loader
    - Imported by evaluation module: ai-image-detector.evaluation.evaluate uses this loader
    - Exported via __init__.py: Available as 'from ai-image-detector.data import SynthBusterDataset'
    - Configuration: Dataset parameters (root_dir, image_size) are passed from config files
      loaded by ai-image-detector.utils.config_loader in training/evaluation scripts
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SynthBusterDataset(Dataset):
    """
    PyTorch Dataset for SynthBuster dataset.
    
    The dataset structure is expected to be:
    - RAISE/ (real images, label=0)
    - <generator_name>/ (fake images, label=1)
    
    Images are loaded as RGB tensors normalized to [0, 1].
    When native_resolution=False (default), images are resized to 256x256.
    When native_resolution=True, images preserve their original dimensions.
    
    Args:
        root_dir: Root directory containing the dataset folders
        transform: Optional transform to apply to images (default: resize to 256x256 and normalize)
        native_resolution: If True, preserve original image dimensions without resizing (default: False)
        
    Returns:
        Tuple of (image_tensor, label, generator_name) where:
        - image_tensor: torch.Tensor of shape (3, H, W) normalized to [0, 1]
        - label: int (0 for real/RAISE, 1 for fake)
        - generator_name: str (folder name, e.g., "RAISE", "stable-diffusion-v1-4", etc.)
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        native_resolution: bool = False
    ):
        """
        Initialize the SynthBuster dataset.
        
        Args:
            root_dir: Path to the root directory containing dataset folders
            transform: Optional custom transform (if None, default transform is used)
            native_resolution: If True, preserve original image dimensions without resizing (default: False)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.native_resolution = native_resolution
        
        # Default transform: conditionally resize based on native_resolution flag
        if self.transform is None:
            if self.native_resolution:
                # Native resolution mode: only convert to tensor, no resizing
                self.transform = transforms.Compose([
                    transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Standard mode: resize to 256x256 for backward compatibility
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        # Build dataset index
        self.samples = []
        self._build_dataset_index()
        
    def _build_dataset_index(self):
        """
        Scan the root directory and build an index of all valid image files.
        
        Note: RAISE images should be in JPEG format (.jpg, .jpeg).
        Other generators may use various formats.
        """
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
        
        # Supported image extensions
        # Note: RAISE images are expected to be .jpg (converted from TIFF during download)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Iterate through all subdirectories
        for generator_dir in self.root_dir.iterdir():
            if not generator_dir.is_dir():
                continue
                
            generator_name = generator_dir.name
            
            # Determine label: 0 for RAISE (real), 1 for others (fake)
            label = 0 if generator_name == "RAISE" else 1
            
            # Scan for image files
            for img_path in generator_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append({
                        'path': img_path,
                        'label': label,
                        'generator_name': generator_name
                    })
        
        if len(self.samples) == 0:
            warnings.warn(f"No valid images found in {self.root_dir}")
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load and return a sample from the dataset.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Tuple of (image_tensor, label, generator_name)
            
        Note:
            If an image is corrupted, a warning is issued and the next valid image is returned.
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        # Try to load the image, skip corrupted ones
        max_attempts = len(self.samples)
        attempts = 0
        
        while attempts < max_attempts:
            try:
                sample = self.samples[idx]
                img_path = sample['path']
                label = sample['label']
                generator_name = sample['generator_name']
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Apply transform
                image_tensor = self.transform(image)
                
                return image_tensor, label, generator_name
                
            except Exception as e:
                warnings.warn(
                    f"Failed to load image {sample['path']}: {str(e)}. "
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


def create_train_val_split(
    root_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[list, list]:
    """
    Create train/validation split for SynthBuster dataset.
    
    Note: SynthBuster is typically used as a test-only dataset in research.
    This function is provided for flexibility, but standard practice is to
    train on other datasets (e.g., ProGAN, StyleGAN) and test on SynthBuster
    to evaluate generalization to unseen generators.
    
    Args:
        root_dir: Root directory containing the dataset folders
        val_ratio: Ratio of validation samples (default: 0.2 for 80/20 split)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_paths, val_paths) where each is a list of Path objects
        
    Example:
        >>> train_paths, val_paths = create_train_val_split('data/synthbuster', val_ratio=0.2)
        >>> print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    """
    import random
    
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Root directory does not exist: {root_dir}")
    
    # Collect all image paths
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_paths = []
    
    for generator_dir in root_path.iterdir():
        if not generator_dir.is_dir():
            continue
        for img_path in generator_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                all_paths.append(img_path)
    
    if len(all_paths) == 0:
        raise ValueError(f"No valid images found in {root_dir}")
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(all_paths)
    
    # Split into train and validation
    val_size = int(len(all_paths) * val_ratio)
    val_paths = all_paths[:val_size]
    train_paths = all_paths[val_size:]
    
    return train_paths, val_paths


def get_generator_subsets(root_dir: str) -> dict:
    """
    Get file paths organized by generator name.
    
    Supports the following generators commonly found in SynthBuster:
    - SD_v2 (Stable Diffusion v2)
    - GLIDE
    - Firefly (Adobe Firefly)
    - DALLE (DALL-E)
    - Midjourney
    - RAISE (real images)
    
    Args:
        root_dir: Root directory containing the dataset folders
        
    Returns:
        Dictionary mapping generator names to lists of file paths
        
    Example:
        >>> subsets = get_generator_subsets('data/synthbuster')
        >>> print(f"SD_v2 images: {len(subsets.get('SD_v2', []))}")
        >>> print(f"Available generators: {list(subsets.keys())}")
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Root directory does not exist: {root_dir}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    generator_subsets = {}
    
    # Supported generators
    supported_generators = {'SD_v2', 'GLIDE', 'Firefly', 'DALLE', 'Midjourney', 'RAISE'}
    
    for generator_dir in root_path.iterdir():
        if not generator_dir.is_dir():
            continue
        
        generator_name = generator_dir.name
        
        # Collect image paths for this generator
        image_paths = []
        for img_path in generator_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                image_paths.append(img_path)
        
        # Only add if we found images
        if image_paths:
            generator_subsets[generator_name] = image_paths
            
            # Warn if generator is not in the supported list
            if generator_name not in supported_generators:
                warnings.warn(
                    f"Found generator '{generator_name}' which is not in the standard "
                    f"SynthBuster generator list: {supported_generators}"
                )
    
    return generator_subsets
