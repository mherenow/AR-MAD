"""
Variable-Size Collate Function for PyTorch DataLoader

This module provides a custom collate function for handling batches of variable-sized images.
When native_resolution=True is used with data loaders, images in a batch may have different
dimensions. This collate function handles such cases by returning a list of tensors instead
of attempting to stack them into a single tensor.

Workspace Dependencies:
    This module is self-contained and has NO dependencies on other workspace modules.
    It only depends on standard PyTorch libraries for maximum portability.
    
    - Does NOT use ai-image-detector.utils.config_loader
    - Does NOT use ai-image-detector.models
    - Does NOT use ai-image-detector.training or evaluation modules
    
    This design allows the collate function to be:
    - Used independently in other projects
    - Tested in isolation without workspace dependencies
    - Easily integrated into different training pipelines

Workspace Conventions:
    - Module Structure: Follows the ai-image-detector package structure with separate
      data/, models/, training/, evaluation/, and utils/ modules
    - Import Style: Uses absolute imports (e.g., from ai-image-detector.data import ...)
    - Type Hints: Uses Python type hints for function signatures and return types
    - Docstrings: Follows Google-style docstrings with Args, Returns, and Examples
"""

from typing import List, Tuple, Union
import torch


def variable_size_collate_fn(
    batch: List[Tuple[torch.Tensor, int, str]]
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor, List[str]]:
    """
    Collate function for variable-sized images.
    
    This function handles batches where images may have different dimensions.
    It supports both 2-tuple (image, label) and 3-tuple (image, label, metadata) formats.
    
    When all images in the batch have the same size, they are stacked into a single tensor
    for efficiency (backward compatibility with fixed-size batches). When images have
    different sizes, they are returned as a list of tensors.
    
    Args:
        batch: List of tuples, where each tuple is (image, label, generator_name)
               - image: torch.Tensor of shape (C, H, W)
               - label: int (0 or 1)
               - generator_name: str (e.g., "RAISE", "stable-diffusion-v1-4")
    
    Returns:
        Tuple of (images, labels, generator_names) where:
        - images: Either torch.Tensor of shape (B, C, H, W) if all same size,
                  or List[torch.Tensor] if variable sizes
        - labels: torch.Tensor of shape (B,) containing label values
        - generator_names: List[str] of length B containing generator names
    
    Examples:
        >>> # Fixed-size batch (all 256x256)
        >>> batch = [(torch.rand(3, 256, 256), 0, "RAISE"),
        ...          (torch.rand(3, 256, 256), 1, "SD_v2")]
        >>> images, labels, names = variable_size_collate_fn(batch)
        >>> print(images.shape)  # torch.Size([2, 3, 256, 256])
        >>> print(labels.shape)  # torch.Size([2])
        >>> print(names)  # ["RAISE", "SD_v2"]
        
        >>> # Variable-size batch
        >>> batch = [(torch.rand(3, 256, 256), 0, "RAISE"),
        ...          (torch.rand(3, 512, 512), 1, "SD_v2")]
        >>> images, labels, names = variable_size_collate_fn(batch)
        >>> print(type(images))  # <class 'list'>
        >>> print(len(images))  # 2
        >>> print(images[0].shape)  # torch.Size([3, 256, 256])
        >>> print(images[1].shape)  # torch.Size([3, 512, 512])
    
    Note:
        This function maintains backward compatibility with fixed-size batches by
        stacking images when all have the same dimensions. This allows existing
        code to work without modification when native_resolution=False.
    """
    # Separate the batch into components
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    generator_names = [item[2] for item in batch]
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Check if all images have the same size
    first_shape = images[0].shape
    all_same_size = all(img.shape == first_shape for img in images)
    
    if all_same_size:
        # Stack into a single tensor for efficiency (backward compatibility)
        images_tensor = torch.stack(images, dim=0)
        return images_tensor, labels_tensor, generator_names
    else:
        # Return as list of tensors for variable sizes
        return images, labels_tensor, generator_names


def variable_size_collate_fn_2tuple(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
    """
    Collate function for variable-sized images with 2-tuple format (image, label).
    
    This is a simplified version of variable_size_collate_fn for datasets that
    only return (image, label) tuples without metadata.
    
    Args:
        batch: List of tuples, where each tuple is (image, label)
               - image: torch.Tensor of shape (C, H, W)
               - label: int (0 or 1)
    
    Returns:
        Tuple of (images, labels) where:
        - images: Either torch.Tensor of shape (B, C, H, W) if all same size,
                  or List[torch.Tensor] if variable sizes
        - labels: torch.Tensor of shape (B,) containing label values
    
    Examples:
        >>> # Fixed-size batch
        >>> batch = [(torch.rand(3, 256, 256), 0),
        ...          (torch.rand(3, 256, 256), 1)]
        >>> images, labels = variable_size_collate_fn_2tuple(batch)
        >>> print(images.shape)  # torch.Size([2, 3, 256, 256])
        
        >>> # Variable-size batch
        >>> batch = [(torch.rand(3, 256, 256), 0),
        ...          (torch.rand(3, 512, 512), 1)]
        >>> images, labels = variable_size_collate_fn_2tuple(batch)
        >>> print(type(images))  # <class 'list'>
        >>> print(len(images))  # 2
    """
    # Separate the batch into components
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Check if all images have the same size
    first_shape = images[0].shape
    all_same_size = all(img.shape == first_shape for img in images)
    
    if all_same_size:
        # Stack into a single tensor for efficiency
        images_tensor = torch.stack(images, dim=0)
        return images_tensor, labels_tensor
    else:
        # Return as list of tensors for variable sizes
        return images, labels_tensor
