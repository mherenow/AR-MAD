"""
Noise Analysis Module - Fixed SRM Approach

Generates Fixed SRM (Spatial Rich Model) high-pass filter noise maps to detect
spatial manipulation artifacts through predefined filter kernels.
"""

import numpy as np
from scipy import ndimage
from typing import Dict


def get_srm_kernels() -> Dict[str, np.ndarray]:
    """
    Get predefined SRM filter kernels for spatial artifact detection.
    
    Returns:
        Dictionary mapping kernel names to kernel arrays
    """
    kernels = {}
    
    # Basic SRM 3x3 high-pass kernel
    kernels['srm_3x3'] = np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [-1, 2, -1]
    ], dtype=np.float32)
    
    # SPAM (Subtractive Pixel Adjacency Matrix) kernel
    kernels['spam'] = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    
    # Edge detection kernel for spatial inconsistencies
    kernels['edge'] = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    # 5x5 SRM kernel for broader spatial context
    kernels['srm_5x5'] = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, 2, -1, 0],
        [-1, 2, -4, 2, -1],
        [0, -1, 2, -1, 0],
        [0, 0, -1, 0, 0]
    ], dtype=np.float32)
    
    return kernels


def apply_srm_filter(image: np.ndarray, kernel_type: str = 'srm_3x3') -> np.ndarray:
    """
    Apply Fixed SRM high-pass filter to extract spatial manipulation artifacts.
    
    Args:
        image: Input image as numpy array (RGB or grayscale)
        kernel_type: Type of SRM kernel to use
        
    Returns:
        SRM filtered residual as numpy array with same shape as input
    """
    # Get the specified kernel
    kernels = get_srm_kernels()
    if kernel_type not in kernels:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available: {list(kernels.keys())}")
    
    kernel = kernels[kernel_type]
    
    # Convert to float for accurate computation
    image_float = image.astype(np.float32)
    
    # Apply SRM filter
    if len(image.shape) == 3:
        # RGB image - apply to each channel separately
        filtered = np.zeros_like(image_float)
        for channel in range(image.shape[2]):
            filtered[:, :, channel] = ndimage.convolve(
                image_float[:, :, channel], kernel, mode='reflect'
            )
    else:
        # Grayscale image
        filtered = ndimage.convolve(image_float, kernel, mode='reflect')
    
    return filtered


def generate_srm_noise_map(image: np.ndarray, kernel_type: str = 'srm_3x3') -> np.ndarray:
    """
    Generate a Fixed SRM high-pass filter noise map to reveal spatial manipulation artifacts.
    
    Args:
        image: Input image as numpy array (RGB or grayscale)
        kernel_type: Type of SRM kernel to use ('srm_3x3', 'spam', 'edge', 'srm_5x5')
        
    Returns:
        Fixed SRM filtered noise map as numpy array with same dimensions as input,
        normalized to [0, 1] for visualization
    """
    # Apply SRM filter to extract spatial artifacts
    srm_residual = apply_srm_filter(image, kernel_type)
    
    # Take absolute value to show artifact magnitude
    noise_map = np.abs(srm_residual)
    
    # If RGB, convert to grayscale for visualization
    if len(noise_map.shape) == 3:
        # Average across channels
        noise_map = np.mean(noise_map, axis=2)
    
    # Normalize for visualization (0 to 1 range)
    if noise_map.max() > 0:
        noise_map = noise_map / noise_map.max()
    
    return noise_map



