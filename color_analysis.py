"""
Color Analysis Module

Analyzes color distribution and detects anomalies that may indicate
image manipulation through splicing or cloning.

Implements 4-bit quantized color distribution analysis (CVPR 2025 "Secret Lies in Color")
to expose non-uniform color tendencies characteristic of AI-generated images.
"""

import numpy as np
from typing import Dict
from scipy.ndimage import gaussian_filter


def compute_block_color_stats(block: np.ndarray) -> Dict[str, float]:
    """
    Compute color statistics for an image block.
    
    Args:
        block: Image block as numpy array (RGB or grayscale)
        
    Returns:
        Dictionary with statistics (mean, std, variance per channel)
    """
    stats = {}
    
    if len(block.shape) == 3:
        # RGB image - compute stats per channel
        for i, channel_name in enumerate(['R', 'G', 'B']):
            channel_data = block[:, :, i].astype(np.float32)
            stats[f'{channel_name}_mean'] = float(np.mean(channel_data))
            stats[f'{channel_name}_std'] = float(np.std(channel_data))
            stats[f'{channel_name}_var'] = float(np.var(channel_data))
    else:
        # Grayscale image
        channel_data = block.astype(np.float32)
        stats['mean'] = float(np.mean(channel_data))
        stats['std'] = float(np.std(channel_data))
        stats['var'] = float(np.var(channel_data))
    
    return stats


def apply_4bit_quantization(image: np.ndarray) -> np.ndarray:
    """
    Apply 4-bit quantization to reduce color depth to 16 levels per channel.
    
    This is part of the CVPR 2025 "Secret Lies in Color" method for detecting
    AI-generated images through non-uniform color tendencies.
    
    Args:
        image: Input image as numpy array (RGB, values 0-255)
        
    Returns:
        4-bit quantized image (16 levels per channel)
    """
    # Ensure image is in the correct format
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # 4-bit quantization: 256 levels -> 16 levels
    # Divide by 16 to get 16 bins (0-15), then multiply by 16 to get back to 0-240 range
    # This creates 16 evenly spaced levels: 0, 16, 32, 48, ..., 240
    quantized = (image // 16) * 16
    
    return quantized.astype(np.uint8)


def apply_gaussian_restoration(quantized_image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian noise restoration to quantized image.
    
    This smoothing step is part of the CVPR 2025 method to reveal AI color artifacts.
    
    Args:
        quantized_image: 4-bit quantized image
        sigma: Standard deviation for Gaussian filter
        
    Returns:
        Restored image after Gaussian filtering
    """
    # Apply Gaussian filter to each channel if RGB, or to the whole image if grayscale
    if len(quantized_image.shape) == 3:
        # RGB image - apply filter to each channel
        restored = np.zeros_like(quantized_image, dtype=np.float32)
        for i in range(quantized_image.shape[2]):
            restored[:, :, i] = gaussian_filter(quantized_image[:, :, i].astype(np.float32), sigma=sigma)
        # Clip to valid range and convert back to uint8
        restored = np.clip(restored, 0, 255).astype(np.uint8)
    else:
        # Grayscale image
        restored = gaussian_filter(quantized_image.astype(np.float32), sigma=sigma)
        restored = np.clip(restored, 0, 255).astype(np.uint8)
    
    return restored


def generate_color_distribution_map(image: np.ndarray) -> np.ndarray:
    """
    Generate a 4-bit quantized color map with Gaussian noise restoration.
    
    Implements the CVPR 2025 "Secret Lies in Color" method to expose non-uniform
    color tendencies of AI-generated images.
    
    Args:
        image: Input image as numpy array (RGB)
        
    Returns:
        4-bit quantized and restored color map as numpy array
    """
    # Apply 4-bit quantization
    quantized = apply_4bit_quantization(image)
    
    # Apply Gaussian restoration
    restored = apply_gaussian_restoration(quantized, sigma=1.0)
    
    return restored


def generate_color_difference_map(image: np.ndarray) -> np.ndarray:
    """
    Generate a difference map between original and restored quantized image.
    
    Implements the CVPR 2025 "Secret Lies in Color" method. The difference reveals
    AI color artifacts - AI-generated images show non-uniform color tendencies that
    become visible through this process.
    
    Args:
        image: Input image as numpy array (RGB)
        
    Returns:
        Color difference map as numpy array (original - restored)
    """
    # Apply 4-bit quantization
    quantized = apply_4bit_quantization(image)
    
    # Apply Gaussian restoration
    restored = apply_gaussian_restoration(quantized, sigma=1.0)
    
    # Compute difference: original - restored
    # Convert to float to avoid underflow
    original_float = image.astype(np.float32)
    restored_float = restored.astype(np.float32)
    
    difference = np.abs(original_float - restored_float)
    
    # Normalize to [0, 1] for visualization
    if difference.max() > 0:
        difference = difference / difference.max()
    
    return difference.astype(np.float32)
