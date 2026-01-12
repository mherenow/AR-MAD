"""
Noise Analysis Module - NIRNet Approach

Generates high-pass residual noise maps to distinguish between sensor noise patterns
(real images) and generator noise patterns (AI-generated images).
"""

import numpy as np
import cv2


def apply_highpass_filter(image: np.ndarray, filter_size: int = 5) -> np.ndarray:
    """
    Apply a high-pass filter to extract high-frequency noise residuals.
    
    This implements the NIRNet approach: extract high-frequency residuals by
    subtracting a low-pass filtered version from the original image.
    
    Args:
        image: Input image as numpy array (RGB or grayscale)
        filter_size: Size of the filter kernel (must be odd)
        
    Returns:
        High-pass filtered residual as numpy array with same shape as input
    """
    # Ensure filter size is odd
    if filter_size % 2 == 0:
        filter_size += 1
    
    # Convert to float for accurate computation
    image_float = image.astype(np.float32)
    
    # Apply Gaussian low-pass filter
    if len(image.shape) == 3:
        # RGB image - apply to each channel
        lowpass = cv2.GaussianBlur(image_float, (filter_size, filter_size), 0)
    else:
        # Grayscale image
        lowpass = cv2.GaussianBlur(image_float, (filter_size, filter_size), 0)
    
    # High-pass residual = original - lowpass
    # This extracts high-frequency components (noise patterns)
    highpass_residual = image_float - lowpass
    
    return highpass_residual


def generate_noise_map(image: np.ndarray, filter_size: int = 5) -> np.ndarray:
    """
    Generate a high-pass residual noise map to reveal sensor vs generator noise patterns.
    
    This implements the NIRNet approach for detecting AI-generated images:
    - Real camera images show characteristic PRNU-like sensor noise
    - AI-generated images show different noise patterns from the generation process
    
    Args:
        image: Input image as numpy array (RGB or grayscale)
        filter_size: Size of the high-pass filter kernel
        
    Returns:
        High-pass residual noise map as numpy array with same dimensions as input,
        normalized to [0, 1] for visualization
    """
    # Apply high-pass filter to extract noise residuals
    highpass_residual = apply_highpass_filter(image, filter_size)
    
    # Take absolute value to show noise magnitude
    noise_map = np.abs(highpass_residual)
    
    # If RGB, convert to grayscale for visualization
    if len(noise_map.shape) == 3:
        # Average across channels
        noise_map = np.mean(noise_map, axis=2)
    
    # Normalize for visualization (0 to 1 range)
    if noise_map.max() > 0:
        noise_map = noise_map / noise_map.max()
    
    return noise_map
