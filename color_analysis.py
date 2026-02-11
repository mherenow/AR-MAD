"""
Color Analysis Module

Analyzes color distribution and detects anomalies that may indicate
image manipulation through splicing or cloning.

Implements color difference and chromatic residual analysis to detect
color-based manipulation artifacts and inconsistencies.
"""

import numpy as np




def compute_channel_differences(image: np.ndarray) -> np.ndarray:
    """
    Compute differences between color channels (R-G, G-B, B-R).
    
    Args:
        image: Input image as numpy array (RGB, values 0-255)
        
    Returns:
        Channel difference maps as numpy array with shape (height, width, 3)
        where channels represent R-G, G-B, B-R differences
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB with 3 channels")
    
    # Convert to float to handle negative differences
    image_float = image.astype(np.float32)
    
    # Extract individual channels
    R = image_float[:, :, 0]
    G = image_float[:, :, 1]
    B = image_float[:, :, 2]
    
    # Compute channel differences
    R_minus_G = R - G
    G_minus_B = G - B
    B_minus_R = B - R
    
    # Stack differences into a 3-channel array
    differences = np.stack([R_minus_G, G_minus_B, B_minus_R], axis=2)
    
    return differences


def extract_chromatic_residuals(image: np.ndarray) -> np.ndarray:
    """
    Extract chromatic residuals that reveal color inconsistencies.
    
    Chromatic residuals are computed by removing the luminance component
    and analyzing the remaining chromatic information for manipulation artifacts.
    
    Args:
        image: Input image as numpy array (RGB)
        
    Returns:
        Chromatic residual map as numpy array
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB with 3 channels")
    
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Extract RGB channels
    R = image_float[:, :, 0]
    G = image_float[:, :, 1]
    B = image_float[:, :, 2]
    
    # Compute luminance (grayscale) using standard weights
    luminance = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Compute chromatic residuals by subtracting luminance from each channel
    # This reveals color information independent of brightness
    chrom_R = R - luminance
    chrom_G = G - luminance
    chrom_B = B - luminance
    
    # Stack chromatic residuals
    chromatic_residuals = np.stack([chrom_R, chrom_G, chrom_B], axis=2)
    
    # Normalize to [0, 1] range for visualization
    # Add 0.5 to center around 0.5 (since residuals can be negative)
    chromatic_residuals = chromatic_residuals + 0.5
    chromatic_residuals = np.clip(chromatic_residuals, 0, 1)
    
    return chromatic_residuals


def generate_color_difference_map(image: np.ndarray) -> np.ndarray:
    """
    Generate a color difference map between color channels.
    
    Args:
        image: Input image as numpy array (RGB)
        
    Returns:
        Color difference map as numpy array
    """
    # Compute channel differences
    differences = compute_channel_differences(image)
    
    # Compute magnitude of differences for visualization
    # Take absolute values and combine channels
    abs_differences = np.abs(differences)
    
    # Combine the three difference channels into a single map
    # Use the maximum difference across channels for each pixel
    difference_map = np.max(abs_differences, axis=2)
    
    # Normalize to [0, 1] for visualization
    if difference_map.max() > 0:
        difference_map = difference_map / difference_map.max()
    
    return difference_map.astype(np.float32)


def generate_chromatic_residual_map(image: np.ndarray) -> np.ndarray:
    """
    Generate a chromatic residual map to reveal color manipulation artifacts.
    
    Args:
        image: Input image as numpy array (RGB)
        
    Returns:
        Chromatic residual map as numpy array
    """
    # Extract chromatic residuals
    chromatic_residuals = extract_chromatic_residuals(image)
    
    # Compute magnitude of chromatic residuals for visualization
    # Convert back to centered around 0 for magnitude calculation
    centered_residuals = chromatic_residuals - 0.5
    
    # Compute magnitude across channels
    residual_magnitude = np.sqrt(np.sum(centered_residuals**2, axis=2))
    
    # Normalize to [0, 1] for visualization
    if residual_magnitude.max() > 0:
        residual_magnitude = residual_magnitude / residual_magnitude.max()
    
    return residual_magnitude.astype(np.float32)
