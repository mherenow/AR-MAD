"""
Frequency domain analysis module for image forensics.

This module provides functions for analyzing images in the frequency domain
using patch-wise Fast Fourier Transform (FFT) to detect localized frequency
domain manipulation artifacts and compression inconsistencies.
"""

import numpy as np
from typing import Tuple, List



def apply_patchwise_fft(image: np.ndarray, patch_size: int = 64) -> List[np.ndarray]:
    """
    Apply FFT to image patches for localized frequency analysis.
    
    Args:
        image: Input image as numpy array (grayscale)
        patch_size: Size of square patches
        
    Returns:
        List of FFT results for each patch
    """
    if len(image.shape) == 3:
        # Convert to grayscale if RGB
        grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        grayscale = image.copy()
    
    # Ensure float type for FFT operations
    grayscale = grayscale.astype(np.float64)
    
    height, width = grayscale.shape
    fft_results = []
    
    # Divide image into patches
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Extract patch (handle boundary cases)
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            patch = grayscale[y:y_end, x:x_end]
            
            # Apply FFT to patch
            fft_patch = np.fft.fft2(patch)
            fft_shifted = np.fft.fftshift(fft_patch)
            fft_results.append(fft_shifted)
    
    return fft_results


def compute_radial_average(fft_magnitude: np.ndarray) -> np.ndarray:
    """
    Compute radial average of FFT magnitude spectrum.
    
    Args:
        fft_magnitude: 2D FFT magnitude spectrum
        
    Returns:
        1D radial frequency profile
    """
    height, width = fft_magnitude.shape
    center_y, center_x = height // 2, width // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    distances = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    
    # Get maximum distance for binning
    max_distance = int(np.sqrt(center_y**2 + center_x**2))
    
    # Compute radial average
    radial_profile = np.zeros(max_distance + 1)
    
    for r in range(max_distance + 1):
        # Create mask for pixels at distance r (with tolerance)
        mask = (distances >= r - 0.5) & (distances < r + 0.5)
        if np.any(mask):
            radial_profile[r] = np.mean(fft_magnitude[mask])
    
    return radial_profile


def generate_patchwise_fft_map(image: np.ndarray, patch_size: int = 64) -> np.ndarray:
    """
    Generate patch-wise FFT log magnitude map.
    
    Args:
        image: Input image as numpy array
        patch_size: Size of patches for localized FFT analysis
        
    Returns:
        Patch-wise log magnitude map as numpy array
    """
    if len(image.shape) == 3:
        # Convert to grayscale if RGB
        grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        grayscale = image.copy()
    
    height, width = grayscale.shape
    result_map = np.zeros((height, width), dtype=np.float64)
    
    # Apply patch-wise FFT
    patch_y = 0
    for y in range(0, height, patch_size):
        patch_x = 0
        for x in range(0, width, patch_size):
            # Extract patch (handle boundary cases)
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            patch = grayscale[y:y_end, x:x_end]
            
            # Apply FFT to patch
            fft_patch = np.fft.fft2(patch)
            fft_shifted = np.fft.fftshift(fft_patch)
            
            # Compute magnitude and apply log scaling
            magnitude = np.abs(fft_shifted)
            log_magnitude = apply_log_scaling(magnitude)
            
            # Place result back in the map
            result_map[y:y_end, x:x_end] = log_magnitude
            
            patch_x += 1
        patch_y += 1
    
    # Normalize to [0, 1] range for visualization
    min_val = np.min(result_map)
    max_val = np.max(result_map)
    
    if max_val > min_val:
        normalized = (result_map - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(result_map)
    
    return normalized


def generate_radial_frequency_stats(image: np.ndarray, patch_size: int = 64) -> np.ndarray:
    """
    Generate 1D radial frequency statistics from patch-wise FFT.
    
    Args:
        image: Input image as numpy array
        patch_size: Size of patches for analysis
        
    Returns:
        1D radial frequency statistics as numpy array
    """
    # Apply patch-wise FFT
    fft_results = apply_patchwise_fft(image, patch_size)
    
    if not fft_results:
        return np.array([])
    
    # Compute magnitude for each patch and collect radial profiles
    all_radial_profiles = []
    
    for fft_patch in fft_results:
        magnitude = np.abs(fft_patch)
        radial_profile = compute_radial_average(magnitude)
        all_radial_profiles.append(radial_profile)
    
    # Average all radial profiles
    if all_radial_profiles:
        # Find the minimum length to handle patches of different sizes
        min_length = min(len(profile) for profile in all_radial_profiles)
        
        # Truncate all profiles to the same length and average
        truncated_profiles = [profile[:min_length] for profile in all_radial_profiles]
        averaged_profile = np.mean(truncated_profiles, axis=0)
        
        return averaged_profile
    else:
        return np.array([])


def apply_log_scaling(magnitude: np.ndarray) -> np.ndarray:
    """
    Apply logarithmic scaling for better visualization.
    
    Args:
        magnitude: Magnitude spectrum
        
    Returns:
        Log-scaled magnitude for visualization
    """
    return np.log(1 + magnitude)



