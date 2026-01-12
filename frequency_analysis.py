"""
Frequency domain analysis module for image forensics.

This module provides functions for analyzing images in the frequency domain
using Fast Fourier Transform (FFT) and high-pass filtering to detect
compression artifacts and edge inconsistencies.
"""

import numpy as np
from typing import Tuple


def apply_fft(image: np.ndarray) -> np.ndarray:
    """
    Apply 2D Fast Fourier Transform to image.
    
    Args:
        image: Input image as numpy array (grayscale)
        
    Returns:
        Frequency domain representation (complex values)
    """
    # Apply 2D FFT and shift zero frequency to center
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)
    return fft_shifted


def create_highpass_filter(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Create a high-pass filter mask in frequency domain.
    
    Args:
        shape: Shape of the filter (height, width)
        cutoff: Cutoff frequency (0.0 to 1.0), where 1.0 is Nyquist frequency
        
    Returns:
        High-pass filter mask as numpy array (values 0.0 to 1.0)
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate grids centered at the image center
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center (normalized by image size)
    # This gives us the frequency magnitude
    distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)
    
    # Normalize distance to [0, 1] range
    # Maximum distance is from center to corner
    max_distance = np.sqrt(center_row**2 + center_col**2)
    normalized_distance = distance / max_distance
    
    # Create high-pass filter: 0 at low frequencies, 1 at high frequencies
    # Use smooth transition to avoid ringing artifacts
    filter_mask = 1.0 - np.exp(-(normalized_distance**2) / (2 * cutoff**2))
    
    return filter_mask


def apply_ifft(freq_data: np.ndarray) -> np.ndarray:
    """
    Apply inverse FFT to return to spatial domain.
    
    Args:
        freq_data: Frequency domain data (complex values, shifted)
        
    Returns:
        Spatial domain image (real values)
    """
    # Unshift the frequency data
    freq_unshifted = np.fft.ifftshift(freq_data)
    
    # Apply inverse FFT
    spatial_data = np.fft.ifft2(freq_unshifted)
    
    # Take real part and ensure it's real-valued
    result = np.real(spatial_data)
    
    return result


def compute_magnitude_spectrum(fft_data: np.ndarray) -> np.ndarray:
    """
    Compute magnitude spectrum from FFT data.
    
    Args:
        fft_data: Complex FFT output
        
    Returns:
        Magnitude spectrum as numpy array
    """
    return np.abs(fft_data)


def apply_log_scaling(magnitude: np.ndarray) -> np.ndarray:
    """
    Apply logarithmic scaling for better visualization.
    
    Args:
        magnitude: Magnitude spectrum
        
    Returns:
        Log-scaled magnitude for visualization
    """
    return np.log(1 + magnitude)


def generate_frequency_map(image: np.ndarray, cutoff_frequency: float = 0.1) -> np.ndarray:
    """
    Generate FFT magnitude spectrum visualization.
    
    This function transforms the image to frequency domain, computes the magnitude
    spectrum, and applies logarithmic scaling for better visualization of frequency
    components. This helps detect upsampling and smoothing artifacts characteristic
    of AI-generated images.
    
    Args:
        image: Input image as numpy array (RGB or grayscale)
        cutoff_frequency: Unused parameter (kept for backward compatibility)
        
    Returns:
        Log-scaled magnitude spectrum as numpy array for visualization
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        # Use standard RGB to grayscale conversion weights
        grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        grayscale = image.copy()
    
    # Ensure float type for FFT operations
    grayscale = grayscale.astype(np.float64)
    
    # Apply FFT with fftshift to center zero frequency
    freq_data = apply_fft(grayscale)
    
    # Compute magnitude spectrum
    magnitude = compute_magnitude_spectrum(freq_data)
    
    # Apply log scaling for better visualization
    log_magnitude = apply_log_scaling(magnitude)
    
    # Normalize to [0, 1] range for visualization
    min_val = np.min(log_magnitude)
    max_val = np.max(log_magnitude)
    
    if max_val > min_val:
        normalized = (log_magnitude - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(log_magnitude)
    
    return normalized
