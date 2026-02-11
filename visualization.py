"""
Visualization manager for displaying forensic analysis results.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def create_subplot_grid(num_images: int) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions for subplots.
    
    Args:
        num_images: Number of images to display
        
    Returns:
        Tuple of (rows, columns) for subplot grid
    """
    if num_images <= 0:
        return (0, 0)
    
    # Calculate rows and columns for a roughly square grid
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    return (rows, cols)


def display_forensic_analysis(
    original_image: np.ndarray,
    srm_noise_map: np.ndarray,
    color_diff_map: np.ndarray,
    chromatic_residual_map: np.ndarray,
    patchwise_fft_map: np.ndarray,
    radial_freq_stats: np.ndarray
) -> None:
    """
    Display all forensic maps in a grid layout using matplotlib.
    
    Args:
        original_image: Original input image
        srm_noise_map: Fixed SRM high-pass filtered noise map
        color_diff_map: Color difference map between channels
        chromatic_residual_map: Chromatic residual map
        patchwise_fft_map: Patch-wise FFT log magnitude map
        radial_freq_stats: 1D radial frequency statistics
    """
    # Create 2x3 grid for 6 panels
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Display original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Display Fixed SRM noise map with 'hot' colormap
    axes[0, 1].imshow(srm_noise_map, cmap='hot')
    axes[0, 1].set_title('Fixed SRM Noise Map')
    axes[0, 1].axis('off')
    
    # Display color difference map with 'viridis' colormap
    axes[0, 2].imshow(color_diff_map, cmap='viridis')
    axes[0, 2].set_title('Color Difference Map')
    axes[0, 2].axis('off')
    
    # Display chromatic residual map with 'viridis' colormap
    axes[1, 0].imshow(chromatic_residual_map, cmap='viridis')
    axes[1, 0].set_title('Chromatic Residual Map')
    axes[1, 0].axis('off')
    
    # Display patch-wise FFT log magnitude with 'gray' colormap
    axes[1, 1].imshow(patchwise_fft_map, cmap='gray')
    axes[1, 1].set_title('Patch-wise FFT Log Magnitude')
    axes[1, 1].axis('off')
    
    # Display radial frequency statistics as 1D plot
    axes[1, 2].plot(radial_freq_stats)
    axes[1, 2].set_title('Radial Frequency Statistics')
    axes[1, 2].set_xlabel('Frequency Bin')
    axes[1, 2].set_ylabel('Average Magnitude')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
