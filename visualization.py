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
    noise_map: np.ndarray,
    color_dist_map: np.ndarray,
    color_diff_map: np.ndarray,
    frequency_map: np.ndarray
) -> None:
    """
    Display all forensic maps in a grid layout using matplotlib.
    
    Args:
        original_image: Original input image
        noise_map: High-pass residual noise map
        color_dist_map: 4-bit quantized color map
        color_diff_map: Color difference map (original - restored)
        frequency_map: FFT magnitude spectrum
    """
    # Total number of images to display
    num_images = 5
    rows, cols = create_subplot_grid(num_images)
    
    # Create figure with consistent sizing
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    
    # Flatten axes array for easier indexing
    if num_images > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Display original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display high-pass residual noise map with 'hot' colormap
    axes[1].imshow(noise_map, cmap='hot')
    axes[1].set_title('High-Pass Residual')
    axes[1].axis('off')
    
    # Display 4-bit quantized color map with 'viridis' colormap
    axes[2].imshow(color_dist_map, cmap='viridis')
    axes[2].set_title('4-Bit Quantized')
    axes[2].axis('off')
    
    # Display color difference map with 'viridis' colormap
    axes[3].imshow(color_diff_map, cmap='viridis')
    axes[3].set_title('Color Difference Map')
    axes[3].axis('off')
    
    # Display FFT magnitude spectrum with 'gray' colormap
    axes[4].imshow(frequency_map, cmap='gray')
    axes[4].set_title('FFT Magnitude Spectrum')
    axes[4].axis('off')
    
    # Hide any unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
