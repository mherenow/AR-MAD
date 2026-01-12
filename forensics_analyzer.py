"""
Image Forensics Analyzer

This module provides forensic analysis capabilities for detecting potential
image manipulation through multiple techniques including noise analysis,
color distribution analysis, and frequency domain analysis.
"""

import argparse
from typing import Optional, Tuple
import numpy as np

from image_loader import load_image, get_supported_formats
from noise_analysis import generate_noise_map
from color_analysis import generate_color_distribution_map, generate_color_difference_map
from frequency_analysis import generate_frequency_map
from visualization import display_forensic_analysis


def analyze_image(
    image_path: str,
    filter_size: int = 5,
    gaussian_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform complete forensic analysis on an image.
    
    This function coordinates all forensic analysis modules to generate
    multiple forensic maps that can help identify potential image manipulation.
    
    Args:
        image_path: Path to the image file to analyze
        filter_size: Size of high-pass filter kernel for noise analysis (default: 5)
        gaussian_sigma: Sigma for Gaussian restoration in color analysis (default: 1.0)
        
    Returns:
        Tuple containing:
            - original_image: The loaded original image
            - noise_map: High-pass residual noise map
            - color_dist_map: 4-bit quantized color distribution map
            - color_diff_map: Color difference map
            - frequency_map: FFT magnitude spectrum
            
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image format is unsupported or parameters are invalid
        IOError: If the image file is corrupted
    """
    try:
        # Validate parameters
        if filter_size <= 0:
            raise ValueError(f"Filter size must be positive, got {filter_size}")
        
        if gaussian_sigma <= 0:
            raise ValueError(f"Gaussian sigma must be positive, got {gaussian_sigma}")
        
        # Load the image
        print(f"Loading image: {image_path}")
        original_image = load_image(image_path)
        print(f"Image loaded successfully. Shape: {original_image.shape}")
        
        # Generate high-pass residual noise map (NIRNet approach)
        print("Generating high-pass residual noise map...")
        noise_map = generate_noise_map(original_image, filter_size=filter_size)
        print("Noise map generated.")
        
        # Generate 4-bit quantized color distribution map (CVPR 2025 method)
        print("Generating 4-bit quantized color distribution map...")
        color_dist_map = generate_color_distribution_map(original_image)
        print("Color distribution map generated.")
        
        # Generate color difference map
        print("Generating color difference map...")
        color_diff_map = generate_color_difference_map(original_image)
        print("Color difference map generated.")
        
        # Generate FFT magnitude spectrum
        print("Generating FFT magnitude spectrum...")
        frequency_map = generate_frequency_map(original_image)
        print("Frequency domain analysis complete.")
        
        return original_image, noise_map, color_dist_map, color_diff_map, frequency_map
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except IOError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during forensic analysis: {e}")
        raise


def main():
    """Main entry point for the forensics analyzer CLI."""
    parser = argparse.ArgumentParser(
        description='Analyze images for potential AI generation or manipulation using forensic techniques.\n'
                    'Implements: High-Pass Residual Noise Analysis (NIRNet), '
                    '4-Bit Quantized Color Distribution (CVPR 2025), and FFT Magnitude Spectrum Analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file to analyze'
    )
    parser.add_argument(
        '--filter-size',
        type=int,
        default=5,
        help='Size of high-pass filter kernel for noise analysis (default: 5, must be positive)'
    )
    parser.add_argument(
        '--gaussian-sigma',
        type=float,
        default=1.0,
        help='Sigma for Gaussian restoration in color analysis (default: 1.0, must be positive)'
    )
    
    args = parser.parse_args()
    
    try:
        # Run the complete forensic analysis
        original_image, noise_map, color_dist_map, color_diff_map, frequency_map = analyze_image(
            args.image_path,
            filter_size=args.filter_size,
            gaussian_sigma=args.gaussian_sigma
        )
        
        # Display all forensic maps
        print("\nDisplaying forensic analysis results...")
        display_forensic_analysis(
            original_image,
            noise_map,
            color_dist_map,
            color_diff_map,
            frequency_map
        )
        
        print("\nAnalysis complete!")
        
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"\nAnalysis failed: {e}")
        print(f"\nSupported formats: {', '.join(get_supported_formats())}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    main()
