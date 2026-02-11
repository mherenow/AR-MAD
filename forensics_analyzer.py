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
from noise_analysis import generate_srm_noise_map
from color_analysis import generate_color_difference_map, generate_chromatic_residual_map
from frequency_analysis import generate_patchwise_fft_map, generate_radial_frequency_stats
from visualization import display_forensic_analysis


def analyze_image(
    image_path: str,
    srm_kernel_type: str = 'srm_3x3',
    patch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform complete forensic analysis on an image using updated methods.
    
    This function coordinates all forensic analysis modules to generate
    multiple forensic maps using Fixed SRM filtering, color difference analysis,
    chromatic residual analysis, and patch-wise FFT analysis.
    
    Args:
        image_path: Path to the image file to analyze
        srm_kernel_type: Type of SRM kernel for spatial analysis (default: 'srm_3x3')
        patch_size: Size of patches for FFT analysis (default: 64)
        
    Returns:
        Tuple containing:
            - original_image: The loaded original image
            - srm_noise_map: Fixed SRM high-pass filtered noise map
            - color_diff_map: Color difference map between channels
            - chromatic_residual_map: Chromatic residual map
            - patchwise_fft_map: Patch-wise FFT log magnitude map
            - radial_freq_stats: 1D radial frequency statistics
            
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image format is unsupported or parameters are invalid
        IOError: If the image file is corrupted
    """
    try:
        # Validate parameters
        if patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {patch_size}")
        
        # Load the image
        print(f"Loading image: {image_path}")
        original_image = load_image(image_path)
        print(f"Image loaded successfully. Shape: {original_image.shape}")
        
        # Generate Fixed SRM noise map
        print("Generating Fixed SRM noise map...")
        srm_noise_map = generate_srm_noise_map(original_image, kernel_type=srm_kernel_type)
        print("Fixed SRM noise map generated.")
        
        # Generate color difference map (only for RGB images)
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            print("Generating color difference map...")
            color_diff_map = generate_color_difference_map(original_image)
            print("Color difference map generated.")
            
            # Generate chromatic residual map (only for RGB images)
            print("Generating chromatic residual map...")
            chromatic_residual_map = generate_chromatic_residual_map(original_image)
            print("Chromatic residual map generated.")
        else:
            # For non-RGB images, create placeholder maps
            print("Skipping color analysis (not RGB image)...")
            color_diff_map = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.float32)
            chromatic_residual_map = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.float32)
        
        # Generate patch-wise FFT log magnitude map
        print("Generating patch-wise FFT log magnitude map...")
        patchwise_fft_map = generate_patchwise_fft_map(original_image, patch_size=patch_size)
        print("Patch-wise FFT map generated.")
        
        # Generate radial frequency statistics
        print("Generating radial frequency statistics...")
        radial_freq_stats = generate_radial_frequency_stats(original_image, patch_size=patch_size)
        print("Radial frequency statistics generated.")
        
        return (original_image, srm_noise_map, color_diff_map, 
                chromatic_residual_map, patchwise_fft_map, radial_freq_stats)
        
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
                    'Implements: Fixed SRM High-Pass Filtering, Color Difference Analysis, '
                    'Chromatic Residual Analysis, and Patch-wise FFT Analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file to analyze'
    )
    parser.add_argument(
        '--srm-kernel',
        type=str,
        default='srm_3x3',
        choices=['srm_3x3', 'spam', 'edge', 'srm_5x5'],
        help='Type of SRM kernel for spatial analysis (default: srm_3x3)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=64,
        help='Size of patches for FFT analysis (default: 64, must be positive)'
    )
    
    args = parser.parse_args()
    
    try:
        # Run the complete forensic analysis
        (original_image, srm_noise_map, color_diff_map, 
         chromatic_residual_map, patchwise_fft_map, radial_freq_stats) = analyze_image(
            args.image_path,
            srm_kernel_type=args.srm_kernel,
            patch_size=args.patch_size
        )
        
        # Display all forensic maps
        print("\nDisplaying forensic analysis results...")
        display_forensic_analysis(
            original_image,
            srm_noise_map,
            color_diff_map,
            chromatic_residual_map,
            patchwise_fft_map,
            radial_freq_stats
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
