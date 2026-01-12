"""
Image Loader Module

Handles loading and validation of image files for forensic analysis.
"""

import os
from typing import List
import numpy as np
import cv2


# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


def get_supported_formats() -> List[str]:
    """
    Return list of supported image formats.
    
    Returns:
        List of supported file extensions (e.g., ['.jpg', '.png', '.bmp'])
    """
    return SUPPORTED_FORMATS.copy()


def load_image(file_path: str) -> np.ndarray:
    """
    Load an image from the specified file path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        numpy array containing the image data in RGB format
        
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image format is unsupported
        IOError: If the image file is corrupted
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Check if file has a supported extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in SUPPORTED_FORMATS:
        supported_str = ', '.join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Unsupported image format: {ext}. "
            f"Supported formats are: {supported_str}"
        )
    
    # Try to load the image
    try:
        # Load image using OpenCV
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # Check if image was loaded successfully
        if image is None:
            raise IOError(f"Failed to load image. The file may be corrupted: {file_path}")
        
        # Convert BGR to RGB if image is color
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                # BGRA to RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        
        return image
        
    except cv2.error as e:
        raise IOError(f"Failed to load image. The file may be corrupted: {file_path}. Error: {str(e)}")
    except Exception as e:
        # Catch any other unexpected errors
        if isinstance(e, (FileNotFoundError, ValueError, IOError)):
            raise
        raise IOError(f"Unexpected error loading image: {file_path}. Error: {str(e)}")
