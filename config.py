"""Configuration module for forensic analysis."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ForensicConfig:
    """Configuration for forensic analysis."""
    
    # High-pass residual noise analysis (NIRNet approach)
    noise_filter_size: int = 5  # Size of high-pass filter kernel
    
    # 4-bit quantized color analysis (CVPR 2025 method)
    quantization_bits: int = 4  # Bits per channel (4-bit = 16 levels)
    gaussian_sigma: float = 1.0  # Sigma for Gaussian restoration
    
    # FFT magnitude spectrum analysis
    use_log_scaling: bool = True  # Apply log scaling for visualization
    
    # Visualization
    colormap_noise: str = 'hot'
    colormap_color: str = 'viridis'
    colormap_frequency: str = 'gray'
    figure_size: Tuple[int, int] = (15, 10)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate noise filter size
        if self.noise_filter_size <= 0:
            raise ValueError(
                f"noise_filter_size must be positive, got {self.noise_filter_size}"
            )
        
        # Validate quantization bits
        if self.quantization_bits <= 0 or self.quantization_bits > 8:
            raise ValueError(
                f"quantization_bits must be between 1 and 8, got {self.quantization_bits}"
            )
        
        # Validate gaussian sigma
        if self.gaussian_sigma <= 0:
            raise ValueError(
                f"gaussian_sigma must be positive, got {self.gaussian_sigma}"
            )
        
        # Validate use_log_scaling is boolean (type checking)
        if not isinstance(self.use_log_scaling, bool):
            raise ValueError(
                f"use_log_scaling must be a boolean, got {type(self.use_log_scaling).__name__}"
            )
        
        # Validate figure size
        if len(self.figure_size) != 2:
            raise ValueError(
                f"figure_size must be a tuple of 2 integers, got {self.figure_size}"
            )
        if self.figure_size[0] <= 0 or self.figure_size[1] <= 0:
            raise ValueError(
                f"figure_size dimensions must be positive, got {self.figure_size}"
            )
