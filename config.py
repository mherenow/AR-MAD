"""Configuration module for forensic analysis."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ForensicConfig:
    """Configuration for forensic analysis."""
    
    # Fixed SRM high-pass filtering
    srm_kernel_type: str = 'srm'  # Type of SRM kernel ('srm', 'spam', etc.)
    
    # Color difference and chromatic residual analysis
    color_diff_method: str = 'channel_diff'  # Method for color difference computation
    chromatic_residual_method: str = 'standard'  # Method for chromatic residual extraction
    
    # Patch-wise FFT analysis
    patch_size: int = 64  # Size of patches for localized FFT
    patch_overlap: float = 0.5  # Overlap ratio between patches (0.0 to 1.0)
    use_log_scaling: bool = True  # Apply log scaling for visualization
    
    # Visualization
    colormap_srm: str = 'hot'
    colormap_color: str = 'viridis'
    colormap_frequency: str = 'gray'
    figure_size: Tuple[int, int] = (18, 12)  # Larger for 6 subplots
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate SRM kernel type
        valid_srm_kernels = ['srm', 'spam', 'edge', 'custom']
        if self.srm_kernel_type not in valid_srm_kernels:
            raise ValueError(
                f"srm_kernel_type must be one of {valid_srm_kernels}, got '{self.srm_kernel_type}'"
            )
        
        # Validate color difference method
        valid_color_methods = ['channel_diff', 'rgb_diff', 'lab_diff']
        if self.color_diff_method not in valid_color_methods:
            raise ValueError(
                f"color_diff_method must be one of {valid_color_methods}, got '{self.color_diff_method}'"
            )
        
        # Validate chromatic residual method
        valid_chromatic_methods = ['standard', 'enhanced', 'adaptive']
        if self.chromatic_residual_method not in valid_chromatic_methods:
            raise ValueError(
                f"chromatic_residual_method must be one of {valid_chromatic_methods}, got '{self.chromatic_residual_method}'"
            )
        
        # Validate patch size
        if self.patch_size <= 0 or self.patch_size > 512:
            raise ValueError(
                f"patch_size must be between 1 and 512, got {self.patch_size}"
            )
        
        # Validate patch overlap
        if self.patch_overlap < 0.0 or self.patch_overlap >= 1.0:
            raise ValueError(
                f"patch_overlap must be between 0.0 and 1.0 (exclusive), got {self.patch_overlap}"
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
