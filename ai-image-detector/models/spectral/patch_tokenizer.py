"""Spectral patch tokenizer with ViT-style architecture."""

import torch
import torch.nn as nn
from typing import Tuple


class SpectralPatchTokenizer(nn.Module):
    """
    ViT-style patch tokenizer for frequency domain spectrograms.
    
    This module divides the frequency spectrum into non-overlapping patches,
    applies linear projection to embed each patch, and adds learnable positional
    embeddings. This is analogous to the patch embedding layer in Vision Transformers
    but operates on frequency domain representations.
    
    Args:
        patch_size: Size of each square patch (default: 16)
        embed_dim: Embedding dimension for each patch (default: 256)
        in_channels: Number of input channels (default: 3 for RGB)
    
    Example:
        >>> tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        >>> spectrum = torch.randn(2, 3, 256, 256)  # Batch of 2 spectrograms
        >>> tokens = tokenizer(spectrum)
        >>> tokens.shape
        torch.Size([2, 256, 256])  # (batch_size, num_patches, embed_dim)
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 256,
        in_channels: int = 3
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # Linear projection layer using Conv2d for efficiency
        # This is equivalent to dividing into patches and applying linear projection
        # Conv2d with kernel_size=patch_size and stride=patch_size extracts non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Learnable positional embeddings will be initialized in forward pass
        # based on the actual input size (to support variable resolutions)
        self.pos_embedding = None
        self.num_patches = None
    
    def _initialize_pos_embedding(self, num_patches_h: int, num_patches_w: int, device: torch.device):
        """
        Initialize or update positional embeddings based on input size.
        
        Args:
            num_patches_h: Number of patches along height
            num_patches_w: Number of patches along width
            device: Device to create embeddings on
        """
        num_patches = num_patches_h * num_patches_w
        
        # Only reinitialize if size changed
        if self.pos_embedding is None or self.num_patches != num_patches:
            self.num_patches = num_patches
            # Shape: (1, num_patches, embed_dim)
            self.pos_embedding = nn.Parameter(
                torch.randn(1, num_patches, self.embed_dim, device=device) * 0.02
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize frequency spectrum into patch embeddings.
        
        Args:
            x: Input frequency spectrum of shape (B, C, H, W) where:
               - B is batch size
               - C is number of channels (typically 3)
               - H is height (must be divisible by patch_size)
               - W is width (must be divisible by patch_size)
        
        Returns:
            tokens: Patch embeddings of shape (B, num_patches, embed_dim) where
                   num_patches = (H // patch_size) * (W // patch_size)
        
        Raises:
            ValueError: If H or W is not divisible by patch_size
        """
        B, C, H, W = x.shape
        
        # Validate input dimensions
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input height ({H}) and width ({W}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Initialize positional embeddings if needed
        self._initialize_pos_embedding(num_patches_h, num_patches_w, x.device)
        
        # Apply linear projection via convolution
        # Output shape: (B, embed_dim, num_patches_h, num_patches_w)
        x = self.projection(x)
        
        # Flatten spatial dimensions and transpose
        # (B, embed_dim, num_patches_h, num_patches_w) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        return x
    
    def get_num_patches(self, height: int, width: int) -> int:
        """
        Calculate the number of patches for given input dimensions.
        
        Args:
            height: Input height
            width: Input width
        
        Returns:
            num_patches: Total number of patches
        
        Raises:
            ValueError: If height or width is not divisible by patch_size
        """
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Height ({height}) and width ({width}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        
        return (height // self.patch_size) * (width // self.patch_size)
