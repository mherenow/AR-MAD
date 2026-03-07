"""
SpectralContextAttention module for any-resolution image processing.

This module implements attention mechanisms with interpolated positional encodings
to handle images of arbitrary resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PositionalEncodingInterpolator(nn.Module):
    """
    Positional encoding with interpolation support for variable resolutions.
    
    Stores base positional encodings and interpolates them for different image sizes
    to maintain relative position information.
    
    Args:
        embed_dim: Embedding dimension
        base_size: Base image size for positional encodings (default: 256)
        patch_size: Size of patches (default: 16)
    """
    
    def __init__(self, embed_dim: int, base_size: int = 256, patch_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_size = base_size
        self.patch_size = patch_size
        
        # Create base positional encodings for base_size
        # Using learnable positional embeddings
        num_patches = (base_size // patch_size) ** 2
        self.base_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )
    
    def forward(self, height: int, width: int) -> torch.Tensor:
        """
        Generate positional encodings for given height and width.
        
        Args:
            height: Image height
            width: Image width
        
        Returns:
            pos_embed: Positional encodings (1, num_patches, embed_dim)
        """
        # Calculate number of patches for target size
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        
        # If size matches base size, return base encodings
        base_patches = self.base_size // self.patch_size
        if h_patches == base_patches and w_patches == base_patches:
            return self.base_pos_embed
        
        # Reshape base encodings to 2D grid
        base_pos_2d = self.base_pos_embed.reshape(
            1, base_patches, base_patches, self.embed_dim
        ).permute(0, 3, 1, 2)  # (1, embed_dim, base_patches, base_patches)
        
        # Interpolate to target size using bilinear interpolation
        pos_2d = F.interpolate(
            base_pos_2d,
            size=(h_patches, w_patches),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape back to sequence format
        pos_embed = pos_2d.permute(0, 2, 3, 1).reshape(
            1, h_patches * w_patches, self.embed_dim
        )
        
        return pos_embed


class PatchEmbedding(nn.Module):
    """
    Converts image to patch embeddings with adaptive patch size.
    
    Args:
        in_channels: Number of input channels (default: 3)
        embed_dim: Embedding dimension (default: 256)
        patch_size: Size of patches (default: 16)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: int = 16
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Convolutional projection for patch embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Convert image to patch embeddings.
        
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            embeddings: Patch embeddings (B, num_patches, embed_dim)
            h_patches: Number of patches in height
            w_patches: Number of patches in width
        """
        B, C, H, W = x.shape
        
        # Apply convolutional projection
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        
        # Get patch dimensions
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        # Reshape to sequence format
        embeddings = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return embeddings, h_patches, w_patches


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor (B, N, embed_dim)
        
        Returns:
            output: Attention output (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        
        # Output projection
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class SpectralContextAttention(nn.Module):
    """
    Attention module with interpolated positional encodings for any resolution.
    
    This module processes images of arbitrary resolution using patch embeddings,
    interpolated positional encodings, and multi-head self-attention.
    
    Args:
        embed_dim: Embedding dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        base_size: Base image size for positional encodings (default: 256)
        in_channels: Number of input channels (default: 3)
        patch_size: Size of patches (default: 16)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        base_size: int = 256,
        in_channels: int = 3,
        patch_size: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.base_size = base_size
        self.patch_size = patch_size
        
        # Components
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        self.pos_interpolator = PositionalEncodingInterpolator(
            embed_dim=embed_dim,
            base_size=base_size,
            patch_size=patch_size
        )
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image of any resolution through attention mechanism.
        
        Args:
            x: Input image of any resolution (B, 3, H, W)
        
        Returns:
            features: Attention-weighted features (B, embed_dim, H', W')
                     where H' = H // patch_size, W' = W // patch_size
        """
        B, C, H, W = x.shape
        
        # Convert to patch embeddings
        embeddings, h_patches, w_patches = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add interpolated positional encodings
        pos_embed = self.pos_interpolator(H, W)  # (1, N, embed_dim)
        embeddings = embeddings + pos_embed
        
        # Apply attention with residual connection
        attn_out = self.attention(self.norm1(embeddings))
        embeddings = embeddings + attn_out
        
        # Apply feed-forward network with residual connection
        mlp_out = self.mlp(self.norm2(embeddings))
        embeddings = embeddings + mlp_out
        
        # Reshape back to spatial format
        features = embeddings.transpose(1, 2).reshape(
            B, self.embed_dim, h_patches, w_patches
        )
        
        return features
