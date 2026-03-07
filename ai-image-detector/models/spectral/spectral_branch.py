"""Spectral branch module integrating all spectral components."""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .fft_processor import FFTProcessor
from .frequency_masking import FrequencyMasking
from .patch_tokenizer import SpectralPatchTokenizer
from .srs_extractor import SRSExtractor
from .scv_computer import SCVComputer


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with multi-head self-attention.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder layer.
        
        Args:
            x: Input tokens (B, num_patches, embed_dim)
        
        Returns:
            output: Transformed tokens (B, num_patches, embed_dim)
        """
        # Multi-head self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        # Feed-forward network with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multiple layers.
    
    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Stack of transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            x: Input tokens (B, num_patches, embed_dim)
        
        Returns:
            output: Encoded tokens (B, num_patches, embed_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralBranch(nn.Module):
    """
    Spectral analysis branch for frequency domain feature extraction.
    
    This module integrates all spectral components to extract frequency domain
    features from images. The pipeline consists of:
    1. FFT processing to convert to frequency domain
    2. Frequency masking to filter specific bands
    3. Patch tokenization with ViT-style architecture
    4. Transformer encoding for contextual processing
    5. SRS extraction for spectral response signatures
    6. SCV computation for spectral consistency vectors
    
    Args:
        patch_size: Size of patches for tokenization (default: 16)
        embed_dim: Embedding dimension (default: 256)
        depth: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        mask_type: Type of frequency filter ('low_pass', 'high_pass', 'band_pass')
                   (default: 'high_pass')
        cutoff_freq: Cutoff frequency for masking (default: 0.3)
        num_bands: Number of frequency bands for SRS/SCV (default: 4)
        consistency_dim: Output dimension for SCV (default: 128)
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        dropout: Dropout probability (default: 0.1)
    
    Example:
        >>> spectral_branch = SpectralBranch(patch_size=16, embed_dim=256, depth=4, num_heads=8)
        >>> image = torch.randn(2, 3, 256, 256)
        >>> srs, scv = spectral_branch(image)
        >>> srs.shape, scv.shape
        (torch.Size([2, 256]), torch.Size([2, 128]))
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mask_type: str = 'high_pass',
        cutoff_freq: float = 0.3,
        num_bands: int = 4,
        consistency_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_bands = num_bands
        self.consistency_dim = consistency_dim
        
        # 1. FFT Processor
        self.fft_processor = FFTProcessor(log_scale=True)
        
        # 2. Frequency Masking
        self.frequency_masking = FrequencyMasking(
            mask_type=mask_type,
            cutoff_freq=cutoff_freq,
            preserve_dc=True
        )
        
        # 3. Spectral Patch Tokenizer
        self.patch_tokenizer = SpectralPatchTokenizer(
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=3
        )
        
        # 4. Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # 5. SRS Extractor
        self.srs_extractor = SRSExtractor(
            embed_dim=embed_dim,
            num_bands=num_bands,
            aggregation_method='mean'
        )
        
        # 6. SCV Computer
        self.scv_computer = SCVComputer(
            embed_dim=embed_dim,
            num_bands=num_bands,
            consistency_dim=consistency_dim
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through spectral branch.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W) where:
               - B is batch size
               - 3 is RGB channels
               - H is height (must be divisible by patch_size)
               - W is width (must be divisible by patch_size)
        
        Returns:
            srs: Spectral Response Signatures of shape (B, embed_dim)
            scv: Spectral Consistency Vectors of shape (B, consistency_dim)
        
        Raises:
            ValueError: If H or W is not divisible by patch_size
        
        Example:
            >>> spectral_branch = SpectralBranch()
            >>> image = torch.randn(4, 3, 256, 256)
            >>> srs, scv = spectral_branch(image)
            >>> print(f"SRS shape: {srs.shape}, SCV shape: {scv.shape}")
            SRS shape: torch.Size([4, 256]), SCV shape: torch.Size([4, 128])
        """
        B, C, H, W = x.shape
        
        # Validate input dimensions
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input height ({H}) and width ({W}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        
        # 1. Convert to frequency domain
        # (B, 3, H, W) -> (B, 3, H, W)
        magnitude_spectrum = self.fft_processor(x)
        
        # 2. Apply frequency masking
        # (B, 3, H, W) -> (B, 3, H, W)
        masked_spectrum = self.frequency_masking(magnitude_spectrum)
        
        # 3. Tokenize spectral patches
        # (B, 3, H, W) -> (B, num_patches, embed_dim)
        tokens = self.patch_tokenizer(masked_spectrum)
        
        # Calculate spatial dimensions for later use
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        spatial_dims = (num_patches_h, num_patches_w)
        
        # 4. Apply transformer encoding
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        encoded_tokens = self.transformer_encoder(tokens)
        
        # 5. Extract Spectral Response Signatures
        # (B, num_patches, embed_dim) -> (B, embed_dim)
        srs = self.srs_extractor(encoded_tokens, spatial_dims=spatial_dims)
        
        # 6. Compute Spectral Consistency Vectors
        # (B, num_patches, embed_dim) -> (B, consistency_dim)
        scv = self.scv_computer(encoded_tokens, spatial_dims=spatial_dims)
        
        return srs, scv
    
    def get_intermediate_features(
        self,
        x: torch.Tensor
    ) -> dict:
        """
        Get intermediate features for visualization and debugging.
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            features: Dictionary containing intermediate features:
                - 'magnitude_spectrum': FFT magnitude spectrum
                - 'masked_spectrum': Frequency-masked spectrum
                - 'tokens': Tokenized patches
                - 'encoded_tokens': Transformer-encoded tokens
                - 'srs': Spectral Response Signatures
                - 'scv': Spectral Consistency Vectors
        """
        B, C, H, W = x.shape
        
        # Validate input dimensions
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input height ({H}) and width ({W}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        
        features = {}
        
        # 1. FFT processing
        magnitude_spectrum = self.fft_processor(x)
        features['magnitude_spectrum'] = magnitude_spectrum
        
        # 2. Frequency masking
        masked_spectrum = self.frequency_masking(magnitude_spectrum)
        features['masked_spectrum'] = masked_spectrum
        
        # 3. Tokenization
        tokens = self.patch_tokenizer(masked_spectrum)
        features['tokens'] = tokens
        
        # Calculate spatial dimensions
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        spatial_dims = (num_patches_h, num_patches_w)
        
        # 4. Transformer encoding
        encoded_tokens = self.transformer_encoder(tokens)
        features['encoded_tokens'] = encoded_tokens
        
        # 5. SRS extraction
        srs = self.srs_extractor(encoded_tokens, spatial_dims=spatial_dims)
        features['srs'] = srs
        
        # 6. SCV computation
        scv = self.scv_computer(encoded_tokens, spatial_dims=spatial_dims)
        features['scv'] = scv
        
        return features
