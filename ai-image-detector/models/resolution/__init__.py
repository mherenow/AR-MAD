"""
Resolution module for any-resolution image processing.
"""

from .context_attention import (
    SpectralContextAttention,
    PositionalEncodingInterpolator,
    PatchEmbedding,
    MultiHeadAttention
)
from .any_resolution_wrapper import AnyResolutionWrapper

__all__ = [
    'SpectralContextAttention',
    'PositionalEncodingInterpolator',
    'PatchEmbedding',
    'MultiHeadAttention',
    'AnyResolutionWrapper'
]
