"""
Unit tests for SpectralContextAttention module.
"""

import pytest
import torch
from .context_attention import (
    SpectralContextAttention,
    PositionalEncodingInterpolator,
    PatchEmbedding,
    MultiHeadAttention
)


class TestPositionalEncodingInterpolator:
    """Tests for PositionalEncodingInterpolator."""
    
    def test_base_size_returns_base_encodings(self):
        """Test that base size returns base encodings without interpolation."""
        embed_dim = 256
        base_size = 256
        patch_size = 16
        interpolator = PositionalEncodingInterpolator(embed_dim, base_size, patch_size)
        
        pos_embed = interpolator(base_size, base_size)
        
        # Should return base encodings
        assert pos_embed.shape == interpolator.base_pos_embed.shape
        assert torch.allclose(pos_embed, interpolator.base_pos_embed)
    
    def test_different_size_interpolates(self):
        """Test that different sizes trigger interpolation."""
        embed_dim = 256
        base_size = 256
        patch_size = 16
        interpolator = PositionalEncodingInterpolator(embed_dim, base_size, patch_size)
        
        # Test with different size
        target_size = 512
        pos_embed = interpolator(target_size, target_size)
        
        # Calculate expected number of patches
        expected_patches = (target_size // patch_size) ** 2
        assert pos_embed.shape == (1, expected_patches, embed_dim)
    
    def test_non_square_interpolation(self):
        """Test interpolation with non-square dimensions."""
        embed_dim = 256
        base_size = 256
        patch_size = 16
        interpolator = PositionalEncodingInterpolator(embed_dim, base_size, patch_size)
        
        # Test with non-square size
        h, w = 256, 512
        pos_embed = interpolator(h, w)
        
        h_patches = h // patch_size
        w_patches = w // patch_size
        expected_patches = h_patches * w_patches
        assert pos_embed.shape == (1, expected_patches, embed_dim)


class TestPatchEmbedding:
    """Tests for PatchEmbedding."""
    
    def test_patch_embedding_output_shape(self):
        """Test that patch embedding produces correct output shape."""
        in_channels = 3
        embed_dim = 256
        patch_size = 16
        patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        
        # Create test input
        B, H, W = 2, 256, 256
        x = torch.randn(B, in_channels, H, W)
        
        embeddings, h_patches, w_patches = patch_embed(x)
        
        # Check output shape
        expected_patches = (H // patch_size) * (W // patch_size)
        assert embeddings.shape == (B, expected_patches, embed_dim)
        assert h_patches == H // patch_size
        assert w_patches == W // patch_size
    
    def test_patch_embedding_variable_size(self):
        """Test patch embedding with variable input sizes."""
        patch_embed = PatchEmbedding(3, 256, 16)
        
        # Test different sizes
        sizes = [(256, 256), (512, 512), (256, 512), (128, 256)]
        
        for H, W in sizes:
            x = torch.randn(1, 3, H, W)
            embeddings, h_patches, w_patches = patch_embed(x)
            
            expected_patches = (H // 16) * (W // 16)
            assert embeddings.shape == (1, expected_patches, 256)
            assert h_patches == H // 16
            assert w_patches == W // 16


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""
    
    def test_attention_output_shape(self):
        """Test that attention produces correct output shape."""
        embed_dim = 256
        num_heads = 8
        attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Create test input
        B, N = 2, 64
        x = torch.randn(B, N, embed_dim)
        
        output = attention(x)
        
        # Output should have same shape as input
        assert output.shape == x.shape
    
    def test_attention_with_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        attention = MultiHeadAttention(256, 8)
        
        # Test different sequence lengths
        for N in [16, 64, 256]:
            x = torch.randn(2, N, 256)
            output = attention(x)
            assert output.shape == (2, N, 256)
    
    def test_embed_dim_divisible_by_num_heads(self):
        """Test that embed_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(embed_dim=256, num_heads=7)


class TestSpectralContextAttention:
    """Tests for SpectralContextAttention."""
    
    def test_forward_pass_base_size(self):
        """Test forward pass with base size image."""
        model = SpectralContextAttention(
            embed_dim=256,
            num_heads=8,
            base_size=256
        )
        
        # Create test input at base size
        B = 2
        x = torch.randn(B, 3, 256, 256)
        
        features = model(x)
        
        # Check output shape
        expected_h = 256 // 16
        expected_w = 256 // 16
        assert features.shape == (B, 256, expected_h, expected_w)
    
    def test_forward_pass_variable_resolution(self):
        """Test forward pass with variable resolution images."""
        model = SpectralContextAttention(
            embed_dim=256,
            num_heads=8,
            base_size=256
        )
        
        # Test different resolutions
        test_sizes = [(256, 256), (512, 512), (256, 512), (128, 256)]
        
        for H, W in test_sizes:
            x = torch.randn(1, 3, H, W)
            features = model(x)
            
            expected_h = H // 16
            expected_w = W // 16
            assert features.shape == (1, 256, expected_h, expected_w)
    
    def test_batch_processing(self):
        """Test that model handles batches correctly."""
        model = SpectralContextAttention(embed_dim=256, num_heads=8)
        
        # Test different batch sizes
        for B in [1, 2, 4, 8]:
            x = torch.randn(B, 3, 256, 256)
            features = model(x)
            assert features.shape[0] == B
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SpectralContextAttention(embed_dim=256, num_heads=8)
        
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        features = model(x)
        
        # Compute loss and backward
        loss = features.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_custom_parameters(self):
        """Test model with custom parameters."""
        model = SpectralContextAttention(
            embed_dim=128,
            num_heads=4,
            base_size=128,
            in_channels=3,
            patch_size=8,
            dropout=0.1
        )
        
        x = torch.randn(2, 3, 128, 128)
        features = model(x)
        
        expected_h = 128 // 8
        expected_w = 128 // 8
        assert features.shape == (2, 128, expected_h, expected_w)
    
    def test_deterministic_output(self):
        """Test that model produces deterministic output in eval mode."""
        model = SpectralContextAttention(embed_dim=256, num_heads=8, dropout=0.0)
        model.eval()
        
        x = torch.randn(1, 3, 256, 256)
        
        # Run twice
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)
