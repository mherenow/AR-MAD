"""Unit tests for SRS extractor."""

import pytest
import torch
from .srs_extractor import SRSExtractor


class TestSRSExtractor:
    """Test suite for SRSExtractor."""
    
    def test_initialization(self):
        """Test SRSExtractor initialization with default parameters."""
        extractor = SRSExtractor()
        assert extractor.embed_dim == 256
        assert extractor.num_bands == 4
        assert extractor.aggregation_method == 'mean'
    
    def test_initialization_custom_params(self):
        """Test SRSExtractor initialization with custom parameters."""
        extractor = SRSExtractor(embed_dim=512, num_bands=8, aggregation_method='max')
        assert extractor.embed_dim == 512
        assert extractor.num_bands == 8
        assert extractor.aggregation_method == 'max'
    
    def test_invalid_aggregation_method(self):
        """Test that invalid aggregation method raises ValueError."""
        with pytest.raises(ValueError, match="aggregation_method must be one of"):
            SRSExtractor(aggregation_method='invalid')
    
    def test_forward_square_layout(self):
        """Test forward pass with square patch layout."""
        extractor = SRSExtractor(embed_dim=256, num_bands=4)
        
        # Create input: 16x16 = 256 patches
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        # Forward pass
        srs = extractor(tokens)
        
        # Check output shape
        assert srs.shape == (batch_size, embed_dim)
    
    def test_forward_with_spatial_dims(self):
        """Test forward pass with explicit spatial dimensions."""
        extractor = SRSExtractor(embed_dim=256, num_bands=4)
        
        # Create input: 8x16 = 128 patches (non-square)
        batch_size = 2
        height, width = 8, 16
        num_patches = height * width
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        # Forward pass with spatial_dims
        srs = extractor(tokens, spatial_dims=(height, width))
        
        # Check output shape
        assert srs.shape == (batch_size, embed_dim)
    
    def test_forward_mismatched_spatial_dims(self):
        """Test that mismatched spatial_dims raises ValueError."""
        extractor = SRSExtractor()
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        # Provide incorrect spatial_dims
        with pytest.raises(ValueError, match="doesn't match"):
            extractor(tokens, spatial_dims=(10, 10))  # 10*10 = 100 != 256
    
    def test_forward_non_square_without_spatial_dims(self):
        """Test that non-square layout without spatial_dims raises ValueError."""
        extractor = SRSExtractor()
        
        batch_size = 2
        num_patches = 128  # Not a perfect square
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        with pytest.raises(ValueError, match="Cannot infer square spatial layout"):
            extractor(tokens)
    
    def test_aggregation_mean(self):
        """Test mean aggregation method."""
        extractor = SRSExtractor(embed_dim=256, aggregation_method='mean')
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        srs = extractor(tokens)
        
        # Check output is valid
        assert srs.shape == (batch_size, embed_dim)
        assert not torch.isnan(srs).any()
        assert not torch.isinf(srs).any()
    
    def test_aggregation_max(self):
        """Test max aggregation method."""
        extractor = SRSExtractor(embed_dim=256, aggregation_method='max')
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        srs = extractor(tokens)
        
        # Check output is valid
        assert srs.shape == (batch_size, embed_dim)
        assert not torch.isnan(srs).any()
        assert not torch.isinf(srs).any()
    
    def test_aggregation_attention(self):
        """Test attention-based aggregation method."""
        extractor = SRSExtractor(embed_dim=256, aggregation_method='attention')
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        srs = extractor(tokens)
        
        # Check output is valid
        assert srs.shape == (batch_size, embed_dim)
        assert not torch.isnan(srs).any()
        assert not torch.isinf(srs).any()
    
    def test_different_batch_sizes(self):
        """Test that extractor works with different batch sizes."""
        extractor = SRSExtractor()
        
        for batch_size in [1, 2, 4, 8]:
            tokens = torch.randn(batch_size, 256, 256)
            srs = extractor(tokens)
            assert srs.shape == (batch_size, 256)
    
    def test_different_num_bands(self):
        """Test extractor with different numbers of frequency bands."""
        for num_bands in [2, 4, 8]:
            extractor = SRSExtractor(num_bands=num_bands)
            tokens = torch.randn(2, 256, 256)
            srs = extractor(tokens)
            assert srs.shape == (2, 256)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the extractor."""
        extractor = SRSExtractor(aggregation_method='attention')
        
        tokens = torch.randn(2, 256, 256, requires_grad=True)
        srs = extractor(tokens)
        
        # Compute a simple loss and backpropagate
        loss = srs.sum()
        loss.backward()
        
        # Check that gradients exist
        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()
    
    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        extractor = SRSExtractor(aggregation_method='mean')
        extractor.eval()
        
        tokens = torch.randn(2, 256, 256)
        
        srs1 = extractor(tokens)
        srs2 = extractor(tokens)
        
        # Outputs should be identical
        assert torch.allclose(srs1, srs2)
    
    def test_small_image(self):
        """Test with small image (few patches)."""
        extractor = SRSExtractor(embed_dim=256, num_bands=4)
        
        # 4x4 = 16 patches (small)
        batch_size = 2
        num_patches = 16
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        srs = extractor(tokens, spatial_dims=(4, 4))
        
        # Should still produce valid output
        assert srs.shape == (batch_size, embed_dim)
        assert not torch.isnan(srs).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
