"""Unit tests for SCV computer."""

import pytest
import torch
from .scv_computer import SCVComputer


class TestSCVComputer:
    """Test suite for SCVComputer."""
    
    def test_initialization(self):
        """Test SCVComputer initialization with default parameters."""
        computer = SCVComputer()
        assert computer.embed_dim == 256
        assert computer.num_bands == 4
        assert computer.consistency_dim == 128
    
    def test_initialization_custom_params(self):
        """Test SCVComputer initialization with custom parameters."""
        computer = SCVComputer(embed_dim=512, num_bands=8, consistency_dim=256)
        assert computer.embed_dim == 512
        assert computer.num_bands == 8
        assert computer.consistency_dim == 256
    
    def test_raw_feature_dim_calculation(self):
        """Test that raw feature dimension is calculated correctly."""
        computer = SCVComputer(num_bands=4)
        # variance: 4, correlation: 3, energy: 4 -> total: 11
        assert computer.raw_feature_dim == 4 + 3 + 4
        
        computer = SCVComputer(num_bands=8)
        # variance: 8, correlation: 7, energy: 8 -> total: 23
        assert computer.raw_feature_dim == 8 + 7 + 8
    
    def test_forward_square_layout(self):
        """Test forward pass with square patch layout."""
        computer = SCVComputer(embed_dim=256, num_bands=4, consistency_dim=128)
        
        # Create input: 16x16 = 256 patches
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        # Forward pass
        scv = computer(tokens)
        
        # Check output shape
        assert scv.shape == (batch_size, 128)
    
    def test_forward_with_spatial_dims(self):
        """Test forward pass with explicit spatial dimensions."""
        computer = SCVComputer(embed_dim=256, num_bands=4, consistency_dim=128)
        
        # Create input: 8x16 = 128 patches (non-square)
        batch_size = 2
        height, width = 8, 16
        num_patches = height * width
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        # Forward pass with spatial_dims
        scv = computer(tokens, spatial_dims=(height, width))
        
        # Check output shape
        assert scv.shape == (batch_size, 128)
    
    def test_forward_mismatched_spatial_dims(self):
        """Test that mismatched spatial_dims raises ValueError."""
        computer = SCVComputer()
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        # Provide incorrect spatial_dims
        with pytest.raises(ValueError, match="doesn't match"):
            computer(tokens, spatial_dims=(10, 10))
    
    def test_forward_non_square_without_spatial_dims(self):
        """Test that non-square layout without spatial_dims raises ValueError."""
        computer = SCVComputer()
        
        batch_size = 2
        num_patches = 128  # Not a perfect square
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        with pytest.raises(ValueError, match="Cannot infer square spatial layout"):
            computer(tokens)
    
    def test_variance_computation(self):
        """Test variance computation for frequency bands."""
        computer = SCVComputer(embed_dim=256, num_bands=4)
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        scv = computer(tokens)
        
        # Check output is valid (no NaN or Inf)
        assert not torch.isnan(scv).any()
        assert not torch.isinf(scv).any()
    
    def test_energy_computation(self):
        """Test energy computation for frequency bands."""
        computer = SCVComputer(embed_dim=256, num_bands=4)
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        scv = computer(tokens)
        
        # Energy should be non-negative (after projection, this may not hold)
        # But output should be valid
        assert not torch.isnan(scv).any()
        assert not torch.isinf(scv).any()
    
    def test_correlation_computation(self):
        """Test correlation computation between adjacent bands."""
        computer = SCVComputer(embed_dim=256, num_bands=4)
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        scv = computer(tokens)
        
        # Correlation should be in [-1, 1] before projection
        # After projection, just check validity
        assert not torch.isnan(scv).any()
        assert not torch.isinf(scv).any()
    
    def test_different_batch_sizes(self):
        """Test that computer works with different batch sizes."""
        computer = SCVComputer(consistency_dim=128)
        
        for batch_size in [1, 2, 4, 8]:
            tokens = torch.randn(batch_size, 256, 256)
            scv = computer(tokens)
            assert scv.shape == (batch_size, 128)
    
    def test_different_num_bands(self):
        """Test computer with different numbers of frequency bands."""
        for num_bands in [2, 4, 8]:
            computer = SCVComputer(num_bands=num_bands, consistency_dim=128)
            tokens = torch.randn(2, 256, 256)
            scv = computer(tokens)
            assert scv.shape == (2, 128)
    
    def test_different_consistency_dims(self):
        """Test computer with different output dimensions."""
        for consistency_dim in [64, 128, 256]:
            computer = SCVComputer(consistency_dim=consistency_dim)
            tokens = torch.randn(2, 256, 256)
            scv = computer(tokens)
            assert scv.shape == (2, consistency_dim)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the computer."""
        computer = SCVComputer()
        
        tokens = torch.randn(2, 256, 256, requires_grad=True)
        scv = computer(tokens)
        
        # Compute a simple loss and backpropagate
        loss = scv.sum()
        loss.backward()
        
        # Check that gradients exist
        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()
    
    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        computer = SCVComputer()
        computer.eval()
        
        tokens = torch.randn(2, 256, 256)
        
        scv1 = computer(tokens)
        scv2 = computer(tokens)
        
        # Outputs should be identical
        assert torch.allclose(scv1, scv2)
    
    def test_small_image(self):
        """Test with small image (few patches)."""
        computer = SCVComputer(embed_dim=256, num_bands=4, consistency_dim=128)
        
        # 4x4 = 16 patches (small)
        batch_size = 2
        num_patches = 16
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        scv = computer(tokens, spatial_dims=(4, 4))
        
        # Should still produce valid output
        assert scv.shape == (batch_size, 128)
        assert not torch.isnan(scv).any()
    
    def test_single_band(self):
        """Test with single frequency band (edge case)."""
        computer = SCVComputer(num_bands=1, consistency_dim=64)
        
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        scv = computer(tokens)
        
        # Should handle single band (no correlations)
        assert scv.shape == (batch_size, 64)
        assert not torch.isnan(scv).any()
    
    def test_consistency_properties(self):
        """Test that SCV captures consistency properties."""
        computer = SCVComputer(embed_dim=256, num_bands=4, consistency_dim=128)
        
        # Create two different token sets
        tokens1 = torch.randn(2, 256, 256)
        tokens2 = torch.randn(2, 256, 256)
        
        scv1 = computer(tokens1)
        scv2 = computer(tokens2)
        
        # SCVs should be different for different inputs
        assert not torch.allclose(scv1, scv2)
    
    def test_empty_band_handling(self):
        """Test handling of potentially empty bands with very small images."""
        computer = SCVComputer(num_bands=8, consistency_dim=64)  # Many bands
        
        # Very small image: 2x2 = 4 patches
        batch_size = 2
        num_patches = 4
        embed_dim = 256
        tokens = torch.randn(batch_size, num_patches, embed_dim)
        
        scv = computer(tokens, spatial_dims=(2, 2))
        
        # Should handle empty bands gracefully
        assert scv.shape == (batch_size, 64)
        assert not torch.isnan(scv).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
