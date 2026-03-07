"""Unit tests for SpectralBranch module."""

import pytest
import torch
from .spectral_branch import SpectralBranch


class TestSpectralBranch:
    """Test suite for SpectralBranch module."""
    
    def test_initialization(self):
        """Test that SpectralBranch initializes correctly with default parameters."""
        branch = SpectralBranch()
        
        assert branch.patch_size == 16
        assert branch.embed_dim == 256
        assert branch.depth == 4
        assert branch.num_heads == 8
        assert branch.num_bands == 4
        assert branch.consistency_dim == 128
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        branch = SpectralBranch(
            patch_size=32,
            embed_dim=512,
            depth=6,
            num_heads=16,
            num_bands=8,
            consistency_dim=256
        )
        
        assert branch.patch_size == 32
        assert branch.embed_dim == 512
        assert branch.depth == 6
        assert branch.num_heads == 16
        assert branch.num_bands == 8
        assert branch.consistency_dim == 256
    
    def test_forward_pass_basic(self):
        """Test basic forward pass with valid input."""
        branch = SpectralBranch(patch_size=16, embed_dim=256, depth=4, num_heads=8)
        
        # Create input image (batch_size=2, channels=3, height=256, width=256)
        x = torch.randn(2, 3, 256, 256)
        
        # Forward pass
        srs, scv = branch(x)
        
        # Check output shapes
        assert srs.shape == (2, 256), f"Expected SRS shape (2, 256), got {srs.shape}"
        assert scv.shape == (2, 128), f"Expected SCV shape (2, 128), got {scv.shape}"
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        branch = SpectralBranch()
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 256, 256)
            srs, scv = branch(x)
            
            assert srs.shape == (batch_size, 256)
            assert scv.shape == (batch_size, 128)
    
    def test_forward_pass_different_image_sizes(self):
        """Test forward pass with different image sizes (all divisible by patch_size)."""
        branch = SpectralBranch(patch_size=16)
        
        # Test various image sizes divisible by 16
        for size in [128, 256, 512]:
            x = torch.randn(2, 3, size, size)
            srs, scv = branch(x)
            
            assert srs.shape == (2, 256)
            assert scv.shape == (2, 128)
    
    def test_forward_pass_invalid_dimensions(self):
        """Test that forward pass raises error for invalid dimensions."""
        branch = SpectralBranch(patch_size=16)
        
        # Image size not divisible by patch_size
        x = torch.randn(2, 3, 255, 255)
        
        with pytest.raises(ValueError, match="must be divisible by patch_size"):
            branch(x)
    
    def test_output_types(self):
        """Test that outputs are torch tensors."""
        branch = SpectralBranch()
        x = torch.randn(2, 3, 256, 256)
        
        srs, scv = branch(x)
        
        assert isinstance(srs, torch.Tensor)
        assert isinstance(scv, torch.Tensor)
    
    def test_output_device_consistency(self):
        """Test that outputs are on the same device as inputs."""
        branch = SpectralBranch()
        x = torch.randn(2, 3, 256, 256)
        
        srs, scv = branch(x)
        
        assert srs.device == x.device
        assert scv.device == x.device
    
    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        branch = SpectralBranch()
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        
        srs, scv = branch(x)
        
        # Compute a simple loss
        loss = srs.sum() + scv.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_get_intermediate_features(self):
        """Test get_intermediate_features method."""
        branch = SpectralBranch()
        x = torch.randn(2, 3, 256, 256)
        
        features = branch.get_intermediate_features(x)
        
        # Check that all expected keys are present
        expected_keys = [
            'magnitude_spectrum',
            'masked_spectrum',
            'tokens',
            'encoded_tokens',
            'srs',
            'scv'
        ]
        
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"
        
        # Check shapes
        assert features['magnitude_spectrum'].shape == (2, 3, 256, 256)
        assert features['masked_spectrum'].shape == (2, 3, 256, 256)
        assert features['tokens'].shape == (2, 256, 256)  # (B, num_patches, embed_dim)
        assert features['encoded_tokens'].shape == (2, 256, 256)
        assert features['srs'].shape == (2, 256)
        assert features['scv'].shape == (2, 128)
    
    def test_deterministic_output(self):
        """Test that forward pass is deterministic (no randomness in inference)."""
        branch = SpectralBranch()
        branch.eval()  # Set to evaluation mode
        
        x = torch.randn(2, 3, 256, 256)
        
        # Run forward pass twice
        srs1, scv1 = branch(x)
        srs2, scv2 = branch(x)
        
        # Outputs should be identical
        assert torch.allclose(srs1, srs2, atol=1e-6)
        assert torch.allclose(scv1, scv2, atol=1e-6)
    
    def test_different_mask_types(self):
        """Test that different frequency mask types work correctly."""
        mask_types = ['low_pass', 'high_pass', 'band_pass']
        
        for mask_type in mask_types:
            branch = SpectralBranch(mask_type=mask_type)
            x = torch.randn(2, 3, 256, 256)
            
            srs, scv = branch(x)
            
            assert srs.shape == (2, 256)
            assert scv.shape == (2, 128)
    
    def test_edge_case_small_image(self):
        """Test with smallest valid image size."""
        branch = SpectralBranch(patch_size=16)
        
        # Smallest valid size: 16x16 (1 patch)
        x = torch.randn(2, 3, 16, 16)
        
        srs, scv = branch(x)
        
        assert srs.shape == (2, 256)
        assert scv.shape == (2, 128)
    
    def test_edge_case_uniform_image(self):
        """Test with uniform (constant) image."""
        branch = SpectralBranch()
        
        # Uniform image (all pixels same value)
        x = torch.ones(2, 3, 256, 256) * 0.5
        
        srs, scv = branch(x)
        
        # Should still produce valid outputs
        assert srs.shape == (2, 256)
        assert scv.shape == (2, 128)
        assert not torch.isnan(srs).any()
        assert not torch.isnan(scv).any()
    
    def test_components_exist(self):
        """Test that all required components are initialized."""
        branch = SpectralBranch()
        
        assert hasattr(branch, 'fft_processor')
        assert hasattr(branch, 'frequency_masking')
        assert hasattr(branch, 'patch_tokenizer')
        assert hasattr(branch, 'transformer_encoder')
        assert hasattr(branch, 'srs_extractor')
        assert hasattr(branch, 'scv_computer')
    
    def test_transformer_depth(self):
        """Test that transformer has correct number of layers."""
        for depth in [2, 4, 6, 8]:
            branch = SpectralBranch(depth=depth)
            assert len(branch.transformer_encoder.layers) == depth


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
