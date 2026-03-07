"""Tests for masked spectral pretraining module."""

import pytest
import torch
import torch.nn as nn

from .spectral_branch import SpectralBranch
from .pretraining import MaskedSpectralPretraining


class TestMaskedSpectralPretraining:
    """Test suite for MaskedSpectralPretraining."""
    
    def test_initialization(self):
        """Test that MaskedSpectralPretraining initializes correctly."""
        spectral_branch = SpectralBranch(
            patch_size=16,
            embed_dim=256,
            depth=4,
            num_heads=8
        )
        
        pretraining_model = MaskedSpectralPretraining(
            spectral_branch=spectral_branch,
            decoder_embed_dim=128,
            decoder_depth=2,
            mask_ratio=0.75
        )
        
        assert pretraining_model.spectral_branch is spectral_branch
        assert pretraining_model.decoder_embed_dim == 128
        assert pretraining_model.decoder_depth == 2
        assert pretraining_model.mask_ratio == 0.75
        assert pretraining_model.patch_size == 16
        assert pretraining_model.embed_dim == 256
    
    def test_invalid_mask_ratio(self):
        """Test that invalid mask ratios raise ValueError."""
        spectral_branch = SpectralBranch()
        
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            MaskedSpectralPretraining(spectral_branch, mask_ratio=0.0)
        
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            MaskedSpectralPretraining(spectral_branch, mask_ratio=1.0)
        
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            MaskedSpectralPretraining(spectral_branch, mask_ratio=1.5)
    
    def test_forward_pass(self):
        """Test forward pass produces expected outputs."""
        spectral_branch = SpectralBranch(
            patch_size=16,
            embed_dim=256,
            depth=4,
            num_heads=8
        )
        
        pretraining_model = MaskedSpectralPretraining(
            spectral_branch=spectral_branch,
            decoder_embed_dim=128,
            decoder_depth=2,
            mask_ratio=0.75
        )
        
        # Create input
        batch_size = 2
        image = torch.randn(batch_size, 3, 256, 256)
        
        # Forward pass
        loss, pred, mask = pretraining_model(image)
        
        # Check outputs
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
        
        num_patches = (256 // 16) ** 2  # 256 patches for 256x256 image with patch_size=16
        patch_pixels = 16 * 16 * 3  # 768 pixels per patch
        
        assert pred.shape == (batch_size, num_patches, patch_pixels)
        assert mask.shape == (batch_size, num_patches)
        
        # Check mask ratio is approximately correct
        mask_ratio = mask.float().mean().item()
        assert 0.7 < mask_ratio < 0.8  # Should be close to 0.75
    
    def test_random_masking(self):
        """Test random masking produces correct outputs."""
        spectral_branch = SpectralBranch()
        pretraining_model = MaskedSpectralPretraining(spectral_branch, mask_ratio=0.75)
        
        # Create input tokens
        batch_size = 2
        num_patches = 256
        embed_dim = 256
        x = torch.randn(batch_size, num_patches, embed_dim)
        
        # Apply masking
        x_masked, mask, ids_restore = pretraining_model.random_masking(x, mask_ratio=0.75)
        
        # Check shapes
        num_keep = int(num_patches * 0.25)
        assert x_masked.shape == (batch_size, num_keep, embed_dim)
        assert mask.shape == (batch_size, num_patches)
        assert ids_restore.shape == (batch_size, num_patches)
        
        # Check mask values
        assert torch.all((mask == 0) | (mask == 1))
        
        # Check mask ratio
        mask_ratio = mask.float().mean().item()
        assert 0.7 < mask_ratio < 0.8
    
    def test_patchify_unpatchify(self):
        """Test patchify and unpatchify are inverse operations."""
        spectral_branch = SpectralBranch(patch_size=16)
        pretraining_model = MaskedSpectralPretraining(spectral_branch)
        
        # Create input
        batch_size = 2
        image = torch.randn(batch_size, 3, 256, 256)
        
        # Patchify
        patches = pretraining_model._patchify(image)
        
        # Check shape
        num_patches = (256 // 16) ** 2
        patch_pixels = 16 * 16 * 3
        assert patches.shape == (batch_size, num_patches, patch_pixels)
        
        # Unpatchify
        reconstructed = pretraining_model._unpatchify(patches, h=16, w=16)
        
        # Check shape
        assert reconstructed.shape == image.shape
        
        # Check values are close (should be exact for this operation)
        assert torch.allclose(reconstructed, image, atol=1e-6)
    
    def test_different_image_sizes(self):
        """Test pretraining works with different valid image sizes."""
        spectral_branch = SpectralBranch(patch_size=16)
        pretraining_model = MaskedSpectralPretraining(spectral_branch)
        
        # Test different sizes (all divisible by 16)
        sizes = [128, 256, 512]
        
        for size in sizes:
            image = torch.randn(1, 3, size, size)
            loss, pred, mask = pretraining_model(image)
            
            num_patches = (size // 16) ** 2
            patch_pixels = 16 * 16 * 3
            
            assert pred.shape == (1, num_patches, patch_pixels)
            assert mask.shape == (1, num_patches)
            assert loss.item() >= 0
    
    def test_invalid_image_size(self):
        """Test that images not divisible by patch_size raise ValueError."""
        spectral_branch = SpectralBranch(patch_size=16)
        pretraining_model = MaskedSpectralPretraining(spectral_branch)
        
        # Image size not divisible by patch_size
        image = torch.randn(1, 3, 255, 255)
        
        with pytest.raises(ValueError, match="must be divisible by"):
            pretraining_model(image)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the model."""
        spectral_branch = SpectralBranch(
            patch_size=16,
            embed_dim=128,  # Smaller for faster test
            depth=2,
            num_heads=4
        )
        
        pretraining_model = MaskedSpectralPretraining(
            spectral_branch=spectral_branch,
            decoder_embed_dim=64,
            decoder_depth=1
        )
        
        # Create input
        image = torch.randn(1, 3, 128, 128, requires_grad=True)
        
        # Forward pass
        loss, pred, mask = pretraining_model(image)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert image.grad is not None
        assert not torch.all(image.grad == 0)
        
        # Check that decoder parameters have gradients
        # Note: SRS and SCV extractors are not used in pretraining forward pass
        decoder_params = [
            'decoder_embed',
            'mask_token',
            'decoder_blocks',
            'decoder_norm',
            'decoder_pred'
        ]
        
        for name, param in pretraining_model.named_parameters():
            if param.requires_grad and any(dp in name for dp in decoder_params):
                assert param.grad is not None, f"No gradient for {name}"
        
        # Check that encoder (transformer) has gradients
        for name, param in pretraining_model.spectral_branch.transformer_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for encoder {name}"
    
    def test_decoder_positional_embeddings(self):
        """Test decoder positional embeddings are created correctly."""
        spectral_branch = SpectralBranch(patch_size=16)
        pretraining_model = MaskedSpectralPretraining(
            spectral_branch,
            decoder_embed_dim=128
        )
        
        # Get positional embeddings
        device = torch.device('cpu')
        pos_embed = pretraining_model._get_decoder_pos_embed(16, 16, device)
        
        # Check shape
        assert pos_embed.shape == (1, 256, 128)  # (1, 16*16, decoder_embed_dim)
        
        # Check it's cached
        pos_embed2 = pretraining_model._get_decoder_pos_embed(16, 16, device)
        assert pos_embed is pos_embed2  # Same object
    
    def test_mask_token_initialization(self):
        """Test mask token is initialized correctly."""
        spectral_branch = SpectralBranch()
        pretraining_model = MaskedSpectralPretraining(
            spectral_branch,
            decoder_embed_dim=128
        )
        
        # Check mask token exists and has correct shape
        assert hasattr(pretraining_model, 'mask_token')
        assert pretraining_model.mask_token.shape == (1, 1, 128)
        assert isinstance(pretraining_model.mask_token, nn.Parameter)
        
        # Check it's not all zeros (should be initialized)
        assert not torch.all(pretraining_model.mask_token == 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
