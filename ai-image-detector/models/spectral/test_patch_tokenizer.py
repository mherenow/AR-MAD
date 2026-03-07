"""Unit tests for SpectralPatchTokenizer."""

import pytest
import torch
from .patch_tokenizer import SpectralPatchTokenizer


class TestSpectralPatchTokenizer:
    """Test suite for SpectralPatchTokenizer."""
    
    def test_initialization(self):
        """Test tokenizer initialization with default parameters."""
        tokenizer = SpectralPatchTokenizer()
        assert tokenizer.patch_size == 16
        assert tokenizer.embed_dim == 256
        assert tokenizer.in_channels == 3
    
    def test_initialization_custom_params(self):
        """Test tokenizer initialization with custom parameters."""
        tokenizer = SpectralPatchTokenizer(patch_size=8, embed_dim=128, in_channels=1)
        assert tokenizer.patch_size == 8
        assert tokenizer.embed_dim == 128
        assert tokenizer.in_channels == 1
    
    def test_forward_pass_basic(self):
        """Test basic forward pass with valid input."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        x = torch.randn(2, 3, 256, 256)
        
        tokens = tokenizer(x)
        
        # Expected number of patches: (256/16) * (256/16) = 16 * 16 = 256
        expected_num_patches = 256
        assert tokens.shape == (2, expected_num_patches, 256)
    
    def test_forward_pass_different_sizes(self):
        """Test forward pass with different input sizes."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        # Test 128x128 input
        x1 = torch.randn(1, 3, 128, 128)
        tokens1 = tokenizer(x1)
        assert tokens1.shape == (1, 64, 256)  # (128/16)^2 = 64 patches
        
        # Test 512x512 input
        x2 = torch.randn(1, 3, 512, 512)
        tokens2 = tokenizer(x2)
        assert tokens2.shape == (1, 1024, 256)  # (512/16)^2 = 1024 patches
    
    def test_forward_pass_non_square(self):
        """Test forward pass with non-square input."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        x = torch.randn(2, 3, 256, 512)
        
        tokens = tokenizer(x)
        
        # Expected patches: (256/16) * (512/16) = 16 * 32 = 512
        assert tokens.shape == (2, 512, 256)
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        # Height not divisible by patch_size
        x1 = torch.randn(1, 3, 255, 256)
        with pytest.raises(ValueError, match="must be divisible by patch_size"):
            tokenizer(x1)
        
        # Width not divisible by patch_size
        x2 = torch.randn(1, 3, 256, 255)
        with pytest.raises(ValueError, match="must be divisible by patch_size"):
            tokenizer(x2)
    
    def test_positional_embeddings_added(self):
        """Test that positional embeddings are properly added."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        x = torch.randn(2, 3, 256, 256)
        
        # First forward pass initializes positional embeddings
        tokens = tokenizer(x)
        
        # Check that positional embeddings were created
        assert tokenizer.pos_embedding is not None
        assert tokenizer.pos_embedding.shape == (1, 256, 256)
        assert tokenizer.num_patches == 256
    
    def test_positional_embeddings_reinitialized(self):
        """Test that positional embeddings are reinitialized for different sizes."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        # First input size
        x1 = torch.randn(1, 3, 256, 256)
        tokens1 = tokenizer(x1)
        pos_emb_1 = tokenizer.pos_embedding
        assert pos_emb_1.shape == (1, 256, 256)
        
        # Different input size should reinitialize
        x2 = torch.randn(1, 3, 128, 128)
        tokens2 = tokenizer(x2)
        pos_emb_2 = tokenizer.pos_embedding
        assert pos_emb_2.shape == (1, 64, 256)
        
        # Verify they are different objects
        assert pos_emb_1 is not pos_emb_2
    
    def test_get_num_patches(self):
        """Test get_num_patches utility method."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        assert tokenizer.get_num_patches(256, 256) == 256
        assert tokenizer.get_num_patches(128, 128) == 64
        assert tokenizer.get_num_patches(512, 512) == 1024
        assert tokenizer.get_num_patches(256, 512) == 512
    
    def test_get_num_patches_invalid(self):
        """Test get_num_patches with invalid dimensions."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        with pytest.raises(ValueError):
            tokenizer.get_num_patches(255, 256)
        
        with pytest.raises(ValueError):
            tokenizer.get_num_patches(256, 255)
    
    def test_batch_processing(self):
        """Test processing multiple batches."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        # Process different batch sizes
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 256, 256)
            tokens = tokenizer(x)
            assert tokens.shape == (batch_size, 256, 256)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the tokenizer."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        
        tokens = tokenizer(x)
        loss = tokens.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_different_patch_sizes(self):
        """Test tokenizer with different patch sizes."""
        for patch_size in [8, 16, 32]:
            tokenizer = SpectralPatchTokenizer(patch_size=patch_size, embed_dim=256)
            x = torch.randn(2, 3, 256, 256)
            
            tokens = tokenizer(x)
            expected_patches = (256 // patch_size) ** 2
            assert tokens.shape == (2, expected_patches, 256)
    
    def test_different_embed_dims(self):
        """Test tokenizer with different embedding dimensions."""
        for embed_dim in [128, 256, 512]:
            tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=embed_dim)
            x = torch.randn(2, 3, 256, 256)
            
            tokens = tokenizer(x)
            assert tokens.shape == (2, 256, embed_dim)
    
    def test_single_channel_input(self):
        """Test tokenizer with single channel input (grayscale)."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256, in_channels=1)
        x = torch.randn(2, 1, 256, 256)
        
        tokens = tokenizer(x)
        assert tokens.shape == (2, 256, 256)
    
    def test_device_compatibility(self):
        """Test tokenizer works on different devices."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        # CPU
        x_cpu = torch.randn(2, 3, 256, 256)
        tokens_cpu = tokenizer(x_cpu)
        assert tokens_cpu.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            tokenizer_gpu = tokenizer.cuda()
            x_gpu = x_cpu.cuda()
            tokens_gpu = tokenizer_gpu(x_gpu)
            assert tokens_gpu.device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
