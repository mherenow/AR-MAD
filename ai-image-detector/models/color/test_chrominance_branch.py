"""Unit tests for ChrominanceBranch module."""

import pytest
import torch
from .chrominance_branch import ChrominanceBranch
from .color_space import RGBtoYCbCr


class TestChroMinanceBranch:
    """Test ChrominanceBranch feature extraction."""
    
    def test_output_shape(self):
        """Test that output has correct feature dimension."""
        branch = ChrominanceBranch(num_bins=64, feature_dim=256)
        ycbcr = torch.rand(4, 3, 224, 224) * 255
        features = branch(ycbcr)
        
        assert features.shape == (4, 256)
    
    def test_custom_feature_dim(self):
        """Test with custom feature dimension."""
        branch = ChrominanceBranch(num_bins=32, feature_dim=128)
        ycbcr = torch.rand(2, 3, 128, 128) * 255
        features = branch(ycbcr)
        
        assert features.shape == (2, 128)
    
    def test_channel_count_validation(self):
        """Test that branch validates channel count."""
        branch = ChrominanceBranch()
        invalid_input = torch.rand(2, 4, 64, 64) * 255
        
        with pytest.raises(ValueError, match="Expected 3 channels"):
            branch(invalid_input)
    
    def test_single_image(self):
        """Test with single image (batch size 1)."""
        branch = ChrominanceBranch()
        branch.eval()  # Set to eval mode to avoid BatchNorm issues with batch size 1
        ycbcr = torch.rand(1, 3, 64, 64) * 255
        features = branch(ycbcr)
        
        assert features.shape == (1, 256)
    
    def test_batch_processing(self):
        """Test with different batch sizes."""
        branch = ChrominanceBranch()
        
        for batch_size in [1, 4, 8, 16]:
            if batch_size == 1:
                branch.eval()  # Use eval mode for batch size 1
            else:
                branch.train()
            ycbcr = torch.rand(batch_size, 3, 128, 128) * 255
            features = branch(ycbcr)
            assert features.shape == (batch_size, 256)
    
    def test_different_image_sizes(self):
        """Test with different image resolutions."""
        branch = ChrominanceBranch()
        
        for size in [64, 128, 224, 256]:
            ycbcr = torch.rand(2, 3, size, size) * 255
            features = branch(ycbcr)
            assert features.shape == (2, 256)
    
    def test_small_image_handling(self):
        """Test with very small images (smaller than patch size)."""
        branch = ChrominanceBranch()
        # Image smaller than 8x8 patch size
        ycbcr = torch.rand(2, 3, 4, 4) * 255
        features = branch(ycbcr)
        
        assert features.shape == (2, 256)
    
    def test_uniform_color_image(self):
        """Test with uniform color image."""
        branch = ChrominanceBranch()
        branch.eval()  # Set to eval mode for batch size 1
        # Create uniform YCbCr image
        ycbcr = torch.ones(1, 3, 64, 64) * 128
        features = branch(ycbcr)
        
        assert features.shape == (1, 256)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        branch = ChrominanceBranch()
        ycbcr = torch.rand(2, 3, 64, 64) * 255
        ycbcr.requires_grad = True
        features = branch(ycbcr)
        loss = features.sum()
        loss.backward()
        
        assert ycbcr.grad is not None
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        branch = ChrominanceBranch()
        branch.eval()
        
        ycbcr = torch.rand(2, 3, 64, 64) * 255
        
        with torch.no_grad():
            features1 = branch(ycbcr)
            features2 = branch(ycbcr)
        
        assert torch.allclose(features1, features2)
    
    def test_different_num_bins(self):
        """Test with different histogram bin counts."""
        for num_bins in [16, 32, 64, 128]:
            branch = ChrominanceBranch(num_bins=num_bins, feature_dim=256)
            ycbcr = torch.rand(2, 3, 64, 64) * 255
            features = branch(ycbcr)
            
            assert features.shape == (2, 256)
    
    def test_device_compatibility(self):
        """Test that branch works on different devices."""
        branch = ChrominanceBranch()
        ycbcr = torch.rand(2, 3, 64, 64) * 255
        
        # CPU
        features_cpu = branch(ycbcr)
        assert features_cpu.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            branch_gpu = branch.cuda()
            ycbcr_gpu = ycbcr.cuda()
            features_gpu = branch_gpu(ycbcr_gpu)
            assert features_gpu.device.type == 'cuda'


class TestChroMinanceBranchIntegration:
    """Test ChrominanceBranch integration with color space conversion."""
    
    def test_rgb_to_chrominance_pipeline(self):
        """Test full pipeline from RGB to chrominance features."""
        rgb_to_ycbcr = RGBtoYCbCr()
        branch = ChrominanceBranch()
        
        # Create RGB image
        rgb = torch.rand(4, 3, 128, 128) * 255
        
        # Convert to YCbCr
        ycbcr = rgb_to_ycbcr(rgb)
        
        # Extract chrominance features
        features = branch(ycbcr)
        
        assert features.shape == (4, 256)
    
    def test_different_color_distributions(self):
        """Test with images having different color distributions."""
        rgb_to_ycbcr = RGBtoYCbCr()
        branch = ChrominanceBranch()
        branch.eval()  # Set to eval mode for batch size 1
        
        # Grayscale-like image (low chrominance)
        gray = torch.ones(1, 3, 64, 64) * 128
        ycbcr_gray = rgb_to_ycbcr(gray)
        features_gray = branch(ycbcr_gray)
        
        # Colorful image (high chrominance)
        colorful = torch.rand(1, 3, 64, 64) * 255
        ycbcr_colorful = rgb_to_ycbcr(colorful)
        features_colorful = branch(ycbcr_colorful)
        
        # Both should produce valid features
        assert features_gray.shape == (1, 256)
        assert features_colorful.shape == (1, 256)
        
        # Features should be different
        assert not torch.allclose(features_gray, features_colorful)
    
    def test_primary_colors(self):
        """Test with primary color images."""
        rgb_to_ycbcr = RGBtoYCbCr()
        branch = ChrominanceBranch()
        branch.eval()  # Set to eval mode for batch size 1
        
        # Create images with primary colors (B, C, H, W)
        red = torch.zeros(1, 3, 64, 64)
        red[:, 0, :, :] = 255.0  # Red channel
        
        green = torch.zeros(1, 3, 64, 64)
        green[:, 1, :, :] = 255.0  # Green channel
        
        blue = torch.zeros(1, 3, 64, 64)
        blue[:, 2, :, :] = 255.0  # Blue channel
        
        # Convert and extract features
        features_red = branch(rgb_to_ycbcr(red))
        features_green = branch(rgb_to_ycbcr(green))
        features_blue = branch(rgb_to_ycbcr(blue))
        
        # All should produce valid features
        assert features_red.shape == (1, 256)
        assert features_green.shape == (1, 256)
        assert features_blue.shape == (1, 256)
        
        # Features should be different for different colors
        assert not torch.allclose(features_red, features_green)
        assert not torch.allclose(features_red, features_blue)
        assert not torch.allclose(features_green, features_blue)
