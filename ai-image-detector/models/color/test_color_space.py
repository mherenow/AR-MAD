"""Unit tests for color space conversion modules."""

import pytest
import torch
from .color_space import RGBtoYCbCr, YCbCrtoRGB


class TestRGBtoYCbCr:
    """Test RGB to YCbCr color space conversion."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        converter = RGBtoYCbCr()
        rgb = torch.rand(2, 3, 64, 64) * 255
        ycbcr = converter(rgb)
        
        assert ycbcr.shape == rgb.shape
    
    def test_channel_count_validation(self):
        """Test that converter validates channel count."""
        converter = RGBtoYCbCr()
        invalid_input = torch.rand(2, 4, 64, 64) * 255
        
        with pytest.raises(ValueError, match="Expected 3 channels"):
            converter(invalid_input)
    
    def test_pure_white_conversion(self):
        """Test conversion of pure white (255, 255, 255)."""
        converter = RGBtoYCbCr()
        white = torch.ones(1, 3, 1, 1) * 255
        ycbcr = converter(white)
        
        # White should convert to Y=255, Cb=128, Cr=128
        assert torch.allclose(ycbcr[0, 0, 0, 0], torch.tensor(255.0), atol=1.0)
        assert torch.allclose(ycbcr[0, 1, 0, 0], torch.tensor(128.0), atol=1.0)
        assert torch.allclose(ycbcr[0, 2, 0, 0], torch.tensor(128.0), atol=1.0)
    
    def test_pure_black_conversion(self):
        """Test conversion of pure black (0, 0, 0)."""
        converter = RGBtoYCbCr()
        black = torch.zeros(1, 3, 1, 1)
        ycbcr = converter(black)
        
        # Black should convert to Y=0, Cb=128, Cr=128
        assert torch.allclose(ycbcr[0, 0, 0, 0], torch.tensor(0.0), atol=1.0)
        assert torch.allclose(ycbcr[0, 1, 0, 0], torch.tensor(128.0), atol=1.0)
        assert torch.allclose(ycbcr[0, 2, 0, 0], torch.tensor(128.0), atol=1.0)
    
    def test_pure_red_conversion(self):
        """Test conversion of pure red (255, 0, 0)."""
        converter = RGBtoYCbCr()
        red = torch.tensor([[[[255.0]], [[0.0]], [[0.0]]]])
        ycbcr = converter(red)
        
        # Red: Y = 0.299*255 ≈ 76.2, Cb = -0.169*255 + 128 ≈ 84.9, Cr = 0.500*255 + 128 ≈ 255.5
        assert torch.allclose(ycbcr[0, 0, 0, 0], torch.tensor(76.245), atol=1.0)
        assert torch.allclose(ycbcr[0, 1, 0, 0], torch.tensor(84.895), atol=1.0)
        assert torch.allclose(ycbcr[0, 2, 0, 0], torch.tensor(255.5), atol=1.0)
    
    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        converter = RGBtoYCbCr()
        rgb = torch.rand(8, 3, 32, 32) * 255
        ycbcr = converter(rgb)
        
        assert ycbcr.shape == (8, 3, 32, 32)
    
    def test_device_compatibility(self):
        """Test that converter works on different devices."""
        converter = RGBtoYCbCr()
        rgb = torch.rand(1, 3, 16, 16) * 255
        
        # CPU
        ycbcr_cpu = converter(rgb)
        assert ycbcr_cpu.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            converter_gpu = converter.cuda()
            rgb_gpu = rgb.cuda()
            ycbcr_gpu = converter_gpu(rgb_gpu)
            assert ycbcr_gpu.device.type == 'cuda'


class TestYCbCrtoRGB:
    """Test YCbCr to RGB color space conversion."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        converter = YCbCrtoRGB()
        ycbcr = torch.rand(2, 3, 64, 64) * 255
        rgb = converter(ycbcr)
        
        assert rgb.shape == ycbcr.shape
    
    def test_channel_count_validation(self):
        """Test that converter validates channel count."""
        converter = YCbCrtoRGB()
        invalid_input = torch.rand(2, 4, 64, 64) * 255
        
        with pytest.raises(ValueError, match="Expected 3 channels"):
            converter(invalid_input)
    
    def test_pure_white_conversion(self):
        """Test conversion of YCbCr white (255, 128, 128) to RGB."""
        converter = YCbCrtoRGB()
        white_ycbcr = torch.tensor([[[[255.0]], [[128.0]], [[128.0]]]])
        rgb = converter(white_ycbcr)
        
        # Should convert back to approximately (255, 255, 255)
        assert torch.allclose(rgb[0, 0, 0, 0], torch.tensor(255.0), atol=1.0)
        assert torch.allclose(rgb[0, 1, 0, 0], torch.tensor(255.0), atol=1.0)
        assert torch.allclose(rgb[0, 2, 0, 0], torch.tensor(255.0), atol=1.0)
    
    def test_pure_black_conversion(self):
        """Test conversion of YCbCr black (0, 128, 128) to RGB."""
        converter = YCbCrtoRGB()
        black_ycbcr = torch.tensor([[[[0.0]], [[128.0]], [[128.0]]]])
        rgb = converter(black_ycbcr)
        
        # Should convert back to approximately (0, 0, 0)
        assert torch.allclose(rgb[0, 0, 0, 0], torch.tensor(0.0), atol=1.0)
        assert torch.allclose(rgb[0, 1, 0, 0], torch.tensor(0.0), atol=1.0)
        assert torch.allclose(rgb[0, 2, 0, 0], torch.tensor(0.0), atol=1.0)
    
    def test_clamping(self):
        """Test that output is clamped to [0, 255]."""
        converter = YCbCrtoRGB()
        # Create extreme values that might produce out-of-range RGB
        extreme_ycbcr = torch.tensor([[[[255.0]], [[0.0]], [[255.0]]]])
        rgb = converter(extreme_ycbcr)
        
        assert torch.all(rgb >= 0)
        assert torch.all(rgb <= 255)
    
    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        converter = YCbCrtoRGB()
        ycbcr = torch.rand(8, 3, 32, 32) * 255
        rgb = converter(ycbcr)
        
        assert rgb.shape == (8, 3, 32, 32)


class TestRoundTripConversion:
    """Test round-trip conversion RGB -> YCbCr -> RGB."""
    
    def test_round_trip_accuracy(self):
        """Test that round-trip conversion is approximately accurate."""
        rgb_to_ycbcr = RGBtoYCbCr()
        ycbcr_to_rgb = YCbCrtoRGB()
        
        # Create test RGB image
        original_rgb = torch.rand(1, 3, 64, 64) * 255
        
        # Convert RGB -> YCbCr -> RGB
        ycbcr = rgb_to_ycbcr(original_rgb)
        reconstructed_rgb = ycbcr_to_rgb(ycbcr)
        
        # Check that reconstruction is close to original
        # Allow some tolerance due to floating point arithmetic
        assert torch.allclose(original_rgb, reconstructed_rgb, atol=2.0)
    
    def test_round_trip_with_known_colors(self):
        """Test round-trip with known color values."""
        rgb_to_ycbcr = RGBtoYCbCr()
        ycbcr_to_rgb = YCbCrtoRGB()
        
        # Test with primary colors
        colors = torch.tensor([
            [[[255.0]], [[0.0]], [[0.0]]],      # Red
            [[[0.0]], [[255.0]], [[0.0]]],      # Green
            [[[0.0]], [[0.0]], [[255.0]]],      # Blue
            [[[255.0]], [[255.0]], [[255.0]]],  # White
            [[[0.0]], [[0.0]], [[0.0]]],        # Black
        ])
        
        for color in colors:
            ycbcr = rgb_to_ycbcr(color.unsqueeze(0))
            reconstructed = ycbcr_to_rgb(ycbcr)
            assert torch.allclose(color.unsqueeze(0), reconstructed, atol=2.0)
    
    def test_round_trip_batch(self):
        """Test round-trip conversion with batches."""
        rgb_to_ycbcr = RGBtoYCbCr()
        ycbcr_to_rgb = YCbCrtoRGB()
        
        original_rgb = torch.rand(16, 3, 32, 32) * 255
        ycbcr = rgb_to_ycbcr(original_rgb)
        reconstructed_rgb = ycbcr_to_rgb(ycbcr)
        
        assert torch.allclose(original_rgb, reconstructed_rgb, atol=2.0)
