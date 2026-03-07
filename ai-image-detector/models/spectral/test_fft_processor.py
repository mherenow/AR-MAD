"""Unit tests for FFTProcessor module."""

import pytest
import torch
from .fft_processor import FFTProcessor


class TestFFTProcessor:
    """Test suite for FFTProcessor."""
    
    def test_initialization(self):
        """Test FFTProcessor initialization with default parameters."""
        processor = FFTProcessor()
        assert processor.log_scale is True
        assert processor.eps == 1e-8
    
    def test_initialization_custom_params(self):
        """Test FFTProcessor initialization with custom parameters."""
        processor = FFTProcessor(log_scale=False, eps=1e-6)
        assert processor.log_scale is False
        assert processor.eps == 1e-6
    
    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        processor = FFTProcessor()
        batch_size, channels, height, width = 2, 3, 64, 64
        x = torch.randn(batch_size, channels, height, width)
        
        output = processor(x)
        
        assert output.shape == (batch_size, channels, height, width)
    
    def test_forward_output_type(self):
        """Test that forward pass produces real-valued output."""
        processor = FFTProcessor()
        x = torch.randn(2, 3, 64, 64)
        
        output = processor(x)
        
        assert output.dtype == torch.float32
        assert not torch.is_complex(output)
    
    def test_forward_non_negative_with_log_scale(self):
        """Test that log-scaled output is non-negative."""
        processor = FFTProcessor(log_scale=True)
        x = torch.randn(2, 3, 64, 64)
        
        output = processor(x)
        
        assert torch.all(output >= 0), "Log-scaled magnitude should be non-negative"
    
    def test_forward_non_negative_without_log_scale(self):
        """Test that magnitude output is non-negative."""
        processor = FFTProcessor(log_scale=False)
        x = torch.randn(2, 3, 64, 64)
        
        output = processor(x)
        
        assert torch.all(output >= 0), "Magnitude should be non-negative"
    
    def test_dc_component_at_center(self):
        """Test that DC component is at the center after fftshift."""
        processor = FFTProcessor(log_scale=False)
        # Create a constant image (all DC component)
        x = torch.ones(1, 1, 64, 64)
        
        output = processor(x)
        
        # DC component should be at center (32, 32) and be the maximum
        center_y, center_x = 32, 32
        center_value = output[0, 0, center_y, center_x]
        
        # Center should have the highest value for a constant image
        assert center_value > output[0, 0, 0, 0]
    
    def test_different_image_sizes(self):
        """Test FFT processing with different image sizes."""
        processor = FFTProcessor()
        
        sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
        for height, width in sizes:
            x = torch.randn(1, 3, height, width)
            output = processor(x)
            assert output.shape == (1, 3, height, width)
    
    def test_batch_processing(self):
        """Test processing multiple images in a batch."""
        processor = FFTProcessor()
        
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 64, 64)
            output = processor(x)
            assert output.shape == (batch_size, 3, 64, 64)
    
    def test_single_channel_image(self):
        """Test FFT processing with single-channel (grayscale) images."""
        processor = FFTProcessor()
        x = torch.randn(2, 1, 64, 64)
        
        output = processor(x)
        
        assert output.shape == (2, 1, 64, 64)
    
    def test_log_scale_effect(self):
        """Test that log scaling reduces dynamic range."""
        x = torch.randn(1, 1, 64, 64)
        
        processor_log = FFTProcessor(log_scale=True)
        processor_no_log = FFTProcessor(log_scale=False)
        
        output_log = processor_log(x)
        output_no_log = processor_no_log(x)
        
        # Log scaling should compress the range
        range_log = output_log.max() - output_log.min()
        range_no_log = output_no_log.max() - output_no_log.min()
        
        assert range_log < range_no_log
    
    def test_gradient_flow(self):
        """Test that gradients flow through the FFT processor."""
        processor = FFTProcessor()
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        
        output = processor(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        processor = FFTProcessor()
        x = torch.randn(1, 3, 64, 64)
        
        output1 = processor(x)
        output2 = processor(x)
        
        assert torch.allclose(output1, output2)
    
    def test_inverse_reconstruction(self):
        """Test inverse FFT reconstruction (utility method)."""
        processor = FFTProcessor(log_scale=False)
        x = torch.randn(1, 1, 64, 64)
        
        # Get magnitude and phase
        fft_result = torch.fft.fft2(x, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)
        
        # Reconstruct
        reconstructed = processor.inverse(magnitude, phase)
        
        # Should be close to original (within numerical precision)
        assert torch.allclose(reconstructed, x, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
