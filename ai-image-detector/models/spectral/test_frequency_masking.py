"""Unit tests for FrequencyMasking module."""

import pytest
import torch
from .frequency_masking import FrequencyMasking


class TestFrequencyMasking:
    """Test suite for FrequencyMasking."""
    
    def test_initialization_low_pass(self):
        """Test initialization with low-pass filter."""
        masking = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.3)
        assert masking.mask_type == 'low_pass'
        assert masking.cutoff_freq == 0.3
        assert masking.preserve_dc is True
    
    def test_initialization_high_pass(self):
        """Test initialization with high-pass filter."""
        masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.5)
        assert masking.mask_type == 'high_pass'
        assert masking.cutoff_freq == 0.5
    
    def test_initialization_band_pass(self):
        """Test initialization with band-pass filter."""
        masking = FrequencyMasking(mask_type='band_pass', cutoff_freq=0.4, bandwidth=0.2)
        assert masking.mask_type == 'band_pass'
        assert masking.cutoff_freq == 0.4
        assert masking.bandwidth == 0.2
    
    def test_invalid_mask_type(self):
        """Test that invalid mask type raises ValueError."""
        with pytest.raises(ValueError, match="mask_type must be"):
            FrequencyMasking(mask_type='invalid')
    
    def test_invalid_cutoff_freq_too_low(self):
        """Test that cutoff frequency below 0 raises ValueError."""
        with pytest.raises(ValueError, match="cutoff_freq must be in"):
            FrequencyMasking(cutoff_freq=-0.1)
    
    def test_invalid_cutoff_freq_too_high(self):
        """Test that cutoff frequency above 1 raises ValueError."""
        with pytest.raises(ValueError, match="cutoff_freq must be in"):
            FrequencyMasking(cutoff_freq=1.5)
    
    def test_forward_output_shape(self):
        """Test that forward pass preserves input shape."""
        masking = FrequencyMasking()
        batch_size, channels, height, width = 2, 3, 64, 64
        spectrum = torch.randn(batch_size, channels, height, width)
        
        output = masking(spectrum)
        
        assert output.shape == (batch_size, channels, height, width)
    
    def test_low_pass_filter_removes_high_frequencies(self):
        """Test that low-pass filter zeros out high frequencies."""
        masking = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.3)
        spectrum = torch.ones(1, 1, 64, 64)  # Uniform spectrum
        
        output = masking(spectrum)
        
        # Check that corners (high frequencies) are zeroed
        assert output[0, 0, 0, 0] == 0.0
        assert output[0, 0, 0, -1] == 0.0
        assert output[0, 0, -1, 0] == 0.0
        assert output[0, 0, -1, -1] == 0.0
        
        # Center (DC component) should be preserved
        assert output[0, 0, 32, 32] == 1.0
    
    def test_high_pass_filter_removes_low_frequencies(self):
        """Test that high-pass filter zeros out low frequencies."""
        masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3, preserve_dc=False)
        spectrum = torch.ones(1, 1, 64, 64)
        
        output = masking(spectrum)
        
        # Center (low frequencies) should be zeroed
        center_region = output[0, 0, 28:36, 28:36]
        assert torch.all(center_region == 0.0)
        
        # Corners (high frequencies) should be preserved
        assert output[0, 0, 0, 0] == 1.0
    
    def test_dc_preservation(self):
        """Test that DC component is preserved when preserve_dc=True."""
        masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3, preserve_dc=True)
        spectrum = torch.ones(1, 1, 64, 64)
        
        output = masking(spectrum)
        
        # DC component at center should be preserved
        assert output[0, 0, 32, 32] == 1.0
    
    def test_dc_not_preserved(self):
        """Test that DC component is not preserved when preserve_dc=False."""
        masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3, preserve_dc=False)
        spectrum = torch.ones(1, 1, 64, 64)
        
        output = masking(spectrum)
        
        # DC component should be zeroed (it's in the low-frequency region)
        assert output[0, 0, 32, 32] == 0.0
    
    def test_band_pass_filter(self):
        """Test that band-pass filter keeps only specified band."""
        masking = FrequencyMasking(
            mask_type='band_pass',
            cutoff_freq=0.5,
            bandwidth=0.2,
            preserve_dc=False
        )
        spectrum = torch.ones(1, 1, 64, 64)
        
        output = masking(spectrum)
        
        # Center (DC) should be zeroed
        assert output[0, 0, 32, 32] == 0.0
        
        # Some mid-frequency regions should be preserved
        # Check that not all values are zero
        assert torch.sum(output > 0) > 0
        
        # Check that not all values are preserved
        assert torch.sum(output == 0) > 0
    
    def test_different_image_sizes(self):
        """Test masking with different image sizes."""
        masking = FrequencyMasking()
        
        sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
        for height, width in sizes:
            spectrum = torch.randn(1, 1, height, width)
            output = masking(spectrum)
            assert output.shape == (1, 1, height, width)
    
    def test_batch_processing(self):
        """Test processing multiple spectra in a batch."""
        masking = FrequencyMasking()
        
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            spectrum = torch.randn(batch_size, 3, 64, 64)
            output = masking(spectrum)
            assert output.shape == (batch_size, 3, 64, 64)
    
    def test_multi_channel_processing(self):
        """Test that masking is applied consistently across channels."""
        masking = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.3)
        spectrum = torch.ones(1, 3, 64, 64)
        
        output = masking(spectrum)
        
        # All channels should have the same mask applied
        assert torch.allclose(output[0, 0], output[0, 1])
        assert torch.allclose(output[0, 1], output[0, 2])
    
    def test_get_mask_method(self):
        """Test get_mask utility method."""
        masking = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.3)
        
        mask = masking.get_mask(64, 64)
        
        assert mask.shape == (1, 1, 64, 64)
        assert torch.all((mask == 0) | (mask == 1))  # Binary mask
    
    def test_mask_symmetry(self):
        """Test that frequency mask is approximately radially symmetric."""
        masking = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.3)
        
        mask = masking.get_mask(64, 64)
        mask_2d = mask[0, 0]
        
        # Check that mask is radially symmetric by verifying that
        # points at the same distance from center have the same value
        center_y, center_x = 32, 32
        
        # Sample a few points at the same distance
        # Point at (32+10, 32) and (32, 32+10) should have same value
        assert mask_2d[center_y + 10, center_x] == mask_2d[center_y, center_x + 10]
        
        # Point at (32+5, 32+5) and (32-5, 32+5) should have same value (approximately)
        # Due to discrete grid, we allow small differences
        val1 = mask_2d[center_y + 5, center_x + 5]
        val2 = mask_2d[center_y - 5, center_x + 5]
        assert abs(val1 - val2) < 0.1
    
    def test_gradient_flow(self):
        """Test that gradients flow through the masking operation."""
        masking = FrequencyMasking()
        spectrum = torch.randn(1, 1, 64, 64, requires_grad=True)
        
        output = masking(spectrum)
        loss = output.sum()
        loss.backward()
        
        assert spectrum.grad is not None
        # Some gradients should be non-zero (where mask is 1)
        assert torch.any(spectrum.grad != 0)
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        masking = FrequencyMasking()
        spectrum = torch.randn(1, 3, 64, 64)
        
        output1 = masking(spectrum)
        output2 = masking(spectrum)
        
        assert torch.allclose(output1, output2)
    
    def test_cutoff_frequency_effect(self):
        """Test that different cutoff frequencies produce different masks."""
        spectrum = torch.ones(1, 1, 64, 64)
        
        masking_low = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.2)
        masking_high = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.5)
        
        output_low = masking_low(spectrum)
        output_high = masking_high(spectrum)
        
        # Higher cutoff should preserve more frequencies
        assert torch.sum(output_high > 0) > torch.sum(output_low > 0)
    
    def test_rectangular_images(self):
        """Test masking with non-square images."""
        masking = FrequencyMasking()
        
        # Test different aspect ratios
        spectrum1 = torch.randn(1, 1, 64, 128)
        output1 = masking(spectrum1)
        assert output1.shape == (1, 1, 64, 128)
        
        spectrum2 = torch.randn(1, 1, 128, 64)
        output2 = masking(spectrum2)
        assert output2.shape == (1, 1, 128, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
