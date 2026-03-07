"""Integration tests for FFTProcessor and FrequencyMasking."""

import pytest
import torch
from .fft_processor import FFTProcessor
from .frequency_masking import FrequencyMasking


class TestSpectralIntegration:
    """Integration tests for spectral processing pipeline."""
    
    def test_fft_and_masking_pipeline(self):
        """Test complete pipeline: FFT -> Masking."""
        fft_processor = FFTProcessor()
        masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3)
        
        # Create sample image
        image = torch.randn(2, 3, 128, 128)
        
        # Process through pipeline
        spectrum = fft_processor(image)
        filtered_spectrum = masking(spectrum)
        
        # Verify shapes
        assert spectrum.shape == (2, 3, 128, 128)
        assert filtered_spectrum.shape == (2, 3, 128, 128)
        
        # Verify filtering occurred
        assert not torch.allclose(spectrum, filtered_spectrum)
    
    def test_low_pass_filtering_effect(self):
        """Test that low-pass filtering removes high-frequency content."""
        fft_processor = FFTProcessor(log_scale=False)
        masking = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.2)
        
        # Create image with high-frequency noise
        image = torch.randn(1, 1, 64, 64)
        
        # Process
        spectrum = fft_processor(image)
        filtered_spectrum = masking(spectrum)
        
        # High frequencies (corners) should be zeroed
        assert filtered_spectrum[0, 0, 0, 0] == 0.0
        assert filtered_spectrum[0, 0, -1, -1] == 0.0
        
        # Center (DC) should be preserved
        assert filtered_spectrum[0, 0, 32, 32] > 0
    
    def test_high_pass_filtering_effect(self):
        """Test that high-pass filtering removes low-frequency content."""
        fft_processor = FFTProcessor(log_scale=False)
        masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3, preserve_dc=False)
        
        # Create smooth image (mostly low frequencies)
        image = torch.ones(1, 1, 64, 64)
        
        # Process
        spectrum = fft_processor(image)
        filtered_spectrum = masking(spectrum)
        
        # Center region should be zeroed
        center_region = filtered_spectrum[0, 0, 28:36, 28:36]
        assert torch.all(center_region == 0.0)
    
    def test_band_pass_filtering(self):
        """Test band-pass filtering isolates specific frequency band."""
        fft_processor = FFTProcessor(log_scale=False)
        masking = FrequencyMasking(
            mask_type='band_pass',
            cutoff_freq=0.5,
            bandwidth=0.2,
            preserve_dc=False
        )
        
        image = torch.randn(1, 1, 64, 64)
        
        # Process
        spectrum = fft_processor(image)
        filtered_spectrum = masking(spectrum)
        
        # Both center and corners should have some zeros
        assert torch.sum(filtered_spectrum == 0) > 0
        # But not everything should be zero
        assert torch.sum(filtered_spectrum > 0) > 0
    
    def test_gradient_flow_through_pipeline(self):
        """Test that gradients flow through the entire pipeline."""
        fft_processor = FFTProcessor()
        masking = FrequencyMasking()
        
        image = torch.randn(1, 1, 64, 64, requires_grad=True)
        
        # Forward pass
        spectrum = fft_processor(image)
        filtered_spectrum = masking(spectrum)
        loss = filtered_spectrum.sum()
        
        # Backward pass
        loss.backward()
        
        assert image.grad is not None
        assert torch.any(image.grad != 0)
    
    def test_batch_processing_consistency(self):
        """Test that batch processing is consistent with individual processing."""
        fft_processor = FFTProcessor()
        masking = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.3)
        
        # Create batch
        images = torch.randn(4, 3, 64, 64)
        
        # Process as batch
        spectrum_batch = fft_processor(images)
        filtered_batch = masking(spectrum_batch)
        
        # Process individually
        for i in range(4):
            spectrum_single = fft_processor(images[i:i+1])
            filtered_single = masking(spectrum_single)
            
            assert torch.allclose(filtered_batch[i], filtered_single[0])
    
    def test_different_filter_types(self):
        """Test all three filter types produce different results."""
        fft_processor = FFTProcessor()
        image = torch.randn(1, 1, 64, 64)
        spectrum = fft_processor(image)
        
        # Apply different filters
        low_pass = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.3)
        high_pass = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3)
        band_pass = FrequencyMasking(mask_type='band_pass', cutoff_freq=0.5, bandwidth=0.2)
        
        filtered_low = low_pass(spectrum)
        filtered_high = high_pass(spectrum)
        filtered_band = band_pass(spectrum)
        
        # All should be different
        assert not torch.allclose(filtered_low, filtered_high)
        assert not torch.allclose(filtered_low, filtered_band)
        assert not torch.allclose(filtered_high, filtered_band)
    
    def test_log_scaling_effect_on_masking(self):
        """Test that log scaling affects the magnitude but not the mask application."""
        image = torch.randn(1, 1, 64, 64)
        masking = FrequencyMasking(mask_type='low_pass', cutoff_freq=0.3)
        
        # With log scaling
        fft_log = FFTProcessor(log_scale=True)
        spectrum_log = fft_log(image)
        filtered_log = masking(spectrum_log)
        
        # Without log scaling
        fft_no_log = FFTProcessor(log_scale=False)
        spectrum_no_log = fft_no_log(image)
        filtered_no_log = masking(spectrum_no_log)
        
        # Magnitudes should be different
        assert not torch.allclose(spectrum_log, spectrum_no_log)
        
        # But mask pattern should be the same (zeros in same places)
        mask_log = (filtered_log == 0)
        mask_no_log = (filtered_no_log == 0)
        assert torch.all(mask_log == mask_no_log)
    
    def test_realistic_image_processing(self):
        """Test processing with realistic image values [0, 1]."""
        fft_processor = FFTProcessor()
        masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3)
        
        # Create realistic image in [0, 1] range
        image = torch.rand(2, 3, 128, 128)
        
        # Process
        spectrum = fft_processor(image)
        filtered_spectrum = masking(spectrum)
        
        # Should produce valid output
        assert not torch.any(torch.isnan(filtered_spectrum))
        assert not torch.any(torch.isinf(filtered_spectrum))
        assert torch.all(filtered_spectrum >= 0)
    
    def test_fft_to_tokenizer_pipeline(self):
        """Test FFT processor followed by patch tokenizer."""
        from .patch_tokenizer import SpectralPatchTokenizer
        
        fft_processor = FFTProcessor()
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        # Create sample image
        x = torch.randn(2, 3, 256, 256)
        
        # Process through pipeline
        spectrum = fft_processor(x)
        tokens = tokenizer(spectrum)
        
        # Verify output shape
        expected_num_patches = (256 // 16) ** 2
        assert tokens.shape == (2, expected_num_patches, 256)
    
    def test_full_spectral_pipeline(self):
        """Test complete spectral pipeline: FFT -> Masking -> Tokenization."""
        from .patch_tokenizer import SpectralPatchTokenizer
        
        fft_processor = FFTProcessor()
        freq_masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3)
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        # Create sample image
        x = torch.randn(2, 3, 256, 256)
        
        # Process through full pipeline
        spectrum = fft_processor(x)
        masked_spectrum = freq_masking(spectrum)
        tokens = tokenizer(masked_spectrum)
        
        # Verify output
        expected_num_patches = (256 // 16) ** 2
        assert tokens.shape == (2, expected_num_patches, 256)
        assert not torch.isnan(tokens).any()
        assert not torch.isinf(tokens).any()
    
    def test_tokenizer_with_different_spectrum_sizes(self):
        """Test tokenizer with different spectrum sizes from FFT."""
        from .patch_tokenizer import SpectralPatchTokenizer
        
        fft_processor = FFTProcessor()
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        # Test different image sizes
        for size in [128, 256, 512]:
            x = torch.randn(2, 3, size, size)
            spectrum = fft_processor(x)
            tokens = tokenizer(spectrum)
            
            expected_num_patches = (size // 16) ** 2
            assert tokens.shape == (2, expected_num_patches, 256)
    
    def test_gradient_flow_through_full_pipeline(self):
        """Test gradient flow through FFT -> Masking -> Tokenization."""
        from .patch_tokenizer import SpectralPatchTokenizer
        
        fft_processor = FFTProcessor()
        freq_masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3)
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        
        # Forward pass
        spectrum = fft_processor(x)
        masked_spectrum = freq_masking(spectrum)
        tokens = tokenizer(masked_spectrum)
        
        # Compute loss and backpropagate
        loss = tokens.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
