"""Integration tests for SRS and SCV extractors with spectral pipeline."""

import pytest
import torch
from .fft_processor import FFTProcessor
from .frequency_masking import FrequencyMasking
from .patch_tokenizer import SpectralPatchTokenizer
from .srs_extractor import SRSExtractor
from .scv_computer import SCVComputer


class TestSRSSCVIntegration:
    """Integration tests for SRS and SCV with full spectral pipeline."""
    
    def test_full_pipeline(self):
        """Test complete spectral pipeline: FFT -> Masking -> Tokenizer -> SRS/SCV."""
        # Initialize components
        fft_processor = FFTProcessor(log_scale=True)
        freq_masking = FrequencyMasking(mask_type='high_pass', cutoff_freq=0.3)
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        srs_extractor = SRSExtractor(embed_dim=256, num_bands=4)
        scv_computer = SCVComputer(embed_dim=256, num_bands=4, consistency_dim=128)
        
        # Create input image
        batch_size = 2
        image = torch.randn(batch_size, 3, 256, 256)
        
        # Process through pipeline
        # 1. FFT
        magnitude = fft_processor(image)
        assert magnitude.shape == (batch_size, 3, 256, 256)
        
        # 2. Frequency masking
        masked = freq_masking(magnitude)
        assert masked.shape == (batch_size, 3, 256, 256)
        
        # 3. Tokenization
        tokens = tokenizer(masked)
        num_patches = (256 // 16) * (256 // 16)
        assert tokens.shape == (batch_size, num_patches, 256)
        
        # 4. SRS extraction
        srs = srs_extractor(tokens)
        assert srs.shape == (batch_size, 256)
        
        # 5. SCV computation
        scv = scv_computer(tokens)
        assert scv.shape == (batch_size, 128)
        
        # Verify outputs are valid
        assert not torch.isnan(srs).any()
        assert not torch.isinf(srs).any()
        assert not torch.isnan(scv).any()
        assert not torch.isinf(scv).any()
    
    def test_pipeline_with_different_image_sizes(self):
        """Test pipeline with various image sizes."""
        # Initialize components
        fft_processor = FFTProcessor()
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        srs_extractor = SRSExtractor(embed_dim=256)
        scv_computer = SCVComputer(embed_dim=256, consistency_dim=128)
        
        # Test different sizes
        sizes = [128, 256, 512]
        batch_size = 2
        
        for size in sizes:
            image = torch.randn(batch_size, 3, size, size)
            
            # Process
            magnitude = fft_processor(image)
            tokens = tokenizer(magnitude)
            srs = srs_extractor(tokens)
            scv = scv_computer(tokens)
            
            # Check shapes
            assert srs.shape == (batch_size, 256)
            assert scv.shape == (batch_size, 128)
            
            # Check validity
            assert not torch.isnan(srs).any()
            assert not torch.isnan(scv).any()
    
    def test_srs_scv_consistency(self):
        """Test that SRS and SCV produce consistent results for same input."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        srs_extractor = SRSExtractor(embed_dim=256)
        scv_computer = SCVComputer(embed_dim=256, consistency_dim=128)
        
        # Create input
        image = torch.randn(2, 3, 256, 256)
        tokens = tokenizer(image)
        
        # Extract features multiple times
        srs1 = srs_extractor(tokens)
        srs2 = srs_extractor(tokens)
        scv1 = scv_computer(tokens)
        scv2 = scv_computer(tokens)
        
        # Should be deterministic
        assert torch.allclose(srs1, srs2)
        assert torch.allclose(scv1, scv2)
    
    def test_srs_scv_different_inputs(self):
        """Test that SRS and SCV produce different outputs for different inputs."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        srs_extractor = SRSExtractor(embed_dim=256)
        scv_computer = SCVComputer(embed_dim=256, consistency_dim=128)
        
        # Create two different inputs
        image1 = torch.randn(2, 3, 256, 256)
        image2 = torch.randn(2, 3, 256, 256)
        
        tokens1 = tokenizer(image1)
        tokens2 = tokenizer(image2)
        
        # Extract features
        srs1 = srs_extractor(tokens1)
        srs2 = srs_extractor(tokens2)
        scv1 = scv_computer(tokens1)
        scv2 = scv_computer(tokens2)
        
        # Should be different
        assert not torch.allclose(srs1, srs2)
        assert not torch.allclose(scv1, scv2)
    
    def test_gradient_flow_through_pipeline(self):
        """Test that gradients flow through the entire pipeline."""
        # Initialize components
        fft_processor = FFTProcessor()
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        srs_extractor = SRSExtractor(embed_dim=256, aggregation_method='attention')
        scv_computer = SCVComputer(embed_dim=256, consistency_dim=128)
        
        # Create input with gradient tracking
        image = torch.randn(2, 3, 256, 256, requires_grad=True)
        
        # Forward pass
        magnitude = fft_processor(image)
        tokens = tokenizer(magnitude)
        srs = srs_extractor(tokens)
        scv = scv_computer(tokens)
        
        # Compute loss and backpropagate
        loss = srs.sum() + scv.sum()
        loss.backward()
        
        # Check gradients exist
        assert image.grad is not None
        assert not torch.isnan(image.grad).any()
    
    def test_batch_processing(self):
        """Test that pipeline handles different batch sizes correctly."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        srs_extractor = SRSExtractor(embed_dim=256)
        scv_computer = SCVComputer(embed_dim=256, consistency_dim=128)
        
        for batch_size in [1, 2, 4, 8, 16]:
            image = torch.randn(batch_size, 3, 256, 256)
            tokens = tokenizer(image)
            srs = srs_extractor(tokens)
            scv = scv_computer(tokens)
            
            assert srs.shape == (batch_size, 256)
            assert scv.shape == (batch_size, 128)
    
    def test_feature_concatenation(self):
        """Test concatenating SRS and SCV for downstream tasks."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        srs_extractor = SRSExtractor(embed_dim=256)
        scv_computer = SCVComputer(embed_dim=256, consistency_dim=128)
        
        # Process image
        image = torch.randn(2, 3, 256, 256)
        tokens = tokenizer(image)
        srs = srs_extractor(tokens)
        scv = scv_computer(tokens)
        
        # Concatenate features
        combined = torch.cat([srs, scv], dim=1)
        
        # Check combined shape
        assert combined.shape == (2, 256 + 128)
        assert not torch.isnan(combined).any()
    
    def test_different_aggregation_methods(self):
        """Test SRS with different aggregation methods in pipeline."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        image = torch.randn(2, 3, 256, 256)
        tokens = tokenizer(image)
        
        for method in ['mean', 'max', 'attention']:
            srs_extractor = SRSExtractor(embed_dim=256, aggregation_method=method)
            srs = srs_extractor(tokens)
            
            assert srs.shape == (2, 256)
            assert not torch.isnan(srs).any()
    
    def test_different_num_bands(self):
        """Test SRS and SCV with different numbers of frequency bands."""
        tokenizer = SpectralPatchTokenizer(patch_size=16, embed_dim=256)
        
        image = torch.randn(2, 3, 256, 256)
        tokens = tokenizer(image)
        
        for num_bands in [2, 4, 8]:
            srs_extractor = SRSExtractor(embed_dim=256, num_bands=num_bands)
            scv_computer = SCVComputer(embed_dim=256, num_bands=num_bands, consistency_dim=128)
            
            srs = srs_extractor(tokens)
            scv = scv_computer(tokens)
            
            assert srs.shape == (2, 256)
            assert scv.shape == (2, 128)
            assert not torch.isnan(srs).any()
            assert not torch.isnan(scv).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
