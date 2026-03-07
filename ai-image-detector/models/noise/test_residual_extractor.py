"""
Unit tests for NoiseResidualExtractor.
"""

import pytest
import torch
import warnings

from .residual_extractor import NoiseResidualExtractor, DIFFUSERS_AVAILABLE


class TestNoiseResidualExtractor:
    """Test suite for NoiseResidualExtractor."""
    
    def test_gaussian_method_initialization(self):
        """Test that Gaussian method initializes correctly."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        
        assert extractor.method == 'gaussian'
        assert extractor.gaussian_sigma == 2.0
        assert hasattr(extractor, 'gaussian_kernel')
        assert extractor.gaussian_kernel is not None
    
    def test_diffusion_method_initialization(self):
        """Test that diffusion method initializes (or falls back gracefully)."""
        extractor = NoiseResidualExtractor(method='diffusion')
        
        # Should either be 'diffusion' (if available) or 'gaussian' (fallback)
        assert extractor.method in ['diffusion', 'gaussian']
        
        if not DIFFUSERS_AVAILABLE:
            # Should have fallen back to gaussian
            assert extractor.method == 'gaussian'
    
    def test_gaussian_kernel_creation(self):
        """Test that Gaussian kernel is created with correct properties."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        kernel = extractor.gaussian_kernel
        
        # Check shape
        assert kernel.ndim == 4
        assert kernel.shape[0] == 1
        assert kernel.shape[1] == 1
        
        # Kernel should be symmetric
        kernel_2d = kernel.squeeze()
        assert torch.allclose(kernel_2d, kernel_2d.T, atol=1e-6)
        
        # Kernel should sum to approximately 1
        assert torch.allclose(kernel_2d.sum(), torch.tensor(1.0), atol=1e-5)
        
        # All values should be positive
        assert (kernel_2d >= 0).all()
    
    def test_gaussian_residual_extraction_shape(self):
        """Test that Gaussian residual extraction preserves shape."""
        extractor = NoiseResidualExtractor(method='gaussian')
        
        # Test with different batch sizes and image sizes
        for batch_size in [1, 4]:
            for size in [64, 128, 256]:
                x = torch.rand(batch_size, 3, size, size)
                residual = extractor(x)
                
                assert residual.shape == x.shape
    
    def test_gaussian_residual_extraction_range(self):
        """Test that Gaussian residual values are in reasonable range."""
        extractor = NoiseResidualExtractor(method='gaussian')
        
        # Create test image in [0, 1] range
        x = torch.rand(2, 3, 128, 128)
        residual = extractor(x)
        
        # Residual should be in [-1, 1] range (since input is [0, 1])
        assert residual.min() >= -1.0
        assert residual.max() <= 1.0
    
    def test_gaussian_blur_removes_high_frequency(self):
        """Test that Gaussian blur removes high-frequency content."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        
        # Create an image with high-frequency noise
        x = torch.rand(1, 3, 128, 128)
        noise = torch.randn(1, 3, 128, 128) * 0.1
        x_noisy = torch.clamp(x + noise, 0, 1)
        
        # Extract residual
        residual = extractor(x_noisy)
        
        # Residual should contain the high-frequency noise
        # The residual should have non-zero values
        assert residual.abs().mean() > 0.001
    
    def test_uniform_image_produces_zero_residual(self):
        """Test that a uniform image produces near-zero residual in the center."""
        extractor = NoiseResidualExtractor(method='gaussian')
        
        # Create uniform image
        x = torch.ones(1, 3, 128, 128) * 0.5
        residual = extractor(x)
        
        # Residual in the center should be very close to zero (ignoring edge effects)
        # Check the center 64x64 region
        center_residual = residual[:, :, 32:96, 32:96]
        assert center_residual.abs().mean() < 0.01
    
    @pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="diffusers not available")
    def test_diffusion_residual_extraction_shape(self):
        """Test that diffusion residual extraction preserves shape."""
        extractor = NoiseResidualExtractor(method='diffusion')
        
        # Only test if diffusion is actually available
        if extractor.method != 'diffusion':
            pytest.skip("Diffusion model failed to load")
        
        # Test with smaller images for speed
        x = torch.rand(1, 3, 256, 256)
        residual = extractor(x)
        
        assert residual.shape == x.shape
    
    @pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="diffusers not available")
    def test_diffusion_residual_extraction_range(self):
        """Test that diffusion residual values are in reasonable range."""
        extractor = NoiseResidualExtractor(method='diffusion')
        
        # Only test if diffusion is actually available
        if extractor.method != 'diffusion':
            pytest.skip("Diffusion model failed to load")
        
        # Create test image in [0, 1] range
        x = torch.rand(1, 3, 256, 256)
        residual = extractor(x)
        
        # Residual should be in reasonable range
        assert residual.min() >= -2.0
        assert residual.max() <= 2.0
    
    def test_fallback_warning_when_diffusers_unavailable(self):
        """Test that a warning is issued when falling back to Gaussian."""
        if DIFFUSERS_AVAILABLE:
            pytest.skip("diffusers is available, cannot test fallback warning")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extractor = NoiseResidualExtractor(method='diffusion')
            
            # Should have issued a warning
            assert len(w) > 0
            assert "diffusers library not available" in str(w[0].message).lower()
            
            # Should have fallen back to gaussian
            assert extractor.method == 'gaussian'
    
    def test_different_sigma_values(self):
        """Test that different sigma values produce different kernels."""
        extractor1 = NoiseResidualExtractor(method='gaussian', gaussian_sigma=1.0)
        extractor2 = NoiseResidualExtractor(method='gaussian', gaussian_sigma=3.0)
        
        kernel1 = extractor1.gaussian_kernel
        kernel2 = extractor2.gaussian_kernel
        
        # Larger sigma should produce larger kernel
        assert kernel2.shape[-1] > kernel1.shape[-1]
        
        # Test that different sigmas produce different results on same image
        x = torch.rand(1, 3, 128, 128)
        residual1 = extractor1(x)
        residual2 = extractor2(x)
        
        # Residuals should be different
        assert not torch.allclose(residual1, residual2)
    
    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        extractor = NoiseResidualExtractor(method='gaussian')
        
        # Process batch
        x_batch = torch.rand(8, 3, 128, 128)
        residual_batch = extractor(x_batch)
        
        # Process individually
        residuals_individual = []
        for i in range(8):
            residual = extractor(x_batch[i:i+1])
            residuals_individual.append(residual)
        residuals_individual = torch.cat(residuals_individual, dim=0)
        
        # Results should be identical
        assert torch.allclose(residual_batch, residuals_individual, atol=1e-5)
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the extractor (for Gaussian method)."""
        extractor = NoiseResidualExtractor(method='gaussian')
        
        # Create input with gradient tracking
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        
        # Forward pass
        residual = extractor(x)
        
        # Compute loss and backward
        loss = residual.sum()
        loss.backward()
        
        # Gradients should exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_device_compatibility(self):
        """Test that extractor works on different devices."""
        extractor = NoiseResidualExtractor(method='gaussian')
        
        # Test on CPU
        x_cpu = torch.rand(2, 3, 64, 64)
        residual_cpu = extractor(x_cpu)
        assert residual_cpu.device == x_cpu.device
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            extractor_cuda = extractor.cuda()
            x_cuda = x_cpu.cuda()
            residual_cuda = extractor_cuda(x_cuda)
            assert residual_cuda.device == x_cuda.device
    
    def test_eval_mode(self):
        """Test that extractor works in eval mode."""
        extractor = NoiseResidualExtractor(method='gaussian')
        extractor.eval()
        
        x = torch.rand(2, 3, 128, 128)
        residual = extractor(x)
        
        assert residual.shape == x.shape
    
    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        extractor = NoiseResidualExtractor(method='gaussian')
        
        x = torch.rand(2, 3, 128, 128)
        
        residual1 = extractor(x)
        residual2 = extractor(x)
        
        assert torch.allclose(residual1, residual2)
