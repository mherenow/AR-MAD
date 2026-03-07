"""
Integration tests for noise imprint detection pipeline.

Tests the complete pipeline: NoiseResidualExtractor -> NoiseImprintBranch
"""

import pytest
import torch

from .residual_extractor import NoiseResidualExtractor
from .noise_branch import NoiseImprintBranch


class TestNoiseImprintIntegration:
    """Test suite for noise imprint detection pipeline integration."""
    
    def test_gaussian_extractor_to_branch(self):
        """Test integration of Gaussian extractor with noise branch."""
        # Create extractor and branch
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256, enable_attribution=False)
        
        # Create sample image
        image = torch.rand(2, 3, 256, 256)
        
        # Extract residual
        residual = extractor(image)
        
        # Process through branch
        features = branch(residual)
        
        # Verify output
        assert features.shape == (2, 256)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_gaussian_extractor_to_branch_with_attribution(self):
        """Test integration with attribution enabled."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(
            feature_dim=256,
            enable_attribution=True,
            num_generators=5
        )
        
        image = torch.rand(2, 3, 256, 256)
        
        # Full pipeline
        residual = extractor(image)
        features, attribution = branch(residual)
        
        # Verify outputs
        assert features.shape == (2, 256)
        assert attribution.shape == (2, 5)
        
        # Verify attribution is valid probability distribution
        prob_sums = attribution.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(2), atol=1e-6)
    
    def test_end_to_end_gradient_flow(self):
        """Test that gradients flow through the entire pipeline."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256, enable_attribution=False)
        
        image = torch.rand(2, 3, 256, 256, requires_grad=True)
        
        # Forward pass
        residual = extractor(image)
        features = branch(residual)
        
        # Backward pass
        loss = features.sum()
        loss.backward()
        
        # Verify gradients exist
        assert image.grad is not None
        assert not torch.isnan(image.grad).any()
    
    def test_different_image_sizes(self):
        """Test pipeline with different image sizes."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256)
        
        sizes = [(128, 128), (256, 256), (512, 512), (224, 224)]
        
        for h, w in sizes:
            image = torch.rand(2, 3, h, w)
            residual = extractor(image)
            features = branch(residual)
            
            # Output should always be (batch_size, feature_dim)
            assert features.shape == (2, 256), f"Failed for size {h}x{w}"
    
    def test_batch_processing(self):
        """Test pipeline with different batch sizes."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256)
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            image = torch.rand(batch_size, 3, 256, 256)
            residual = extractor(image)
            features = branch(residual)
            
            assert features.shape == (batch_size, 256)
    
    def test_residual_properties(self):
        """Test that extracted residuals have expected properties."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        
        image = torch.rand(2, 3, 256, 256)
        residual = extractor(image)
        
        # Residual should have same shape as input
        assert residual.shape == image.shape
        
        # Residual should be relatively small (image - blurred_image)
        assert residual.abs().mean() < 0.5
        
        # Residual should be processable by branch
        branch = NoiseImprintBranch(feature_dim=256)
        features = branch(residual)
        assert features.shape == (2, 256)
    
    def test_eval_mode_consistency(self):
        """Test that pipeline produces consistent results in eval mode."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256)
        
        extractor.eval()
        branch.eval()
        
        image = torch.rand(2, 3, 256, 256)
        
        with torch.no_grad():
            # First pass
            residual1 = extractor(image)
            features1 = branch(residual1)
            
            # Second pass
            residual2 = extractor(image)
            features2 = branch(residual2)
        
        # Results should be identical
        assert torch.allclose(residual1, residual2)
        assert torch.allclose(features1, features2)
    
    def test_different_sigma_values(self):
        """Test pipeline with different Gaussian sigma values."""
        branch = NoiseImprintBranch(feature_dim=256)
        image = torch.rand(2, 3, 256, 256)
        
        sigma_values = [0.5, 1.0, 2.0, 3.0]
        
        for sigma in sigma_values:
            extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=sigma)
            residual = extractor(image)
            features = branch(residual)
            
            assert features.shape == (2, 256)
            assert not torch.isnan(features).any()
    
    def test_attribution_with_different_generators(self):
        """Test attribution with different numbers of generators."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        image = torch.rand(2, 3, 256, 256)
        residual = extractor(image)
        
        num_generators_list = [2, 5, 10, 20]
        
        for num_gen in num_generators_list:
            branch = NoiseImprintBranch(
                feature_dim=256,
                enable_attribution=True,
                num_generators=num_gen
            )
            features, attribution = branch(residual)
            
            assert features.shape == (2, 256)
            assert attribution.shape == (2, num_gen)
    
    def test_zero_image_input(self):
        """Test pipeline with zero image (edge case)."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256)
        
        image = torch.zeros(2, 3, 256, 256)
        
        residual = extractor(image)
        features = branch(residual)
        
        # Should handle gracefully
        assert features.shape == (2, 256)
        assert not torch.isnan(features).any()
    
    def test_uniform_image_input(self):
        """Test pipeline with uniform image (edge case)."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256)
        
        # Uniform gray image
        image = torch.ones(2, 3, 256, 256) * 0.5
        
        residual = extractor(image)
        features = branch(residual)
        
        # Should handle gracefully
        assert features.shape == (2, 256)
        assert not torch.isnan(features).any()
        
        # Residual of uniform image should be relatively small (edge artifacts exist)
        # The center should be very small, edges may have artifacts from padding
        center_residual = residual[:, :, 64:192, 64:192]
        assert center_residual.abs().max() < 0.01
    
    def test_high_frequency_image(self):
        """Test pipeline with high-frequency image (checkerboard pattern)."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256)
        
        # Create checkerboard pattern
        image = torch.zeros(2, 3, 256, 256)
        image[:, :, ::2, ::2] = 1.0
        image[:, :, 1::2, 1::2] = 1.0
        
        residual = extractor(image)
        features = branch(residual)
        
        # Should handle high-frequency content
        assert features.shape == (2, 256)
        assert not torch.isnan(features).any()
        
        # High-frequency image should produce larger residuals
        assert residual.abs().mean() > 0.01
    
    def test_device_compatibility(self):
        """Test that pipeline works on the same device."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0)
        branch = NoiseImprintBranch(feature_dim=256)
        
        image = torch.rand(2, 3, 256, 256)
        
        residual = extractor(image)
        features = branch(residual)
        
        # All tensors should be on CPU
        assert image.device.type == 'cpu'
        assert residual.device.type == 'cpu'
        assert features.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda(self):
        """Test that pipeline works on CUDA."""
        extractor = NoiseResidualExtractor(method='gaussian', gaussian_sigma=2.0).cuda()
        branch = NoiseImprintBranch(feature_dim=256).cuda()
        
        image = torch.rand(2, 3, 256, 256).cuda()
        
        residual = extractor(image)
        features = branch(residual)
        
        # All tensors should be on CUDA
        assert image.device.type == 'cuda'
        assert residual.device.type == 'cuda'
        assert features.device.type == 'cuda'
