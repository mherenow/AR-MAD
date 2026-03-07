"""
Unit tests for MixUp augmentation module.
"""

import pytest
import torch
import numpy as np

from .mixup import MixUpAugmentation


class TestMixUpAugmentation:
    """Test suite for MixUpAugmentation class."""
    
    def test_initialization_valid_params(self):
        """Test initialization with valid parameters."""
        aug = MixUpAugmentation(alpha=0.2, prob=0.5)
        assert aug.alpha == 0.2
        assert aug.prob == 0.5
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        aug = MixUpAugmentation()
        assert aug.alpha == 0.2
        assert aug.prob == 0.5
    
    def test_initialization_invalid_alpha(self):
        """Test initialization with invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            MixUpAugmentation(alpha=0.0)
        
        with pytest.raises(ValueError, match="alpha must be positive"):
            MixUpAugmentation(alpha=-0.5)
    
    def test_initialization_invalid_prob(self):
        """Test initialization with invalid prob raises ValueError."""
        with pytest.raises(ValueError, match="prob must be in"):
            MixUpAugmentation(prob=-0.1)
        
        with pytest.raises(ValueError, match="prob must be in"):
            MixUpAugmentation(prob=1.5)
    
    def test_sample_lambda_range(self):
        """Test that sampled lambda is in valid range [0, 1]."""
        aug = MixUpAugmentation(alpha=0.2)
        
        # Sample multiple times to check range
        for _ in range(100):
            lam = aug._sample_lambda()
            assert 0.0 <= lam <= 1.0
    
    def test_sample_lambda_distribution(self):
        """Test that lambda follows Beta distribution characteristics."""
        aug = MixUpAugmentation(alpha=0.2)
        
        # Sample many times
        samples = [aug._sample_lambda() for _ in range(1000)]
        
        # With alpha=0.2, Beta distribution should favor values near 0 and 1
        # Check that mean is around 0.5 (symmetric distribution)
        mean = np.mean(samples)
        assert 0.3 <= mean <= 0.7  # Allow some variance
    
    def test_mixup_basic_blending(self):
        """Test basic MixUp blending with known lambda."""
        aug = MixUpAugmentation(alpha=0.2, prob=1.0)  # Always apply
        
        # Create simple test images
        batch_size, channels, height, width = 2, 3, 4, 4
        image1 = torch.ones(batch_size, channels, height, width)
        image2 = torch.zeros(batch_size, channels, height, width)
        label1 = torch.ones(batch_size, 1)
        label2 = torch.zeros(batch_size, 1)
        
        # Mock _sample_lambda to return fixed value
        aug._sample_lambda = lambda: 0.7
        
        mixed_image, mixed_label = aug(image1, label1, image2, label2)
        
        # Check blending: 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert torch.allclose(mixed_image, torch.full_like(image1, 0.7))
        assert torch.allclose(mixed_label, torch.full_like(label1, 0.7))
    
    def test_mixup_label_mixing(self):
        """Test that labels are mixed correctly."""
        aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        batch_size = 4
        image1 = torch.randn(batch_size, 3, 8, 8)
        image2 = torch.randn(batch_size, 3, 8, 8)
        label1 = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        label2 = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        
        # Mock lambda
        aug._sample_lambda = lambda: 0.6
        
        _, mixed_label = aug(image1, label1, image2, label2)
        
        # Expected: 0.6 * label1 + 0.4 * label2
        expected = torch.tensor([[0.6], [0.4], [0.6], [0.4]])
        assert torch.allclose(mixed_label, expected)
    
    def test_mixup_probability_zero(self):
        """Test that MixUp is not applied when prob=0."""
        aug = MixUpAugmentation(alpha=0.2, prob=0.0)
        
        image1 = torch.randn(2, 3, 4, 4)
        image2 = torch.randn(2, 3, 4, 4)
        label1 = torch.tensor([[1.0], [0.0]])
        label2 = torch.tensor([[0.0], [1.0]])
        
        mixed_image, mixed_label = aug(image1, label1, image2, label2)
        
        # Should return original image1 and label1
        assert torch.equal(mixed_image, image1)
        assert torch.equal(mixed_label, label1)
    
    def test_mixup_1d_labels(self):
        """Test that 1D labels are handled correctly."""
        aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        image1 = torch.randn(2, 3, 4, 4)
        image2 = torch.randn(2, 3, 4, 4)
        label1 = torch.tensor([1.0, 0.0])  # 1D
        label2 = torch.tensor([0.0, 1.0])  # 1D
        
        aug._sample_lambda = lambda: 0.5
        
        mixed_image, mixed_label = aug(image1, label1, image2, label2)
        
        # Labels should be converted to 2D
        assert mixed_label.dim() == 2
        assert mixed_label.shape == (2, 1)
        assert torch.allclose(mixed_label, torch.tensor([[0.5], [0.5]]))
    
    def test_mixup_shape_validation(self):
        """Test that mismatched shapes raise ValueError."""
        aug = MixUpAugmentation()
        
        image1 = torch.randn(2, 3, 4, 4)
        image2 = torch.randn(2, 3, 8, 8)  # Different size
        label1 = torch.tensor([[1.0], [0.0]])
        label2 = torch.tensor([[0.0], [1.0]])
        
        with pytest.raises(ValueError, match="Images must have same shape"):
            aug(image1, label1, image2, label2)
    
    def test_mixup_dimension_validation(self):
        """Test that wrong dimensions raise ValueError."""
        aug = MixUpAugmentation()
        
        image1 = torch.randn(3, 4, 4)  # 3D instead of 4D
        image2 = torch.randn(3, 4, 4)
        label1 = torch.tensor([1.0])
        label2 = torch.tensor([0.0])
        
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            aug(image1, label1, image2, label2)
    
    def test_mixup_preserves_batch_size(self):
        """Test that batch size is preserved."""
        aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            image1 = torch.randn(batch_size, 3, 8, 8)
            image2 = torch.randn(batch_size, 3, 8, 8)
            label1 = torch.randn(batch_size, 1)
            label2 = torch.randn(batch_size, 1)
            
            mixed_image, mixed_label = aug(image1, label1, image2, label2)
            
            assert mixed_image.shape[0] == batch_size
            assert mixed_label.shape[0] == batch_size
    
    def test_mixup_preserves_image_dimensions(self):
        """Test that image dimensions are preserved."""
        aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        shapes = [(2, 3, 8, 8), (4, 3, 16, 16), (1, 3, 32, 32)]
        for shape in shapes:
            image1 = torch.randn(*shape)
            image2 = torch.randn(*shape)
            label1 = torch.randn(shape[0], 1)
            label2 = torch.randn(shape[0], 1)
            
            mixed_image, mixed_label = aug(image1, label1, image2, label2)
            
            assert mixed_image.shape == shape
            assert mixed_label.shape == (shape[0], 1)
    
    def test_mixup_weighted_average_property(self):
        """Test that MixUp produces a weighted average."""
        aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        image1 = torch.randn(2, 3, 4, 4)
        image2 = torch.randn(2, 3, 4, 4)
        label1 = torch.tensor([[1.0], [0.0]])
        label2 = torch.tensor([[0.0], [1.0]])
        
        # Test with different lambda values
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            aug._sample_lambda = lambda l=lam: l
            
            mixed_image, mixed_label = aug(image1, label1, image2, label2)
            
            # Verify weighted average
            expected_image = lam * image1 + (1.0 - lam) * image2
            expected_label = lam * label1 + (1.0 - lam) * label2
            
            assert torch.allclose(mixed_image, expected_image)
            assert torch.allclose(mixed_label, expected_label)
    
    def test_mixup_extreme_lambda_values(self):
        """Test MixUp with extreme lambda values (0 and 1)."""
        aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        image1 = torch.randn(2, 3, 4, 4)
        image2 = torch.randn(2, 3, 4, 4)
        label1 = torch.tensor([[1.0], [0.0]])
        label2 = torch.tensor([[0.0], [1.0]])
        
        # Lambda = 1.0 should return image1
        aug._sample_lambda = lambda: 1.0
        mixed_image, mixed_label = aug(image1, label1, image2, label2)
        assert torch.allclose(mixed_image, image1)
        assert torch.allclose(mixed_label, label1)
        
        # Lambda = 0.0 should return image2
        aug._sample_lambda = lambda: 0.0
        mixed_image, mixed_label = aug(image1, label1, image2, label2)
        assert torch.allclose(mixed_image, image2)
        assert torch.allclose(mixed_label, label2)
    
    def test_mixup_different_alpha_values(self):
        """Test MixUp with different alpha values."""
        # Test that different alpha values produce different distributions
        alphas = [0.1, 0.2, 0.5, 1.0, 2.0]
        
        for alpha in alphas:
            aug = MixUpAugmentation(alpha=alpha, prob=1.0)
            
            # Sample multiple times
            samples = [aug._sample_lambda() for _ in range(100)]
            
            # All samples should be in valid range
            assert all(0.0 <= s <= 1.0 for s in samples)
    
    def test_mixup_gradient_flow(self):
        """Test that gradients flow through MixUp operation."""
        aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        image1 = torch.randn(2, 3, 4, 4, requires_grad=True)
        image2 = torch.randn(2, 3, 4, 4, requires_grad=True)
        label1 = torch.tensor([[1.0], [0.0]])
        label2 = torch.tensor([[0.0], [1.0]])
        
        aug._sample_lambda = lambda: 0.6
        
        mixed_image, mixed_label = aug(image1, label1, image2, label2)
        
        # Compute a simple loss
        loss = mixed_image.sum()
        loss.backward()
        
        # Check that gradients exist
        assert image1.grad is not None
        assert image2.grad is not None
        
        # Check gradient values (should be proportional to lambda)
        # grad_image1 should be approximately 0.6 * ones
        # grad_image2 should be approximately 0.4 * ones
        assert torch.allclose(image1.grad, torch.full_like(image1, 0.6), atol=1e-6)
        assert torch.allclose(image2.grad, torch.full_like(image2, 0.4), atol=1e-6)
