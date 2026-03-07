"""
Unit tests for CutMix augmentation module.
"""

import pytest
import torch
import numpy as np

from .cutmix import CutMixAugmentation


class TestCutMixAugmentation:
    """Test suite for CutMixAugmentation class."""
    
    def test_initialization_valid_params(self):
        """Test initialization with valid parameters."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=0.5)
        assert cutmix.alpha == 1.0
        assert cutmix.prob == 0.5
    
    def test_initialization_invalid_alpha(self):
        """Test initialization with invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            CutMixAugmentation(alpha=0.0)
        
        with pytest.raises(ValueError, match="alpha must be positive"):
            CutMixAugmentation(alpha=-1.0)
    
    def test_initialization_invalid_prob(self):
        """Test initialization with invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="prob must be in"):
            CutMixAugmentation(alpha=1.0, prob=1.5)
        
        with pytest.raises(ValueError, match="prob must be in"):
            CutMixAugmentation(alpha=1.0, prob=-0.1)
    
    def test_sample_lambda_range(self):
        """Test that sampled lambda is in valid range [0, 1]."""
        cutmix = CutMixAugmentation(alpha=1.0)
        
        # Sample multiple times to check range
        for _ in range(100):
            lam = cutmix._sample_lambda()
            assert 0.0 <= lam <= 1.0, f"Lambda {lam} out of range [0, 1]"
    
    def test_get_bbox_valid_coordinates(self):
        """Test that bounding box coordinates are valid."""
        cutmix = CutMixAugmentation(alpha=1.0)
        width, height = 256, 256
        
        # Test with various lambda values
        for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
            x1, y1, x2, y2 = cutmix._get_bbox(width, height, lam)
            
            # Check coordinates are within bounds
            assert 0 <= x1 < width
            assert 0 <= y1 < height
            assert 0 < x2 <= width
            assert 0 < y2 <= height
            
            # Check x1 < x2 and y1 < y2
            assert x1 < x2, f"x1 ({x1}) should be less than x2 ({x2})"
            assert y1 < y2, f"y1 ({y1}) should be less than y2 ({y2})"
    
    def test_get_bbox_area_proportional_to_lambda(self):
        """Test that bounding box area is related to (1 - lambda)."""
        cutmix = CutMixAugmentation(alpha=1.0)
        width, height = 256, 256
        total_area = width * height
        
        # Test that smaller lambda produces larger cut regions on average
        lambda_values = [0.9, 0.7, 0.5, 0.3, 0.1]
        avg_areas = []
        
        for lam in lambda_values:
            areas = []
            for _ in range(50):
                x1, y1, x2, y2 = cutmix._get_bbox(width, height, lam)
                bbox_area = (x2 - x1) * (y2 - y1)
                areas.append(bbox_area / total_area)
            avg_areas.append(np.mean(areas))
        
        # Check that average area decreases as lambda increases
        # (smaller lambda -> larger cut region)
        for i in range(len(avg_areas) - 1):
            assert avg_areas[i] < avg_areas[i + 1], \
                f"Expected decreasing area with increasing lambda, but got {avg_areas}"
    
    def test_cutmix_output_shape(self):
        """Test that CutMix output has correct shape."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=1.0)  # Always apply
        
        batch_size, channels, height, width = 4, 3, 64, 64
        image1 = torch.rand(batch_size, channels, height, width)
        label1 = torch.rand(batch_size, 1)
        image2 = torch.rand(batch_size, channels, height, width)
        label2 = torch.rand(batch_size, 1)
        
        mixed_image, mixed_label = cutmix(image1, label1, image2, label2)
        
        assert mixed_image.shape == image1.shape
        assert mixed_label.shape == (batch_size, 1)
    
    def test_cutmix_with_1d_labels(self):
        """Test that CutMix handles 1D labels correctly."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        batch_size = 4
        image1 = torch.rand(batch_size, 3, 64, 64)
        label1 = torch.rand(batch_size)  # 1D labels
        image2 = torch.rand(batch_size, 3, 64, 64)
        label2 = torch.rand(batch_size)  # 1D labels
        
        mixed_image, mixed_label = cutmix(image1, label1, image2, label2)
        
        assert mixed_image.shape == image1.shape
        assert mixed_label.shape == (batch_size, 1)
    
    def test_cutmix_label_mixing(self):
        """Test that labels are mixed proportionally to area ratio."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        # Create images with distinct values
        batch_size = 2
        image1 = torch.zeros(batch_size, 3, 64, 64)
        label1 = torch.zeros(batch_size, 1)
        image2 = torch.ones(batch_size, 3, 64, 64)
        label2 = torch.ones(batch_size, 1)
        
        mixed_image, mixed_label = cutmix(image1, label1, image2, label2)
        
        # Mixed label should be between 0 and 1
        assert torch.all(mixed_label >= 0.0)
        assert torch.all(mixed_label <= 1.0)
        
        # Calculate actual area ratio from mixed image
        # Count pixels that are 1.0 (from image2)
        for i in range(batch_size):
            pixels_from_image2 = (mixed_image[i] == 1.0).all(dim=0).sum().item()
            total_pixels = 64 * 64
            actual_area_ratio = pixels_from_image2 / total_pixels
            
            # Mixed label should match area ratio
            expected_label = (1.0 - actual_area_ratio) * 0.0 + actual_area_ratio * 1.0
            assert abs(mixed_label[i].item() - expected_label) < 1e-5, \
                f"Label mixing doesn't match area ratio: {mixed_label[i].item()} vs {expected_label}"
    
    def test_cutmix_probability(self):
        """Test that CutMix is applied with correct probability."""
        # Test with prob=0.0 (never apply)
        cutmix_never = CutMixAugmentation(alpha=1.0, prob=0.0)
        
        image1 = torch.rand(2, 3, 64, 64)
        label1 = torch.tensor([[0.0], [0.0]])
        image2 = torch.rand(2, 3, 64, 64)
        label2 = torch.tensor([[1.0], [1.0]])
        
        mixed_image, mixed_label = cutmix_never(image1, label1, image2, label2)
        
        # Should return original image and label
        assert torch.allclose(mixed_image, image1)
        assert torch.allclose(mixed_label, label1)
    
    def test_cutmix_mismatched_shapes(self):
        """Test that mismatched image shapes raise ValueError."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        image1 = torch.rand(2, 3, 64, 64)
        label1 = torch.rand(2, 1)
        image2 = torch.rand(2, 3, 128, 128)  # Different size
        label2 = torch.rand(2, 1)
        
        with pytest.raises(ValueError, match="Images must have same shape"):
            cutmix(image1, label1, image2, label2)
    
    def test_cutmix_invalid_dimensions(self):
        """Test that invalid tensor dimensions raise ValueError."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        # 3D tensor instead of 4D
        image1 = torch.rand(3, 64, 64)
        label1 = torch.rand(1)
        image2 = torch.rand(3, 64, 64)
        label2 = torch.rand(1)
        
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            cutmix(image1, label1, image2, label2)
    
    def test_cutmix_preserves_image_range(self):
        """Test that CutMix preserves image value range."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        # Create images in [0, 1] range
        image1 = torch.rand(2, 3, 64, 64)
        label1 = torch.rand(2, 1)
        image2 = torch.rand(2, 3, 64, 64)
        label2 = torch.rand(2, 1)
        
        mixed_image, mixed_label = cutmix(image1, label1, image2, label2)
        
        # Check that mixed image is still in [0, 1] range
        assert torch.all(mixed_image >= 0.0)
        assert torch.all(mixed_image <= 1.0)
    
    def test_cutmix_different_alpha_values(self):
        """Test CutMix with different alpha values."""
        # Alpha affects the distribution of lambda
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            cutmix = CutMixAugmentation(alpha=alpha, prob=1.0)
            
            image1 = torch.rand(2, 3, 64, 64)
            label1 = torch.rand(2, 1)
            image2 = torch.rand(2, 3, 64, 64)
            label2 = torch.rand(2, 1)
            
            mixed_image, mixed_label = cutmix(image1, label1, image2, label2)
            
            # Should produce valid output regardless of alpha
            assert mixed_image.shape == image1.shape
            assert mixed_label.shape == label1.shape
            assert torch.all(mixed_label >= 0.0)
            assert torch.all(mixed_label <= 1.0)
    
    def test_cutmix_edge_case_small_lambda(self):
        """Test CutMix with very small lambda (large cut region)."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        # Manually set a small lambda to test edge case
        np.random.seed(42)
        torch.manual_seed(42)
        
        image1 = torch.zeros(1, 3, 64, 64)
        label1 = torch.tensor([[0.0]])
        image2 = torch.ones(1, 3, 64, 64)
        label2 = torch.tensor([[1.0]])
        
        mixed_image, mixed_label = cutmix(image1, label1, image2, label2)
        
        # Should still produce valid output
        assert mixed_image.shape == image1.shape
        assert 0.0 <= mixed_label.item() <= 1.0
    
    def test_cutmix_edge_case_large_lambda(self):
        """Test CutMix with very large lambda (small cut region)."""
        cutmix = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        # Run multiple times to potentially hit large lambda values
        for _ in range(10):
            image1 = torch.zeros(1, 3, 64, 64)
            label1 = torch.tensor([[0.0]])
            image2 = torch.ones(1, 3, 64, 64)
            label2 = torch.tensor([[1.0]])
            
            mixed_image, mixed_label = cutmix(image1, label1, image2, label2)
            
            # Should still produce valid output
            assert mixed_image.shape == image1.shape
            assert 0.0 <= mixed_label.item() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
