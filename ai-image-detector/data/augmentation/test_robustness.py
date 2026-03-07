"""
Unit tests for RobustnessAugmentation module.
"""

import pytest
import torch
from PIL import Image

from .robustness import RobustnessAugmentation


class TestRobustnessAugmentation:
    """Test suite for RobustnessAugmentation."""
    
    def test_initialization_valid_params(self):
        """Test initialization with valid parameters."""
        aug = RobustnessAugmentation(
            jpeg_prob=0.5,
            blur_prob=0.3,
            noise_prob=0.2,
            severity_range=(1, 5)
        )
        assert aug.jpeg_prob == 0.5
        assert aug.blur_prob == 0.3
        assert aug.noise_prob == 0.2
        assert aug.severity_range == (1, 5)
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        aug = RobustnessAugmentation()
        assert aug.jpeg_prob == 0.3
        assert aug.blur_prob == 0.3
        assert aug.noise_prob == 0.3
        assert aug.severity_range == (1, 5)
    
    def test_initialization_invalid_jpeg_prob(self):
        """Test initialization with invalid JPEG probability."""
        with pytest.raises(ValueError, match="jpeg_prob must be in"):
            RobustnessAugmentation(jpeg_prob=1.5)
        
        with pytest.raises(ValueError, match="jpeg_prob must be in"):
            RobustnessAugmentation(jpeg_prob=-0.1)
    
    def test_initialization_invalid_blur_prob(self):
        """Test initialization with invalid blur probability."""
        with pytest.raises(ValueError, match="blur_prob must be in"):
            RobustnessAugmentation(blur_prob=2.0)
    
    def test_initialization_invalid_noise_prob(self):
        """Test initialization with invalid noise probability."""
        with pytest.raises(ValueError, match="noise_prob must be in"):
            RobustnessAugmentation(noise_prob=-0.5)
    
    def test_initialization_invalid_severity_range(self):
        """Test initialization with invalid severity range."""
        with pytest.raises(ValueError, match="severity_range must be in"):
            RobustnessAugmentation(severity_range=(0, 5))
        
        with pytest.raises(ValueError, match="severity_range must be in"):
            RobustnessAugmentation(severity_range=(1, 6))
        
        with pytest.raises(ValueError, match="severity_range must be in"):
            RobustnessAugmentation(severity_range=(3, 2))
    
    def test_jpeg_compression_severity_levels(self):
        """Test JPEG compression at all severity levels."""
        aug = RobustnessAugmentation(jpeg_prob=1.0, blur_prob=0.0, noise_prob=0.0)
        image = torch.rand(3, 64, 64)
        
        for severity in range(1, 6):
            compressed = aug._apply_jpeg_compression(image, severity)
            assert compressed.shape == image.shape
            assert compressed.dtype == image.dtype
            assert 0.0 <= compressed.min() <= compressed.max() <= 1.0
    
    def test_jpeg_compression_quality_mapping(self):
        """Test that JPEG quality levels are correctly mapped."""
        assert RobustnessAugmentation.JPEG_QUALITY[1] == 95
        assert RobustnessAugmentation.JPEG_QUALITY[2] == 85
        assert RobustnessAugmentation.JPEG_QUALITY[3] == 75
        assert RobustnessAugmentation.JPEG_QUALITY[4] == 65
        assert RobustnessAugmentation.JPEG_QUALITY[5] == 50
    
    def test_gaussian_blur_severity_levels(self):
        """Test Gaussian blur at all severity levels."""
        aug = RobustnessAugmentation(jpeg_prob=0.0, blur_prob=1.0, noise_prob=0.0)
        image = torch.rand(3, 64, 64)
        
        for severity in range(1, 6):
            blurred = aug._apply_gaussian_blur(image, severity)
            assert blurred.shape == image.shape
            assert blurred.dtype == image.dtype
            assert 0.0 <= blurred.min() <= blurred.max() <= 1.0
    
    def test_blur_sigma_mapping(self):
        """Test that blur sigma levels are correctly mapped."""
        assert RobustnessAugmentation.BLUR_SIGMA[1] == 0.5
        assert RobustnessAugmentation.BLUR_SIGMA[2] == 1.0
        assert RobustnessAugmentation.BLUR_SIGMA[3] == 1.5
        assert RobustnessAugmentation.BLUR_SIGMA[4] == 2.0
        assert RobustnessAugmentation.BLUR_SIGMA[5] == 2.5
    
    def test_gaussian_noise_severity_levels(self):
        """Test Gaussian noise at all severity levels."""
        aug = RobustnessAugmentation(jpeg_prob=0.0, blur_prob=0.0, noise_prob=1.0)
        image = torch.rand(3, 64, 64)
        
        for severity in range(1, 6):
            noisy = aug._apply_gaussian_noise(image, severity)
            assert noisy.shape == image.shape
            assert noisy.dtype == image.dtype
            assert 0.0 <= noisy.min() <= noisy.max() <= 1.0
    
    def test_noise_std_mapping(self):
        """Test that noise std levels are correctly mapped."""
        assert RobustnessAugmentation.NOISE_STD[1] == 0.01
        assert RobustnessAugmentation.NOISE_STD[2] == 0.02
        assert RobustnessAugmentation.NOISE_STD[3] == 0.03
        assert RobustnessAugmentation.NOISE_STD[4] == 0.04
        assert RobustnessAugmentation.NOISE_STD[5] == 0.05
    
    def test_gaussian_noise_clamping(self):
        """Test that Gaussian noise properly clamps values to [0, 1]."""
        aug = RobustnessAugmentation(jpeg_prob=0.0, blur_prob=0.0, noise_prob=1.0)
        
        # Create image with values near boundaries
        image = torch.ones(3, 32, 32) * 0.95
        
        # Apply severe noise multiple times
        for _ in range(10):
            noisy = aug._apply_gaussian_noise(image, severity=5)
            assert noisy.min() >= 0.0
            assert noisy.max() <= 1.0
    
    def test_single_image_augmentation(self):
        """Test augmentation on single image (3D tensor)."""
        aug = RobustnessAugmentation(jpeg_prob=1.0, blur_prob=1.0, noise_prob=1.0)
        image = torch.rand(3, 64, 64)
        
        augmented = aug(image)
        assert augmented.shape == image.shape
        assert augmented.dtype == image.dtype
        assert 0.0 <= augmented.min() <= augmented.max() <= 1.0
    
    def test_batch_augmentation(self):
        """Test augmentation on batch of images (4D tensor)."""
        aug = RobustnessAugmentation(jpeg_prob=1.0, blur_prob=1.0, noise_prob=1.0)
        batch = torch.rand(4, 3, 64, 64)
        
        augmented = aug(batch)
        assert augmented.shape == batch.shape
        assert augmented.dtype == batch.dtype
        assert 0.0 <= augmented.min() <= augmented.max() <= 1.0
    
    def test_invalid_tensor_dimension(self):
        """Test that invalid tensor dimensions raise error."""
        aug = RobustnessAugmentation()
        
        # 2D tensor should fail
        with pytest.raises(ValueError, match="Expected 3D or 4D tensor"):
            aug(torch.rand(64, 64))
        
        # 5D tensor should fail
        with pytest.raises(ValueError, match="Expected 3D or 4D tensor"):
            aug(torch.rand(2, 3, 3, 64, 64))
    
    def test_no_augmentation_with_zero_probabilities(self):
        """Test that no augmentation is applied when all probabilities are 0."""
        aug = RobustnessAugmentation(jpeg_prob=0.0, blur_prob=0.0, noise_prob=0.0)
        image = torch.rand(3, 64, 64)
        
        augmented = aug(image)
        # Should be identical (no augmentation applied)
        assert torch.allclose(augmented, image, atol=1e-6)
    
    def test_severity_range_respected(self):
        """Test that severity range is respected."""
        # Test with limited severity range
        aug = RobustnessAugmentation(
            jpeg_prob=1.0,
            blur_prob=0.0,
            noise_prob=0.0,
            severity_range=(3, 3)  # Only severity 3
        )
        image = torch.rand(3, 64, 64)
        
        # Apply multiple times to ensure consistency
        for _ in range(5):
            augmented = aug(image)
            assert augmented.shape == image.shape
    
    def test_augmentation_changes_image(self):
        """Test that augmentation actually modifies the image."""
        aug = RobustnessAugmentation(jpeg_prob=1.0, blur_prob=1.0, noise_prob=1.0)
        image = torch.rand(3, 64, 64)
        
        augmented = aug(image)
        # Should be different (augmentation applied)
        assert not torch.allclose(augmented, image, atol=1e-3)
    
    def test_jpeg_compression_reduces_quality(self):
        """Test that JPEG compression at higher severity reduces quality more."""
        aug = RobustnessAugmentation(jpeg_prob=1.0, blur_prob=0.0, noise_prob=0.0)
        
        # Create a high-frequency pattern (checkerboard)
        image = torch.zeros(3, 64, 64)
        image[:, ::2, ::2] = 1.0
        image[:, 1::2, 1::2] = 1.0
        
        # Apply different severity levels
        compressed_mild = aug._apply_jpeg_compression(image, severity=1)
        compressed_severe = aug._apply_jpeg_compression(image, severity=5)
        
        # Severe compression should differ more from original
        diff_mild = torch.abs(image - compressed_mild).mean()
        diff_severe = torch.abs(image - compressed_severe).mean()
        
        assert diff_severe > diff_mild
    
    def test_blur_increases_with_severity(self):
        """Test that blur effect increases with severity."""
        aug = RobustnessAugmentation(jpeg_prob=0.0, blur_prob=1.0, noise_prob=0.0)
        
        # Create a sharp edge
        image = torch.zeros(3, 64, 64)
        image[:, :, :32] = 1.0
        
        # Apply different severity levels
        blurred_mild = aug._apply_gaussian_blur(image, severity=1)
        blurred_severe = aug._apply_gaussian_blur(image, severity=5)
        
        # Severe blur should differ more from original
        diff_mild = torch.abs(image - blurred_mild).mean()
        diff_severe = torch.abs(image - blurred_severe).mean()
        
        assert diff_severe > diff_mild
    
    def test_noise_increases_with_severity(self):
        """Test that noise effect increases with severity (statistically)."""
        aug = RobustnessAugmentation(jpeg_prob=0.0, blur_prob=0.0, noise_prob=1.0)
        
        # Create uniform image
        image = torch.ones(3, 64, 64) * 0.5
        
        # Apply different severity levels multiple times and average
        diffs_mild = []
        diffs_severe = []
        
        for _ in range(10):
            noisy_mild = aug._apply_gaussian_noise(image, severity=1)
            noisy_severe = aug._apply_gaussian_noise(image, severity=5)
            
            diffs_mild.append(torch.abs(image - noisy_mild).mean().item())
            diffs_severe.append(torch.abs(image - noisy_severe).mean().item())
        
        # On average, severe noise should have larger difference
        assert sum(diffs_severe) / len(diffs_severe) > sum(diffs_mild) / len(diffs_mild)
    
    def test_output_range_preservation(self):
        """Test that output values stay in valid range [0, 1]."""
        aug = RobustnessAugmentation(jpeg_prob=1.0, blur_prob=1.0, noise_prob=1.0)
        
        # Test with various input ranges
        for _ in range(10):
            image = torch.rand(3, 64, 64)
            augmented = aug(image)
            
            assert augmented.min() >= 0.0, f"Min value {augmented.min()} < 0"
            assert augmented.max() <= 1.0, f"Max value {augmented.max()} > 1"
    
    def test_deterministic_with_seed(self):
        """Test that augmentation is deterministic with fixed random seed (excluding JPEG)."""
        # Note: JPEG compression is not deterministic due to PIL's internal implementation
        # So we test only blur and noise which are deterministic
        aug = RobustnessAugmentation(jpeg_prob=0.0, blur_prob=0.5, noise_prob=0.5)
        image = torch.rand(3, 64, 64)
        
        # Apply with same seed
        torch.manual_seed(42)
        result1 = aug(image)
        
        torch.manual_seed(42)
        result2 = aug(image)
        
        assert torch.allclose(result1, result2, atol=1e-6)
    
    def test_different_results_without_seed(self):
        """Test that augmentation produces different results without fixed seed."""
        aug = RobustnessAugmentation(jpeg_prob=1.0, blur_prob=0.0, noise_prob=1.0)
        image = torch.rand(3, 64, 64)
        
        results = [aug(image) for _ in range(5)]
        
        # At least some results should be different
        all_same = all(torch.allclose(results[0], r, atol=1e-6) for r in results[1:])
        assert not all_same
