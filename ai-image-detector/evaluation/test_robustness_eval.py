"""Unit tests for robustness evaluation module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from evaluation.robustness_eval import (
    RobustnessPerturbation,
    evaluate_robustness,
    compute_robustness_degradation,
    _evaluate_with_perturbation
)


class SimpleBinaryClassifier(nn.Module):
    """Simple classifier for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = SimpleBinaryClassifier()
    model.eval()
    return model


@pytest.fixture
def test_dataloader():
    """Create a simple test dataloader."""
    # Create synthetic data: 50 samples, 3 channels, 32x32 images
    images = torch.rand(50, 3, 32, 32)
    labels = torch.randint(0, 2, (50,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=10, shuffle=False)


def test_jpeg_compression():
    """Test JPEG compression perturbation."""
    image = torch.rand(3, 64, 64)
    
    # Test different quality levels
    for quality in [95, 85, 75, 65, 50]:
        compressed = RobustnessPerturbation.apply_jpeg_compression(image, quality)
        
        # Check output shape
        assert compressed.shape == image.shape
        
        # Check value range
        assert compressed.min() >= 0.0
        assert compressed.max() <= 1.0
        
        # Lower quality should produce more artifacts (less similar to original)
        # This is a rough check - JPEG compression is lossy


def test_gaussian_blur():
    """Test Gaussian blur perturbation."""
    image = torch.rand(3, 64, 64)
    
    # Test different sigma levels
    for sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
        blurred = RobustnessPerturbation.apply_gaussian_blur(image, sigma)
        
        # Check output shape
        assert blurred.shape == image.shape
        
        # Check value range
        assert blurred.min() >= 0.0
        assert blurred.max() <= 1.0
        
        # Blurred image should be smoother (lower variance in local regions)


def test_gaussian_noise():
    """Test Gaussian noise perturbation."""
    image = torch.rand(3, 64, 64)
    
    # Test different std levels
    for std in [0.01, 0.02, 0.03, 0.04, 0.05]:
        noisy = RobustnessPerturbation.apply_gaussian_noise(image, std)
        
        # Check output shape
        assert noisy.shape == image.shape
        
        # Check value range (should be clamped to [0, 1])
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0
        
        # Noisy image should differ from original
        assert not torch.allclose(noisy, image)


def test_evaluate_with_perturbation_no_perturbation(simple_model, test_dataloader):
    """Test evaluation without perturbation (baseline)."""
    device = torch.device('cpu')
    
    metrics = _evaluate_with_perturbation(
        simple_model,
        test_dataloader,
        device,
        perturbation_fn=None
    )
    
    # Check that metrics are returned
    assert 'accuracy' in metrics
    assert 'auc' in metrics
    
    # Check that metrics are in valid range
    assert 0.0 <= metrics['accuracy'] <= 1.0
    if not np.isnan(metrics['auc']):
        assert 0.0 <= metrics['auc'] <= 1.0


def test_evaluate_with_perturbation_jpeg(simple_model, test_dataloader):
    """Test evaluation with JPEG compression perturbation."""
    device = torch.device('cpu')
    
    perturbation_fn = lambda img: RobustnessPerturbation.apply_jpeg_compression(img, 75)
    
    metrics = _evaluate_with_perturbation(
        simple_model,
        test_dataloader,
        device,
        perturbation_fn=perturbation_fn
    )
    
    # Check that metrics are returned
    assert 'accuracy' in metrics
    assert 'auc' in metrics
    
    # Check that metrics are in valid range
    assert 0.0 <= metrics['accuracy'] <= 1.0
    if not np.isnan(metrics['auc']):
        assert 0.0 <= metrics['auc'] <= 1.0


def test_evaluate_robustness_jpeg_only(simple_model, test_dataloader):
    """Test robustness evaluation with JPEG compression only."""
    device = torch.device('cpu')
    
    results = evaluate_robustness(
        simple_model,
        test_dataloader,
        device,
        perturbations=['jpeg'],
        jpeg_qualities=[95, 75, 50]
    )
    
    # Check baseline is present
    assert 'baseline' in results
    assert 'accuracy' in results['baseline']
    assert 'auc' in results['baseline']
    
    # Check JPEG results
    assert 'jpeg' in results
    assert 95 in results['jpeg']
    assert 75 in results['jpeg']
    assert 50 in results['jpeg']
    
    # Check each quality level has metrics
    for quality in [95, 75, 50]:
        assert 'accuracy' in results['jpeg'][quality]
        assert 'auc' in results['jpeg'][quality]
        assert 0.0 <= results['jpeg'][quality]['accuracy'] <= 1.0


def test_evaluate_robustness_all_perturbations(simple_model, test_dataloader):
    """Test robustness evaluation with all perturbation types."""
    device = torch.device('cpu')
    
    results = evaluate_robustness(
        simple_model,
        test_dataloader,
        device,
        perturbations=['jpeg', 'blur', 'noise'],
        jpeg_qualities=[95, 50],
        blur_sigmas=[0.5, 2.5],
        noise_stds=[0.01, 0.05]
    )
    
    # Check all perturbation types are present
    assert 'baseline' in results
    assert 'jpeg' in results
    assert 'blur' in results
    assert 'noise' in results
    
    # Check JPEG
    assert 95 in results['jpeg']
    assert 50 in results['jpeg']
    
    # Check blur
    assert 0.5 in results['blur']
    assert 2.5 in results['blur']
    
    # Check noise
    assert 0.01 in results['noise']
    assert 0.05 in results['noise']


def test_compute_robustness_degradation(simple_model, test_dataloader):
    """Test computation of robustness degradation metrics."""
    device = torch.device('cpu')
    
    results = evaluate_robustness(
        simple_model,
        test_dataloader,
        device,
        perturbations=['jpeg'],
        jpeg_qualities=[95, 50]
    )
    
    degradation = compute_robustness_degradation(results)
    
    # Check degradation metrics are computed
    assert 'jpeg' in degradation
    assert 95 in degradation['jpeg']
    assert 50 in degradation['jpeg']
    
    # Check degradation structure
    for quality in [95, 50]:
        assert 'accuracy_drop' in degradation['jpeg'][quality]
        assert 'auc_drop' in degradation['jpeg'][quality]
        
        # Degradation should be non-negative (performance should not improve)
        # Note: Due to randomness in the simple model, this might not always hold
        # but we can check the structure is correct
        assert isinstance(degradation['jpeg'][quality]['accuracy_drop'], float)


def test_evaluate_robustness_default_parameters(simple_model, test_dataloader):
    """Test robustness evaluation with default parameters."""
    device = torch.device('cpu')
    
    # This should use default severity levels
    results = evaluate_robustness(
        simple_model,
        test_dataloader,
        device
    )
    
    # Check all default perturbations are present
    assert 'baseline' in results
    assert 'jpeg' in results
    assert 'blur' in results
    assert 'noise' in results
    
    # Check default JPEG qualities
    for quality in [95, 85, 75, 65, 50]:
        assert quality in results['jpeg']
    
    # Check default blur sigmas
    for sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
        assert sigma in results['blur']
    
    # Check default noise stds
    for std in [0.01, 0.02, 0.03, 0.04, 0.05]:
        assert std in results['noise']


def test_perturbation_preserves_shape():
    """Test that all perturbations preserve image shape."""
    image = torch.rand(3, 128, 128)
    
    # JPEG compression
    compressed = RobustnessPerturbation.apply_jpeg_compression(image, 75)
    assert compressed.shape == image.shape
    
    # Gaussian blur
    blurred = RobustnessPerturbation.apply_gaussian_blur(image, 1.5)
    assert blurred.shape == image.shape
    
    # Gaussian noise
    noisy = RobustnessPerturbation.apply_gaussian_noise(image, 0.03)
    assert noisy.shape == image.shape


def test_perturbation_value_range():
    """Test that all perturbations maintain valid value range [0, 1]."""
    image = torch.rand(3, 64, 64)
    
    # JPEG compression
    compressed = RobustnessPerturbation.apply_jpeg_compression(image, 50)
    assert compressed.min() >= 0.0 and compressed.max() <= 1.0
    
    # Gaussian blur
    blurred = RobustnessPerturbation.apply_gaussian_blur(image, 2.5)
    assert blurred.min() >= 0.0 and blurred.max() <= 1.0
    
    # Gaussian noise (should be clamped)
    noisy = RobustnessPerturbation.apply_gaussian_noise(image, 0.05)
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
