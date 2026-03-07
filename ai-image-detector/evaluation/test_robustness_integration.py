"""Integration test for robustness evaluation module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from evaluation.robustness_eval import (
    evaluate_robustness,
    print_robustness_report,
    compute_robustness_degradation
)


class MockClassifier(nn.Module):
    """Mock classifier that returns deterministic predictions."""
    
    def __init__(self, base_accuracy=0.9):
        super().__init__()
        self.base_accuracy = base_accuracy
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Use actual forward pass for realistic behavior
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MockClassifier(base_accuracy=0.9)
    model.eval()
    return model


@pytest.fixture
def test_dataset():
    """Create a test dataset with balanced classes."""
    # Create 100 samples: 50 real (label=0), 50 fake (label=1)
    images = torch.rand(100, 3, 64, 64)
    labels = torch.cat([torch.zeros(50), torch.ones(50)])
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=20, shuffle=False)


def test_full_robustness_evaluation_workflow(mock_model, test_dataset):
    """Test the complete robustness evaluation workflow."""
    device = torch.device('cpu')
    
    # Run robustness evaluation with subset of perturbations
    results = evaluate_robustness(
        mock_model,
        test_dataset,
        device,
        perturbations=['jpeg', 'blur'],
        jpeg_qualities=[95, 75, 50],
        blur_sigmas=[0.5, 1.5, 2.5]
    )
    
    # Verify results structure
    assert 'baseline' in results
    assert 'jpeg' in results
    assert 'blur' in results
    
    # Verify baseline metrics
    assert 'accuracy' in results['baseline']
    assert 'auc' in results['baseline']
    assert 0.0 <= results['baseline']['accuracy'] <= 1.0
    
    # Verify JPEG results
    for quality in [95, 75, 50]:
        assert quality in results['jpeg']
        assert 'accuracy' in results['jpeg'][quality]
        assert 'auc' in results['jpeg'][quality]
    
    # Verify blur results
    for sigma in [0.5, 1.5, 2.5]:
        assert sigma in results['blur']
        assert 'accuracy' in results['blur'][sigma]
        assert 'auc' in results['blur'][sigma]
    
    # Compute degradation
    degradation = compute_robustness_degradation(results)
    
    # Verify degradation structure
    assert 'jpeg' in degradation
    assert 'blur' in degradation
    
    for quality in [95, 75, 50]:
        assert quality in degradation['jpeg']
        assert 'accuracy_drop' in degradation['jpeg'][quality]
        assert 'auc_drop' in degradation['jpeg'][quality]
    
    # Print report (should not raise errors)
    print_robustness_report(results)


def test_robustness_evaluation_with_all_perturbations(mock_model, test_dataset):
    """Test robustness evaluation with all perturbation types."""
    device = torch.device('cpu')
    
    # Run with all default perturbations
    results = evaluate_robustness(
        mock_model,
        test_dataset,
        device,
        perturbations=['jpeg', 'blur', 'noise'],
        jpeg_qualities=[95, 50],
        blur_sigmas=[0.5, 2.5],
        noise_stds=[0.01, 0.05]
    )
    
    # Verify all perturbation types are present
    assert 'baseline' in results
    assert 'jpeg' in results
    assert 'blur' in results
    assert 'noise' in results
    
    # Verify each perturbation has expected severity levels
    assert set(results['jpeg'].keys()) == {95, 50}
    assert set(results['blur'].keys()) == {0.5, 2.5}
    assert set(results['noise'].keys()) == {0.01, 0.05}
    
    # Print report
    print_robustness_report(results)


def test_robustness_evaluation_single_perturbation(mock_model, test_dataset):
    """Test robustness evaluation with a single perturbation type."""
    device = torch.device('cpu')
    
    # Test with only JPEG compression
    results = evaluate_robustness(
        mock_model,
        test_dataset,
        device,
        perturbations=['jpeg'],
        jpeg_qualities=[95, 85, 75, 65, 50]
    )
    
    # Verify only JPEG is present (besides baseline)
    assert 'baseline' in results
    assert 'jpeg' in results
    assert 'blur' not in results
    assert 'noise' not in results
    
    # Verify all quality levels
    for quality in [95, 85, 75, 65, 50]:
        assert quality in results['jpeg']
    
    # Compute and verify degradation
    degradation = compute_robustness_degradation(results)
    assert 'jpeg' in degradation
    assert 'blur' not in degradation
    assert 'noise' not in degradation


def test_robustness_report_formatting(mock_model, test_dataset, capsys):
    """Test that robustness report is properly formatted."""
    device = torch.device('cpu')
    
    results = evaluate_robustness(
        mock_model,
        test_dataset,
        device,
        perturbations=['jpeg'],
        jpeg_qualities=[95, 50]
    )
    
    # Print report and capture output
    print_robustness_report(results)
    captured = capsys.readouterr()
    
    # Verify report contains expected sections
    assert "ROBUSTNESS EVALUATION REPORT" in captured.out
    assert "Baseline (No Perturbation)" in captured.out
    assert "JPEG Compression" in captured.out
    assert "Quality 95" in captured.out
    assert "Quality 50" in captured.out
    assert "Accuracy:" in captured.out
    assert "AUC:" in captured.out


def test_degradation_computation_accuracy(mock_model, test_dataset):
    """Test that degradation is computed correctly."""
    device = torch.device('cpu')
    
    results = evaluate_robustness(
        mock_model,
        test_dataset,
        device,
        perturbations=['jpeg'],
        jpeg_qualities=[95, 50]
    )
    
    degradation = compute_robustness_degradation(results)
    
    baseline_acc = results['baseline']['accuracy']
    
    # Verify degradation is computed as baseline - perturbed
    for quality in [95, 50]:
        expected_drop = baseline_acc - results['jpeg'][quality]['accuracy']
        actual_drop = degradation['jpeg'][quality]['accuracy_drop']
        assert abs(expected_drop - actual_drop) < 1e-6


def test_evaluation_with_model_returning_tuple(test_dataset):
    """Test evaluation with model that returns tuple (prediction, attribution)."""
    
    class ModelWithAttribution(nn.Module):
        """Model that returns both prediction and attribution."""
        
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 1)
            self.attribution_fc = nn.Linear(16, 5)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            prediction = self.sigmoid(self.fc(x))
            attribution = self.softmax(self.attribution_fc(x))
            return prediction, attribution
    
    model = ModelWithAttribution()
    model.eval()
    device = torch.device('cpu')
    
    # Should handle tuple output correctly
    results = evaluate_robustness(
        model,
        test_dataset,
        device,
        perturbations=['jpeg'],
        jpeg_qualities=[95]
    )
    
    # Verify evaluation completed successfully
    assert 'baseline' in results
    assert 'jpeg' in results
    assert 95 in results['jpeg']
