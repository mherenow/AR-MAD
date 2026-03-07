"""
Unit tests for any-resolution evaluation module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from evaluation.resolution_eval import (
    evaluate_any_resolution,
    generate_size_performance_matrix,
    compute_size_variance,
    _find_size_range,
    _compute_metrics
)


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, fixed_output=0.7):
        super().__init__()
        self.fixed_output = fixed_output
    
    def forward(self, x):
        """Return fixed predictions."""
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
        return torch.full((batch_size, 1), self.fixed_output)


class VariableSizeDataset(torch.utils.data.Dataset):
    """Dataset that returns variable-sized images."""
    
    def __init__(self, sizes, labels):
        """
        Args:
            sizes: List of (H, W) tuples
            labels: List of labels
        """
        self.sizes = sizes
        self.labels = labels
    
    def __len__(self):
        return len(self.sizes)
    
    def __getitem__(self, idx):
        h, w = self.sizes[idx]
        image = torch.randn(3, h, w)
        label = self.labels[idx]
        return image, label


def variable_size_collate_fn(batch):
    """Collate function for variable-sized images."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


def test_find_size_range():
    """Test _find_size_range helper function."""
    size_ranges = [(128, 256), (256, 512), (512, 1024)]
    
    # Test within ranges
    assert _find_size_range(150, size_ranges) == "128-256"
    assert _find_size_range(300, size_ranges) == "256-512"
    assert _find_size_range(700, size_ranges) == "512-1024"
    
    # Test boundaries
    assert _find_size_range(128, size_ranges) == "128-256"
    assert _find_size_range(256, size_ranges) == "256-512"
    assert _find_size_range(512, size_ranges) == "512-1024"
    
    # Test outside ranges
    assert _find_size_range(100, size_ranges) is None
    assert _find_size_range(1024, size_ranges) is None


def test_compute_metrics():
    """Test _compute_metrics helper function."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_prob = np.array([0.2, 0.3, 0.8, 0.9, 0.4, 0.1, 0.7, 0.6])
    heights = np.array([200, 200, 200, 200, 200, 200, 200, 200])
    widths = np.array([180, 180, 180, 180, 180, 180, 180, 180])
    
    metrics = _compute_metrics(y_true, y_pred, y_prob, heights, widths)
    
    # Check metric keys
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics
    assert 'num_samples' in metrics
    assert 'confusion_matrix' in metrics
    assert 'avg_height' in metrics
    assert 'avg_width' in metrics
    
    # Check values
    assert metrics['num_samples'] == 8
    assert metrics['avg_height'] == 200.0
    assert metrics['avg_width'] == 180.0
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert 0.0 <= metrics['f1'] <= 1.0
    
    # Check confusion matrix shape
    cm = np.array(metrics['confusion_matrix'])
    assert cm.shape == (2, 2)


def test_compute_metrics_single_class():
    """Test _compute_metrics with single class (edge case)."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    y_prob = np.array([0.8, 0.9, 0.7, 0.85])
    heights = np.array([200, 200, 200, 200])
    widths = np.array([200, 200, 200, 200])
    
    metrics = _compute_metrics(y_true, y_pred, y_prob, heights, widths)
    
    # AUC should be NaN for single class
    assert np.isnan(metrics['auc'])
    
    # Other metrics should still be valid
    assert metrics['accuracy'] == 1.0
    assert metrics['num_samples'] == 4


def test_evaluate_any_resolution_variable_size():
    """Test evaluate_any_resolution with variable-sized images."""
    # Create dataset with different sizes
    sizes = [
        (150, 150),  # 128-256 range
        (200, 180),  # 128-256 range
        (300, 320),  # 256-512 range
        (400, 380),  # 256-512 range
        (600, 650),  # 512-1024 range
        (800, 750),  # 512-1024 range
    ]
    labels = [0, 1, 0, 1, 0, 1]
    
    dataset = VariableSizeDataset(sizes, labels)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=variable_size_collate_fn
    )
    
    model = MockModel(fixed_output=0.7)
    device = torch.device('cpu')
    
    results = evaluate_any_resolution(model, loader, device)
    
    # Check that we have results for each size range
    assert '128-256' in results
    assert '256-512' in results
    assert '512-1024' in results
    
    # Check that each range has the expected number of samples
    assert results['128-256']['num_samples'] == 2
    assert results['256-512']['num_samples'] == 2
    assert results['512-1024']['num_samples'] == 2
    
    # Check that metrics are present
    for range_key in ['128-256', '256-512', '512-1024']:
        assert 'accuracy' in results[range_key]
        assert 'precision' in results[range_key]
        assert 'recall' in results[range_key]
        assert 'f1' in results[range_key]
        assert 'avg_height' in results[range_key]
        assert 'avg_width' in results[range_key]


def test_evaluate_any_resolution_fixed_size():
    """Test evaluate_any_resolution with fixed-size batched images."""
    # Create fixed-size dataset
    images = torch.randn(10, 3, 300, 300)  # 256-512 range
    labels = torch.randint(0, 2, (10,))
    
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=5)
    
    model = MockModel(fixed_output=0.7)
    device = torch.device('cpu')
    
    results = evaluate_any_resolution(model, loader, device)
    
    # Should only have results for 256-512 range
    assert '256-512' in results
    assert results['256-512']['num_samples'] == 10
    
    # Check average size
    assert results['256-512']['avg_height'] == 300.0
    assert results['256-512']['avg_width'] == 300.0


def test_evaluate_any_resolution_custom_ranges():
    """Test evaluate_any_resolution with custom size ranges."""
    sizes = [
        (100, 100),  # 64-128 range
        (150, 150),  # 128-256 range
        (300, 300),  # 256-512 range
    ]
    labels = [0, 1, 0]
    
    dataset = VariableSizeDataset(sizes, labels)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=variable_size_collate_fn
    )
    
    model = MockModel(fixed_output=0.7)
    device = torch.device('cpu')
    
    # Custom size ranges
    custom_ranges = [(64, 128), (128, 256), (256, 512)]
    results = evaluate_any_resolution(model, loader, device, size_ranges=custom_ranges)
    
    # Check that we have results for custom ranges
    assert '64-128' in results
    assert '128-256' in results
    assert '256-512' in results


def test_evaluate_any_resolution_outside_ranges():
    """Test that images outside all ranges are skipped."""
    sizes = [
        (50, 50),    # Too small
        (200, 200),  # 128-256 range
        (2000, 2000),  # Too large
    ]
    labels = [0, 1, 0]
    
    dataset = VariableSizeDataset(sizes, labels)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=variable_size_collate_fn
    )
    
    model = MockModel(fixed_output=0.7)
    device = torch.device('cpu')
    
    results = evaluate_any_resolution(model, loader, device)
    
    # Should only have results for 128-256 range
    assert '128-256' in results
    assert results['128-256']['num_samples'] == 1
    
    # Other ranges should not be present or have 0 samples
    assert '512-1024' not in results or results['512-1024']['num_samples'] == 0


def test_generate_size_performance_matrix():
    """Test generate_size_performance_matrix function."""
    results = {
        '128-256': {
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.96,
            'f1': 0.95,
            'auc': 0.98
        },
        '256-512': {
            'accuracy': 0.93,
            'precision': 0.92,
            'recall': 0.94,
            'f1': 0.93,
            'auc': 0.96
        }
    }
    
    matrix = generate_size_performance_matrix(results)
    
    # Check structure
    assert 'accuracy' in matrix
    assert 'precision' in matrix
    assert 'recall' in matrix
    assert 'f1' in matrix
    assert 'auc' in matrix
    
    # Check values
    assert matrix['accuracy']['128-256'] == 0.95
    assert matrix['accuracy']['256-512'] == 0.93
    assert matrix['precision']['128-256'] == 0.94
    assert matrix['f1']['256-512'] == 0.93


def test_generate_size_performance_matrix_custom_metrics():
    """Test generate_size_performance_matrix with custom metrics."""
    results = {
        '128-256': {
            'accuracy': 0.95,
            'custom_metric': 0.88
        }
    }
    
    matrix = generate_size_performance_matrix(results, metrics=['accuracy', 'custom_metric'])
    
    assert 'accuracy' in matrix
    assert 'custom_metric' in matrix
    assert matrix['accuracy']['128-256'] == 0.95
    assert matrix['custom_metric']['128-256'] == 0.88


def test_compute_size_variance():
    """Test compute_size_variance function."""
    results = {
        '128-256': {
            'accuracy': 0.95,
            'precision': 0.94,
            'f1': 0.95
        },
        '256-512': {
            'accuracy': 0.93,
            'precision': 0.92,
            'f1': 0.93
        },
        '512-1024': {
            'accuracy': 0.94,
            'precision': 0.93,
            'f1': 0.94
        }
    }
    
    variance = compute_size_variance(results)
    
    # Check structure
    assert 'accuracy' in variance
    assert 'precision' in variance
    assert 'f1' in variance
    
    # Check statistics keys
    for metric in ['accuracy', 'precision', 'f1']:
        assert 'mean' in variance[metric]
        assert 'std' in variance[metric]
        assert 'min' in variance[metric]
        assert 'max' in variance[metric]
        assert 'range' in variance[metric]
    
    # Check accuracy statistics
    acc_values = [0.95, 0.93, 0.94]
    assert abs(variance['accuracy']['mean'] - np.mean(acc_values)) < 1e-6
    assert abs(variance['accuracy']['std'] - np.std(acc_values)) < 1e-6
    assert variance['accuracy']['min'] == 0.93
    assert variance['accuracy']['max'] == 0.95
    assert abs(variance['accuracy']['range'] - 0.02) < 1e-6


def test_compute_size_variance_with_nan():
    """Test compute_size_variance with NaN values."""
    results = {
        '128-256': {
            'accuracy': 0.95,
            'auc': float('nan')  # Single class case
        },
        '256-512': {
            'accuracy': 0.93,
            'auc': 0.96
        }
    }
    
    variance = compute_size_variance(results)
    
    # Accuracy should have valid statistics
    assert not np.isnan(variance['accuracy']['mean'])
    
    # AUC should only use non-NaN values
    assert not np.isnan(variance['auc']['mean'])
    assert variance['auc']['mean'] == 0.96


def test_compute_size_variance_empty_metric():
    """Test compute_size_variance when a metric has no valid values."""
    results = {
        '128-256': {
            'accuracy': 0.95
        }
    }
    
    variance = compute_size_variance(results, metrics=['accuracy', 'missing_metric'])
    
    # Accuracy should have valid statistics
    assert not np.isnan(variance['accuracy']['mean'])
    
    # Missing metric should have NaN statistics
    assert np.isnan(variance['missing_metric']['mean'])


def test_model_with_attribution():
    """Test that evaluation handles models with attribution output."""
    class ModelWithAttribution(nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            prediction = torch.full((batch_size, 1), 0.7)
            attribution = torch.randn(batch_size, 10)
            return prediction, attribution
    
    sizes = [(200, 200), (300, 300)]
    labels = [0, 1]
    
    dataset = VariableSizeDataset(sizes, labels)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=variable_size_collate_fn
    )
    
    model = ModelWithAttribution()
    device = torch.device('cpu')
    
    # Should not raise an error
    results = evaluate_any_resolution(model, loader, device)
    
    assert '128-256' in results
    assert '256-512' in results


def test_empty_size_range():
    """Test that empty size ranges are handled correctly."""
    sizes = [(200, 200), (220, 220)]  # Only 128-256 range
    labels = [0, 1]
    
    dataset = VariableSizeDataset(sizes, labels)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=variable_size_collate_fn
    )
    
    model = MockModel(fixed_output=0.7)
    device = torch.device('cpu')
    
    results = evaluate_any_resolution(model, loader, device)
    
    # Should only have 128-256 range
    assert '128-256' in results
    assert results['128-256']['num_samples'] == 2
    
    # Other ranges should not be in results
    assert '256-512' not in results
    assert '512-1024' not in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
