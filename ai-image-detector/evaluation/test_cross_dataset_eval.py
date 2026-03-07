"""
Unit tests for cross-dataset evaluation module.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from evaluation.cross_dataset_eval import (
    evaluate_cross_dataset,
    generate_performance_matrix,
    compute_cross_dataset_variance,
    _evaluate_single_dataset
)


class SimpleBinaryClassifier(nn.Module):
    """Simple classifier for testing."""
    
    def __init__(self, return_attribution=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)
        self.return_attribution = return_attribution
        
        if return_attribution:
            self.attribution_head = nn.Linear(16, 5)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        output = torch.sigmoid(self.fc(x))
        
        if self.return_attribution:
            attribution = torch.softmax(self.attribution_head(x), dim=1)
            return output, attribution
        return output


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = SimpleBinaryClassifier()
    model.eval()
    return model


@pytest.fixture
def model_with_attribution():
    """Create a model that returns attribution."""
    model = SimpleBinaryClassifier(return_attribution=True)
    model.eval()
    return model


@pytest.fixture
def dataset_loaders():
    """Create multiple dataset loaders for testing."""
    # Create synthetic datasets
    loaders = {}
    
    # Dataset 1: 100 samples
    images1 = torch.randn(100, 3, 32, 32)
    labels1 = torch.randint(0, 2, (100,))
    dataset1 = TensorDataset(images1, labels1)
    loaders['dataset1'] = DataLoader(dataset1, batch_size=16, shuffle=False)
    
    # Dataset 2: 80 samples
    images2 = torch.randn(80, 3, 32, 32)
    labels2 = torch.randint(0, 2, (80,))
    dataset2 = TensorDataset(images2, labels2)
    loaders['dataset2'] = DataLoader(dataset2, batch_size=16, shuffle=False)
    
    # Dataset 3: 120 samples
    images3 = torch.randn(120, 3, 32, 32)
    labels3 = torch.randint(0, 2, (120,))
    dataset3 = TensorDataset(images3, labels3)
    loaders['dataset3'] = DataLoader(dataset3, batch_size=16, shuffle=False)
    
    return loaders


def test_evaluate_single_dataset(simple_model, device):
    """Test evaluation on a single dataset."""
    # Create test dataset
    images = torch.randn(50, 3, 32, 32)
    labels = torch.randint(0, 2, (50,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    metrics = _evaluate_single_dataset(simple_model, loader, device)
    
    # Check metrics exist
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics
    assert 'num_samples' in metrics
    assert 'confusion_matrix' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert metrics['num_samples'] == 50
    
    # Check confusion matrix shape
    cm = np.array(metrics['confusion_matrix'])
    assert cm.shape == (2, 2)


def test_evaluate_single_dataset_with_attribution(model_with_attribution, device):
    """Test evaluation with model that returns attribution."""
    # Create test dataset
    images = torch.randn(50, 3, 32, 32)
    labels = torch.randint(0, 2, (50,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate (should handle tuple output)
    metrics = _evaluate_single_dataset(model_with_attribution, loader, device)
    
    # Check metrics exist
    assert 'accuracy' in metrics
    assert 'num_samples' in metrics
    assert metrics['num_samples'] == 50


def test_evaluate_cross_dataset(simple_model, dataset_loaders, device):
    """Test cross-dataset evaluation."""
    results = evaluate_cross_dataset(simple_model, dataset_loaders, device)
    
    # Check all datasets are evaluated
    assert 'dataset1' in results
    assert 'dataset2' in results
    assert 'dataset3' in results
    
    # Check each dataset has metrics
    for dataset_name, metrics in results.items():
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        assert 'num_samples' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    # Check sample counts
    assert results['dataset1']['num_samples'] == 100
    assert results['dataset2']['num_samples'] == 80
    assert results['dataset3']['num_samples'] == 120


def test_generate_performance_matrix(simple_model, dataset_loaders, device):
    """Test performance matrix generation."""
    results = evaluate_cross_dataset(simple_model, dataset_loaders, device)
    matrix = generate_performance_matrix(results)
    
    # Check metrics are present
    assert 'accuracy' in matrix
    assert 'precision' in matrix
    assert 'recall' in matrix
    assert 'f1' in matrix
    assert 'auc' in matrix
    
    # Check each metric has all datasets
    for metric, dataset_values in matrix.items():
        assert 'dataset1' in dataset_values
        assert 'dataset2' in dataset_values
        assert 'dataset3' in dataset_values


def test_generate_performance_matrix_custom_metrics(simple_model, dataset_loaders, device):
    """Test performance matrix with custom metrics."""
    results = evaluate_cross_dataset(simple_model, dataset_loaders, device)
    matrix = generate_performance_matrix(results, metrics=['accuracy', 'f1'])
    
    # Check only requested metrics are present
    assert 'accuracy' in matrix
    assert 'f1' in matrix
    assert 'precision' not in matrix
    assert 'recall' not in matrix


def test_compute_cross_dataset_variance(simple_model, dataset_loaders, device):
    """Test variance computation across datasets."""
    results = evaluate_cross_dataset(simple_model, dataset_loaders, device)
    variance = compute_cross_dataset_variance(results)
    
    # Check metrics are present
    assert 'accuracy' in variance
    assert 'precision' in variance
    assert 'recall' in variance
    assert 'f1' in variance
    assert 'auc' in variance
    
    # Check statistics for each metric
    for metric, stats in variance.items():
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'range' in stats
        
        # Check ranges (if not NaN)
        if not np.isnan(stats['mean']):
            assert 0 <= stats['mean'] <= 1
            assert stats['std'] >= 0
            assert 0 <= stats['min'] <= 1
            assert 0 <= stats['max'] <= 1
            assert stats['range'] >= 0
            assert stats['range'] == stats['max'] - stats['min']


def test_compute_cross_dataset_variance_custom_metrics(simple_model, dataset_loaders, device):
    """Test variance computation with custom metrics."""
    results = evaluate_cross_dataset(simple_model, dataset_loaders, device)
    variance = compute_cross_dataset_variance(results, metrics=['accuracy', 'f1'])
    
    # Check only requested metrics are present
    assert 'accuracy' in variance
    assert 'f1' in variance
    assert 'precision' not in variance
    assert 'recall' not in variance


def test_single_class_dataset(simple_model, device):
    """Test evaluation on dataset with single class (AUC should be NaN)."""
    # Create dataset with only class 0
    images = torch.randn(50, 3, 32, 32)
    labels = torch.zeros(50, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    metrics = _evaluate_single_dataset(simple_model, loader, device)
    
    # AUC should be NaN for single class
    assert np.isnan(metrics['auc'])
    
    # Other metrics should still be valid
    assert 'accuracy' in metrics
    assert 'num_samples' in metrics


def test_empty_dataset_loader(simple_model, device):
    """Test evaluation with empty dataset."""
    # Create empty dataset
    images = torch.randn(0, 3, 32, 32)
    labels = torch.zeros(0, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    metrics = _evaluate_single_dataset(simple_model, loader, device)
    
    # Should handle empty dataset gracefully
    assert metrics['num_samples'] == 0


def test_perfect_predictions(device):
    """Test evaluation with perfect predictions."""
    # Create a model that always predicts correctly
    class PerfectModel(nn.Module):
        def forward(self, x):
            # Return labels directly (cheating for test purposes)
            batch_size = x.size(0)
            return torch.ones(batch_size, 1)
    
    model = PerfectModel()
    model.eval()
    
    # Create dataset with all positive labels
    images = torch.randn(50, 3, 32, 32)
    labels = torch.ones(50, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    metrics = _evaluate_single_dataset(model, loader, device)
    
    # Should have perfect metrics
    assert metrics['accuracy'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1'] == 1.0


def test_dataloader_with_dataset_name(simple_model, device):
    """Test evaluation with DataLoader that returns (images, labels, dataset_name)."""
    # Create dataset with dataset name
    images = torch.randn(50, 3, 32, 32)
    labels = torch.randint(0, 2, (50,))
    dataset_names = ['test'] * 50
    dataset = TensorDataset(images, labels, torch.tensor([0] * 50))  # Dummy third element
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate (should handle 3-element tuples)
    metrics = _evaluate_single_dataset(simple_model, loader, device)
    
    # Check metrics exist
    assert 'accuracy' in metrics
    assert 'num_samples' in metrics


def test_confusion_matrix_values(device):
    """Test that confusion matrix values are correct."""
    # Create a deterministic model
    class DeterministicModel(nn.Module):
        def __init__(self, threshold=0.5):
            super().__init__()
            self.threshold = threshold
        
        def forward(self, x):
            # Return constant predictions
            batch_size = x.size(0)
            return torch.full((batch_size, 1), self.threshold + 0.1)
    
    model = DeterministicModel(threshold=0.5)
    model.eval()
    
    # Create balanced dataset
    images = torch.randn(100, 3, 32, 32)
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    metrics = _evaluate_single_dataset(model, loader, device)
    
    # Model predicts all positive (threshold + 0.1 > 0.5)
    # So: TN=0, FP=50, FN=0, TP=50
    cm = np.array(metrics['confusion_matrix'])
    assert cm[0, 0] == 0  # TN
    assert cm[0, 1] == 50  # FP
    assert cm[1, 0] == 0  # FN
    assert cm[1, 1] == 50  # TP


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
