"""
Unit tests for noise imprint clustering analysis module.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from evaluation.noise_clustering import (
    evaluate_noise_imprint_clustering,
    print_clustering_report,
    extract_noise_features,
    compute_pairwise_separability
)


class MockNoiseExtractor(nn.Module):
    """Mock noise residual extractor for testing."""
    
    def forward(self, x):
        # Return a simple residual (just scaled input)
        return x * 0.1


class MockNoiseBranch(nn.Module):
    """Mock noise imprint branch for testing."""
    
    def __init__(self, feature_dim=256, enable_attribution=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.enable_attribution = enable_attribution
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, feature_dim)
        
        if enable_attribution:
            self.attribution_head = nn.Linear(feature_dim, 5)
    
    def forward(self, residual):
        x = self.conv(residual)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        if self.enable_attribution:
            attribution = self.attribution_head(features)
            return features, attribution
        return features


class MockModel(nn.Module):
    """Mock model with noise imprint components."""
    
    def __init__(self, feature_dim=256, enable_attribution=False):
        super().__init__()
        self.noise_extractor = MockNoiseExtractor()
        self.noise_branch = MockNoiseBranch(feature_dim, enable_attribution)


def create_test_dataloader(num_samples=100, num_generators=3, batch_size=16):
    """Create a test dataloader with synthetic data."""
    # Create images
    images = torch.randn(num_samples, 3, 64, 64)
    
    # Create generator labels (evenly distributed)
    labels = torch.tensor([i % num_generators for i in range(num_samples)])
    
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader


class TestEvaluateNoiseImprintClustering:
    """Tests for evaluate_noise_imprint_clustering function."""
    
    def test_basic_evaluation(self):
        """Test basic clustering evaluation with valid inputs."""
        model = MockModel()
        loader = create_test_dataloader(num_samples=90, num_generators=3)
        device = torch.device('cpu')
        
        metrics = evaluate_noise_imprint_clustering(model, loader, device)
        
        # Check all expected keys are present
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_index' in metrics
        assert 'num_samples' in metrics
        assert 'num_generators' in metrics
        
        # Check values are in expected ranges
        assert -1 <= metrics['silhouette_score'] <= 1
        assert metrics['davies_bouldin_index'] >= 0
        assert metrics['num_samples'] == 90
        assert metrics['num_generators'] == 3
    
    def test_with_generator_labels(self):
        """Test evaluation with generator label names."""
        model = MockModel()
        loader = create_test_dataloader(num_samples=60, num_generators=3)
        device = torch.device('cpu')
        generator_labels = ['DALL-E', 'Midjourney', 'Stable Diffusion']
        
        metrics = evaluate_noise_imprint_clustering(
            model, loader, device, generator_labels
        )
        
        assert 'generator_labels' in metrics
        assert metrics['generator_labels'] == generator_labels
    
    def test_with_attribution_model(self):
        """Test evaluation with model that has attribution head."""
        model = MockModel(enable_attribution=True)
        loader = create_test_dataloader(num_samples=60, num_generators=3)
        device = torch.device('cpu')
        
        # Should handle tuple output from noise_branch
        metrics = evaluate_noise_imprint_clustering(model, loader, device)
        
        assert 'silhouette_score' in metrics
        assert metrics['num_samples'] == 60
    
    def test_missing_noise_extractor(self):
        """Test error when model lacks noise_extractor."""
        model = nn.Module()  # Empty model
        loader = create_test_dataloader()
        device = torch.device('cpu')
        
        with pytest.raises(ValueError, match="noise_extractor"):
            evaluate_noise_imprint_clustering(model, loader, device)
    
    def test_missing_noise_branch(self):
        """Test error when model lacks noise_branch."""
        model = nn.Module()
        model.noise_extractor = MockNoiseExtractor()
        loader = create_test_dataloader()
        device = torch.device('cpu')
        
        with pytest.raises(ValueError, match="noise_branch"):
            evaluate_noise_imprint_clustering(model, loader, device)
    
    def test_single_generator_error(self):
        """Test error when only one generator is present."""
        model = MockModel()
        loader = create_test_dataloader(num_samples=30, num_generators=1)
        device = torch.device('cpu')
        
        with pytest.raises(ValueError, match="at least 2 generators"):
            evaluate_noise_imprint_clustering(model, loader, device)
    
    def test_insufficient_samples_per_generator(self):
        """Test error when a generator has only 1 sample."""
        model = MockModel()
        
        # Create dataset with one generator having only 1 sample
        images = torch.randn(5, 3, 64, 64)
        labels = torch.tensor([0, 0, 1, 1, 2])  # Generator 2 has only 1 sample
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=5)
        device = torch.device('cpu')
        
        with pytest.raises(ValueError, match="at least 2 samples"):
            evaluate_noise_imprint_clustering(model, loader, device)
    
    def test_multiple_generators(self):
        """Test with more than 3 generators."""
        model = MockModel()
        loader = create_test_dataloader(num_samples=100, num_generators=5)
        device = torch.device('cpu')
        
        metrics = evaluate_noise_imprint_clustering(model, loader, device)
        
        assert metrics['num_generators'] == 5
        assert metrics['num_samples'] == 100


class TestPrintClusteringReport:
    """Tests for print_clustering_report function."""
    
    def test_basic_report(self, capsys):
        """Test basic report printing."""
        metrics = {
            'silhouette_score': 0.65,
            'davies_bouldin_index': 0.85,
            'num_samples': 100,
            'num_generators': 3
        }
        
        print_clustering_report(metrics, verbose=False)
        
        captured = capsys.readouterr()
        assert "NOISE IMPRINT CLUSTERING REPORT" in captured.out
        assert "0.6500" in captured.out
        assert "0.8500" in captured.out
        assert "100" in captured.out
        assert "3" in captured.out
    
    def test_report_with_generator_labels(self, capsys):
        """Test report with generator label names."""
        metrics = {
            'silhouette_score': 0.65,
            'davies_bouldin_index': 0.85,
            'num_samples': 100,
            'num_generators': 3,
            'generator_labels': ['DALL-E', 'Midjourney', 'Stable Diffusion']
        }
        
        print_clustering_report(metrics, verbose=False)
        
        captured = capsys.readouterr()
        assert "DALL-E" in captured.out
        assert "Midjourney" in captured.out
        assert "Stable Diffusion" in captured.out
    
    def test_verbose_report(self, capsys):
        """Test verbose report with interpretation."""
        metrics = {
            'silhouette_score': 0.65,
            'davies_bouldin_index': 0.85,
            'num_samples': 100,
            'num_generators': 3
        }
        
        print_clustering_report(metrics, verbose=True)
        
        captured = capsys.readouterr()
        assert "Interpretation:" in captured.out
        assert "feature space" in captured.out
    
    def test_interpretation_excellent(self, capsys):
        """Test interpretation for excellent separation."""
        metrics = {
            'silhouette_score': 0.85,
            'davies_bouldin_index': 0.35,
            'num_samples': 100,
            'num_generators': 3
        }
        
        print_clustering_report(metrics, verbose=False)
        
        captured = capsys.readouterr()
        assert "Excellent separation" in captured.out
    
    def test_interpretation_poor(self, capsys):
        """Test interpretation for poor separation."""
        metrics = {
            'silhouette_score': 0.15,
            'davies_bouldin_index': 2.5,
            'num_samples': 100,
            'num_generators': 3
        }
        
        print_clustering_report(metrics, verbose=False)
        
        captured = capsys.readouterr()
        assert "Weak separation" in captured.out or "Poor separation" in captured.out


class TestExtractNoiseFeatures:
    """Tests for extract_noise_features function."""
    
    def test_basic_extraction(self):
        """Test basic feature extraction."""
        model = MockModel(feature_dim=128)
        images = torch.randn(10, 3, 64, 64)
        device = torch.device('cpu')
        
        features = extract_noise_features(model, images, device)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (10, 128)
    
    def test_with_attribution_model(self):
        """Test feature extraction with attribution model."""
        model = MockModel(feature_dim=256, enable_attribution=True)
        images = torch.randn(5, 3, 64, 64)
        device = torch.device('cpu')
        
        features = extract_noise_features(model, images, device)
        
        # Should extract features and ignore attribution
        assert features.shape == (5, 256)
    
    def test_missing_components(self):
        """Test error when model lacks required components."""
        model = nn.Module()
        images = torch.randn(5, 3, 64, 64)
        device = torch.device('cpu')
        
        with pytest.raises(ValueError, match="noise_extractor"):
            extract_noise_features(model, images, device)


class TestComputePairwiseSeparability:
    """Tests for compute_pairwise_separability function."""
    
    def test_basic_pairwise(self):
        """Test basic pairwise separability computation."""
        # Create features with clear separation
        features = np.vstack([
            np.random.randn(30, 10) + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.random.randn(30, 10) + [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            np.random.randn(30, 10) + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        ])
        labels = np.repeat([0, 1, 2], 30)
        
        pairwise = compute_pairwise_separability(features, labels)
        
        # Should have 3 pairs: (0,1), (0,2), (1,2)
        assert len(pairwise) == 3
        assert (0, 1) in pairwise
        assert (0, 2) in pairwise
        assert (1, 2) in pairwise
        
        # All scores should be in valid range
        for score in pairwise.values():
            assert -1 <= score <= 1
    
    def test_two_generators(self):
        """Test with only two generators."""
        features = np.vstack([
            np.random.randn(20, 5) + [1, 0, 0, 0, 0],
            np.random.randn(20, 5) + [0, 1, 0, 0, 0]
        ])
        labels = np.repeat([0, 1], 20)
        
        pairwise = compute_pairwise_separability(features, labels)
        
        # Should have only 1 pair
        assert len(pairwise) == 1
        assert (0, 1) in pairwise
    
    def test_with_generator_labels(self):
        """Test with generator label names."""
        features = np.random.randn(60, 10)
        labels = np.repeat([0, 1, 2], 20)
        generator_labels = ['DALL-E', 'Midjourney', 'Stable Diffusion']
        
        pairwise = compute_pairwise_separability(features, labels, generator_labels)
        
        # Should still use numeric indices as keys
        assert (0, 1) in pairwise
        assert (0, 2) in pairwise
        assert (1, 2) in pairwise


class TestIntegration:
    """Integration tests for the full workflow."""
    
    def test_full_workflow(self):
        """Test complete workflow from model to report."""
        # Create model and data
        model = MockModel(feature_dim=128)
        loader = create_test_dataloader(num_samples=90, num_generators=3)
        device = torch.device('cpu')
        generator_labels = ['Generator A', 'Generator B', 'Generator C']
        
        # Evaluate clustering
        metrics = evaluate_noise_imprint_clustering(
            model, loader, device, generator_labels
        )
        
        # Print report (should not raise errors)
        print_clustering_report(metrics, verbose=True)
        
        # Extract features for further analysis
        images = torch.randn(10, 3, 64, 64)
        features = extract_noise_features(model, images, device)
        
        assert features.shape[0] == 10
        assert metrics['num_generators'] == 3
    
    def test_cuda_if_available(self):
        """Test with CUDA device if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = MockModel().cuda()
        loader = create_test_dataloader(num_samples=60, num_generators=3)
        device = torch.device('cuda')
        
        metrics = evaluate_noise_imprint_clustering(model, loader, device)
        
        assert metrics['num_samples'] == 60
        assert 'silhouette_score' in metrics
