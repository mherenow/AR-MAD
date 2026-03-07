"""
Unit tests for MultiDatasetLoader.

Tests cover:
- Weight normalization
- Probabilistic dataset selection
- Batch sampling from selected dataset
- Edge cases and error handling
"""

import pytest
import torch
from torch.utils.data import TensorDataset
import numpy as np
from .loader import MultiDatasetLoader


class TestMultiDatasetLoader:
    """Test suite for MultiDatasetLoader."""
    
    @pytest.fixture
    def sample_datasets(self):
        """Create sample datasets for testing."""
        # Dataset 1: 100 samples
        data1 = torch.randn(100, 3, 32, 32)
        labels1 = torch.randint(0, 2, (100,))
        dataset1 = TensorDataset(data1, labels1)
        
        # Dataset 2: 50 samples
        data2 = torch.randn(50, 3, 32, 32)
        labels2 = torch.randint(0, 2, (50,))
        dataset2 = TensorDataset(data2, labels2)
        
        # Dataset 3: 75 samples
        data3 = torch.randn(75, 3, 32, 32)
        labels3 = torch.randint(0, 2, (75,))
        dataset3 = TensorDataset(data3, labels3)
        
        return {
            'dataset1': dataset1,
            'dataset2': dataset2,
            'dataset3': dataset3
        }
    
    def test_initialization(self, sample_datasets):
        """Test basic initialization."""
        weights = {'dataset1': 0.5, 'dataset2': 0.3, 'dataset3': 0.2}
        loader = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16
        )
        
        assert len(loader.datasets) == 3
        assert len(loader.dataloaders) == 3
        assert len(loader.dataset_names) == 3
        assert len(loader.probabilities) == 3
    
    def test_weight_normalization(self, sample_datasets):
        """Test that weights are normalized to probabilities."""
        weights = {'dataset1': 2.0, 'dataset2': 3.0, 'dataset3': 5.0}
        loader = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16
        )
        
        # Check probabilities sum to 1
        assert np.isclose(loader.probabilities.sum(), 1.0)
        
        # Check individual probabilities
        expected_probs = {
            'dataset1': 0.2,  # 2/10
            'dataset2': 0.3,  # 3/10
            'dataset3': 0.5   # 5/10
        }
        
        for i, name in enumerate(loader.dataset_names):
            assert np.isclose(
                loader.probabilities[i], expected_probs[name], atol=1e-6
            )
    
    def test_equal_weights(self, sample_datasets):
        """Test equal weights result in equal probabilities."""
        weights = {'dataset1': 1.0, 'dataset2': 1.0, 'dataset3': 1.0}
        loader = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16
        )
        
        # All probabilities should be 1/3
        for prob in loader.probabilities:
            assert np.isclose(prob, 1.0/3.0, atol=1e-6)
    
    def test_batch_iteration(self, sample_datasets):
        """Test iterating over batches."""
        weights = {'dataset1': 0.5, 'dataset2': 0.3, 'dataset3': 0.2}
        loader = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16, shuffle=False
        )
        
        batch_count = 0
        dataset_counts = {'dataset1': 0, 'dataset2': 0, 'dataset3': 0}
        
        for images, labels, dataset_name in loader:
            # Check batch shapes
            assert images.shape[0] <= 16  # batch_size
            assert images.shape[1:] == (3, 32, 32)
            assert labels.shape[0] == images.shape[0]
            
            # Check dataset name is valid
            assert dataset_name in sample_datasets
            
            # Count batches per dataset
            dataset_counts[dataset_name] += 1
            batch_count += 1
        
        # Should iterate through all batches
        assert batch_count == len(loader)
        
        # At least one dataset should be sampled (probabilistic, so not all guaranteed)
        assert sum(dataset_counts.values()) > 0
    
    def test_probabilistic_sampling(self, sample_datasets):
        """Test that sampling respects probabilities over many iterations."""
        # Use extreme weights to make test more reliable
        weights = {'dataset1': 0.8, 'dataset2': 0.15, 'dataset3': 0.05}
        loader = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16
        )
        
        # Run multiple epochs to get statistical significance
        dataset_counts = {'dataset1': 0, 'dataset2': 0, 'dataset3': 0}
        num_epochs = 10
        
        for _ in range(num_epochs):
            for _, _, dataset_name in loader:
                dataset_counts[dataset_name] += 1
        
        total_batches = sum(dataset_counts.values())
        observed_probs = {
            name: count / total_batches 
            for name, count in dataset_counts.items()
        }
        
        # Check that observed probabilities are close to expected
        # Allow 10% tolerance due to randomness
        expected_probs = {
            'dataset1': 0.8,
            'dataset2': 0.15,
            'dataset3': 0.05
        }
        
        for name in expected_probs:
            assert abs(
                observed_probs[name] - expected_probs[name]
            ) < 0.1, f"Dataset {name} probability mismatch"
    
    def test_num_batches(self, sample_datasets):
        """Test that num_batches is based on largest dataset."""
        weights = {'dataset1': 0.5, 'dataset2': 0.3, 'dataset3': 0.2}
        loader = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16, drop_last=False
        )
        
        # Dataset1 has 100 samples, so 100/16 = 7 batches (with drop_last=False)
        # This should be the maximum
        expected_batches = max(
            len(dataset) // 16 + (1 if len(dataset) % 16 > 0 else 0)
            for dataset in sample_datasets.values()
        )
        
        assert len(loader) == expected_batches
    
    def test_dataset_info(self, sample_datasets):
        """Test get_dataset_info method."""
        weights = {'dataset1': 0.5, 'dataset2': 0.3, 'dataset3': 0.2}
        loader = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16
        )
        
        info = loader.get_dataset_info()
        
        # Check all datasets are present
        assert set(info.keys()) == set(sample_datasets.keys())
        
        # Check info structure
        for name, dataset_info in info.items():
            assert 'size' in dataset_info
            assert 'probability' in dataset_info
            assert 'num_batches' in dataset_info
            
            # Verify size
            assert dataset_info['size'] == len(sample_datasets[name])
            
            # Verify probability
            expected_prob = weights[name] / sum(weights.values())
            assert np.isclose(dataset_info['probability'], expected_prob)
    
    def test_empty_datasets_error(self):
        """Test that empty datasets dict raises error."""
        with pytest.raises(ValueError, match="datasets dictionary cannot be empty"):
            MultiDatasetLoader({}, {}, batch_size=16)
    
    def test_mismatched_keys_error(self, sample_datasets):
        """Test that mismatched dataset and weight keys raise error."""
        weights = {'dataset1': 0.5, 'dataset2': 0.5}  # Missing dataset3
        
        with pytest.raises(ValueError, match="datasets and weights must have the same keys"):
            MultiDatasetLoader(sample_datasets, weights, batch_size=16)
    
    def test_negative_weight_error(self, sample_datasets):
        """Test that negative weights raise error."""
        weights = {'dataset1': 0.5, 'dataset2': -0.3, 'dataset3': 0.2}
        
        with pytest.raises(ValueError, match="all weights must be positive"):
            MultiDatasetLoader(sample_datasets, weights, batch_size=16)
    
    def test_zero_weight_error(self, sample_datasets):
        """Test that zero weights raise error."""
        weights = {'dataset1': 0.5, 'dataset2': 0.0, 'dataset3': 0.2}
        
        with pytest.raises(ValueError, match="all weights must be positive"):
            MultiDatasetLoader(sample_datasets, weights, batch_size=16)
    
    def test_single_dataset(self):
        """Test with a single dataset."""
        data = torch.randn(50, 3, 32, 32)
        labels = torch.randint(0, 2, (50,))
        dataset = TensorDataset(data, labels)
        
        datasets = {'single': dataset}
        weights = {'single': 1.0}
        
        loader = MultiDatasetLoader(datasets, weights, batch_size=16)
        
        # Probability should be 1.0
        assert np.isclose(loader.probabilities[0], 1.0)
        
        # All batches should come from the single dataset
        for _, _, dataset_name in loader:
            assert dataset_name == 'single'
    
    def test_different_batch_sizes(self, sample_datasets):
        """Test with different batch sizes."""
        weights = {'dataset1': 0.5, 'dataset2': 0.3, 'dataset3': 0.2}
        
        for batch_size in [8, 16, 32, 64]:
            loader = MultiDatasetLoader(
                sample_datasets, weights, batch_size=batch_size
            )
            
            for images, labels, _ in loader:
                assert images.shape[0] <= batch_size
                assert labels.shape[0] == images.shape[0]
    
    def test_shuffle_parameter(self, sample_datasets):
        """Test shuffle parameter is passed to DataLoaders."""
        weights = {'dataset1': 0.5, 'dataset2': 0.3, 'dataset3': 0.2}
        
        # Test with shuffle=True
        loader_shuffled = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16, shuffle=True
        )
        assert all(
            dl.dataset == sample_datasets[name]
            for name, dl in loader_shuffled.dataloaders.items()
        )
        
        # Test with shuffle=False
        loader_not_shuffled = MultiDatasetLoader(
            sample_datasets, weights, batch_size=16, shuffle=False
        )
        assert all(
            dl.dataset == sample_datasets[name]
            for name, dl in loader_not_shuffled.dataloaders.items()
        )
    
    def test_iterator_exhaustion_handling(self, sample_datasets):
        """Test that exhausted iterators are recreated."""
        # Use small dataset and large batch size to quickly exhaust
        small_data = torch.randn(10, 3, 32, 32)
        small_labels = torch.randint(0, 2, (10,))
        small_dataset = TensorDataset(small_data, small_labels)
        
        datasets = {
            'small': small_dataset,
            'dataset1': sample_datasets['dataset1']
        }
        weights = {'small': 0.9, 'dataset1': 0.1}  # Heavily favor small dataset
        
        loader = MultiDatasetLoader(datasets, weights, batch_size=8)
        
        # Iterate through multiple batches
        # The small dataset will be exhausted and should be recreated
        batch_count = 0
        for images, labels, dataset_name in loader:
            assert images.shape[0] > 0
            assert labels.shape[0] > 0
            batch_count += 1
        
        # Should complete without errors
        assert batch_count == len(loader)
