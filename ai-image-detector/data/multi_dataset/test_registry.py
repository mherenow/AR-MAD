"""Unit tests for DatasetRegistry."""

import pytest
import torch
from torch.utils.data import Dataset
from .registry import DatasetRegistry


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, root_dir: str, size: int = 100, transform=None):
        """Initialize mock dataset."""
        self.root_dir = root_dir
        self.size = size
        self.transform = transform
    
    def __len__(self):
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx):
        """Return a mock sample."""
        if idx >= self.size:
            raise IndexError("Index out of range")
        
        # Return mock image and label
        image = torch.randn(3, 224, 224)
        label = idx % 2  # Alternate between 0 and 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class InvalidDataset:
    """Invalid dataset that doesn't inherit from Dataset."""
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir


class TestDatasetRegistry:
    """Test suite for DatasetRegistry."""
    
    def test_initialization(self):
        """Test registry initialization."""
        registry = DatasetRegistry()
        assert len(registry) == 0
        assert registry.list() == []
    
    def test_register_dataset(self):
        """Test registering a dataset."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 50}
        
        registry.register('test_dataset', MockDataset, config)
        
        assert len(registry) == 1
        assert 'test_dataset' in registry
        assert registry.is_registered('test_dataset')
        assert registry.list() == ['test_dataset']
    
    def test_register_multiple_datasets(self):
        """Test registering multiple datasets."""
        registry = DatasetRegistry()
        
        config1 = {'root_dir': 'data/dataset1', 'size': 100}
        config2 = {'root_dir': 'data/dataset2', 'size': 200}
        
        registry.register('dataset1', MockDataset, config1)
        registry.register('dataset2', MockDataset, config2)
        
        assert len(registry) == 2
        assert set(registry.list()) == {'dataset1', 'dataset2'}
    
    def test_register_duplicate_name_raises_error(self):
        """Test that registering duplicate name raises ValueError."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 50}
        
        registry.register('test_dataset', MockDataset, config)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register('test_dataset', MockDataset, config)
    
    def test_register_invalid_dataset_class_raises_error(self):
        """Test that registering non-Dataset class raises TypeError."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test'}
        
        with pytest.raises(TypeError, match="must be a subclass"):
            registry.register('invalid', InvalidDataset, config)
    
    def test_register_with_invalid_config_raises_error(self):
        """Test that invalid config raises ValueError."""
        registry = DatasetRegistry()
        config = {'invalid_param': 'value'}  # Missing required 'root_dir'
        
        with pytest.raises(ValueError, match="Failed to instantiate"):
            registry.register('test_dataset', MockDataset, config)
    
    def test_get_dataset(self):
        """Test retrieving a registered dataset."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 50}
        
        registry.register('test_dataset', MockDataset, config)
        dataset = registry.get('test_dataset')
        
        assert isinstance(dataset, MockDataset)
        assert dataset.root_dir == 'data/test'
        assert len(dataset) == 50
    
    def test_get_nonexistent_dataset_raises_error(self):
        """Test that getting nonexistent dataset raises KeyError."""
        registry = DatasetRegistry()
        
        with pytest.raises(KeyError, match="not found"):
            registry.get('nonexistent')
    
    def test_get_config(self):
        """Test retrieving dataset configuration."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 50}
        
        registry.register('test_dataset', MockDataset, config)
        retrieved_config = registry.get_config('test_dataset')
        
        assert retrieved_config == config
        assert retrieved_config is not config  # Should be a copy
    
    def test_get_config_nonexistent_raises_error(self):
        """Test that getting config for nonexistent dataset raises KeyError."""
        registry = DatasetRegistry()
        
        with pytest.raises(KeyError, match="not found"):
            registry.get_config('nonexistent')
    
    def test_unregister_dataset(self):
        """Test unregistering a dataset."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 50}
        
        registry.register('test_dataset', MockDataset, config)
        assert 'test_dataset' in registry
        
        registry.unregister('test_dataset')
        
        assert 'test_dataset' not in registry
        assert len(registry) == 0
    
    def test_unregister_nonexistent_raises_error(self):
        """Test that unregistering nonexistent dataset raises KeyError."""
        registry = DatasetRegistry()
        
        with pytest.raises(KeyError, match="not found"):
            registry.unregister('nonexistent')
    
    def test_is_registered(self):
        """Test checking if dataset is registered."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 50}
        
        assert not registry.is_registered('test_dataset')
        
        registry.register('test_dataset', MockDataset, config)
        assert registry.is_registered('test_dataset')
        
        registry.unregister('test_dataset')
        assert not registry.is_registered('test_dataset')
    
    def test_clear(self):
        """Test clearing all datasets."""
        registry = DatasetRegistry()
        
        config1 = {'root_dir': 'data/dataset1', 'size': 100}
        config2 = {'root_dir': 'data/dataset2', 'size': 200}
        
        registry.register('dataset1', MockDataset, config1)
        registry.register('dataset2', MockDataset, config2)
        
        assert len(registry) == 2
        
        registry.clear()
        
        assert len(registry) == 0
        assert registry.list() == []
    
    def test_contains_operator(self):
        """Test 'in' operator."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 50}
        
        assert 'test_dataset' not in registry
        
        registry.register('test_dataset', MockDataset, config)
        assert 'test_dataset' in registry
    
    def test_len_operator(self):
        """Test len() operator."""
        registry = DatasetRegistry()
        
        assert len(registry) == 0
        
        config1 = {'root_dir': 'data/dataset1', 'size': 100}
        registry.register('dataset1', MockDataset, config1)
        assert len(registry) == 1
        
        config2 = {'root_dir': 'data/dataset2', 'size': 200}
        registry.register('dataset2', MockDataset, config2)
        assert len(registry) == 2
    
    def test_repr(self):
        """Test string representation."""
        registry = DatasetRegistry()
        
        assert repr(registry) == "DatasetRegistry(datasets=[])"
        
        config = {'root_dir': 'data/test', 'size': 50}
        registry.register('test_dataset', MockDataset, config)
        
        assert repr(registry) == "DatasetRegistry(datasets=['test_dataset'])"
    
    def test_dataset_functionality(self):
        """Test that registered dataset works correctly."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 10}
        
        registry.register('test_dataset', MockDataset, config)
        dataset = registry.get('test_dataset')
        
        # Test dataset functionality
        assert len(dataset) == 10
        
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)
        assert label in [0, 1]
        
        # Test all samples
        for i in range(len(dataset)):
            image, label = dataset[i]
            assert image.shape == (3, 224, 224)
            assert label == i % 2
    
    def test_dataset_with_transform(self):
        """Test dataset with transform."""
        registry = DatasetRegistry()
        
        # Define a simple transform
        def normalize_transform(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        
        config = {'root_dir': 'data/test', 'size': 5, 'transform': normalize_transform}
        
        registry.register('test_dataset', MockDataset, config)
        dataset = registry.get('test_dataset')
        
        image, label = dataset[0]
        
        # Check that transform was applied (mean should be close to 0)
        assert abs(image.mean().item()) < 0.1
    
    def test_config_isolation(self):
        """Test that config modifications don't affect registry."""
        registry = DatasetRegistry()
        config = {'root_dir': 'data/test', 'size': 50}
        
        registry.register('test_dataset', MockDataset, config)
        
        # Modify original config
        config['size'] = 100
        
        # Retrieved config should be unchanged
        retrieved_config = registry.get_config('test_dataset')
        assert retrieved_config['size'] == 50
    
    def test_multiple_registries_independent(self):
        """Test that multiple registries are independent."""
        registry1 = DatasetRegistry()
        registry2 = DatasetRegistry()
        
        config = {'root_dir': 'data/test', 'size': 50}
        
        registry1.register('dataset1', MockDataset, config)
        
        assert 'dataset1' in registry1
        assert 'dataset1' not in registry2
        assert len(registry1) == 1
        assert len(registry2) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
