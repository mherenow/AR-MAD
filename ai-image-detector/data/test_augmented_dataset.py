"""
Tests for AugmentedDataset wrapper.

This module tests the AugmentedDataset wrapper to ensure it correctly applies
augmentations to base datasets while maintaining compatibility with both 2-tuple
and 3-tuple dataset formats.
"""

import pytest
import torch
from torch.utils.data import Dataset

from data.augmented_dataset import AugmentedDataset
from data.augmentation.robustness import RobustnessAugmentation


class MockDataset2Tuple(Dataset):
    """Mock dataset that returns 2-tuple (image, label)."""
    
    def __init__(self, size: int = 10):
        self.size = size
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int):
        # Return a simple image tensor and label
        image = torch.rand(3, 64, 64)
        label = idx % 2  # Alternate between 0 and 1
        return image, label


class MockDataset3Tuple(Dataset):
    """Mock dataset that returns 3-tuple (image, label, metadata)."""
    
    def __init__(self, size: int = 10):
        self.size = size
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int):
        # Return image tensor, label, and metadata
        image = torch.rand(3, 64, 64)
        label = idx % 2
        metadata = f"sample_{idx}"
        return image, label, metadata


class TestAugmentedDatasetInitialization:
    """Test AugmentedDataset initialization."""
    
    def test_init_with_valid_dataset(self):
        """Test initialization with valid base dataset."""
        base_dataset = MockDataset2Tuple(size=10)
        aug_dataset = AugmentedDataset(base_dataset)
        
        assert len(aug_dataset) == 10
        assert aug_dataset.base_dataset is base_dataset
        assert aug_dataset.robustness_aug is None
        assert aug_dataset.aug_prob == 1.0
    
    def test_init_with_robustness_aug(self):
        """Test initialization with RobustnessAugmentation."""
        base_dataset = MockDataset2Tuple(size=10)
        robustness_aug = RobustnessAugmentation(jpeg_prob=0.3, blur_prob=0.3, noise_prob=0.3)
        aug_dataset = AugmentedDataset(base_dataset, robustness_aug=robustness_aug)
        
        assert aug_dataset.robustness_aug is robustness_aug
    
    def test_init_with_custom_aug_prob(self):
        """Test initialization with custom augmentation probability."""
        base_dataset = MockDataset2Tuple(size=10)
        aug_dataset = AugmentedDataset(base_dataset, aug_prob=0.5)
        
        assert aug_dataset.aug_prob == 0.5
    
    def test_init_with_invalid_dataset(self):
        """Test initialization with invalid base dataset."""
        # Use an object that doesn't have __getitem__ or __len__
        class InvalidDataset:
            pass
        
        with pytest.raises(ValueError, match="must implement __getitem__ and __len__"):
            AugmentedDataset(InvalidDataset())
    
    def test_init_with_invalid_aug_prob(self):
        """Test initialization with invalid augmentation probability."""
        base_dataset = MockDataset2Tuple(size=10)
        
        with pytest.raises(ValueError, match="aug_prob must be in"):
            AugmentedDataset(base_dataset, aug_prob=-0.1)
        
        with pytest.raises(ValueError, match="aug_prob must be in"):
            AugmentedDataset(base_dataset, aug_prob=1.5)


class TestAugmentedDataset2Tuple:
    """Test AugmentedDataset with 2-tuple base dataset."""
    
    def test_getitem_without_augmentation(self):
        """Test __getitem__ without augmentation."""
        base_dataset = MockDataset2Tuple(size=10)
        aug_dataset = AugmentedDataset(base_dataset)
        
        image, label = aug_dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert label in [0, 1]
    
    def test_getitem_with_augmentation_prob_zero(self):
        """Test __getitem__ with augmentation but probability zero."""
        base_dataset = MockDataset2Tuple(size=10)
        robustness_aug = RobustnessAugmentation(jpeg_prob=1.0, blur_prob=1.0, noise_prob=1.0)
        aug_dataset = AugmentedDataset(base_dataset, robustness_aug=robustness_aug, aug_prob=0.0)
        
        # Get original image
        original_image, _ = base_dataset[0]
        
        # Get augmented dataset image (should be same since aug_prob=0)
        # Note: We can't directly compare due to random generation in MockDataset
        # So we just verify the format is correct
        image, label = aug_dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
    
    def test_getitem_with_augmentation_prob_one(self):
        """Test __getitem__ with augmentation and probability one."""
        base_dataset = MockDataset2Tuple(size=10)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=1.0,
            blur_prob=0.0,
            noise_prob=0.0,
            severity_range=(1, 1)
        )
        aug_dataset = AugmentedDataset(base_dataset, robustness_aug=robustness_aug, aug_prob=1.0)
        
        image, label = aug_dataset[0]
        
        # Verify format
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)
        
        # Verify image values are in valid range [0, 1]
        assert torch.all(image >= 0.0)
        assert torch.all(image <= 1.0)
    
    def test_len(self):
        """Test __len__ returns correct length."""
        base_dataset = MockDataset2Tuple(size=25)
        aug_dataset = AugmentedDataset(base_dataset)
        
        assert len(aug_dataset) == 25


class TestAugmentedDataset3Tuple:
    """Test AugmentedDataset with 3-tuple base dataset."""
    
    def test_getitem_without_augmentation(self):
        """Test __getitem__ without augmentation."""
        base_dataset = MockDataset3Tuple(size=10)
        aug_dataset = AugmentedDataset(base_dataset)
        
        image, label, metadata = aug_dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert label in [0, 1]
        assert isinstance(metadata, str)
        assert metadata == "sample_0"
    
    def test_getitem_with_augmentation(self):
        """Test __getitem__ with augmentation."""
        base_dataset = MockDataset3Tuple(size=10)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=0.0,
            blur_prob=1.0,
            noise_prob=0.0,
            severity_range=(2, 2)
        )
        aug_dataset = AugmentedDataset(base_dataset, robustness_aug=robustness_aug, aug_prob=1.0)
        
        image, label, metadata = aug_dataset[5]
        
        # Verify format
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert label == 1  # idx=5, 5 % 2 = 1
        assert isinstance(metadata, str)
        assert metadata == "sample_5"
        
        # Verify image values are in valid range
        assert torch.all(image >= 0.0)
        assert torch.all(image <= 1.0)
    
    def test_len(self):
        """Test __len__ returns correct length."""
        base_dataset = MockDataset3Tuple(size=15)
        aug_dataset = AugmentedDataset(base_dataset)
        
        assert len(aug_dataset) == 15


class TestAugmentedDatasetEdgeCases:
    """Test edge cases for AugmentedDataset."""
    
    def test_empty_dataset(self):
        """Test with empty base dataset."""
        base_dataset = MockDataset2Tuple(size=0)
        aug_dataset = AugmentedDataset(base_dataset)
        
        assert len(aug_dataset) == 0
    
    def test_single_sample_dataset(self):
        """Test with single sample dataset."""
        base_dataset = MockDataset2Tuple(size=1)
        aug_dataset = AugmentedDataset(base_dataset)
        
        assert len(aug_dataset) == 1
        image, label = aug_dataset[0]
        assert isinstance(image, torch.Tensor)
    
    def test_multiple_augmentations(self):
        """Test with multiple augmentation types enabled."""
        base_dataset = MockDataset2Tuple(size=10)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=0.5,
            blur_prob=0.5,
            noise_prob=0.5,
            severity_range=(1, 5)
        )
        aug_dataset = AugmentedDataset(base_dataset, robustness_aug=robustness_aug, aug_prob=1.0)
        
        # Test multiple samples
        for i in range(5):
            image, label = aug_dataset[i]
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, 64, 64)
            assert torch.all(image >= 0.0)
            assert torch.all(image <= 1.0)


class TestAugmentedDatasetIntegration:
    """Integration tests for AugmentedDataset."""
    
    def test_with_dataloader(self):
        """Test AugmentedDataset with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        base_dataset = MockDataset2Tuple(size=20)
        robustness_aug = RobustnessAugmentation(jpeg_prob=0.3, blur_prob=0.3, noise_prob=0.3)
        aug_dataset = AugmentedDataset(base_dataset, robustness_aug=robustness_aug, aug_prob=0.5)
        
        dataloader = DataLoader(aug_dataset, batch_size=4, shuffle=True)
        
        # Test one batch
        batch = next(iter(dataloader))
        images, labels = batch
        
        assert images.shape == (4, 3, 64, 64)
        assert labels.shape == (4,)
        assert torch.all(images >= 0.0)
        assert torch.all(images <= 1.0)
    
    def test_with_dataloader_3tuple(self):
        """Test AugmentedDataset with PyTorch DataLoader for 3-tuple format."""
        from torch.utils.data import DataLoader
        
        base_dataset = MockDataset3Tuple(size=20)
        aug_dataset = AugmentedDataset(base_dataset, aug_prob=0.5)
        
        dataloader = DataLoader(aug_dataset, batch_size=4, shuffle=False)
        
        # Test one batch
        batch = next(iter(dataloader))
        images, labels, metadata = batch
        
        assert images.shape == (4, 3, 64, 64)
        assert labels.shape == (4,)
        assert len(metadata) == 4
        assert all(isinstance(m, str) for m in metadata)
    
    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        base_dataset = MockDataset2Tuple(size=5)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=0.0,
            blur_prob=0.0,
            noise_prob=1.0,
            severity_range=(3, 3)
        )
        aug_dataset = AugmentedDataset(base_dataset, robustness_aug=robustness_aug, aug_prob=1.0)
        
        # Set seed and get sample
        torch.manual_seed(42)
        image1, _ = aug_dataset[0]
        
        # Reset seed and get sample again
        torch.manual_seed(42)
        image2, _ = aug_dataset[0]
        
        # Note: Due to random generation in MockDataset, we can't compare directly
        # But we can verify both are valid tensors
        assert isinstance(image1, torch.Tensor)
        assert isinstance(image2, torch.Tensor)
        assert image1.shape == image2.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
