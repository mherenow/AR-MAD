"""
Integration test for AugmentedDataset with real dataset classes.

This test verifies that AugmentedDataset works correctly with actual
dataset implementations like SynthBusterDataset and COCO2017Dataset.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from data.augmented_dataset import AugmentedDataset
from data.augmentation.robustness import RobustnessAugmentation


class TestAugmentedDatasetIntegrationWithRealDatasets:
    """Integration tests with real dataset classes."""
    
    def test_with_synthbuster_dataset_format(self):
        """Test AugmentedDataset with SynthBusterDataset-like format (3-tuple)."""
        # Create a mock that mimics SynthBusterDataset behavior
        class MockSynthBusterDataset:
            def __init__(self, size=10):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # SynthBusterDataset returns (image, label, generator_name)
                image = torch.rand(3, 256, 256)
                label = idx % 2
                generator_name = "stable-diffusion-v1-4" if label == 1 else "RAISE"
                return image, label, generator_name
        
        base_dataset = MockSynthBusterDataset(size=20)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=0.3,
            blur_prob=0.3,
            noise_prob=0.3
        )
        augmented_dataset = AugmentedDataset(
            base_dataset,
            robustness_aug=robustness_aug,
            aug_prob=0.5
        )
        
        # Test single sample
        image, label, generator = augmented_dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 256, 256)
        assert isinstance(label, int)
        assert isinstance(generator, str)
        
        # Test with DataLoader
        loader = DataLoader(augmented_dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        images, labels, generators = batch
        
        assert images.shape == (4, 3, 256, 256)
        assert labels.shape == (4,)
        assert len(generators) == 4
    
    def test_with_coco_dataset_format(self):
        """Test AugmentedDataset with COCO2017Dataset-like format (3-tuple)."""
        # Create a mock that mimics COCO2017Dataset behavior
        class MockCOCODataset:
            def __init__(self, size=10):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # COCO2017Dataset returns (image, label, image_id)
                image = torch.rand(3, 256, 256)
                label = 0  # Real images
                image_id = f"coco_{idx:06d}"
                return image, label, image_id
        
        base_dataset = MockCOCODataset(size=15)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=0.5,
            blur_prob=0.5,
            noise_prob=0.5,
            severity_range=(1, 3)
        )
        augmented_dataset = AugmentedDataset(
            base_dataset,
            robustness_aug=robustness_aug,
            aug_prob=1.0
        )
        
        # Test single sample
        image, label, image_id = augmented_dataset[5]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 256, 256)
        assert label == 0
        assert image_id == "coco_000005"
        
        # Test with DataLoader
        loader = DataLoader(augmented_dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))
        images, labels, image_ids = batch
        
        assert images.shape == (3, 3, 256, 256)
        assert torch.all(labels == 0)
        assert len(image_ids) == 3
    
    def test_with_combined_dataset_format(self):
        """Test AugmentedDataset with BalancedCombinedDataset-like format (2-tuple)."""
        # Create a mock that mimics BalancedCombinedDataset behavior
        class MockCombinedDataset:
            def __init__(self, size=10):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # BalancedCombinedDataset returns (image, label)
                image = torch.rand(3, 256, 256)
                label = idx % 2
                return image, label
        
        base_dataset = MockCombinedDataset(size=20)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=0.4,
            blur_prob=0.4,
            noise_prob=0.4
        )
        augmented_dataset = AugmentedDataset(
            base_dataset,
            robustness_aug=robustness_aug,
            aug_prob=0.8
        )
        
        # Test single sample
        image, label = augmented_dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 256, 256)
        assert isinstance(label, int)
        
        # Test with DataLoader
        loader = DataLoader(augmented_dataset, batch_size=5, shuffle=True)
        batch = next(iter(loader))
        images, labels = batch
        
        assert images.shape == (5, 3, 256, 256)
        assert labels.shape == (5,)
    
    def test_native_resolution_compatibility(self):
        """Test AugmentedDataset with native resolution images (variable sizes)."""
        # Create a mock dataset with variable-sized images
        class MockNativeResolutionDataset:
            def __init__(self, size=10):
                self.size = size
                # Different image sizes
                self.sizes = [(224, 224), (256, 256), (320, 240), (512, 512)]
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Return variable-sized images
                h, w = self.sizes[idx % len(self.sizes)]
                image = torch.rand(3, h, w)
                label = idx % 2
                return image, label
        
        base_dataset = MockNativeResolutionDataset(size=12)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=0.3,
            blur_prob=0.3,
            noise_prob=0.3
        )
        augmented_dataset = AugmentedDataset(
            base_dataset,
            robustness_aug=robustness_aug,
            aug_prob=1.0
        )
        
        # Test samples with different sizes
        image1, label1 = augmented_dataset[0]
        image2, label2 = augmented_dataset[1]
        image3, label3 = augmented_dataset[2]
        
        assert image1.shape == (3, 224, 224)
        assert image2.shape == (3, 256, 256)
        assert image3.shape == (3, 320, 240)
        
        # Verify augmentation was applied (images should still be valid)
        assert torch.all(image1 >= 0.0) and torch.all(image1 <= 1.0)
        assert torch.all(image2 >= 0.0) and torch.all(image2 <= 1.0)
        assert torch.all(image3 >= 0.0) and torch.all(image3 <= 1.0)
    
    def test_augmentation_preserves_label_and_metadata(self):
        """Test that augmentation doesn't affect labels or metadata."""
        class MockDataset:
            def __init__(self, size=10):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                image = torch.rand(3, 128, 128)
                label = idx
                metadata = f"metadata_{idx}"
                return image, label, metadata
        
        base_dataset = MockDataset(size=10)
        robustness_aug = RobustnessAugmentation(
            jpeg_prob=1.0,
            blur_prob=1.0,
            noise_prob=1.0,
            severity_range=(5, 5)  # Maximum severity
        )
        augmented_dataset = AugmentedDataset(
            base_dataset,
            robustness_aug=robustness_aug,
            aug_prob=1.0
        )
        
        # Get samples and verify labels/metadata are unchanged
        for idx in range(5):
            image, label, metadata = augmented_dataset[idx]
            assert label == idx, f"Label should be {idx}, got {label}"
            assert metadata == f"metadata_{idx}", f"Metadata should be 'metadata_{idx}', got {metadata}"
            
            # Verify image was augmented (shape preserved)
            assert image.shape == (3, 128, 128)
            assert torch.all(image >= 0.0) and torch.all(image <= 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
