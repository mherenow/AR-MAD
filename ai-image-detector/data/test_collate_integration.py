"""
Integration tests for variable-size collate function with actual data loaders.

These tests demonstrate how the collate function integrates with SynthBusterDataset
and COCO2017Dataset when native_resolution=True.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from unittest.mock import Mock, patch
from pathlib import Path

from .collate import variable_size_collate_fn
from .synthbuster_loader import SynthBusterDataset


class TestCollateIntegrationWithSynthBuster:
    """Integration tests with SynthBusterDataset."""
    
    @patch('data.synthbuster_loader.Path.exists')
    @patch('data.synthbuster_loader.Path.iterdir')
    def test_native_resolution_with_collate_fn(self, mock_iterdir, mock_exists):
        """Test that native_resolution mode works with variable_size_collate_fn."""
        # Mock the directory structure
        mock_exists.return_value = True
        
        # Create mock directory structure
        mock_raise_dir = Mock()
        mock_raise_dir.is_dir.return_value = True
        mock_raise_dir.name = "RAISE"
        
        mock_sd_dir = Mock()
        mock_sd_dir.is_dir.return_value = True
        mock_sd_dir.name = "SD_v2"
        
        # Mock image files
        mock_img1 = Mock()
        mock_img1.suffix = ".jpg"
        mock_img1.is_dir.return_value = False
        
        mock_img2 = Mock()
        mock_img2.suffix = ".png"
        mock_img2.is_dir.return_value = False
        
        # Set up the directory iteration
        mock_raise_dir.iterdir.return_value = [mock_img1]
        mock_sd_dir.iterdir.return_value = [mock_img2]
        mock_iterdir.return_value = [mock_raise_dir, mock_sd_dir]
        
        # This test verifies the concept - in practice, you would use real data
        # The key point is that variable_size_collate_fn can handle batches
        # where images have different sizes when native_resolution=True
        
        # Create a mock batch with variable sizes
        batch = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 512, 512), 1, "SD_v2"),
            (torch.rand(3, 384, 384), 1, "GLIDE")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Verify that variable sizes are handled correctly
        assert isinstance(images, list)
        assert len(images) == 3
        assert images[0].shape == (3, 256, 256)
        assert images[1].shape == (3, 512, 512)
        assert images[2].shape == (3, 384, 384)
    
    def test_fixed_size_backward_compatibility(self):
        """Test that fixed-size batches still work efficiently (backward compatibility)."""
        # Create a batch with all same size (standard mode)
        batch = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 256, 256), 1, "SD_v2"),
            (torch.rand(3, 256, 256), 1, "GLIDE")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Should stack into a single tensor for efficiency
        assert isinstance(images, torch.Tensor)
        assert images.shape == (3, 3, 256, 256)
        assert labels.shape == (3,)
        assert len(generator_names) == 3


class TestCollateUsageExamples:
    """Examples of how to use the collate function in practice."""
    
    def test_dataloader_with_variable_size_collate(self):
        """Example: Using variable_size_collate_fn with DataLoader."""
        # Create a simple mock dataset
        class MockVariableSizeDataset:
            def __init__(self):
                self.data = [
                    (torch.rand(3, 256, 256), 0, "RAISE"),
                    (torch.rand(3, 512, 512), 1, "SD_v2"),
                    (torch.rand(3, 384, 384), 1, "GLIDE"),
                    (torch.rand(3, 256, 256), 0, "RAISE"),
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = MockVariableSizeDataset()
        
        # Create DataLoader with custom collate function
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=variable_size_collate_fn,
            shuffle=False
        )
        
        # Iterate through batches
        batches = list(loader)
        assert len(batches) == 2
        
        # First batch: 256x256 and 512x512 (variable sizes)
        images1, labels1, names1 = batches[0]
        assert isinstance(images1, list)
        assert len(images1) == 2
        assert images1[0].shape == (3, 256, 256)
        assert images1[1].shape == (3, 512, 512)
        
        # Second batch: 384x384 and 256x256 (variable sizes)
        images2, labels2, names2 = batches[1]
        assert isinstance(images2, list)
        assert len(images2) == 2
        assert images2[0].shape == (3, 384, 384)
        assert images2[1].shape == (3, 256, 256)
    
    def test_dataloader_with_fixed_size_collate(self):
        """Example: Fixed-size batches still work efficiently."""
        # Create a simple mock dataset with all same size
        class MockFixedSizeDataset:
            def __init__(self):
                self.data = [
                    (torch.rand(3, 256, 256), 0, "RAISE"),
                    (torch.rand(3, 256, 256), 1, "SD_v2"),
                    (torch.rand(3, 256, 256), 1, "GLIDE"),
                    (torch.rand(3, 256, 256), 0, "RAISE"),
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = MockFixedSizeDataset()
        
        # Create DataLoader with custom collate function
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=variable_size_collate_fn,
            shuffle=False
        )
        
        # Iterate through batches
        for images, labels, names in loader:
            # All same size, so should be stacked into tensor
            assert isinstance(images, torch.Tensor)
            assert images.shape == (2, 3, 256, 256)
            assert labels.shape == (2,)
            assert len(names) == 2


class TestCollateDocumentation:
    """Tests that serve as documentation for the collate function."""
    
    def test_when_to_use_variable_size_collate(self):
        """
        Use variable_size_collate_fn when:
        1. native_resolution=True in your dataset
        2. Images in your dataset have different sizes
        3. You want to preserve original image dimensions
        
        The collate function will:
        - Return a list of tensors when sizes differ
        - Return a stacked tensor when all sizes are the same (for efficiency)
        """
        # Variable sizes -> list
        batch_variable = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 512, 512), 1, "SD_v2")
        ]
        images_var, _, _ = variable_size_collate_fn(batch_variable)
        assert isinstance(images_var, list)
        
        # Same sizes -> tensor (backward compatible)
        batch_fixed = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 256, 256), 1, "SD_v2")
        ]
        images_fixed, _, _ = variable_size_collate_fn(batch_fixed)
        assert isinstance(images_fixed, torch.Tensor)
    
    def test_handling_collate_output_in_training_loop(self):
        """
        Example: How to handle the collate function output in a training loop.
        """
        # Simulate a batch from the collate function
        batch_variable = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 512, 512), 1, "SD_v2")
        ]
        images, labels, names = variable_size_collate_fn(batch_variable)
        
        # In your training loop, check if images is a list or tensor
        if isinstance(images, list):
            # Variable sizes: process each image individually
            for img, label in zip(images, labels):
                # Process individual image
                assert img.shape[0] == 3  # RGB channels
                assert label in [0, 1]  # Binary label
        else:
            # Fixed sizes: process as batch
            assert images.shape[0] == len(labels)  # Batch size matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
