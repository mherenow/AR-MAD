"""
Unit tests for variable-size collate function.

Tests cover:
- Fixed-size batches (backward compatibility)
- Variable-size batches
- 2-tuple and 3-tuple formats
- Edge cases (single item, empty batch handling)
"""

import pytest
import torch
from .collate import (
    variable_size_collate_fn,
    variable_size_collate_fn_2tuple
)


class TestVariableSizeCollateFn:
    """Tests for variable_size_collate_fn (3-tuple format)."""
    
    def test_fixed_size_batch_stacks_tensors(self):
        """Test that fixed-size batches are stacked into a single tensor."""
        # Create batch with all same size (3, 256, 256)
        batch = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 256, 256), 1, "SD_v2"),
            (torch.rand(3, 256, 256), 1, "GLIDE")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Images should be stacked into a single tensor
        assert isinstance(images, torch.Tensor)
        assert images.shape == (3, 3, 256, 256)
        
        # Labels should be a tensor
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (3,)
        assert labels.tolist() == [0, 1, 1]
        
        # Generator names should be a list
        assert isinstance(generator_names, list)
        assert generator_names == ["RAISE", "SD_v2", "GLIDE"]
    
    def test_variable_size_batch_returns_list(self):
        """Test that variable-size batches return a list of tensors."""
        # Create batch with different sizes
        batch = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 512, 512), 1, "SD_v2"),
            (torch.rand(3, 384, 384), 1, "GLIDE")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Images should be a list of tensors
        assert isinstance(images, list)
        assert len(images) == 3
        assert images[0].shape == (3, 256, 256)
        assert images[1].shape == (3, 512, 512)
        assert images[2].shape == (3, 384, 384)
        
        # Labels should be a tensor
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (3,)
        assert labels.tolist() == [0, 1, 1]
        
        # Generator names should be a list
        assert generator_names == ["RAISE", "SD_v2", "GLIDE"]
    
    def test_single_item_batch(self):
        """Test batch with a single item."""
        batch = [(torch.rand(3, 256, 256), 1, "SD_v2")]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Single item should still be stacked (all same size trivially)
        assert isinstance(images, torch.Tensor)
        assert images.shape == (1, 3, 256, 256)
        assert labels.shape == (1,)
        assert labels.item() == 1
        assert generator_names == ["SD_v2"]
    
    def test_different_heights_same_width(self):
        """Test batch with different heights but same width."""
        batch = [
            (torch.rand(3, 256, 512), 0, "RAISE"),
            (torch.rand(3, 384, 512), 1, "SD_v2")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Should return list due to different heights
        assert isinstance(images, list)
        assert len(images) == 2
        assert images[0].shape == (3, 256, 512)
        assert images[1].shape == (3, 384, 512)
    
    def test_different_widths_same_height(self):
        """Test batch with different widths but same height."""
        batch = [
            (torch.rand(3, 512, 256), 0, "RAISE"),
            (torch.rand(3, 512, 384), 1, "SD_v2")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Should return list due to different widths
        assert isinstance(images, list)
        assert len(images) == 2
        assert images[0].shape == (3, 512, 256)
        assert images[1].shape == (3, 512, 384)
    
    def test_label_types(self):
        """Test that labels are correctly converted to tensor."""
        batch = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 256, 256), 1, "SD_v2")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        assert labels.dtype == torch.long
        assert labels.tolist() == [0, 1]
    
    def test_preserves_image_values(self):
        """Test that image tensor values are preserved during collation."""
        # Create specific tensors to verify values are preserved
        img1 = torch.ones(3, 256, 256) * 0.5
        img2 = torch.ones(3, 256, 256) * 0.8
        
        batch = [
            (img1, 0, "RAISE"),
            (img2, 1, "SD_v2")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Check that values are preserved
        assert torch.allclose(images[0], img1)
        assert torch.allclose(images[1], img2)
    
    def test_large_batch(self):
        """Test with a larger batch size."""
        batch_size = 32
        batch = [
            (torch.rand(3, 256, 256), i % 2, f"generator_{i}")
            for i in range(batch_size)
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        assert isinstance(images, torch.Tensor)
        assert images.shape == (batch_size, 3, 256, 256)
        assert labels.shape == (batch_size,)
        assert len(generator_names) == batch_size


class TestVariableSizeCollateFn2Tuple:
    """Tests for variable_size_collate_fn_2tuple (2-tuple format)."""
    
    def test_fixed_size_batch_stacks_tensors(self):
        """Test that fixed-size batches are stacked into a single tensor."""
        batch = [
            (torch.rand(3, 256, 256), 0),
            (torch.rand(3, 256, 256), 1),
            (torch.rand(3, 256, 256), 1)
        ]
        
        images, labels = variable_size_collate_fn_2tuple(batch)
        
        assert isinstance(images, torch.Tensor)
        assert images.shape == (3, 3, 256, 256)
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (3,)
        assert labels.tolist() == [0, 1, 1]
    
    def test_variable_size_batch_returns_list(self):
        """Test that variable-size batches return a list of tensors."""
        batch = [
            (torch.rand(3, 256, 256), 0),
            (torch.rand(3, 512, 512), 1),
            (torch.rand(3, 384, 384), 1)
        ]
        
        images, labels = variable_size_collate_fn_2tuple(batch)
        
        assert isinstance(images, list)
        assert len(images) == 3
        assert images[0].shape == (3, 256, 256)
        assert images[1].shape == (3, 512, 512)
        assert images[2].shape == (3, 384, 384)
        assert labels.tolist() == [0, 1, 1]
    
    def test_single_item_batch(self):
        """Test batch with a single item."""
        batch = [(torch.rand(3, 256, 256), 1)]
        
        images, labels = variable_size_collate_fn_2tuple(batch)
        
        assert isinstance(images, torch.Tensor)
        assert images.shape == (1, 3, 256, 256)
        assert labels.shape == (1,)
        assert labels.item() == 1
    
    def test_preserves_image_values(self):
        """Test that image tensor values are preserved during collation."""
        img1 = torch.ones(3, 256, 256) * 0.3
        img2 = torch.ones(3, 256, 256) * 0.7
        
        batch = [(img1, 0), (img2, 1)]
        
        images, labels = variable_size_collate_fn_2tuple(batch)
        
        assert torch.allclose(images[0], img1)
        assert torch.allclose(images[1], img2)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_grayscale_images(self):
        """Test with grayscale images (1 channel)."""
        batch = [
            (torch.rand(1, 256, 256), 0, "RAISE"),
            (torch.rand(1, 256, 256), 1, "SD_v2")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        assert isinstance(images, torch.Tensor)
        assert images.shape == (2, 1, 256, 256)
    
    def test_different_channel_counts(self):
        """Test with images having different channel counts."""
        batch = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(4, 256, 256), 1, "SD_v2")  # RGBA
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Should return list due to different channel counts
        assert isinstance(images, list)
        assert images[0].shape == (3, 256, 256)
        assert images[1].shape == (4, 256, 256)
    
    def test_very_small_images(self):
        """Test with very small images."""
        batch = [
            (torch.rand(3, 32, 32), 0, "RAISE"),
            (torch.rand(3, 32, 32), 1, "SD_v2")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        assert isinstance(images, torch.Tensor)
        assert images.shape == (2, 3, 32, 32)
    
    def test_very_large_images(self):
        """Test with very large images."""
        batch = [
            (torch.rand(3, 2048, 2048), 0, "RAISE"),
            (torch.rand(3, 2048, 2048), 1, "SD_v2")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        assert isinstance(images, torch.Tensor)
        assert images.shape == (2, 3, 2048, 2048)
    
    def test_mixed_label_types(self):
        """Test that integer labels are correctly handled."""
        batch = [
            (torch.rand(3, 256, 256), 0, "RAISE"),
            (torch.rand(3, 256, 256), 1, "SD_v2")
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        assert labels.dtype == torch.long
        assert labels[0].item() == 0
        assert labels[1].item() == 1


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""
    
    def test_standard_256x256_batch(self):
        """Test standard 256x256 batch (most common case)."""
        batch = [
            (torch.rand(3, 256, 256), i % 2, f"gen_{i}")
            for i in range(16)
        ]
        
        images, labels, generator_names = variable_size_collate_fn(batch)
        
        # Should stack into tensor for efficiency
        assert isinstance(images, torch.Tensor)
        assert images.shape == (16, 3, 256, 256)
        assert labels.shape == (16,)
        assert len(generator_names) == 16
    
    def test_can_be_used_with_dataloader(self):
        """Test that collate function works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create a simple dataset
        images = torch.rand(10, 3, 256, 256)
        labels = torch.randint(0, 2, (10,))
        
        # Create a custom dataset that returns 3-tuples
        class CustomDataset:
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx].item(), f"gen_{idx}"
        
        dataset = CustomDataset(images, labels)
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=variable_size_collate_fn
        )
        
        # Test that we can iterate through the loader
        for batch_images, batch_labels, batch_names in loader:
            assert isinstance(batch_images, torch.Tensor)
            assert batch_images.shape[0] == 4  # batch size
            assert batch_images.shape[1:] == (3, 256, 256)
            assert len(batch_labels) == 4
            assert len(batch_names) == 4
            break  # Just test first batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
