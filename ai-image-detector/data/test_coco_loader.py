"""
Unit tests for COCO 2017 Dataset Loader

Tests cover:
- Loading images with correct shapes and labels
- Native resolution mode
- Max samples parameter

Requirements:
    pip install torch torchvision numpy Pillow pytest

Usage:
    # Run all tests
    pytest ai-image-detector/data/test_coco_loader.py -v
    
    # Run specific test
    pytest ai-image-detector/data/test_coco_loader.py::TestCOCO2017Dataset::test_native_resolution_mode -v
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from .coco_loader import COCO2017Dataset


class TestCOCO2017Dataset(unittest.TestCase):
    """Test cases for COCO2017Dataset class."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary dataset directory with sample images."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dataset_root = Path(cls.temp_dir) / "coco_test"
        cls.dataset_root.mkdir()
        
        # Create train2017 directory
        cls.train_dir = cls.dataset_root / "train2017"
        cls.train_dir.mkdir()
        
        # Create sample images with different sizes
        cls.num_images = 5
        cls.image_sizes = [(512, 512), (640, 480), (800, 600), (1024, 768), (256, 256)]
        
        for i in range(cls.num_images):
            # Create a random RGB image with varying sizes
            size = cls.image_sizes[i]
            img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(cls.train_dir / f"image_{i:06d}.jpg")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_dataset_initialization(self):
        """Test that dataset initializes correctly."""
        dataset = COCO2017Dataset(str(self.dataset_root))
        
        # Should have all images
        self.assertEqual(len(dataset), self.num_images)
    
    def test_image_shape_and_type_default(self):
        """Test that loaded images have correct shape and type with default settings."""
        dataset = COCO2017Dataset(str(self.dataset_root))
        
        # Get first sample
        image_tensor, label = dataset[0]
        
        # Check tensor properties
        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(image_tensor.shape, (3, 256, 256))  # Default resize to 256x256
        self.assertEqual(image_tensor.dtype, torch.float32)
        
        # Check normalization (values should be in [0, 1])
        self.assertTrue(torch.all(image_tensor >= 0))
        self.assertTrue(torch.all(image_tensor <= 1))
    
    def test_label_assignment(self):
        """Test that all labels are 0 (real images)."""
        dataset = COCO2017Dataset(str(self.dataset_root))
        
        # Check all samples
        for i in range(len(dataset)):
            _, label = dataset[i]
            self.assertEqual(label, 0, "COCO images should always have label 0 (real)")
    
    def test_native_resolution_mode(self):
        """Test that native_resolution=True preserves original image dimensions."""
        # Create dataset with native_resolution=True
        dataset = COCO2017Dataset(str(self.dataset_root), native_resolution=True)
        
        # Check that images preserve their original dimensions
        for i in range(len(dataset)):
            image_tensor, label = dataset[i]
            
            # Image should be a tensor
            self.assertIsInstance(image_tensor, torch.Tensor)
            
            # Should have 3 channels
            self.assertEqual(image_tensor.shape[0], 3)
            
            # Dimensions should match one of our original sizes (H, W)
            height, width = image_tensor.shape[1], image_tensor.shape[2]
            original_size = self.image_sizes[i]
            self.assertEqual((height, width), original_size)
            
            # Check normalization
            self.assertTrue(torch.all(image_tensor >= 0))
            self.assertTrue(torch.all(image_tensor <= 1))
    
    def test_native_resolution_default_false(self):
        """Test that native_resolution defaults to False for backward compatibility."""
        # Create dataset without specifying native_resolution
        dataset = COCO2017Dataset(str(self.dataset_root))
        
        # Get first sample - should be resized to 256x256
        image_tensor, label = dataset[0]
        self.assertEqual(image_tensor.shape, (3, 256, 256))
    
    def test_native_resolution_with_custom_transform(self):
        """Test that custom transform overrides native_resolution flag."""
        from torchvision import transforms
        
        # Create custom transform that resizes to 128x128
        custom_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        # Even with native_resolution=True, custom transform should be used
        dataset = COCO2017Dataset(
            str(self.dataset_root),
            transform=custom_transform,
            native_resolution=True
        )
        
        # Get first sample - should be 128x128 due to custom transform
        image_tensor, label = dataset[0]
        self.assertEqual(image_tensor.shape, (3, 128, 128))
    
    def test_max_samples_parameter(self):
        """Test that max_samples parameter limits the dataset size."""
        max_samples = 3
        dataset = COCO2017Dataset(str(self.dataset_root), max_samples=max_samples)
        
        # Should only have max_samples images
        self.assertEqual(len(dataset), max_samples)
    
    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = COCO2017Dataset(str(self.dataset_root))
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get first batch
        batch = next(iter(dataloader))
        images, labels = batch
        
        # Check batch properties
        self.assertEqual(images.shape, (2, 3, 256, 256))
        self.assertEqual(labels.shape, (2,))
        self.assertTrue(torch.all(labels == 0))


if __name__ == "__main__":
    unittest.main()
