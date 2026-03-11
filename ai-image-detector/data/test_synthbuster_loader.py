"""
Unit tests for SynthBuster Dataset Loader

Tests cover:
- Loading images with correct shapes and labels
- Handling corrupted images
- Train/validation split ratios
- Generator subset mapping

Requirements:
    pip install torch torchvision numpy Pillow pytest

Usage:
    # Run all tests
    pytest ai-image-detector/data/test_synthbuster_loader.py -v
    
    # Run specific test class
    pytest ai-image-detector/data/test_synthbuster_loader.py::TestSynthBusterDataset -v
    
    # Run specific test
    pytest ai-image-detector/data/test_synthbuster_loader.py::TestSynthBusterDataset::test_image_shape_and_type -v
    
    # Run with unittest
    python -m unittest ai-image-detector.data.test_synthbuster_loader
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from .synthbuster_loader import (
    SynthBusterDataset,
    create_train_val_split,
    get_generator_subsets
)


class TestSynthBusterDataset(unittest.TestCase):
    """Test cases for SynthBusterDataset class."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary dataset directory with sample images."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dataset_root = Path(cls.temp_dir) / "synthbuster_test"
        cls.dataset_root.mkdir()
        
        # Create generator directories
        cls.generators = {
            'RAISE': 0,  # Real images, label=0
            'SD_v2': 1,  # Fake images, label=1
            'GLIDE': 1,
            'Midjourney': 1
        }
        
        # Create sample images for each generator
        cls.images_per_generator = 5
        for generator_name, label in cls.generators.items():
            generator_dir = cls.dataset_root / generator_name
            generator_dir.mkdir()
            
            for i in range(cls.images_per_generator):
                # Create a random RGB image
                img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(generator_dir / f"image_{i}.jpg")
        
        # Create a corrupted image for testing error handling
        corrupted_dir = cls.dataset_root / "RAISE"
        with open(corrupted_dir / "corrupted.jpg", 'w') as f:
            f.write("This is not a valid image file")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_dataset_initialization(self):
        """Test that dataset initializes correctly."""
        dataset = SynthBusterDataset(str(self.dataset_root))
        
        # Should have images from all generators plus one corrupted
        expected_count = len(self.generators) * self.images_per_generator + 1
        self.assertEqual(len(dataset), expected_count)
    
    def test_image_shape_and_type(self):
        """Test that loaded images have correct shape and type."""
        dataset = SynthBusterDataset(str(self.dataset_root))
        
        # Get first valid sample
        image_tensor, label, generator_name = dataset[0]
        
        # Check tensor properties
        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(image_tensor.shape, (3, 256, 256))  # (C, H, W)
        self.assertEqual(image_tensor.dtype, torch.float32)
        
        # Check ImageNet normalization (values should be approximately in [-2.1, 2.6])
        self.assertLess(image_tensor.min().item(), 0)  # Should have negative values
        self.assertGreater(image_tensor.max().item(), 1)  # Should exceed 1.0
    
    def test_label_assignment(self):
        """Test that labels are correctly assigned (0 for RAISE, 1 for others)."""
        dataset = SynthBusterDataset(str(self.dataset_root))
        
        # Check all samples
        raise_count = 0
        fake_count = 0
        
        for i in range(len(dataset)):
            _, label, generator_name = dataset[i]
            
            if generator_name == "RAISE":
                self.assertEqual(label, 0, f"RAISE images should have label 0")
                raise_count += 1
            else:
                self.assertEqual(label, 1, f"{generator_name} images should have label 1")
                fake_count += 1
        
        # Verify we have both real and fake images
        self.assertGreater(raise_count, 0)
        self.assertGreater(fake_count, 0)
    
    def test_generator_name_returned(self):
        """Test that generator names are correctly returned."""
        dataset = SynthBusterDataset(str(self.dataset_root))
        
        generator_names_found = set()
        for i in range(len(dataset)):
            _, _, generator_name = dataset[i]
            generator_names_found.add(generator_name)
        
        # Should find all our test generators
        expected_generators = set(self.generators.keys())
        self.assertEqual(generator_names_found, expected_generators)
    
    def test_corrupted_image_handling(self):
        """Test that corrupted images are skipped gracefully."""
        dataset = SynthBusterDataset(str(self.dataset_root))
        
        # Find the index of the corrupted image
        corrupted_idx = None
        for i, sample in enumerate(dataset.samples):
            if sample['path'].name == 'corrupted.jpg':
                corrupted_idx = i
                break
        
        self.assertIsNotNone(corrupted_idx, "Corrupted image should be in dataset index")
        
        # Accessing corrupted image should return next valid image with warning
        with self.assertWarns(UserWarning):
            image_tensor, label, generator_name = dataset[corrupted_idx]
            
            # Should still return a valid tensor
            self.assertIsInstance(image_tensor, torch.Tensor)
            self.assertEqual(image_tensor.shape, (3, 256, 256))
    
    def test_index_out_of_range(self):
        """Test that accessing invalid index raises IndexError."""
        dataset = SynthBusterDataset(str(self.dataset_root))
        
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset) + 10]
    
    def test_empty_directory_warning(self):
        """Test that empty directory triggers warning."""
        empty_dir = Path(self.temp_dir) / "empty_dataset"
        empty_dir.mkdir()
        
        with self.assertWarns(UserWarning):
            dataset = SynthBusterDataset(str(empty_dir))
            self.assertEqual(len(dataset), 0)
        
        shutil.rmtree(empty_dir)
    
    def test_nonexistent_directory(self):
        """Test that nonexistent directory raises ValueError."""
        with self.assertRaises(ValueError):
            SynthBusterDataset("/nonexistent/path/to/dataset")
    
    def test_custom_transform(self):
        """Test that custom transforms are applied correctly."""
        from torchvision import transforms
        
        # Custom transform: resize to 128x128 instead of 256x256
        custom_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        dataset = SynthBusterDataset(str(self.dataset_root), transform=custom_transform)
        image_tensor, _, _ = dataset[0]
        
        # Should have custom size
        self.assertEqual(image_tensor.shape, (3, 128, 128))
    
    def test_native_resolution_mode(self):
        """Test that native_resolution=True preserves original image dimensions."""
        # Create dataset with native_resolution=True
        dataset = SynthBusterDataset(str(self.dataset_root), native_resolution=True)
        
        # Get first sample
        image_tensor, label, generator_name = dataset[0]
        
        # Check tensor properties
        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(image_tensor.dtype, torch.float32)
        
        # Check that dimensions are preserved (original images are 512x512)
        self.assertEqual(image_tensor.shape, (3, 512, 512))
        
        # Check ImageNet normalization (values should be approximately in [-2.1, 2.6])
        self.assertLess(image_tensor.min().item(), 0)  # Should have negative values
        self.assertGreater(image_tensor.max().item(), 1)  # Should exceed 1.0
    
    def test_native_resolution_default_false(self):
        """Test that native_resolution defaults to False for backward compatibility."""
        # Create dataset without specifying native_resolution
        dataset = SynthBusterDataset(str(self.dataset_root))
        
        # Should resize to 256x256 by default
        image_tensor, _, _ = dataset[0]
        self.assertEqual(image_tensor.shape, (3, 256, 256))
    
    def test_native_resolution_with_custom_transform(self):
        """Test that custom transform overrides native_resolution flag."""
        from torchvision import transforms
        
        # Custom transform that resizes to 128x128
        custom_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        # Even with native_resolution=True, custom transform should be used
        dataset = SynthBusterDataset(
            str(self.dataset_root),
            transform=custom_transform,
            native_resolution=True
        )
        
        image_tensor, _, _ = dataset[0]
        
        # Should use custom transform size, not native resolution
        self.assertEqual(image_tensor.shape, (3, 128, 128))


class TestTrainValSplit(unittest.TestCase):
    """Test cases for create_train_val_split function."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary dataset directory."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dataset_root = Path(cls.temp_dir) / "synthbuster_split"
        cls.dataset_root.mkdir()
        
        # Create sample images
        cls.total_images = 100
        generator_dir = cls.dataset_root / "RAISE"
        generator_dir.mkdir()
        
        for i in range(cls.total_images):
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(generator_dir / f"image_{i}.jpg")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_split_ratio_default(self):
        """Test default 80/20 train/val split."""
        train_paths, val_paths = create_train_val_split(str(self.dataset_root))
        
        # Check split ratio (default 0.2 for validation)
        expected_val_size = int(self.total_images * 0.2)
        expected_train_size = self.total_images - expected_val_size
        
        self.assertEqual(len(val_paths), expected_val_size)
        self.assertEqual(len(train_paths), expected_train_size)
    
    def test_split_ratio_custom(self):
        """Test custom split ratios."""
        test_ratios = [0.1, 0.3, 0.5]
        
        for val_ratio in test_ratios:
            train_paths, val_paths = create_train_val_split(
                str(self.dataset_root),
                val_ratio=val_ratio
            )
            
            expected_val_size = int(self.total_images * val_ratio)
            expected_train_size = self.total_images - expected_val_size
            
            self.assertEqual(len(val_paths), expected_val_size,
                           f"Failed for val_ratio={val_ratio}")
            self.assertEqual(len(train_paths), expected_train_size,
                           f"Failed for val_ratio={val_ratio}")
    
    def test_split_no_overlap(self):
        """Test that train and val sets don't overlap."""
        train_paths, val_paths = create_train_val_split(str(self.dataset_root))
        
        train_set = set(train_paths)
        val_set = set(val_paths)
        
        # No overlap
        self.assertEqual(len(train_set & val_set), 0)
        
        # All images accounted for
        self.assertEqual(len(train_set) + len(val_set), self.total_images)
    
    def test_split_reproducibility(self):
        """Test that same seed produces same split."""
        train1, val1 = create_train_val_split(str(self.dataset_root), seed=42)
        train2, val2 = create_train_val_split(str(self.dataset_root), seed=42)
        
        self.assertEqual(train1, train2)
        self.assertEqual(val1, val2)
    
    def test_split_different_seeds(self):
        """Test that different seeds produce different splits."""
        train1, val1 = create_train_val_split(str(self.dataset_root), seed=42)
        train2, val2 = create_train_val_split(str(self.dataset_root), seed=123)
        
        # Should be different (with high probability)
        self.assertNotEqual(train1, train2)
    
    def test_split_nonexistent_directory(self):
        """Test that nonexistent directory raises ValueError."""
        with self.assertRaises(ValueError):
            create_train_val_split("/nonexistent/path")
    
    def test_split_empty_directory(self):
        """Test that empty directory raises ValueError."""
        empty_dir = Path(self.temp_dir) / "empty_split"
        empty_dir.mkdir()
        
        with self.assertRaises(ValueError):
            create_train_val_split(str(empty_dir))
        
        shutil.rmtree(empty_dir)


class TestGeneratorSubsets(unittest.TestCase):
    """Test cases for get_generator_subsets function."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary dataset directory with multiple generators."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dataset_root = Path(cls.temp_dir) / "synthbuster_subsets"
        cls.dataset_root.mkdir()
        
        # Create standard generators with different numbers of images
        cls.generator_counts = {
            'RAISE': 10,
            'SD_v2': 15,
            'GLIDE': 8,
            'Firefly': 12,
            'DALLE': 20,
            'Midjourney': 5
        }
        
        for generator_name, count in cls.generator_counts.items():
            generator_dir = cls.dataset_root / generator_name
            generator_dir.mkdir()
            
            for i in range(count):
                img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(generator_dir / f"image_{i}.jpg")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_all_generators_found(self):
        """Test that all generator directories are found."""
        subsets = get_generator_subsets(str(self.dataset_root))
        
        # Should find all generators
        self.assertEqual(set(subsets.keys()), set(self.generator_counts.keys()))
    
    def test_correct_image_counts(self):
        """Test that correct number of images per generator."""
        subsets = get_generator_subsets(str(self.dataset_root))
        
        for generator_name, expected_count in self.generator_counts.items():
            self.assertIn(generator_name, subsets)
            self.assertEqual(len(subsets[generator_name]), expected_count,
                           f"Wrong count for {generator_name}")
    
    def test_paths_are_valid(self):
        """Test that returned paths are valid Path objects."""
        subsets = get_generator_subsets(str(self.dataset_root))
        
        for generator_name, paths in subsets.items():
            for path in paths:
                self.assertIsInstance(path, Path)
                self.assertTrue(path.exists())
                self.assertTrue(path.is_file())
    
    def test_unsupported_generator_warning(self):
        """Test that unsupported generator triggers warning."""
        # Add an unsupported generator
        unsupported_dir = self.dataset_root / "UnknownGenerator"
        unsupported_dir.mkdir()
        
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(unsupported_dir / "image_0.jpg")
        
        with self.assertWarns(UserWarning):
            subsets = get_generator_subsets(str(self.dataset_root))
            
            # Should still include the unsupported generator
            self.assertIn("UnknownGenerator", subsets)
        
        shutil.rmtree(unsupported_dir)
    
    def test_empty_generator_directory(self):
        """Test that empty generator directories are not included."""
        # Create empty directory
        empty_dir = self.dataset_root / "EmptyGenerator"
        empty_dir.mkdir()
        
        subsets = get_generator_subsets(str(self.dataset_root))
        
        # Should not include empty directory
        self.assertNotIn("EmptyGenerator", subsets)
        
        shutil.rmtree(empty_dir)
    
    def test_nonexistent_directory(self):
        """Test that nonexistent directory raises ValueError."""
        with self.assertRaises(ValueError):
            get_generator_subsets("/nonexistent/path")
    
    def test_multiple_image_formats(self):
        """Test that multiple image formats are recognized."""
        # Create a generator with different image formats
        mixed_dir = self.dataset_root / "MixedFormats"
        mixed_dir.mkdir()
        
        formats = ['.jpg', '.png', '.jpeg', '.bmp']
        for i, ext in enumerate(formats):
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(mixed_dir / f"image_{i}{ext}")
        
        subsets = get_generator_subsets(str(self.dataset_root))
        
        # Should find all formats
        self.assertIn("MixedFormats", subsets)
        self.assertEqual(len(subsets["MixedFormats"]), len(formats))
        
        shutil.rmtree(mixed_dir)
    
    def test_subset_filtering_by_generator(self):
        """Test that we can filter specific generators."""
        subsets = get_generator_subsets(str(self.dataset_root))
        
        # Get only real images (RAISE)
        raise_images = subsets.get('RAISE', [])
        self.assertEqual(len(raise_images), self.generator_counts['RAISE'])
        
        # Get only SD_v2 images
        sd_images = subsets.get('SD_v2', [])
        self.assertEqual(len(sd_images), self.generator_counts['SD_v2'])


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    @classmethod
    def setUpClass(cls):
        """Create a complete test dataset."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dataset_root = Path(cls.temp_dir) / "synthbuster_integration"
        cls.dataset_root.mkdir()
        
        # Create realistic dataset structure
        cls.generators = ['RAISE', 'SD_v2', 'GLIDE']
        cls.images_per_gen = 20
        
        for generator_name in cls.generators:
            generator_dir = cls.dataset_root / generator_name
            generator_dir.mkdir()
            
            for i in range(cls.images_per_gen):
                img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(generator_dir / f"image_{i}.jpg")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_dataset_and_split_consistency(self):
        """Test that dataset and split functions work together."""
        # Create dataset
        dataset = SynthBusterDataset(str(self.dataset_root))
        
        # Create split
        train_paths, val_paths = create_train_val_split(str(self.dataset_root))
        
        # Total should match
        total_images = len(self.generators) * self.images_per_gen
        self.assertEqual(len(dataset), total_images)
        self.assertEqual(len(train_paths) + len(val_paths), total_images)
    
    def test_dataset_and_subsets_consistency(self):
        """Test that dataset and subsets functions agree on counts."""
        dataset = SynthBusterDataset(str(self.dataset_root))
        subsets = get_generator_subsets(str(self.dataset_root))
        
        # Count images per generator in dataset
        dataset_counts = {}
        for i in range(len(dataset)):
            _, _, generator_name = dataset[i]
            dataset_counts[generator_name] = dataset_counts.get(generator_name, 0) + 1
        
        # Should match subset counts
        for generator_name, paths in subsets.items():
            self.assertEqual(len(paths), dataset_counts[generator_name])
    
    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = SynthBusterDataset(str(self.dataset_root))
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Get one batch
        batch = next(iter(dataloader))
        images, labels, generator_names = batch
        
        # Check batch properties
        self.assertEqual(images.shape, (4, 3, 256, 256))
        self.assertEqual(labels.shape, (4,))
        self.assertEqual(len(generator_names), 4)


if __name__ == '__main__':
    unittest.main()
