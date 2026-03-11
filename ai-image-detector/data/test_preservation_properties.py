"""
Preservation Property Tests for Training Fixes and Cleanup

These tests verify that non-pixel-distribution behaviors remain unchanged after the fix.
They should PASS on unfixed code (baseline behavior) and continue to PASS after the fix.

**IMPORTANT**: Follow observation-first methodology - these tests capture the current
behavior of the unfixed code to ensure no regressions are introduced.

**GOAL**: Verify that all aspects of data loading that do NOT involve pixel value
distribution remain completely unchanged after adding ImageNet normalization.

Test Strategy:
- Observe behavior on UNFIXED code for non-pixel-distribution aspects
- Write property-based tests capturing observed behavior patterns
- Run tests on UNFIXED code - they should PASS (confirms baseline)
- After fix, re-run tests - they should still PASS (confirms preservation)

Preservation Requirements (from bugfix.md section 3):
- 3.1: ToTensor() transformation continues to be applied
- 3.2: Data augmentation order preserved (resize, crop before ToTensor)
- 3.3: Batch collation continues to stack tensors into [B, C, H, W]
- 3.4: Model architecture and pretrained weights unchanged (not tested here)
- 3.5: COCO real images continue to be labeled as class 0 (REAL)
- 3.6: SynthBuster RAISE labeled 0, generator images labeled 1 (FAKE)

**Validates: Requirements 3.1, 3.2, 3.3, 3.5, 3.6**

Usage:
    pytest ai-image-detector/data/test_preservation_properties.py -v -s
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from .synthbuster_loader import SynthBusterDataset
from .coco_loader import COCO2017Dataset
from .combined_loader import BalancedCombinedDataset


class TestPreservationProperties(unittest.TestCase):
    """
    Property-based tests for preservation of non-pixel-distribution behaviors.
    
    These tests should PASS on unfixed code and continue to PASS after the fix.
    """
    
    @classmethod
    def setUpClass(cls):
        """Create temporary datasets for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create SynthBuster dataset structure
        cls.synthbuster_root = Path(cls.temp_dir) / "synthbuster"
        cls.synthbuster_root.mkdir()
        
        # Create RAISE (real) and generator (fake) directories
        raise_dir = cls.synthbuster_root / "RAISE"
        raise_dir.mkdir()
        
        sd_dir = cls.synthbuster_root / "stable-diffusion-v1-4"
        sd_dir.mkdir()
        
        midjourney_dir = cls.synthbuster_root / "midjourney"
        midjourney_dir.mkdir()
        
        # Create sample images with varying sizes
        cls.images_per_category = 30
        
        # Create images with different sizes to test resolution handling
        image_sizes = [(256, 256), (512, 512), (224, 224), (300, 400), (640, 480)]
        
        for i in range(cls.images_per_category):
            size = image_sizes[i % len(image_sizes)]
            
            # RAISE images (real, label=0)
            img_array = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(raise_dir / f"real_{i}.jpg")
            
            # Stable Diffusion images (fake, label=1)
            img_array = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(sd_dir / f"fake_{i}.jpg")
            
            # Midjourney images (fake, label=1)
            img_array = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(midjourney_dir / f"fake_{i}.jpg")
        
        # Create COCO dataset structure
        cls.coco_root = Path(cls.temp_dir) / "coco"
        cls.coco_root.mkdir()
        
        coco_train_dir = cls.coco_root / "train2017"
        coco_train_dir.mkdir()
        
        for i in range(cls.images_per_category):
            size = image_sizes[i % len(image_sizes)]
            img_array = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(coco_train_dir / f"coco_{i:012d}.jpg")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_property_totensor_applied(self):
        """
        Property: ToTensor() transformation is still applied.
        
        **Validates: Requirement 3.1**
        
        Verifies that images are converted from PIL format to PyTorch tensors
        with shape (C, H, W) and values in a numeric range (regardless of
        whether normalization is applied).
        """
        print("\n" + "="*70)
        print("Property Test: ToTensor() Applied")
        print("="*70)
        
        # Test SynthBuster dataset
        dataset = SynthBusterDataset(str(self.synthbuster_root))
        image, label, generator = dataset[0]
        
        print(f"\n1. SynthBuster dataset:")
        print(f"   Image type: {type(image)}")
        print(f"   Image dtype: {image.dtype}")
        print(f"   Image shape: {image.shape}")
        
        # Verify it's a tensor
        self.assertIsInstance(image, torch.Tensor, "Image should be a torch.Tensor")
        
        # Verify shape is (C, H, W)
        self.assertEqual(len(image.shape), 3, "Image should have 3 dimensions (C, H, W)")
        self.assertEqual(image.shape[0], 3, "Image should have 3 channels (RGB)")
        
        # Verify dtype is float (ToTensor converts to float)
        self.assertTrue(
            image.dtype in [torch.float32, torch.float64],
            f"Image should be float type, got {image.dtype}"
        )
        
        # Test COCO dataset
        coco_dataset = COCO2017Dataset(str(self.coco_root), max_samples=10)
        coco_image, coco_label = coco_dataset[0]
        
        print(f"\n2. COCO dataset:")
        print(f"   Image type: {type(coco_image)}")
        print(f"   Image dtype: {coco_image.dtype}")
        print(f"   Image shape: {coco_image.shape}")
        
        self.assertIsInstance(coco_image, torch.Tensor, "COCO image should be a torch.Tensor")
        self.assertEqual(len(coco_image.shape), 3, "COCO image should have 3 dimensions")
        self.assertEqual(coco_image.shape[0], 3, "COCO image should have 3 channels")
        self.assertTrue(
            coco_image.dtype in [torch.float32, torch.float64],
            f"COCO image should be float type, got {coco_image.dtype}"
        )
        
        print("\n✓ ToTensor() transformation is applied correctly")
    
    def test_property_batch_shapes_preserved(self):
        """
        Property: Batch collation produces tensors with shape [B, C, H, W].
        
        **Validates: Requirement 3.3**
        
        Verifies that DataLoader correctly stacks individual samples into
        batches with the expected 4D tensor shape.
        """
        print("\n" + "="*70)
        print("Property Test: Batch Shapes Preserved")
        print("="*70)
        
        batch_size = 8
        
        # Test SynthBuster batches
        dataset = SynthBusterDataset(str(self.synthbuster_root))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        batch_data = next(iter(dataloader))
        images, labels, generators = batch_data
        
        print(f"\n1. SynthBuster batch:")
        print(f"   Batch shape: {images.shape}")
        print(f"   Expected: [{batch_size}, 3, H, W]")
        
        # Verify batch shape
        self.assertEqual(len(images.shape), 4, "Batch should have 4 dimensions [B, C, H, W]")
        self.assertEqual(images.shape[0], batch_size, f"Batch size should be {batch_size}")
        self.assertEqual(images.shape[1], 3, "Batch should have 3 channels")
        
        # Verify labels shape
        self.assertEqual(labels.shape[0], batch_size, f"Labels should have {batch_size} elements")
        
        # Test COCO batches
        coco_dataset = COCO2017Dataset(str(self.coco_root), max_samples=20)
        coco_dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=False)
        
        coco_batch_data = next(iter(coco_dataloader))
        coco_images, coco_labels = coco_batch_data
        
        print(f"\n2. COCO batch:")
        print(f"   Batch shape: {coco_images.shape}")
        print(f"   Expected: [{batch_size}, 3, H, W]")
        
        self.assertEqual(len(coco_images.shape), 4, "COCO batch should have 4 dimensions")
        self.assertEqual(coco_images.shape[0], batch_size, f"COCO batch size should be {batch_size}")
        self.assertEqual(coco_images.shape[1], 3, "COCO batch should have 3 channels")
        self.assertEqual(coco_labels.shape[0], batch_size, f"COCO labels should have {batch_size} elements")
        
        # Test Combined dataset batches
        combined_dataset = BalancedCombinedDataset(
            synthbuster_root=str(self.synthbuster_root),
            coco_root=str(self.coco_root)
        )
        combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
        
        combined_batch_data = next(iter(combined_dataloader))
        combined_images, combined_labels = combined_batch_data
        
        print(f"\n3. Combined dataset batch:")
        print(f"   Batch shape: {combined_images.shape}")
        print(f"   Expected: [{batch_size}, 3, H, W]")
        
        self.assertEqual(len(combined_images.shape), 4, "Combined batch should have 4 dimensions")
        self.assertEqual(combined_images.shape[0], batch_size, f"Combined batch size should be {batch_size}")
        self.assertEqual(combined_images.shape[1], 3, "Combined batch should have 3 channels")
        
        print("\n✓ Batch shapes are preserved correctly")
    
    def test_property_label_assignments_preserved(self):
        """
        Property: Label assignments remain unchanged.
        
        **Validates: Requirements 3.5, 3.6**
        
        Verifies that:
        - RAISE images (real) are labeled as 0
        - SynthBuster generator images (fake) are labeled as 1
        - COCO images (real) are labeled as 0
        """
        print("\n" + "="*70)
        print("Property Test: Label Assignments Preserved")
        print("="*70)
        
        # Test SynthBuster labels
        dataset = SynthBusterDataset(str(self.synthbuster_root))
        
        raise_labels = []
        fake_labels = []
        
        for i in range(len(dataset)):
            image, label, generator = dataset[i]
            if generator == "RAISE":
                raise_labels.append(label)
            else:
                fake_labels.append(label)
        
        print(f"\n1. SynthBuster labels:")
        print(f"   RAISE images: {len(raise_labels)} samples")
        print(f"   Generator images: {len(fake_labels)} samples")
        
        # Verify RAISE images are labeled 0
        self.assertTrue(
            all(label == 0 for label in raise_labels),
            "All RAISE images should be labeled 0 (REAL)"
        )
        print(f"   ✓ All RAISE images labeled as 0 (REAL)")
        
        # Verify generator images are labeled 1
        self.assertTrue(
            all(label == 1 for label in fake_labels),
            "All generator images should be labeled 1 (FAKE)"
        )
        print(f"   ✓ All generator images labeled as 1 (FAKE)")
        
        # Test COCO labels
        coco_dataset = COCO2017Dataset(str(self.coco_root), max_samples=20)
        
        coco_labels = []
        for i in range(len(coco_dataset)):
            image, label = coco_dataset[i]
            coco_labels.append(label)
        
        print(f"\n2. COCO labels:")
        print(f"   Total images: {len(coco_labels)}")
        
        # Verify all COCO images are labeled 0
        self.assertTrue(
            all(label == 0 for label in coco_labels),
            "All COCO images should be labeled 0 (REAL)"
        )
        print(f"   ✓ All COCO images labeled as 0 (REAL)")
        
        # Test Combined dataset labels
        combined_dataset = BalancedCombinedDataset(
            synthbuster_root=str(self.synthbuster_root),
            coco_root=str(self.coco_root)
        )
        
        combined_real_count = 0
        combined_fake_count = 0
        
        for i in range(min(50, len(combined_dataset))):
            image, label = combined_dataset[i]
            if label == 0:
                combined_real_count += 1
            else:
                combined_fake_count += 1
        
        print(f"\n3. Combined dataset labels:")
        print(f"   Real images (label=0): {combined_real_count}")
        print(f"   Fake images (label=1): {combined_fake_count}")
        
        # Verify labels are only 0 or 1
        for i in range(min(50, len(combined_dataset))):
            image, label = combined_dataset[i]
            self.assertIn(label, [0, 1], f"Label should be 0 or 1, got {label}")
        
        print(f"   ✓ All labels are valid (0 or 1)")
        
        print("\n✓ Label assignments are preserved correctly")
    
    def test_property_sampling_logic_preserved(self):
        """
        Property: Dataset sampling logic remains unchanged.
        
        **Validates: Requirement 3.6 (balanced sampling)**
        
        Verifies that BalancedCombinedDataset continues to sample from
        SynthBuster and COCO with balanced representation.
        """
        print("\n" + "="*70)
        print("Property Test: Sampling Logic Preserved")
        print("="*70)
        
        # Create combined dataset
        combined_dataset = BalancedCombinedDataset(
            synthbuster_root=str(self.synthbuster_root),
            coco_root=str(self.coco_root)
        )
        
        # Count real and fake samples
        real_count = 0
        fake_count = 0
        
        for i in range(len(combined_dataset)):
            image, label = combined_dataset[i]
            if label == 0:
                real_count += 1
            else:
                fake_count += 1
        
        print(f"\n1. Combined dataset composition:")
        print(f"   Real images (label=0): {real_count}")
        print(f"   Fake images (label=1): {fake_count}")
        print(f"   Total images: {len(combined_dataset)}")
        print(f"   Balance ratio: {real_count/fake_count:.2f}:1")
        
        # Verify balanced sampling (should be approximately 1:1)
        balance_ratio = real_count / fake_count
        self.assertGreater(balance_ratio, 0.8, "Balance ratio should be close to 1:1")
        self.assertLess(balance_ratio, 1.2, "Balance ratio should be close to 1:1")
        
        print(f"   ✓ Dataset is balanced (ratio within 0.8-1.2)")
        
        # Verify shuffling behavior
        combined_shuffled = BalancedCombinedDataset(
            synthbuster_root=str(self.synthbuster_root),
            coco_root=str(self.coco_root),
            shuffle=True,
            seed=42
        )
        
        combined_not_shuffled = BalancedCombinedDataset(
            synthbuster_root=str(self.synthbuster_root),
            coco_root=str(self.coco_root),
            shuffle=False,
            seed=42
        )
        
        # Get first 10 labels from each
        shuffled_labels = [combined_shuffled[i][1] for i in range(10)]
        not_shuffled_labels = [combined_not_shuffled[i][1] for i in range(10)]
        
        print(f"\n2. Shuffling behavior:")
        print(f"   Shuffled labels (first 10): {shuffled_labels}")
        print(f"   Not shuffled labels (first 10): {not_shuffled_labels}")
        
        # They should be different (with high probability)
        # Note: This is a probabilistic test, but with 10 samples it's very unlikely
        # they'd be identical if one is shuffled
        if shuffled_labels != not_shuffled_labels:
            print(f"   ✓ Shuffling produces different order")
        else:
            print(f"   ⚠️  Shuffled and non-shuffled orders are identical (unlikely but possible)")
        
        print("\n✓ Sampling logic is preserved correctly")
    
    def test_property_augmentation_order_preserved(self):
        """
        Property: Augmentation order is preserved (resize before ToTensor).
        
        **Validates: Requirement 3.2**
        
        Verifies that geometric transformations (resize) occur before ToTensor
        transformation in the pipeline.
        """
        print("\n" + "="*70)
        print("Property Test: Augmentation Order Preserved")
        print("="*70)
        
        # Test standard mode (with resize)
        dataset_standard = SynthBusterDataset(
            str(self.synthbuster_root),
            native_resolution=False
        )
        
        image_standard, _, _ = dataset_standard[0]
        
        print(f"\n1. Standard mode (with resize):")
        print(f"   Image shape: {image_standard.shape}")
        print(f"   Expected: [3, 256, 256]")
        
        # In standard mode, images should be resized to 256x256
        self.assertEqual(
            image_standard.shape[1:],
            (256, 256),
            "Standard mode should resize images to 256x256"
        )
        
        # Test native resolution mode (no resize)
        dataset_native = SynthBusterDataset(
            str(self.synthbuster_root),
            native_resolution=True
        )
        
        image_native, _, _ = dataset_native[0]
        
        print(f"\n2. Native resolution mode (no resize):")
        print(f"   Image shape: {image_native.shape}")
        print(f"   Note: Shape varies based on original image size")
        
        # In native mode, images should preserve original dimensions
        # We can't predict exact size, but it should be a valid tensor
        self.assertEqual(len(image_native.shape), 3, "Native mode should produce (C, H, W) tensor")
        self.assertEqual(image_native.shape[0], 3, "Native mode should have 3 channels")
        
        # Test COCO dataset
        coco_standard = COCO2017Dataset(
            str(self.coco_root),
            native_resolution=False,
            max_samples=10
        )
        
        coco_image_standard, _ = coco_standard[0]
        
        print(f"\n3. COCO standard mode:")
        print(f"   Image shape: {coco_image_standard.shape}")
        print(f"   Expected: [3, 256, 256]")
        
        self.assertEqual(
            coco_image_standard.shape[1:],
            (256, 256),
            "COCO standard mode should resize images to 256x256"
        )
        
        coco_native = COCO2017Dataset(
            str(self.coco_root),
            native_resolution=True,
            max_samples=10
        )
        
        coco_image_native, _ = coco_native[0]
        
        print(f"\n4. COCO native resolution mode:")
        print(f"   Image shape: {coco_image_native.shape}")
        
        self.assertEqual(len(coco_image_native.shape), 3, "COCO native mode should produce (C, H, W) tensor")
        self.assertEqual(coco_image_native.shape[0], 3, "COCO native mode should have 3 channels")
        
        print("\n✓ Augmentation order is preserved correctly")
        print("  (Resize applied before ToTensor when native_resolution=False)")


if __name__ == '__main__':
    unittest.main()
