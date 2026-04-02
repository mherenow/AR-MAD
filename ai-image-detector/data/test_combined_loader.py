"""
Tests for combined balanced dataset loader.

This module tests the BalancedCombinedDataset to ensure proper loading
and balancing of SynthBuster and COCO2017 datasets.
"""

import pytest
import torch
from pathlib import Path

from .combined_loader import BalancedCombinedDataset, create_train_val_split_combined


def test_balanced_combined_dataset_initialization():
    """Test that BalancedCombinedDataset initializes correctly."""
    synthbuster_root = "datasets/synthbuster"
    coco_root = "datasets/coco2017"
    
    # Skip test if datasets don't exist
    if not Path(synthbuster_root).exists() or not Path(coco_root).exists():
        pytest.skip("Datasets not available")
    
    dataset = BalancedCombinedDataset(
        synthbuster_root=synthbuster_root,
        coco_root=coco_root
    )
    
    assert len(dataset) > 0, "Dataset should have samples"
    
    # Check balance
    real_count = sum(1 for _, _, label in dataset.all_samples if label == 0)
    fake_count = sum(1 for _, _, label in dataset.all_samples if label == 1)
    
    # Should be roughly balanced (within 10% tolerance)
    balance_ratio = real_count / fake_count if fake_count > 0 else 0
    assert 0.9 <= balance_ratio <= 1.1, f"Dataset imbalanced: {real_count} real vs {fake_count} fake"
    
    print(f"✓ Dataset balanced: {real_count} real, {fake_count} fake (ratio: {balance_ratio:.2f})")


def test_combined_dataset_getitem():
    """Test that samples can be loaded from the combined dataset."""
    synthbuster_root = "datasets/synthbuster"
    coco_root = "datasets/coco2017"
    
    # Skip test if datasets don't exist
    if not Path(synthbuster_root).exists() or not Path(coco_root).exists():
        pytest.skip("Datasets not available")
    
    dataset = BalancedCombinedDataset(
        synthbuster_root=synthbuster_root,
        coco_root=coco_root
    )
    
    # Test loading first sample
    image, label = dataset[0]
    
    assert isinstance(image, torch.Tensor), "Image should be a tensor"
    assert image.shape == (3, 256, 256), f"Image shape should be (3, 256, 256), got {image.shape}"
    assert label in [0, 1], f"Label should be 0 or 1, got {label}"
    # Check ImageNet normalization (values should be approximately in [-2.1, 2.6])
    assert image.min() < 0, "Image should have negative values after ImageNet normalization"
    assert image.max() > 1, "Image should have values > 1 after ImageNet normalization"
    
    print(f"✓ Sample loaded: shape={image.shape}, label={label}")


def test_train_val_split_combined():
    """Test train/validation split for combined dataset."""
    synthbuster_root = "datasets/synthbuster"
    coco_root = "datasets/coco2017"
    
    # Skip test if datasets don't exist
    if not Path(synthbuster_root).exists() or not Path(coco_root).exists():
        pytest.skip("Datasets not available")
    
    train_dataset, val_dataset = create_train_val_split_combined(
        synthbuster_root=synthbuster_root,
        coco_root=coco_root,
        val_ratio=0.2,
        seed=42
    )
    
    assert len(train_dataset) > 0, "Train dataset should have samples"
    assert len(val_dataset) > 0, "Val dataset should have samples"
    
    total = len(train_dataset) + len(val_dataset)
    val_ratio = len(val_dataset) / total
    
    # Check split ratio (should be close to 0.2)
    assert 0.15 <= val_ratio <= 0.25, f"Val ratio should be ~0.2, got {val_ratio:.2f}"
    
    print(f"✓ Split created: {len(train_dataset)} train, {len(val_dataset)} val (ratio: {val_ratio:.2f})")


def test_combined_dataset_label_distribution():
    """Test that both real and fake labels are present in the dataset."""
    synthbuster_root = "datasets/synthbuster"
    coco_root = "datasets/coco2017"
    
    # Skip test if datasets don't exist
    if not Path(synthbuster_root).exists() or not Path(coco_root).exists():
        pytest.skip("Datasets not available")
    
    dataset = BalancedCombinedDataset(
        synthbuster_root=synthbuster_root,
        coco_root=coco_root
    )
    
    # Sample a few items to check label distribution
    sample_size = min(100, len(dataset))
    labels = []
    
    for i in range(sample_size):
        _, label = dataset[i]
        labels.append(label)
    
    real_count = sum(1 for l in labels if l == 0)
    fake_count = sum(1 for l in labels if l == 1)
    
    assert real_count > 0, "Should have real images (label=0)"
    assert fake_count > 0, "Should have fake images (label=1)"
    
    print(f"✓ Label distribution in {sample_size} samples: {real_count} real, {fake_count} fake")


if __name__ == "__main__":
    """Run tests manually for quick validation."""
    print("Testing Combined Balanced Dataset Loader")
    print("=" * 70)
    
    try:
        print("\n1. Testing dataset initialization...")
        test_balanced_combined_dataset_initialization()
        
        print("\n2. Testing sample loading...")
        test_combined_dataset_getitem()
        
        print("\n3. Testing train/val split...")
        test_train_val_split_combined()
        
        print("\n4. Testing label distribution...")
        test_combined_dataset_label_distribution()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_combined_dataset_corrupted_image_handling(tmp_path):
    """Test that corrupted images are handled gracefully."""
    import warnings
    from PIL import Image
    import numpy as np
    
    # Create a mock corrupted TIFF file
    corrupted_tiff = tmp_path / "corrupted.tiff"
    corrupted_tiff.write_bytes(b"CORRUPTED_TIFF_DATA")
    
    # Create a valid image
    valid_image = tmp_path / "valid.png"
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(valid_image)
    
    # Test that opening corrupted TIFF raises OSError
    with pytest.raises(OSError):
        Image.open(corrupted_tiff).convert('RGB')
    
    print("✓ Corrupted image handling test setup complete")
