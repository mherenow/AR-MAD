#!/usr/bin/env python
"""
Show dataset statistics including label counts for training data.

Usage:
    python show_dataset_stats.py --config configs/default_config.yaml
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Subset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from data.synthbuster_loader import SynthBusterDataset, create_train_val_split
from data.combined_loader import BalancedCombinedDataset, create_train_val_split_combined


def count_labels(dataset):
    """Count labels in a dataset."""
    label_counts = Counter()
    
    print("Counting labels...")
    for i in range(len(dataset)):
        try:
            # Handle both (image, label) and (image, label, generator) formats
            sample = dataset[i]
            if len(sample) == 3:
                _, label, _ = sample
            else:
                _, label = sample
            
            # Convert tensor to int if needed
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            label_counts[int(label)] += 1
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")
        except Exception as e:
            print(f"  Warning: Error processing sample {i}: {e}")
            continue
    
    return label_counts


def show_dataset_statistics(config):
    """Show dataset statistics including label counts."""
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    # Initialize dataset based on mode
    dataset_mode = config['dataset'].get('mode', 'synthbuster')
    
    print(f"\nDataset Mode: {dataset_mode.upper()}")
    print("-" * 70)
    
    if dataset_mode == 'combined':
        # Use combined balanced dataset (SynthBuster + COCO2017)
        print("\nLoading COMBINED dataset (SynthBuster + COCO2017)...")
        synthbuster_root = config['dataset']['synthbuster_root']
        coco_root = config['dataset']['coco_root']
        val_ratio = config['dataset'].get('val_ratio', 0.2)
        
        # Create train/val split
        train_dataset, val_dataset = create_train_val_split_combined(
            synthbuster_root=synthbuster_root,
            coco_root=coco_root,
            val_ratio=val_ratio,
            seed=42
        )
        
    else:
        # Use SynthBuster only (original mode)
        print("\nLoading SYNTHBUSTER-ONLY dataset...")
        root_dir = config['dataset'].get('root_dir') or config['dataset'].get('synthbuster_root')
        
        # Create train/val split
        val_ratio = config['dataset'].get('val_ratio', 0.2)
        train_paths, val_paths = create_train_val_split(root_dir, val_ratio=val_ratio)
        
        # Create full dataset
        full_dataset = SynthBusterDataset(root_dir=root_dir)
        
        # Create train/val subsets
        path_to_idx = {sample['path']: idx for idx, sample in enumerate(full_dataset.samples)}
        train_indices = [path_to_idx[path] for path in train_paths if path in path_to_idx]
        val_indices = [path_to_idx[path] for path in val_paths if path in path_to_idx]
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    
    # Show basic statistics
    print(f"\nTotal Samples:")
    print(f"  Training:   {len(train_dataset):,}")
    print(f"  Validation: {len(val_dataset):,}")
    print(f"  Total:      {len(train_dataset) + len(val_dataset):,}")
    
    # Count labels in training set
    print("\n" + "=" * 70)
    print("TRAINING SET LABEL DISTRIBUTION")
    print("=" * 70)
    train_label_counts = count_labels(train_dataset)
    
    total_train = sum(train_label_counts.values())
    print(f"\nLabel Counts (Training):")
    for label in sorted(train_label_counts.keys()):
        count = train_label_counts[label]
        percentage = (count / total_train * 100) if total_train > 0 else 0
        label_name = "REAL" if label == 0 else "FAKE (AI-Generated)"
        print(f"  Label {label} ({label_name:20s}): {count:,} samples ({percentage:.2f}%)")
    
    print(f"\nTotal Training Samples: {total_train:,}")
    
    # Calculate balance ratio
    if len(train_label_counts) == 2:
        real_count = train_label_counts.get(0, 0)
        fake_count = train_label_counts.get(1, 0)
        if real_count > 0 and fake_count > 0:
            ratio = fake_count / real_count
            print(f"Balance Ratio (Fake:Real): {ratio:.3f}:1")
            
            if abs(ratio - 1.0) < 0.1:
                print("✓ Dataset is well-balanced")
            elif ratio < 0.5 or ratio > 2.0:
                print("⚠ Dataset is significantly imbalanced")
            else:
                print("⚠ Dataset has moderate imbalance")
    
    # Count labels in validation set
    print("\n" + "=" * 70)
    print("VALIDATION SET LABEL DISTRIBUTION")
    print("=" * 70)
    val_label_counts = count_labels(val_dataset)
    
    total_val = sum(val_label_counts.values())
    print(f"\nLabel Counts (Validation):")
    for label in sorted(val_label_counts.keys()):
        count = val_label_counts[label]
        percentage = (count / total_val * 100) if total_val > 0 else 0
        label_name = "REAL" if label == 0 else "FAKE (AI-Generated)"
        print(f"  Label {label} ({label_name:20s}): {count:,} samples ({percentage:.2f}%)")
    
    print(f"\nTotal Validation Samples: {total_val:,}")
    
    # Calculate balance ratio
    if len(val_label_counts) == 2:
        real_count = val_label_counts.get(0, 0)
        fake_count = val_label_counts.get(1, 0)
        if real_count > 0 and fake_count > 0:
            ratio = fake_count / real_count
            print(f"Balance Ratio (Fake:Real): {ratio:.3f}:1")
            
            if abs(ratio - 1.0) < 0.1:
                print("✓ Dataset is well-balanced")
            elif ratio < 0.5 or ratio > 2.0:
                print("⚠ Dataset has moderate imbalance")
            else:
                print("⚠ Dataset has slight imbalance")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    total_samples = total_train + total_val
    overall_label_counts = Counter()
    for label, count in train_label_counts.items():
        overall_label_counts[label] += count
    for label, count in val_label_counts.items():
        overall_label_counts[label] += count
    
    print(f"\nOverall Label Distribution:")
    for label in sorted(overall_label_counts.keys()):
        count = overall_label_counts[label]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        label_name = "REAL" if label == 0 else "FAKE (AI-Generated)"
        print(f"  Label {label} ({label_name:20s}): {count:,} samples ({percentage:.2f}%)")
    
    print(f"\nTotal Samples: {total_samples:,}")
    
    # Show configuration details
    print("\n" + "=" * 70)
    print("CONFIGURATION DETAILS")
    print("=" * 70)
    print(f"\nDataset Configuration:")
    print(f"  Mode: {dataset_mode}")
    print(f"  Image Size: {config['dataset'].get('image_size', 'N/A')}")
    print(f"  Validation Ratio: {config['dataset'].get('val_ratio', 0.2)}")
    print(f"  Num Workers: {config['dataset'].get('num_workers', 4)}")
    
    if dataset_mode == 'combined':
        print(f"\nDataset Paths:")
        print(f"  SynthBuster: {config['dataset']['synthbuster_root']}")
        print(f"  COCO2017: {config['dataset']['coco_root']}")
    else:
        root_dir = config['dataset'].get('root_dir') or config['dataset'].get('synthbuster_root')
        print(f"\nDataset Path:")
        print(f"  SynthBuster: {root_dir}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Show dataset statistics for AI Image Detector training'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Show statistics
    show_dataset_statistics(config)


if __name__ == '__main__':
    main()
