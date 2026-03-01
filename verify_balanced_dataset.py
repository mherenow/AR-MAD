"""
Quick verification script for balanced dataset configuration.

This script checks that both SynthBuster and COCO2017 datasets are available
and shows the balanced dataset statistics.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "ai-image-detector"))

from data.combined_loader import BalancedCombinedDataset


def main():
    """Verify balanced dataset setup."""
    print("\n" + "="*70)
    print("BALANCED DATASET VERIFICATION")
    print("="*70)
    
    # Check if datasets exist
    synthbuster_root = "datasets/synthbuster"
    coco_root = "datasets/coco2017"
    
    print("\n1. Checking dataset availability...")
    
    synthbuster_path = Path(synthbuster_root)
    coco_path = Path(coco_root)
    
    if not synthbuster_path.exists():
        print(f"   ✗ SynthBuster not found at: {synthbuster_root}")
        print(f"     Please ensure the dataset is downloaded and extracted.")
        return False
    else:
        print(f"   ✓ SynthBuster found at: {synthbuster_root}")
    
    if not coco_path.exists():
        print(f"   ✗ COCO2017 not found at: {coco_root}")
        print(f"     Please ensure the dataset is downloaded and extracted.")
        return False
    else:
        print(f"   ✓ COCO2017 found at: {coco_root}")
    
    # Create balanced dataset
    print("\n2. Creating balanced combined dataset...")
    try:
        dataset = BalancedCombinedDataset(
            synthbuster_root=synthbuster_root,
            coco_root=coco_root,
            balance_mode='equal'
        )
        
        print(f"\n3. Dataset statistics:")
        print(f"   Total samples: {len(dataset)}")
        
        # Count labels
        real_count = sum(1 for _, _, label in dataset.all_samples if label == 0)
        fake_count = sum(1 for _, _, label in dataset.all_samples if label == 1)
        
        print(f"   Real images (label=0): {real_count}")
        print(f"   Fake images (label=1): {fake_count}")
        print(f"   Balance ratio: {real_count/fake_count:.3f}:1")
        
        # Test loading a sample
        print("\n4. Testing sample loading...")
        image, label = dataset[0]
        print(f"   ✓ Sample loaded successfully")
        print(f"     Shape: {image.shape}")
        print(f"     Label: {label} ({'real' if label == 0 else 'fake'})")
        print(f"     Value range: [{image.min():.3f}, {image.max():.3f}]")
        
        print("\n" + "="*70)
        print("✓ VERIFICATION SUCCESSFUL")
        print("="*70)
        print("\nYou can now train the model with:")
        print("  python -m ai_image_detector.training --config ai-image-detector/configs/default_config.yaml")
        print("\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
