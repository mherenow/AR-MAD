"""
Script to identify the specific image causing the decoder error.

This script simulates the DataLoader iteration to find which image
is causing the OSError: decoder error -2.
"""

import sys
from pathlib import Path
from PIL import Image
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from relative path
import importlib.util
spec = importlib.util.spec_from_file_location(
    "combined_loader",
    Path(__file__).parent.parent / "data" / "combined_loader.py"
)
combined_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combined_loader)
BalancedCombinedDataset = combined_loader.BalancedCombinedDataset


def test_all_images(dataset, max_samples=None):
    """
    Test loading all images in the dataset.
    
    Args:
        dataset: The dataset to test
        max_samples: Maximum number of samples to test (None for all)
    """
    problematic = []
    total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"Testing {total} images...")
    
    for idx in range(total):
        if idx % 100 == 0:
            print(f"Progress: {idx}/{total} ({idx*100//total}%)")
        
        try:
            source, source_idx, label = dataset.all_samples[idx]
            
            if source == 'coco':
                # COCO dataset handles its own loading
                _ = dataset.coco_dataset[source_idx]
            elif source == 'synthbuster_real':
                sample = dataset.synthbuster_real_samples[source_idx]
                img = Image.open(sample['path'])
                img.convert('RGB')
                img.load()  # Force load to catch decoder errors
                img.close()
            elif source == 'synthbuster_fake':
                sample = dataset.synthbuster_fake_samples[source_idx]
                img = Image.open(sample['path'])
                img.convert('RGB')
                img.load()  # Force load to catch decoder errors
                img.close()
                
        except Exception as e:
            error_info = {
                'idx': idx,
                'source': source,
                'source_idx': source_idx,
                'label': label,
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            if source in ['synthbuster_real', 'synthbuster_fake']:
                if source == 'synthbuster_real':
                    sample = dataset.synthbuster_real_samples[source_idx]
                else:
                    sample = dataset.synthbuster_fake_samples[source_idx]
                error_info['path'] = str(sample['path'])
            
            problematic.append(error_info)
            print(f"\n⚠️  Found problematic image at index {idx}:")
            print(f"   Source: {source}")
            print(f"   Path: {error_info.get('path', 'N/A')}")
            print(f"   Error: {error_info['error_type']}: {error_info['error']}")
    
    return problematic


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find problematic images in the dataset'
    )
    parser.add_argument(
        '--synthbuster-root',
        type=str,
        default='datasets/synthbuster',
        help='Path to SynthBuster dataset'
    )
    parser.add_argument(
        '--coco-root',
        type=str,
        default='datasets/coco2017',
        help='Path to COCO dataset'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to test'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to save problematic image list'
    )
    
    args = parser.parse_args()
    
    print("Loading dataset...")
    dataset = BalancedCombinedDataset(
        synthbuster_root=args.synthbuster_root,
        coco_root=args.coco_root
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print()
    
    problematic = test_all_images(dataset, args.max_samples)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total tested: {args.max_samples or len(dataset)}")
    print(f"  Problematic: {len(problematic)}")
    
    if problematic:
        print(f"\nProblematic images:")
        for info in problematic:
            print(f"\n  Index: {info['idx']}")
            print(f"  Source: {info['source']}")
            print(f"  Path: {info.get('path', 'N/A')}")
            print(f"  Error: {info['error_type']}: {info['error']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write("Problematic Images Report\n")
                f.write("=" * 60 + "\n\n")
                
                for info in problematic:
                    f.write(f"Index: {info['idx']}\n")
                    f.write(f"Source: {info['source']}\n")
                    f.write(f"Path: {info.get('path', 'N/A')}\n")
                    f.write(f"Error: {info['error_type']}: {info['error']}\n")
                    f.write("\n")
            
            print(f"\nReport saved to {args.output}")
    else:
        print("\n✓ All images loaded successfully!")


if __name__ == '__main__':
    main()
