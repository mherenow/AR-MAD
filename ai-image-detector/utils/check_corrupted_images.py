"""
Utility script to check for corrupted images in the dataset.

This script scans through the dataset directories and identifies images
that cannot be loaded by PIL, helping to clean up the dataset before training.
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys


def check_image(image_path: Path) -> tuple[bool, str]:
    """
    Check if an image can be loaded successfully.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            img.convert('RGB')
            img.load()  # Force loading to catch decoder errors
        return True, ""
    except Exception as e:
        return False, str(e)


def scan_directory(directory: Path, extensions: list[str] = None) -> dict:
    """
    Scan a directory for corrupted images.
    
    Args:
        directory: Directory to scan
        extensions: List of file extensions to check (default: common image formats)
        
    Returns:
        Dictionary with scan results
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(directory.rglob(f'*{ext}'))
        image_files.extend(directory.rglob(f'*{ext.upper()}'))
    
    corrupted = []
    valid = 0
    
    print(f"Scanning {len(image_files)} images in {directory}...")
    
    for img_path in tqdm(image_files, desc="Checking images"):
        is_valid, error = check_image(img_path)
        if is_valid:
            valid += 1
        else:
            corrupted.append((img_path, error))
    
    return {
        'total': len(image_files),
        'valid': valid,
        'corrupted': corrupted
    }


def main():
    parser = argparse.ArgumentParser(
        description='Check for corrupted images in dataset directories'
    )
    parser.add_argument(
        'directories',
        nargs='+',
        type=str,
        help='Directories to scan for corrupted images'
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help='Remove corrupted images (use with caution!)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to save list of corrupted images'
    )
    
    args = parser.parse_args()
    
    all_results = {}
    
    for directory_str in args.directories:
        directory = Path(directory_str)
        
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist, skipping...")
            continue
        
        if not directory.is_dir():
            print(f"Warning: {directory} is not a directory, skipping...")
            continue
        
        results = scan_directory(directory)
        all_results[directory_str] = results
        
        print(f"\n{'='*60}")
        print(f"Results for {directory}:")
        print(f"  Total images: {results['total']}")
        print(f"  Valid images: {results['valid']}")
        print(f"  Corrupted images: {len(results['corrupted'])}")
        
        if results['corrupted']:
            print(f"\nCorrupted images:")
            for img_path, error in results['corrupted']:
                print(f"  - {img_path}")
                print(f"    Error: {error}")
        
        # Remove corrupted images if requested
        if args.remove and results['corrupted']:
            print(f"\nRemoving {len(results['corrupted'])} corrupted images...")
            for img_path, _ in results['corrupted']:
                try:
                    img_path.unlink()
                    print(f"  Removed: {img_path}")
                except Exception as e:
                    print(f"  Failed to remove {img_path}: {e}")
    
    # Save results to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write("Corrupted Images Report\n")
            f.write("=" * 60 + "\n\n")
            
            for directory, results in all_results.items():
                f.write(f"Directory: {directory}\n")
                f.write(f"Total: {results['total']}, Valid: {results['valid']}, "
                       f"Corrupted: {len(results['corrupted'])}\n\n")
                
                if results['corrupted']:
                    f.write("Corrupted files:\n")
                    for img_path, error in results['corrupted']:
                        f.write(f"  {img_path}\n")
                        f.write(f"    Error: {error}\n")
                f.write("\n")
        
        print(f"\nResults saved to {output_path}")
    
    # Exit with error code if corrupted images found
    total_corrupted = sum(len(r['corrupted']) for r in all_results.values())
    if total_corrupted > 0:
        print(f"\n⚠️  Found {total_corrupted} corrupted images across all directories")
        sys.exit(1)
    else:
        print(f"\n✓ All images are valid!")
        sys.exit(0)


if __name__ == '__main__':
    main()
