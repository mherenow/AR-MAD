"""
Verify that the dataset is properly formatted and ready for training.

This script checks:
- RAISE images are in JPEG format
- No corrupted images exist
- Dataset structure is correct
"""

from pathlib import Path
from collections import Counter
from PIL import Image
import sys


def verify_raise_format(raise_dir: Path) -> tuple[bool, str]:
    """Verify RAISE images are in JPEG format."""
    if not raise_dir.exists():
        return False, f"RAISE directory not found: {raise_dir}"
    
    extensions = Counter()
    for file in raise_dir.iterdir():
        if file.is_file():
            extensions[file.suffix.lower()] += 1
    
    jpeg_count = extensions.get('.jpg', 0) + extensions.get('.jpeg', 0)
    tiff_count = extensions.get('.tif', 0) + extensions.get('.tiff', 0)
    
    if tiff_count > 0:
        return False, f"Found {tiff_count} TIFF files. RAISE images should be JPEG."
    
    if jpeg_count == 0:
        return False, "No JPEG files found in RAISE directory."
    
    return True, f"Found {jpeg_count} JPEG files (correct format)"


def verify_images_loadable(directory: Path, max_check: int = 100) -> tuple[bool, str]:
    """Verify that images can be loaded without errors."""
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(directory.rglob(f'*{ext}'))
    
    if not image_files:
        return True, "No images to check"
    
    # Check up to max_check images
    to_check = min(len(image_files), max_check)
    corrupted = []
    
    for i, img_path in enumerate(image_files[:to_check]):
        try:
            with Image.open(img_path) as img:
                img.load()  # Force load to catch decoder errors
        except Exception as e:
            corrupted.append((img_path, str(e)))
    
    if corrupted:
        msg = f"Found {len(corrupted)} corrupted images:\n"
        for path, error in corrupted[:5]:  # Show first 5
            msg += f"  - {path}: {error}\n"
        return False, msg
    
    return True, f"Checked {to_check} images - all valid"


def main():
    print("Dataset Verification")
    print("=" * 60)
    
    synthbuster_root = Path("datasets/synthbuster")
    raise_dir = synthbuster_root / "RAISE"
    
    all_passed = True
    
    # Check 1: RAISE format
    print("\n1. Checking RAISE image format...")
    passed, message = verify_raise_format(raise_dir)
    print(f"   {'✓' if passed else '✗'} {message}")
    all_passed = all_passed and passed
    
    # Check 2: RAISE images loadable
    if raise_dir.exists():
        print("\n2. Checking RAISE images are loadable...")
        passed, message = verify_images_loadable(raise_dir, max_check=100)
        print(f"   {'✓' if passed else '✗'} {message}")
        all_passed = all_passed and passed
    
    # Check 3: AI generator directories
    print("\n3. Checking AI generator directories...")
    if synthbuster_root.exists():
        generators = [d for d in synthbuster_root.iterdir() 
                     if d.is_dir() and d.name != "RAISE"]
        if generators:
            print(f"   ✓ Found {len(generators)} AI generator directories:")
            for gen in generators[:5]:  # Show first 5
                count = len(list(gen.glob('*.jpg'))) + len(list(gen.glob('*.png')))
                print(f"     - {gen.name}: {count} images")
            if len(generators) > 5:
                print(f"     ... and {len(generators) - 5} more")
        else:
            print("   ✗ No AI generator directories found")
            all_passed = False
    else:
        print(f"   ✗ SynthBuster directory not found: {synthbuster_root}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ Dataset verification passed!")
        print("  Your dataset is ready for training.")
        sys.exit(0)
    else:
        print("✗ Dataset verification failed!")
        print("  Please fix the issues above before training.")
        sys.exit(1)


if __name__ == '__main__':
    main()
