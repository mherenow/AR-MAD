"""
Download RAISE-1k images from the official server.

By default, downloads TIFF files and converts them to JPEG to save disk space
(~20MB TIFF -> ~2MB JPEG). The JPEG format is recommended for training as it:
- Saves significant disk space
- Loads faster during training
- Is the expected format for the RAISE dataset in this project

Usage:
    # Download as JPEG (recommended, default):
    python download_raise_images.py --csv RAISE_1k.csv --output datasets/synthbuster/RAISE
    
    # Download as TIFF (if you need lossless format):
    python download_raise_images.py --csv RAISE_1k.csv --output datasets/synthbuster/RAISE --tiff
"""

import csv
import os
import sys
import time
import argparse
import urllib.request
from pathlib import Path


def download_raise(csv_path: str, output_dir: str, keep_tiff: bool = False,
                   max_images: int = None, delay: float = 0.5):
    """
    Download RAISE-1k images from CSV.

    Args:
        csv_path:    Path to RAISE_1k.csv
        output_dir:  Destination folder (will be created)
        keep_tiff:   If True, keep raw TIFF. If False, convert to JPEG and delete TIFF.
        max_images:  Download only first N images (None = all 1000)
        delay:       Seconds between requests to avoid hammering the server
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read TIFF URLs from column index 2 (0=File, 1=NEF, 2=TIFF)
    urls = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row
        for row in reader:
            if len(row) >= 3:
                tiff_url = row[2].strip()
                file_id = row[0].strip()
                if tiff_url.startswith('http'):
                    urls.append((file_id, tiff_url))

    if max_images:
        urls = urls[:max_images]

    total = len(urls)
    print(f"Found {total} images to download")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Format: {'TIFF (raw)' if keep_tiff else 'JPEG (converted)'}")
    print("=" * 60)

    success = 0
    failed = []

    for i, (file_id, tiff_url) in enumerate(urls, 1):
        # Determine output filename
        if keep_tiff:
            out_file = output_path / f"{file_id}.tif"
        else:
            out_file = output_path / f"{file_id}.jpg"

        # Skip if already downloaded
        if out_file.exists():
            print(f"[{i}/{total}] SKIP {file_id} (already exists)")
            success += 1
            continue

        print(f"[{i}/{total}] Downloading {file_id}...", end=' ', flush=True)

        try:
            if keep_tiff:
                # Download TIFF directly
                urllib.request.urlretrieve(tiff_url, out_file)
                size_mb = out_file.stat().st_size / (1024 * 1024)
                print(f"✓ ({size_mb:.1f} MB)")
            else:
                # Download TIFF to temp file, convert to JPEG, delete TIFF
                tiff_temp = output_path / f"{file_id}_temp.tif"
                urllib.request.urlretrieve(tiff_url, tiff_temp)

                # Convert TIFF -> JPEG using Pillow
                try:
                    from PIL import Image
                    with Image.open(tiff_temp) as img:
                        # Convert to RGB (TIFF may be 16-bit or have alpha)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(out_file, 'JPEG', quality=95)
                    tiff_temp.unlink()  # delete temp TIFF
                    size_mb = out_file.stat().st_size / (1024 * 1024)
                    print(f"✓ ({size_mb:.1f} MB JPEG)")
                except Exception as conv_err:
                    # If conversion fails, keep the TIFF
                    tiff_fallback = output_path / f"{file_id}.tif"
                    tiff_temp.rename(tiff_fallback)
                    print(f"✓ TIFF kept (conversion failed: {conv_err})")

            success += 1

        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed.append((file_id, str(e)))
            # Clean up partial downloads
            for f in [out_file, output_path / f"{file_id}_temp.tif"]:
                if f.exists():
                    f.unlink()

        # Polite delay between requests
        if i < total:
            time.sleep(delay)

    print("\n" + "=" * 60)
    print(f"Downloaded: {success}/{total}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for fid, err in failed:
            print(f"  {fid}: {err}")
    else:
        print("All downloads successful!")

    print(f"\nImages saved to: {output_path.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download RAISE-1k dataset')
    parser.add_argument('--csv', required=True,
                        help='Path to RAISE_1k.csv')
    parser.add_argument('--output', default='datasets/synthbuster/RAISE',
                        help='Output directory (default: datasets/synthbuster/RAISE)')
    parser.add_argument('--tiff', action='store_true',
                        help='Keep raw TIFF files instead of converting to JPEG')
    parser.add_argument('--limit', type=int, default=None,
                        help='Download only first N images (default: all 1000)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Seconds between requests (default: 0.5)')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    download_raise(
        csv_path=args.csv,
        output_dir=args.output,
        keep_tiff=args.tiff,
        max_images=args.limit,
        delay=args.delay
    )