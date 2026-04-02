# Corrupted Image Checker

This utility helps identify and optionally remove corrupted images from your dataset before training.

## Problem

During training, corrupted images (especially TIFF files with decoder errors) can cause the DataLoader to crash with errors like:
```
OSError: decoder error -2
```

## Solution

The training code now includes error handling that automatically skips corrupted images, but it's better to identify and fix them beforehand.

## Usage

### Check for corrupted images:

```bash
python -m ai-image-detector.utils.check_corrupted_images datasets/synthbuster datasets/coco2017
```

### Save report to file:

```bash
python -m ai-image-detector.utils.check_corrupted_images datasets/synthbuster --output corrupted_report.txt
```

### Remove corrupted images (use with caution!):

```bash
python -m ai-image-detector.utils.check_corrupted_images datasets/synthbuster --remove
```

## What it does

1. Scans all image files in the specified directories
2. Attempts to load each image with PIL
3. Reports any images that fail to load
4. Optionally removes corrupted images
5. Saves a detailed report

## Error Handling in Training

Even with this utility, the training code includes robust error handling:

- `combined_loader.py`: Automatically skips corrupted images and loads a different sample
- `synthbuster_loader.py`: Already has error handling built-in
- `coco_loader.py`: Already has error handling built-in

This ensures training continues even if a few corrupted images slip through.
