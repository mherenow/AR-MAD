# Corrupted Image Fix

## Problem
Training was crashing with:
```
OSError: decoder error -2
```

This error occurred when PIL encountered a corrupted TIFF file in the RAISE dataset.

## Root Cause
The RAISE dataset should contain JPEG files (converted from TIFF during download), but one corrupted TIFF file (`r0bf7f938t.tif`) remained in the dataset. This file had a decoder error and couldn't be loaded.

## Solution Applied

### 1. Identified and Removed Corrupted File
- Found 1 corrupted TIFF file among 999 JPEG files in the RAISE directory
- The file `r0bf7f938t.tif` had decoder error -2 and couldn't be converted
- Removed the corrupted file
- Dataset now has 999 valid RAISE images (all in JPEG format)

### 2. Added Robust Error Handling to `combined_loader.py`
The `__getitem__` method now catches `OSError` and `IOError` exceptions when loading images:
- Logs a detailed warning with the corrupted image path and source
- Automatically selects a different random sample
- Training continues without interruption

The warning will show:
```
Skipping corrupted image at index X (source: synthbuster_fake, path: /path/to/image.jpg): decoder error -2
```

### 3. Updated `synthbuster_loader.py`
- Removed `.tiff` and `.tif` from supported extensions
- Now expects RAISE images in JPEG format (`.jpg`, `.jpeg`)
- Added documentation clarifying expected formats

### 4. Updated `download_raise_images.py`
- Clarified that JPEG is the recommended and default format
- TIFF files are converted to JPEG during download to save space
- Added better documentation about format expectations

### 5. Created Utility Script
`ai-image-detector/utils/check_corrupted_images.py` helps identify corrupted images:
- Scans dataset directories
- Tests each image with PIL
- Reports corrupted files
- Optionally removes them

### 6. Verified Existing Error Handling
Confirmed that `synthbuster_loader.py` and `coco_loader.py` already have similar error handling.

## Expected File Formats

- RAISE images: `.jpg` or `.jpeg` (converted from TIFF during download)
- AI-generated images: `.jpg`, `.jpeg`, `.png`, or `.bmp`
- COCO images: `.jpg` (standard COCO format)

## Usage

### Run training (will now work without errors):
```bash
python -m ai-image-detector.training
```

### Check for corrupted images beforehand:
```bash
python -m ai-image-detector.utils.check_corrupted_images datasets/synthbuster
```

### Remove corrupted images:
```bash
python -m ai-image-detector.utils.check_corrupted_images datasets/synthbuster --remove
```

## Testing
Added test case `test_combined_dataset_corrupted_image_handling` to verify the error handling works correctly.

## Result
- Corrupted TIFF file removed
- All RAISE images now in JPEG format (999 images)
- Training will continue even when encountering corrupted images
- Exact file paths logged for manual inspection
