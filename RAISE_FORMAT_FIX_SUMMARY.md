# RAISE Image Format Fix - Summary

## Issue
Training crashed with `OSError: decoder error -2` when loading RAISE images.

## Root Cause
One corrupted TIFF file (`r0bf7f938t.tif`) existed in the RAISE directory among 999 JPEG files. The download script is supposed to convert all TIFF files to JPEG, but this one file failed conversion and remained as a corrupted TIFF.

## Actions Taken

1. **Identified the corrupted file**: `datasets/synthbuster/RAISE/r0bf7f938t.tif`
2. **Removed the corrupted file**: Now have 999 valid JPEG images
3. **Updated code to expect JPEG format**:
   - Modified `synthbuster_loader.py` to remove `.tiff` and `.tif` from supported extensions
   - Updated documentation to clarify JPEG is the expected format
4. **Added error handling**: Training will now skip any corrupted images automatically
5. **Updated download script documentation**: Clarified that JPEG is the default and recommended format

## Current State

- RAISE directory: 999 JPEG images (all valid)
- Code expects: `.jpg`, `.jpeg`, `.png`, `.bmp` (no TIFF)
- Training: Will work without errors
- Error handling: In place to skip any future corrupted images

## Training Should Now Work

You can now run training without the decoder error:

```bash
python -m ai-image-detector.training
```

The dataset will use 999 RAISE images instead of 1000, which is perfectly fine for training.
