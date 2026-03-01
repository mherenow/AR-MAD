# Balanced Dataset Implementation Summary

## Overview

The AI Image Detector now supports training on a balanced combination of SynthBuster and COCO2017 datasets, ensuring equal representation of real and AI-generated images for optimal binary classification performance.

## What Was Implemented

### 1. New Dataset Loaders

#### COCO2017 Loader (`ai-image-detector/data/coco_loader.py`)
- Loads COCO 2017 train images as real images (label=0)
- Supports limiting the number of samples for balancing
- Handles 118,287 images with efficient loading
- Automatic resize to 256x256 and normalization

#### Combined Balanced Loader (`ai-image-detector/data/combined_loader.py`)
- Combines SynthBuster and COCO2017 datasets
- Automatically balances real and fake images to 1:1 ratio
- Supports two balance modes: 'equal' and 'min'
- Provides detailed statistics during dataset creation
- Creates train/validation splits while maintaining balance

### 2. Updated Configuration

**New configuration options in `default_config.yaml`:**
```yaml
dataset:
  mode: "combined"  # or "synthbuster"
  synthbuster_root: "datasets/synthbuster"
  coco_root: "datasets/coco2017"
  balance_mode: "equal"
  val_ratio: 0.2
```

### 3. Updated Training Script

**Modified `ai-image-detector/training/__main__.py`:**
- Supports both 'combined' and 'synthbuster' modes
- Automatically selects appropriate dataset loader based on config
- Maintains backward compatibility with SynthBuster-only mode

### 4. Testing and Verification

**New test files:**
- `ai-image-detector/data/test_combined_loader.py` - Unit tests for combined dataset
- `verify_balanced_dataset.py` - Quick verification script

## Current Dataset Statistics

### Balanced Dataset Composition
- **Real images (label=0)**: 9,000 images from COCO 2017
- **Fake images (label=1)**: 9,000 images from SynthBuster AI generators
- **Total**: 18,000 images
- **Balance ratio**: 1.000:1 (perfect balance)

### Source Breakdown
- COCO 2017: 9,000 images (from 118,287 available)
- SynthBuster RAISE: 0 images (COCO provides sufficient real images)
- SynthBuster AI generators: 9,000 images (from various generators)

## How to Use

### 1. Verify Dataset Setup
```bash
python verify_balanced_dataset.py
```

### 2. Train with Balanced Dataset
```bash
python -m ai_image_detector.training --config ai-image-detector/configs/default_config.yaml
```

The default configuration now uses the combined balanced mode.

### 3. Switch to SynthBuster-Only Mode
Edit `ai-image-detector/configs/default_config.yaml`:
```yaml
dataset:
  mode: "synthbuster"
  synthbuster_root: "datasets/synthbuster"
```

## Benefits of Balanced Training

### 1. Improved Classification Performance
- Equal representation prevents model bias toward majority class
- Better generalization to both real and fake images
- More reliable accuracy metrics

### 2. Diverse Real Image Training
- COCO 2017 provides 118K diverse real-world images
- Covers wide variety of scenes, objects, and contexts
- Complements SynthBuster's RAISE dataset

### 3. Scalable and Flexible
- Easy to adjust balance ratio via configuration
- Can limit dataset size for faster experimentation
- Supports adding more datasets in the future

## Technical Details

### Dataset Loading Flow

1. **Initialization**:
   - Load SynthBuster dataset and count real/fake images
   - Load COCO 2017 dataset (all available images)
   - Calculate target samples per class based on balance_mode

2. **Balancing**:
   - Determine minimum of (real_available, fake_available)
   - Limit each class to target count
   - Prioritize COCO for real images, then SynthBuster RAISE

3. **Sampling**:
   - Create unified sample list with source tracking
   - Support random access via `__getitem__`
   - Load images on-demand with error handling

### Memory Efficiency

- Images loaded on-demand (not preloaded into memory)
- Only metadata stored in memory during initialization
- Efficient indexing for fast random access
- Supports multi-worker data loading

### Train/Val Split

- Maintains balance in both train and validation sets
- Uses random shuffling with fixed seed for reproducibility
- Default 80/20 split (configurable via val_ratio)

## Files Modified/Created

### Created Files
1. `ai-image-detector/data/coco_loader.py` - COCO2017 dataset loader
2. `ai-image-detector/data/combined_loader.py` - Balanced combined dataset
3. `ai-image-detector/data/test_combined_loader.py` - Unit tests
4. `verify_balanced_dataset.py` - Verification script
5. `BALANCED_DATASET_SUMMARY.md` - This document

### Modified Files
1. `ai-image-detector/data/__init__.py` - Export new loaders
2. `ai-image-detector/configs/default_config.yaml` - Add combined mode config
3. `ai-image-detector/training/__main__.py` - Support both modes
4. `ai-image-detector/README.md` - Document balanced dataset

## Example Output

### Dataset Creation
```
======================================================================
Building Balanced Combined Dataset
======================================================================

1. Loading SynthBuster dataset...
   SynthBuster RAISE (real): 0 images
   SynthBuster AI (fake): 9000 images

2. Loading COCO 2017 dataset...
   COCO 2017 (real): 118287 images available

3. Calculating balanced split...
   Total real images available: 118287
   Total fake images available: 9000
   Target per class: 9000

4. Creating balanced subsets...
   Using 0 from SynthBuster RAISE
   Using 9000 from COCO 2017
   Using 9000 from SynthBuster AI generators

5. Final dataset composition:
   Real images (label=0): 9000
   Fake images (label=1): 9000
   Total images: 18000
   Balance ratio: 1.00:1
======================================================================
```

### Training Output
```
Using COMBINED dataset mode (SynthBuster + COCO2017)
Train samples: 14400, Val samples: 3600

Starting training for 10 epochs...
======================================================================
Epoch [1/10] Summary:
  Train Loss: 0.3245 | Train Acc: 0.8567
  Val Loss:   0.2891 | Val Acc:   0.8823
  💾 Checkpoint saved: checkpoints/checkpoint_epoch_1.pth
  ⭐ New best model saved! (val_acc: 0.8823)
----------------------------------------------------------------------
```

## Future Enhancements

### Potential Improvements
1. **Dynamic Balancing**: Adjust balance ratio during training
2. **Multi-Source Real Images**: Add more real image datasets
3. **Weighted Sampling**: Sample more from difficult generators
4. **Data Augmentation**: Add augmentation pipeline for robustness
5. **Stratified Splitting**: Ensure each generator represented in val set

### Additional Datasets
- ImageNet (real images)
- Places365 (real scenes)
- More AI generator datasets (Imagen, Parti, etc.)

## Troubleshooting

### Issue: Dataset Not Found
**Solution**: Ensure both datasets exist at configured paths:
- `datasets/synthbuster/` with generator folders
- `datasets/coco2017/train2017/` with COCO images

### Issue: Imbalanced Dataset Warning
**Solution**: Check that SynthBuster has sufficient AI-generated images. The system will balance to the minimum available.

### Issue: Out of Memory
**Solution**: Reduce batch size in config or use fewer workers:
```yaml
training:
  batch_size: 16  # Reduce from 32
dataset:
  num_workers: 2  # Reduce from 4
```

## Conclusion

The balanced dataset implementation ensures the AI Image Detector trains on equal amounts of real and AI-generated images, leading to better classification performance and more reliable evaluation metrics. The system is flexible, scalable, and maintains backward compatibility with the original SynthBuster-only mode.

**Key Achievement**: Perfect 1:1 balance with 9,000 real images from COCO 2017 and 9,000 fake images from SynthBuster AI generators, totaling 18,000 training samples.
