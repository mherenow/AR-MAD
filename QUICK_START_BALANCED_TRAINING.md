# Quick Start: Balanced Training Guide

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] PyTorch and dependencies installed
- [ ] SynthBuster dataset downloaded to `datasets/synthbuster/`
- [ ] COCO 2017 train set downloaded to `datasets/coco2017/train2017/`

## Step-by-Step Guide

### 1. Verify Dataset Setup (2 minutes)

```bash
# Run verification script
python verify_balanced_dataset.py
```

**Expected output:**
```
✓ SynthBuster found at: datasets/synthbuster
✓ COCO2017 found at: datasets/coco2017
✓ Dataset balanced: 9000 real, 9000 fake (ratio: 1.00)
✓ VERIFICATION SUCCESSFUL
```

### 2. Review Configuration (1 minute)

Check `ai-image-detector/configs/default_config.yaml`:

```yaml
dataset:
  mode: "combined"  # ✓ Using balanced mode
  synthbuster_root: "datasets/synthbuster"
  coco_root: "datasets/coco2017"
  balance_mode: "equal"  # ✓ 1:1 ratio
```

### 3. Start Training (30-60 minutes)

```bash
# Train with balanced dataset
python -m ai_image_detector.training --config ai-image-detector/configs/default_config.yaml
```

**What to expect:**
- Dataset loading: ~30 seconds
- Per epoch: ~3-5 minutes (GPU) or ~15-20 minutes (CPU)
- Total time: ~30-50 minutes for 10 epochs
- Checkpoints saved every epoch to `checkpoints/`

### 4. Monitor Training Progress

Watch for these metrics in the output:

```
Epoch [1/10] Summary:
  Train Loss: 0.3245 | Train Acc: 0.8567  ← Should decrease/increase
  Val Loss:   0.2891 | Val Acc:   0.8823  ← Should improve over time
  ⭐ New best model saved! (val_acc: 0.8823)
```

**Good signs:**
- Train loss decreasing
- Train/val accuracy increasing
- Val accuracy > 85% by epoch 5
- Best model being saved regularly

**Warning signs:**
- Val accuracy not improving after 5 epochs
- Large gap between train and val accuracy (overfitting)
- Loss increasing or oscillating wildly

### 5. Evaluate Trained Model (5 minutes)

```bash
# Evaluate best model
python -m ai_image_detector.evaluation \
    --config ai-image-detector/configs/default_config.yaml \
    --checkpoint checkpoints/best_model.pth
```

**Expected results:**
- Overall accuracy: 85-92%
- AUC: 0.90-0.96
- Per-generator breakdown showing performance on each AI model

## Quick Configuration Changes

### Faster Training (for testing)

Edit `default_config.yaml`:
```yaml
model:
  backbone_type: "simple_cnn"  # Faster than resnet18

training:
  batch_size: 64  # Larger batches
  num_epochs: 5   # Fewer epochs
```

### Higher Accuracy (for production)

Edit `default_config.yaml`:
```yaml
model:
  backbone_type: "resnet50"  # More powerful
  pretrained: true

training:
  batch_size: 16      # Smaller for larger model
  num_epochs: 20      # More training
  learning_rate: 0.0001  # Lower LR for fine-tuning
```

### Switch to SynthBuster-Only

Edit `default_config.yaml`:
```yaml
dataset:
  mode: "synthbuster"  # Change from "combined"
  synthbuster_root: "datasets/synthbuster"
```

## Common Issues & Quick Fixes

### Issue: "CUDA out of memory"
**Fix:** Reduce batch size
```yaml
training:
  batch_size: 16  # or 8
```

### Issue: "Dataset not found"
**Fix:** Check paths in config match your actual directories
```bash
ls datasets/synthbuster/
ls datasets/coco2017/train2017/
```

### Issue: Training too slow
**Fix:** Increase workers or use GPU
```yaml
dataset:
  num_workers: 8  # More parallel loading

device: "cuda"  # Use GPU if available
```

### Issue: Low accuracy (<80%)
**Fix:** Train longer or use pretrained model
```yaml
model:
  pretrained: true  # Use ImageNet weights

training:
  num_epochs: 15  # More training
```

## Expected Timeline

| Task | Time | Notes |
|------|------|-------|
| Dataset verification | 2 min | One-time setup |
| Training (10 epochs) | 30-50 min | GPU recommended |
| Evaluation | 5 min | Quick metrics |
| **Total** | **~40-60 min** | First run |

## Success Criteria

After training, you should see:

✓ **Training completed** without errors
✓ **Best model saved** to `checkpoints/best_model.pth`
✓ **Validation accuracy** > 85%
✓ **Balanced dataset** used (9,000 real + 9,000 fake)
✓ **Evaluation metrics** showing good performance across generators

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes
2. **Test different backbones**: Compare simple_cnn, resnet18, resnet50
3. **Analyze per-generator performance**: Identify which AI models are hardest to detect
4. **Add data augmentation**: Improve robustness (future enhancement)
5. **Deploy model**: Use for inference on new images

## Quick Commands Reference

```bash
# Verify setup
python verify_balanced_dataset.py

# Train model
python -m ai_image_detector.training --config ai-image-detector/configs/default_config.yaml

# Evaluate model
python -m ai_image_detector.evaluation --config ai-image-detector/configs/default_config.yaml --checkpoint checkpoints/best_model.pth

# Run tests
python -m pytest ai-image-detector/data/test_combined_loader.py -v

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Getting Help

If you encounter issues:
1. Check error messages carefully
2. Verify dataset paths and structure
3. Review configuration file syntax
4. Check `BALANCED_DATASET_SUMMARY.md` for detailed information
5. Review `ai-image-detector/README.md` for comprehensive documentation

## Summary

You now have a balanced training setup that:
- Uses 9,000 real images from COCO 2017
- Uses 9,000 fake images from SynthBuster
- Maintains perfect 1:1 balance for optimal classification
- Supports easy switching between combined and SynthBuster-only modes

**Ready to train!** 🚀
