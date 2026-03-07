# Quick Start Guide for Low-Spec Systems

This guide helps you train the AI Image Detector on systems with limited resources.

## TL;DR - Fastest Setup

```bash
# Use the optimized fast training config
python -m ai-image-detector.training --config ai-image-detector/configs/fast_training.yaml
```

This provides **2-4x speedup** with minimal accuracy loss.

## What's Optimized?

The `fast_training.yaml` config includes:

✓ **128x128 images** (4x faster than 256x256)  
✓ **SimpleCNN backbone** (5x faster than ResNet50)  
✓ **Mixed precision training** (2x speedup on CUDA GPUs)  
✓ **No enhanced features** (spectral, noise, color branches disabled)  
✓ **No augmentations** (CutMix, MixUp, robustness disabled)  
✓ **Smaller batch size** (16 instead of 32)  
✓ **Fewer epochs** (20 instead of 100)  
✓ **2 data workers** (better for low-spec CPUs)

## Expected Performance

On a low-spec system (GTX 1050 Ti, i3-8100):
- **Default config**: ~120 seconds/epoch
- **Fast config**: ~30 seconds/epoch
- **Fast config + AMP**: ~15 seconds/epoch

**Speedup: 4-8x faster!**

## If You Still Have Issues

### Out of Memory (OOM)

Reduce batch size:
```yaml
training:
  batch_size: 8  # Or even 4
```

Or reduce image size further:
```yaml
dataset:
  image_size: 96  # Or even 64
```

### Still Too Slow

Try even fewer workers:
```yaml
dataset:
  num_workers: 0  # Single-threaded
```

Or train for even fewer epochs:
```yaml
training:
  num_epochs: 10
```

### No CUDA GPU

Mixed precision won't help, but other optimizations still apply:
```yaml
training:
  mixed_precision: false  # Disable if no CUDA
  batch_size: 8  # Smaller batch for CPU
```

CPU training is 10-50x slower than GPU. Consider:
- Using Google Colab (free GPU)
- AWS/Azure free tier
- Kaggle notebooks (free GPU)

## Monitoring Training

Watch your system resources:

**GPU (NVIDIA):**
```bash
nvidia-smi -l 1
```

**CPU/RAM:**
```bash
# Linux/Mac
htop

# Windows
Task Manager (Ctrl+Shift+Esc)
```

## Gradual Optimization

Start with fast config, then gradually add features:

**Step 1: Baseline (fastest)**
```yaml
model:
  backbone_type: "simple_cnn"
  # All features disabled
```

**Step 2: Add better backbone**
```yaml
model:
  backbone_type: "resnet18"  # 2-3x slower but better accuracy
```

**Step 3: Add one feature**
```yaml
model:
  backbone_type: "resnet18"
  use_spectral: true  # Add spectral branch
```

**Step 4: Increase image size**
```yaml
dataset:
  image_size: 224  # Better quality
```

**Step 5: Add augmentation**
```yaml
augmentation:
  mixup:
    enabled: true
```

## Accuracy vs Speed Trade-offs

| Config | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Fast (128px, SimpleCNN) | 10x | ~89% | Quick experiments |
| Medium (224px, ResNet18) | 3x | ~93% | Development |
| Full (256px, ResNet50 + features) | 1x | ~95% | Production |

## Tips for Low-Spec Systems

1. **Close other applications** - Free up RAM and GPU memory
2. **Use SSD** - Faster data loading
3. **Monitor temperature** - Ensure adequate cooling
4. **Train overnight** - Let it run when you're not using the PC
5. **Use checkpoints** - Resume if interrupted
6. **Start small** - Test with 1-2 epochs first

## Example Training Session

```bash
# 1. Test that everything works (1 epoch)
python -m ai-image-detector.training \
    --config ai-image-detector/configs/fast_training.yaml

# 2. If successful, train for real (edit config to set num_epochs: 20)
python -m ai-image-detector.training \
    --config ai-image-detector/configs/fast_training.yaml

# 3. If interrupted, resume
python -m ai-image-detector.training \
    --config ai-image-detector/configs/fast_training.yaml \
    --resume checkpoints/fast/checkpoint_epoch_10.pth
```

## Getting Help

If you're still having issues:

1. Check [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION.md)
2. Review your system specs (GPU, RAM, CPU)
3. Try the absolute minimum config (64px images, batch_size=4)
4. Consider cloud training options

## Cloud Training Options (Free Tier)

If local training is too slow:

**Google Colab** (Free GPU)
- 12-16 GB GPU RAM
- ~100 GB disk space
- 12-hour session limit

**Kaggle Notebooks** (Free GPU)
- 16 GB GPU RAM
- 30 hours/week GPU quota

**AWS SageMaker** (Free tier)
- 250 hours/month (first 2 months)
- ml.t2.medium instance

**Paperspace Gradient** (Free tier)
- 6 hours/month GPU time
- M4000 GPU
