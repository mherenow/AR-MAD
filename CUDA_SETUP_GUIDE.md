# CUDA Setup Guide for AI Image Detector

## Current Situation

**Your Hardware:**
- GPU: NVIDIA GeForce GTX 1650 Ti (4GB VRAM)
- NVIDIA Driver: Version 581.57
- CUDA Version: 13.0 (supported by driver)

**Your Software:**
- Python: 3.14.3
- PyTorch: 2.10.0+cpu (CPU-only version)
- Issue: PyTorch with CUDA support is not available for Python 3.14 yet

## Problem

PyTorch does not yet provide CUDA-enabled builds for Python 3.14. The latest Python version with CUDA support is Python 3.12.

## Solutions

### Option 1: Use Python 3.12 (Recommended for GPU Training)

This is the best option if you want to use GPU acceleration.

#### Steps:

1. **Install Python 3.12**
   - Download from: https://www.python.org/downloads/
   - Install Python 3.12.x (latest 3.12 version)

2. **Create new virtual environment with Python 3.12**
   ```bash
   # Navigate to project directory
   cd C:\MajorProject
   
   # Create new venv with Python 3.12
   py -3.12 -m venv .venv312
   
   # Activate the new environment
   .venv312\Scripts\activate
   ```

3. **Install PyTorch with CUDA support**
   ```bash
   # For CUDA 11.8 (most compatible)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # OR for CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install other dependencies**
   ```bash
   pip install -r ai-image-detector/requirements.txt
   ```

5. **Verify CUDA is working**
   ```bash
   python check_cuda_setup.py
   ```

**Expected output:**
```
✓ PyTorch is installed
  Version: 2.x.x+cu118
  CUDA available: True
  CUDA version: 11.8
  GPU device count: 1
  GPU 0: NVIDIA GeForce GTX 1650 Ti
    Memory: 4.00 GB
```

### Option 2: Continue with CPU Training (Current Setup)

You can continue training on CPU with your current Python 3.14 setup. It will be slower but will work.

#### Performance Comparison:

| Hardware | Training Time (10 epochs) | Notes |
|----------|---------------------------|-------|
| GTX 1650 Ti (GPU) | ~30-50 minutes | Recommended |
| CPU | ~3-5 hours | Slower but functional |

#### To use CPU training:

1. **Update config to use CPU**
   Edit `ai-image-detector/configs/default_config.yaml`:
   ```yaml
   device: "cpu"  # Change from "cuda"
   ```

2. **Reduce batch size for CPU**
   ```yaml
   training:
     batch_size: 16  # Smaller batch for CPU
   ```

3. **Train normally**
   ```bash
   python -m ai_image_detector.training --config ai-image-detector/configs/default_config.yaml
   ```

### Option 3: Wait for PyTorch CUDA Support for Python 3.14

PyTorch will eventually release CUDA builds for Python 3.14. You can:
- Monitor PyTorch releases: https://pytorch.org/get-started/locally/
- Check periodically: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Recommendations

### For Best Performance (GPU Training):
✅ **Use Option 1** - Install Python 3.12 and PyTorch with CUDA

**Benefits:**
- 5-10x faster training
- Can use larger batch sizes
- Better for experimentation and iteration

### For Quick Start (CPU Training):
✅ **Use Option 2** - Continue with current setup

**Benefits:**
- No additional setup required
- Works immediately
- Good for testing and small experiments

## Training Configuration for Your GPU

Your GTX 1650 Ti has 4GB VRAM. Here are recommended settings:

### For GPU Training (Python 3.12 + CUDA):

```yaml
# ai-image-detector/configs/default_config.yaml

model:
  backbone_type: "resnet18"  # Good balance for 4GB VRAM
  pretrained: true

training:
  batch_size: 32  # Safe for 4GB VRAM
  num_epochs: 10
  learning_rate: 0.001

dataset:
  num_workers: 4  # Parallel data loading

device: "cuda"
```

### For CPU Training (Current Python 3.14):

```yaml
# ai-image-detector/configs/default_config.yaml

model:
  backbone_type: "simple_cnn"  # Faster on CPU
  pretrained: false

training:
  batch_size: 16  # Smaller for CPU
  num_epochs: 5   # Fewer epochs for testing
  learning_rate: 0.001

dataset:
  num_workers: 2  # Fewer workers for CPU

device: "cpu"
```

## Memory Management Tips for 4GB GPU

If you get "CUDA out of memory" errors with GPU:

1. **Reduce batch size**
   ```yaml
   training:
     batch_size: 16  # or even 8
   ```

2. **Use smaller model**
   ```yaml
   model:
     backbone_type: "simple_cnn"  # Instead of resnet50
   ```

3. **Reduce workers**
   ```yaml
   dataset:
     num_workers: 2  # Less memory overhead
   ```

4. **Clear cache between runs**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## Quick Decision Guide

**Choose GPU Training (Option 1) if:**
- You want faster training (30-50 min vs 3-5 hours)
- You plan to experiment with different models
- You're willing to set up Python 3.12

**Choose CPU Training (Option 2) if:**
- You want to start immediately
- You're just testing the setup
- You don't mind slower training

## Verification Commands

After setup, verify your configuration:

```bash
# Check CUDA availability
python check_cuda_setup.py

# Check balanced dataset
python verify_balanced_dataset.py

# Quick test training (1 epoch)
# Edit config to set num_epochs: 1 first
python -m ai_image_detector.training --config ai-image-detector/configs/default_config.yaml
```

## Current Status Summary

✅ **Working:**
- NVIDIA GPU detected (GTX 1650 Ti)
- NVIDIA drivers installed (581.57)
- PyTorch installed (CPU version)
- Balanced dataset configured
- Training code ready

⚠️ **Needs Attention:**
- PyTorch CUDA support not available for Python 3.14
- Choose Option 1 (Python 3.12) or Option 2 (CPU training)

## Next Steps

1. **Decide:** GPU (Option 1) or CPU (Option 2) training
2. **Configure:** Update `default_config.yaml` with appropriate settings
3. **Verify:** Run verification scripts
4. **Train:** Start training with balanced dataset

The model will train successfully on either CPU or GPU - the only difference is speed!
