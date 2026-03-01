"""
Test device auto-detection for training.

This script demonstrates how the training script automatically detects
and uses the best available device (CUDA, MPS, or CPU).
"""

import torch
import sys
from pathlib import Path

# Add ai-image-detector to path
sys.path.insert(0, str(Path(__file__).parent / "ai-image-detector"))

from utils.config_loader import load_config


def test_device_detection():
    """Test the device detection logic."""
    print("=" * 70)
    print("Device Auto-Detection Test")
    print("=" * 70)
    
    # Load config
    config = load_config("ai-image-detector/configs/default_config.yaml")
    device_config = config.get('device', 'auto')
    
    print(f"\nConfiguration device setting: '{device_config}'")
    print("\nDetecting available devices...")
    print("-" * 70)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Check MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    # Determine device based on config
    print("\n" + "-" * 70)
    print("Device Selection Logic:")
    print("-" * 70)
    
    if device_config == 'auto':
        print("Mode: AUTO (automatic detection)")
        if cuda_available:
            device = torch.device('cuda')
            print(f"✓ Selected: {device} (CUDA detected)")
        elif mps_available:
            device = torch.device('mps')
            print(f"✓ Selected: {device} (MPS detected)")
        else:
            device = torch.device('cpu')
            print(f"✓ Selected: {device} (no GPU detected)")
    elif device_config == 'cuda':
        print("Mode: CUDA (explicitly requested)")
        if cuda_available:
            device = torch.device('cuda')
            print(f"✓ Selected: {device}")
        else:
            device = torch.device('cpu')
            print(f"⚠ Selected: {device} (CUDA not available, falling back)")
    else:
        print(f"Mode: {device_config.upper()} (explicitly requested)")
        device = torch.device(device_config)
        print(f"✓ Selected: {device}")
    
    # Test tensor creation
    print("\n" + "-" * 70)
    print("Testing tensor creation on selected device...")
    print("-" * 70)
    
    try:
        test_tensor = torch.randn(3, 256, 256).to(device)
        print(f"✓ Successfully created tensor on {device}")
        print(f"  Tensor shape: {test_tensor.shape}")
        print(f"  Tensor device: {test_tensor.device}")
        print(f"  Memory allocated: {test_tensor.element_size() * test_tensor.nelement() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"✗ Failed to create tensor: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Config setting: {device_config}")
    print(f"Selected device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if device.type == 'cuda':
        print("\n✓ GPU training will be used (fast)")
        print("  Expected training time: ~30-50 minutes for 10 epochs")
    else:
        print("\n⚠ CPU training will be used (slower)")
        print("  Expected training time: ~3-5 hours for 10 epochs")
        if cuda_available:
            print("\n  Note: CUDA is available but not being used.")
            print("  Check your configuration or PyTorch installation.")
    
    print("=" * 70)
    
    return device


if __name__ == "__main__":
    device = test_device_detection()
    
    print("\nTo start training with this device:")
    print("  python -m ai_image_detector.training --config ai-image-detector/configs/default_config.yaml")
