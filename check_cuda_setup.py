"""
CUDA Setup Checker for AI Image Detector

This script checks your PyTorch and CUDA installation and provides
instructions for installing the correct version with GPU support.
"""

import sys
import subprocess

def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("=" * 70)
    print("PyTorch & CUDA Setup Checker")
    print("=" * 70)
    
    try:
        import torch
        print(f"\n✓ PyTorch is installed")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("\n⚠ WARNING: CUDA is NOT available!")
            print("  You have the CPU-only version of PyTorch installed.")
            print("\n" + "=" * 70)
            print("To install PyTorch with CUDA support:")
            print("=" * 70)
            print("\n1. First, uninstall current PyTorch:")
            print("   pip uninstall torch torchvision torchaudio")
            print("\n2. Check your CUDA version:")
            print("   nvidia-smi")
            print("\n3. Install PyTorch with CUDA support:")
            print("   For CUDA 11.8:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n   For CUDA 12.1:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("\n   For CUDA 12.4:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("\n4. Verify installation:")
            print("   python check_cuda_setup.py")
            print("\nFor more info, visit: https://pytorch.org/get-started/locally/")
            
    except ImportError:
        print("\n✗ PyTorch is NOT installed")
        print("\nInstall PyTorch with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "=" * 70)

def check_nvidia_driver():
    """Check if NVIDIA driver is installed."""
    print("\nChecking NVIDIA Driver...")
    print("-" * 70)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA driver is installed")
            print("\nGPU Information:")
            print(result.stdout)
        else:
            print("✗ nvidia-smi command failed")
            print("  Make sure NVIDIA drivers are installed")
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("  NVIDIA drivers may not be installed")
        print("\nInstall NVIDIA drivers from:")
        print("  https://www.nvidia.com/Download/index.aspx")
    except Exception as e:
        print(f"✗ Error checking NVIDIA driver: {e}")

if __name__ == '__main__':
    check_nvidia_driver()
    check_pytorch()
