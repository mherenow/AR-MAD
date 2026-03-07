"""
Example script demonstrating robustness evaluation usage.

This script shows how to use the robustness evaluation module to test
model performance under various perturbations.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.robustness_eval import (
    evaluate_robustness,
    print_robustness_report,
    compute_robustness_degradation
)


class ExampleClassifier(nn.Module):
    """Example classifier for demonstration."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)


def create_example_dataset(num_samples=200):
    """Create an example dataset for demonstration."""
    # Create synthetic images and labels
    images = torch.rand(num_samples, 3, 128, 128)
    labels = torch.randint(0, 2, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return dataloader


def main():
    """Run example robustness evaluation."""
    print("=" * 60)
    print("Robustness Evaluation Example")
    print("=" * 60)
    
    # Create model and dataset
    print("\n1. Creating model and dataset...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExampleClassifier()
    model.to(device)
    model.eval()
    test_loader = create_example_dataset(num_samples=200)
    print(f"   Using device: {device}")
    
    # Run robustness evaluation
    print("\n2. Running robustness evaluation...")
    print("   This will test the model under:")
    print("   - JPEG compression at quality levels [95, 85, 75, 65, 50]")
    print("   - Gaussian blur at sigma levels [0.5, 1.0, 1.5, 2.0, 2.5]")
    print("   - Gaussian noise at std levels [0.01, 0.02, 0.03, 0.04, 0.05]")
    print()
    
    results = evaluate_robustness(
        model=model,
        test_loader=test_loader,
        device=device,
        perturbations=['jpeg', 'blur', 'noise']
    )
    
    # Print formatted report
    print("\n3. Evaluation Results:")
    print_robustness_report(results)
    
    # Compute degradation metrics
    print("\n4. Computing performance degradation...")
    degradation = compute_robustness_degradation(results)
    
    print("\nPerformance Degradation Summary:")
    print("-" * 60)
    
    # Show worst-case degradation for each perturbation type
    if 'jpeg' in degradation:
        worst_jpeg = max(degradation['jpeg'].items(), 
                        key=lambda x: x[1]['accuracy_drop'])
        print(f"JPEG Compression:")
        print(f"  Worst case: Quality {worst_jpeg[0]}")
        print(f"  Accuracy drop: {worst_jpeg[1]['accuracy_drop']:.4f}")
    
    if 'blur' in degradation:
        worst_blur = max(degradation['blur'].items(),
                        key=lambda x: x[1]['accuracy_drop'])
        print(f"\nGaussian Blur:")
        print(f"  Worst case: Sigma {worst_blur[0]}")
        print(f"  Accuracy drop: {worst_blur[1]['accuracy_drop']:.4f}")
    
    if 'noise' in degradation:
        worst_noise = max(degradation['noise'].items(),
                         key=lambda x: x[1]['accuracy_drop'])
        print(f"\nGaussian Noise:")
        print(f"  Worst case: Std {worst_noise[0]}")
        print(f"  Accuracy drop: {worst_noise[1]['accuracy_drop']:.4f}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
