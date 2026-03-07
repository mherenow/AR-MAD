"""
Example usage of any-resolution evaluation module.

This script demonstrates how to evaluate a model's performance across
different image resolutions using size stratification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from resolution_eval import (
    evaluate_any_resolution,
    generate_size_performance_matrix,
    print_resolution_report,
    print_size_performance_matrix,
    compute_size_variance,
    print_size_variance_report
)


class DummyVariableSizeDataset(Dataset):
    """
    Dummy dataset with variable-sized images for demonstration.
    
    In practice, use your actual dataset with native_resolution=True.
    """
    
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        
        # Generate random sizes across different ranges
        self.sizes = []
        self.labels = []
        
        for _ in range(num_samples):
            # Randomly choose a size range
            range_choice = np.random.choice(['small', 'medium', 'large'])
            
            if range_choice == 'small':
                h = np.random.randint(128, 256)
                w = np.random.randint(128, 256)
            elif range_choice == 'medium':
                h = np.random.randint(256, 512)
                w = np.random.randint(256, 512)
            else:
                h = np.random.randint(512, 1024)
                w = np.random.randint(512, 1024)
            
            self.sizes.append((h, w))
            self.labels.append(np.random.randint(0, 2))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        h, w = self.sizes[idx]
        image = torch.randn(3, h, w)
        label = self.labels[idx]
        return image, label


def variable_size_collate_fn(batch):
    """Collate function for variable-sized images."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


class DummyModel(nn.Module):
    """
    Dummy model that simulates size-dependent performance.
    
    In practice, use your actual model with any-resolution support:
    - AnyResolutionWrapper for tiling strategy
    - SpectralContextAttention for native variable-resolution
    - Or any model with adaptive pooling
    """
    
    def __init__(self):
        super().__init__()
        # Simulate size-dependent accuracy
        # Small images: 92% accuracy
        # Medium images: 95% accuracy
        # Large images: 90% accuracy
    
    def forward(self, x):
        """
        Simulate predictions with size-dependent accuracy.
        
        Args:
            x: Input image tensor (B, 3, H, W) or single image (3, H, W)
        
        Returns:
            Predictions (B, 1) or (1, 1)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        predictions = []
        
        for i in range(batch_size):
            h, w = x[i].shape[1], x[i].shape[2]
            size = max(h, w)
            
            # Simulate size-dependent performance
            if size < 256:
                # Small images: 92% accuracy
                prob = 0.92 if torch.rand(1).item() < 0.92 else 0.08
            elif size < 512:
                # Medium images: 95% accuracy
                prob = 0.95 if torch.rand(1).item() < 0.95 else 0.05
            else:
                # Large images: 90% accuracy
                prob = 0.90 if torch.rand(1).item() < 0.90 else 0.10
            
            predictions.append(prob)
        
        return torch.tensor(predictions).unsqueeze(1)


def main():
    """Run example any-resolution evaluation."""
    print("=" * 60)
    print("ANY-RESOLUTION EVALUATION EXAMPLE")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dummy dataset with variable-sized images
    print("\n1. Creating dataset with variable-sized images...")
    dataset = DummyVariableSizeDataset(num_samples=150)
    
    # Create data loader with variable-size collate function
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=variable_size_collate_fn
    )
    
    print(f"   Created dataset with {len(dataset)} samples")
    print(f"   Size ranges: 128-256, 256-512, 512-1024")
    
    # Create dummy model
    print("\n2. Loading model...")
    model = DummyModel()
    model.eval()
    device = torch.device('cpu')
    
    print("   Model loaded (simulates size-dependent performance)")
    
    # Run any-resolution evaluation
    print("\n3. Running any-resolution evaluation...")
    print("-" * 60)
    
    results = evaluate_any_resolution(
        model=model,
        test_loader=loader,
        device=device,
        size_ranges=[(128, 256), (256, 512), (512, 1024)]
    )
    
    # Print comprehensive report
    print("\n4. Comprehensive Report:")
    print_resolution_report(results, verbose=True)
    
    # Generate and print performance matrix
    print("\n5. Performance Matrix:")
    matrix = generate_size_performance_matrix(results)
    print_size_performance_matrix(matrix)
    
    # Compute and print variance statistics
    print("\n6. Variance Analysis:")
    variance = compute_size_variance(results)
    print_size_variance_report(variance)
    
    # Analyze results
    print("\n7. Analysis:")
    print("-" * 60)
    
    # Check for high variance
    if variance['accuracy']['std'] > 0.05:
        print(f"⚠ Warning: High accuracy variance detected!")
        print(f"  Standard deviation: {variance['accuracy']['std']:.4f}")
        print(f"  Range: {variance['accuracy']['range']:.4f}")
        print(f"  This indicates size-dependent performance issues.")
    else:
        print(f"✓ Good: Low accuracy variance across sizes")
        print(f"  Standard deviation: {variance['accuracy']['std']:.4f}")
    
    # Identify best and worst performing size ranges
    best_range = max(results.items(), key=lambda x: x[1]['accuracy'])
    worst_range = min(results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\n  Best performing size: {best_range[0]}")
    print(f"    Accuracy: {best_range[1]['accuracy']:.4f}")
    print(f"    Samples: {best_range[1]['num_samples']}")
    
    print(f"\n  Worst performing size: {worst_range[0]}")
    print(f"    Accuracy: {worst_range[1]['accuracy']:.4f}")
    print(f"    Samples: {worst_range[1]['num_samples']}")
    
    # Check sample distribution
    print(f"\n  Sample distribution:")
    for size_range, metrics in results.items():
        pct = metrics['num_samples'] / len(dataset) * 100
        print(f"    {size_range}: {metrics['num_samples']} samples ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    # Return results for further analysis
    return results, matrix, variance


if __name__ == '__main__':
    results, matrix, variance = main()
    
    # Additional custom analysis can be done here
    print("\n\nCustom Analysis Example:")
    print("-" * 60)
    
    # Example: Check if any size range has insufficient samples
    for size_range, metrics in results.items():
        if metrics['num_samples'] < 30:
            print(f"⚠ Warning: {size_range} has only {metrics['num_samples']} samples")
            print(f"  Consider collecting more data in this size range")
    
    # Example: Compare F1 scores across sizes
    print("\nF1 Score Comparison:")
    for size_range in sorted(results.keys()):
        f1 = results[size_range]['f1']
        print(f"  {size_range}: {f1:.4f}")
