"""
Example usage of AnyResolutionWrapper.

This script demonstrates how to use the AnyResolutionWrapper to process
images of arbitrary resolution using a tiling strategy.
"""

import torch
import torch.nn as nn
from any_resolution_wrapper import AnyResolutionWrapper


class SimpleClassifier(nn.Module):
    """Simple CNN classifier for demonstration."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    """Demonstrate AnyResolutionWrapper usage."""
    
    print("=" * 60)
    print("AnyResolutionWrapper Example Usage")
    print("=" * 60)
    
    # Create a simple classifier
    model = SimpleClassifier()
    model.eval()
    
    # Wrap it with AnyResolutionWrapper
    print("\n1. Creating wrapper with weighted averaging...")
    wrapper_avg = AnyResolutionWrapper(
        model,
        tile_size=256,
        stride=128,
        aggregation='average'
    )
    
    # Test with different image sizes
    test_sizes = [
        (128, 128),   # Small image (direct processing)
        (256, 256),   # Exact tile size
        (512, 512),   # Large square image (tiling)
        (768, 512),   # Large non-square image
    ]
    
    print("\n2. Testing with different image sizes:")
    print("-" * 60)
    
    for h, w in test_sizes:
        # Create random image
        x = torch.randn(1, 3, h, w)
        
        # Process through wrapper
        with torch.no_grad():
            output = wrapper_avg(x)
        
        print(f"   Image size: {h}x{w} -> Prediction: {output.item():.4f}")
    
    # Test voting aggregation
    print("\n3. Testing voting aggregation...")
    wrapper_vote = AnyResolutionWrapper(
        model,
        tile_size=256,
        stride=128,
        aggregation='voting'
    )
    
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = wrapper_vote(x)
    
    print(f"   Voting result for 512x512 image: {output.item():.4f}")
    
    # Test batch processing
    print("\n4. Testing batch processing...")
    x_batch = torch.randn(4, 3, 512, 512)
    with torch.no_grad():
        outputs = wrapper_avg(x_batch)
    
    print(f"   Batch size: {x_batch.shape[0]}")
    print(f"   Predictions: {outputs.squeeze().tolist()}")
    
    # Demonstrate tile extraction
    print("\n5. Demonstrating tile extraction...")
    x = torch.randn(1, 3, 512, 512)
    tiles, positions = wrapper_avg._extract_tiles(x)
    
    print(f"   Input image: {x.shape}")
    print(f"   Number of tiles: {len(positions)}")
    print(f"   Tile shape: {tiles.shape}")
    print(f"   First 3 tile positions: {positions[:3]}")
    
    # Demonstrate weight computation
    print("\n6. Demonstrating tile weighting...")
    weights = wrapper_avg._compute_tile_weights(positions, 512, 512)
    print(f"   Tile weights (first 5): {weights[:5].tolist()}")
    print(f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
