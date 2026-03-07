"""
Example usage of noise imprint clustering analysis.

This script demonstrates how to evaluate noise imprint clustering metrics
to measure how well different generators are separated in feature space.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.noise_clustering import (
    evaluate_noise_imprint_clustering,
    print_clustering_report,
    extract_noise_features,
    compute_pairwise_separability
)


def create_synthetic_dataset(num_samples_per_generator=100, num_generators=5):
    """
    Create a synthetic dataset with multiple generators.
    
    In practice, you would load real images from different generators
    (e.g., DALL-E, Midjourney, Stable Diffusion, etc.)
    """
    images_list = []
    labels_list = []
    
    for gen_id in range(num_generators):
        # Create synthetic images for this generator
        images = torch.randn(num_samples_per_generator, 3, 256, 256)
        labels = torch.full((num_samples_per_generator,), gen_id, dtype=torch.long)
        
        images_list.append(images)
        labels_list.append(labels)
    
    all_images = torch.cat(images_list)
    all_labels = torch.cat(labels_list)
    
    # Shuffle the dataset
    indices = torch.randperm(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    return all_images, all_labels


def main():
    """Main example workflow."""
    print("=" * 60)
    print("Noise Imprint Clustering Analysis Example")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    num_generators = 5
    generator_labels = [
        'DALL-E 3',
        'Midjourney v6',
        'Stable Diffusion XL',
        'Firefly',
        'Imagen'
    ]
    
    images, labels = create_synthetic_dataset(
        num_samples_per_generator=50,
        num_generators=num_generators
    )
    print(f"   Created {len(images)} images from {num_generators} generators")
    
    # Create dataloader
    dataset = TensorDataset(images, labels)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Load or create model
    print("\n2. Loading model with noise imprint branch...")
    # In practice, you would load a trained model:
    # model = BinaryClassifier(use_noise_imprint=True)
    # model.load_state_dict(torch.load('checkpoint.pth'))
    
    # For this example, we'll create a mock model
    from evaluation.test_noise_clustering import MockModel
    model = MockModel(feature_dim=256, enable_attribution=False)
    model = model.to(device)
    print("   Model loaded successfully")
    
    # Evaluate clustering metrics
    print("\n3. Evaluating noise imprint clustering...")
    metrics = evaluate_noise_imprint_clustering(
        model=model,
        test_loader=test_loader,
        device=device,
        generator_labels=generator_labels
    )
    
    # Print detailed report
    print_clustering_report(metrics, verbose=True)
    
    # Extract features for additional analysis
    print("\n4. Extracting features for pairwise analysis...")
    all_features = []
    all_labels_np = []
    
    for batch_images, batch_labels in test_loader:
        features = extract_noise_features(model, batch_images, device)
        all_features.append(features)
        all_labels_np.append(batch_labels.numpy())
    
    all_features = np.concatenate(all_features)
    all_labels_np = np.concatenate(all_labels_np)
    
    # Compute pairwise separability
    print("\n5. Computing pairwise generator separability...")
    pairwise_scores = compute_pairwise_separability(
        all_features,
        all_labels_np,
        generator_labels
    )
    
    print("\nPairwise Silhouette Scores:")
    print("-" * 60)
    for (gen_i, gen_j), score in sorted(pairwise_scores.items()):
        name_i = generator_labels[gen_i]
        name_j = generator_labels[gen_j]
        print(f"  {name_i:20s} vs {name_j:20s}: {score:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"  - Overall silhouette score: {metrics['silhouette_score']:.4f}")
    print(f"  - Davies-Bouldin index: {metrics['davies_bouldin_index']:.4f}")
    print(f"  - Number of generators: {metrics['num_generators']}")
    print(f"  - Total samples analyzed: {metrics['num_samples']}")
    
    # Interpretation
    if metrics['silhouette_score'] > 0.5:
        print("\n  ✓ Good separation between generators detected!")
        print("    The noise imprint features successfully distinguish")
        print("    between different generator models.")
    elif metrics['silhouette_score'] > 0.25:
        print("\n  ~ Moderate separation between generators.")
        print("    Some generators may be difficult to distinguish.")
    else:
        print("\n  ✗ Poor separation between generators.")
        print("    The model may need more training or different features.")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
