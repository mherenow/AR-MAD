"""
Noise imprint clustering analysis module for evaluating generator separation.

This module evaluates how well noise imprint features separate different generators
using clustering metrics like silhouette score and Davies-Bouldin index.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def evaluate_noise_imprint_clustering(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    generator_labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate noise imprint detection using cluster separation metrics.
    
    This function extracts noise imprint features from a test set and computes
    clustering metrics to measure how well different generators are separated
    in the feature space. Higher silhouette scores (closer to 1) and lower
    Davies-Bouldin indices indicate better separation.
    
    Args:
        model: Model with noise imprint branch (must have noise_extractor and noise_branch)
        test_loader: Test data loader that returns (images, labels) or (images, labels, generator_name)
                    where labels are generator indices (0, 1, 2, ...)
        device: Device to run evaluation on (cuda/cpu)
        generator_labels: Optional list of generator names for reporting
                         (e.g., ['DALL-E', 'Midjourney', 'Stable Diffusion'])
    
    Returns:
        Dictionary with clustering metrics:
        {
            'silhouette_score': float in [-1, 1], higher is better
            'davies_bouldin_index': float >= 0, lower is better
            'num_samples': int, number of samples evaluated
            'num_generators': int, number of unique generators
        }
    
    Raises:
        ValueError: If model doesn't have noise_extractor or noise_branch
        ValueError: If test set has fewer than 2 generators
        ValueError: If any generator has fewer than 2 samples
    
    Example:
        >>> model = BinaryClassifier(use_noise_imprint=True)
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> device = torch.device('cuda')
        >>> generator_labels = ['DALL-E', 'Midjourney', 'Stable Diffusion']
        >>> metrics = evaluate_noise_imprint_clustering(
        ...     model, test_loader, device, generator_labels
        ... )
        >>> print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
        >>> print(f"Davies-Bouldin index: {metrics['davies_bouldin_index']:.4f}")
    """
    # Validate model has required components
    if not hasattr(model, 'noise_extractor'):
        raise ValueError(
            "Model must have 'noise_extractor' attribute. "
            "Ensure model was created with use_noise_imprint=True"
        )
    if not hasattr(model, 'noise_branch'):
        raise ValueError(
            "Model must have 'noise_branch' attribute. "
            "Ensure model was created with use_noise_imprint=True"
        )
    
    # Extract noise imprint features
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle both (images, labels) and (images, labels, generator_name) formats
            if len(batch_data) >= 2:
                images, labels = batch_data[0], batch_data[1]
            else:
                raise ValueError("DataLoader must return at least (images, labels)")
            
            images = images.to(device)
            
            # Extract noise residual
            residual = model.noise_extractor(images)
            
            # Extract noise imprint features
            # Handle both single output and tuple output (with attribution)
            features_output = model.noise_branch(residual)
            if isinstance(features_output, tuple):
                features = features_output[0]  # Get features, ignore attribution
            else:
                features = features_output
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # Concatenate all features and labels
    features = np.concatenate(features_list, axis=0)  # (N, feature_dim)
    labels = np.concatenate(labels_list, axis=0)      # (N,)
    
    # Validate data for clustering metrics
    unique_labels = np.unique(labels)
    num_generators = len(unique_labels)
    num_samples = len(labels)
    
    if num_generators < 2:
        raise ValueError(
            f"Need at least 2 generators for clustering metrics, got {num_generators}. "
            "Ensure test set contains samples from multiple generators."
        )
    
    # Check that each generator has at least 2 samples
    for label in unique_labels:
        count = np.sum(labels == label)
        if count < 2:
            raise ValueError(
                f"Generator {label} has only {count} sample(s). "
                "Each generator needs at least 2 samples for clustering metrics."
            )
    
    # Compute clustering metrics
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    
    # Build results dictionary
    results = {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'num_samples': int(num_samples),
        'num_generators': int(num_generators)
    }
    
    # Add generator names if provided
    if generator_labels is not None:
        results['generator_labels'] = generator_labels
    
    return results


def print_clustering_report(
    metrics: Dict[str, float],
    verbose: bool = True
) -> None:
    """
    Print formatted clustering evaluation report.
    
    Args:
        metrics: Dictionary returned by evaluate_noise_imprint_clustering()
        verbose: If True, print detailed interpretation of metrics
    
    Example:
        >>> metrics = evaluate_noise_imprint_clustering(model, test_loader, device)
        >>> print_clustering_report(metrics)
        
        ========================================
        NOISE IMPRINT CLUSTERING REPORT
        ========================================
        
        Dataset Statistics:
          Samples:    1000
          Generators: 5
        
        Clustering Metrics:
          Silhouette Score:      0.6234 (Good separation)
          Davies-Bouldin Index:  0.8123 (Good separation)
        
        Interpretation:
          The silhouette score of 0.62 indicates that noise imprints
          from different generators are well-separated in feature space.
          The Davies-Bouldin index of 0.81 confirms good cluster quality.
    """
    print("\n" + "=" * 40)
    print("NOISE IMPRINT CLUSTERING REPORT")
    print("=" * 40)
    
    # Dataset statistics
    print("\nDataset Statistics:")
    print(f"  Samples:    {metrics['num_samples']}")
    print(f"  Generators: {metrics['num_generators']}")
    
    if 'generator_labels' in metrics:
        print(f"  Labels:     {', '.join(metrics['generator_labels'])}")
    
    # Clustering metrics
    print("\nClustering Metrics:")
    silhouette = metrics['silhouette_score']
    davies_bouldin = metrics['davies_bouldin_index']
    
    # Interpret silhouette score
    if silhouette > 0.7:
        sil_interp = "(Excellent separation)"
    elif silhouette > 0.5:
        sil_interp = "(Good separation)"
    elif silhouette > 0.25:
        sil_interp = "(Moderate separation)"
    elif silhouette > 0:
        sil_interp = "(Weak separation)"
    else:
        sil_interp = "(Poor separation)"
    
    # Interpret Davies-Bouldin index
    if davies_bouldin < 0.5:
        db_interp = "(Excellent separation)"
    elif davies_bouldin < 1.0:
        db_interp = "(Good separation)"
    elif davies_bouldin < 1.5:
        db_interp = "(Moderate separation)"
    else:
        db_interp = "(Poor separation)"
    
    print(f"  Silhouette Score:      {silhouette:.4f} {sil_interp}")
    print(f"  Davies-Bouldin Index:  {davies_bouldin:.4f} {db_interp}")
    
    # Verbose interpretation
    if verbose:
        print("\nInterpretation:")
        print(f"  The silhouette score of {silhouette:.2f} indicates that noise imprints")
        if silhouette > 0.5:
            print("  from different generators are well-separated in feature space.")
        elif silhouette > 0:
            print("  from different generators show some separation in feature space.")
        else:
            print("  from different generators are poorly separated in feature space.")
        
        print(f"  The Davies-Bouldin index of {davies_bouldin:.2f} ", end="")
        if davies_bouldin < 1.0:
            print("confirms good cluster quality.")
        elif davies_bouldin < 1.5:
            print("suggests moderate cluster quality.")
        else:
            print("indicates poor cluster quality.")
    
    print("\n" + "=" * 40)


def extract_noise_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Extract noise imprint features from a batch of images.
    
    This is a utility function for extracting features without computing
    clustering metrics, useful for visualization or further analysis.
    
    Args:
        model: Model with noise imprint branch
        images: Batch of images (B, C, H, W)
        device: Device to run on
    
    Returns:
        Noise imprint features as numpy array (B, feature_dim)
    
    Example:
        >>> model = BinaryClassifier(use_noise_imprint=True)
        >>> images = torch.randn(10, 3, 256, 256)
        >>> features = extract_noise_features(model, images, device)
        >>> features.shape
        (10, 256)
    """
    if not hasattr(model, 'noise_extractor') or not hasattr(model, 'noise_branch'):
        raise ValueError(
            "Model must have 'noise_extractor' and 'noise_branch' attributes"
        )
    
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        residual = model.noise_extractor(images)
        
        features_output = model.noise_branch(residual)
        if isinstance(features_output, tuple):
            features = features_output[0]
        else:
            features = features_output
        
        return features.cpu().numpy()


def compute_pairwise_separability(
    features: np.ndarray,
    labels: np.ndarray,
    generator_labels: Optional[List[str]] = None
) -> Dict[Tuple[int, int], float]:
    """
    Compute pairwise silhouette scores between generators.
    
    This function computes silhouette scores for each pair of generators,
    which can help identify which generators are most/least distinguishable.
    
    Args:
        features: Feature matrix (N, feature_dim)
        labels: Generator labels (N,)
        generator_labels: Optional list of generator names
    
    Returns:
        Dictionary mapping (generator_i, generator_j) to silhouette score
    
    Example:
        >>> features = np.random.randn(100, 256)
        >>> labels = np.repeat([0, 1, 2], [30, 40, 30])
        >>> pairwise = compute_pairwise_separability(features, labels)
        >>> print(f"Generators 0 vs 1: {pairwise[(0, 1)]:.4f}")
    """
    unique_labels = np.unique(labels)
    pairwise_scores = {}
    
    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i+1:]:
            # Extract samples from these two generators
            mask = (labels == label_i) | (labels == label_j)
            subset_features = features[mask]
            subset_labels = labels[mask]
            
            # Compute silhouette score for this pair
            if len(np.unique(subset_labels)) == 2:
                score = silhouette_score(subset_features, subset_labels)
                pairwise_scores[(int(label_i), int(label_j))] = float(score)
    
    return pairwise_scores
