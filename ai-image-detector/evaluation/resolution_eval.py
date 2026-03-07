"""
Any-resolution evaluation module for testing model performance across image sizes.

This module evaluates model performance on images of varying resolutions by
stratifying images into size ranges and computing metrics for each stratum.
This helps assess how well the model handles different image sizes.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_any_resolution(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    size_ranges: Optional[List[Tuple[int, int]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on images of varying resolutions with size stratification.
    
    This function groups images by size ranges and computes comprehensive metrics
    for each size stratum. This helps assess how well the model handles different
    image resolutions and identifies potential size-dependent performance issues.
    
    Args:
        model: Trained model to evaluate (should support variable-sized inputs)
        test_loader: DataLoader with variable-sized images
                    Should use variable_size_collate_fn or similar
        device: Device to run evaluation on (cuda/cpu)
        size_ranges: List of (min_size, max_size) tuples for stratification
                    Default: [(128, 256), (256, 512), (512, 1024)]
                    Images are grouped by max(height, width)
    
    Returns:
        Dictionary mapping size range strings to metrics:
        {
            '128-256': {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1': float,
                'auc': float,
                'num_samples': int,
                'confusion_matrix': [[TN, FP], [FN, TP]],
                'avg_height': float,
                'avg_width': float
            },
            '256-512': {...},
            ...
        }
    
    Example:
        >>> from ai_image_detector.models.resolution import AnyResolutionWrapper
        >>> base_model = BinaryClassifier(backbone_type='resnet18')
        >>> model = AnyResolutionWrapper(base_model, tile_size=256, stride=128)
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> device = torch.device('cuda')
        >>> # DataLoader with native_resolution=True and variable_size_collate_fn
        >>> results = evaluate_any_resolution(model, test_loader, device)
        >>> print(f"128-256 accuracy: {results['128-256']['accuracy']:.4f}")
        >>> print(f"256-512 accuracy: {results['256-512']['accuracy']:.4f}")
    """
    if size_ranges is None:
        size_ranges = [(128, 256), (256, 512), (512, 1024)]
    
    model.eval()
    
    # Initialize storage for each size range
    size_groups = {}
    for min_s, max_s in size_ranges:
        range_key = f"{min_s}-{max_s}"
        size_groups[range_key] = {
            'labels': [],
            'predictions': [],
            'probabilities': [],
            'heights': [],
            'widths': []
        }
    
    # Collect predictions grouped by size
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle both (images, labels) and (images, labels, dataset_name) formats
            if len(batch_data) >= 2:
                images, labels = batch_data[0], batch_data[1]
            else:
                raise ValueError("DataLoader must return at least (images, labels)")
            
            # Handle both batched tensors and lists of tensors (variable size)
            if isinstance(images, list):
                # Variable-sized images (list of tensors)
                for img, label in zip(images, labels):
                    # Get image size
                    if img.dim() == 3:  # (C, H, W)
                        h, w = img.shape[1], img.shape[2]
                    else:
                        raise ValueError(f"Expected 3D image tensor, got shape {img.shape}")
                    
                    size = max(h, w)
                    
                    # Find appropriate size range
                    range_key = _find_size_range(size, size_ranges)
                    if range_key is None:
                        continue  # Skip images outside all ranges
                    
                    # Process single image
                    img_batch = img.unsqueeze(0).to(device)
                    outputs = model(img_batch)
                    
                    # Handle both single output and tuple output (with attribution)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    probability = outputs.squeeze().cpu().item()
                    prediction = float(probability > 0.5)
                    
                    # Store results
                    size_groups[range_key]['labels'].append(label.item())
                    size_groups[range_key]['predictions'].append(prediction)
                    size_groups[range_key]['probabilities'].append(probability)
                    size_groups[range_key]['heights'].append(h)
                    size_groups[range_key]['widths'].append(w)
            else:
                # Fixed-size images (batched tensor)
                images = images.to(device)
                
                # Get image size (assuming all images in batch have same size)
                if images.dim() == 4:  # (B, C, H, W)
                    h, w = images.shape[2], images.shape[3]
                else:
                    raise ValueError(f"Expected 4D image tensor, got shape {images.shape}")
                
                size = max(h, w)
                range_key = _find_size_range(size, size_ranges)
                if range_key is None:
                    continue  # Skip images outside all ranges
                
                # Forward pass
                outputs = model(images)
                
                # Handle both single output and tuple output (with attribution)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probabilities = outputs.squeeze(1).cpu().numpy()
                predictions = (outputs > 0.5).float().squeeze(1).cpu().numpy()
                
                # Store results
                size_groups[range_key]['labels'].extend(labels.numpy())
                size_groups[range_key]['predictions'].extend(predictions)
                size_groups[range_key]['probabilities'].extend(probabilities)
                size_groups[range_key]['heights'].extend([h] * len(labels))
                size_groups[range_key]['widths'].extend([w] * len(labels))
    
    # Compute metrics for each size range
    results = {}
    for range_key, data in size_groups.items():
        if len(data['labels']) > 0:
            metrics = _compute_metrics(
                np.array(data['labels']),
                np.array(data['predictions']),
                np.array(data['probabilities']),
                np.array(data['heights']),
                np.array(data['widths'])
            )
            results[range_key] = metrics
            
            # Print summary
            print(f"Size range {range_key}:")
            print(f"  Samples:   {metrics['num_samples']}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            if not np.isnan(metrics['auc']):
                print(f"  AUC:       {metrics['auc']:.4f}")
            else:
                print(f"  AUC:       N/A")
            print(f"  Avg size:  {metrics['avg_height']:.1f} x {metrics['avg_width']:.1f}")
        else:
            print(f"Size range {range_key}: No samples")
    
    return results


def _find_size_range(
    size: int,
    size_ranges: List[Tuple[int, int]]
) -> Optional[str]:
    """
    Find the appropriate size range for a given image size.
    
    Args:
        size: Image size (max of height and width)
        size_ranges: List of (min_size, max_size) tuples
    
    Returns:
        Range key string (e.g., '128-256') or None if outside all ranges
    """
    for min_s, max_s in size_ranges:
        if min_s <= size < max_s:
            return f"{min_s}-{max_s}"
    return None


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    heights: np.ndarray,
    widths: np.ndarray
) -> Dict[str, Union[float, int, List]]:
    """
    Compute evaluation metrics for a size stratum.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        heights: Image heights
        widths: Image widths
    
    Returns:
        Dictionary with metrics
    """
    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1 with zero_division handling
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC requires at least one sample from each class
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float('nan')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Size statistics
    avg_height = float(np.mean(heights))
    avg_width = float(np.mean(widths))
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'num_samples': int(len(y_true)),
        'confusion_matrix': cm.tolist(),
        'avg_height': avg_height,
        'avg_width': avg_width
    }


def generate_size_performance_matrix(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Generate a size-stratified performance matrix.
    
    This function reorganizes the results to show how each metric varies
    across size ranges, making it easier to identify size-dependent patterns.
    
    Args:
        results: Dictionary returned by evaluate_any_resolution()
        metrics: List of metrics to include in matrix
                (default: ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    Returns:
        Dictionary mapping metrics to size-range-value mappings:
        {
            'accuracy': {'128-256': 0.95, '256-512': 0.94, ...},
            'precision': {'128-256': 0.94, '256-512': 0.93, ...},
            ...
        }
    
    Example:
        >>> results = evaluate_any_resolution(model, test_loader, device)
        >>> matrix = generate_size_performance_matrix(results)
        >>> print("Accuracy across size ranges:")
        >>> for size_range, acc in matrix['accuracy'].items():
        ...     print(f"  {size_range}: {acc:.4f}")
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    matrix = {metric: {} for metric in metrics}
    
    for size_range, size_results in results.items():
        for metric in metrics:
            if metric in size_results:
                matrix[metric][size_range] = size_results[metric]
    
    return matrix


def print_resolution_report(
    results: Dict[str, Dict[str, float]],
    verbose: bool = True
) -> None:
    """
    Print formatted any-resolution evaluation report.
    
    Args:
        results: Dictionary returned by evaluate_any_resolution()
        verbose: If True, print confusion matrices for each size range
    
    Example:
        >>> results = evaluate_any_resolution(model, test_loader, device)
        >>> print_resolution_report(results)
        
        ========================================
        ANY-RESOLUTION EVALUATION REPORT
        ========================================
        
        Size Range: 128-256
        ----------------------------------------
          Samples:   500
          Avg Size:  192.3 x 189.7
          Accuracy:  94.80%
          Precision: 94.20%
          Recall:    95.40%
          F1 Score:  94.80%
          AUC:       0.978
        
        Confusion Matrix:
          [[230  12]
           [ 14 244]]
        
        Size Range: 256-512
        ----------------------------------------
          ...
    """
    print("\n" + "=" * 40)
    print("ANY-RESOLUTION EVALUATION REPORT")
    print("=" * 40)
    
    for size_range, metrics in results.items():
        print(f"\nSize Range: {size_range}")
        print("-" * 40)
        print(f"  Samples:   {metrics['num_samples']}")
        print(f"  Avg Size:  {metrics['avg_height']:.1f} x {metrics['avg_width']:.1f}")
        print(f"  Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        print(f"  Precision: {metrics['precision'] * 100:.2f}%")
        print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
        print(f"  F1 Score:  {metrics['f1'] * 100:.2f}%")
        
        if not np.isnan(metrics['auc']):
            print(f"  AUC:       {metrics['auc']:.3f}")
        else:
            print(f"  AUC:       N/A")
        
        if verbose and 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            print("\nConfusion Matrix:")
            print(f"  [[{cm[0, 0]:4d} {cm[0, 1]:4d}]")
            print(f"   [{cm[1, 0]:4d} {cm[1, 1]:4d}]]")
            print("  (TN FP)")
            print("  (FN TP)")
    
    print("\n" + "=" * 40)


def print_size_performance_matrix(
    matrix: Dict[str, Dict[str, float]]
) -> None:
    """
    Print formatted performance matrix showing metrics across size ranges.
    
    Args:
        matrix: Dictionary returned by generate_size_performance_matrix()
    
    Example:
        >>> results = evaluate_any_resolution(model, test_loader, device)
        >>> matrix = generate_size_performance_matrix(results)
        >>> print_size_performance_matrix(matrix)
        
        ========================================
        SIZE-STRATIFIED PERFORMANCE MATRIX
        ========================================
        
        Metric: accuracy
        ----------------------------------------
          128-256:   94.80%
          256-512:   95.20%
          512-1024:  93.60%
        
        Metric: precision
        ----------------------------------------
          ...
    """
    print("\n" + "=" * 40)
    print("SIZE-STRATIFIED PERFORMANCE MATRIX")
    print("=" * 40)
    
    for metric, size_values in matrix.items():
        print(f"\nMetric: {metric}")
        print("-" * 40)
        
        for size_range, value in size_values.items():
            if metric in ['accuracy', 'precision', 'recall', 'f1']:
                # Display as percentage
                print(f"  {size_range:10s}: {value * 100:5.2f}%")
            elif not np.isnan(value):
                # Display as decimal
                print(f"  {size_range:10s}: {value:.3f}")
            else:
                print(f"  {size_range:10s}: N/A")
    
    print("\n" + "=" * 40)


def compute_size_variance(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute variance statistics across size ranges for each metric.
    
    This function helps identify which metrics show high variance across
    different image sizes, indicating potential size-dependent issues.
    
    Args:
        results: Dictionary returned by evaluate_any_resolution()
        metrics: List of metrics to analyze
                (default: ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    Returns:
        Dictionary mapping metrics to statistics:
        {
            'accuracy': {
                'mean': float,
                'std': float,
                'min': float,
                'max': float,
                'range': float
            },
            ...
        }
    
    Example:
        >>> results = evaluate_any_resolution(model, test_loader, device)
        >>> variance = compute_size_variance(results)
        >>> print(f"Accuracy mean: {variance['accuracy']['mean']:.4f}")
        >>> print(f"Accuracy std:  {variance['accuracy']['std']:.4f}")
        >>> if variance['accuracy']['std'] > 0.05:
        ...     print("Warning: High variance across sizes!")
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    variance_stats = {}
    
    for metric in metrics:
        # Collect values for this metric across all size ranges
        values = []
        for size_results in results.values():
            if metric in size_results:
                value = size_results[metric]
                if not np.isnan(value):
                    values.append(value)
        
        if len(values) > 0:
            values = np.array(values)
            variance_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values))
            }
        else:
            variance_stats[metric] = {
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
                'range': float('nan')
            }
    
    return variance_stats


def print_size_variance_report(
    variance_stats: Dict[str, Dict[str, float]]
) -> None:
    """
    Print formatted variance report showing size-stratified statistics.
    
    Args:
        variance_stats: Dictionary returned by compute_size_variance()
    
    Example:
        >>> results = evaluate_any_resolution(model, test_loader, device)
        >>> variance = compute_size_variance(results)
        >>> print_size_variance_report(variance)
        
        ========================================
        SIZE-STRATIFIED VARIANCE REPORT
        ========================================
        
        Metric: accuracy
        ----------------------------------------
          Mean:  94.53%
          Std:    0.82%
          Min:   93.60%
          Max:   95.20%
          Range:  1.60%
        
        Metric: precision
        ----------------------------------------
          ...
    """
    print("\n" + "=" * 40)
    print("SIZE-STRATIFIED VARIANCE REPORT")
    print("=" * 40)
    
    for metric, stats in variance_stats.items():
        print(f"\nMetric: {metric}")
        print("-" * 40)
        
        if not np.isnan(stats['mean']):
            if metric in ['accuracy', 'precision', 'recall', 'f1']:
                # Display as percentage
                print(f"  Mean:  {stats['mean'] * 100:5.2f}%")
                print(f"  Std:   {stats['std'] * 100:5.2f}%")
                print(f"  Min:   {stats['min'] * 100:5.2f}%")
                print(f"  Max:   {stats['max'] * 100:5.2f}%")
                print(f"  Range: {stats['range'] * 100:5.2f}%")
            else:
                # Display as decimal
                print(f"  Mean:  {stats['mean']:.3f}")
                print(f"  Std:   {stats['std']:.3f}")
                print(f"  Min:   {stats['min']:.3f}")
                print(f"  Max:   {stats['max']:.3f}")
                print(f"  Range: {stats['range']:.3f}")
        else:
            print("  No valid data")
    
    print("\n" + "=" * 40)
