"""
Cross-dataset evaluation module for testing model performance across datasets.

This module evaluates model performance separately on each dataset to assess
generalization and identify dataset-specific biases. It computes per-dataset
metrics (accuracy, precision, recall, F1) and generates a cross-dataset
performance matrix.
"""

from typing import Dict, List, Optional, Tuple

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


def evaluate_cross_dataset(
    model: nn.Module,
    dataset_loaders: Dict[str, DataLoader],
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance across multiple datasets.
    
    This function evaluates the model separately on each dataset and computes
    comprehensive metrics including accuracy, precision, recall, F1 score, and AUC.
    This helps assess how well the model generalizes across different data distributions.
    
    Args:
        model: Trained model to evaluate
        dataset_loaders: Dictionary mapping dataset names to DataLoader objects
                        e.g., {'synthbuster': loader1, 'coco2017': loader2}
        device: Device to run evaluation on (cuda/cpu)
    
    Returns:
        Dictionary mapping dataset names to metrics:
        {
            'synthbuster': {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1': float,
                'auc': float,
                'num_samples': int,
                'confusion_matrix': [[TN, FP], [FN, TP]]
            },
            'coco2017': {...},
            ...
        }
    
    Example:
        >>> model = BinaryClassifier(backbone_type='resnet18')
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> device = torch.device('cuda')
        >>> dataset_loaders = {
        ...     'synthbuster': synthbuster_loader,
        ...     'coco2017': coco_loader
        ... }
        >>> results = evaluate_cross_dataset(model, dataset_loaders, device)
        >>> print(f"SynthBuster accuracy: {results['synthbuster']['accuracy']:.4f}")
        >>> print(f"COCO accuracy: {results['coco2017']['accuracy']:.4f}")
    """
    model.eval()
    results = {}
    
    for dataset_name, loader in dataset_loaders.items():
        print(f"Evaluating on {dataset_name}...")
        metrics = _evaluate_single_dataset(model, loader, device)
        results[dataset_name] = metrics
        
        # Print summary
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if not np.isnan(metrics['auc']):
            print(f"  AUC:       {metrics['auc']:.4f}")
        else:
            print(f"  AUC:       N/A")
        print(f"  Samples:   {metrics['num_samples']}")
    
    return results


def _evaluate_single_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on a single dataset.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for the dataset
        device: Device to run on
    
    Returns:
        Dictionary with metrics for this dataset
    """
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_data in loader:
            # Handle both (images, labels) and (images, labels, dataset_name) formats
            if len(batch_data) >= 2:
                images, labels = batch_data[0], batch_data[1]
            else:
                raise ValueError("DataLoader must return at least (images, labels)")
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle both single output and tuple output (with attribution)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probabilities = outputs.squeeze(1).cpu().numpy()
            predictions = (outputs > 0.5).float().squeeze(1).cpu().numpy()
            
            # Collect results
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Handle empty dataset
    if len(y_true) == 0:
        return {
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'auc': float('nan'),
            'num_samples': 0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }
    
    # Compute metrics
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
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'num_samples': int(len(y_true)),
        'confusion_matrix': cm.tolist()
    }


def generate_performance_matrix(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Generate a cross-dataset performance matrix.
    
    This function reorganizes the results to show how each metric varies
    across datasets, making it easier to compare performance.
    
    Args:
        results: Dictionary returned by evaluate_cross_dataset()
        metrics: List of metrics to include in matrix
                (default: ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    Returns:
        Dictionary mapping metrics to dataset-value mappings:
        {
            'accuracy': {'synthbuster': 0.95, 'coco2017': 0.92, ...},
            'precision': {'synthbuster': 0.94, 'coco2017': 0.91, ...},
            ...
        }
    
    Example:
        >>> results = evaluate_cross_dataset(model, dataset_loaders, device)
        >>> matrix = generate_performance_matrix(results)
        >>> print("Accuracy across datasets:")
        >>> for dataset, acc in matrix['accuracy'].items():
        ...     print(f"  {dataset}: {acc:.4f}")
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    matrix = {metric: {} for metric in metrics}
    
    for dataset_name, dataset_results in results.items():
        for metric in metrics:
            if metric in dataset_results:
                matrix[metric][dataset_name] = dataset_results[metric]
    
    return matrix


def print_cross_dataset_report(
    results: Dict[str, Dict[str, float]],
    verbose: bool = True
) -> None:
    """
    Print formatted cross-dataset evaluation report.
    
    Args:
        results: Dictionary returned by evaluate_cross_dataset()
        verbose: If True, print confusion matrices for each dataset
    
    Example:
        >>> results = evaluate_cross_dataset(model, dataset_loaders, device)
        >>> print_cross_dataset_report(results)
        
        ========================================
        CROSS-DATASET EVALUATION REPORT
        ========================================
        
        Dataset: synthbuster
        ----------------------------------------
          Samples:   1000
          Accuracy:  95.20%
          Precision: 94.80%
          Recall:    95.60%
          F1 Score:  95.20%
          AUC:       0.982
        
        Confusion Matrix:
          [[450  20]
           [ 24 506]]
        
        Dataset: coco2017
        ----------------------------------------
          ...
    """
    print("\n" + "=" * 40)
    print("CROSS-DATASET EVALUATION REPORT")
    print("=" * 40)
    
    for dataset_name, metrics in results.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 40)
        print(f"  Samples:   {metrics['num_samples']}")
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


def print_performance_matrix(
    matrix: Dict[str, Dict[str, float]]
) -> None:
    """
    Print formatted performance matrix showing metrics across datasets.
    
    Args:
        matrix: Dictionary returned by generate_performance_matrix()
    
    Example:
        >>> results = evaluate_cross_dataset(model, dataset_loaders, device)
        >>> matrix = generate_performance_matrix(results)
        >>> print_performance_matrix(matrix)
        
        ========================================
        CROSS-DATASET PERFORMANCE MATRIX
        ========================================
        
        Metric: accuracy
        ----------------------------------------
          synthbuster:  95.20%
          coco2017:     92.40%
          imagenet:     93.80%
        
        Metric: precision
        ----------------------------------------
          ...
    """
    print("\n" + "=" * 40)
    print("CROSS-DATASET PERFORMANCE MATRIX")
    print("=" * 40)
    
    for metric, dataset_values in matrix.items():
        print(f"\nMetric: {metric}")
        print("-" * 40)
        
        for dataset_name, value in dataset_values.items():
            if metric in ['accuracy', 'precision', 'recall', 'f1']:
                # Display as percentage
                print(f"  {dataset_name:15s}: {value * 100:5.2f}%")
            elif not np.isnan(value):
                # Display as decimal
                print(f"  {dataset_name:15s}: {value:.3f}")
            else:
                print(f"  {dataset_name:15s}: N/A")
    
    print("\n" + "=" * 40)


def compute_cross_dataset_variance(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute variance statistics across datasets for each metric.
    
    This function helps identify which metrics show high variance across
    datasets, indicating potential generalization issues or dataset biases.
    
    Args:
        results: Dictionary returned by evaluate_cross_dataset()
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
        >>> results = evaluate_cross_dataset(model, dataset_loaders, device)
        >>> variance = compute_cross_dataset_variance(results)
        >>> print(f"Accuracy mean: {variance['accuracy']['mean']:.4f}")
        >>> print(f"Accuracy std:  {variance['accuracy']['std']:.4f}")
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    variance_stats = {}
    
    for metric in metrics:
        # Collect values for this metric across all datasets
        values = []
        for dataset_results in results.values():
            if metric in dataset_results:
                value = dataset_results[metric]
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


def print_variance_report(
    variance_stats: Dict[str, Dict[str, float]]
) -> None:
    """
    Print formatted variance report showing cross-dataset statistics.
    
    Args:
        variance_stats: Dictionary returned by compute_cross_dataset_variance()
    
    Example:
        >>> results = evaluate_cross_dataset(model, dataset_loaders, device)
        >>> variance = compute_cross_dataset_variance(results)
        >>> print_variance_report(variance)
        
        ========================================
        CROSS-DATASET VARIANCE REPORT
        ========================================
        
        Metric: accuracy
        ----------------------------------------
          Mean:  93.80%
          Std:    2.10%
          Min:   91.20%
          Max:   95.60%
          Range:  4.40%
        
        Metric: precision
        ----------------------------------------
          ...
    """
    print("\n" + "=" * 40)
    print("CROSS-DATASET VARIANCE REPORT")
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
