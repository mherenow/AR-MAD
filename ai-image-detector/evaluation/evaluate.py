"""Evaluation metrics for AI image detection."""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List, Tuple


def compute_per_generator_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    generator_labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy and AUC for each generator subset.
    
    Args:
        y_true: Ground truth labels (0 for real, 1 for AI-generated)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for the positive class
        generator_labels: List of generator names corresponding to each sample
        
    Returns:
        Dictionary mapping generator names to {'accuracy': float, 'auc': float}
    """
    generator_labels = np.array(generator_labels)
    unique_generators = np.unique(generator_labels)
    
    results = {}
    
    for generator in unique_generators:
        # Get indices for this generator
        mask = generator_labels == generator
        
        # Extract subset for this generator
        gen_y_true = y_true[mask]
        gen_y_pred = y_pred[mask]
        gen_y_prob = y_prob[mask]
        
        # Compute metrics
        accuracy = accuracy_score(gen_y_true, gen_y_pred)
        
        # AUC requires at least one sample from each class
        if len(np.unique(gen_y_true)) > 1:
            auc = roc_auc_score(gen_y_true, gen_y_prob)
        else:
            auc = float('nan')
        
        results[generator] = {
            'accuracy': float(accuracy),
            'auc': float(auc)
        }
    
    return results


def evaluate_model(
    checkpoint_path: str,
    model: 'torch.nn.Module',
    dataloader: 'torch.utils.data.DataLoader',
    device: 'torch.device'
) -> Dict[str, any]:
    """
    Evaluate model on test set with comprehensive metrics.
    
    Loads model from checkpoint, computes overall accuracy and AUC,
    and provides per-generator breakdown using compute_per_generator_metrics().
    
    Args:
        checkpoint_path: Path to model checkpoint file (.pth)
        model: BinaryClassifier model instance (architecture must match checkpoint)
        dataloader: Test data loader (should return images, labels, generator_names)
        device: Device to run evaluation on (cuda/cpu)
        
    Returns:
        Dictionary containing:
            - 'overall_accuracy': float, overall accuracy on test set
            - 'overall_auc': float, overall AUC-ROC score
            - 'per_generator_metrics': dict, breakdown by generator (from compute_per_generator_metrics)
            - 'num_samples': int, total number of test samples
            
    Example:
        >>> model = BinaryClassifier(backbone_type='resnet18')
        >>> device = torch.device('cuda')
        >>> metrics = evaluate_model('checkpoints/best_model.pth', model, test_loader, device)
        >>> print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        >>> print(f"Overall AUC: {metrics['overall_auc']:.4f}")
    """
    import torch
    
    # Load model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Collect predictions and labels
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_generator_names = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Handle both (images, labels) and (images, labels, generator_name) formats
            if len(batch_data) == 3:
                images, labels, generator_names = batch_data
            else:
                images, labels = batch_data
                generator_names = ['unknown'] * len(labels)
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)  # (B, 1)
            probabilities = outputs.squeeze(1).cpu().numpy()  # (B,)
            predictions = (outputs > 0.5).float().squeeze(1).cpu().numpy()  # (B,)
            
            # Collect results
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_generator_names.extend(generator_names)
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Compute overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_auc = roc_auc_score(y_true, y_prob)
    
    # Compute per-generator metrics
    per_generator_metrics = compute_per_generator_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        generator_labels=all_generator_names
    )
    
    # TODO: Add spectral/frequency-domain evaluation metrics
    # Implement evaluation metrics that analyze model performance on frequency domain
    # artifacts. Consider computing detection accuracy specifically for images with
    # known spectral artifacts (e.g., checkerboard patterns, upsampling artifacts).
    # This helps assess whether the model is learning frequency-based features.
    
    # TODO: Add noise-based imprint evaluation
    # Implement evaluation that measures the model's ability to detect and attribute
    # images based on generator-specific noise patterns. Consider computing metrics
    # for noise residual analysis and generator fingerprint matching. This validates
    # whether the model learns meaningful noise-based features.
    
    # TODO: Add multi-dataset evaluation support
    # Implement cross-dataset evaluation to assess generalization across different
    # data sources. Evaluate on held-out datasets not seen during training to measure
    # domain adaptation performance. Report per-dataset metrics to identify dataset-
    # specific biases or failure modes.
    
    # TODO: Add robustness testing
    # Implement evaluation under various image perturbations to assess model robustness:
    # - JPEG compression at different quality levels (e.g., 50, 75, 90)
    # - Gaussian blur with varying kernel sizes (e.g., 3x3, 5x5, 7x7)
    # - Additive Gaussian noise at different SNR levels
    # - Resizing and rescaling operations
    # This will help understand model performance degradation under real-world conditions
    # and identify vulnerabilities to adversarial perturbations.
    
    # TODO: Add any-resolution evaluation
    # Implement evaluation across multiple input resolutions to assess the model's
    # ability to handle variable-sized images. Test on images at native resolutions
    # (e.g., 512x512, 1024x1024, 2048x2048) without resizing to validate any-resolution
    # processing capabilities. Report performance metrics stratified by resolution.
    
    return {
        'overall_accuracy': float(overall_accuracy),
        'overall_auc': float(overall_auc),
        'per_generator_metrics': per_generator_metrics,
        'num_samples': len(y_true)
    }


def print_evaluation_report(metrics: Dict[str, any]) -> None:
    """
    Print formatted evaluation report with overall and per-generator metrics.
    
    Args:
        metrics: Dictionary returned by evaluate_model() containing:
            - 'overall_accuracy': float
            - 'overall_auc': float
            - 'per_generator_metrics': dict
            - 'num_samples': int
    
    Example:
        >>> metrics = evaluate_model(checkpoint_path, model, test_loader, device)
        >>> print_evaluation_report(metrics)
        
        ========================================
        EVALUATION REPORT
        ========================================
        Total Samples: 1000
        
        Overall Metrics:
          Accuracy: 92.50%
          AUC:      0.956
        
        Per-Generator Metrics:
        ----------------------------------------
        DALL-E 2:
          Accuracy: 94.20%
          AUC:      0.968
        
        Stable Diffusion:
          Accuracy: 91.30%
          AUC:      0.945
        ----------------------------------------
    """
    print("\n" + "=" * 40)
    print("EVALUATION REPORT")
    print("=" * 40)
    print(f"Total Samples: {metrics['num_samples']}")
    print()
    
    # Overall metrics
    print("Overall Metrics:")
    print(f"  Accuracy: {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"  AUC:      {metrics['overall_auc']:.3f}")
    print()
    
    # Per-generator metrics
    print("Per-Generator Metrics:")
    print("-" * 40)
    
    per_gen = metrics['per_generator_metrics']
    for generator_name in sorted(per_gen.keys()):
        gen_metrics = per_gen[generator_name]
        print(f"{generator_name}:")
        print(f"  Accuracy: {gen_metrics['accuracy'] * 100:.2f}%")
        
        # Handle NaN AUC (when only one class present)
        if np.isnan(gen_metrics['auc']):
            print(f"  AUC:      N/A (single class)")
        else:
            print(f"  AUC:      {gen_metrics['auc']:.3f}")
        print()
    
    print("-" * 40)
