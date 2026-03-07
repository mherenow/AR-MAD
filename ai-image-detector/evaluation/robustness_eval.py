"""
Robustness evaluation module for testing model performance under perturbations.

This module evaluates model robustness by applying various perturbations
(JPEG compression, Gaussian blur, Gaussian noise) at multiple severity levels
and measuring performance degradation.
"""

import io
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


class RobustnessPerturbation:
    """
    Applies specific perturbations at configurable severity levels.
    
    This class provides methods to apply JPEG compression, Gaussian blur,
    and Gaussian noise at specific severity levels for evaluation purposes.
    """
    
    # JPEG compression quality levels by severity
    JPEG_QUALITY = {
        1: 95,
        2: 85,
        3: 75,
        4: 65,
        5: 50
    }
    
    # Gaussian blur sigma levels by severity
    BLUR_SIGMA = {
        1: 0.5,
        2: 1.0,
        3: 1.5,
        4: 2.0,
        5: 2.5
    }
    
    # Gaussian noise standard deviation by severity
    NOISE_STD = {
        1: 0.01,
        2: 0.02,
        3: 0.03,
        4: 0.04,
        5: 0.05
    }
    
    @staticmethod
    def apply_jpeg_compression(
        image: torch.Tensor,
        quality: int
    ) -> torch.Tensor:
        """
        Apply JPEG compression at specified quality level.
        
        Args:
            image: Input image tensor (C, H, W) in range [0, 1]
            quality: JPEG quality (1-100, lower = more compression)
        
        Returns:
            Compressed image tensor (C, H, W)
        """
        # Convert tensor to PIL Image
        pil_image = TF.to_pil_image(image)
        
        # Apply JPEG compression by encoding and decoding
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to tensor
        return TF.to_tensor(compressed_image)
    
    @staticmethod
    def apply_gaussian_blur(
        image: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        Apply Gaussian blur at specified sigma level.
        
        Args:
            image: Input image tensor (C, H, W)
            sigma: Gaussian blur sigma
        
        Returns:
            Blurred image tensor (C, H, W)
        """
        # Calculate kernel size (must be odd and large enough for sigma)
        kernel_size = int(2 * (3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian blur
        return TF.gaussian_blur(
            image,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma]
        )
    
    @staticmethod
    def apply_gaussian_noise(
        image: torch.Tensor,
        std: float
    ) -> torch.Tensor:
        """
        Apply Gaussian noise at specified standard deviation.
        
        Args:
            image: Input image tensor (C, H, W)
            std: Gaussian noise standard deviation
        
        Returns:
            Noisy image tensor (C, H, W)
        """
        # Generate Gaussian noise
        noise = torch.randn_like(image) * std
        
        # Add noise and clamp to valid range [0, 1]
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0.0, 1.0)


def evaluate_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    perturbations: Optional[List[str]] = None,
    jpeg_qualities: Optional[List[int]] = None,
    blur_sigmas: Optional[List[float]] = None,
    noise_stds: Optional[List[float]] = None
) -> Dict[str, Dict]:
    """
    Evaluate model robustness against perturbations at multiple severity levels.
    
    This function tests the model's performance under various perturbations:
    - JPEG compression at quality levels [95, 85, 75, 65, 50]
    - Gaussian blur at sigma levels [0.5, 1.0, 1.5, 2.0, 2.5]
    - Gaussian noise at std levels [0.01, 0.02, 0.03, 0.04, 0.05]
    
    Args:
        model: Trained model to evaluate
        test_loader: Test data loader (returns images, labels)
        device: Device to run evaluation on (cuda/cpu)
        perturbations: List of perturbation types to test
                      (default: ['jpeg', 'blur', 'noise'])
        jpeg_qualities: JPEG quality levels to test (default: [95, 85, 75, 65, 50])
        blur_sigmas: Gaussian blur sigma levels to test
                    (default: [0.5, 1.0, 1.5, 2.0, 2.5])
        noise_stds: Gaussian noise std levels to test
                   (default: [0.01, 0.02, 0.03, 0.04, 0.05])
    
    Returns:
        Dictionary mapping perturbation types to results:
        {
            'baseline': {'accuracy': float, 'auc': float},
            'jpeg': {
                95: {'accuracy': float, 'auc': float},
                85: {'accuracy': float, 'auc': float},
                ...
            },
            'blur': {
                0.5: {'accuracy': float, 'auc': float},
                1.0: {'accuracy': float, 'auc': float},
                ...
            },
            'noise': {
                0.01: {'accuracy': float, 'auc': float},
                0.02: {'accuracy': float, 'auc': float},
                ...
            }
        }
    
    Example:
        >>> model = BinaryClassifier(backbone_type='resnet18')
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> device = torch.device('cuda')
        >>> results = evaluate_robustness(model, test_loader, device)
        >>> print(f"Baseline accuracy: {results['baseline']['accuracy']:.4f}")
        >>> print(f"JPEG-75 accuracy: {results['jpeg'][75]['accuracy']:.4f}")
    """
    # Set default perturbations
    if perturbations is None:
        perturbations = ['jpeg', 'blur', 'noise']
    
    # Set default severity levels
    if jpeg_qualities is None:
        jpeg_qualities = [95, 85, 75, 65, 50]
    if blur_sigmas is None:
        blur_sigmas = [0.5, 1.0, 1.5, 2.0, 2.5]
    if noise_stds is None:
        noise_stds = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    model.eval()
    results = {}
    
    # Evaluate baseline (no perturbation)
    print("Evaluating baseline (no perturbation)...")
    baseline_metrics = _evaluate_with_perturbation(
        model, test_loader, device, perturbation_fn=None
    )
    results['baseline'] = baseline_metrics
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}, "
          f"AUC: {baseline_metrics['auc']:.4f}")
    
    # Evaluate JPEG compression
    if 'jpeg' in perturbations:
        print("\nEvaluating JPEG compression...")
        results['jpeg'] = {}
        for quality in jpeg_qualities:
            print(f"  Quality {quality}...", end=' ')
            perturbation_fn = lambda img: RobustnessPerturbation.apply_jpeg_compression(
                img, quality
            )
            metrics = _evaluate_with_perturbation(
                model, test_loader, device, perturbation_fn
            )
            results['jpeg'][quality] = metrics
            print(f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # Evaluate Gaussian blur
    if 'blur' in perturbations:
        print("\nEvaluating Gaussian blur...")
        results['blur'] = {}
        for sigma in blur_sigmas:
            print(f"  Sigma {sigma}...", end=' ')
            perturbation_fn = lambda img, s=sigma: RobustnessPerturbation.apply_gaussian_blur(
                img, s
            )
            metrics = _evaluate_with_perturbation(
                model, test_loader, device, perturbation_fn
            )
            results['blur'][sigma] = metrics
            print(f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # Evaluate Gaussian noise
    if 'noise' in perturbations:
        print("\nEvaluating Gaussian noise...")
        results['noise'] = {}
        for std in noise_stds:
            print(f"  Std {std}...", end=' ')
            perturbation_fn = lambda img, s=std: RobustnessPerturbation.apply_gaussian_noise(
                img, s
            )
            metrics = _evaluate_with_perturbation(
                model, test_loader, device, perturbation_fn
            )
            results['noise'][std] = metrics
            print(f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    return results


def _evaluate_with_perturbation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    perturbation_fn: Optional[callable] = None
) -> Dict[str, float]:
    """
    Evaluate model with optional perturbation applied to images.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        perturbation_fn: Optional function to apply to each image
    
    Returns:
        Dictionary with 'accuracy' and 'auc' metrics
    """
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle both (images, labels) and (images, labels, generator_name) formats
            if len(batch_data) >= 2:
                images, labels = batch_data[0], batch_data[1]
            else:
                raise ValueError("DataLoader must return at least (images, labels)")
            
            # Apply perturbation if provided
            if perturbation_fn is not None:
                # Apply perturbation to each image in batch
                perturbed_images = []
                for img in images:
                    perturbed_img = perturbation_fn(img)
                    perturbed_images.append(perturbed_img)
                images = torch.stack(perturbed_images)
            
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
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # AUC requires at least one sample from each class
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float('nan')
    
    return {
        'accuracy': float(accuracy),
        'auc': float(auc)
    }


def print_robustness_report(results: Dict[str, Dict]) -> None:
    """
    Print formatted robustness evaluation report.
    
    Args:
        results: Dictionary returned by evaluate_robustness()
    
    Example:
        >>> results = evaluate_robustness(model, test_loader, device)
        >>> print_robustness_report(results)
        
        ========================================
        ROBUSTNESS EVALUATION REPORT
        ========================================
        
        Baseline (No Perturbation):
          Accuracy: 95.20%
          AUC:      0.982
        
        JPEG Compression:
        ----------------------------------------
          Quality 95: Accuracy: 94.80%, AUC: 0.978
          Quality 85: Accuracy: 93.50%, AUC: 0.971
          Quality 75: Accuracy: 91.20%, AUC: 0.958
          Quality 65: Accuracy: 88.40%, AUC: 0.941
          Quality 50: Accuracy: 84.10%, AUC: 0.918
        
        Gaussian Blur:
        ----------------------------------------
          Sigma 0.5: Accuracy: 94.50%, AUC: 0.976
          Sigma 1.0: Accuracy: 92.80%, AUC: 0.965
          ...
    """
    print("\n" + "=" * 40)
    print("ROBUSTNESS EVALUATION REPORT")
    print("=" * 40)
    
    # Baseline metrics
    if 'baseline' in results:
        baseline = results['baseline']
        print("\nBaseline (No Perturbation):")
        print(f"  Accuracy: {baseline['accuracy'] * 100:.2f}%")
        if not np.isnan(baseline['auc']):
            print(f"  AUC:      {baseline['auc']:.3f}")
        else:
            print(f"  AUC:      N/A")
    
    # JPEG compression results
    if 'jpeg' in results:
        print("\nJPEG Compression:")
        print("-" * 40)
        for quality in sorted(results['jpeg'].keys(), reverse=True):
            metrics = results['jpeg'][quality]
            auc_str = f"{metrics['auc']:.3f}" if not np.isnan(metrics['auc']) else "N/A"
            print(f"  Quality {quality:2d}: "
                  f"Accuracy: {metrics['accuracy'] * 100:5.2f}%, "
                  f"AUC: {auc_str}")
    
    # Gaussian blur results
    if 'blur' in results:
        print("\nGaussian Blur:")
        print("-" * 40)
        for sigma in sorted(results['blur'].keys()):
            metrics = results['blur'][sigma]
            auc_str = f"{metrics['auc']:.3f}" if not np.isnan(metrics['auc']) else "N/A"
            print(f"  Sigma {sigma:.1f}: "
                  f"Accuracy: {metrics['accuracy'] * 100:5.2f}%, "
                  f"AUC: {auc_str}")
    
    # Gaussian noise results
    if 'noise' in results:
        print("\nGaussian Noise:")
        print("-" * 40)
        for std in sorted(results['noise'].keys()):
            metrics = results['noise'][std]
            auc_str = f"{metrics['auc']:.3f}" if not np.isnan(metrics['auc']) else "N/A"
            print(f"  Std {std:.2f}: "
                  f"Accuracy: {metrics['accuracy'] * 100:5.2f}%, "
                  f"AUC: {auc_str}")
    
    print("\n" + "=" * 40)


def compute_robustness_degradation(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute performance degradation relative to baseline.
    
    Args:
        results: Dictionary returned by evaluate_robustness()
    
    Returns:
        Dictionary mapping perturbation types to degradation metrics:
        {
            'jpeg': {
                95: {'accuracy_drop': 0.004, 'auc_drop': 0.004},
                85: {'accuracy_drop': 0.017, 'auc_drop': 0.011},
                ...
            },
            ...
        }
    """
    if 'baseline' not in results:
        raise ValueError("Results must contain 'baseline' metrics")
    
    baseline_acc = results['baseline']['accuracy']
    baseline_auc = results['baseline']['auc']
    
    degradation = {}
    
    for pert_type in ['jpeg', 'blur', 'noise']:
        if pert_type in results:
            degradation[pert_type] = {}
            for level, metrics in results[pert_type].items():
                acc_drop = baseline_acc - metrics['accuracy']
                auc_drop = baseline_auc - metrics['auc'] if not np.isnan(metrics['auc']) else float('nan')
                degradation[pert_type][level] = {
                    'accuracy_drop': float(acc_drop),
                    'auc_drop': float(auc_drop)
                }
    
    return degradation
