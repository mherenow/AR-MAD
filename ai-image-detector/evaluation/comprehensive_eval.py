"""
Comprehensive evaluation runner for orchestrating all evaluation modules.

This module provides a unified interface to run all evaluation types:
- Robustness evaluation (JPEG, blur, noise perturbations)
- Spectral artifact visualization (GradCAM)
- Noise imprint clustering analysis
- Cross-dataset performance evaluation
- Any-resolution capability testing

Results are saved to JSON files and visualizations are saved as images.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# Import evaluation modules
from .robustness_eval import (
    evaluate_robustness,
    print_robustness_report,
    compute_robustness_degradation
)
from .spectral_viz import (
    visualize_spectral_artifacts,
    check_gradcam_availability
)
from .noise_clustering import (
    evaluate_noise_imprint_clustering,
    print_clustering_report
)
from .cross_dataset_eval import (
    evaluate_cross_dataset,
    print_cross_dataset_report,
    generate_performance_matrix,
    compute_cross_dataset_variance
)
from .resolution_eval import (
    evaluate_any_resolution,
    print_resolution_report,
    generate_size_performance_matrix,
    compute_size_variance
)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation orchestrator for ML detector models.
    
    This class coordinates all evaluation types and generates a unified
    evaluation report with metrics, visualizations, and analysis.
    
    Args:
        model: Trained model to evaluate
        device: Device to run evaluation on (cuda/cpu)
        output_dir: Directory to save results (default: 'evaluation_results')
        run_name: Optional name for this evaluation run (default: timestamp)
    
    Example:
        >>> model = BinaryClassifier(use_spectral=True, use_noise_imprint=True)
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> device = torch.device('cuda')
        >>> evaluator = ComprehensiveEvaluator(model, device)
        >>> 
        >>> # Run all evaluations
        >>> results = evaluator.run_all_evaluations(
        ...     test_loader=test_loader,
        ...     dataset_loaders={'synthbuster': sb_loader, 'coco': coco_loader},
        ...     sample_images=sample_images
        ... )
        >>> 
        >>> # Save results
        >>> evaluator.save_results(results)
        >>> evaluator.generate_report(results)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = 'evaluation_results',
        run_name: Optional[str] = None
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Generate run name if not provided
        if run_name is None:
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = run_name
        
        # Create output directory
        self.run_dir = self.output_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / 'visualizations').mkdir(exist_ok=True)
        (self.run_dir / 'metrics').mkdir(exist_ok=True)
        
        print(f"Comprehensive Evaluator initialized")
        print(f"Output directory: {self.run_dir}")
    
    def run_all_evaluations(
        self,
        test_loader: Optional[DataLoader] = None,
        dataset_loaders: Optional[Dict[str, DataLoader]] = None,
        sample_images: Optional[torch.Tensor] = None,
        enable_robustness: bool = True,
        enable_spectral_viz: bool = True,
        enable_noise_clustering: bool = True,
        enable_cross_dataset: bool = True,
        enable_resolution: bool = True,
        generator_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run all enabled evaluations and collect results.
        
        Args:
            test_loader: Test data loader for robustness and noise clustering
            dataset_loaders: Dictionary of dataset-specific loaders for cross-dataset eval
            sample_images: Sample images for spectral visualization (B, 3, H, W)
            enable_robustness: Whether to run robustness evaluation
            enable_spectral_viz: Whether to run spectral visualization
            enable_noise_clustering: Whether to run noise clustering analysis
            enable_cross_dataset: Whether to run cross-dataset evaluation
            enable_resolution: Whether to run any-resolution evaluation
            generator_labels: Optional list of generator names for clustering
        
        Returns:
            Dictionary containing all evaluation results:
            {
                'robustness': {...},
                'spectral_viz': {...},
                'noise_clustering': {...},
                'cross_dataset': {...},
                'resolution': {...},
                'metadata': {...}
            }
        """
        results = {
            'metadata': {
                'run_name': self.run_name,
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device)
            }
        }
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # 1. Robustness Evaluation
        if enable_robustness and test_loader is not None:
            print("\n[1/5] Running robustness evaluation...")
            try:
                robustness_results = evaluate_robustness(
                    self.model,
                    test_loader,
                    self.device
                )
                results['robustness'] = robustness_results
                print_robustness_report(robustness_results)
                
                # Compute degradation metrics
                degradation = compute_robustness_degradation(robustness_results)
                results['robustness_degradation'] = degradation
                
                print("✓ Robustness evaluation completed")
            except Exception as e:
                print(f"✗ Robustness evaluation failed: {e}")
                results['robustness'] = {'error': str(e)}
        else:
            print("\n[1/5] Robustness evaluation skipped")
        
        # 2. Spectral Artifact Visualization
        if enable_spectral_viz and sample_images is not None:
            print("\n[2/5] Running spectral artifact visualization...")
            try:
                if not check_gradcam_availability():
                    print("✗ pytorch-grad-cam not available, skipping visualization")
                    results['spectral_viz'] = {'error': 'pytorch-grad-cam not installed'}
                elif not hasattr(self.model, 'spectral_branch'):
                    print("✗ Model does not have spectral branch, skipping visualization")
                    results['spectral_viz'] = {'error': 'No spectral branch in model'}
                else:
                    overlays = visualize_spectral_artifacts(
                        self.model,
                        sample_images,
                        device=self.device
                    )
                    
                    if overlays is not None:
                        # Save visualizations
                        viz_paths = self._save_spectral_visualizations(overlays)
                        results['spectral_viz'] = {
                            'num_visualizations': len(viz_paths),
                            'visualization_paths': viz_paths
                        }
                        print(f"✓ Spectral visualization completed ({len(viz_paths)} images saved)")
                    else:
                        results['spectral_viz'] = {'error': 'Visualization failed'}
            except Exception as e:
                print(f"✗ Spectral visualization failed: {e}")
                results['spectral_viz'] = {'error': str(e)}
        else:
            print("\n[2/5] Spectral visualization skipped")
        
        # 3. Noise Imprint Clustering
        if enable_noise_clustering and test_loader is not None:
            print("\n[3/5] Running noise imprint clustering analysis...")
            try:
                if not hasattr(self.model, 'noise_extractor') or not hasattr(self.model, 'noise_branch'):
                    print("✗ Model does not have noise imprint branch, skipping clustering")
                    results['noise_clustering'] = {'error': 'No noise imprint branch in model'}
                else:
                    clustering_results = evaluate_noise_imprint_clustering(
                        self.model,
                        test_loader,
                        self.device,
                        generator_labels=generator_labels
                    )
                    results['noise_clustering'] = clustering_results
                    print_clustering_report(clustering_results)
                    print("✓ Noise clustering analysis completed")
            except Exception as e:
                print(f"✗ Noise clustering analysis failed: {e}")
                results['noise_clustering'] = {'error': str(e)}
        else:
            print("\n[3/5] Noise clustering analysis skipped")
        
        # 4. Cross-Dataset Evaluation
        if enable_cross_dataset and dataset_loaders is not None:
            print("\n[4/5] Running cross-dataset evaluation...")
            try:
                cross_dataset_results = evaluate_cross_dataset(
                    self.model,
                    dataset_loaders,
                    self.device
                )
                results['cross_dataset'] = cross_dataset_results
                print_cross_dataset_report(cross_dataset_results)
                
                # Compute performance matrix and variance
                perf_matrix = generate_performance_matrix(cross_dataset_results)
                variance = compute_cross_dataset_variance(cross_dataset_results)
                results['cross_dataset_matrix'] = perf_matrix
                results['cross_dataset_variance'] = variance
                
                print("✓ Cross-dataset evaluation completed")
            except Exception as e:
                print(f"✗ Cross-dataset evaluation failed: {e}")
                results['cross_dataset'] = {'error': str(e)}
        else:
            print("\n[4/5] Cross-dataset evaluation skipped")
        
        # 5. Any-Resolution Evaluation
        if enable_resolution and test_loader is not None:
            print("\n[5/5] Running any-resolution evaluation...")
            try:
                resolution_results = evaluate_any_resolution(
                    self.model,
                    test_loader,
                    self.device
                )
                results['resolution'] = resolution_results
                print_resolution_report(resolution_results)
                
                # Compute size performance matrix and variance
                size_matrix = generate_size_performance_matrix(resolution_results)
                size_variance = compute_size_variance(resolution_results)
                results['resolution_matrix'] = size_matrix
                results['resolution_variance'] = size_variance
                
                print("✓ Any-resolution evaluation completed")
            except Exception as e:
                print(f"✗ Any-resolution evaluation failed: {e}")
                results['resolution'] = {'error': str(e)}
        else:
            print("\n[5/5] Any-resolution evaluation skipped")
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION COMPLETED")
        print("=" * 60)
        
        return results
    
    def _save_spectral_visualizations(
        self,
        overlays: np.ndarray
    ) -> List[str]:
        """
        Save spectral visualization overlays as images.
        
        Args:
            overlays: Array of overlay images (B, H, W, 3)
        
        Returns:
            List of saved file paths
        """
        viz_dir = self.run_dir / 'visualizations'
        saved_paths = []
        
        for i, overlay in enumerate(overlays):
            filename = f'spectral_viz_{i:03d}.png'
            filepath = viz_dir / filename
            
            # Convert to PIL Image and save
            img = Image.fromarray(overlay.astype(np.uint8))
            img.save(filepath)
            
            saved_paths.append(str(filepath))
        
        return saved_paths
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str = 'comprehensive_results.json'
    ) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Dictionary returned by run_all_evaluations()
            filename: Output filename (default: 'comprehensive_results.json')
        
        Returns:
            Path to saved JSON file
        """
        output_path = self.run_dir / 'metrics' / filename
        
        # Convert numpy types to Python types for JSON serialization
        results_serializable = self._make_json_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return str(output_path)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to convert
        
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return obj
    
    def generate_report(
        self,
        results: Dict[str, Any],
        filename: str = 'evaluation_report.txt'
    ) -> str:
        """
        Generate a comprehensive text report summarizing all evaluations.
        
        Args:
            results: Dictionary returned by run_all_evaluations()
            filename: Output filename (default: 'evaluation_report.txt')
        
        Returns:
            Path to saved report file
        """
        output_path = self.run_dir / filename
        
        with open(output_path, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Write metadata
            f.write("Metadata:\n")
            f.write("-" * 80 + "\n")
            for key, value in results.get('metadata', {}).items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Write robustness results
            if 'robustness' in results and 'error' not in results['robustness']:
                f.write("Robustness Evaluation:\n")
                f.write("-" * 80 + "\n")
                self._write_robustness_section(f, results['robustness'])
                f.write("\n")
            
            # Write spectral visualization results
            if 'spectral_viz' in results:
                f.write("Spectral Artifact Visualization:\n")
                f.write("-" * 80 + "\n")
                if 'error' in results['spectral_viz']:
                    f.write(f"  Error: {results['spectral_viz']['error']}\n")
                else:
                    f.write(f"  Visualizations generated: {results['spectral_viz']['num_visualizations']}\n")
                    f.write(f"  Saved to: visualizations/\n")
                f.write("\n")
            
            # Write noise clustering results
            if 'noise_clustering' in results and 'error' not in results['noise_clustering']:
                f.write("Noise Imprint Clustering:\n")
                f.write("-" * 80 + "\n")
                self._write_clustering_section(f, results['noise_clustering'])
                f.write("\n")
            
            # Write cross-dataset results
            if 'cross_dataset' in results and 'error' not in results['cross_dataset']:
                f.write("Cross-Dataset Evaluation:\n")
                f.write("-" * 80 + "\n")
                self._write_cross_dataset_section(f, results['cross_dataset'])
                f.write("\n")
            
            # Write resolution results
            if 'resolution' in results and 'error' not in results['resolution']:
                f.write("Any-Resolution Evaluation:\n")
                f.write("-" * 80 + "\n")
                self._write_resolution_section(f, results['resolution'])
                f.write("\n")
            
            # Write summary
            f.write("=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n")
            self._write_summary_section(f, results)
        
        print(f"Report saved to: {output_path}")
        return str(output_path)
    
    def _write_robustness_section(self, f, robustness_results: Dict) -> None:
        """Write robustness evaluation section to report."""
        baseline = robustness_results.get('baseline', {})
        f.write(f"  Baseline Accuracy: {baseline.get('accuracy', 0) * 100:.2f}%\n")
        f.write(f"  Baseline AUC: {baseline.get('auc', 0):.3f}\n\n")
        
        for pert_type in ['jpeg', 'blur', 'noise']:
            if pert_type in robustness_results:
                f.write(f"  {pert_type.upper()} Perturbation:\n")
                for level, metrics in sorted(robustness_results[pert_type].items()):
                    f.write(f"    Level {level}: Accuracy={metrics['accuracy']*100:.2f}%, ")
                    f.write(f"AUC={metrics['auc']:.3f}\n")
                f.write("\n")
    
    def _write_clustering_section(self, f, clustering_results: Dict) -> None:
        """Write noise clustering section to report."""
        f.write(f"  Samples: {clustering_results.get('num_samples', 0)}\n")
        f.write(f"  Generators: {clustering_results.get('num_generators', 0)}\n")
        f.write(f"  Silhouette Score: {clustering_results.get('silhouette_score', 0):.4f}\n")
        f.write(f"  Davies-Bouldin Index: {clustering_results.get('davies_bouldin_index', 0):.4f}\n")
    
    def _write_cross_dataset_section(self, f, cross_dataset_results: Dict) -> None:
        """Write cross-dataset evaluation section to report."""
        for dataset_name, metrics in cross_dataset_results.items():
            f.write(f"  {dataset_name}:\n")
            f.write(f"    Accuracy:  {metrics.get('accuracy', 0) * 100:.2f}%\n")
            f.write(f"    Precision: {metrics.get('precision', 0) * 100:.2f}%\n")
            f.write(f"    Recall:    {metrics.get('recall', 0) * 100:.2f}%\n")
            f.write(f"    F1 Score:  {metrics.get('f1', 0) * 100:.2f}%\n")
            f.write(f"    Samples:   {metrics.get('num_samples', 0)}\n\n")
    
    def _write_resolution_section(self, f, resolution_results: Dict) -> None:
        """Write any-resolution evaluation section to report."""
        for size_range, metrics in resolution_results.items():
            f.write(f"  {size_range}:\n")
            f.write(f"    Samples:   {metrics.get('num_samples', 0)}\n")
            f.write(f"    Avg Size:  {metrics.get('avg_height', 0):.1f} x {metrics.get('avg_width', 0):.1f}\n")
            f.write(f"    Accuracy:  {metrics.get('accuracy', 0) * 100:.2f}%\n")
            f.write(f"    F1 Score:  {metrics.get('f1', 0) * 100:.2f}%\n\n")
    
    def _write_summary_section(self, f, results: Dict) -> None:
        """Write summary section to report."""
        completed = []
        failed = []
        
        for eval_type in ['robustness', 'spectral_viz', 'noise_clustering', 'cross_dataset', 'resolution']:
            if eval_type in results:
                if 'error' in results[eval_type]:
                    failed.append(eval_type)
                else:
                    completed.append(eval_type)
        
        f.write(f"  Completed evaluations: {len(completed)}\n")
        f.write(f"  Failed evaluations: {len(failed)}\n\n")
        
        if completed:
            f.write("  Completed:\n")
            for eval_type in completed:
                f.write(f"    - {eval_type}\n")
        
        if failed:
            f.write("\n  Failed:\n")
            for eval_type in failed:
                f.write(f"    - {eval_type}: {results[eval_type]['error']}\n")


def run_comprehensive_evaluation(
    model: nn.Module,
    device: torch.device,
    test_loader: Optional[DataLoader] = None,
    dataset_loaders: Optional[Dict[str, DataLoader]] = None,
    sample_images: Optional[torch.Tensor] = None,
    output_dir: str = 'evaluation_results',
    run_name: Optional[str] = None,
    generator_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive evaluation with default settings.
    
    This function provides a simple interface to run all evaluations and
    save results without manually creating a ComprehensiveEvaluator instance.
    
    Args:
        model: Trained model to evaluate
        device: Device to run evaluation on
        test_loader: Test data loader for robustness and noise clustering
        dataset_loaders: Dictionary of dataset-specific loaders
        sample_images: Sample images for spectral visualization
        output_dir: Directory to save results
        run_name: Optional name for this evaluation run
        generator_labels: Optional list of generator names
    
    Returns:
        Dictionary containing all evaluation results
    
    Example:
        >>> model = BinaryClassifier(use_spectral=True, use_noise_imprint=True)
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> device = torch.device('cuda')
        >>> 
        >>> results = run_comprehensive_evaluation(
        ...     model=model,
        ...     device=device,
        ...     test_loader=test_loader,
        ...     dataset_loaders={'synthbuster': sb_loader, 'coco': coco_loader},
        ...     sample_images=sample_images[:10]
        ... )
    """
    evaluator = ComprehensiveEvaluator(
        model=model,
        device=device,
        output_dir=output_dir,
        run_name=run_name
    )
    
    results = evaluator.run_all_evaluations(
        test_loader=test_loader,
        dataset_loaders=dataset_loaders,
        sample_images=sample_images,
        generator_labels=generator_labels
    )
    
    # Save results and generate report
    evaluator.save_results(results)
    evaluator.generate_report(results)
    
    return results
