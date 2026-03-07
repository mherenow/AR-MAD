"""
Unit tests for comprehensive evaluation runner.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .comprehensive_eval import (
    ComprehensiveEvaluator,
    run_comprehensive_evaluation
)


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, has_spectral=False, has_noise=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        
        if has_spectral:
            self.spectral_branch = nn.Linear(64, 256)
        
        if has_noise:
            self.noise_extractor = nn.Identity()
            self.noise_branch = nn.Linear(64, 256)
    
    def forward(self, x):
        # Simple forward pass
        return torch.sigmoid(torch.randn(x.shape[0], 1))


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return MockModel(has_spectral=True, has_noise=True)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')


@pytest.fixture
def test_loader():
    """Create a test data loader."""
    images = torch.randn(20, 3, 64, 64)
    labels = torch.randint(0, 2, (20,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def dataset_loaders(test_loader):
    """Create dataset loaders dictionary."""
    return {
        'dataset1': test_loader,
        'dataset2': test_loader
    }


@pytest.fixture
def sample_images():
    """Create sample images for visualization."""
    return torch.randn(5, 3, 64, 64)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestComprehensiveEvaluator:
    """Tests for ComprehensiveEvaluator class."""
    
    def test_initialization(self, mock_model, device, temp_output_dir):
        """Test evaluator initialization."""
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir,
            run_name='test_run'
        )
        
        assert evaluator.model is mock_model
        assert evaluator.device == device
        assert evaluator.run_name == 'test_run'
        assert evaluator.run_dir.exists()
        assert (evaluator.run_dir / 'visualizations').exists()
        assert (evaluator.run_dir / 'metrics').exists()
    
    def test_initialization_auto_run_name(self, mock_model, device, temp_output_dir):
        """Test evaluator initialization with auto-generated run name."""
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        # Run name should be timestamp format
        assert len(evaluator.run_name) > 0
        assert evaluator.run_dir.exists()
    
    @patch('evaluation.comprehensive_eval.evaluate_robustness')
    def test_run_robustness_only(self, mock_eval_robustness, mock_model, device, test_loader, temp_output_dir):
        """Test running only robustness evaluation."""
        # Mock robustness evaluation
        mock_eval_robustness.return_value = {
            'baseline': {'accuracy': 0.95, 'auc': 0.98},
            'jpeg': {95: {'accuracy': 0.94, 'auc': 0.97}}
        }
        
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            test_loader=test_loader,
            enable_robustness=True,
            enable_spectral_viz=False,
            enable_noise_clustering=False,
            enable_cross_dataset=False,
            enable_resolution=False
        )
        
        assert 'robustness' in results
        assert 'metadata' in results
        assert results['robustness']['baseline']['accuracy'] == 0.95
        mock_eval_robustness.assert_called_once()
    
    @patch('evaluation.comprehensive_eval.visualize_spectral_artifacts')
    @patch('evaluation.comprehensive_eval.check_gradcam_availability')
    def test_run_spectral_viz_only(
        self,
        mock_check_gradcam,
        mock_viz,
        mock_model,
        device,
        sample_images,
        temp_output_dir
    ):
        """Test running only spectral visualization."""
        # Mock GradCAM availability and visualization
        mock_check_gradcam.return_value = True
        mock_viz.return_value = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            sample_images=sample_images,
            enable_robustness=False,
            enable_spectral_viz=True,
            enable_noise_clustering=False,
            enable_cross_dataset=False,
            enable_resolution=False
        )
        
        assert 'spectral_viz' in results
        assert 'num_visualizations' in results['spectral_viz']
        assert results['spectral_viz']['num_visualizations'] == 5
        mock_viz.assert_called_once()
    
    @patch('evaluation.comprehensive_eval.check_gradcam_availability')
    def test_spectral_viz_no_gradcam(
        self,
        mock_check_gradcam,
        mock_model,
        device,
        sample_images,
        temp_output_dir
    ):
        """Test spectral visualization when GradCAM is not available."""
        mock_check_gradcam.return_value = False
        
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            sample_images=sample_images,
            enable_robustness=False,
            enable_spectral_viz=True,
            enable_noise_clustering=False,
            enable_cross_dataset=False,
            enable_resolution=False
        )
        
        assert 'spectral_viz' in results
        assert 'error' in results['spectral_viz']
    
    def test_spectral_viz_no_spectral_branch(self, device, sample_images, temp_output_dir):
        """Test spectral visualization when model has no spectral branch."""
        model_no_spectral = MockModel(has_spectral=False, has_noise=False)
        
        evaluator = ComprehensiveEvaluator(
            model=model_no_spectral,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            sample_images=sample_images,
            enable_robustness=False,
            enable_spectral_viz=True,
            enable_noise_clustering=False,
            enable_cross_dataset=False,
            enable_resolution=False
        )
        
        assert 'spectral_viz' in results
        assert 'error' in results['spectral_viz']
    
    @patch('evaluation.comprehensive_eval.evaluate_noise_imprint_clustering')
    def test_run_noise_clustering_only(
        self,
        mock_eval_clustering,
        mock_model,
        device,
        test_loader,
        temp_output_dir
    ):
        """Test running only noise clustering analysis."""
        mock_eval_clustering.return_value = {
            'silhouette_score': 0.65,
            'davies_bouldin_index': 0.82,
            'num_samples': 100,
            'num_generators': 5
        }
        
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            test_loader=test_loader,
            enable_robustness=False,
            enable_spectral_viz=False,
            enable_noise_clustering=True,
            enable_cross_dataset=False,
            enable_resolution=False
        )
        
        assert 'noise_clustering' in results
        assert results['noise_clustering']['silhouette_score'] == 0.65
        mock_eval_clustering.assert_called_once()
    
    def test_noise_clustering_no_noise_branch(self, device, test_loader, temp_output_dir):
        """Test noise clustering when model has no noise branch."""
        model_no_noise = MockModel(has_spectral=False, has_noise=False)
        
        evaluator = ComprehensiveEvaluator(
            model=model_no_noise,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            test_loader=test_loader,
            enable_robustness=False,
            enable_spectral_viz=False,
            enable_noise_clustering=True,
            enable_cross_dataset=False,
            enable_resolution=False
        )
        
        assert 'noise_clustering' in results
        assert 'error' in results['noise_clustering']
    
    @patch('evaluation.comprehensive_eval.evaluate_cross_dataset')
    def test_run_cross_dataset_only(
        self,
        mock_eval_cross,
        mock_model,
        device,
        dataset_loaders,
        temp_output_dir
    ):
        """Test running only cross-dataset evaluation."""
        mock_eval_cross.return_value = {
            'dataset1': {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.96, 'f1': 0.95, 'auc': 0.98, 'num_samples': 100},
            'dataset2': {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.94, 'f1': 0.93, 'auc': 0.96, 'num_samples': 100}
        }
        
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            dataset_loaders=dataset_loaders,
            enable_robustness=False,
            enable_spectral_viz=False,
            enable_noise_clustering=False,
            enable_cross_dataset=True,
            enable_resolution=False
        )
        
        assert 'cross_dataset' in results
        assert 'dataset1' in results['cross_dataset']
        assert 'dataset2' in results['cross_dataset']
        mock_eval_cross.assert_called_once()
    
    @patch('evaluation.comprehensive_eval.evaluate_any_resolution')
    def test_run_resolution_only(
        self,
        mock_eval_resolution,
        mock_model,
        device,
        test_loader,
        temp_output_dir
    ):
        """Test running only any-resolution evaluation."""
        mock_eval_resolution.return_value = {
            '128-256': {'accuracy': 0.94, 'precision': 0.93, 'recall': 0.95, 'f1': 0.94, 'auc': 0.97, 'num_samples': 50, 'avg_height': 192, 'avg_width': 190},
            '256-512': {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.96, 'f1': 0.95, 'auc': 0.98, 'num_samples': 50, 'avg_height': 384, 'avg_width': 380}
        }
        
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            test_loader=test_loader,
            enable_robustness=False,
            enable_spectral_viz=False,
            enable_noise_clustering=False,
            enable_cross_dataset=False,
            enable_resolution=True
        )
        
        assert 'resolution' in results
        assert '128-256' in results['resolution']
        assert '256-512' in results['resolution']
        mock_eval_resolution.assert_called_once()
    
    def test_save_results(self, mock_model, device, temp_output_dir):
        """Test saving results to JSON."""
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = {
            'metadata': {'run_name': 'test', 'timestamp': '2024-01-01'},
            'robustness': {'baseline': {'accuracy': 0.95}}
        }
        
        output_path = evaluator.save_results(results)
        
        assert Path(output_path).exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['metadata']['run_name'] == 'test'
        assert loaded_results['robustness']['baseline']['accuracy'] == 0.95
    
    def test_save_results_with_numpy(self, mock_model, device, temp_output_dir):
        """Test saving results with numpy types."""
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = {
            'metadata': {'run_name': 'test'},
            'robustness': {
                'baseline': {
                    'accuracy': np.float64(0.95),
                    'num_samples': np.int64(100),
                    'confusion_matrix': np.array([[45, 5], [3, 47]])
                }
            }
        }
        
        output_path = evaluator.save_results(results)
        
        assert Path(output_path).exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['robustness']['baseline']['accuracy'] == 0.95
        assert loaded_results['robustness']['baseline']['num_samples'] == 100
        assert loaded_results['robustness']['baseline']['confusion_matrix'] == [[45, 5], [3, 47]]
    
    def test_generate_report(self, mock_model, device, temp_output_dir):
        """Test generating text report."""
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = {
            'metadata': {'run_name': 'test', 'timestamp': '2024-01-01', 'device': 'cpu'},
            'robustness': {
                'baseline': {'accuracy': 0.95, 'auc': 0.98},
                'jpeg': {95: {'accuracy': 0.94, 'auc': 0.97}}
            },
            'noise_clustering': {
                'silhouette_score': 0.65,
                'davies_bouldin_index': 0.82,
                'num_samples': 100,
                'num_generators': 5
            }
        }
        
        output_path = evaluator.generate_report(results)
        
        assert Path(output_path).exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert 'COMPREHENSIVE EVALUATION REPORT' in content
        assert 'Metadata:' in content
        assert 'Robustness Evaluation:' in content
        assert 'Noise Imprint Clustering:' in content
        assert 'SUMMARY' in content
    
    def test_generate_report_with_errors(self, mock_model, device, temp_output_dir):
        """Test generating report with failed evaluations."""
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = {
            'metadata': {'run_name': 'test'},
            'robustness': {'baseline': {'accuracy': 0.95}},
            'spectral_viz': {'error': 'pytorch-grad-cam not installed'},
            'noise_clustering': {'error': 'No noise branch in model'}
        }
        
        output_path = evaluator.generate_report(results)
        
        assert Path(output_path).exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert 'Failed:' in content
        assert 'spectral_viz' in content
        assert 'noise_clustering' in content


class TestRunComprehensiveEvaluation:
    """Tests for run_comprehensive_evaluation convenience function."""
    
    @patch('evaluation.comprehensive_eval.ComprehensiveEvaluator')
    def test_run_comprehensive_evaluation(
        self,
        mock_evaluator_class,
        mock_model,
        device,
        test_loader,
        temp_output_dir
    ):
        """Test convenience function."""
        # Mock evaluator instance
        mock_evaluator = MagicMock()
        mock_evaluator.run_all_evaluations.return_value = {
            'metadata': {'run_name': 'test'},
            'robustness': {'baseline': {'accuracy': 0.95}}
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        results = run_comprehensive_evaluation(
            model=mock_model,
            device=device,
            test_loader=test_loader,
            output_dir=temp_output_dir
        )
        
        # Verify evaluator was created
        mock_evaluator_class.assert_called_once_with(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir,
            run_name=None
        )
        
        # Verify methods were called
        mock_evaluator.run_all_evaluations.assert_called_once()
        mock_evaluator.save_results.assert_called_once()
        mock_evaluator.generate_report.assert_called_once()
        
        assert 'robustness' in results


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_results(self, mock_model, device, temp_output_dir):
        """Test handling empty results."""
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        results = evaluator.run_all_evaluations(
            enable_robustness=False,
            enable_spectral_viz=False,
            enable_noise_clustering=False,
            enable_cross_dataset=False,
            enable_resolution=False
        )
        
        # Should only have metadata
        assert 'metadata' in results
        assert len(results) == 1
    
    def test_all_evaluations_skipped(self, mock_model, device, temp_output_dir):
        """Test when all evaluations are skipped due to missing data."""
        evaluator = ComprehensiveEvaluator(
            model=mock_model,
            device=device,
            output_dir=temp_output_dir
        )
        
        # No data loaders or images provided
        results = evaluator.run_all_evaluations()
        
        # Should only have metadata
        assert 'metadata' in results
        assert 'robustness' not in results
        assert 'spectral_viz' not in results
