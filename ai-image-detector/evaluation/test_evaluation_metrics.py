"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
from .evaluate import (
    compute_per_generator_metrics,
    print_evaluation_report
)


class TestAccuracyComputation:
    """Test accuracy computation with known predictions and labels."""
    
    def test_perfect_accuracy(self):
        """Test accuracy when all predictions are correct."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
        generator_labels = ['gen1'] * 6
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert 'gen1' in metrics
        assert metrics['gen1']['accuracy'] == 1.0
    
    def test_zero_accuracy(self):
        """Test accuracy when all predictions are wrong."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        generator_labels = ['gen1'] * 4
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert metrics['gen1']['accuracy'] == 0.0
    
    def test_partial_accuracy(self):
        """Test accuracy with mixed correct and incorrect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        y_prob = np.array([0.2, 0.6, 0.8, 0.3, 0.1, 0.9, 0.4, 0.7])
        generator_labels = ['gen1'] * 8
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        # Correct predictions at indices: 0 (0==0), 4 (0==0), 5 (1==1) = 3 wrong, 4 correct
        # Actually: 0✓, 1✗, 2✓, 3✗, 4✓, 5✓, 6✗, 7✗ = 4 correct out of 8
        expected_accuracy = 4.0 / 8.0
        assert abs(metrics['gen1']['accuracy'] - expected_accuracy) < 1e-6
    
    def test_accuracy_with_single_sample(self):
        """Test accuracy computation with a single sample."""
        y_true = np.array([1])
        y_pred = np.array([1])
        y_prob = np.array([0.9])
        generator_labels = ['gen1']
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert metrics['gen1']['accuracy'] == 1.0


class TestAUCComputation:
    """Test AUC computation with known scores."""
    
    def test_perfect_auc(self):
        """Test AUC when model perfectly separates classes."""
        # All real images have low probabilities, all AI images have high probabilities
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        generator_labels = ['gen1'] * 6
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert metrics['gen1']['auc'] == 1.0
    
    def test_worst_auc(self):
        """Test AUC when model predictions are completely reversed."""
        # All real images have high probabilities, all AI images have low probabilities
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        generator_labels = ['gen1'] * 6
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert metrics['gen1']['auc'] == 0.0
    
    def test_random_auc(self):
        """Test AUC close to 0.5 for random predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        y_prob = np.array([0.45, 0.55, 0.52, 0.48, 0.49, 0.51, 0.53, 0.47])
        generator_labels = ['gen1'] * 8
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        # Should be close to 0.5 for random-like predictions
        assert 0.3 < metrics['gen1']['auc'] < 0.7
    
    def test_auc_with_single_class(self):
        """Test AUC returns NaN when only one class is present."""
        # Only class 0 (real images)
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.6, 0.3])
        generator_labels = ['gen1'] * 4
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert np.isnan(metrics['gen1']['auc'])
    
    def test_auc_with_only_positive_class(self):
        """Test AUC returns NaN when only positive class is present."""
        # Only class 1 (AI-generated images)
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.4, 0.7])
        generator_labels = ['gen1'] * 4
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert np.isnan(metrics['gen1']['auc'])
    
    def test_auc_intermediate_values(self):
        """Test AUC with known intermediate values."""
        # Create a scenario with known AUC
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        y_pred = (y_prob > 0.5).astype(int)
        generator_labels = ['gen1'] * 8
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        # Perfect separation with threshold 0.5
        assert metrics['gen1']['auc'] == 1.0


class TestPerGeneratorMetricAggregation:
    """Test per-generator metric aggregation."""
    
    def test_multiple_generators(self):
        """Test metrics are computed separately for each generator."""
        # Generator 1: perfect accuracy
        # Generator 2: 50% accuracy
        y_true = np.array([0, 0, 1, 1,  0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1,  1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8,  0.6, 0.7, 0.3, 0.4])
        generator_labels = ['gen1', 'gen1', 'gen1', 'gen1', 
                           'gen2', 'gen2', 'gen2', 'gen2']
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert 'gen1' in metrics
        assert 'gen2' in metrics
        assert metrics['gen1']['accuracy'] == 1.0
        assert metrics['gen2']['accuracy'] == 0.0
    
    def test_three_generators_different_performance(self):
        """Test with three generators having different performance levels."""
        y_true = np.array([0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1,  0, 0, 1, 1,  1, 1, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8,  
                          0.3, 0.4, 0.6, 0.7,  
                          0.5, 0.9, 0.8, 0.95])
        generator_labels = ['dalle', 'dalle', 'dalle', 'dalle',
                           'midjourney', 'midjourney', 'midjourney', 'midjourney',
                           'stable_diffusion', 'stable_diffusion', 'stable_diffusion', 'stable_diffusion']
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert len(metrics) == 3
        assert 'dalle' in metrics
        assert 'midjourney' in metrics
        assert 'stable_diffusion' in metrics
        
        # DALL-E: perfect (4/4) - indices 0-3: all match
        assert metrics['dalle']['accuracy'] == 1.0
        
        # Midjourney: 50% (2/4) - indices 4-7: 4✓, 5✗, 6✗, 7✓ = 2 correct
        assert metrics['midjourney']['accuracy'] == 0.5
        
        # Stable Diffusion: 50% (2/4) - indices 8-11: 8✗, 9✓, 10✗, 11✓ = 2 correct
        assert metrics['stable_diffusion']['accuracy'] == 0.5
    
    def test_generator_with_unbalanced_samples(self):
        """Test generators with different numbers of samples."""
        y_true = np.array([0, 1, 0, 1, 0, 1,  1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1,  1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7,  0.9, 0.8])
        generator_labels = ['gen1', 'gen1', 'gen1', 'gen1', 'gen1', 'gen1',
                           'gen2', 'gen2']
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert metrics['gen1']['accuracy'] == 1.0
        assert metrics['gen2']['accuracy'] == 1.0
    
    def test_generator_names_with_special_characters(self):
        """Test generator names with special characters and spaces."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        generator_labels = ['DALL-E 2', 'DALL-E 2', 'Stable Diffusion v1.5', 'Stable Diffusion v1.5']
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert 'DALL-E 2' in metrics
        assert 'Stable Diffusion v1.5' in metrics
        assert metrics['DALL-E 2']['accuracy'] == 1.0
        assert metrics['Stable Diffusion v1.5']['accuracy'] == 1.0
    
    def test_single_generator(self):
        """Test with only one generator (edge case)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
        generator_labels = ['only_gen'] * 6
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert len(metrics) == 1
        assert 'only_gen' in metrics
        assert metrics['only_gen']['accuracy'] == 1.0
    
    def test_empty_generator_name(self):
        """Test with empty string as generator name."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_prob = np.array([0.2, 0.8])
        generator_labels = ['', '']
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert '' in metrics
        assert metrics['']['accuracy'] == 1.0


class TestMetricsDataTypes:
    """Test that metrics return correct data types."""
    
    def test_metrics_return_float_types(self):
        """Test that all metrics are returned as Python floats."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        generator_labels = ['gen1'] * 4
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert isinstance(metrics['gen1']['accuracy'], float)
        assert isinstance(metrics['gen1']['auc'], float)
    
    def test_nan_is_float(self):
        """Test that NaN AUC is still a float type."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3])
        generator_labels = ['gen1'] * 3
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert isinstance(metrics['gen1']['auc'], float)
        assert np.isnan(metrics['gen1']['auc'])


class TestPrintEvaluationReport:
    """Test the print_evaluation_report function."""
    
    def test_print_report_basic(self, capsys):
        """Test that report prints without errors."""
        metrics = {
            'overall_accuracy': 0.925,
            'overall_auc': 0.956,
            'num_samples': 1000,
            'per_generator_metrics': {
                'DALL-E 2': {'accuracy': 0.942, 'auc': 0.968},
                'Stable Diffusion': {'accuracy': 0.913, 'auc': 0.945}
            }
        }
        
        print_evaluation_report(metrics)
        
        captured = capsys.readouterr()
        assert 'EVALUATION REPORT' in captured.out
        assert 'Total Samples: 1000' in captured.out
        assert '92.50%' in captured.out
        assert '0.956' in captured.out
        assert 'DALL-E 2' in captured.out
        assert 'Stable Diffusion' in captured.out
    
    def test_print_report_with_nan_auc(self, capsys):
        """Test report printing when AUC is NaN."""
        metrics = {
            'overall_accuracy': 0.85,
            'overall_auc': 0.90,
            'num_samples': 500,
            'per_generator_metrics': {
                'gen1': {'accuracy': 0.85, 'auc': float('nan')}
            }
        }
        
        print_evaluation_report(metrics)
        
        captured = capsys.readouterr()
        assert 'N/A (single class)' in captured.out
    
    def test_print_report_multiple_generators(self, capsys):
        """Test report with multiple generators."""
        metrics = {
            'overall_accuracy': 0.88,
            'overall_auc': 0.92,
            'num_samples': 2000,
            'per_generator_metrics': {
                'DALL-E 2': {'accuracy': 0.90, 'auc': 0.95},
                'Midjourney': {'accuracy': 0.87, 'auc': 0.91},
                'Stable Diffusion': {'accuracy': 0.86, 'auc': 0.89}
            }
        }
        
        print_evaluation_report(metrics)
        
        captured = capsys.readouterr()
        assert 'DALL-E 2' in captured.out
        assert 'Midjourney' in captured.out
        assert 'Stable Diffusion' in captured.out
        assert '90.00%' in captured.out
        assert '87.00%' in captured.out
        assert '86.00%' in captured.out


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_probabilities(self):
        """Test with very small probability values."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([1e-10, 1e-9, 0.999999, 0.9999999])
        generator_labels = ['gen1'] * 4
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert metrics['gen1']['accuracy'] == 1.0
        assert metrics['gen1']['auc'] == 1.0
    
    def test_probabilities_at_threshold(self):
        """Test with probabilities exactly at decision threshold (0.5)."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.49, 0.51])
        generator_labels = ['gen1'] * 4
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        # Should handle threshold cases gracefully
        assert 0.0 <= metrics['gen1']['accuracy'] <= 1.0
        assert 0.0 <= metrics['gen1']['auc'] <= 1.0
    
    def test_large_dataset(self):
        """Test with a large number of samples."""
        n_samples = 10000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true.copy()  # Perfect predictions
        y_prob = y_true.astype(float) + np.random.uniform(-0.1, 0.1, n_samples)
        y_prob = np.clip(y_prob, 0, 1)
        generator_labels = ['gen1'] * n_samples
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        assert metrics['gen1']['accuracy'] == 1.0
        assert metrics['gen1']['auc'] > 0.95  # Should be very high
    
    def test_all_same_predictions(self):
        """Test when model always predicts the same class."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])  # Always predicts 1
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        generator_labels = ['gen1'] * 6
        
        metrics = compute_per_generator_metrics(y_true, y_pred, y_prob, generator_labels)
        
        # Accuracy should be 50% (3 correct out of 6)
        assert metrics['gen1']['accuracy'] == 0.5
        # AUC should be 0.5 (no discrimination)
        assert metrics['gen1']['auc'] == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
