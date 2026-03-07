"""Unit tests for LocalPatchClassifier module."""

import pytest
import torch
from .local_patch_classifier import LocalPatchClassifier


class TestLocalPatchClassifier:
    """Tests for LocalPatchClassifier module."""
    
    def test_local_patch_classifier_output_shape(self):
        """Test that LocalPatchClassifier produces correct output shape."""
        feature_dim = 256
        classifier = LocalPatchClassifier(feature_dim, patch_size=8, num_classes=1)
        features = torch.randn(4, feature_dim, 32, 32)
        
        out = classifier(features)
        
        assert out.shape == (4, 1)
        
    def test_local_patch_classifier_with_heatmap(self):
        """Test LocalPatchClassifier with heatmap output."""
        classifier = LocalPatchClassifier(256, patch_size=8, num_classes=1)
        features = torch.randn(4, 256, 32, 32)
        
        prediction, heatmap = classifier(features, return_heatmap=True)
        
        assert prediction.shape == (4, 1)
        # 32 / 8 = 4 patches per dimension
        assert heatmap.shape == (4, 4, 4)
        
    def test_local_patch_classifier_heatmap_values_range(self):
        """Test that heatmap values are in valid range [0, 1]."""
        classifier = LocalPatchClassifier(256, patch_size=8)
        features = torch.randn(2, 256, 32, 32)
        
        _, heatmap = classifier(features, return_heatmap=True)
        
        assert torch.all(heatmap >= 0)
        assert torch.all(heatmap <= 1)
        
    def test_local_patch_classifier_prediction_values_range(self):
        """Test that predictions are in valid range [0, 1]."""
        classifier = LocalPatchClassifier(256, patch_size=8)
        features = torch.randn(2, 256, 32, 32)
        
        prediction = classifier(features)
        
        assert torch.all(prediction >= 0)
        assert torch.all(prediction <= 1)
        
    def test_local_patch_classifier_average_aggregation(self):
        """Test LocalPatchClassifier with average aggregation."""
        classifier = LocalPatchClassifier(
            256, patch_size=8, aggregation='average'
        )
        features = torch.randn(2, 256, 32, 32)
        
        prediction = classifier(features)
        
        assert prediction.shape == (2, 1)
        
    def test_local_patch_classifier_max_aggregation(self):
        """Test LocalPatchClassifier with max aggregation."""
        classifier = LocalPatchClassifier(
            256, patch_size=8, aggregation='max'
        )
        features = torch.randn(2, 256, 32, 32)
        
        prediction = classifier(features)
        
        assert prediction.shape == (2, 1)
        
    def test_local_patch_classifier_different_patch_sizes(self):
        """Test LocalPatchClassifier with different patch sizes."""
        feature_dim = 256
        
        for patch_size in [4, 8, 16]:
            classifier = LocalPatchClassifier(feature_dim, patch_size=patch_size)
            features = torch.randn(2, feature_dim, 32, 32)
            
            prediction, heatmap = classifier(features, return_heatmap=True)
            
            assert prediction.shape == (2, 1)
            expected_patches = 32 // patch_size
            assert heatmap.shape == (2, expected_patches, expected_patches)
            
    def test_local_patch_classifier_different_feature_sizes(self):
        """Test LocalPatchClassifier with various feature map sizes."""
        classifier = LocalPatchClassifier(128, patch_size=8)
        
        for h, w in [(32, 32), (64, 64), (32, 64)]:
            features = torch.randn(2, 128, h, w)
            prediction, heatmap = classifier(features, return_heatmap=True)
            
            assert prediction.shape == (2, 1)
            assert heatmap.shape == (2, h // 8, w // 8)
            
    def test_local_patch_classifier_gradient_flow(self):
        """Test that gradients flow through LocalPatchClassifier."""
        classifier = LocalPatchClassifier(256, patch_size=8)
        features = torch.randn(2, 256, 32, 32, requires_grad=True)
        
        prediction = classifier(features)
        loss = prediction.sum()
        loss.backward()
        
        assert features.grad is not None
        assert not torch.all(features.grad == 0)
        
    def test_local_patch_classifier_parameters(self):
        """Test that LocalPatchClassifier has the expected parameters."""
        feature_dim = 256
        hidden_dim = 128
        classifier = LocalPatchClassifier(
            feature_dim, patch_size=8, hidden_dim=hidden_dim
        )
        
        # Check that parameters exist
        params = list(classifier.parameters())
        assert len(params) > 0
        
        # Check classifier architecture
        assert classifier.classifier[0].in_channels == feature_dim
        assert classifier.classifier[0].out_channels == hidden_dim
        
    def test_local_patch_classifier_eval_mode(self):
        """Test LocalPatchClassifier in evaluation mode."""
        classifier = LocalPatchClassifier(256, patch_size=8)
        classifier.eval()
        
        features = torch.randn(2, 256, 32, 32)
        
        with torch.no_grad():
            prediction = classifier(features)
            
        assert prediction.shape == (2, 1)
        
    def test_local_patch_classifier_batch_independence(self):
        """Test that LocalPatchClassifier processes batch samples independently."""
        classifier = LocalPatchClassifier(256, patch_size=8)
        
        # Process samples individually
        f1 = torch.randn(1, 256, 32, 32)
        f2 = torch.randn(1, 256, 32, 32)
        out1 = classifier(f1)
        out2 = classifier(f2)
        
        # Process as batch
        f_batch = torch.cat([f1, f2], dim=0)
        out_batch = classifier(f_batch)
        
        # Results should match
        assert torch.allclose(out_batch[0:1], out1, atol=1e-6)
        assert torch.allclose(out_batch[1:2], out2, atol=1e-6)
        
    def test_local_patch_classifier_get_patch_grid_size(self):
        """Test get_patch_grid_size method."""
        classifier = LocalPatchClassifier(256, patch_size=8)
        
        h_patches, w_patches = classifier.get_patch_grid_size(32, 32)
        assert h_patches == 4
        assert w_patches == 4
        
        h_patches, w_patches = classifier.get_patch_grid_size(64, 32)
        assert h_patches == 8
        assert w_patches == 4
        
    def test_local_patch_classifier_invalid_aggregation(self):
        """Test that invalid aggregation method raises error."""
        classifier = LocalPatchClassifier(
            256, patch_size=8, aggregation='invalid'
        )
        features = torch.randn(2, 256, 32, 32)
        
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            classifier(features)
            
    def test_local_patch_classifier_deterministic(self):
        """Test that LocalPatchClassifier produces deterministic outputs."""
        classifier = LocalPatchClassifier(256, patch_size=8)
        classifier.eval()
        
        features = torch.randn(2, 256, 32, 32)
        
        with torch.no_grad():
            out1 = classifier(features)
            out2 = classifier(features)
            
        assert torch.allclose(out1, out2)
        
    def test_local_patch_classifier_multiclass(self):
        """Test LocalPatchClassifier with multiple classes."""
        num_classes = 3
        classifier = LocalPatchClassifier(
            256, patch_size=8, num_classes=num_classes
        )
        features = torch.randn(2, 256, 32, 32)
        
        prediction = classifier(features)
        
        assert prediction.shape == (2, num_classes)
