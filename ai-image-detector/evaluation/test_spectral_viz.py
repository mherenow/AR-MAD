"""Unit tests for spectral artifact visualization module."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from .spectral_viz import (
    SpectralGradCAM,
    visualize_spectral_artifacts,
    get_available_target_layers,
    check_gradcam_availability,
    GRADCAM_AVAILABLE
)


class MockSpectralBranch(nn.Module):
    """Mock spectral branch for testing."""
    
    def __init__(self):
        super().__init__()
        self.transformer_encoder = nn.ModuleDict({
            'layers': nn.ModuleList([
                nn.Linear(256, 256),
                nn.Linear(256, 256),
                nn.Linear(256, 256),
                nn.Linear(256, 256)
            ])
        })
    
    def forward(self, x):
        return x, x


class MockModel(nn.Module):
    """Mock model with spectral branch for testing."""
    
    def __init__(self):
        super().__init__()
        self.spectral_branch = MockSpectralBranch()
        self.classifier = nn.Linear(256, 1)
    
    def forward(self, x):
        # Simple forward pass
        B = x.shape[0]
        return torch.sigmoid(torch.randn(B, 1))


class TestGradCAMAvailability:
    """Test GradCAM availability checking."""
    
    def test_check_gradcam_availability(self):
        """Test that check_gradcam_availability returns a boolean."""
        result = check_gradcam_availability()
        assert isinstance(result, bool)
    
    def test_gradcam_available_constant(self):
        """Test that GRADCAM_AVAILABLE is a boolean."""
        assert isinstance(GRADCAM_AVAILABLE, bool)


class TestGetAvailableTargetLayers:
    """Test layer discovery functionality."""
    
    def test_get_available_target_layers_returns_list(self):
        """Test that get_available_target_layers returns a list."""
        model = MockModel()
        layers = get_available_target_layers(model)
        assert isinstance(layers, list)
    
    def test_get_available_target_layers_finds_linear_layers(self):
        """Test that Linear layers are found."""
        model = MockModel()
        layers = get_available_target_layers(model)
        
        # Should find at least the classifier Linear layer
        assert any('classifier' in layer for layer in layers)
    
    def test_get_available_target_layers_finds_transformer_layers(self):
        """Test that transformer layers are found."""
        model = MockModel()
        layers = get_available_target_layers(model)
        
        # Should find transformer encoder layers
        transformer_layers = [l for l in layers if 'transformer' in l.lower()]
        assert len(transformer_layers) > 0
    
    def test_get_available_target_layers_empty_model(self):
        """Test with a model that has no suitable layers."""
        model = nn.Module()
        layers = get_available_target_layers(model)
        assert isinstance(layers, list)
        assert len(layers) == 0


@pytest.mark.skipif(not GRADCAM_AVAILABLE, reason="pytorch-grad-cam not installed")
class TestSpectralGradCAM:
    """Test SpectralGradCAM class (only if pytorch-grad-cam is available)."""
    
    def test_initialization_with_default_layer(self):
        """Test SpectralGradCAM initialization with default target layer."""
        model = MockModel()
        viz = SpectralGradCAM(model, use_cuda=False)
        
        assert viz.model is model
        assert viz.target_layer is not None
        assert hasattr(viz, 'cam')
    
    def test_initialization_with_layer_name(self):
        """Test SpectralGradCAM initialization with layer name."""
        model = MockModel()
        layer_name = 'spectral_branch.transformer_encoder.layers.0'
        viz = SpectralGradCAM(model, target_layer=layer_name, use_cuda=False)
        
        assert viz.target_layer is not None
    
    def test_initialization_with_layer_module(self):
        """Test SpectralGradCAM initialization with layer module."""
        model = MockModel()
        target_layer = model.spectral_branch.transformer_encoder['layers'][0]
        viz = SpectralGradCAM(model, target_layer=target_layer, use_cuda=False)
        
        assert viz.target_layer is target_layer
    
    def test_initialization_invalid_layer_name(self):
        """Test that invalid layer name raises ValueError."""
        model = MockModel()
        with pytest.raises(ValueError, match="Could not find target layer"):
            SpectralGradCAM(model, target_layer='nonexistent.layer', use_cuda=False)
    
    def test_find_default_target_layer(self):
        """Test finding default target layer."""
        model = MockModel()
        viz = SpectralGradCAM(model, use_cuda=False)
        
        # Should find the last transformer layer
        assert viz.target_layer is not None
        assert isinstance(viz.target_layer, nn.Module)
    
    def test_get_layer_by_name_valid(self):
        """Test getting layer by valid name."""
        model = MockModel()
        viz = SpectralGradCAM(model, use_cuda=False)
        
        layer = viz._get_layer_by_name('spectral_branch.transformer_encoder.layers.0')
        assert layer is not None
        assert isinstance(layer, nn.Module)
    
    def test_get_layer_by_name_invalid(self):
        """Test getting layer by invalid name returns None."""
        model = MockModel()
        viz = SpectralGradCAM(model, use_cuda=False)
        
        layer = viz._get_layer_by_name('nonexistent.layer')
        assert layer is None
    
    def test_get_layer_by_name_invalid_index(self):
        """Test getting layer with out-of-bounds index returns None."""
        model = MockModel()
        viz = SpectralGradCAM(model, use_cuda=False)
        
        layer = viz._get_layer_by_name('spectral_branch.transformer_encoder.layers.999')
        assert layer is None
    
    @patch('spectral_viz.GradCAM')
    def test_generate_heatmaps_shape(self, mock_gradcam_class):
        """Test that generate_heatmaps returns correct shape."""
        model = MockModel()
        
        # Mock GradCAM to return dummy heatmaps
        mock_cam = MagicMock()
        mock_cam.return_value = np.random.rand(2, 256, 256)
        mock_gradcam_class.return_value = mock_cam
        
        viz = SpectralGradCAM(model, use_cuda=False)
        viz.cam = mock_cam
        
        images = torch.randn(2, 3, 256, 256)
        heatmaps = viz.generate_heatmaps(images)
        
        assert isinstance(heatmaps, torch.Tensor)
        assert heatmaps.shape == (2, 256, 256)
    
    @patch('spectral_viz.GradCAM')
    def test_generate_heatmaps_with_target_class(self, mock_gradcam_class):
        """Test generate_heatmaps with specified target class."""
        model = MockModel()
        
        mock_cam = MagicMock()
        mock_cam.return_value = np.random.rand(2, 256, 256)
        mock_gradcam_class.return_value = mock_cam
        
        viz = SpectralGradCAM(model, use_cuda=False)
        viz.cam = mock_cam
        
        images = torch.randn(2, 3, 256, 256)
        heatmaps = viz.generate_heatmaps(images, target_class=1)
        
        assert isinstance(heatmaps, torch.Tensor)
        assert heatmaps.shape == (2, 256, 256)
    
    @patch('spectral_viz.GradCAM')
    def test_visualize_spectral_artifacts_shape(self, mock_gradcam_class):
        """Test that visualize_spectral_artifacts returns correct shape."""
        model = MockModel()
        
        mock_cam = MagicMock()
        mock_cam.return_value = np.random.rand(2, 256, 256)
        mock_gradcam_class.return_value = mock_cam
        
        viz = SpectralGradCAM(model, use_cuda=False)
        viz.cam = mock_cam
        
        images = torch.rand(2, 3, 256, 256)  # Use rand for [0, 1] range
        overlays = viz.visualize_spectral_artifacts(images)
        
        assert isinstance(overlays, np.ndarray)
        assert overlays.shape == (2, 256, 256, 3)
        assert overlays.dtype == np.uint8
    
    @patch('spectral_viz.GradCAM')
    def test_visualize_spectral_artifacts_value_range(self, mock_gradcam_class):
        """Test that overlay values are in valid range [0, 255]."""
        model = MockModel()
        
        mock_cam = MagicMock()
        mock_cam.return_value = np.random.rand(2, 256, 256)
        mock_gradcam_class.return_value = mock_cam
        
        viz = SpectralGradCAM(model, use_cuda=False)
        viz.cam = mock_cam
        
        images = torch.rand(2, 3, 256, 256)
        overlays = viz.visualize_spectral_artifacts(images)
        
        assert np.all(overlays >= 0)
        assert np.all(overlays <= 255)
    
    @patch('spectral_viz.GradCAM')
    def test_visualize_spectral_artifacts_custom_alpha(self, mock_gradcam_class):
        """Test visualize_spectral_artifacts with custom alpha."""
        model = MockModel()
        
        mock_cam = MagicMock()
        mock_cam.return_value = np.random.rand(2, 256, 256)
        mock_gradcam_class.return_value = mock_cam
        
        viz = SpectralGradCAM(model, use_cuda=False)
        viz.cam = mock_cam
        
        images = torch.rand(2, 3, 256, 256)
        overlays = viz.visualize_spectral_artifacts(images, overlay_alpha=0.7)
        
        assert isinstance(overlays, np.ndarray)
        assert overlays.shape == (2, 256, 256, 3)


@pytest.mark.skipif(not GRADCAM_AVAILABLE, reason="pytorch-grad-cam not installed")
class TestVisualizationConvenienceFunction:
    """Test the convenience function for visualization."""
    
    @patch('spectral_viz.SpectralGradCAM')
    def test_visualize_spectral_artifacts_function(self, mock_gradcam_class):
        """Test convenience function for visualization."""
        model = MockModel()
        images = torch.rand(2, 3, 256, 256)
        
        # Mock SpectralGradCAM instance
        mock_viz = MagicMock()
        mock_viz.visualize_spectral_artifacts.return_value = np.random.randint(
            0, 256, (2, 256, 256, 3), dtype=np.uint8
        )
        mock_gradcam_class.return_value = mock_viz
        
        overlays = visualize_spectral_artifacts(model, images)
        
        assert overlays is not None
        assert isinstance(overlays, np.ndarray)
        mock_gradcam_class.assert_called_once()
        mock_viz.visualize_spectral_artifacts.assert_called_once()
    
    @patch('spectral_viz.SpectralGradCAM')
    def test_visualize_spectral_artifacts_with_device(self, mock_gradcam_class):
        """Test convenience function with explicit device."""
        model = MockModel()
        images = torch.rand(2, 3, 256, 256)
        device = torch.device('cpu')
        
        mock_viz = MagicMock()
        mock_viz.visualize_spectral_artifacts.return_value = np.random.randint(
            0, 256, (2, 256, 256, 3), dtype=np.uint8
        )
        mock_gradcam_class.return_value = mock_viz
        
        overlays = visualize_spectral_artifacts(model, images, device=device)
        
        assert overlays is not None
        mock_gradcam_class.assert_called_once()


@pytest.mark.skipif(GRADCAM_AVAILABLE, reason="Test for when pytorch-grad-cam is not installed")
class TestGradCAMUnavailable:
    """Test behavior when pytorch-grad-cam is not available."""
    
    def test_spectral_gradcam_raises_import_error(self):
        """Test that SpectralGradCAM raises ImportError when library unavailable."""
        model = MockModel()
        with pytest.raises(ImportError, match="pytorch-grad-cam is required"):
            SpectralGradCAM(model)
    
    def test_visualize_function_returns_none(self):
        """Test that convenience function returns None with warning."""
        model = MockModel()
        images = torch.rand(2, 3, 256, 256)
        
        with pytest.warns(UserWarning, match="pytorch-grad-cam not available"):
            result = visualize_spectral_artifacts(model, images)
        
        assert result is None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_get_available_target_layers_with_conv_layers(self):
        """Test that Conv2d layers are detected."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3)
        )
        layers = get_available_target_layers(model)
        
        # Should find Conv2d layers
        assert len(layers) >= 2
    
    def test_get_available_target_layers_with_multihead_attention(self):
        """Test that MultiheadAttention layers are detected."""
        model = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        layers = get_available_target_layers(model)
        
        # Should find the MultiheadAttention layer
        assert len(layers) > 0
    
    @pytest.mark.skipif(not GRADCAM_AVAILABLE, reason="pytorch-grad-cam not installed")
    def test_spectral_gradcam_with_model_without_spectral_branch(self):
        """Test SpectralGradCAM with model that doesn't have spectral branch."""
        model = nn.Sequential(nn.Linear(256, 1))
        
        # Should raise ValueError because no default target layer can be found
        with pytest.raises(ValueError, match="Could not find target layer"):
            SpectralGradCAM(model)
    
    @pytest.mark.skipif(not GRADCAM_AVAILABLE, reason="pytorch-grad-cam not installed")
    @patch('spectral_viz.GradCAM')
    def test_generate_heatmaps_with_batch_size_one(self, mock_gradcam_class):
        """Test generate_heatmaps with batch size 1."""
        model = MockModel()
        
        mock_cam = MagicMock()
        mock_cam.return_value = np.random.rand(1, 256, 256)
        mock_gradcam_class.return_value = mock_cam
        
        viz = SpectralGradCAM(model, use_cuda=False)
        viz.cam = mock_cam
        
        images = torch.randn(1, 3, 256, 256)
        heatmaps = viz.generate_heatmaps(images)
        
        assert heatmaps.shape == (1, 256, 256)
    
    @pytest.mark.skipif(not GRADCAM_AVAILABLE, reason="pytorch-grad-cam not installed")
    @patch('spectral_viz.GradCAM')
    def test_generate_heatmaps_with_different_image_sizes(self, mock_gradcam_class):
        """Test generate_heatmaps with different image sizes."""
        model = MockModel()
        
        mock_cam = MagicMock()
        mock_cam.return_value = np.random.rand(2, 512, 512)
        mock_gradcam_class.return_value = mock_cam
        
        viz = SpectralGradCAM(model, use_cuda=False)
        viz.cam = mock_cam
        
        images = torch.randn(2, 3, 512, 512)
        heatmaps = viz.generate_heatmaps(images)
        
        assert heatmaps.shape == (2, 512, 512)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
