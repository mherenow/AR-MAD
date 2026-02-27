"""
Unit tests for model architecture components.

Tests cover:
- Forward pass with dummy tensors
- Output shape validation
- Output value range validation
- Backbone swapping (simple_cnn, resnet18, resnet50)
"""

import pytest
import torch
import torch.nn as nn
from .classifier import BinaryClassifier, ClassificationHead
from .backbones import SimpleCNN, get_resnet18, get_resnet50


class TestBackbones:
    """Test suite for backbone architectures."""
    
    @pytest.fixture
    def dummy_input(self):
        """Create dummy input tensor (B=4, C=3, H=256, W=256)."""
        return torch.randn(4, 3, 256, 256)
    
    def test_simple_cnn_forward_pass(self, dummy_input):
        """Test SimpleCNN forward pass with dummy input."""
        model = SimpleCNN(input_channels=3)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output shape: (B, 512, 16, 16) after 4 maxpool layers
        assert output.shape == (4, 512, 16, 16), \
            f"Expected shape (4, 512, 16, 16), got {output.shape}"
        
        # Check output is not NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_resnet18_forward_pass(self, dummy_input):
        """Test ResNet18 backbone forward pass."""
        model = get_resnet18(pretrained=False, freeze_backbone=False)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # ResNet18 outputs (B, 512, 1, 1) after global avg pool
        assert output.shape == (4, 512, 1, 1), \
            f"Expected shape (4, 512, 1, 1), got {output.shape}"
        
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_resnet50_forward_pass(self, dummy_input):
        """Test ResNet50 backbone forward pass."""
        model = get_resnet50(pretrained=False, freeze_backbone=False)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # ResNet50 outputs (B, 2048, 1, 1) after global avg pool
        assert output.shape == (4, 2048, 1, 1), \
            f"Expected shape (4, 2048, 1, 1), got {output.shape}"
        
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_resnet18_freeze_backbone(self):
        """Test that freeze_backbone parameter works for ResNet18."""
        model = get_resnet18(pretrained=False, freeze_backbone=True)
        
        # Check that all parameters are frozen
        for param in model.parameters():
            assert not param.requires_grad, \
                "Expected all parameters to be frozen"
    
    def test_resnet50_freeze_backbone(self):
        """Test that freeze_backbone parameter works for ResNet50."""
        model = get_resnet50(pretrained=False, freeze_backbone=True)
        
        # Check that all parameters are frozen
        for param in model.parameters():
            assert not param.requires_grad, \
                "Expected all parameters to be frozen"


class TestClassificationHead:
    """Test suite for classification head."""
    
    def test_classification_head_forward_pass(self):
        """Test ClassificationHead forward pass."""
        head = ClassificationHead(feature_dim=512)
        head.eval()
        
        # Create dummy features (B=4, feature_dim=512)
        dummy_features = torch.randn(4, 512)
        
        with torch.no_grad():
            output = head(dummy_features)
        
        # Check output shape: (B, 1)
        assert output.shape == (4, 1), \
            f"Expected shape (4, 1), got {output.shape}"
        
        # Check output range [0, 1] due to Sigmoid
        assert (output >= 0).all() and (output <= 1).all(), \
            f"Output values should be in [0, 1], got min={output.min()}, max={output.max()}"
    
    def test_classification_head_different_batch_sizes(self):
        """Test ClassificationHead with different batch sizes."""
        head = ClassificationHead(feature_dim=512)
        head.eval()
        
        batch_sizes = [1, 2, 8, 16, 32]
        
        for batch_size in batch_sizes:
            dummy_features = torch.randn(batch_size, 512)
            
            with torch.no_grad():
                output = head(dummy_features)
            
            assert output.shape == (batch_size, 1), \
                f"Expected shape ({batch_size}, 1), got {output.shape}"
            assert (output >= 0).all() and (output <= 1).all(), \
                "Output values should be in [0, 1]"


class TestBinaryClassifier:
    """Test suite for complete binary classifier."""
    
    @pytest.fixture
    def dummy_input(self):
        """Create dummy input tensor (B=4, C=3, H=256, W=256)."""
        return torch.randn(4, 3, 256, 256)
    
    @pytest.mark.parametrize("backbone_type", ["simple_cnn", "resnet18", "resnet50"])
    def test_binary_classifier_forward_pass(self, dummy_input, backbone_type):
        """Test BinaryClassifier forward pass with different backbones."""
        model = BinaryClassifier(backbone_type=backbone_type, pretrained=False)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output shape: (B, 1)
        assert output.shape == (4, 1), \
            f"Expected shape (4, 1), got {output.shape} for backbone {backbone_type}"
        
        # Check output range [0, 1]
        assert (output >= 0).all() and (output <= 1).all(), \
            f"Output values should be in [0, 1], got min={output.min()}, max={output.max()} for backbone {backbone_type}"
        
        # Check output is not NaN or Inf
        assert not torch.isnan(output).any(), \
            f"Output contains NaN values for backbone {backbone_type}"
        assert not torch.isinf(output).any(), \
            f"Output contains Inf values for backbone {backbone_type}"
    
    def test_backbone_swapping_simple_cnn(self, dummy_input):
        """Test backbone swapping: simple_cnn."""
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        model.eval()
        
        assert model.backbone_type == 'simple_cnn'
        assert isinstance(model.backbone, SimpleCNN)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (4, 1)
        assert (output >= 0).all() and (output <= 1).all()
    
    def test_backbone_swapping_resnet18(self, dummy_input):
        """Test backbone swapping: resnet18."""
        model = BinaryClassifier(backbone_type='resnet18', pretrained=False)
        model.eval()
        
        assert model.backbone_type == 'resnet18'
        assert isinstance(model.backbone, nn.Sequential)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (4, 1)
        assert (output >= 0).all() and (output <= 1).all()
    
    def test_backbone_swapping_resnet50(self, dummy_input):
        """Test backbone swapping: resnet50."""
        model = BinaryClassifier(backbone_type='resnet50', pretrained=False)
        model.eval()
        
        assert model.backbone_type == 'resnet50'
        assert isinstance(model.backbone, nn.Sequential)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (4, 1)
        assert (output >= 0).all() and (output <= 1).all()
    
    def test_invalid_backbone_type(self):
        """Test that invalid backbone type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backbone type"):
            BinaryClassifier(backbone_type='invalid_backbone')
    
    def test_different_batch_sizes(self):
        """Test BinaryClassifier with different batch sizes."""
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        model.eval()
        
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 256, 256)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape == (batch_size, 1), \
                f"Expected shape ({batch_size}, 1), got {output.shape}"
            assert (output >= 0).all() and (output <= 1).all(), \
                "Output values should be in [0, 1]"
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        model.train()
        
        dummy_input = torch.randn(2, 3, 256, 256, requires_grad=True)
        output = model(dummy_input)
        
        # Compute a simple loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for model parameters
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in model parameters"
    
    def test_model_determinism(self):
        """Test that model produces deterministic outputs in eval mode."""
        torch.manual_seed(42)
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = model(dummy_input)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2), \
            "Model outputs are not deterministic in eval mode"
    
    def test_output_range_comprehensive(self):
        """Comprehensive test for output value range across multiple forward passes."""
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        model.eval()
        
        # Test with multiple random inputs
        for _ in range(10):
            dummy_input = torch.randn(4, 3, 256, 256)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Verify all outputs are in [0, 1]
            assert (output >= 0).all(), \
                f"Found output values < 0: min={output.min()}"
            assert (output <= 1).all(), \
                f"Found output values > 1: max={output.max()}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
