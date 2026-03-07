"""
Integration tests for feature extraction and fusion in BinaryClassifier.

This module tests that all feature branches work together correctly and
produce valid outputs when combined.
"""

import pytest
import torch
from .classifier import BinaryClassifier


class TestFeatureIntegration:
    """Test suite for integrated feature extraction and fusion."""
    
    def test_spectral_branch_integration(self):
        """Test that spectral branch integrates correctly."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_spectral=True
        )
        model.eval()
        
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_noise_imprint_branch_integration(self):
        """Test that noise imprint branch integrates correctly."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_noise_imprint=True
        )
        model.eval()
        
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_chrominance_branch_integration(self):
        """Test that chrominance branch integrates correctly."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_color_features=True
        )
        model.eval()
        
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_attention_integration(self):
        """Test that attention modules integrate correctly."""
        for attention_type in ['cbam', 'se']:
            model = BinaryClassifier(
                backbone_type='simple_cnn',
                use_attention=attention_type
            )
            model.eval()
            
            x = torch.randn(2, 3, 256, 256)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 1)
            assert torch.all((output >= 0) & (output <= 1))
    
    def test_fpn_integration(self):
        """Test that FPN integrates correctly."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_fpn=True
        )
        model.eval()
        
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_all_features_combined(self):
        """Test that all features work together."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_spectral=True,
            use_noise_imprint=True,
            use_color_features=True,
            use_fpn=True,
            use_attention='cbam'
        )
        model.eval()
        
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_attribution_with_noise_imprint(self):
        """Test that attribution works with noise imprint."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_noise_imprint=True,
            enable_attribution=True
        )
        model.eval()
        
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            prediction, attribution = model(x)
        
        assert prediction.shape == (2, 1)
        assert attribution.shape == (2, 10)
        assert torch.all((prediction >= 0) & (prediction <= 1))
        assert torch.allclose(attribution.sum(dim=1), torch.ones(2), atol=1e-5)
    
    def test_fusion_layer_combines_features(self):
        """Test that fusion layer properly combines multiple feature sources."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_spectral=True,
            use_noise_imprint=True,
            use_color_features=True
        )
        model.eval()
        
        # Verify fusion layer exists
        assert model.fusion_layer is not None
        
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_different_backbones_with_features(self):
        """Test that features work with different backbone types."""
        for backbone_type in ['simple_cnn', 'resnet18', 'resnet50']:
            model = BinaryClassifier(
                backbone_type=backbone_type,
                pretrained=False,
                use_spectral=True,
                use_color_features=True
            )
            model.eval()
            
            x = torch.randn(2, 3, 256, 256)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 1)
            assert torch.all((output >= 0) & (output <= 1))
    
    def test_fpn_with_different_backbones(self):
        """Test that FPN works with different backbone architectures."""
        for backbone_type in ['simple_cnn', 'resnet18', 'resnet50']:
            model = BinaryClassifier(
                backbone_type=backbone_type,
                pretrained=False,
                use_fpn=True
            )
            model.eval()
            
            x = torch.randn(2, 3, 256, 256)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 1)
            assert torch.all((output >= 0) & (output <= 1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
