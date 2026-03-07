"""
Tests for BinaryClassifier feature flag functionality.

This module tests the new feature flag parameters added to BinaryClassifier
for task 2.1 of the ml-detector-enhancements spec.
"""

import pytest
import torch
from .classifier import BinaryClassifier


class TestFeatureFlags:
    """Test suite for BinaryClassifier feature flags."""
    
    def test_default_feature_flags(self):
        """Test that all feature flags default to False/None for backward compatibility."""
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        
        flags = model.get_feature_flags()
        
        assert flags['use_spectral'] is False
        assert flags['use_noise_imprint'] is False
        assert flags['use_color_features'] is False
        assert flags['use_local_patches'] is False
        assert flags['use_fpn'] is False
        assert flags['use_attention'] is None
        assert flags['enable_attribution'] is False
    
    def test_feature_flags_stored_as_instance_variables(self):
        """Test that feature flags are stored as instance variables."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            pretrained=False,
            use_spectral=True,
            use_noise_imprint=True,
            use_color_features=True,
            use_local_patches=True,
            use_fpn=True,
            use_attention='cbam',
            enable_attribution=True
        )
        
        assert model.use_spectral is True
        assert model.use_noise_imprint is True
        assert model.use_color_features is True
        assert model.use_local_patches is True
        assert model.use_fpn is True
        assert model.use_attention == 'cbam'
        assert model.enable_attribution is True
    
    def test_get_feature_flags_returns_correct_values(self):
        """Test that get_feature_flags() returns the correct flag values."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            pretrained=False,
            use_spectral=True,
            use_noise_imprint=False,
            use_color_features=True,
            use_local_patches=False,
            use_fpn=True,
            use_attention='se',
            enable_attribution=False
        )
        
        flags = model.get_feature_flags()
        
        assert flags['use_spectral'] is True
        assert flags['use_noise_imprint'] is False
        assert flags['use_color_features'] is True
        assert flags['use_local_patches'] is False
        assert flags['use_fpn'] is True
        assert flags['use_attention'] == 'se'
        assert flags['enable_attribution'] is False
    
    def test_feature_flags_with_different_backbones(self):
        """Test that feature flags work with different backbone types."""
        for backbone_type in ['simple_cnn', 'resnet18', 'resnet50']:
            model = BinaryClassifier(
                backbone_type=backbone_type,
                pretrained=False,
                use_spectral=True,
                use_attention='cbam'
            )
            
            flags = model.get_feature_flags()
            assert flags['use_spectral'] is True
            assert flags['use_attention'] == 'cbam'
    
    def test_backward_compatibility_forward_pass(self):
        """Test that forward pass still works with default flags (backward compatibility)."""
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        model.eval()
        
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 256, 256)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (batch_size, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_attention_flag_accepts_valid_values(self):
        """Test that use_attention accepts valid values: 'cbam', 'se', or None."""
        # Test with 'cbam'
        model1 = BinaryClassifier(
            backbone_type='simple_cnn',
            pretrained=False,
            use_attention='cbam'
        )
        assert model1.use_attention == 'cbam'
        
        # Test with 'se'
        model2 = BinaryClassifier(
            backbone_type='simple_cnn',
            pretrained=False,
            use_attention='se'
        )
        assert model2.use_attention == 'se'
        
        # Test with None
        model3 = BinaryClassifier(
            backbone_type='simple_cnn',
            pretrained=False,
            use_attention=None
        )
        assert model3.use_attention is None
    
    def test_get_feature_flags_returns_dict(self):
        """Test that get_feature_flags() returns a dictionary."""
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        flags = model.get_feature_flags()
        
        assert isinstance(flags, dict)
        assert len(flags) == 7  # Should have 7 feature flags
        
        # Verify all expected keys are present
        expected_keys = {
            'use_spectral',
            'use_noise_imprint',
            'use_color_features',
            'use_local_patches',
            'use_fpn',
            'use_attention',
            'enable_attribution'
        }
        assert set(flags.keys()) == expected_keys
