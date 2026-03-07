"""
Tests for conditional module instantiation based on feature flags.

This module tests that optional branches are only instantiated when their
corresponding feature flags are enabled, and that backward compatibility
is maintained when all flags are disabled.
"""

import pytest
import torch
from .classifier import BinaryClassifier


class TestConditionalInstantiation:
    """Test suite for conditional module instantiation."""
    
    def test_all_flags_disabled_no_optional_modules(self):
        """Test that no optional modules are instantiated when all flags are False."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_spectral=False,
            use_noise_imprint=False,
            use_color_features=False,
            use_local_patches=False,
            use_fpn=False,
            use_attention=None,
            enable_attribution=False
        )
        
        # Verify that optional modules are None
        assert model.spectral_branch is None
        assert model.noise_extractor is None
        assert model.noise_branch is None
        assert model.rgb_to_ycbcr is None
        assert model.chrominance_branch is None
        assert model.attention_module is None
        assert model.fpn is None
        assert model.local_patch_classifier is None
        assert model.fusion_layer is None
    
    def test_spectral_flag_enables_spectral_branch(self):
        """Test that spectral branch is instantiated when use_spectral=True."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_spectral=True
        )
        
        # Spectral branch should now be instantiated
        assert model.use_spectral is True
        assert model.spectral_branch is not None
        assert model.fusion_layer is not None  # Fusion layer should be created
    
    def test_noise_imprint_flag_enables_noise_branch(self):
        """Test that noise branch is instantiated when use_noise_imprint=True."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_noise_imprint=True
        )
        
        # Noise branch should now be instantiated
        assert model.use_noise_imprint is True
        assert model.noise_extractor is not None
        assert model.noise_branch is not None
        assert model.fusion_layer is not None  # Fusion layer should be created
    
    def test_color_features_flag_enables_chrominance_branch(self):
        """Test that chrominance branch is instantiated when use_color_features=True."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_color_features=True
        )
        
        # Chrominance branch should now be instantiated
        assert model.use_color_features is True
        assert model.rgb_to_ycbcr is not None
        assert model.chrominance_branch is not None
        assert model.fusion_layer is not None  # Fusion layer should be created
    
    def test_attention_flag_enables_attention_module(self):
        """Test that attention module is instantiated when use_attention is set."""
        model_cbam = BinaryClassifier(
            backbone_type='simple_cnn',
            use_attention='cbam'
        )
        
        model_se = BinaryClassifier(
            backbone_type='simple_cnn',
            use_attention='se'
        )
        
        # Attention modules should now be instantiated
        assert model_cbam.use_attention == 'cbam'
        assert model_cbam.attention_module is not None
        assert model_se.use_attention == 'se'
        assert model_se.attention_module is not None
    
    def test_fpn_flag_enables_fpn_module(self):
        """Test that FPN module is instantiated when use_fpn=True."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_fpn=True
        )
        
        # FPN should now be instantiated
        assert model.use_fpn is True
        assert model.fpn is not None
        assert model.fusion_layer is not None  # Fusion layer should be created
    
    def test_local_patches_flag_enables_patch_classifier(self):
        """Test that local patch classifier is prepared when use_local_patches=True."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_local_patches=True
        )
        
        # Currently a placeholder (None), but the flag is stored
        assert model.use_local_patches is True
        # When implemented, this will be: assert model.local_patch_classifier is not None
    
    def test_fusion_layer_created_when_multiple_features_enabled(self):
        """Test that fusion layer is created when multiple features are enabled."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_spectral=True,
            use_noise_imprint=True
        )
        
        # Fusion layer should now be created
        assert model.use_spectral is True
        assert model.use_noise_imprint is True
        assert model.fusion_layer is not None
    
    def test_backward_compatibility_forward_pass(self):
        """Test that forward pass works with all flags disabled (backward compatibility)."""
        model = BinaryClassifier(backbone_type='simple_cnn')
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 256, 256)
        
        # Forward pass should work and return only prediction
        with torch.no_grad():
            output = model(x)
        
        # Verify output shape and type
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_forward_pass_with_flags_enabled(self):
        """Test that forward pass works with various flags enabled."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_spectral=True,
            use_noise_imprint=True,
            use_color_features=True,
            use_fpn=True
        )
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 256, 256)
        
        # Forward pass should work (placeholders don't affect output yet)
        with torch.no_grad():
            output = model(x)
        
        # Verify output shape and type
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_attribution_return_type(self):
        """Test that attribution flag affects return type (when implemented)."""
        model = BinaryClassifier(
            backbone_type='simple_cnn',
            use_noise_imprint=True,
            enable_attribution=True
        )
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 256, 256)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Now returns (prediction, attribution) tuple
        assert isinstance(output, tuple)
        assert len(output) == 2
        prediction, attribution = output
        assert prediction.shape == (2, 1)
        assert attribution.shape == (2, 10)  # 10 generators by default
        # Check that attribution is a valid probability distribution
        assert torch.allclose(attribution.sum(dim=1), torch.ones(2), atol=1e-5)
    
    def test_multiple_flags_combination(self):
        """Test various combinations of feature flags."""
        combinations = [
            {'use_spectral': True, 'use_attention': 'cbam'},
            {'use_noise_imprint': True, 'use_color_features': True},
            {'use_fpn': True, 'use_local_patches': True},
            {'use_spectral': True, 'use_noise_imprint': True, 'use_color_features': True, 'use_fpn': True}
        ]
        
        for flags in combinations:
            model = BinaryClassifier(backbone_type='simple_cnn', **flags)
            model.eval()  # Set to eval mode to avoid batch norm issues with batch size 1
            
            # Verify flags are stored correctly
            for flag_name, flag_value in flags.items():
                assert getattr(model, flag_name) == flag_value
            
            # Verify forward pass works
            x = torch.randn(2, 3, 256, 256)  # Use batch size 2 to avoid batch norm issues
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
