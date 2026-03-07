"""Unit tests for FeaturePyramidFusion module."""

import pytest
import torch
from .fpn import FeaturePyramidFusion


class TestFeaturePyramidFusion:
    """Tests for FeaturePyramidFusion module."""
    
    def test_fpn_output_shape(self):
        """Test that FPN produces correct output shape."""
        in_channels_list = [512, 256, 128]
        out_channels = 256
        fpn = FeaturePyramidFusion(in_channels_list, out_channels)
        
        features = [
            torch.randn(4, 512, 8, 8),
            torch.randn(4, 256, 16, 16),
            torch.randn(4, 128, 32, 32)
        ]
        
        fused = fpn(features)
        
        # Output should match highest resolution with out_channels
        assert fused.shape == (4, out_channels, 32, 32)
        
    def test_fpn_two_scales(self):
        """Test FPN with two scales."""
        fpn = FeaturePyramidFusion([256, 128], out_channels=256)
        
        features = [
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 128, 32, 32)
        ]
        
        fused = fpn(features)
        
        assert fused.shape == (2, 256, 32, 32)
        
    def test_fpn_four_scales(self):
        """Test FPN with four scales."""
        fpn = FeaturePyramidFusion([1024, 512, 256, 128], out_channels=256)
        
        features = [
            torch.randn(2, 1024, 4, 4),
            torch.randn(2, 512, 8, 8),
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 128, 32, 32)
        ]
        
        fused = fpn(features)
        
        assert fused.shape == (2, 256, 32, 32)
        
    def test_fpn_different_output_channels(self):
        """Test FPN with different output channel dimensions."""
        in_channels_list = [512, 256, 128]
        
        for out_channels in [128, 256, 512]:
            fpn = FeaturePyramidFusion(in_channels_list, out_channels)
            features = [
                torch.randn(2, 512, 8, 8),
                torch.randn(2, 256, 16, 16),
                torch.randn(2, 128, 32, 32)
            ]
            
            fused = fpn(features)
            
            assert fused.shape == (2, out_channels, 32, 32)
            
    def test_fpn_gradient_flow(self):
        """Test that gradients flow through FPN."""
        fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
        
        features = [
            torch.randn(2, 512, 8, 8, requires_grad=True),
            torch.randn(2, 256, 16, 16, requires_grad=True),
            torch.randn(2, 128, 32, 32, requires_grad=True)
        ]
        
        fused = fpn(features)
        loss = fused.sum()
        loss.backward()
        
        for feature in features:
            assert feature.grad is not None
            assert not torch.all(feature.grad == 0)
            
    def test_fpn_parameters(self):
        """Test that FPN has the expected parameters."""
        in_channels_list = [512, 256, 128]
        out_channels = 256
        fpn = FeaturePyramidFusion(in_channels_list, out_channels)
        
        # Check that parameters exist
        params = list(fpn.parameters())
        assert len(params) > 0
        
        # Check lateral convs
        assert len(fpn.lateral_convs) == len(in_channels_list)
        for i, lateral_conv in enumerate(fpn.lateral_convs):
            assert lateral_conv.in_channels == in_channels_list[i]
            assert lateral_conv.out_channels == out_channels
            
        # Check output convs
        assert len(fpn.output_convs) == len(in_channels_list)
        
    def test_fpn_eval_mode(self):
        """Test FPN in evaluation mode."""
        fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
        fpn.eval()
        
        features = [
            torch.randn(2, 512, 8, 8),
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 128, 32, 32)
        ]
        
        with torch.no_grad():
            fused = fpn(features)
            
        assert fused.shape == (2, 256, 32, 32)
        
    def test_fpn_batch_independence(self):
        """Test that FPN processes batch samples independently."""
        fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
        
        # Process samples individually
        f1 = [
            torch.randn(1, 512, 8, 8),
            torch.randn(1, 256, 16, 16),
            torch.randn(1, 128, 32, 32)
        ]
        f2 = [
            torch.randn(1, 512, 8, 8),
            torch.randn(1, 256, 16, 16),
            torch.randn(1, 128, 32, 32)
        ]
        
        out1 = fpn(f1)
        out2 = fpn(f2)
        
        # Process as batch
        f_batch = [
            torch.cat([f1[i], f2[i]], dim=0)
            for i in range(3)
        ]
        out_batch = fpn(f_batch)
        
        # Results should match
        assert torch.allclose(out_batch[0:1], out1, atol=1e-5)
        assert torch.allclose(out_batch[1:2], out2, atol=1e-5)
        
    def test_fpn_wrong_number_of_features(self):
        """Test that FPN raises error with wrong number of features."""
        fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
        
        # Provide only 2 features instead of 3
        features = [
            torch.randn(2, 512, 8, 8),
            torch.randn(2, 256, 16, 16)
        ]
        
        with pytest.raises(ValueError, match="Expected 3 feature maps"):
            fpn(features)
            
    def test_fpn_deterministic(self):
        """Test that FPN produces deterministic outputs."""
        fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
        fpn.eval()
        
        features = [
            torch.randn(2, 512, 8, 8),
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 128, 32, 32)
        ]
        
        with torch.no_grad():
            out1 = fpn(features)
            out2 = fpn(features)
            
        assert torch.allclose(out1, out2)
        
    def test_fpn_non_square_features(self):
        """Test FPN with non-square feature maps."""
        fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
        
        features = [
            torch.randn(2, 512, 8, 16),
            torch.randn(2, 256, 16, 32),
            torch.randn(2, 128, 32, 64)
        ]
        
        fused = fpn(features)
        
        assert fused.shape == (2, 256, 32, 64)
        
    def test_fpn_single_scale(self):
        """Test FPN with single scale (edge case)."""
        fpn = FeaturePyramidFusion([256], out_channels=256)
        
        features = [torch.randn(2, 256, 32, 32)]
        
        fused = fpn(features)
        
        assert fused.shape == (2, 256, 32, 32)
        
    def test_fpn_upsampling_consistency(self):
        """Test that FPN upsamples features consistently."""
        fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
        
        features = [
            torch.randn(2, 512, 8, 8),
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 128, 32, 32)
        ]
        
        fused = fpn(features)
        
        # All features should be upsampled to highest resolution
        assert fused.shape[2:] == features[-1].shape[2:]
        
    def test_fpn_different_batch_sizes(self):
        """Test FPN with different batch sizes."""
        fpn = FeaturePyramidFusion([512, 256, 128], out_channels=256)
        
        for batch_size in [1, 2, 4, 8]:
            features = [
                torch.randn(batch_size, 512, 8, 8),
                torch.randn(batch_size, 256, 16, 16),
                torch.randn(batch_size, 128, 32, 32)
            ]
            
            fused = fpn(features)
            
            assert fused.shape == (batch_size, 256, 32, 32)
