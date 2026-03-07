"""Unit tests for CBAM attention module."""

import pytest
import torch
from .cbam import CBAM, ChannelAttention, SpatialAttention


class TestChannelAttention:
    """Tests for ChannelAttention module."""
    
    def test_channel_attention_output_shape(self):
        """Test that channel attention produces correct output shape."""
        channels = 64
        ca = ChannelAttention(channels, reduction_ratio=16)
        x = torch.randn(2, channels, 32, 32)
        
        out = ca(x)
        
        assert out.shape == (2, channels, 1, 1)
        
    def test_channel_attention_values_in_range(self):
        """Test that channel attention weights are in valid range [0, 1]."""
        ca = ChannelAttention(64, reduction_ratio=16)
        x = torch.randn(2, 64, 32, 32)
        
        weights = ca(x)
        
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        
    def test_channel_attention_reduction_ratio(self):
        """Test that reduction ratio is applied correctly."""
        channels = 64
        reduction_ratio = 16
        ca = ChannelAttention(channels, reduction_ratio)
        
        # Check MLP dimensions
        assert ca.mlp[0].out_features == channels // reduction_ratio
        assert ca.mlp[2].out_features == channels


class TestSpatialAttention:
    """Tests for SpatialAttention module."""
    
    def test_spatial_attention_output_shape(self):
        """Test that spatial attention produces correct output shape."""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 32, 32)
        
        out = sa(x)
        
        assert out.shape == (2, 1, 32, 32)
        
    def test_spatial_attention_values_in_range(self):
        """Test that spatial attention weights are in valid range [0, 1]."""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 32, 32)
        
        weights = sa(x)
        
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        
    def test_spatial_attention_kernel_size(self):
        """Test that kernel size is applied correctly."""
        kernel_size = 7
        sa = SpatialAttention(kernel_size)
        
        assert sa.conv.kernel_size == (kernel_size, kernel_size)


class TestCBAM:
    """Tests for CBAM module."""
    
    def test_cbam_output_shape(self):
        """Test that CBAM preserves input shape."""
        channels = 256
        cbam = CBAM(channels, reduction_ratio=16, kernel_size=7)
        x = torch.randn(4, channels, 32, 32)
        
        out = cbam(x)
        
        assert out.shape == x.shape
        
    def test_cbam_applies_both_attentions(self):
        """Test that CBAM applies both channel and spatial attention."""
        cbam = CBAM(64, reduction_ratio=16, kernel_size=7)
        x = torch.randn(2, 64, 32, 32)
        
        # Forward pass should not raise errors
        out = cbam(x)
        
        # Output should be different from input (attention applied)
        assert not torch.allclose(out, x)
        
    def test_cbam_with_different_input_sizes(self):
        """Test CBAM with various input spatial dimensions."""
        cbam = CBAM(128, reduction_ratio=16, kernel_size=7)
        
        for h, w in [(16, 16), (32, 32), (64, 64), (32, 64)]:
            x = torch.randn(2, 128, h, w)
            out = cbam(x)
            assert out.shape == (2, 128, h, w)
            
    def test_cbam_gradient_flow(self):
        """Test that gradients flow through CBAM."""
        cbam = CBAM(64, reduction_ratio=16, kernel_size=7)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        
        out = cbam(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        
    def test_cbam_parameters(self):
        """Test that CBAM has the expected parameters."""
        channels = 256
        reduction_ratio = 16
        cbam = CBAM(channels, reduction_ratio, kernel_size=7)
        
        # Check that parameters exist
        params = list(cbam.parameters())
        assert len(params) > 0
        
        # Check channel attention MLP parameters
        assert cbam.channel_attention.mlp[0].out_features == channels // reduction_ratio
        
    def test_cbam_eval_mode(self):
        """Test CBAM in evaluation mode."""
        cbam = CBAM(64, reduction_ratio=16, kernel_size=7)
        cbam.eval()
        
        x = torch.randn(2, 64, 32, 32)
        
        with torch.no_grad():
            out = cbam(x)
            
        assert out.shape == x.shape
        
    def test_cbam_batch_independence(self):
        """Test that CBAM processes batch samples independently."""
        cbam = CBAM(64, reduction_ratio=16, kernel_size=7)
        
        # Process samples individually
        x1 = torch.randn(1, 64, 32, 32)
        x2 = torch.randn(1, 64, 32, 32)
        out1 = cbam(x1)
        out2 = cbam(x2)
        
        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        out_batch = cbam(x_batch)
        
        # Results should match
        assert torch.allclose(out_batch[0:1], out1, atol=1e-6)
        assert torch.allclose(out_batch[1:2], out2, atol=1e-6)
