"""Unit tests for SEBlock attention module."""

import pytest
import torch
from .se_block import SEBlock


class TestSEBlock:
    """Tests for SEBlock module."""
    
    def test_se_block_output_shape(self):
        """Test that SEBlock preserves input shape."""
        channels = 256
        se = SEBlock(channels, reduction=16)
        x = torch.randn(4, channels, 32, 32)
        
        out = se(x)
        
        assert out.shape == x.shape
        
    def test_se_block_channel_recalibration(self):
        """Test that SEBlock applies channel recalibration."""
        se = SEBlock(64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        
        out = se(x)
        
        # Output should be different from input (recalibration applied)
        assert not torch.allclose(out, x)
        
    def test_se_block_reduction_ratio(self):
        """Test that reduction ratio is applied correctly."""
        channels = 64
        reduction = 16
        se = SEBlock(channels, reduction)
        
        # Check FC layer dimensions
        assert se.fc1.out_features == channels // reduction
        assert se.fc2.out_features == channels
        
    def test_se_block_with_different_input_sizes(self):
        """Test SEBlock with various input spatial dimensions."""
        se = SEBlock(128, reduction=16)
        
        for h, w in [(16, 16), (32, 32), (64, 64), (32, 64)]:
            x = torch.randn(2, 128, h, w)
            out = se(x)
            assert out.shape == (2, 128, h, w)
            
    def test_se_block_gradient_flow(self):
        """Test that gradients flow through SEBlock."""
        se = SEBlock(64, reduction=16)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        
        out = se(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        
    def test_se_block_parameters(self):
        """Test that SEBlock has the expected parameters."""
        channels = 256
        reduction = 16
        se = SEBlock(channels, reduction)
        
        # Check that parameters exist
        params = list(se.parameters())
        assert len(params) == 2  # fc1 and fc2 weights (no bias)
        
        # Check dimensions
        assert se.fc1.in_features == channels
        assert se.fc1.out_features == channels // reduction
        assert se.fc2.in_features == channels // reduction
        assert se.fc2.out_features == channels
        
    def test_se_block_eval_mode(self):
        """Test SEBlock in evaluation mode."""
        se = SEBlock(64, reduction=16)
        se.eval()
        
        x = torch.randn(2, 64, 32, 32)
        
        with torch.no_grad():
            out = se(x)
            
        assert out.shape == x.shape
        
    def test_se_block_batch_independence(self):
        """Test that SEBlock processes batch samples independently."""
        se = SEBlock(64, reduction=16)
        
        # Process samples individually
        x1 = torch.randn(1, 64, 32, 32)
        x2 = torch.randn(1, 64, 32, 32)
        out1 = se(x1)
        out2 = se(x2)
        
        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        out_batch = se(x_batch)
        
        # Results should match
        assert torch.allclose(out_batch[0:1], out1, atol=1e-6)
        assert torch.allclose(out_batch[1:2], out2, atol=1e-6)
        
    def test_se_block_small_channels(self):
        """Test SEBlock with small number of channels."""
        # Test with channels smaller than reduction ratio
        se = SEBlock(8, reduction=16)
        x = torch.randn(2, 8, 16, 16)
        
        out = se(x)
        
        assert out.shape == x.shape
        # Should have at least 1 channel in bottleneck
        assert se.fc1.out_features >= 1
        
    def test_se_block_channel_weights_range(self):
        """Test that channel weights are in valid range [0, 1]."""
        se = SEBlock(64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        
        # Extract channel weights by modifying forward pass
        batch_size, channels, _, _ = x.size()
        squeezed = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        excited = se.fc1(squeezed)
        excited = torch.nn.functional.relu(excited, inplace=True)
        excited = se.fc2(excited)
        weights = torch.sigmoid(excited)
        
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        
    def test_se_block_deterministic(self):
        """Test that SEBlock produces deterministic outputs."""
        se = SEBlock(64, reduction=16)
        se.eval()
        
        x = torch.randn(2, 64, 32, 32)
        
        with torch.no_grad():
            out1 = se(x)
            out2 = se(x)
            
        assert torch.allclose(out1, out2)
        
    def test_se_block_different_reductions(self):
        """Test SEBlock with different reduction ratios."""
        channels = 64
        
        for reduction in [4, 8, 16, 32]:
            se = SEBlock(channels, reduction)
            x = torch.randn(2, channels, 32, 32)
            out = se(x)
            
            assert out.shape == x.shape
            assert se.fc1.out_features == max(channels // reduction, 1)
