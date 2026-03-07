"""
Unit tests for NoiseImprintBranch module.
"""

import pytest
import torch

from .noise_branch import NoiseImprintBranch


class TestNoiseImprintBranch:
    """Test suite for NoiseImprintBranch."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        branch = NoiseImprintBranch()
        
        assert branch.input_channels == 3
        assert branch.feature_dim == 256
        assert branch.enable_attribution is False
        assert branch.num_generators == 10
        assert branch.attribution_head is None
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        branch = NoiseImprintBranch(
            input_channels=1,
            feature_dim=128,
            enable_attribution=True,
            num_generators=5
        )
        
        assert branch.input_channels == 1
        assert branch.feature_dim == 128
        assert branch.enable_attribution is True
        assert branch.num_generators == 5
        assert branch.attribution_head is not None
    
    def test_forward_without_attribution(self):
        """Test forward pass without attribution."""
        branch = NoiseImprintBranch(feature_dim=256, enable_attribution=False)
        residual = torch.randn(2, 3, 256, 256)
        
        output = branch(residual)
        
        # Should return only features
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 256)
    
    def test_forward_with_attribution(self):
        """Test forward pass with attribution."""
        branch = NoiseImprintBranch(
            feature_dim=256,
            enable_attribution=True,
            num_generators=5
        )
        residual = torch.randn(2, 3, 256, 256)
        
        features, attribution = branch(residual)
        
        # Should return features and attribution
        assert isinstance(features, torch.Tensor)
        assert isinstance(attribution, torch.Tensor)
        assert features.shape == (2, 256)
        assert attribution.shape == (2, 5)
    
    def test_attribution_is_probability_distribution(self):
        """Test that attribution output is a valid probability distribution."""
        branch = NoiseImprintBranch(
            feature_dim=256,
            enable_attribution=True,
            num_generators=5
        )
        residual = torch.randn(4, 3, 256, 256)
        
        _, attribution = branch(residual)
        
        # Check that probabilities sum to 1 for each sample
        prob_sums = attribution.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-6)
        
        # Check that all probabilities are in [0, 1]
        assert (attribution >= 0).all()
        assert (attribution <= 1).all()
    
    def test_different_input_sizes(self):
        """Test that branch handles different input sizes."""
        branch = NoiseImprintBranch(feature_dim=256)
        
        # Test various input sizes
        sizes = [(128, 128), (256, 256), (512, 512), (224, 224)]
        
        for h, w in sizes:
            residual = torch.randn(2, 3, h, w)
            output = branch(residual)
            
            # Output should always be (batch_size, feature_dim)
            assert output.shape == (2, 256), f"Failed for size {h}x{w}"
    
    def test_batch_size_variations(self):
        """Test that branch handles different batch sizes."""
        branch = NoiseImprintBranch(feature_dim=256)
        
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            residual = torch.randn(batch_size, 3, 256, 256)
            output = branch(residual)
            
            assert output.shape == (batch_size, 256)
    
    def test_custom_feature_dim(self):
        """Test custom feature dimensions."""
        feature_dims = [64, 128, 256, 512]
        
        for dim in feature_dims:
            branch = NoiseImprintBranch(feature_dim=dim)
            residual = torch.randn(2, 3, 256, 256)
            output = branch(residual)
            
            assert output.shape == (2, dim)
    
    def test_custom_input_channels(self):
        """Test custom input channels."""
        # Test single channel (grayscale residual)
        branch = NoiseImprintBranch(input_channels=1, feature_dim=256)
        residual = torch.randn(2, 1, 256, 256)
        output = branch(residual)
        
        assert output.shape == (2, 256)
        
        # Test 4 channels (e.g., RGBA)
        branch = NoiseImprintBranch(input_channels=4, feature_dim=256)
        residual = torch.randn(2, 4, 256, 256)
        output = branch(residual)
        
        assert output.shape == (2, 256)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        branch = NoiseImprintBranch(feature_dim=256, enable_attribution=False)
        residual = torch.randn(2, 3, 256, 256, requires_grad=True)
        
        output = branch(residual)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert residual.grad is not None
        assert not torch.isnan(residual.grad).any()
    
    def test_gradient_flow_with_attribution(self):
        """Test gradient flow with attribution head."""
        branch = NoiseImprintBranch(
            feature_dim=256,
            enable_attribution=True,
            num_generators=5
        )
        residual = torch.randn(2, 3, 256, 256, requires_grad=True)
        
        features, attribution = branch(residual)
        loss = features.sum() + attribution.sum()
        loss.backward()
        
        # Check that gradients exist
        assert residual.grad is not None
        assert not torch.isnan(residual.grad).any()
    
    def test_eval_mode(self):
        """Test that eval mode works correctly."""
        branch = NoiseImprintBranch(feature_dim=256)
        branch.eval()
        
        residual = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            output = branch(residual)
        
        assert output.shape == (2, 256)
    
    def test_train_mode(self):
        """Test that train mode works correctly."""
        branch = NoiseImprintBranch(feature_dim=256)
        branch.train()
        
        residual = torch.randn(2, 3, 256, 256)
        output = branch(residual)
        
        assert output.shape == (2, 256)
    
    def test_deterministic_output(self):
        """Test that output is deterministic in eval mode."""
        branch = NoiseImprintBranch(feature_dim=256)
        branch.eval()
        
        residual = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            output1 = branch(residual)
            output2 = branch(residual)
        
        assert torch.allclose(output1, output2)
    
    def test_different_attribution_classes(self):
        """Test different numbers of generator classes."""
        num_classes_list = [2, 5, 10, 20]
        
        for num_classes in num_classes_list:
            branch = NoiseImprintBranch(
                feature_dim=256,
                enable_attribution=True,
                num_generators=num_classes
            )
            residual = torch.randn(2, 3, 256, 256)
            
            _, attribution = branch(residual)
            
            assert attribution.shape == (2, num_classes)
    
    def test_edge_case_small_input(self):
        """Test with very small input size."""
        branch = NoiseImprintBranch(feature_dim=256)
        
        # Minimum size that won't cause issues with stride 2 convolutions
        # After 3 stride-2 convolutions: 32 -> 16 -> 8 -> 4
        residual = torch.randn(2, 3, 32, 32)
        output = branch(residual)
        
        assert output.shape == (2, 256)
    
    def test_residual_range(self):
        """Test that branch handles residuals in typical range."""
        branch = NoiseImprintBranch(feature_dim=256)
        
        # Residuals are typically in range [-1, 1] or smaller
        residual = torch.randn(2, 3, 256, 256) * 0.1  # Small residuals
        output = branch(residual)
        
        assert output.shape == (2, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_zero_residual(self):
        """Test with zero residual (edge case)."""
        branch = NoiseImprintBranch(feature_dim=256)
        
        residual = torch.zeros(2, 3, 256, 256)
        output = branch(residual)
        
        assert output.shape == (2, 256)
        assert not torch.isnan(output).any()
    
    def test_module_parameters(self):
        """Test that module has trainable parameters."""
        branch = NoiseImprintBranch(feature_dim=256, enable_attribution=False)
        
        params = list(branch.parameters())
        assert len(params) > 0
        
        # Check that parameters require gradients
        trainable_params = [p for p in params if p.requires_grad]
        assert len(trainable_params) > 0
    
    def test_module_parameters_with_attribution(self):
        """Test that attribution head adds parameters."""
        branch_without = NoiseImprintBranch(feature_dim=256, enable_attribution=False)
        branch_with = NoiseImprintBranch(feature_dim=256, enable_attribution=True, num_generators=5)
        
        params_without = sum(p.numel() for p in branch_without.parameters())
        params_with = sum(p.numel() for p in branch_with.parameters())
        
        # Branch with attribution should have more parameters
        assert params_with > params_without
    
    def test_device_compatibility_cpu(self):
        """Test that branch works on CPU."""
        branch = NoiseImprintBranch(feature_dim=256)
        residual = torch.randn(2, 3, 256, 256)
        
        output = branch(residual)
        
        assert output.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda(self):
        """Test that branch works on CUDA."""
        branch = NoiseImprintBranch(feature_dim=256).cuda()
        residual = torch.randn(2, 3, 256, 256).cuda()
        
        output = branch(residual)
        
        assert output.device.type == 'cuda'
    
    def test_output_dtype(self):
        """Test that output has correct dtype."""
        branch = NoiseImprintBranch(feature_dim=256)
        residual = torch.randn(2, 3, 256, 256, dtype=torch.float32)
        
        output = branch(residual)
        
        assert output.dtype == torch.float32
    
    def test_cnn_architecture_layers(self):
        """Test that CNN has the expected architecture."""
        branch = NoiseImprintBranch(feature_dim=256)
        
        # Check that all expected layers exist
        assert hasattr(branch, 'conv1')
        assert hasattr(branch, 'conv2')
        assert hasattr(branch, 'conv3')
        assert hasattr(branch, 'conv4')
        assert hasattr(branch, 'global_pool')
        assert hasattr(branch, 'fc')
    
    def test_global_average_pooling(self):
        """Test that global average pooling produces correct output size."""
        branch = NoiseImprintBranch(feature_dim=256)
        
        # Test with different input sizes
        for size in [128, 256, 512]:
            residual = torch.randn(2, 3, size, size)
            output = branch(residual)
            
            # Output should be same regardless of input spatial size
            assert output.shape == (2, 256)
