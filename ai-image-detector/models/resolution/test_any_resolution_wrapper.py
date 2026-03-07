"""
Unit tests for AnyResolutionWrapper module.
"""

import pytest
import torch
import torch.nn as nn
from .any_resolution_wrapper import AnyResolutionWrapper


class DummyModel(nn.Module):
    """Simple dummy model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class TestAnyResolutionWrapper:
    """Test suite for AnyResolutionWrapper."""
    
    def test_initialization(self):
        """Test wrapper initialization with valid parameters."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        assert wrapper.tile_size == 256
        assert wrapper.stride == 128
        assert wrapper.aggregation == 'average'
        assert wrapper.model is model
    
    def test_initialization_with_voting(self):
        """Test wrapper initialization with voting aggregation."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(
            model,
            tile_size=256,
            stride=128,
            aggregation='voting'
        )
        
        assert wrapper.aggregation == 'voting'
    
    def test_invalid_aggregation(self):
        """Test that invalid aggregation method raises error."""
        model = DummyModel()
        with pytest.raises(AssertionError):
            AnyResolutionWrapper(model, aggregation='invalid')
    
    def test_invalid_tile_size(self):
        """Test that invalid tile_size raises error."""
        model = DummyModel()
        with pytest.raises(AssertionError):
            AnyResolutionWrapper(model, tile_size=0)
    
    def test_invalid_stride(self):
        """Test that stride > tile_size raises error."""
        model = DummyModel()
        with pytest.raises(AssertionError):
            AnyResolutionWrapper(model, tile_size=256, stride=300)
    
    def test_small_image_direct_processing(self):
        """Test that small images are processed directly without tiling."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        # Image smaller than tile_size
        x = torch.randn(2, 3, 128, 128)
        output = wrapper(x)
        
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_exact_tile_size_image(self):
        """Test image exactly matching tile_size."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        x = torch.randn(2, 3, 256, 256)
        output = wrapper(x)
        
        assert output.shape == (2, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_large_image_tiling(self):
        """Test that large images are tiled and processed."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        # Image larger than tile_size
        x = torch.randn(1, 3, 512, 512)
        output = wrapper(x)
        
        assert output.shape == (1, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_extract_tiles_shape(self):
        """Test tile extraction produces correct shapes."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        x = torch.randn(1, 3, 512, 512)
        tiles, positions = wrapper._extract_tiles(x)
        
        # For 512x512 image with tile_size=256 and stride=128:
        # n_rows = ceil((512 - 256) / 128) + 1 = 3
        # n_cols = ceil((512 - 256) / 128) + 1 = 3
        # Total tiles = 3 * 3 = 9
        expected_num_tiles = 9
        
        assert tiles.shape == (expected_num_tiles, 3, 256, 256)
        assert len(positions) == expected_num_tiles
    
    def test_extract_tiles_positions(self):
        """Test that tile positions are calculated correctly."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        x = torch.randn(1, 3, 512, 512)
        tiles, positions = wrapper._extract_tiles(x)
        
        # Check first tile position
        assert positions[0] == (0, 0)
        
        # Check that all positions are valid
        for row, col in positions:
            assert row >= 0 and row <= 512 - 256
            assert col >= 0 and col <= 512 - 256
    
    def test_compute_tile_weights(self):
        """Test tile weight computation."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        positions = [(0, 0), (0, 256), (256, 0), (256, 256)]
        weights = wrapper._compute_tile_weights(positions, 512, 512)
        
        assert len(weights) == 4
        assert torch.all(weights > 0)
        assert torch.all(weights <= 1)
        
        # Center tile should have highest weight
        # For 512x512 image, center is at (256, 256)
        # Tile at (256, 256) has center at (256 + 128, 256 + 128) = (384, 384)
        # This is closest to image center
        center_idx = 3  # Position (256, 256)
        assert weights[center_idx] >= weights[0]  # Higher than corner tile
    
    def test_weighted_average_aggregation(self):
        """Test weighted average aggregation."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(
            model,
            tile_size=256,
            stride=128,
            aggregation='average'
        )
        
        # Create dummy tile predictions
        tile_predictions = torch.tensor([
            [0.2], [0.4], [0.6], [0.8]
        ])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        batch_size = 1
        
        result = wrapper._aggregate_predictions(tile_predictions, weights, batch_size)
        
        # With equal weights, should be simple average
        expected = (0.2 + 0.4 + 0.6 + 0.8) / 4
        assert result.shape == (1, 1)
        assert torch.isclose(result[0, 0], torch.tensor(expected), atol=1e-5)
    
    def test_voting_aggregation(self):
        """Test voting aggregation."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(
            model,
            tile_size=256,
            stride=128,
            aggregation='voting'
        )
        
        # Create dummy tile predictions (3 above 0.5, 1 below)
        tile_predictions = torch.tensor([
            [0.2], [0.6], [0.7], [0.8]
        ])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        batch_size = 1
        
        result = wrapper._aggregate_predictions(tile_predictions, weights, batch_size)
        
        # Majority vote should be 1.0 (3 out of 4 above 0.5)
        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0
    
    def test_batch_processing(self):
        """Test processing multiple images in a batch."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        # Batch of 3 images
        x = torch.randn(3, 3, 512, 512)
        output = wrapper(x)
        
        assert output.shape == (3, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_non_square_image(self):
        """Test processing non-square images."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        # Non-square image
        x = torch.randn(1, 3, 512, 768)
        output = wrapper(x)
        
        assert output.shape == (1, 1)
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_small_image_with_padding(self):
        """Test that small images are padded correctly."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        # Very small image that needs padding
        x = torch.randn(1, 3, 100, 100)
        tiles, positions = wrapper._extract_tiles(x)
        
        # Should be padded to at least tile_size
        assert tiles.shape[2] == 256
        assert tiles.shape[3] == 256
    
    def test_gradient_flow(self):
        """Test that gradients flow through the wrapper."""
        model = DummyModel()
        wrapper = AnyResolutionWrapper(model, tile_size=256, stride=128)
        
        x = torch.randn(1, 3, 512, 512, requires_grad=True)
        output = wrapper(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
