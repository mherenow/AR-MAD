"""
AnyResolutionWrapper module for processing images of arbitrary resolution.

This module implements a tiling strategy to process large images by dividing them
into overlapping tiles, processing each tile independently, and aggregating the results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Literal
import math


class AnyResolutionWrapper(nn.Module):
    """
    Wrapper for processing images of any resolution using tiling strategy.
    
    This wrapper divides large images into overlapping tiles, processes each tile
    independently through the base model, and aggregates predictions using weighted
    averaging or voting. Tiles are weighted by distance from center using Gaussian
    weighting to reduce edge artifacts.
    
    Args:
        model: Base model to wrap (should accept images and return predictions)
        tile_size: Size of tiles (default: 256)
        stride: Stride between tiles (default: 128, 50% overlap)
        aggregation: Aggregation method ('average' or 'voting', default: 'average')
        sigma: Gaussian weighting sigma for distance-based weighting (default: 0.5)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tile_size: int = 256,
        stride: int = 128,
        aggregation: Literal['average', 'voting'] = 'average',
        sigma: float = 0.5
    ):
        super().__init__()
        self.model = model
        self.tile_size = tile_size
        self.stride = stride
        self.aggregation = aggregation
        self.sigma = sigma
        
        # Validate parameters
        assert tile_size > 0, "tile_size must be positive"
        assert stride > 0, "stride must be positive"
        assert stride <= tile_size, "stride should not exceed tile_size"
        assert aggregation in ['average', 'voting'], "aggregation must be 'average' or 'voting'"
    
    def _extract_tiles(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Extract overlapping tiles from input image.
        
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            tiles: Extracted tiles (N, C, tile_size, tile_size)
            positions: List of (row, col) positions for each tile
        """
        B, C, H, W = x.shape
        
        # If image is smaller than tile_size, pad it
        if H < self.tile_size or W < self.tile_size:
            pad_h = max(0, self.tile_size - H)
            pad_w = max(0, self.tile_size - W)
            
            # Use reflect padding if possible, otherwise use constant padding
            # Reflect padding requires padding size < input dimension
            if pad_h < H and pad_w < W:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                # Use constant padding (zero padding) for large padding amounts
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
            H, W = x.shape[2], x.shape[3]
        
        # Calculate number of tiles
        n_rows = max(1, math.ceil((H - self.tile_size) / self.stride) + 1)
        n_cols = max(1, math.ceil((W - self.tile_size) / self.stride) + 1)
        
        tiles = []
        positions = []
        
        for i in range(n_rows):
            for j in range(n_cols):
                # Calculate tile position
                row_start = min(i * self.stride, H - self.tile_size)
                col_start = min(j * self.stride, W - self.tile_size)
                row_end = row_start + self.tile_size
                col_end = col_start + self.tile_size
                
                # Extract tile
                tile = x[:, :, row_start:row_end, col_start:col_end]
                tiles.append(tile)
                positions.append((row_start, col_start))
        
        # Stack tiles (B * num_tiles, C, tile_size, tile_size)
        tiles = torch.cat(tiles, dim=0)
        
        return tiles, positions
    
    def _compute_tile_weights(
        self,
        positions: List[Tuple[int, int]],
        image_height: int,
        image_width: int
    ) -> torch.Tensor:
        """
        Compute Gaussian weights for tiles based on distance from image center.
        
        Tiles closer to the center of the image receive higher weights to reduce
        edge artifacts and improve aggregation quality.
        
        Args:
            positions: List of (row, col) positions for each tile
            image_height: Height of the original image
            image_width: Width of the original image
        
        Returns:
            weights: Tile weights (num_tiles,)
        """
        # Calculate image center
        center_h = image_height / 2
        center_w = image_width / 2
        
        weights = []
        for row, col in positions:
            # Calculate tile center
            tile_center_h = row + self.tile_size / 2
            tile_center_w = col + self.tile_size / 2
            
            # Calculate normalized distance from image center
            dist_h = (tile_center_h - center_h) / image_height
            dist_w = (tile_center_w - center_w) / image_width
            dist = math.sqrt(dist_h ** 2 + dist_w ** 2)
            
            # Gaussian weighting: exp(-dist^2 / (2 * sigma^2))
            weight = math.exp(-(dist ** 2) / (2 * self.sigma ** 2))
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _aggregate_predictions(
        self,
        tile_predictions: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Aggregate tile predictions using weighted averaging or voting.
        
        Args:
            tile_predictions: Predictions for all tiles (B * num_tiles, 1)
            weights: Weights for each tile (num_tiles,)
            batch_size: Original batch size
        
        Returns:
            prediction: Aggregated prediction (B, 1)
        """
        # Reshape predictions to (B, num_tiles, 1)
        num_tiles = len(weights)
        tile_predictions = tile_predictions.view(batch_size, num_tiles, -1)
        
        # Move weights to same device as predictions
        weights = weights.to(tile_predictions.device)
        
        if self.aggregation == 'average':
            # Weighted average: pred = Σ(w_i * pred_i) / Σ(w_i)
            weights = weights.view(1, num_tiles, 1)  # (1, num_tiles, 1)
            weighted_preds = tile_predictions * weights  # (B, num_tiles, 1)
            prediction = weighted_preds.sum(dim=1) / weights.sum()  # (B, 1)
        
        elif self.aggregation == 'voting':
            # Majority voting: threshold at 0.5 and take majority vote
            votes = (tile_predictions > 0.5).float()  # (B, num_tiles, 1)
            weights = weights.view(1, num_tiles, 1)  # (1, num_tiles, 1)
            weighted_votes = votes * weights  # (B, num_tiles, 1)
            
            # Compute weighted vote ratio
            vote_ratio = weighted_votes.sum(dim=1) / weights.sum()  # (B, 1)
            prediction = (vote_ratio > 0.5).float()  # (B, 1)
        
        return prediction
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image of any size using tiling strategy.
        
        For images smaller than or equal to tile_size, processes directly.
        For larger images, divides into overlapping tiles, processes each tile,
        and aggregates predictions using the configured aggregation method.
        
        Args:
            x: Input image of any size (B, C, H, W)
        
        Returns:
            prediction: Aggregated prediction (B, 1)
        """
        B, C, H, W = x.shape
        
        # If image is small enough, process directly
        if H <= self.tile_size and W <= self.tile_size:
            return self.model(x)
        
        # Extract tiles
        tiles, positions = self._extract_tiles(x)  # (B * num_tiles, C, tile_size, tile_size)
        
        # Process tiles through model
        tile_predictions = self.model(tiles)  # (B * num_tiles, 1)
        
        # Compute tile weights
        weights = self._compute_tile_weights(positions, H, W)  # (num_tiles,)
        
        # Aggregate predictions
        prediction = self._aggregate_predictions(tile_predictions, weights, B)  # (B, 1)
        
        return prediction
