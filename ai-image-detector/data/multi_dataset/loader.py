"""
MultiDatasetLoader for weighted sampling from multiple datasets.

This module implements a loader that samples from multiple datasets with
configurable weights, enabling balanced training across different data distributions.
"""

from typing import Dict, Iterator, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MultiDatasetLoader:
    """
    Loader for multiple datasets with weighted sampling.
    
    Enables training on multiple datasets simultaneously by sampling datasets
    according to configurable weights. For each batch, a dataset is selected
    probabilistically based on normalized weights, and a batch is sampled from
    that dataset.
    
    Args:
        datasets: Dictionary mapping dataset names to Dataset objects
        weights: Dictionary mapping dataset names to sampling weights
        batch_size: Batch size for sampling (default: 32)
        shuffle: Whether to shuffle datasets (default: True)
        num_workers: Number of worker processes for data loading (default: 0)
        drop_last: Whether to drop the last incomplete batch (default: False)
    
    Example:
        >>> datasets = {
        ...     'synthbuster': SynthBusterDataset(...),
        ...     'coco': CocoDataset(...)
        ... }
        >>> weights = {'synthbuster': 0.7, 'coco': 0.3}
        >>> loader = MultiDatasetLoader(datasets, weights, batch_size=32)
        >>> for images, labels, dataset_name in loader:
        ...     # Train on batch from dataset_name
        ...     pass
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        weights: Dict[str, float],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False
    ):
        """Initialize MultiDatasetLoader with datasets and weights."""
        if not datasets:
            raise ValueError("datasets dictionary cannot be empty")
        
        if set(datasets.keys()) != set(weights.keys()):
            raise ValueError("datasets and weights must have the same keys")
        
        if any(w <= 0 for w in weights.values()):
            raise ValueError("all weights must be positive")
        
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        # Normalize weights to probabilities
        self.dataset_names = list(datasets.keys())
        total_weight = sum(weights.values())
        self.probabilities = np.array([
            weights[name] / total_weight for name in self.dataset_names
        ])
        
        # Create DataLoader for each dataset
        self.dataloaders = {
            name: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last
            )
            for name, dataset in datasets.items()
        }
        
        # Calculate total number of batches (based on largest dataset)
        self.num_batches = max(
            len(loader) for loader in self.dataloaders.values()
        )
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, str]]:
        """
        Iterate over batches sampled from multiple datasets.
        
        For each batch:
        1. Sample a dataset according to normalized weights
        2. Sample a batch from the selected dataset
        3. Yield (images, labels, dataset_name)
        
        Yields:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            dataset_name: Name of the source dataset
        """
        # Create iterators for each dataset
        iterators = {
            name: iter(loader) for name, loader in self.dataloaders.items()
        }
        
        for _ in range(self.num_batches):
            # Sample dataset based on probabilities
            dataset_name = np.random.choice(
                self.dataset_names, p=self.probabilities
            )
            
            # Get batch from selected dataset
            try:
                batch = next(iterators[dataset_name])
            except StopIteration:
                # If iterator exhausted, create new iterator and sample again
                iterators[dataset_name] = iter(self.dataloaders[dataset_name])
                batch = next(iterators[dataset_name])
            
            # Unpack batch (assuming standard format: images, labels)
            if len(batch) == 2:
                images, labels = batch
            else:
                # Handle other batch formats if needed
                images, labels = batch[0], batch[1]
            
            yield images, labels, dataset_name
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about registered datasets.
        
        Returns:
            Dictionary with dataset names as keys and info dictionaries as values.
            Each info dict contains:
                - size: Number of samples in the dataset
                - probability: Sampling probability
                - num_batches: Number of batches in the dataset
        """
        return {
            name: {
                'size': len(self.datasets[name]),
                'probability': float(self.probabilities[i]),
                'num_batches': len(self.dataloaders[name])
            }
            for i, name in enumerate(self.dataset_names)
        }
