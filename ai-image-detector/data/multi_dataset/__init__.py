"""Multi-dataset support for training on multiple datasets simultaneously."""

from .loader import MultiDatasetLoader
from .registry import DatasetRegistry

__all__ = ['MultiDatasetLoader', 'DatasetRegistry']
