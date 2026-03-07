"""Dataset registry for managing multiple datasets."""

from typing import Dict, List, Type, Any, Optional
from torch.utils.data import Dataset


class DatasetRegistry:
    """
    Registry for managing multiple datasets.
    
    Provides a centralized way to register and retrieve datasets with their
    configurations. Supports extensibility by allowing any dataset class
    implementing the Dataset protocol to be registered.
    
    Usage:
        registry = DatasetRegistry()
        registry.register('synthbuster', SynthBusterDataset, config)
        dataset = registry.get('synthbuster')
        all_datasets = registry.list()
    
    Attributes:
        _datasets: Dictionary mapping dataset names to dataset instances
        _configs: Dictionary mapping dataset names to their configurations
    """
    
    def __init__(self):
        """Initialize an empty dataset registry."""
        self._datasets: Dict[str, Dataset] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        dataset_class: Type[Dataset],
        config: Dict[str, Any]
    ) -> None:
        """
        Register a dataset with its configuration.
        
        Args:
            name: Unique identifier for the dataset
            dataset_class: Dataset class to instantiate
            config: Configuration dictionary for dataset initialization
        
        Raises:
            ValueError: If dataset name is already registered
            TypeError: If dataset_class doesn't implement Dataset protocol
        
        Example:
            >>> registry = DatasetRegistry()
            >>> config = {'root_dir': 'data/synthbuster', 'transform': None}
            >>> registry.register('synthbuster', SynthBusterDataset, config)
        """
        if name in self._datasets:
            raise ValueError(f"Dataset '{name}' is already registered")
        
        # Verify dataset_class implements Dataset protocol
        if not issubclass(dataset_class, Dataset):
            raise TypeError(
                f"dataset_class must be a subclass of torch.utils.data.Dataset, "
                f"got {type(dataset_class)}"
            )
        
        # Instantiate the dataset with the provided configuration
        try:
            dataset_instance = dataset_class(**config)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate dataset '{name}' with provided config: {e}"
            )
        
        # Store the dataset instance and configuration
        self._datasets[name] = dataset_instance
        self._configs[name] = config.copy()
    
    def get(self, name: str) -> Dataset:
        """
        Retrieve a registered dataset.
        
        Args:
            name: Name of the dataset to retrieve
        
        Returns:
            Dataset instance
        
        Raises:
            KeyError: If dataset name is not registered
        
        Example:
            >>> dataset = registry.get('synthbuster')
        """
        if name not in self._datasets:
            raise KeyError(
                f"Dataset '{name}' not found. Available datasets: {self.list()}"
            )
        
        return self._datasets[name]
    
    def list(self) -> List[str]:
        """
        List all registered dataset names.
        
        Returns:
            List of dataset names in registration order
        
        Example:
            >>> registry.list()
            ['synthbuster', 'coco2017']
        """
        return list(self._datasets.keys())
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """
        Get the configuration for a registered dataset.
        
        Args:
            name: Name of the dataset
        
        Returns:
            Configuration dictionary (copy)
        
        Raises:
            KeyError: If dataset name is not registered
        
        Example:
            >>> config = registry.get_config('synthbuster')
        """
        if name not in self._configs:
            raise KeyError(
                f"Dataset '{name}' not found. Available datasets: {self.list()}"
            )
        
        return self._configs[name].copy()
    
    def unregister(self, name: str) -> None:
        """
        Unregister a dataset.
        
        Args:
            name: Name of the dataset to unregister
        
        Raises:
            KeyError: If dataset name is not registered
        
        Example:
            >>> registry.unregister('synthbuster')
        """
        if name not in self._datasets:
            raise KeyError(
                f"Dataset '{name}' not found. Available datasets: {self.list()}"
            )
        
        del self._datasets[name]
        del self._configs[name]
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a dataset is registered.
        
        Args:
            name: Name of the dataset to check
        
        Returns:
            True if dataset is registered, False otherwise
        
        Example:
            >>> registry.is_registered('synthbuster')
            True
        """
        return name in self._datasets
    
    def clear(self) -> None:
        """
        Clear all registered datasets.
        
        Example:
            >>> registry.clear()
            >>> registry.list()
            []
        """
        self._datasets.clear()
        self._configs.clear()
    
    def __len__(self) -> int:
        """Return the number of registered datasets."""
        return len(self._datasets)
    
    def __contains__(self, name: str) -> bool:
        """Check if a dataset is registered using 'in' operator."""
        return name in self._datasets
    
    def __repr__(self) -> str:
        """Return string representation of the registry."""
        return f"DatasetRegistry(datasets={self.list()})"
