# Data Module Dependencies

## Overview

The `data/` module is responsible for loading and preprocessing the SynthBuster dataset. It is designed to be **self-contained** with minimal dependencies on other workspace modules.

## Module: synthbuster_loader.py

### External Dependencies

The module depends only on standard PyTorch and image processing libraries:

- **torch** (>=2.0.0): PyTorch Dataset class and tensor operations
- **torchvision** (>=0.15.0): Image transforms (Resize, ToTensor)
- **PIL** (Pillow >=10.0.0): Image loading and format conversion
- **pathlib**: File system path handling (Python standard library)
- **warnings**: Warning messages for corrupted images (Python standard library)

### Workspace Dependencies

**NONE** - This module has zero dependencies on other workspace modules.

The module does NOT import or depend on:
- ❌ `ai-image-detector.utils.config_loader`
- ❌ `ai-image-detector.models`
- ❌ `ai-image-detector.training`
- ❌ `ai-image-detector.evaluation`

### Design Rationale

The data loader is intentionally self-contained to:

1. **Portability**: Can be copied to other projects without workspace dependencies
2. **Testability**: Can be tested in isolation without mocking workspace modules
3. **Reusability**: Can be integrated into different training pipelines
4. **Simplicity**: Easier to understand and maintain with minimal coupling

### Integration with Workspace

While the data loader itself has no workspace dependencies, it is used by other modules:

#### Used By

1. **training/__main__.py**
   ```python
   from data.synthbuster_loader import SynthBusterDataset, create_train_val_split
   
   # Create dataset
   full_dataset = SynthBusterDataset(root_dir=config['dataset']['root_dir'])
   
   # Create train/val split
   train_paths, val_paths = create_train_val_split(root_dir, val_ratio=0.2)
   ```

2. **evaluation/__main__.py**
   ```python
   from data.synthbuster_loader import SynthBusterDataset
   
   # Create test dataset
   test_dataset = SynthBusterDataset(root_dir=config['dataset']['root_dir'])
   ```

#### Configuration Flow

The data loader receives configuration parameters indirectly:

```
configs/default_config.yaml
         ↓
utils/config_loader.py (loads and validates)
         ↓
training/__main__.py or evaluation/__main__.py
         ↓
SynthBusterDataset(root_dir=config['dataset']['root_dir'])
```

**Key Point**: The data loader does NOT directly load configuration files. Configuration parameters are extracted by the training/evaluation scripts and passed as function arguments.

### Public API

The module exports three main components via `data/__init__.py`:

```python
from .synthbuster_loader import SynthBusterDataset

__all__ = ['SynthBusterDataset']
```

#### SynthBusterDataset

**Purpose**: PyTorch Dataset for loading SynthBuster images

**Constructor Parameters**:
- `root_dir` (str): Path to dataset root directory
- `transform` (Optional[transforms.Compose]): Custom image transforms (default: resize to 256x256)

**Returns**: Tuple of (image_tensor, label, generator_name)
- `image_tensor`: torch.Tensor of shape (3, 256, 256), normalized to [0, 1]
- `label`: int (0 for real/RAISE, 1 for fake)
- `generator_name`: str (e.g., "RAISE", "SD_v2", "GLIDE")

#### create_train_val_split

**Purpose**: Split dataset into training and validation sets

**Parameters**:
- `root_dir` (str): Path to dataset root directory
- `val_ratio` (float): Validation split ratio (default: 0.2)
- `seed` (int): Random seed for reproducibility (default: 42)

**Returns**: Tuple of (train_paths, val_paths)

**Note**: SynthBuster is typically used as a test-only dataset. This function is provided for flexibility.

#### get_generator_subsets

**Purpose**: Organize dataset by generator name

**Parameters**:
- `root_dir` (str): Path to dataset root directory

**Returns**: Dict mapping generator names to lists of file paths

**Supported Generators**:
- SD_v2 (Stable Diffusion v2)
- GLIDE
- Firefly (Adobe Firefly)
- DALLE (DALL-E)
- Midjourney
- RAISE (real images)

### Workspace Conventions Followed

The module adheres to workspace conventions:

1. **Type Hints**: All functions use Python type hints
   ```python
   def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
   ```

2. **Docstrings**: Google-style docstrings with Args, Returns, Examples
   ```python
   """
   Load and return a sample from the dataset.
   
   Args:
       idx: Index of the sample to load
       
   Returns:
       Tuple of (image_tensor, label, generator_name)
   """
   ```

3. **Testing**: Comprehensive unit tests in `test_synthbuster_loader.py`

4. **Error Handling**: Graceful handling of corrupted images with warnings

5. **Module Structure**: Follows workspace package structure with `__init__.py` exports

### Future Considerations

If workspace utilities are needed in the future:

1. **Configuration Loading**: Could optionally accept config dict parameter
   ```python
   # Potential future enhancement
   dataset = SynthBusterDataset.from_config(config['dataset'])
   ```

2. **Logging**: Could integrate with workspace logging utility if created
   ```python
   # Potential future enhancement
   from ai-image-detector.utils.logger import get_logger
   logger = get_logger(__name__)
   ```

3. **Data Augmentation**: Could integrate with workspace augmentation pipeline
   ```python
   # Potential future enhancement
   from ai-image-detector.utils.augmentation import get_augmentation_pipeline
   ```

**Important**: Any such changes should maintain backward compatibility and keep the module usable as a standalone component.

## Summary

The `data/synthbuster_loader.py` module is a **self-contained, zero-dependency** data loader that integrates with the workspace through:

- ✅ Clean public API exported via `__init__.py`
- ✅ Configuration parameters passed from training/evaluation scripts
- ✅ Adherence to workspace conventions (type hints, docstrings, testing)
- ✅ No direct imports of workspace modules
- ✅ Maximum portability and reusability

This design makes the data loader both a reliable workspace component and a standalone utility that can be easily extracted for use in other projects.
