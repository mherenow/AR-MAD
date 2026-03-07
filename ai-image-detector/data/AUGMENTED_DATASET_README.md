# AugmentedDataset Wrapper

## Overview

The `AugmentedDataset` is a wrapper class that applies augmentations to base datasets. It integrates with the robustness augmentation pipeline to improve model robustness against common image transformations.

## Features

- **Per-Image Augmentations**: Applies RobustnessAugmentation (JPEG compression, blur, noise) to individual images
- **Configurable Probability**: Control how often augmentations are applied with `aug_prob` parameter
- **Format Compatibility**: Supports both 2-tuple `(image, label)` and 3-tuple `(image, label, metadata)` dataset formats
- **Transparent Wrapping**: Maintains the same interface as the base dataset

## Usage

### Basic Usage

```python
from data.synthbuster_loader import SynthBusterDataset
from data.augmentation.robustness import RobustnessAugmentation
from data.augmented_dataset import AugmentedDataset

# Create base dataset
base_dataset = SynthBusterDataset('data/synthbuster')

# Create robustness augmentation
robustness_aug = RobustnessAugmentation(
    jpeg_prob=0.3,
    blur_prob=0.3,
    noise_prob=0.3,
    severity_range=(1, 5)
)

# Wrap with AugmentedDataset
augmented_dataset = AugmentedDataset(
    base_dataset,
    robustness_aug=robustness_aug,
    aug_prob=0.5  # Apply augmentation to 50% of samples
)

# Use like any PyTorch dataset
image, label, generator = augmented_dataset[0]
```

### With DataLoader

```python
from torch.utils.data import DataLoader

# Create augmented dataset
augmented_dataset = AugmentedDataset(
    base_dataset,
    robustness_aug=robustness_aug,
    aug_prob=1.0  # Always apply augmentation during training
)

# Create DataLoader
train_loader = DataLoader(
    augmented_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for images, labels, generators in train_loader:
    # images are augmented according to aug_prob
    outputs = model(images)
    loss = criterion(outputs, labels)
    # ...
```

### Without Augmentation

```python
# Create dataset without augmentation (for validation/testing)
dataset = AugmentedDataset(base_dataset, aug_prob=0.0)

# Or simply use the base dataset directly
dataset = base_dataset
```

### Custom Augmentation Configuration

```python
# Aggressive augmentation for training
train_aug = RobustnessAugmentation(
    jpeg_prob=0.5,
    blur_prob=0.5,
    noise_prob=0.5,
    severity_range=(2, 5)  # Higher severity
)

train_dataset = AugmentedDataset(
    base_dataset,
    robustness_aug=train_aug,
    aug_prob=1.0
)

# Mild augmentation for validation
val_aug = RobustnessAugmentation(
    jpeg_prob=0.2,
    blur_prob=0.2,
    noise_prob=0.2,
    severity_range=(1, 3)  # Lower severity
)

val_dataset = AugmentedDataset(
    base_dataset,
    robustness_aug=val_aug,
    aug_prob=0.3
)
```

## Parameters

### AugmentedDataset

- **base_dataset** (Dataset): Base dataset to wrap. Must implement `__getitem__` and `__len__`.
- **robustness_aug** (RobustnessAugmentation, optional): RobustnessAugmentation instance. If None, no augmentation is applied.
- **aug_prob** (float, default=1.0): Probability of applying augmentation to each sample. Must be in [0, 1].

### RobustnessAugmentation

- **jpeg_prob** (float, default=0.3): Probability of applying JPEG compression
- **blur_prob** (float, default=0.3): Probability of applying Gaussian blur
- **noise_prob** (float, default=0.3): Probability of applying Gaussian noise
- **severity_range** (tuple, default=(1, 5)): Range of severity levels (1=mild, 5=severe)

## Dataset Format Compatibility

The AugmentedDataset automatically detects and maintains the format of the base dataset:

### 2-Tuple Format
```python
# Base dataset returns: (image, label)
image, label = augmented_dataset[0]
```

### 3-Tuple Format
```python
# Base dataset returns: (image, label, metadata)
image, label, metadata = augmented_dataset[0]
```

## CutMix and MixUp Augmentations

**Important**: CutMix and MixUp augmentations require pairs of images and should be applied at the **batch level** in the training loop, not in the dataset wrapper.

```python
from data.augmentation.cutmix import CutMixAugmentation
from data.augmentation.mixup import MixUpAugmentation

# Initialize augmentations
cutmix = CutMixAugmentation(alpha=1.0, prob=0.5)
mixup = MixUpAugmentation(alpha=0.2, prob=0.5)

# Apply in training loop
for images, labels, _ in train_loader:
    # Randomly choose between CutMix and MixUp
    if random.random() < 0.5:
        images, labels = cutmix(images, labels, images2, labels2)
    else:
        images, labels = mixup(images, labels, images2, labels2)
    
    # Forward pass with mixed images
    outputs = model(images)
    loss = criterion(outputs, labels)
```

## Best Practices

1. **Training**: Use `aug_prob=1.0` with moderate augmentation settings
2. **Validation**: Use `aug_prob=0.0` or no augmentation wrapper
3. **Testing**: Use base dataset without augmentation
4. **Severity Levels**: Start with `severity_range=(1, 3)` and increase if needed
5. **Probability**: Balance augmentation probabilities based on your data distribution

## Example: Complete Training Setup

```python
from data.synthbuster_loader import SynthBusterDataset
from data.augmentation.robustness import RobustnessAugmentation
from data.augmented_dataset import AugmentedDataset
from torch.utils.data import DataLoader

# Training dataset with augmentation
train_base = SynthBusterDataset('data/train', native_resolution=False)
train_aug = RobustnessAugmentation(
    jpeg_prob=0.3,
    blur_prob=0.3,
    noise_prob=0.3,
    severity_range=(1, 5)
)
train_dataset = AugmentedDataset(train_base, robustness_aug=train_aug, aug_prob=1.0)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation dataset without augmentation
val_base = SynthBusterDataset('data/val', native_resolution=False)
val_dataset = AugmentedDataset(val_base, aug_prob=0.0)  # No augmentation
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Test dataset (use base dataset directly)
test_dataset = SynthBusterDataset('data/test', native_resolution=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

## Testing

Run the test suite to verify the implementation:

```bash
python -m pytest ai-image-detector/data/test_augmented_dataset.py -v
```

## Related Modules

- `data/augmentation/robustness.py`: RobustnessAugmentation implementation
- `data/augmentation/cutmix.py`: CutMixAugmentation (batch-level)
- `data/augmentation/mixup.py`: MixUpAugmentation (batch-level)
- `data/synthbuster_loader.py`: SynthBusterDataset base dataset
- `data/coco_loader.py`: COCO2017Dataset base dataset
