# Variable-Size Collate Function

## Overview

The `variable_size_collate_fn` is a custom collate function for PyTorch DataLoader that handles batches of images with different dimensions. This is essential when using `native_resolution=True` mode, where images preserve their original dimensions instead of being resized to a fixed size.

## Why Do We Need This?

PyTorch's default collate function (`torch.utils.data.default_collate`) expects all tensors in a batch to have the same shape so it can stack them into a single tensor. When images have different sizes (e.g., 256×256, 512×512, 384×384), the default collate function will fail with a size mismatch error.

The `variable_size_collate_fn` solves this by:
1. **Detecting size differences**: Checks if all images in the batch have the same dimensions
2. **Smart handling**: 
   - If all same size → stacks into a single tensor (efficient, backward compatible)
   - If different sizes → returns a list of tensors (handles variable sizes)

## Usage

### Basic Usage with DataLoader

```python
from torch.utils.data import DataLoader
from data import SynthBusterDataset, variable_size_collate_fn

# Create dataset with native_resolution=True
dataset = SynthBusterDataset(
    root_dir='path/to/synthbuster',
    native_resolution=True  # Preserve original image dimensions
)

# Create DataLoader with custom collate function
loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=variable_size_collate_fn,  # Use custom collate function
    shuffle=True
)

# Iterate through batches
for images, labels, generator_names in loader:
    # images can be either:
    # - torch.Tensor of shape (B, C, H, W) if all same size
    # - List[torch.Tensor] if variable sizes
    
    if isinstance(images, list):
        # Variable sizes: process each image individually
        for img, label in zip(images, labels):
            # Process individual image
            prediction = model(img.unsqueeze(0))  # Add batch dimension
    else:
        # Fixed sizes: process as batch (more efficient)
        predictions = model(images)
```

### With 2-Tuple Format

If your dataset returns only `(image, label)` without metadata:

```python
from data import variable_size_collate_fn_2tuple

loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=variable_size_collate_fn_2tuple
)

for images, labels in loader:
    # Same behavior as variable_size_collate_fn
    # but without generator_names
    pass
```

## Function Signatures

### variable_size_collate_fn

```python
def variable_size_collate_fn(
    batch: List[Tuple[torch.Tensor, int, str]]
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor, List[str]]:
    """
    Collate function for variable-sized images with 3-tuple format.
    
    Args:
        batch: List of (image, label, generator_name) tuples
    
    Returns:
        images: torch.Tensor (B, C, H, W) or List[torch.Tensor]
        labels: torch.Tensor (B,)
        generator_names: List[str]
    """
```

### variable_size_collate_fn_2tuple

```python
def variable_size_collate_fn_2tuple(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
    """
    Collate function for variable-sized images with 2-tuple format.
    
    Args:
        batch: List of (image, label) tuples
    
    Returns:
        images: torch.Tensor (B, C, H, W) or List[torch.Tensor]
        labels: torch.Tensor (B,)
    """
```

## Backward Compatibility

The collate function maintains **full backward compatibility** with existing code:

- When `native_resolution=False` (default), all images are resized to 256×256
- All images in a batch have the same size
- The collate function detects this and stacks them into a single tensor
- Existing training loops work without modification

```python
# Standard mode (native_resolution=False, default)
dataset = SynthBusterDataset(root_dir='path/to/synthbuster')
loader = DataLoader(dataset, batch_size=8, collate_fn=variable_size_collate_fn)

for images, labels, names in loader:
    # images is always a torch.Tensor (B, 3, 256, 256)
    # because all images are resized to 256×256
    assert isinstance(images, torch.Tensor)
    predictions = model(images)  # Works as before
```

## Training Loop Examples

### Example 1: Handling Variable Sizes

```python
for images, labels, names in loader:
    if isinstance(images, list):
        # Variable sizes: process individually
        predictions = []
        for img in images:
            pred = model(img.unsqueeze(0))  # Add batch dimension
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
    else:
        # Fixed sizes: process as batch
        predictions = model(images)
    
    # Compute loss and backpropagate
    loss = criterion(predictions, labels.float().unsqueeze(1))
    loss.backward()
    optimizer.step()
```

### Example 2: Using AnyResolutionWrapper

When using the `AnyResolutionWrapper` model, it can handle both formats:

```python
from models.resolution import AnyResolutionWrapper

# Wrap your model to handle any resolution
model = AnyResolutionWrapper(
    base_model,
    tile_size=256,
    stride=128,
    aggregation='average'
)

for images, labels, names in loader:
    # AnyResolutionWrapper handles both list and tensor inputs
    predictions = model(images)
    loss = criterion(predictions, labels.float().unsqueeze(1))
    loss.backward()
    optimizer.step()
```

## Performance Considerations

### Fixed-Size Batches (Efficient)

When all images have the same size:
- Images are stacked into a single tensor: `(B, C, H, W)`
- GPU can process the entire batch in parallel
- Maximum efficiency

### Variable-Size Batches (Flexible)

When images have different sizes:
- Images are kept as a list: `[Tensor(C, H1, W1), Tensor(C, H2, W2), ...]`
- Must process images individually or use tiling strategy
- More flexible but potentially slower

**Recommendation**: Use fixed-size batches (native_resolution=False) for training when possible. Use variable-size batches (native_resolution=True) for evaluation or when testing any-resolution capabilities.

## Testing

The collate function is thoroughly tested:

```bash
# Run unit tests
pytest ai-image-detector/data/test_collate.py -v

# Run integration tests
pytest ai-image-detector/data/test_collate_integration.py -v
```

Test coverage includes:
- Fixed-size batches (backward compatibility)
- Variable-size batches
- 2-tuple and 3-tuple formats
- Edge cases (single item, different channels, very small/large images)
- Integration with DataLoader
- Usage examples

## Requirements

- PyTorch >= 2.0.0
- No additional dependencies

## Related Components

- `SynthBusterDataset`: Supports `native_resolution` parameter
- `COCO2017Dataset`: Supports `native_resolution` parameter
- `AnyResolutionWrapper`: Model wrapper for handling variable-resolution images
- `SpectralContextAttention`: Attention module with interpolated positional encodings

## See Also

- [Design Document](../../.kiro/specs/ml-detector-enhancements/design.md) - Section 9: Data Pipeline Updates
- [Requirements](../../.kiro/specs/ml-detector-enhancements/requirements.md) - Requirement 10.2
- [Tasks](../../.kiro/specs/ml-detector-enhancements/tasks.md) - Task 16.4
