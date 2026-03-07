# Spectral Artifact Visualization Module

This module provides visualization tools for spectral branch features using GradCAM (Gradient-weighted Class Activation Mapping) to highlight frequency domain regions that contribute most to the model's predictions.

## Features

- **GradCAM Integration**: Leverages pytorch-grad-cam library for attention heatmap generation
- **Graceful Dependency Handling**: Works with or without pytorch-grad-cam installed
- **Flexible Layer Selection**: Automatic or manual target layer selection
- **Batch Processing**: Efficient processing of multiple images
- **Customizable Visualization**: Configurable overlay alpha and colormaps

## Installation

### Required Dependencies

```bash
pip install torch torchvision numpy matplotlib
```

### Optional Dependencies

For full visualization functionality, install pytorch-grad-cam:

```bash
pip install pytorch-grad-cam>=1.4.0
```

**Note**: The module will work without pytorch-grad-cam but will provide informative error messages when visualization functions are called.

## Usage

### Basic Usage

The simplest way to generate spectral visualizations:

```python
from evaluation.spectral_viz import visualize_spectral_artifacts
from models.classifier import BinaryClassifier
import torch

# Load model with spectral branch
model = BinaryClassifier(use_spectral=True)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Load images
images = torch.randn(4, 3, 256, 256)  # Replace with actual images

# Generate visualizations
overlays = visualize_spectral_artifacts(
    model=model,
    images=images,
    overlay_alpha=0.5,
    colormap='jet'
)

# Save visualizations
from PIL import Image
for i, overlay in enumerate(overlays):
    img = Image.fromarray(overlay.astype(np.uint8))
    img.save(f'spectral_viz_{i}.png')
```

### Advanced Usage with Custom Layer

For more control, use the `SpectralGradCAM` class directly:

```python
from evaluation.spectral_viz import SpectralGradCAM, get_available_target_layers

# List available target layers
layers = get_available_target_layers(model)
print("Available layers:", layers)

# Create SpectralGradCAM with specific layer
viz = SpectralGradCAM(
    model=model,
    target_layer='spectral_branch.transformer_encoder.layers.3',
    use_cuda=True
)

# Generate heatmaps only
heatmaps = viz.generate_heatmaps(images)
print(f"Heatmaps shape: {heatmaps.shape}")  # (B, H, W)

# Generate overlays with custom settings
overlays = viz.visualize_spectral_artifacts(
    images=images,
    overlay_alpha=0.6,
    colormap='hot'
)
```

### Checking Availability

Check if pytorch-grad-cam is installed:

```python
from evaluation.spectral_viz import check_gradcam_availability

if check_gradcam_availability():
    print("GradCAM visualization is available")
else:
    print("Install pytorch-grad-cam to enable visualization")
```

## API Reference

### `SpectralGradCAM`

Main class for generating GradCAM visualizations.

**Constructor Parameters:**
- `model` (nn.Module): Model with spectral branch
- `target_layer` (Optional[Union[str, nn.Module]]): Target layer for GradCAM
  - If None, automatically uses last transformer layer
  - Can be layer name string (e.g., 'spectral_branch.transformer_encoder.layers.3')
  - Can be nn.Module instance
- `use_cuda` (bool): Whether to use CUDA if available (default: True)

**Methods:**

#### `generate_heatmaps(images, target_class=None)`

Generate GradCAM heatmaps for input images.

**Parameters:**
- `images` (torch.Tensor): Input images of shape (B, 3, H, W)
- `target_class` (Optional[int]): Target class for GradCAM (default: None, uses predicted class)

**Returns:**
- `heatmaps` (torch.Tensor): GradCAM heatmaps of shape (B, H, W) with values in [0, 1]

#### `visualize_spectral_artifacts(images, overlay_alpha=0.5, colormap='jet')`

Generate visualization overlaying heatmaps on original images.

**Parameters:**
- `images` (torch.Tensor): Input images of shape (B, 3, H, W) in range [0, 1]
- `overlay_alpha` (float): Alpha blending factor for overlay (default: 0.5)
- `colormap` (str): Matplotlib colormap name (default: 'jet')

**Returns:**
- `overlays` (np.ndarray): Overlaid images of shape (B, H, W, 3) in range [0, 255]

### Convenience Functions

#### `visualize_spectral_artifacts(model, images, target_layer=None, device=None, overlay_alpha=0.5, colormap='jet')`

Convenience function for quick visualization without creating SpectralGradCAM instance.

**Parameters:**
- `model` (nn.Module): Model with spectral branch
- `images` (torch.Tensor): Input images of shape (B, 3, H, W)
- `target_layer` (Optional[Union[str, nn.Module]]): Target layer (default: None, auto-detect)
- `device` (Optional[torch.device]): Device to run on (default: None, auto-detect)
- `overlay_alpha` (float): Alpha blending factor (default: 0.5)
- `colormap` (str): Matplotlib colormap name (default: 'jet')

**Returns:**
- `overlays` (Union[np.ndarray, None]): Overlaid images or None if pytorch-grad-cam unavailable

#### `get_available_target_layers(model)`

Get list of available target layers in the model for GradCAM.

**Parameters:**
- `model` (nn.Module): Model to inspect

**Returns:**
- `layers` (List[str]): List of layer names suitable for GradCAM

#### `check_gradcam_availability()`

Check if pytorch-grad-cam is available.

**Returns:**
- `available` (bool): True if pytorch-grad-cam is installed, False otherwise

## Examples

See `example_spectral_viz.py` for comprehensive examples including:

1. **Basic Usage**: Simple visualization with automatic layer detection
2. **Advanced Usage**: Custom layer selection and configuration
3. **Batch Processing**: Processing multiple batches of images
4. **Comparison**: Comparing real vs AI-generated images

Run examples:

```bash
python ai-image-detector/evaluation/example_spectral_viz.py
```

## Colormaps

Common matplotlib colormaps for visualization:

- `'jet'`: Rainbow colormap (default, high contrast)
- `'hot'`: Black-red-yellow-white colormap
- `'viridis'`: Perceptually uniform colormap
- `'plasma'`: Perceptually uniform colormap
- `'inferno'`: Perceptually uniform colormap
- `'coolwarm'`: Blue-white-red diverging colormap

## Target Layer Selection

### Automatic Selection

By default, SpectralGradCAM automatically selects the last transformer layer in the spectral branch:
- `spectral_branch.transformer_encoder.layers[-1]`

### Manual Selection

For better control, specify the target layer explicitly:

```python
# Use specific transformer layer
viz = SpectralGradCAM(model, target_layer='spectral_branch.transformer_encoder.layers.2')

# Use FFT processor output
viz = SpectralGradCAM(model, target_layer='spectral_branch.fft_processor')

# Use patch tokenizer
viz = SpectralGradCAM(model, target_layer='spectral_branch.patch_tokenizer')
```

### Finding Suitable Layers

Use `get_available_target_layers()` to discover suitable layers:

```python
layers = get_available_target_layers(model)
for layer in layers:
    print(layer)
```

## Error Handling

### Missing pytorch-grad-cam

If pytorch-grad-cam is not installed:

```python
from evaluation.spectral_viz import SpectralGradCAM

try:
    viz = SpectralGradCAM(model)
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install pytorch-grad-cam>=1.4.0")
```

### Invalid Target Layer

If the specified target layer doesn't exist:

```python
try:
    viz = SpectralGradCAM(model, target_layer='nonexistent.layer')
except ValueError as e:
    print(f"Error: {e}")
    # Use get_available_target_layers() to find valid layers
```

### Model Without Spectral Branch

If the model doesn't have a spectral branch:

```python
model = BinaryClassifier(use_spectral=False)  # No spectral branch

try:
    viz = SpectralGradCAM(model)  # Will fail
except ValueError as e:
    print(f"Error: {e}")
    # Must specify target_layer explicitly or use model with spectral branch
```

## Performance Considerations

### GPU Acceleration

Enable CUDA for faster processing:

```python
viz = SpectralGradCAM(model, use_cuda=True)
```

### Batch Size

Process images in batches for efficiency:

```python
batch_size = 16
for i in range(0, len(all_images), batch_size):
    batch = all_images[i:i+batch_size]
    overlays = viz.visualize_spectral_artifacts(batch)
    # Process overlays...
```

### Memory Management

For large datasets, process and save incrementally:

```python
for batch in dataloader:
    overlays = viz.visualize_spectral_artifacts(batch)
    # Save immediately
    save_overlays(overlays)
    # Free memory
    del overlays
```

## Integration with Evaluation Pipeline

Integrate spectral visualization into comprehensive evaluation:

```python
from evaluation.spectral_viz import SpectralGradCAM
from evaluation.robustness_eval import evaluate_robustness

# Evaluate robustness
results = evaluate_robustness(model, test_loader, device)

# Visualize spectral artifacts for sample images
viz = SpectralGradCAM(model)
sample_images = next(iter(test_loader))[0][:4]
overlays = viz.visualize_spectral_artifacts(sample_images)

# Save visualizations
for i, overlay in enumerate(overlays):
    Image.fromarray(overlay).save(f'eval_viz_{i}.png')
```

## Troubleshooting

### Issue: Heatmaps are all zeros

**Cause**: Target layer may not have gradients or model is in eval mode without gradient computation.

**Solution**: Ensure model is in eval mode but GradCAM computes gradients internally.

### Issue: Overlays look incorrect

**Cause**: Input images may not be in correct range [0, 1].

**Solution**: Normalize images to [0, 1] range before visualization.

### Issue: Out of memory

**Cause**: Processing too many large images at once.

**Solution**: Reduce batch size or process images sequentially.

## References

- [GradCAM Paper](https://arxiv.org/abs/1610.02391): Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam): PyTorch implementation of GradCAM

## Requirements Validation

This module validates **Requirement 8.2**:

> WHEN evaluating spectral artifacts THEN the system SHALL use GradCAM visualization to highlight frequency domain regions

**Validation:**
- ✅ Implements GradCAM integration for spectral branch
- ✅ Generates attention heatmaps highlighting important frequency regions
- ✅ Handles optional pytorch-grad-cam dependency gracefully
- ✅ Provides clear error messages when dependency is missing
- ✅ Supports batch processing and customization
