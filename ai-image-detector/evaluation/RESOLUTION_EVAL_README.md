# Any-Resolution Evaluation Module

This module provides comprehensive evaluation of model performance across different image resolutions. It stratifies images by size ranges and computes metrics for each stratum to assess size-dependent performance characteristics.

## Features

- **Size Stratification**: Groups images by configurable size ranges (default: 128-256, 256-512, 512-1024)
- **Comprehensive Metrics**: Computes accuracy, precision, recall, F1, and AUC for each size stratum
- **Variance Analysis**: Identifies metrics with high variance across sizes
- **Variable-Size Support**: Handles both fixed-size batched images and variable-sized images
- **Flexible Configuration**: Customizable size ranges for different use cases

## Requirements

- Requirement 8.5: When evaluating any-resolution capability THEN the system SHALL test on images of varying sizes and report size-stratified metrics

## Usage

### Basic Evaluation

```python
import torch
from torch.utils.data import DataLoader
from evaluation.resolution_eval import evaluate_any_resolution

# Load model with any-resolution support
from models.resolution import AnyResolutionWrapper
from models.classifier import BinaryClassifier

base_model = BinaryClassifier(backbone_type='resnet18')
model = AnyResolutionWrapper(base_model, tile_size=256, stride=128)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create test loader with variable-sized images
# Important: Use native_resolution=True and variable_size_collate_fn
from data.collate import variable_size_collate_fn

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    collate_fn=variable_size_collate_fn
)

# Evaluate across size ranges
results = evaluate_any_resolution(model, test_loader, device)

# Results contain metrics for each size range
print(f"128-256 accuracy: {results['128-256']['accuracy']:.4f}")
print(f"256-512 accuracy: {results['256-512']['accuracy']:.4f}")
print(f"512-1024 accuracy: {results['512-1024']['accuracy']:.4f}")
```

### Custom Size Ranges

```python
# Define custom size ranges
custom_ranges = [
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048)
]

results = evaluate_any_resolution(
    model,
    test_loader,
    device,
    size_ranges=custom_ranges
)
```

### Performance Matrix

```python
from evaluation.resolution_eval import (
    generate_size_performance_matrix,
    print_size_performance_matrix
)

# Generate performance matrix
matrix = generate_size_performance_matrix(results)

# Print formatted matrix
print_size_performance_matrix(matrix)

# Output:
# ========================================
# SIZE-STRATIFIED PERFORMANCE MATRIX
# ========================================
# 
# Metric: accuracy
# ----------------------------------------
#   128-256:   94.80%
#   256-512:   95.20%
#   512-1024:  93.60%
# ...
```

### Variance Analysis

```python
from evaluation.resolution_eval import (
    compute_size_variance,
    print_size_variance_report
)

# Compute variance statistics
variance = compute_size_variance(results)

# Print variance report
print_size_variance_report(variance)

# Check for high variance (potential size-dependent issues)
if variance['accuracy']['std'] > 0.05:
    print("Warning: High accuracy variance across sizes!")
    print(f"Range: {variance['accuracy']['range']:.4f}")

# Output:
# ========================================
# SIZE-STRATIFIED VARIANCE REPORT
# ========================================
# 
# Metric: accuracy
# ----------------------------------------
#   Mean:  94.53%
#   Std:    0.82%
#   Min:   93.60%
#   Max:   95.20%
#   Range:  1.60%
# ...
```

### Formatted Report

```python
from evaluation.resolution_eval import print_resolution_report

# Print comprehensive report
print_resolution_report(results, verbose=True)

# Output:
# ========================================
# ANY-RESOLUTION EVALUATION REPORT
# ========================================
# 
# Size Range: 128-256
# ----------------------------------------
#   Samples:   500
#   Avg Size:  192.3 x 189.7
#   Accuracy:  94.80%
#   Precision: 94.20%
#   Recall:    95.40%
#   F1 Score:  94.80%
#   AUC:       0.978
# 
# Confusion Matrix:
#   [[ 230   12]
#    [  14  244]]
#   (TN FP)
#   (FN TP)
# ...
```

## Data Preparation

For any-resolution evaluation, your dataset should:

1. **Preserve native resolution**: Set `native_resolution=True` in dataset
2. **Use variable-size collate**: Use `variable_size_collate_fn` in DataLoader
3. **Include diverse sizes**: Ensure test set covers all size ranges

Example dataset setup:

```python
from data.synthbuster_loader import SynthBusterDataset
from data.collate import variable_size_collate_fn

# Create dataset with native resolution
test_dataset = SynthBusterDataset(
    root_dir='datasets/synthbuster/test',
    native_resolution=True,  # Don't resize images
    transform=None  # No resizing transforms
)

# Create loader with variable-size collate
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=variable_size_collate_fn
)
```

## Model Requirements

The model must support variable-sized inputs. Options:

1. **AnyResolutionWrapper**: Wraps any model with tiling strategy
   ```python
   from models.resolution import AnyResolutionWrapper
   model = AnyResolutionWrapper(base_model, tile_size=256, stride=128)
   ```

2. **SpectralContextAttention**: Native variable-resolution support
   ```python
   from models.resolution import SpectralContextAttention
   # Built-in interpolated positional encodings
   ```

3. **Adaptive pooling**: Models with global pooling can handle any size
   ```python
   # Models using nn.AdaptiveAvgPool2d automatically support variable sizes
   ```

## Output Format

The `evaluate_any_resolution` function returns a dictionary:

```python
{
    '128-256': {
        'accuracy': float,        # Classification accuracy
        'precision': float,       # Precision score
        'recall': float,          # Recall score
        'f1': float,              # F1 score
        'auc': float,             # ROC AUC (NaN if single class)
        'num_samples': int,       # Number of samples in this range
        'confusion_matrix': [[TN, FP], [FN, TP]],
        'avg_height': float,      # Average image height
        'avg_width': float        # Average image width
    },
    '256-512': {...},
    '512-1024': {...}
}
```

## Interpretation

### Good Performance
- Low variance across size ranges (std < 0.02)
- Consistent metrics across all sizes
- No significant drop at any size range

### Potential Issues
- High variance (std > 0.05): Size-dependent performance
- Low accuracy at specific range: Model struggles with that size
- Large range (max - min > 0.10): Inconsistent generalization

### Example Analysis

```python
results = evaluate_any_resolution(model, test_loader, device)
variance = compute_size_variance(results)

# Check overall performance
print(f"Mean accuracy: {variance['accuracy']['mean']:.4f}")
print(f"Accuracy std: {variance['accuracy']['std']:.4f}")

# Identify problematic size ranges
for size_range, metrics in results.items():
    if metrics['accuracy'] < 0.90:
        print(f"Warning: Low accuracy at {size_range}: {metrics['accuracy']:.4f}")
    
    if metrics['num_samples'] < 50:
        print(f"Note: Limited samples at {size_range}: {metrics['num_samples']}")
```

## Integration with Comprehensive Evaluation

```python
from evaluation.comprehensive_eval import run_comprehensive_evaluation

# Any-resolution evaluation is part of comprehensive suite
results = run_comprehensive_evaluation(
    model=model,
    test_loaders=test_loaders,
    device=device,
    output_dir='evaluation_results'
)

# Access any-resolution results
resolution_results = results['any_resolution']
```

## See Also

- `cross_dataset_eval.py`: Cross-dataset evaluation
- `robustness_eval.py`: Robustness evaluation
- `models/resolution/any_resolution_wrapper.py`: Tiling strategy
- `models/resolution/context_attention.py`: Variable-resolution attention
- `data/collate.py`: Variable-size collate function
