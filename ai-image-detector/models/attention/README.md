# Attention Mechanisms and Multi-Scale Fusion

This directory contains attention mechanisms and multi-scale feature fusion modules for the ML-generated image detector.

## Modules

### 1. CBAM (Convolutional Block Attention Module)

**File**: `cbam.py`

CBAM applies both channel attention and spatial attention sequentially to refine feature maps.

**Key Features**:
- Channel attention using global average and max pooling
- Spatial attention using channel-wise pooling
- Reduction ratio of 16 (configurable)
- Spatial kernel size of 7 (configurable)

**Usage**:
```python
from models.attention import CBAM

cbam = CBAM(channels=256, reduction_ratio=16, kernel_size=7)
features = torch.randn(4, 256, 32, 32)
attended = cbam(features)  # Shape: (4, 256, 32, 32)
```

**Reference**: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

### 2. SEBlock (Squeeze-and-Excitation Block)

**File**: `se_block.py`

SEBlock recalibrates channel-wise feature responses through squeeze-and-excitation.

**Key Features**:
- Global average pooling for squeeze operation
- Two FC layers for excitation (with reduction ratio)
- Sigmoid activation for channel weights
- Reduction ratio of 16 (configurable)

**Usage**:
```python
from models.attention import SEBlock

se = SEBlock(channels=256, reduction=16)
features = torch.randn(4, 256, 32, 32)
recalibrated = se(features)  # Shape: (4, 256, 32, 32)
```

**Reference**: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018

### 3. LocalPatchClassifier

**File**: `local_patch_classifier.py`

LocalPatchClassifier performs patch-level classification for fine-grained detection.

**Key Features**:
- Divides feature maps into patches
- Per-patch classification with 1x1 convolutions
- Aggregation methods: average or max pooling
- Optional spatial heatmap output
- Configurable patch size (default: 8)

**Usage**:
```python
from models.attention import LocalPatchClassifier

classifier = LocalPatchClassifier(
    feature_dim=256,
    patch_size=8,
    aggregation='average'
)

features = torch.randn(4, 256, 32, 32)

# Get aggregated prediction
prediction = classifier(features)  # Shape: (4, 1)

# Get prediction with spatial heatmap
prediction, heatmap = classifier(features, return_heatmap=True)
# prediction: (4, 1), heatmap: (4, 4, 4) for 32/8 = 4 patches per dimension
```

### 4. FeaturePyramidFusion (FPN)

**File**: `../fusion/fpn.py`

FeaturePyramidFusion combines features from multiple scales using FPN-style architecture.

**Key Features**:
- Top-down pathway with lateral connections
- Bilinear upsampling for scale matching
- 1x1 convolutions for channel alignment
- 3x3 convolutions to reduce aliasing
- Final fusion layer for multi-scale concatenation

**Usage**:
```python
from models.fusion import FeaturePyramidFusion

fpn = FeaturePyramidFusion(
    in_channels_list=[512, 256, 128],  # Low to high resolution
    out_channels=256
)

features = [
    torch.randn(4, 512, 8, 8),   # Low resolution
    torch.randn(4, 256, 16, 16), # Mid resolution
    torch.randn(4, 128, 32, 32)  # High resolution
]

fused = fpn(features)  # Shape: (4, 256, 32, 32)
```

**Reference**: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017

## Integration with BinaryClassifier

These modules integrate with the main `BinaryClassifier` through feature flags:

```python
from models.classifier import BinaryClassifier

model = BinaryClassifier(
    backbone_type='resnet18',
    use_attention='cbam',  # or 'se'
    use_local_patches=True,
    use_fpn=True
)
```

## Testing

All modules have comprehensive unit tests:

```bash
# Test all attention and fusion modules
pytest ai-image-detector/models/attention/ ai-image-detector/models/fusion/ -v

# Test individual modules
pytest ai-image-detector/models/attention/test_cbam.py -v
pytest ai-image-detector/models/attention/test_se_block.py -v
pytest ai-image-detector/models/attention/test_local_patch_classifier.py -v
pytest ai-image-detector/models/fusion/test_fpn.py -v
```

## Example Usage

See `example_usage.py` for complete examples demonstrating:
1. Individual module usage
2. Combined pipeline with all modules
3. Integration with a simple backbone

Run the examples:
```bash
python ai-image-detector/models/attention/example_usage.py
```

## Design Specifications

These modules implement Requirements 6.1-6.4 from the ML Detector Enhancements specification:

- **Requirement 6.1**: CBAM with channel and spatial attention
- **Requirement 6.2**: SEBlock with squeeze-and-excitation
- **Requirement 6.3**: LocalPatchClassifier for fine-grained detection
- **Requirement 6.4**: FeaturePyramidFusion for multi-scale features

## Architecture Details

### CBAM Architecture
```
Input (B, C, H, W)
    ↓
Channel Attention
    ├─ Global Avg Pool → MLP → Sigmoid
    └─ Global Max Pool → MLP → Sigmoid
    ↓
Channel-weighted features
    ↓
Spatial Attention
    ├─ Channel Avg Pool ┐
    └─ Channel Max Pool ┴→ Conv2d → Sigmoid
    ↓
Output (B, C, H, W)
```

### SEBlock Architecture
```
Input (B, C, H, W)
    ↓
Squeeze: Global Avg Pool → (B, C)
    ↓
Excitation: FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid
    ↓
Scale: Input × Channel Weights
    ↓
Output (B, C, H, W)
```

### LocalPatchClassifier Architecture
```
Input (B, C, H, W)
    ↓
Per-patch Classification: Conv1x1(C→hidden) → ReLU → Conv1x1(hidden→1)
    ↓
Patch Pooling: AvgPool(patch_size)
    ↓
Sigmoid → (B, 1, H_patches, W_patches)
    ↓
Aggregation: Average or Max Pooling
    ↓
Output (B, 1)
```

### FPN Architecture
```
Multi-scale Inputs:
  [Low-res (B, C1, H1, W1), Mid-res (B, C2, H2, W2), High-res (B, C3, H3, W3)]
    ↓
Lateral Connections: Conv1x1 to match channels
    ↓
Top-down Pathway: Upsample + Add
    ↓
Output Convs: Conv3x3 to reduce aliasing
    ↓
Upsample all to highest resolution
    ↓
Concatenate + Conv1x1 fusion
    ↓
Output (B, out_channels, H3, W3)
```

## Performance Considerations

- **CBAM**: Adds ~2-3% computational overhead, provides 1-2% accuracy improvement
- **SEBlock**: Lightweight, adds <1% overhead, effective for channel recalibration
- **LocalPatchClassifier**: Enables spatial interpretability with minimal overhead
- **FPN**: Increases memory usage for multi-scale features, significant accuracy gains

## Future Enhancements

Potential improvements:
1. Efficient attention variants (e.g., ECA-Net, Coordinate Attention)
2. Learnable aggregation weights in LocalPatchClassifier
3. Deformable convolutions in FPN
4. Attention visualization tools
