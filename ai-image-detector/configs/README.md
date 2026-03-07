# Enhanced Configuration System

This directory contains the enhanced configuration system for the AI Image Detector, supporting 8 major feature enhancements with full backward compatibility.

## Configuration Files

### `default_config.yaml`
The original configuration file for basic detector functionality. This config works with the original implementation and maintains backward compatibility.

**Key Features:**
- Simple backbone configuration (ResNet18/50, SimpleCNN)
- Basic training parameters
- Dataset configuration for SynthBuster and COCO2017
- No enhanced features enabled

### `enhanced_config.yaml`
The comprehensive configuration file with all 8 enhanced features available. All features are disabled by default for backward compatibility.

**Enhanced Features:**
1. **Spectral Branch** - Frequency domain analysis using FFT and ViT
2. **Noise Imprint Detection** - Generator-specific noise pattern detection
3. **Chrominance Features** - YCbCr color space analysis
4. **Attention Mechanisms** - CBAM and SE attention modules
5. **Feature Pyramid Fusion** - Multi-scale feature combination
6. **Local Patch Classifier** - Fine-grained patch-level detection
7. **Any-Resolution Processing** - Handle variable-sized images
8. **Robustness Augmentation** - JPEG/blur/noise augmentation, CutMix, MixUp

## Configuration Modules

### `validator.py`
Comprehensive validation logic for enhanced configurations.

**Features:**
- Feature flag dependency validation
- Incompatible combination detection
- Parameter range and type checking
- Optional dependency handling (diffusers, pytorch-grad-cam)
- Automatic fallback for missing dependencies

**Key Functions:**
- `validate_enhanced_config(config)` - Main validation entry point
- `get_feature_flag_summary(config)` - Extract feature flag status
- Individual validators for each feature module

### Updated `utils/config_loader.py`
Enhanced configuration loader with backward compatibility.

**Features:**
- Automatic default application for missing parameters
- Backward-compatible defaults (all features disabled)
- Support for both legacy and enhanced configs
- Configuration summary generation

**Key Functions:**
- `load_config(path)` - Load and validate configuration
- `apply_backward_compatible_defaults(config)` - Apply defaults
- `has_enhanced_features(config)` - Detect enhanced features
- `get_config_summary(config)` - Generate human-readable summary

## Usage Examples

### Loading a Configuration

```python
from utils.config_loader import load_config, get_config_summary

# Load configuration
config = load_config('configs/enhanced_config.yaml')

# Print summary
print(get_config_summary(config))
```

### Backward Compatibility

Old configurations work seamlessly with the new system:

```yaml
# Old-style config (still works!)
dataset:
  mode: "synthbuster"
  synthbuster_root: "datasets/synthbuster"
  image_size: 256

model:
  backbone_type: "resnet18"
  pretrained: true

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 10
```

The system automatically applies defaults:
- All feature flags set to `False`
- All enhanced features disabled
- Behavior identical to original implementation

### Enabling Enhanced Features

To enable specific features, add them to your config:

```yaml
model:
  backbone_type: "resnet18"
  pretrained: true
  
  # Enable enhanced features
  use_spectral: true
  use_noise_imprint: true
  use_color_features: true

# Feature-specific configurations
spectral:
  patch_size: 16
  embed_dim: 256
  depth: 4
  num_heads: 8
  mask_ratio: 0.75

noise_imprint:
  method: 'gaussian'  # or 'diffusion' if diffusers installed
  feature_dim: 256
  gaussian_sigma: 2.0

chrominance:
  num_bins: 64
  feature_dim: 256
```

## Feature Flag Reference

### Model Feature Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_spectral` | bool | `false` | Enable spectral branch (FFT + ViT) |
| `use_noise_imprint` | bool | `false` | Enable noise imprint detection |
| `use_color_features` | bool | `false` | Enable chrominance features |
| `use_local_patches` | bool | `false` | Enable local patch classifier |
| `use_fpn` | bool | `false` | Enable feature pyramid fusion |
| `use_attention` | str/null | `null` | Attention type: 'cbam', 'se', or null |
| `enable_attribution` | bool | `false` | Enable generator attribution |
| `num_generators` | int | `10` | Number of generator classes |

### Data Feature Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `native_resolution` | bool | `false` | Preserve original image dimensions |
| `any_resolution.enabled` | bool | `false` | Enable any-resolution processing |

### Training Feature Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `domain_adversarial.enabled` | bool | `false` | Enable domain adversarial training |
| `augmentation.cutmix.enabled` | bool | `false` | Enable CutMix augmentation |
| `augmentation.mixup.enabled` | bool | `false` | Enable MixUp augmentation |

## Validation Rules

### Feature Dependencies

1. **Spectral Branch** (`use_spectral=true`)
   - Requires: `spectral` configuration section
   - Optional: `pretraining` configuration for masked pretraining

2. **Noise Imprint** (`use_noise_imprint=true`)
   - Requires: `noise_imprint` configuration section
   - Optional: `diffusers` library for diffusion method

3. **Attribution** (`enable_attribution=true`)
   - Requires: `use_noise_imprint=true`
   - Requires: `num_generators >= 2`

4. **Chrominance Features** (`use_color_features=true`)
   - Requires: `chrominance` configuration section

5. **Attention** (`use_attention != null`)
   - Requires: `attention` configuration section
   - Must be one of: 'cbam', 'se'

6. **FPN** (`use_fpn=true`)
   - Requires: `fpn` configuration section

7. **Domain Adversarial** (`domain_adversarial.enabled=true`)
   - Requires: At least 2 datasets in `data.datasets`

### Incompatible Combinations

The validator automatically detects and prevents:
- Attribution without noise imprint branch
- Domain adversarial training with single dataset
- Invalid parameter ranges (e.g., probabilities outside [0, 1])
- Missing required configuration sections

### Optional Dependencies

The system gracefully handles missing optional dependencies:

- **diffusers**: Falls back to Gaussian method for noise extraction
- **pytorch-grad-cam**: Disables spectral artifact visualization
- **scikit-learn**: Required only for clustering metrics

Warnings are issued when optional dependencies are unavailable.

## Testing

### Unit Tests

```bash
# Test enhanced configuration validation
python -m pytest configs/test_enhanced_config.py -v

# Test backward compatibility
python -m pytest utils/test_config_loader.py -v
```

### Integration Tests

```bash
# Run integration tests
python configs/test_integration.py
```

## Configuration Best Practices

### 1. Start with Defaults

Begin with `default_config.yaml` and enable features incrementally:

```yaml
# Start simple
model:
  backbone_type: "resnet18"
  use_spectral: false  # Add features one at a time
```

### 2. Validate Before Training

Always validate your configuration:

```python
from configs.validator import validate_enhanced_config

try:
    validate_enhanced_config(config)
    print("✓ Configuration is valid")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

### 3. Use Feature Flag Summary

Check which features are enabled:

```python
from configs.validator import get_feature_flag_summary

flags = get_feature_flag_summary(config)
print("Enabled features:", [k for k, v in flags.items() if v])
```

### 4. Monitor Optional Dependencies

Check for missing optional dependencies:

```python
try:
    import diffusers
    print("✓ diffusers available")
except ImportError:
    print("⚠ diffusers not available, using Gaussian fallback")
```

## Migration Guide

### From Original to Enhanced

1. **No changes required** - Old configs work as-is
2. **Add feature flags** - Enable features incrementally
3. **Add feature configs** - Provide configuration for enabled features
4. **Validate** - Run validation to catch issues early

Example migration:

```yaml
# Before (original config)
model:
  backbone_type: "resnet18"

# After (enhanced config with spectral branch)
model:
  backbone_type: "resnet18"
  use_spectral: true  # Enable new feature

spectral:  # Add feature configuration
  patch_size: 16
  embed_dim: 256
  depth: 4
  num_heads: 8
  mask_ratio: 0.75
```

## Troubleshooting

### Common Issues

**Issue**: `ValueError: use_spectral=True requires 'spectral' configuration`
- **Solution**: Add `spectral` section to config with required parameters

**Issue**: `ValueError: Attribution requires noise imprint branch`
- **Solution**: Set `use_noise_imprint: true` when enabling attribution

**Issue**: `Warning: diffusers library not available`
- **Solution**: Install diffusers (`pip install diffusers`) or use Gaussian method

**Issue**: Old tests failing with new config loader
- **Solution**: This is expected - error messages have changed slightly but functionality is preserved

## Support

For issues or questions about the enhanced configuration system:
1. Check this README for common patterns
2. Review `test_enhanced_config.py` for usage examples
3. Run `test_integration.py` to verify your setup
4. Check validation error messages for specific guidance
