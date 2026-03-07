"""
Configuration loader and validator for AI Image Detector.

This module provides utilities for loading and validating configuration files
in YAML or JSON format. It ensures all required parameters are present and
validates their values according to the system requirements.

Supports both legacy and enhanced configurations with backward compatibility.
"""

import os
import yaml
import json
from typing import Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from configs.validator import validate_enhanced_config, get_feature_flag_summary
    ENHANCED_VALIDATION_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATION_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Reads a configuration file and validates its contents to ensure all
    required parameters are present and valid. Supports both YAML and JSON
    formats based on file extension.
    
    Args:
        config_path: Path to the configuration file (.yaml, .yml, or .json)
    
    Returns:
        Dictionary containing validated configuration parameters
    
    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the file format is unsupported or configuration is invalid
        yaml.YAMLError: If YAML parsing fails
        json.JSONDecodeError: If JSON parsing fails
    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format and load
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        with open(config_path, 'r') as f:
            if file_ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_ext == '.json':
                config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {file_ext}. "
                    "Supported formats: .yaml, .yml, .json"
                )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON configuration: {e.msg}",
            e.doc, e.pos
        )
    
    # Apply backward-compatible defaults
    config = apply_backward_compatible_defaults(config)
    
    # Validate the loaded configuration (legacy validation)
    config = validate_config(config)
    
    # Validate enhanced features if available
    if ENHANCED_VALIDATION_AVAILABLE and has_enhanced_features(config):
        validate_enhanced_config(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration parameters.
    
    Checks that all required parameters are present and have valid values.
    Validates data types, ranges, and file system paths as appropriate.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        Validated configuration dictionary (same as input if valid)
    
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    if config is None:
        raise ValueError("Configuration is empty or None")
    
    # Validate dataset configuration
    if 'dataset' not in config:
        raise ValueError("Missing required section: 'dataset'")
    
    dataset_config = config['dataset']
    
    # Check dataset mode
    dataset_mode = dataset_config.get('mode', 'synthbuster')
    
    # Validate dataset paths based on mode
    if dataset_mode == 'combined':
        # Combined mode requires both synthbuster_root and coco_root
        if 'synthbuster_root' not in dataset_config:
            raise ValueError("Missing required parameter: 'dataset.synthbuster_root' for combined mode")
        if 'coco_root' not in dataset_config:
            raise ValueError("Missing required parameter: 'dataset.coco_root' for combined mode")
        
        # Note: We don't check if paths exist here as they might not be available during testing
        # The actual loaders will validate paths when they're used
    else:
        # SynthBuster-only mode requires either root_dir or synthbuster_root
        if 'root_dir' not in dataset_config and 'synthbuster_root' not in dataset_config:
            raise ValueError("Missing required parameter: 'dataset.root_dir' or 'dataset.synthbuster_root'")
        
        root_dir = dataset_config.get('root_dir') or dataset_config.get('synthbuster_root')
        if not isinstance(root_dir, str):
            raise ValueError(
                f"Invalid type for dataset root: expected str, got {type(root_dir).__name__}"
            )
    
    # Validate dataset.image_size
    if 'image_size' in dataset_config:
        image_size = dataset_config['image_size']
        if not isinstance(image_size, int):
            raise ValueError(
                f"Invalid type for 'dataset.image_size': expected int, got {type(image_size).__name__}"
            )
        
        if image_size <= 0:
            raise ValueError(
                f"Invalid value for 'dataset.image_size': must be positive, got {image_size}"
            )
    
    # Validate training configuration
    if 'training' not in config:
        raise ValueError("Missing required section: 'training'")
    
    training_config = config['training']
    
    # Validate training.batch_size
    if 'batch_size' not in training_config:
        raise ValueError("Missing required parameter: 'training.batch_size'")
    
    batch_size = training_config['batch_size']
    if not isinstance(batch_size, int):
        raise ValueError(
            f"Invalid type for 'training.batch_size': expected int, got {type(batch_size).__name__}"
        )
    
    if batch_size <= 0:
        raise ValueError(
            f"Invalid value for 'training.batch_size': must be positive, got {batch_size}"
        )
    
    # Validate training.learning_rate
    if 'learning_rate' not in training_config:
        raise ValueError("Missing required parameter: 'training.learning_rate'")
    
    learning_rate = training_config['learning_rate']
    if not isinstance(learning_rate, (int, float)):
        raise ValueError(
            f"Invalid type for 'training.learning_rate': expected float, got {type(learning_rate).__name__}"
        )
    
    if learning_rate <= 0:
        raise ValueError(
            f"Invalid value for 'training.learning_rate': must be positive, got {learning_rate}"
        )
    
    # Validate model.backbone_type if present
    if 'model' in config and 'backbone_type' in config['model']:
        backbone_type = config['model']['backbone_type']
        valid_backbones = ['simple_cnn', 'resnet18', 'resnet50']
        
        if backbone_type not in valid_backbones:
            raise ValueError(
                f"Invalid value for 'model.backbone_type': must be one of {valid_backbones}, "
                f"got '{backbone_type}'"
            )
    
    return config



def apply_backward_compatible_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply backward-compatible defaults for enhanced features.
    
    This ensures that configurations without enhanced feature parameters
    will work correctly by providing sensible defaults that maintain
    backward compatibility with the original implementation.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configuration dictionary with defaults applied
    """
    # Ensure model section exists
    if 'model' not in config:
        config['model'] = {}
    
    model_config = config['model']
    
    # Apply feature flag defaults (all False for backward compatibility)
    feature_flag_defaults = {
        'use_spectral': False,
        'use_noise_imprint': False,
        'use_color_features': False,
        'use_local_patches': False,
        'use_fpn': False,
        'use_attention': None,
        'enable_attribution': False,
        'num_generators': 10,
    }
    
    for key, default_value in feature_flag_defaults.items():
        if key not in model_config:
            model_config[key] = default_value
    
    # Apply dataset defaults
    if 'dataset' not in config:
        config['dataset'] = {}
    
    if 'native_resolution' not in config['dataset']:
        config['dataset']['native_resolution'] = False
    
    # Apply training defaults
    if 'training' not in config:
        config['training'] = {}
    
    if 'domain_adversarial' not in config['training']:
        config['training']['domain_adversarial'] = {
            'enabled': False,
            'lambda': 1.0,
            'hidden_dim': 256
        }
    
    # Apply augmentation defaults
    if 'augmentation' not in config:
        config['augmentation'] = {}
    
    if 'robustness' not in config['augmentation']:
        config['augmentation']['robustness'] = {
            'jpeg_prob': 0.3,
            'blur_prob': 0.3,
            'noise_prob': 0.3,
            'severity_range': [1, 5]
        }
    
    if 'cutmix' not in config['augmentation']:
        config['augmentation']['cutmix'] = {
            'enabled': False,
            'alpha': 1.0,
            'prob': 0.5
        }
    
    if 'mixup' not in config['augmentation']:
        config['augmentation']['mixup'] = {
            'enabled': False,
            'alpha': 0.2,
            'prob': 0.5
        }
    
    # Apply any-resolution defaults
    if 'any_resolution' not in config:
        config['any_resolution'] = {
            'enabled': False,
            'tile_size': 256,
            'stride': 128,
            'aggregation': 'average'
        }
    
    # Apply spectral defaults (only if spectral branch is enabled)
    if model_config.get('use_spectral', False) and 'spectral' not in config:
        config['spectral'] = {
            'patch_size': 16,
            'embed_dim': 256,
            'depth': 4,
            'num_heads': 8,
            'mask_ratio': 0.75,
            'frequency_mask_type': 'high_pass',
            'cutoff_freq': 0.3
        }
    
    # Apply noise imprint defaults (only if noise imprint is enabled)
    if model_config.get('use_noise_imprint', False) and 'noise_imprint' not in config:
        config['noise_imprint'] = {
            'method': 'gaussian',  # Default to gaussian for better compatibility
            'diffusion_steps': 50,
            'gaussian_sigma': 2.0,
            'feature_dim': 256
        }
    
    # Apply chrominance defaults (only if color features are enabled)
    if model_config.get('use_color_features', False) and 'chrominance' not in config:
        config['chrominance'] = {
            'num_bins': 64,
            'feature_dim': 256
        }
    
    # Apply attention defaults (only if attention is enabled)
    if model_config.get('use_attention') is not None and 'attention' not in config:
        config['attention'] = {
            'cbam': {
                'reduction_ratio': 16,
                'kernel_size': 7
            },
            'se': {
                'reduction': 16
            }
        }
    
    # Apply FPN defaults (only if FPN is enabled)
    if model_config.get('use_fpn', False) and 'fpn' not in config:
        config['fpn'] = {
            'out_channels': 256
        }
    
    # Apply pretraining defaults (only if spectral branch is enabled)
    if model_config.get('use_spectral', False) and 'pretraining' not in config:
        config['pretraining'] = {
            'decoder_embed_dim': 128,
            'decoder_depth': 2,
            'num_epochs': 100,
            'learning_rate': 0.001
        }
    
    return config


def has_enhanced_features(config: Dict[str, Any]) -> bool:
    """
    Check if configuration uses any enhanced features.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if any enhanced features are enabled, False otherwise
    """
    if 'model' not in config:
        return False
    
    model_config = config['model']
    
    # Check if any feature flags are enabled
    enhanced_flags = [
        model_config.get('use_spectral', False),
        model_config.get('use_noise_imprint', False),
        model_config.get('use_color_features', False),
        model_config.get('use_local_patches', False),
        model_config.get('use_fpn', False),
        model_config.get('use_attention') is not None,
        model_config.get('enable_attribution', False),
    ]
    
    return any(enhanced_flags)


def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        String summary of key configuration parameters
    """
    lines = ["Configuration Summary:"]
    lines.append("=" * 50)
    
    # Model configuration
    if 'model' in config:
        model_config = config['model']
        lines.append(f"Backbone: {model_config.get('backbone_type', 'unknown')}")
        lines.append(f"Pretrained: {model_config.get('pretrained', False)}")
        
        # Feature flags
        if has_enhanced_features(config):
            lines.append("\nEnhanced Features:")
            if model_config.get('use_spectral', False):
                lines.append("  ✓ Spectral Branch")
            if model_config.get('use_noise_imprint', False):
                lines.append("  ✓ Noise Imprint Detection")
            if model_config.get('use_color_features', False):
                lines.append("  ✓ Chrominance Features")
            if model_config.get('use_local_patches', False):
                lines.append("  ✓ Local Patch Classifier")
            if model_config.get('use_fpn', False):
                lines.append("  ✓ Feature Pyramid Fusion")
            if model_config.get('use_attention') is not None:
                lines.append(f"  ✓ Attention: {model_config['use_attention']}")
            if model_config.get('enable_attribution', False):
                lines.append("  ✓ Generator Attribution")
    
    # Training configuration
    if 'training' in config:
        training_config = config['training']
        lines.append(f"\nTraining:")
        lines.append(f"  Batch Size: {training_config.get('batch_size', 'unknown')}")
        lines.append(f"  Learning Rate: {training_config.get('learning_rate', 'unknown')}")
        lines.append(f"  Epochs: {training_config.get('num_epochs', 'unknown')}")
        lines.append(f"  Optimizer: {training_config.get('optimizer', 'unknown')}")
        
        if training_config.get('domain_adversarial', {}).get('enabled', False):
            lines.append("  ✓ Domain Adversarial Training")
    
    # Dataset configuration
    if 'dataset' in config:
        dataset_config = config['dataset']
        lines.append(f"\nDataset:")
        lines.append(f"  Mode: {dataset_config.get('mode', 'unknown')}")
        lines.append(f"  Image Size: {dataset_config.get('image_size', 'unknown')}")
        lines.append(f"  Native Resolution: {dataset_config.get('native_resolution', False)}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)
