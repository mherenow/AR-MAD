"""
Integration test for enhanced configuration system.

This test verifies that the enhanced configuration system works end-to-end
with backward compatibility.
"""

import os
import sys
import yaml
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, get_config_summary


def test_load_default_config():
    """Test loading the default config file."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        'default_config.yaml'
    )
    
    if os.path.exists(config_path):
        config = load_config(config_path)
        
        print("✓ Successfully loaded default_config.yaml")
        print("\nConfig Summary:")
        print(get_config_summary(config))
        
        # Verify backward-compatible defaults were applied
        assert config['model']['use_spectral'] == False
        assert config['model']['use_noise_imprint'] == False
        assert config['model']['use_color_features'] == False
        print("\n✓ Backward-compatible defaults applied correctly")
    else:
        print(f"⚠ default_config.yaml not found at {config_path}")


def test_load_enhanced_config():
    """Test loading the enhanced config file."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        'enhanced_config.yaml'
    )
    
    if os.path.exists(config_path):
        config = load_config(config_path)
        
        print("\n✓ Successfully loaded enhanced_config.yaml")
        print("\nConfig Summary:")
        print(get_config_summary(config))
        
        # Verify enhanced config structure
        assert 'spectral' in config
        assert 'noise_imprint' in config
        assert 'chrominance' in config
        assert 'augmentation' in config
        print("\n✓ Enhanced configuration structure is valid")
    else:
        print(f"⚠ enhanced_config.yaml not found at {config_path}")


def test_backward_compatibility():
    """Test that old configs still work with new system."""
    # Create a minimal old-style config
    old_config = {
        'dataset': {
            'mode': 'synthbuster',
            'synthbuster_root': 'datasets/synthbuster',
            'image_size': 256,
        },
        'model': {
            'backbone_type': 'resnet18',
            'pretrained': True,
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10,
        }
    }
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(old_config, f)
        temp_path = f.name
    
    try:
        # Load with new system
        config = load_config(temp_path)
        
        print("\n✓ Old-style config loaded successfully")
        
        # Verify all feature flags are False (backward compatible)
        assert config['model']['use_spectral'] == False
        assert config['model']['use_noise_imprint'] == False
        assert config['model']['use_color_features'] == False
        assert config['model']['use_local_patches'] == False
        assert config['model']['use_fpn'] == False
        assert config['model']['use_attention'] is None
        assert config['model']['enable_attribution'] == False
        
        print("✓ All feature flags correctly defaulted to False/None")
        
        # Verify original values preserved
        assert config['model']['backbone_type'] == 'resnet18'
        assert config['training']['batch_size'] == 32
        
        print("✓ Original configuration values preserved")
        
    finally:
        os.unlink(temp_path)


def test_enhanced_features_enabled():
    """Test config with enhanced features enabled."""
    enhanced_config = {
        'dataset': {
            'mode': 'combined',
            'synthbuster_root': 'datasets/synthbuster',
            'coco_root': 'datasets/coco2017',
            'image_size': 256,
        },
        'model': {
            'backbone_type': 'resnet18',
            'pretrained': True,
            'use_spectral': True,
            'use_noise_imprint': True,
            'use_color_features': True,
        },
        'spectral': {
            'patch_size': 16,
            'embed_dim': 256,
            'depth': 4,
            'num_heads': 8,
            'mask_ratio': 0.75,
        },
        'noise_imprint': {
            'method': 'gaussian',
            'feature_dim': 256,
            'gaussian_sigma': 2.0,
        },
        'chrominance': {
            'num_bins': 64,
            'feature_dim': 256,
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.0001,
            'num_epochs': 50,
        }
    }
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(enhanced_config, f)
        temp_path = f.name
    
    try:
        # Load with validation
        config = load_config(temp_path)
        
        print("\n✓ Enhanced config with features enabled loaded successfully")
        
        # Verify features are enabled
        assert config['model']['use_spectral'] == True
        assert config['model']['use_noise_imprint'] == True
        assert config['model']['use_color_features'] == True
        
        print("✓ Enhanced features correctly enabled")
        
        # Verify feature configs are present
        assert 'spectral' in config
        assert 'noise_imprint' in config
        assert 'chrominance' in config
        
        print("✓ Feature-specific configurations present")
        
    finally:
        os.unlink(temp_path)


if __name__ == '__main__':
    print("=" * 70)
    print("Enhanced Configuration System Integration Test")
    print("=" * 70)
    
    try:
        test_load_default_config()
        test_load_enhanced_config()
        test_backward_compatibility()
        test_enhanced_features_enabled()
        
        print("\n" + "=" * 70)
        print("✓ All integration tests passed!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
