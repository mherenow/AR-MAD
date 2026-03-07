"""
Tests for enhanced configuration system.

This module tests the enhanced configuration validation and loading,
including backward compatibility and feature flag validation.
"""

import pytest
import os
import sys
import yaml
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.validator import (
    validate_enhanced_config,
    get_feature_flag_summary,
    _validate_spectral_config,
    _validate_noise_imprint_config,
    _validate_augmentation_config,
)
from utils.config_loader import (
    load_config,
    apply_backward_compatible_defaults,
    has_enhanced_features,
    get_config_summary,
)


class TestEnhancedConfigValidation:
    """Test suite for enhanced configuration validation."""
    
    def test_valid_minimal_config(self):
        """Test that minimal valid config passes validation."""
        config = {
            'model': {
                'backbone_type': 'resnet18',
                'pretrained': True,
            },
            'dataset': {
                'mode': 'synthbuster',
                'synthbuster_root': 'datasets/synthbuster',
                'image_size': 256,
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10,
            }
        }
        
        # Should not raise any exceptions
        validate_enhanced_config(config)
    
    def test_spectral_branch_requires_config(self):
        """Test that enabling spectral branch requires spectral config."""
        config = {
            'model': {
                'use_spectral': True,
            },
            'dataset': {'mode': 'synthbuster', 'synthbuster_root': 'test'},
            'training': {'batch_size': 32, 'learning_rate': 0.001},
        }
        
        with pytest.raises(ValueError, match="use_spectral=True requires 'spectral' configuration"):
            validate_enhanced_config(config)
    
    def test_attribution_requires_noise_imprint(self):
        """Test that attribution requires noise imprint branch."""
        config = {
            'model': {
                'enable_attribution': True,
                'use_noise_imprint': False,
            },
            'dataset': {'mode': 'synthbuster', 'synthbuster_root': 'test'},
            'training': {'batch_size': 32, 'learning_rate': 0.001},
        }
        
        with pytest.raises(ValueError, match="Attribution requires noise imprint branch"):
            validate_enhanced_config(config)
    
    def test_domain_adversarial_requires_multiple_datasets(self):
        """Test that domain adversarial training requires at least 2 datasets."""
        config = {
            'model': {'backbone_type': 'resnet18'},
            'dataset': {'mode': 'synthbuster', 'synthbuster_root': 'test'},
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'domain_adversarial': {
                    'enabled': True,
                }
            },
            'data': {
                'datasets': {
                    'synthbuster': {'weight': 1.0, 'path': 'test'}
                }
            }
        }
        
        with pytest.raises(ValueError, match="requires at least 2 datasets"):
            validate_enhanced_config(config)
    
    def test_spectral_config_validation(self):
        """Test spectral configuration validation."""
        # Valid config
        valid_config = {
            'patch_size': 16,
            'embed_dim': 256,
            'depth': 4,
            'num_heads': 8,
            'mask_ratio': 0.75,
        }
        _validate_spectral_config(valid_config)
        
        # Invalid: embed_dim not divisible by num_heads
        invalid_config = {
            'patch_size': 16,
            'embed_dim': 250,
            'depth': 4,
            'num_heads': 8,
            'mask_ratio': 0.75,
        }
        with pytest.raises(ValueError, match="embed_dim.*must be divisible by num_heads"):
            _validate_spectral_config(invalid_config)
        
        # Invalid: mask_ratio out of range
        invalid_config = {
            'patch_size': 16,
            'embed_dim': 256,
            'depth': 4,
            'num_heads': 8,
            'mask_ratio': 1.5,
        }
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            _validate_spectral_config(invalid_config)
    
    def test_noise_imprint_config_validation(self):
        """Test noise imprint configuration validation."""
        # Valid config
        valid_config = {
            'method': 'gaussian',
            'feature_dim': 256,
            'gaussian_sigma': 2.0,
        }
        _validate_noise_imprint_config(valid_config)
        
        # Invalid method
        invalid_config = {
            'method': 'invalid_method',
            'feature_dim': 256,
        }
        with pytest.raises(ValueError, match="method must be one of"):
            _validate_noise_imprint_config(invalid_config)
    
    def test_augmentation_config_validation(self):
        """Test augmentation configuration validation."""
        # Valid config
        valid_config = {
            'robustness': {
                'jpeg_prob': 0.3,
                'blur_prob': 0.3,
                'noise_prob': 0.3,
                'severity_range': [1, 5],
            },
            'cutmix': {
                'enabled': True,
                'alpha': 1.0,
                'prob': 0.5,
            },
            'mixup': {
                'enabled': True,
                'alpha': 0.2,
                'prob': 0.5,
            }
        }
        _validate_augmentation_config(valid_config)
        
        # Invalid: severity_range out of bounds
        invalid_config = {
            'robustness': {
                'severity_range': [0, 6],
            }
        }
        with pytest.raises(ValueError, match="severity_range must be in"):
            _validate_augmentation_config(invalid_config)
        
        # Invalid: probability out of range
        invalid_config = {
            'robustness': {
                'jpeg_prob': 1.5,
            }
        }
        with pytest.raises(ValueError, match="must be in.*0.0, 1.0"):
            _validate_augmentation_config(invalid_config)


class TestBackwardCompatibility:
    """Test suite for backward compatibility features."""
    
    def test_apply_defaults_to_minimal_config(self):
        """Test that defaults are applied to minimal config."""
        minimal_config = {
            'model': {
                'backbone_type': 'resnet18',
            },
            'dataset': {
                'mode': 'synthbuster',
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
            }
        }
        
        config = apply_backward_compatible_defaults(minimal_config)
        
        # Check that feature flags are set to False
        assert config['model']['use_spectral'] == False
        assert config['model']['use_noise_imprint'] == False
        assert config['model']['use_color_features'] == False
        assert config['model']['use_attention'] is None
        assert config['model']['enable_attribution'] == False
        
        # Check that native_resolution defaults to False
        assert config['dataset']['native_resolution'] == False
        
        # Check that augmentation defaults are set
        assert 'augmentation' in config
        assert config['augmentation']['cutmix']['enabled'] == False
        assert config['augmentation']['mixup']['enabled'] == False
    
    def test_has_enhanced_features_detection(self):
        """Test detection of enhanced features."""
        # Config without enhanced features
        basic_config = {
            'model': {
                'backbone_type': 'resnet18',
                'use_spectral': False,
                'use_noise_imprint': False,
            }
        }
        assert has_enhanced_features(basic_config) == False
        
        # Config with enhanced features
        enhanced_config = {
            'model': {
                'backbone_type': 'resnet18',
                'use_spectral': True,
            }
        }
        assert has_enhanced_features(enhanced_config) == True
    
    def test_feature_flag_summary(self):
        """Test feature flag summary extraction."""
        config = {
            'model': {
                'use_spectral': True,
                'use_noise_imprint': False,
                'use_color_features': True,
                'enable_attribution': False,
            },
            'dataset': {
                'native_resolution': True,
            },
            'training': {
                'domain_adversarial': {
                    'enabled': True,
                }
            }
        }
        
        summary = get_feature_flag_summary(config)
        
        assert summary['use_spectral'] == True
        assert summary['use_noise_imprint'] == False
        assert summary['use_color_features'] == True
        assert summary['enable_attribution'] == False
        assert summary['native_resolution'] == True
        assert summary['domain_adversarial_enabled'] == True


class TestConfigLoading:
    """Test suite for configuration loading."""
    
    def test_load_enhanced_config_file(self):
        """Test loading enhanced config from file."""
        # Create a temporary config file
        config_data = {
            'model': {
                'backbone_type': 'resnet18',
                'pretrained': True,
                'use_spectral': True,
            },
            'spectral': {
                'patch_size': 16,
                'embed_dim': 256,
                'depth': 4,
                'num_heads': 8,
                'mask_ratio': 0.75,
            },
            'dataset': {
                'mode': 'synthbuster',
                'synthbuster_root': 'datasets/synthbuster',
                'image_size': 256,
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10,
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Load the config
            config = load_config(temp_path)
            
            # Verify it loaded correctly
            assert config['model']['use_spectral'] == True
            assert config['spectral']['patch_size'] == 16
            
            # Verify defaults were applied
            assert config['model']['use_noise_imprint'] == False
            assert 'augmentation' in config
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_config_summary_generation(self):
        """Test configuration summary generation."""
        config = {
            'model': {
                'backbone_type': 'resnet18',
                'pretrained': True,
                'use_spectral': True,
                'use_noise_imprint': True,
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 50,
                'optimizer': 'adamw',
            },
            'dataset': {
                'mode': 'combined',
                'image_size': 256,
                'native_resolution': False,
            }
        }
        
        summary = get_config_summary(config)
        
        # Check that summary contains key information
        assert 'resnet18' in summary
        assert 'Spectral Branch' in summary
        assert 'Noise Imprint' in summary
        assert 'Batch Size: 32' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
