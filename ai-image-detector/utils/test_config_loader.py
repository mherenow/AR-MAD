"""
Unit tests for configuration loader.

Tests the load_config() and validate_config() functions to ensure proper
validation of configuration parameters and error handling.
"""

import os
import sys
import pytest
import tempfile
import yaml
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config, validate_config


class TestValidateConfig:
    """Test suite for validate_config() function."""
    
    def test_validate_config_with_valid_config(self, tmp_path):
        """Test validation passes with valid configuration."""
        # Create a temporary dataset directory
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256,
                'val_ratio': 0.2,
                'num_workers': 4
            },
            'model': {
                'backbone_type': 'resnet18',
                'pretrained': True
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10
            }
        }
        
        # Should not raise any exceptions
        result = validate_config(config)
        assert result == config
    
    def test_validate_config_with_none(self):
        """Test validation fails with None config."""
        with pytest.raises(ValueError, match="Configuration is empty or None"):
            validate_config(None)
    
    def test_validate_config_missing_dataset_section(self):
        """Test validation fails when dataset section is missing."""
        config = {
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Missing required section: 'dataset'"):
            validate_config(config)
    
    def test_validate_config_missing_root_dir(self):
        """Test validation fails when dataset.root_dir is missing."""
        config = {
            'dataset': {
                'image_size': 256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Missing required parameter: 'dataset.root_dir'"):
            validate_config(config)
    
    def test_validate_config_invalid_root_dir_type(self):
        """Test validation fails when dataset.root_dir is not a string."""
        config = {
            'dataset': {
                'root_dir': 123,
                'image_size': 256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Invalid type for 'dataset.root_dir'"):
            validate_config(config)
    
    def test_validate_config_nonexistent_root_dir(self):
        """Test validation fails when dataset.root_dir does not exist."""
        config = {
            'dataset': {
                'root_dir': '/nonexistent/path/to/dataset',
                'image_size': 256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Dataset root directory does not exist"):
            validate_config(config)
    
    def test_validate_config_missing_image_size(self, tmp_path):
        """Test validation fails when dataset.image_size is missing."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir)
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Missing required parameter: 'dataset.image_size'"):
            validate_config(config)
    
    def test_validate_config_invalid_image_size_type(self, tmp_path):
        """Test validation fails when dataset.image_size is not an integer."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': "256"
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Invalid type for 'dataset.image_size'"):
            validate_config(config)
    
    def test_validate_config_negative_image_size(self, tmp_path):
        """Test validation fails when dataset.image_size is not positive."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': -256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Invalid value for 'dataset.image_size': must be positive"):
            validate_config(config)
    
    def test_validate_config_missing_training_section(self, tmp_path):
        """Test validation fails when training section is missing."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            }
        }
        
        with pytest.raises(ValueError, match="Missing required section: 'training'"):
            validate_config(config)
    
    def test_validate_config_missing_batch_size(self, tmp_path):
        """Test validation fails when training.batch_size is missing."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'training': {
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Missing required parameter: 'training.batch_size'"):
            validate_config(config)
    
    def test_validate_config_invalid_batch_size_type(self, tmp_path):
        """Test validation fails when training.batch_size is not an integer."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'training': {
                'batch_size': 32.5,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Invalid type for 'training.batch_size'"):
            validate_config(config)
    
    def test_validate_config_negative_batch_size(self, tmp_path):
        """Test validation fails when training.batch_size is not positive."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'training': {
                'batch_size': -32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Invalid value for 'training.batch_size': must be positive"):
            validate_config(config)
    
    def test_validate_config_missing_learning_rate(self, tmp_path):
        """Test validation fails when training.learning_rate is missing."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'training': {
                'batch_size': 32
            }
        }
        
        with pytest.raises(ValueError, match="Missing required parameter: 'training.learning_rate'"):
            validate_config(config)
    
    def test_validate_config_invalid_learning_rate_type(self, tmp_path):
        """Test validation fails when training.learning_rate is not numeric."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': "0.001"
            }
        }
        
        with pytest.raises(ValueError, match="Invalid type for 'training.learning_rate'"):
            validate_config(config)
    
    def test_validate_config_negative_learning_rate(self, tmp_path):
        """Test validation fails when training.learning_rate is not positive."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': -0.001
            }
        }
        
        with pytest.raises(ValueError, match="Invalid value for 'training.learning_rate': must be positive"):
            validate_config(config)
    
    def test_validate_config_invalid_backbone_type(self, tmp_path):
        """Test validation fails when model.backbone_type is invalid."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        config = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'model': {
                'backbone_type': 'invalid_backbone'
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with pytest.raises(ValueError, match="Invalid value for 'model.backbone_type'"):
            validate_config(config)
    
    def test_validate_config_valid_backbone_types(self, tmp_path):
        """Test validation passes with all valid backbone types."""
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        valid_backbones = ['simple_cnn', 'resnet18', 'resnet50']
        
        for backbone in valid_backbones:
            config = {
                'dataset': {
                    'root_dir': str(dataset_dir),
                    'image_size': 256
                },
                'model': {
                    'backbone_type': backbone
                },
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            }
            
            # Should not raise any exceptions
            result = validate_config(config)
            assert result == config


class TestLoadConfig:
    """Test suite for load_config() function."""
    
    def test_load_config_yaml(self, tmp_path):
        """Test loading valid YAML configuration file."""
        # Create temporary dataset directory
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        # Create temporary YAML config file
        config_file = tmp_path / "config.yaml"
        config_data = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and validate
        result = load_config(str(config_file))
        assert result['dataset']['root_dir'] == str(dataset_dir)
        assert result['dataset']['image_size'] == 256
        assert result['training']['batch_size'] == 32
        assert result['training']['learning_rate'] == 0.001
    
    def test_load_config_json(self, tmp_path):
        """Test loading valid JSON configuration file."""
        # Create temporary dataset directory
        dataset_dir = tmp_path / "synthbuster"
        dataset_dir.mkdir()
        
        # Create temporary JSON config file
        config_file = tmp_path / "config.json"
        config_data = {
            'dataset': {
                'root_dir': str(dataset_dir),
                'image_size': 256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load and validate
        result = load_config(str(config_file))
        assert result['dataset']['root_dir'] == str(dataset_dir)
        assert result['dataset']['image_size'] == 256
        assert result['training']['batch_size'] == 32
        assert result['training']['learning_rate'] == 0.001
    
    def test_load_config_file_not_found(self):
        """Test loading fails when config file does not exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("/nonexistent/config.yaml")
    
    def test_load_config_unsupported_format(self, tmp_path):
        """Test loading fails with unsupported file format."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some text")
        
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config(str(config_file))
    
    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading fails with malformed YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            load_config(str(config_file))
    
    def test_load_config_invalid_json(self, tmp_path):
        """Test loading fails with malformed JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"invalid": json content}')
        
        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))
    
    def test_load_config_validates_content(self, tmp_path):
        """Test that load_config validates the loaded configuration."""
        # Create config with missing required parameter
        config_file = tmp_path / "config.yaml"
        config_data = {
            'dataset': {
                'image_size': 256
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Should fail validation
        with pytest.raises(ValueError, match="Missing required parameter: 'dataset.root_dir'"):
            load_config(str(config_file))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
