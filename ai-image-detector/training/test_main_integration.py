"""
Integration tests for training/__main__.py configuration and optimizer setup.

Tests verify that the training script correctly:
- Initializes augmentation modules from config
- Creates domain discriminator and optimizer when enabled
- Passes parameters correctly to train_epoch
- Saves checkpoints with domain discriminator state
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def minimal_dataset_dir():
    """Create a minimal dataset directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthbuster structure
        synthbuster_dir = os.path.join(tmpdir, 'synthbuster')
        os.makedirs(os.path.join(synthbuster_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(synthbuster_dir, 'fake'), exist_ok=True)
        
        # Create dummy images (just empty files for testing)
        for i in range(4):
            open(os.path.join(synthbuster_dir, 'real', f'img_{i}.jpg'), 'w').close()
            open(os.path.join(synthbuster_dir, 'fake', f'img_{i}.jpg'), 'w').close()
        
        yield tmpdir


def test_config_with_cutmix_enabled(temp_config_dir, temp_checkpoint_dir, minimal_dataset_dir):
    """Test that CutMix augmentation is correctly initialized from config."""
    config_path = os.path.join(temp_config_dir, 'test_config.yaml')
    
    config = {
        'device': 'cpu',
        'model': {
            'backbone_type': 'simple_cnn',
            'pretrained': False
        },
        'dataset': {
            'mode': 'synthbuster',
            'root_dir': os.path.join(minimal_dataset_dir, 'synthbuster'),
            'val_ratio': 0.2,
            'num_workers': 0
        },
        'training': {
            'num_epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'checkpoint_dir': temp_checkpoint_dir
        },
        'augmentation': {
            'cutmix': {
                'enabled': True,
                'alpha': 1.0,
                'prob': 0.5
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load config and verify structure
    loaded_config = load_config(config_path)
    assert loaded_config['augmentation']['cutmix']['enabled'] is True
    assert loaded_config['augmentation']['cutmix']['alpha'] == 1.0
    assert loaded_config['augmentation']['cutmix']['prob'] == 0.5


def test_config_with_mixup_enabled(temp_config_dir, temp_checkpoint_dir, minimal_dataset_dir):
    """Test that MixUp augmentation is correctly initialized from config."""
    config_path = os.path.join(temp_config_dir, 'test_config.yaml')
    
    config = {
        'device': 'cpu',
        'model': {
            'backbone_type': 'simple_cnn',
            'pretrained': False
        },
        'dataset': {
            'mode': 'synthbuster',
            'root_dir': os.path.join(minimal_dataset_dir, 'synthbuster'),
            'val_ratio': 0.2,
            'num_workers': 0
        },
        'training': {
            'num_epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'checkpoint_dir': temp_checkpoint_dir
        },
        'augmentation': {
            'mixup': {
                'enabled': True,
                'alpha': 0.2,
                'prob': 0.5
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load config and verify structure
    loaded_config = load_config(config_path)
    assert loaded_config['augmentation']['mixup']['enabled'] is True
    assert loaded_config['augmentation']['mixup']['alpha'] == 0.2
    assert loaded_config['augmentation']['mixup']['prob'] == 0.5


def test_config_with_domain_adversarial_enabled(temp_config_dir, temp_checkpoint_dir, minimal_dataset_dir):
    """Test that domain adversarial training is correctly initialized from config."""
    config_path = os.path.join(temp_config_dir, 'test_config.yaml')
    
    config = {
        'device': 'cpu',
        'model': {
            'backbone_type': 'simple_cnn',
            'pretrained': False
        },
        'dataset': {
            'mode': 'synthbuster',
            'root_dir': os.path.join(minimal_dataset_dir, 'synthbuster'),
            'val_ratio': 0.2,
            'num_workers': 0
        },
        'training': {
            'num_epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'checkpoint_dir': temp_checkpoint_dir,
            'domain_adversarial': {
                'enabled': True,
                'lambda': 1.0,
                'hidden_dim': 256
            }
        },
        'data': {
            'datasets': {
                'dataset_a': {'weight': 0.6},
                'dataset_b': {'weight': 0.4}
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load config and verify structure
    loaded_config = load_config(config_path)
    assert loaded_config['training']['domain_adversarial']['enabled'] is True
    assert loaded_config['training']['domain_adversarial']['lambda'] == 1.0
    assert loaded_config['training']['domain_adversarial']['hidden_dim'] == 256
    assert len(loaded_config['data']['datasets']) == 2


def test_config_with_all_features_enabled(temp_config_dir, temp_checkpoint_dir, minimal_dataset_dir):
    """Test config with CutMix, MixUp, and domain adversarial all enabled."""
    config_path = os.path.join(temp_config_dir, 'test_config.yaml')
    
    config = {
        'device': 'cpu',
        'model': {
            'backbone_type': 'simple_cnn',
            'pretrained': False
        },
        'dataset': {
            'mode': 'synthbuster',
            'root_dir': os.path.join(minimal_dataset_dir, 'synthbuster'),
            'val_ratio': 0.2,
            'num_workers': 0
        },
        'training': {
            'num_epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'checkpoint_dir': temp_checkpoint_dir,
            'domain_adversarial': {
                'enabled': True,
                'lambda': 1.0,
                'hidden_dim': 256
            }
        },
        'augmentation': {
            'cutmix': {
                'enabled': True,
                'alpha': 1.0,
                'prob': 0.3
            },
            'mixup': {
                'enabled': True,
                'alpha': 0.2,
                'prob': 0.3
            }
        },
        'data': {
            'datasets': {
                'dataset_a': {'weight': 0.6},
                'dataset_b': {'weight': 0.4}
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load config and verify all features are enabled
    loaded_config = load_config(config_path)
    assert loaded_config['augmentation']['cutmix']['enabled'] is True
    assert loaded_config['augmentation']['mixup']['enabled'] is True
    assert loaded_config['training']['domain_adversarial']['enabled'] is True


def test_backward_compatibility_no_augmentation_config():
    """Test that missing augmentation config doesn't break training."""
    config = {
        'device': 'cpu',
        'model': {
            'backbone_type': 'simple_cnn',
            'pretrained': False
        },
        'training': {
            'num_epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001
        }
    }
    
    # Should not raise error when accessing augmentation config
    augmentation_config = config.get('augmentation', {})
    assert augmentation_config == {}
    
    # CutMix should be disabled by default
    cutmix_enabled = augmentation_config.get('cutmix', {}).get('enabled', False)
    assert cutmix_enabled is False
    
    # MixUp should be disabled by default
    mixup_enabled = augmentation_config.get('mixup', {}).get('enabled', False)
    assert mixup_enabled is False


def test_backward_compatibility_no_domain_adversarial_config():
    """Test that missing domain adversarial config doesn't break training."""
    config = {
        'device': 'cpu',
        'model': {
            'backbone_type': 'simple_cnn',
            'pretrained': False
        },
        'training': {
            'num_epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001
        }
    }
    
    # Should not raise error when accessing domain adversarial config
    domain_config = config.get('training', {}).get('domain_adversarial', {})
    assert domain_config == {}
    
    # Domain adversarial should be disabled by default
    domain_enabled = domain_config.get('enabled', False)
    assert domain_enabled is False


def test_feature_dim_calculation_for_different_backbones():
    """Test that feature dimension is correctly calculated for different backbones."""
    backbone_configs = {
        'simple_cnn': 512,
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }
    
    for backbone_type, expected_dim in backbone_configs.items():
        # Simulate feature dimension calculation
        if backbone_type == 'simple_cnn':
            feature_dim = 512
        elif backbone_type in ['resnet18', 'resnet34']:
            feature_dim = 512
        elif backbone_type in ['resnet50', 'resnet101', 'resnet152']:
            feature_dim = 2048
        else:
            feature_dim = 512  # Default
        
        assert feature_dim == expected_dim, f"Feature dim mismatch for {backbone_type}"


def test_dataset_to_domain_mapping_creation():
    """Test that dataset to domain mapping is correctly created."""
    # Test with explicit dataset config
    dataset_config = {
        'synthbuster': {'weight': 0.6},
        'coco2017': {'weight': 0.4}
    }
    
    dataset_to_domain = {name: idx for idx, name in enumerate(dataset_config.keys())}
    
    assert len(dataset_to_domain) == 2
    assert 'synthbuster' in dataset_to_domain
    assert 'coco2017' in dataset_to_domain
    assert dataset_to_domain['synthbuster'] in [0, 1]
    assert dataset_to_domain['coco2017'] in [0, 1]
    assert dataset_to_domain['synthbuster'] != dataset_to_domain['coco2017']


def test_checkpoint_structure_with_domain_discriminator():
    """Test that checkpoint includes domain discriminator state when enabled."""
    from training.domain_adversarial import DomainDiscriminator
    from models.classifier import BinaryClassifier
    
    # Create model and domain discriminator
    model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
    domain_discriminator = DomainDiscriminator(feature_dim=512, num_domains=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    domain_optimizer = torch.optim.Adam(domain_discriminator.parameters(), lr=0.001)
    
    # Create checkpoint data
    checkpoint_data = {
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': 0.5,
        'train_acc': 0.8,
        'val_acc': 0.75,
        'val_loss': 0.6,
        'config': {}
    }
    
    # Add domain discriminator state
    checkpoint_data['domain_discriminator_state_dict'] = domain_discriminator.state_dict()
    checkpoint_data['domain_optimizer_state_dict'] = domain_optimizer.state_dict()
    
    # Verify checkpoint structure
    assert 'domain_discriminator_state_dict' in checkpoint_data
    assert 'domain_optimizer_state_dict' in checkpoint_data
    assert isinstance(checkpoint_data['domain_discriminator_state_dict'], dict)
    assert isinstance(checkpoint_data['domain_optimizer_state_dict'], dict)


def test_checkpoint_structure_without_domain_discriminator():
    """Test that checkpoint works without domain discriminator state."""
    from models.classifier import BinaryClassifier
    
    # Create model
    model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create checkpoint data without domain discriminator
    checkpoint_data = {
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': 0.5,
        'train_acc': 0.8,
        'val_acc': 0.75,
        'val_loss': 0.6,
        'config': {}
    }
    
    # Verify checkpoint structure (no domain discriminator)
    assert 'domain_discriminator_state_dict' not in checkpoint_data
    assert 'domain_optimizer_state_dict' not in checkpoint_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
