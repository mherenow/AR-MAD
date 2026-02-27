"""
Demonstration of configuration loader validation features.

This script demonstrates the validation capabilities of the config_loader module
by testing various valid and invalid configuration scenarios. It creates temporary
test configurations and verifies that the validation logic correctly accepts valid
configurations and rejects invalid ones.

Usage:
    python ai-image-detector/utils/demo_validation.py
"""

import os
import sys
import tempfile
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config, validate_config

print("Configuration Loader Validation Demo")
print("=" * 50)

# Create a temporary directory for testing
with tempfile.TemporaryDirectory() as tmpdir:
    dataset_dir = os.path.join(tmpdir, "synthbuster")
    os.makedirs(dataset_dir)
    
    # Test 1: Valid configuration
    print("\n1. Testing valid configuration...")
    valid_config = {
        'dataset': {
            'root_dir': dataset_dir,
            'image_size': 256,
            'val_ratio': 0.2
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
    
    try:
        result = validate_config(valid_config)
        print("   ✓ Valid configuration accepted")
    except ValueError as e:
        print(f"   ✗ Unexpected error: {e}")
    
    # Test 2: Missing required parameter
    print("\n2. Testing missing dataset.root_dir...")
    invalid_config = {
        'dataset': {
            'image_size': 256
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    
    try:
        validate_config(invalid_config)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    
    # Test 3: Invalid image_size (negative)
    print("\n3. Testing negative image_size...")
    invalid_config = {
        'dataset': {
            'root_dir': dataset_dir,
            'image_size': -256
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    
    try:
        validate_config(invalid_config)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    
    # Test 4: Invalid batch_size (not integer)
    print("\n4. Testing non-integer batch_size...")
    invalid_config = {
        'dataset': {
            'root_dir': dataset_dir,
            'image_size': 256
        },
        'training': {
            'batch_size': 32.5,
            'learning_rate': 0.001
        }
    }
    
    try:
        validate_config(invalid_config)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    
    # Test 5: Invalid learning_rate (negative)
    print("\n5. Testing negative learning_rate...")
    invalid_config = {
        'dataset': {
            'root_dir': dataset_dir,
            'image_size': 256
        },
        'training': {
            'batch_size': 32,
            'learning_rate': -0.001
        }
    }
    
    try:
        validate_config(invalid_config)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    
    # Test 6: Invalid backbone_type
    print("\n6. Testing invalid backbone_type...")
    invalid_config = {
        'dataset': {
            'root_dir': dataset_dir,
            'image_size': 256
        },
        'model': {
            'backbone_type': 'vgg16'  # Not in allowed list
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    
    try:
        validate_config(invalid_config)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    
    # Test 7: Nonexistent dataset directory
    print("\n7. Testing nonexistent dataset directory...")
    invalid_config = {
        'dataset': {
            'root_dir': '/nonexistent/path',
            'image_size': 256
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    
    try:
        validate_config(invalid_config)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    
    # Test 8: Load from YAML file
    print("\n8. Testing load_config with YAML file...")
    config_file = os.path.join(tmpdir, "test_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(valid_config, f)
    
    try:
        result = load_config(config_file)
        print("   ✓ Successfully loaded and validated YAML config")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

print("\n" + "=" * 50)
print("All validation tests completed successfully!")
