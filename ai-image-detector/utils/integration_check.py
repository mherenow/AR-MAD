"""
Integration test to verify config loader works with default_config.yaml
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config

# Test loading the default config
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.yaml')

try:
    # This will fail because the dataset directory doesn't exist yet
    # But we can test if it properly validates
    config = load_config(config_path)
    print("✓ Config loaded successfully!")
    print(f"  Dataset root: {config['dataset']['root_dir']}")
    print(f"  Image size: {config['dataset']['image_size']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Backbone: {config['model']['backbone_type']}")
except ValueError as e:
    # Expected error: dataset directory doesn't exist
    if "Dataset root directory does not exist" in str(e):
        print("✓ Config validation working correctly!")
        print(f"  Expected error caught: {e}")
    else:
        print(f"✗ Unexpected validation error: {e}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

print("\n✓ Integration test passed!")
