"""
Configuration loader and validator for AI Image Detector.

This module provides utilities for loading and validating configuration files
in YAML or JSON format. It ensures all required parameters are present and
validates their values according to the system requirements.
"""

import os
import yaml
import json
from typing import Dict, Any


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
    
    # Validate the loaded configuration
    config = validate_config(config)
    
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
