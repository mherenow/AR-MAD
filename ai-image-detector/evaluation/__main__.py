"""
Evaluation script for AI Image Detector.

This module implements the evaluation pipeline for the binary classifier on the
SynthBuster test dataset. It loads a trained model checkpoint, evaluates it on
the test set, and provides comprehensive metrics including per-generator breakdown.

Usage:
    python -m ai-image-detector.evaluation --config configs/default_config.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from data.synthbuster_loader import SynthBusterDataset
from models.classifier import BinaryClassifier
from evaluation.evaluate import evaluate_model, print_evaluation_report


def main():
    """Main evaluation function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate AI Image Detector')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pth)'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Set device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize test dataset
    print("Loading test dataset...")
    root_dir = config['dataset']['root_dir']
    
    test_dataset = SynthBusterDataset(root_dir=root_dir)
    print(f"Test samples: {len(test_dataset)}")
    
    # Create test data loader
    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['dataset'].get('num_workers', 4)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing model...")
    backbone_type = config['model']['backbone_type']
    pretrained = config['model'].get('pretrained', True)
    
    model = BinaryClassifier(
        backbone_type=backbone_type,
        pretrained=pretrained
    )
    
    print(f"Model: BinaryClassifier with {backbone_type} backbone")
    print(f"Loading checkpoint from {args.checkpoint}...")
    
    # Run evaluation
    print("\nEvaluating model on test set...")
    print("=" * 70)
    
    metrics = evaluate_model(
        checkpoint_path=args.checkpoint,
        model=model,
        dataloader=test_loader,
        device=device
    )
    
    # Print evaluation report
    print_evaluation_report(metrics)
    
    print("=" * 70)
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
