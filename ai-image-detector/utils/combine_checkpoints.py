"""
Utility for combining separately trained model branches.

This module allows you to train different branches (spectral, noise, color, etc.)
separately and then combine them into a single unified model for inference or
fine-tuning.
"""

import torch
import argparse
from typing import Dict, Optional
import os


def load_branch_weights(checkpoint_path: str, branch_prefix: str) -> Dict[str, torch.Tensor]:
    """
    Load weights for a specific branch from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        branch_prefix: Prefix of the branch to extract (e.g., 'spectral_branch', 'noise_branch')
    
    Returns:
        Dictionary of state dict entries for the specified branch
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state_dict']
    
    # Extract only the weights for the specified branch
    branch_weights = {}
    for key, value in model_state.items():
        if key.startswith(branch_prefix):
            branch_weights[key] = value
    
    return branch_weights


def combine_checkpoints(
    backbone_checkpoint: str,
    spectral_checkpoint: Optional[str] = None,
    noise_checkpoint: Optional[str] = None,
    color_checkpoint: Optional[str] = None,
    output_path: str = 'combined_model.pth',
    config: Optional[dict] = None
) -> None:
    """
    Combine separately trained branches into a single checkpoint.
    
    Args:
        backbone_checkpoint: Path to checkpoint with trained backbone
        spectral_checkpoint: Path to checkpoint with trained spectral branch (optional)
        noise_checkpoint: Path to checkpoint with trained noise branch (optional)
        color_checkpoint: Path to checkpoint with trained color branch (optional)
        output_path: Path where combined checkpoint will be saved
        config: Configuration dictionary for the combined model (optional)
    """
    print("Combining checkpoints...")
    
    # Load backbone checkpoint as base
    print(f"Loading backbone from: {backbone_checkpoint}")
    base_checkpoint = torch.load(backbone_checkpoint, map_location='cpu')
    combined_state = base_checkpoint['model_state_dict'].copy()
    
    # Load and merge spectral branch if provided
    if spectral_checkpoint:
        print(f"Loading spectral branch from: {spectral_checkpoint}")
        spectral_weights = load_branch_weights(spectral_checkpoint, 'spectral_branch')
        combined_state.update(spectral_weights)
        print(f"  Added {len(spectral_weights)} spectral branch parameters")
    
    # Load and merge noise branch if provided
    if noise_checkpoint:
        print(f"Loading noise branch from: {noise_checkpoint}")
        noise_extractor_weights = load_branch_weights(noise_checkpoint, 'noise_extractor')
        noise_branch_weights = load_branch_weights(noise_checkpoint, 'noise_branch')
        combined_state.update(noise_extractor_weights)
        combined_state.update(noise_branch_weights)
        print(f"  Added {len(noise_extractor_weights) + len(noise_branch_weights)} noise branch parameters")
    
    # Load and merge color branch if provided
    if color_checkpoint:
        print(f"Loading color branch from: {color_checkpoint}")
        rgb_to_ycbcr_weights = load_branch_weights(color_checkpoint, 'rgb_to_ycbcr')
        chrominance_weights = load_branch_weights(color_checkpoint, 'chrominance_branch')
        combined_state.update(rgb_to_ycbcr_weights)
        combined_state.update(chrominance_weights)
        print(f"  Added {len(rgb_to_ycbcr_weights) + len(chrominance_weights)} color branch parameters")
    
    # Create combined checkpoint
    combined_checkpoint = {
        'model_state_dict': combined_state,
        'config': config or base_checkpoint.get('config'),
        'combined_from': {
            'backbone': backbone_checkpoint,
            'spectral': spectral_checkpoint,
            'noise': noise_checkpoint,
            'color': color_checkpoint
        }
    }
    
    # Save combined checkpoint
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(combined_checkpoint, output_path)
    print(f"\n✓ Combined checkpoint saved to: {output_path}")
    print(f"  Total parameters: {len(combined_state)}")


def main():
    """Command-line interface for combining checkpoints."""
    parser = argparse.ArgumentParser(
        description='Combine separately trained model branches into a single checkpoint'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        required=True,
        help='Path to checkpoint with trained backbone'
    )
    parser.add_argument(
        '--spectral',
        type=str,
        default=None,
        help='Path to checkpoint with trained spectral branch'
    )
    parser.add_argument(
        '--noise',
        type=str,
        default=None,
        help='Path to checkpoint with trained noise branch'
    )
    parser.add_argument(
        '--color',
        type=str,
        default=None,
        help='Path to checkpoint with trained color branch'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoints/combined_model.pth',
        help='Path where combined checkpoint will be saved'
    )
    
    args = parser.parse_args()
    
    combine_checkpoints(
        backbone_checkpoint=args.backbone,
        spectral_checkpoint=args.spectral,
        noise_checkpoint=args.noise,
        color_checkpoint=args.color,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
