"""
Utility script to check fusion layer weights in a checkpoint.
"""
import torch
import sys
from pathlib import Path
from datetime import datetime


def check_fusion_weights(checkpoint_path: str, output_file: str = None):
    """
    Load a checkpoint and inspect fusion layer weights.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_file: Optional path to save output (default: fusion_weights_report.txt)
    """
    if not Path(checkpoint_path).exists():
        msg = f"Error: Checkpoint not found at {checkpoint_path}"
        print(msg)
        return
    
    # Determine output file path
    if output_file is None:
        output_file = "fusion_weights_report.txt"
    
    # Collect all output in a list
    output_lines = []
    
    def log(msg):
        """Print and save to output"""
        print(msg)
        output_lines.append(msg)
    
    log(f"Fusion Layer Weight Analysis")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Checkpoint: {checkpoint_path}")
    log("=" * 80)
    log("")
    
    log(f"Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' not in checkpoint:
        log("Error: 'model_state_dict' not found in checkpoint")
        log(f"Available keys: {list(checkpoint.keys())}")
        # Save and return
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        log(f"\nReport saved to: {output_file}")
        return
    
    state = checkpoint['model_state_dict']
    
    log("\n=== Fusion Layer Weights ===")
    fusion_found = False
    
    for k, v in state.items():
        if 'fusion' in k.lower():
            fusion_found = True
            log(f"{k}:")
            log(f"  Shape: {v.shape}")
            log(f"  Mean: {v.mean():.6f}")
            log(f"  Std: {v.std():.6f}")
            log(f"  Max (abs): {v.abs().max():.6f}")
            log(f"  Min: {v.min():.6f}")
            log(f"  Max: {v.max():.6f}")
            log("")
    
    if not fusion_found:
        log("No fusion layer weights found in checkpoint.")
        log("\nAll available keys:")
        for k in sorted(state.keys()):
            log(f"  {k}")
    
    # Also check for other metadata
    log("\n=== Checkpoint Metadata ===")
    for key in checkpoint.keys():
        if key != 'model_state_dict':
            log(f"{key}: {checkpoint[key]}")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    log(f"\n{'=' * 80}")
    log(f"Report saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_fusion_weights.py <checkpoint_path> [output_file]")
        print("Example: python check_fusion_weights.py checkpoints/all_features/checkpoint_epoch_80.pth")
        print("         python check_fusion_weights.py checkpoints/all_features/checkpoint_epoch_80.pth my_report.txt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    check_fusion_weights(checkpoint_path, output_file)
