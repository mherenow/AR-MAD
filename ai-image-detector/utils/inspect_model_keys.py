"""
Inspect model parameter names and compare against a checkpoint.
Saves results to a txt file.
"""
import sys
import os
import torch
from datetime import datetime

# Allow imports from the ai-image-detector package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import BinaryClassifier
from utils.config_loader import load_config


def inspect_model_keys(
    config_path: str,
    checkpoint_path: str,
    output_file: str = "model_key_inspection.txt"
):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"Model Key Inspection Report")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Config:     {config_path}")
    log(f"Checkpoint: {checkpoint_path}")
    log("=" * 80)

    # --- Build model from config ---
    config = load_config(config_path)
    model_cfg = config.get('model', {})

    model = BinaryClassifier(
        backbone_type=model_cfg.get('backbone_type', 'resnet50'),
        pretrained=False,
        use_spectral=model_cfg.get('use_spectral', False),
        use_noise_imprint=model_cfg.get('use_noise_imprint', False),
        use_color_features=model_cfg.get('use_color_features', False),
        use_local_patches=model_cfg.get('use_local_patches', False),
        use_fpn=model_cfg.get('use_fpn', False),
        use_attention=model_cfg.get('use_attention', None),
        enable_attribution=model_cfg.get('enable_attribution', False),
    )

    # --- All model parameter names ---
    log("\n=== All Model Parameter Names ===")
    model_keys = set()
    for name, _ in model.named_parameters():
        log(f"  {name}")
        model_keys.add(name)

    # --- Load checkpoint ---
    log("\n=== Checkpoint vs Model Key Comparison ===")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ckpt_keys = set(checkpoint['model_state_dict'].keys())

    log(f"\nCheckpoint keys : {len(ckpt_keys)}")
    log(f"Model keys      : {len(model_keys)}")

    missing_from_ckpt = sorted(model_keys - ckpt_keys)
    extra_in_ckpt = sorted(ckpt_keys - model_keys)

    log(f"\nIn model but NOT in checkpoint (branches that never trained): {len(missing_from_ckpt)}")
    for k in missing_from_ckpt:
        log(f"  {k}")

    log(f"\nIn checkpoint but NOT in model (stale / renamed keys): {len(extra_in_ckpt)}")
    for k in extra_in_ckpt:
        log(f"  {k}")

    if not missing_from_ckpt and not extra_in_ckpt:
        log("\nAll keys match perfectly.")

    # --- Save ---
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    log(f"\n{'=' * 80}")
    log(f"Report saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_model_keys.py <config_path> <checkpoint_path> [output_file]")
        print("Example: python inspect_model_keys.py configs/all_features.yaml "
              "../checkpoints/all_features/checkpoint_epoch_80.pth")
        sys.exit(1)

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "model_key_inspection.txt"

    inspect_model_keys(config_path, checkpoint_path, output_file)
