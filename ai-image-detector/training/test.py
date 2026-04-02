import torch
import sys
sys.path.insert(0, 'ai-image-detector')

from utils.config_loader import load_config
from models.classifier import BinaryClassifier

config = load_config('ai-image-detector/configs/all_features.yaml')

model = BinaryClassifier(
    backbone_type=config['model']['backbone_type'],
    pretrained=config['model'].get('pretrained', True)
)

# Check what state_dict actually contains
sd = model.state_dict()
branch_prefixes = ['spectral_branch', 'noise_branch', 'chrominance_branch',
                   'fpn', 'fusion_layer', 'attention_module']

print("=== state_dict branch keys ===")
for prefix in branch_prefixes:
    keys = [k for k in sd.keys() if k.startswith(prefix)]
    print(f"{prefix}: {len(keys)} keys in state_dict")

print("\n=== named_parameters branch params ===")
for prefix in branch_prefixes:
    params = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
    print(f"{prefix}: {len(params)} params")

print("\n=== BinaryClassifier __init__ signature ===")
import inspect
print(inspect.getsource(BinaryClassifier.__init__))