# AI Image Detector

A binary classification system for detecting AI-generated images using the SynthBuster benchmark dataset. This project provides a modular, extensible foundation for deepfake image detection research.

## Overview

The AI Image Detector distinguishes between real photographs (from the RAISE dataset) and AI-generated images from multiple diffusion models including:
- Stable Diffusion v2
- Midjourney
- GLIDE
- Adobe Firefly
- DALL·E

## Features

- **Modular Architecture**: Separate components for data loading, model definition, training, and evaluation
- **Flexible Backbones**: Support for SimpleCNN, ResNet18, and ResNet50 architectures
- **Per-Generator Metrics**: Detailed performance breakdown for each AI generation model
- **Configuration Management**: YAML-based configuration for easy experimentation
- **Checkpoint Management**: Save and resume training with full state preservation
- **Comprehensive Testing**: Unit tests for all major components

## Project Structure

```
ai-image-detector/
├── data/                           # Dataset loading and preprocessing
│   ├── __init__.py
│   ├── synthbuster_loader.py       # SynthBuster dataset loader with train/val split
│   └── test_synthbuster_loader.py  # Unit tests for data loading
├── models/                         # Model architectures
│   ├── __init__.py
│   ├── classifier.py               # Binary classifier wrapper
│   ├── backbones.py                # SimpleCNN, ResNet18, ResNet50 backbones
│   └── test_model_architecture.py  # Unit tests for model components
├── training/                       # Training utilities
│   ├── __init__.py
│   ├── __main__.py                 # Main training script (entry point)
│   ├── train.py                    # Training loop and utilities
│   └── test_training_integration.py # Integration tests for training
├── evaluation/                     # Evaluation and metrics
│   ├── __init__.py
│   ├── __main__.py                 # Main evaluation script (entry point)
│   ├── evaluate.py                 # Per-generator metrics computation
│   └── test_evaluation_metrics.py  # Unit tests for evaluation metrics
├── configs/                        # Configuration files
│   └── default_config.yaml         # Default hyperparameters
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── config_loader.py            # YAML configuration file parser
│   └── test_config_loader.py       # Unit tests for config loading
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Workspace Architecture

### Module Dependencies

The project follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Points                              │
│  training/__main__.py          evaluation/__main__.py        │
└────────────┬──────────────────────────────┬─────────────────┘
             │                               │
             ▼                               ▼
┌────────────────────────┐      ┌──────────────────────────┐
│   training/train.py    │      │  evaluation/evaluate.py  │
│  - Training loop       │      │  - Metrics computation   │
│  - Checkpoint saving   │      │  - Per-generator eval    │
└──┬──────────┬──────────┘      └──┬────────────┬──────────┘
   │          │                    │            │
   │          └────────┬───────────┘            │
   │                   ▼                        │
   │         ┌──────────────────┐               │
   │         │  utils/          │               │
   │         │  config_loader   │               │
   │         │  - YAML parsing  │               │
   │         │  - Validation    │               │
   │         └──────────────────┘               │
   │                                            │
   ├────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│                    Core Modules                           │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  data/          │  │  models/                     │  │
│  │  synthbuster_   │  │  - backbones.py              │  │
│  │  loader         │  │  - classifier.py             │  │
│  │  (independent)  │  │  (depends on backbones)      │  │
│  └─────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Module Responsibilities

#### data/synthbuster_loader.py
**Purpose:** Load and preprocess SynthBuster dataset images

**Dependencies:** 
- External only: PyTorch, PIL, torchvision
- NO workspace dependencies (self-contained)

**Used by:**
- training/train.py
- evaluation/evaluate.py

**Design rationale:** Self-contained for maximum portability and reusability

#### models/backbones.py
**Purpose:** Define CNN architectures (SimpleCNN, ResNet18, ResNet50)

**Dependencies:**
- External only: PyTorch, torchvision

**Used by:**
- models/classifier.py
- training/train.py

#### models/classifier.py
**Purpose:** Binary classifier wrapper combining backbone + classification head

**Dependencies:**
- models/backbones.py

**Used by:**
- training/train.py
- evaluation/evaluate.py

#### utils/config_loader.py
**Purpose:** Load and validate YAML configuration files

**Dependencies:**
- External only: PyYAML, os

**Used by:**
- training/__main__.py
- evaluation/__main__.py

**Design rationale:** Centralized configuration management with validation

#### training/train.py
**Purpose:** Training loop, optimizer setup, checkpoint management

**Dependencies:**
- data/synthbuster_loader.py
- models/classifier.py
- utils/config_loader.py (via __main__.py)

#### evaluation/evaluate.py
**Purpose:** Model evaluation with per-generator metrics

**Dependencies:**
- data/synthbuster_loader.py
- models/classifier.py
- utils/config_loader.py (via __main__.py)

### Workspace Conventions

#### Import Style
```python
# Absolute imports from workspace root
from ai-image-detector.data import SynthBusterDataset
from ai-image-detector.models import SimpleCNN, get_resnet18
from ai-image-detector.utils import load_config

# Relative imports within same module
from .backbones import SimpleCNN
```

#### Configuration Flow
1. YAML config file (configs/default_config.yaml)
2. Loaded by utils/config_loader.py with validation
3. Passed to training/evaluation scripts
4. Parameters extracted and passed to modules

Example:
```python
# In training/__main__.py
config = load_config('configs/default_config.yaml')

# Extract parameters
dataset = SynthBusterDataset(
    root_dir=config['dataset']['root_dir']
)

model = BinaryClassifier(
    backbone_type=config['model']['backbone_type'],
    pretrained=config['model']['pretrained']
)
```

#### Testing Conventions
- Each module has a corresponding test_*.py file
- Tests use pytest framework
- Tests are self-contained and can run independently
- Test files include integration tests for cross-module functionality

#### Type Hints
All functions use Python type hints:
```python
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    pass

def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
    """Get dataset item."""
    pass
```

#### Docstring Format
Google-style docstrings with Args, Returns, Raises, and Examples:
```python
def create_train_val_split(
    root_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[list, list]:
    """
    Create train/validation split for SynthBuster dataset.
    
    Args:
        root_dir: Root directory containing the dataset folders
        val_ratio: Ratio of validation samples (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_paths, val_paths)
        
    Example:
        >>> train, val = create_train_val_split('data/synthbuster')
        >>> print(f"Train: {len(train)}, Val: {len(val)}")
    """
    pass
```

### Adding New Modules

When adding new functionality to the workspace:

1. **Determine module location:**
   - Data processing → `data/`
   - Model architectures → `models/`
   - Training utilities → `training/`
   - Evaluation utilities → `evaluation/`
   - Shared utilities → `utils/`

2. **Follow dependency principles:**
   - Minimize cross-module dependencies
   - Keep data loaders self-contained
   - Use utils/ for shared functionality
   - Configuration flows through config_loader

3. **Add corresponding tests:**
   - Create test_*.py file in same directory
   - Use pytest for test framework
   - Include unit and integration tests

4. **Update __init__.py:**
   - Export public APIs via __all__
   - Keep imports clean and explicit

5. **Document dependencies:**
   - Add module docstring explaining dependencies
   - Update this README if adding new patterns
   - Include usage examples in docstrings

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 10+ GB disk space for dataset and checkpoints

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd ai-image-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

The SynthBuster dataset should be organized in the following structure:

```
data/synthbuster/
├── real/
│   └── RAISE/
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
└── fake/
    ├── SD_v2/
    │   ├── image001.png
    │   └── ...
    ├── GLIDE/
    │   ├── image001.png
    │   └── ...
    ├── Firefly/
    │   ├── image001.png
    │   └── ...
    ├── DALLE/
    │   ├── image001.png
    │   └── ...
    └── Midjourney/
        ├── image001.png
        └── ...
```

**Dataset Structure Requirements:**
- `real/RAISE/`: Contains real photographs from the RAISE dataset
- `fake/<generator_name>/`: Contains AI-generated images from each model
- Supported image formats: `.jpg`, `.jpeg`, `.png`
- Images will be automatically resized to the configured size (default: 256x256)

**Download Instructions:**
1. Download the SynthBuster dataset from the official source
2. Extract the dataset to `data/synthbuster/` following the structure above
3. Verify the dataset structure matches the expected format

## Usage

### Training

Train a model using the default configuration:
```bash
python -m ai-image-detector.training --config ai-image-detector/configs/default_config.yaml
```

Train with a custom configuration:
```bash
python -m ai-image-detector.training --config path/to/custom_config.yaml
```

**Training Output:**
- Model checkpoints saved to `checkpoints/` directory (configurable)
- Best model saved as `checkpoints/best_model.pth` based on validation accuracy
- Training progress logged to console with loss and accuracy metrics

**Example Training Output:**
```
Loading configuration from configs/default_config.yaml...
Using device: cuda
Initializing dataset...
Train samples: 8000, Val samples: 2000
Initializing model...
Model: BinaryClassifier with resnet18 backbone
Pretrained: True

Starting training for 10 epochs...
======================================================================
Epoch [1/10]
  Train Loss: 0.3245 | Train Acc: 0.8567
  Val Loss:   0.2891 | Val Acc:   0.8823
  ✓ Saved best model (val_acc: 0.8823)
----------------------------------------------------------------------
Epoch [2/10]
  Train Loss: 0.2134 | Train Acc: 0.9123
  Val Loss:   0.1987 | Val Acc:   0.9245
  ✓ Saved best model (val_acc: 0.9245)
----------------------------------------------------------------------
...
======================================================================
Training complete! Best validation accuracy: 0.9456
Best model saved to: checkpoints/best_model.pth
```

### Evaluation

Evaluate a trained model on the test set:
```bash
python -m ai-image-detector.evaluation --config ai-image-detector/configs/default_config.yaml --checkpoint checkpoints/best_model.pth
```

**Evaluation Output:**
- Overall accuracy and AUC-ROC score
- Per-generator breakdown showing performance on each AI model
- Detailed metrics printed to console

**Example Evaluation Output:**
```
========================================
EVALUATION REPORT
========================================
Total Samples: 10000

Overall Metrics:
  Accuracy: 92.50%
  AUC:      0.956

Per-Generator Metrics:
----------------------------------------
DALLE:
  Accuracy: 94.20%
  AUC:      0.968

Firefly:
  Accuracy: 91.80%
  AUC:      0.952

GLIDE:
  Accuracy: 89.50%
  AUC:      0.941

Midjourney:
  Accuracy: 93.10%
  AUC:      0.961

RAISE:
  Accuracy: 94.50%
  AUC:      0.972

SD_v2:
  Accuracy: 90.40%
  AUC:      0.947
----------------------------------------
```

### Running Tests

Run all unit tests:
```bash
# Test data loading
python -m pytest ai-image-detector/data/test_synthbuster_loader.py -v

# Test model architecture
python -m pytest ai-image-detector/models/test_model_architecture.py -v

# Test evaluation metrics
python -m pytest ai-image-detector/evaluation/test_evaluation_metrics.py -v

# Test configuration loading
python -m pytest ai-image-detector/utils/test_config_loader.py -v

# Run all tests
python -m pytest ai-image-detector/ -v
```

## Configuration

All hyperparameters and settings are managed through YAML configuration files. The default configuration is located at `configs/default_config.yaml`.

### Configuration Parameters

#### Dataset Configuration
```yaml
dataset:
  root_dir: "data/synthbuster"    # Path to SynthBuster dataset root
  image_size: 256                 # Target image size (width and height)
  val_ratio: 0.2                  # Validation split ratio (0.0 to 1.0)
  num_workers: 4                  # Number of data loading workers
```

#### Model Configuration
```yaml
model:
  backbone_type: "resnet18"       # Options: simple_cnn, resnet18, resnet50
  pretrained: true                # Use ImageNet pretrained weights
  freeze_backbone: false          # Freeze backbone layers during training
```

**Backbone Options:**
- `simple_cnn`: Lightweight 4-layer CNN (fast training, baseline performance)
- `resnet18`: ResNet-18 with pretrained weights (balanced speed/accuracy)
- `resnet50`: ResNet-50 with pretrained weights (best accuracy, slower training)

#### Training Configuration
```yaml
training:
  batch_size: 32                  # Training batch size
  learning_rate: 0.001            # Initial learning rate
  num_epochs: 10                  # Number of training epochs
  optimizer: "adam"               # Optimizer type (adam, sgd, adamw)
  weight_decay: 0.0001            # L2 regularization weight decay
  checkpoint_dir: "checkpoints"   # Directory to save model checkpoints
  save_every: 1                   # Save checkpoint every N epochs
```

#### Evaluation Configuration
```yaml
evaluation:
  checkpoint_path: "checkpoints/best_model.pth"  # Path to trained model
  batch_size: 64                  # Evaluation batch size (can be larger)
```

#### Device Configuration
```yaml
device: "cuda"                    # Options: cuda, cpu, mps (Apple Silicon)
```

### Creating Custom Configurations

1. Copy the default configuration:
```bash
cp ai-image-detector/configs/default_config.yaml ai-image-detector/configs/my_config.yaml
```

2. Edit the new configuration file with your desired parameters

3. Use the custom configuration:
```bash
python -m ai-image-detector.training --config ai-image-detector/configs/my_config.yaml
```

### Configuration Examples

**Fast Training (for testing):**
```yaml
model:
  backbone_type: "simple_cnn"
  pretrained: false

training:
  batch_size: 64
  num_epochs: 5
  learning_rate: 0.01
```

**High Accuracy (production):**
```yaml
model:
  backbone_type: "resnet50"
  pretrained: true
  freeze_backbone: false

training:
  batch_size: 16
  num_epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.0001
```

**Transfer Learning (fine-tuning):**
```yaml
model:
  backbone_type: "resnet18"
  pretrained: true
  freeze_backbone: true

training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 0.001
```

## Model Architectures

The system supports three backbone architectures, each with different trade-offs between speed and accuracy.

### SimpleCNN

Lightweight 4-layer convolutional neural network for baseline experiments and fast prototyping.

**Architecture:**
- Conv Layer 1: 3 → 64 channels, 3x3 kernel, ReLU, MaxPool
- Conv Layer 2: 64 → 128 channels, 3x3 kernel, ReLU, MaxPool
- Conv Layer 3: 128 → 256 channels, 3x3 kernel, ReLU, MaxPool
- Conv Layer 4: 256 → 512 channels, 3x3 kernel, ReLU, MaxPool
- Global Average Pooling
- Fully Connected: 512 → 1 (binary classification)

**Use Cases:**
- Quick baseline experiments
- Limited computational resources
- Fast iteration during development

**Expected Performance:**
- Training time: ~5-10 minutes per epoch (GPU)
- Accuracy: 75-85% on SynthBuster

### ResNet18

Standard ResNet-18 architecture with ImageNet pretrained weights for transfer learning.

**Architecture:**
- ResNet-18 backbone (pretrained on ImageNet)
- Custom classification head: 512 → 1
- Optional backbone freezing for faster training

**Use Cases:**
- Balanced speed and accuracy
- Transfer learning from ImageNet features
- Production deployments with moderate resources

**Expected Performance:**
- Training time: ~15-20 minutes per epoch (GPU)
- Accuracy: 85-92% on SynthBuster

### ResNet50

Deeper ResNet-50 architecture for maximum accuracy.

**Architecture:**
- ResNet-50 backbone (pretrained on ImageNet)
- Custom classification head: 2048 → 1
- Optional backbone freezing for faster training

**Use Cases:**
- Maximum accuracy requirements
- Research experiments
- Sufficient computational resources available

**Expected Performance:**
- Training time: ~25-35 minutes per epoch (GPU)
- Accuracy: 90-95% on SynthBuster

### Model Selection Guide

| Backbone | Speed | Accuracy | Parameters | Best For |
|----------|-------|----------|------------|----------|
| SimpleCNN | ⚡⚡⚡ | ⭐⭐ | ~2M | Prototyping, baselines |
| ResNet18 | ⚡⚡ | ⭐⭐⭐ | ~11M | Production, balanced |
| ResNet50 | ⚡ | ⭐⭐⭐⭐ | ~23M | Research, max accuracy |

## Evaluation Metrics

The system provides comprehensive evaluation metrics to assess model performance across different AI generation models.

### Overall Metrics

- **Accuracy**: Classification accuracy across all test images (correct predictions / total predictions)
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between real and AI-generated images

### Per-Generator Metrics

The evaluation pipeline computes separate accuracy and AUC scores for each AI generation model:
- **Stable Diffusion v2 (SD_v2)**: Text-to-image diffusion model
- **GLIDE**: OpenAI's guided diffusion model
- **Adobe Firefly**: Adobe's commercial AI image generator
- **DALL·E**: OpenAI's DALL·E model
- **Midjourney**: Popular commercial AI art generator
- **RAISE**: Real photographs (baseline for real image detection)

### Example Evaluation Report

```
========================================
EVALUATION REPORT
========================================
Total Samples: 10000

Overall Metrics:
  Accuracy: 92.50%
  AUC:      0.956

Per-Generator Metrics:
----------------------------------------
DALLE:
  Accuracy: 94.20%
  AUC:      0.968

Firefly:
  Accuracy: 91.80%
  AUC:      0.952

GLIDE:
  Accuracy: 89.50%
  AUC:      0.941

Midjourney:
  Accuracy: 93.10%
  AUC:      0.961

RAISE:
  Accuracy: 94.50%
  AUC:      0.972

SD_v2:
  Accuracy: 90.40%
  AUC:      0.947
----------------------------------------
```

### Interpreting Results

- **High accuracy on RAISE**: Indicates good detection of real images (low false positive rate)
- **Varying accuracy across generators**: Some AI models are easier to detect than others
- **AUC > 0.9**: Excellent discrimination ability
- **AUC 0.8-0.9**: Good discrimination ability
- **AUC < 0.8**: May need model improvements or more training data

## Examples

### Quick Start Example

Complete workflow from setup to evaluation:

```bash
# 1. Setup environment
cd ai-image-detector
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Prepare dataset (assuming dataset is downloaded)
# Verify dataset structure
ls data/synthbuster/real/RAISE/
ls data/synthbuster/fake/SD_v2/

# 3. Train a model with default settings
python -m ai-image-detector.training --config ai-image-detector/configs/default_config.yaml

# 4. Evaluate the trained model
python -m ai-image-detector.evaluation \
    --config ai-image-detector/configs/default_config.yaml \
    --checkpoint checkpoints/best_model.pth
```

### Custom Training Example

Create a custom configuration for high-accuracy training:

```bash
# Create custom config
cat > ai-image-detector/configs/high_accuracy.yaml << EOF
dataset:
  root_dir: "data/synthbuster"
  image_size: 256
  val_ratio: 0.2
  num_workers: 4

model:
  backbone_type: "resnet50"
  pretrained: true
  freeze_backbone: false

training:
  batch_size: 16
  learning_rate: 0.0001
  num_epochs: 20
  optimizer: "adam"
  weight_decay: 0.0001
  checkpoint_dir: "checkpoints/resnet50"
  save_every: 1

evaluation:
  checkpoint_path: "checkpoints/resnet50/best_model.pth"
  batch_size: 64

device: "cuda"
EOF

# Train with custom config
python -m ai-image-detector.training --config ai-image-detector/configs/high_accuracy.yaml
```

### Transfer Learning Example

Fine-tune a pretrained model with frozen backbone:

```bash
# Create transfer learning config
cat > ai-image-detector/configs/transfer_learning.yaml << EOF
dataset:
  root_dir: "data/synthbuster"
  image_size: 256
  val_ratio: 0.2
  num_workers: 4

model:
  backbone_type: "resnet18"
  pretrained: true
  freeze_backbone: true  # Freeze backbone, only train classifier head

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 10
  optimizer: "adam"
  weight_decay: 0.0001
  checkpoint_dir: "checkpoints/transfer"
  save_every: 1

device: "cuda"
EOF

# Train with frozen backbone
python -m ai-image-detector.training --config ai-image-detector/configs/transfer_learning.yaml
```

### Batch Evaluation Example

Evaluate multiple checkpoints:

```bash
# Evaluate different model checkpoints
for checkpoint in checkpoints/*.pth; do
    echo "Evaluating $checkpoint"
    python -m ai-image-detector.evaluation \
        --config ai-image-detector/configs/default_config.yaml \
        --checkpoint "$checkpoint"
done
```

### Python API Example

Use the model programmatically in your own code:

```python
import torch
from ai-image-detector.models.classifier import BinaryClassifier
from ai-image-detector.data.synthbuster_loader import SynthBusterDataset
from torch.utils.data import DataLoader

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BinaryClassifier(backbone_type='resnet18', pretrained=True)

checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load test data
test_dataset = SynthBusterDataset(root_dir='data/synthbuster')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Make predictions
predictions = []
with torch.no_grad():
    for images, labels, generator_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float().cpu().numpy()
        predictions.extend(preds)

print(f"Total predictions: {len(predictions)}")
```

### Single Image Inference Example

Classify a single image:

```python
import torch
from PIL import Image
from torchvision import transforms
from ai-image-detector.models.classifier import BinaryClassifier

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BinaryClassifier(backbone_type='resnet18', pretrained=True)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probability = output.item()
    prediction = "AI-generated" if probability > 0.5 else "Real"
    
print(f"Prediction: {prediction} (confidence: {probability:.2%})")
```

## Future Enhancements

This is a foundational implementation with planned extensions:
- Spectral/frequency-domain feature extraction
- Noise-based imprint analysis modules
- Multi-dataset support beyond SynthBuster
- Robustness testing (JPEG compression, Gaussian blur)
- Any-resolution image processing
- Attention mechanisms for spatial localization
- Advanced augmentation techniques (CutMix, MixUp)

TODO comments throughout the codebase mark locations for these enhancements.

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size in config: `training.batch_size: 16` or `8`
- Use a smaller model: `model.backbone_type: "simple_cnn"`
- Enable gradient checkpointing (requires code modification)
- Use CPU instead: `device: "cpu"` (slower but works)

#### Dataset Not Found

**Error:** `FileNotFoundError: Dataset directory not found`

**Solutions:**
- Verify dataset path in config matches actual location
- Check dataset structure matches expected format (see Dataset Setup)
- Ensure all required subdirectories exist (real/RAISE, fake/SD_v2, etc.)

#### Import Errors

**Error:** `ModuleNotFoundError: No module named 'ai-image-detector'`

**Solutions:**
- Run commands from project root directory (parent of ai-image-detector/)
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r ai-image-detector/requirements.txt`

#### Low Accuracy

**Issue:** Model accuracy is below expected performance

**Solutions:**
- Increase training epochs: `training.num_epochs: 20`
- Use pretrained backbone: `model.pretrained: true`
- Try a larger model: `model.backbone_type: "resnet50"`
- Verify dataset quality and balance
- Check for data loading issues (corrupted images)

#### Slow Training

**Issue:** Training takes too long

**Solutions:**
- Increase batch size if GPU memory allows: `training.batch_size: 64`
- Use more data loading workers: `dataset.num_workers: 8`
- Use a smaller model: `model.backbone_type: "simple_cnn"`
- Freeze backbone layers: `model.freeze_backbone: true`
- Enable GPU if available: `device: "cuda"`

### Getting Help

If you encounter issues not covered here:
1. Check the error message carefully
2. Verify your configuration file syntax (YAML format)
3. Ensure all dependencies are installed correctly
4. Review the test files for usage examples
5. Check that your dataset structure matches the expected format

## Dependencies

Core dependencies:
- **PyTorch** (>=2.0.0): Deep learning framework
- **torchvision** (>=0.15.0): Pre-trained models and image transforms
- **NumPy** (>=1.24.0): Numerical computing
- **Pillow** (>=10.0.0): Image loading and processing
- **PyYAML** (>=6.0): Configuration file parsing
- **scikit-learn** (>=1.3.0): Evaluation metrics (AUC, accuracy)

Optional:
- **matplotlib** (>=3.7.0): Visualization and plotting

## License

[To be determined]

## Citation

If you use this code in your research, please cite the SynthBuster benchmark:
```
[SynthBuster citation to be added]
```

## Contributing

Contributions are welcome! Please follow the existing code structure and add appropriate tests for new functionality.

## Contact

[Contact information to be added]
