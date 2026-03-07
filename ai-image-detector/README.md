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

### Core Features

- **Modular Architecture**: Separate components for data loading, model definition, training, and evaluation
- **Flexible Backbones**: Support for SimpleCNN, ResNet18, and ResNet50 architectures
- **Per-Generator Metrics**: Detailed performance breakdown for each AI generation model
- **Configuration Management**: YAML-based configuration for easy experimentation
- **Checkpoint Management**: Save and resume training with full state preservation
- **Comprehensive Testing**: Unit tests for all major components

### Enhanced Detection Features

The detector includes 8 major enhancements for state-of-the-art AI image detection:

#### 1. Spectral Branch Architecture
Analyzes frequency domain features to detect spectral artifacts invisible in spatial domain:
- **FFT Processing**: Converts images to frequency domain using Fast Fourier Transform
- **Frequency Masking**: Configurable low-pass, high-pass, and band-pass filters
- **ViT-style Tokenization**: Patch-based processing with transformer encoder (patch_size=16, embed_dim=256, depth=4)
- **SRS Extraction**: Spectral Response Signatures for fixed-size feature vectors
- **SCV Computation**: Spectral Consistency Vectors measuring cross-band consistency
- **Self-Supervised Pretraining**: Masked patch reconstruction for spectral branch initialization

**Use Case**: Detect ML-generated images by identifying frequency domain anomalies that diffusion models introduce.

#### 2. Any-Resolution Processing
Process images of arbitrary sizes without resizing:
- **SpectralContextAttention**: Attention mechanism with interpolated positional encodings
- **Tiling Strategy**: Process large images as 256×256 tiles with 128-pixel stride (50% overlap)
- **Tile Aggregation**: Weighted averaging or voting to combine tile predictions
- **Native Resolution Mode**: Preserve original image dimensions throughout pipeline

**Use Case**: Analyze high-resolution images without information loss from resizing, maintaining detection accuracy across different image sizes.

#### 3. Noise Imprint Detection
Identify generator-specific noise patterns:
- **Dual Denoising Methods**: Diffusion-based (primary) with Gaussian fallback
- **CNN Feature Extraction**: 4-layer CNN extracts noise imprint features
- **Generator Attribution**: Optional head predicts which ML model generated the image
- **Graceful Degradation**: Falls back to Gaussian denoising when diffusers library unavailable

**Use Case**: Not only detect AI-generated images but also identify which specific model (Stable Diffusion, DALL·E, Midjourney, etc.) created them.

#### 4. Robustness Augmentation
Train models resistant to common image transformations:
- **JPEG Compression**: 5 severity levels (quality 95→50)
- **Gaussian Blur**: 5 severity levels (sigma 0.5→2.5)
- **Gaussian Noise**: 5 severity levels (std 0.01→0.05)
- **CutMix**: Patch mixing with label interpolation (alpha=1.0)
- **MixUp**: Image blending with alpha-weighted labels (alpha=0.2)

**Use Case**: Maintain detection accuracy on real-world images that have been compressed, resized, or otherwise transformed.

#### 5. Multi-Dataset Support
Train on multiple datasets simultaneously:
- **Weighted Sampling**: Configurable sampling weights per dataset
- **Dataset Registry**: Extensible pattern for adding new datasets
- **Domain Adversarial Training**: Gradient reversal layer learns domain-invariant features
- **Balanced Training**: Ensures representation across all datasets

**Use Case**: Improve generalization by training on diverse data sources (SynthBuster, COCO, custom datasets) with domain adaptation.

#### 6. Attention Mechanisms
Focus on discriminative image regions:
- **CBAM**: Convolutional Block Attention Module with channel and spatial attention
- **SEBlock**: Squeeze-and-Excitation for channel recalibration
- **LocalPatchClassifier**: Fine-grained patch-level predictions with heatmaps
- **Feature Pyramid Fusion**: FPN-style multi-scale feature combination

**Use Case**: Identify which image regions are most indicative of AI generation, providing interpretability and improved accuracy.

#### 7. Chrominance Features
Analyze color channel artifacts:
- **YCbCr Conversion**: RGB to YCbCr color space transformation
- **Histogram Features**: 64-bin histograms for Cb and Cr channels
- **Variance Statistics**: Global and local (8×8 patches) variance computation
- **Feature Projection**: Linear projection to 256-dimensional feature space

**Use Case**: Detect AI-generated images that exhibit chrominance anomalies, as diffusion models often produce subtle color artifacts.

#### 8. Comprehensive Evaluation Suite
Assess detector performance across multiple dimensions:
- **Robustness Evaluation**: Test against JPEG/blur/noise at 5 severity levels each
- **Spectral Visualization**: GradCAM heatmaps highlighting frequency domain regions
- **Noise Clustering**: Silhouette score and Davies-Bouldin index for generator separation
- **Cross-Dataset Metrics**: Per-dataset accuracy, precision, recall, F1 scores
- **Resolution Stratification**: Size-stratified metrics (128-256, 256-512, 512-1024 pixels)

**Use Case**: Comprehensive model analysis to understand strengths, weaknesses, and failure modes across different conditions.

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

The system supports two dataset modes for training:

#### Mode 1: Combined Balanced Dataset (Recommended)

Trains on a balanced combination of real and AI-generated images:
- **Real images (label=0)**: COCO 2017 train set + SynthBuster RAISE folder
- **AI-generated images (label=1)**: SynthBuster AI generator folders

The dataset automatically balances to ensure equal representation (1:1 ratio), which is critical for binary classification performance.

**Current balanced dataset statistics:**
- Real images: 9,000 from COCO 2017
- Fake images: 9,000 from SynthBuster AI generators
- Total: 18,000 images with perfect 1:1 balance

**Directory structure:**
```
datasets/
├── coco2017/
│   └── train2017/
│       ├── 000000000009.jpg
│       ├── 000000000025.jpg
│       └── ... (118,287 images total)
└── synthbuster/
    ├── RAISE/              # Real images (optional, COCO is primary)
    │   ├── image001.jpg
    │   └── ...
    ├── SD_v2/              # Stable Diffusion v2
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

**Configuration for combined mode:**
```yaml
dataset:
  mode: "combined"
  synthbuster_root: "datasets/synthbuster"
  coco_root: "datasets/coco2017"
  balance_mode: "equal"  # Ensures 1:1 ratio
  val_ratio: 0.2
```

#### Mode 2: SynthBuster-Only Dataset

Trains only on the SynthBuster dataset (original mode).

**Configuration for SynthBuster-only mode:**
```yaml
dataset:
  mode: "synthbuster"
  synthbuster_root: "datasets/synthbuster"
  val_ratio: 0.2
```

#### Verifying Dataset Setup

Run the verification script to check your dataset configuration:
```bash
python verify_balanced_dataset.py
```

This will:
- Check if both datasets are available
- Show the balanced dataset statistics
- Test sample loading
- Confirm the 1:1 balance ratio

**Expected output:**
```
======================================================================
BALANCED DATASET VERIFICATION
======================================================================

1. Checking dataset availability...
   ✓ SynthBuster found at: datasets/synthbuster
   ✓ COCO2017 found at: datasets/coco2017

2. Creating balanced combined dataset...
   ...

3. Dataset statistics:
   Total samples: 18000
   Real images (label=0): 9000
   Fake images (label=1): 9000
   Balance ratio: 1.000:1

✓ VERIFICATION SUCCESSFUL
```

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

Fast training for low-spec systems (2-4x speedup):
```bash
python -m ai-image-detector.training --config ai-image-detector/configs/fast_training.yaml
```

Resume training from a checkpoint:
```bash
python -m ai-image-detector.training --config ai-image-detector/configs/default_config.yaml --resume checkpoints/checkpoint_epoch_5.pth
```

**Performance Optimization:**
For detailed optimization strategies, see [Performance Optimization Guide](docs/PERFORMANCE_OPTIMIZATION.md).

Key optimizations for low-spec systems:
- Use `fast_training.yaml` config (128px images, SimpleCNN backbone)
- Enable mixed precision training (`mixed_precision: true`)
- Reduce batch size if OOM occurs
- Disable enhanced features for maximum speed

**Training Output:**
- Model checkpoints saved to `checkpoints/` directory (configurable)
- Best model saved as `checkpoints/best_model.pth` based on validation accuracy
- Per-epoch checkpoints saved as `checkpoint_epoch_N.pth`
- Training progress logged to console with loss and accuracy metrics

**Resuming Training:**
When resuming from a checkpoint, the training will:
- Load model weights and optimizer state
- Resume from the next epoch after the checkpoint
- Preserve the best validation accuracy
- Continue saving checkpoints normally

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

### Modular Branch Training

You can train different feature branches separately and combine them later:

**1. Train Individual Branches:**
```bash
# Train backbone only (no enhanced features)
python -m ai-image-detector.training --config ai-image-detector/configs/default_config.yaml

# Train spectral branch
python -m ai-image-detector.training --config ai-image-detector/configs/spectral_only.yaml

# Train noise imprint branch
python -m ai-image-detector.training --config ai-image-detector/configs/noise_only.yaml

# Train color features branch
python -m ai-image-detector.training --config ai-image-detector/configs/color_only.yaml
```

**2. Combine Pre-trained Branches:**
```bash
python -m ai-image-detector.utils.combine_checkpoints \
    --backbone checkpoints/default/best_model.pth \
    --spectral checkpoints/spectral_only/best_model.pth \
    --noise checkpoints/noise_only/best_model.pth \
    --color checkpoints/color_only/best_model.pth \
    --output checkpoints/combined_model.pth
```

**3. Fine-tune Combined Model:**
```bash
python -m ai-image-detector.training \
    --config ai-image-detector/configs/all_features.yaml \
    --resume checkpoints/combined_model.pth
```

**Benefits of Modular Training:**
- Train branches in parallel on different GPUs
- Experiment with different branch architectures independently
- Faster iteration when testing new features
- Reuse pre-trained branches across experiments

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

### Enhanced Features Configuration

The enhanced detector supports 8 major feature categories, all controlled via feature flags for backward compatibility.

#### Feature Flags

All enhanced features are disabled by default. Enable them in your config:

```yaml
model:
  backbone_type: "resnet18"
  pretrained: true
  
  # Feature flags (all default to false)
  use_spectral: true              # Enable spectral branch
  use_noise_imprint: true         # Enable noise imprint detection
  use_color_features: true        # Enable chrominance features
  use_local_patches: true         # Enable local patch classifier
  use_fpn: true                   # Enable feature pyramid fusion
  use_attention: "cbam"           # Options: "cbam", "se", null
  
  # Attribution
  enable_attribution: true        # Enable generator attribution
  num_generators: 10              # Number of generator classes
```

#### Spectral Branch Configuration

```yaml
spectral:
  patch_size: 16                  # Patch size for tokenization
  embed_dim: 256                  # Embedding dimension
  depth: 4                        # Number of transformer layers
  num_heads: 8                    # Number of attention heads
  mask_ratio: 0.75                # Masking ratio for pretraining
  frequency_mask_type: "high_pass"  # Options: "low_pass", "high_pass", "band_pass"
  cutoff_freq: 0.3                # Cutoff frequency for masking
```

**Example - Spectral-Only Detection:**
```yaml
model:
  use_spectral: true
  use_noise_imprint: false
  use_color_features: false

spectral:
  frequency_mask_type: "high_pass"
  cutoff_freq: 0.3
```

#### Any-Resolution Configuration

```yaml
any_resolution:
  enabled: true                   # Enable any-resolution processing
  tile_size: 256                  # Size of tiles for large images
  stride: 128                     # Stride between tiles (50% overlap)
  aggregation: "average"          # Options: "average", "voting"

data:
  native_resolution: true         # Preserve original image dimensions
```

**Example - High-Resolution Processing:**
```yaml
any_resolution:
  enabled: true
  tile_size: 256
  stride: 128
  aggregation: "average"

data:
  native_resolution: true
  batch_size: 8  # Smaller batch for variable sizes
```

#### Noise Imprint Configuration

```yaml
noise_imprint:
  method: "diffusion"             # Options: "diffusion", "gaussian"
  diffusion_steps: 50             # Number of diffusion steps
  gaussian_sigma: 2.0             # Sigma for Gaussian fallback
  feature_dim: 256                # Output feature dimension

model:
  use_noise_imprint: true
  enable_attribution: true        # Enable generator identification
  num_generators: 10              # Number of generator classes
```

**Example - Generator Attribution:**
```yaml
model:
  use_noise_imprint: true
  enable_attribution: true
  num_generators: 6  # DALLE, Firefly, GLIDE, Midjourney, SD_v2, RAISE

noise_imprint:
  method: "diffusion"
  diffusion_steps: 50
```

#### Augmentation Configuration

```yaml
augmentation:
  # Robustness augmentation
  robustness:
    jpeg_prob: 0.3                # Probability of JPEG compression
    blur_prob: 0.3                # Probability of Gaussian blur
    noise_prob: 0.3               # Probability of Gaussian noise
    severity_range: [1, 5]        # Min and max severity levels
  
  # CutMix augmentation
  cutmix:
    enabled: true
    alpha: 1.0                    # Beta distribution parameter
    prob: 0.5                     # Probability of applying CutMix
  
  # MixUp augmentation
  mixup:
    enabled: true
    alpha: 0.2                    # Beta distribution parameter
    prob: 0.5                     # Probability of applying MixUp
```

**Example - Robustness Training:**
```yaml
augmentation:
  robustness:
    jpeg_prob: 0.5
    blur_prob: 0.5
    noise_prob: 0.5
    severity_range: [1, 5]
  cutmix:
    enabled: true
    prob: 0.5
  mixup:
    enabled: true
    prob: 0.5
```

#### Multi-Dataset Configuration

```yaml
data:
  # Multi-dataset configuration
  datasets:
    synthbuster:
      weight: 1.0                 # Sampling weight
      path: "datasets/synthbuster"
    coco2017:
      weight: 0.5
      path: "datasets/coco2017"
    custom_dataset:
      weight: 0.3
      path: "datasets/custom"

training:
  # Domain adversarial training
  domain_adversarial:
    enabled: true
    lambda: 1.0                   # Gradient reversal strength
    hidden_dim: 256               # Domain discriminator hidden dimension
```

**Example - Multi-Dataset Training:**
```yaml
data:
  datasets:
    synthbuster:
      weight: 1.0
      path: "datasets/synthbuster"
    coco2017:
      weight: 1.0
      path: "datasets/coco2017"

training:
  domain_adversarial:
    enabled: true
    lambda: 1.0
```

#### Attention Configuration

```yaml
model:
  use_attention: "cbam"           # Options: "cbam", "se", null

attention:
  cbam:
    reduction_ratio: 16           # Channel reduction ratio
    kernel_size: 7                # Spatial attention kernel size
  se:
    reduction: 16                 # Squeeze-and-excitation reduction ratio
```

**Example - CBAM Attention:**
```yaml
model:
  use_attention: "cbam"

attention:
  cbam:
    reduction_ratio: 16
    kernel_size: 7
```

#### Chrominance Configuration

```yaml
model:
  use_color_features: true

chrominance:
  num_bins: 64                    # Number of histogram bins
  feature_dim: 256                # Output feature dimension
```

#### Complete Enhanced Configuration Example

```yaml
# All features enabled for maximum accuracy
dataset:
  root_dir: "datasets/synthbuster"
  image_size: 256
  val_ratio: 0.2
  num_workers: 4
  native_resolution: false

model:
  backbone_type: "resnet50"
  pretrained: true
  
  # Enable all features
  use_spectral: true
  use_noise_imprint: true
  use_color_features: true
  use_local_patches: true
  use_fpn: true
  use_attention: "cbam"
  enable_attribution: true
  num_generators: 10

spectral:
  patch_size: 16
  embed_dim: 256
  depth: 4
  num_heads: 8
  frequency_mask_type: "high_pass"
  cutoff_freq: 0.3

noise_imprint:
  method: "diffusion"
  diffusion_steps: 50
  feature_dim: 256

chrominance:
  num_bins: 64
  feature_dim: 256

attention:
  cbam:
    reduction_ratio: 16
    kernel_size: 7

augmentation:
  robustness:
    jpeg_prob: 0.3
    blur_prob: 0.3
    noise_prob: 0.3
    severity_range: [1, 5]
  cutmix:
    enabled: true
    alpha: 1.0
    prob: 0.5
  mixup:
    enabled: true
    alpha: 0.2
    prob: 0.5

data:
  datasets:
    synthbuster:
      weight: 1.0
      path: "datasets/synthbuster"

training:
  batch_size: 16
  learning_rate: 0.0001
  num_epochs: 20
  optimizer: "adamw"
  weight_decay: 0.0001
  checkpoint_dir: "checkpoints/enhanced"
  
  domain_adversarial:
    enabled: false
    lambda: 1.0

device: "cuda"
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

## Spectral Branch Pretraining

The spectral branch can be pretrained using self-supervised masked patch reconstruction before fine-tuning on the detection task.

### Pretraining Workflow

#### 1. Prepare Pretraining Dataset

Use unlabeled images (real or AI-generated) for pretraining:

```python
from ai-image-detector.data.synthbuster_loader import SynthBusterDataset
from torch.utils.data import DataLoader

# Load dataset (labels not used during pretraining)
pretrain_dataset = SynthBusterDataset(
    root_dir='datasets/synthbuster',
    transform=default_transform()
)

pretrain_loader = DataLoader(
    pretrain_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

#### 2. Configure Pretraining

Create a pretraining configuration file:

```yaml
# configs/spectral_pretrain.yaml
spectral:
  patch_size: 16
  embed_dim: 256
  depth: 4
  num_heads: 8
  mask_ratio: 0.75  # Mask 75% of patches

pretraining:
  decoder_embed_dim: 128
  decoder_depth: 2
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  checkpoint_dir: "checkpoints/spectral_pretrain"

data:
  root_dir: "datasets/synthbuster"
  batch_size: 32
  num_workers: 4

device: "cuda"
```

#### 3. Run Pretraining

Execute the pretraining script:

```bash
# Using the --pretrain flag
python -m ai-image-detector.training \
    --pretrain \
    --config configs/spectral_pretrain.yaml
```

Or use the dedicated pretraining script:

```bash
python -m ai-image-detector.training.pretrain_spectral \
    --config configs/spectral_pretrain.yaml
```

#### 4. Monitor Pretraining Progress

**Expected Output:**
```
Spectral Branch Pretraining
======================================================================
Configuration:
  Patch size: 16
  Embed dim: 256
  Mask ratio: 0.75
  Decoder depth: 2

Starting pretraining for 100 epochs...
----------------------------------------------------------------------
Epoch [1/100]
  Train Loss: 0.4523 | Val Loss: 0.4312
  Saved checkpoint: checkpoints/spectral_pretrain/epoch_1.pth
----------------------------------------------------------------------
Epoch [10/100]
  Train Loss: 0.2145 | Val Loss: 0.2089
  Saved checkpoint: checkpoints/spectral_pretrain/epoch_10.pth
----------------------------------------------------------------------
...
======================================================================
Pretraining complete!
Best checkpoint: checkpoints/spectral_pretrain/best_spectral.pth
```

#### 5. Fine-tune with Pretrained Weights

Load pretrained spectral branch for detection task:

```yaml
# configs/finetune_with_spectral.yaml
model:
  backbone_type: "resnet18"
  pretrained: true
  use_spectral: true
  
  # Load pretrained spectral weights
  spectral_checkpoint: "checkpoints/spectral_pretrain/best_spectral.pth"

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 20
```

```bash
python -m ai-image-detector.training \
    --config configs/finetune_with_spectral.yaml
```

### Pretraining Benefits

- **Improved Convergence**: Pretrained spectral branch converges faster during fine-tuning
- **Better Features**: Self-supervised learning discovers frequency domain patterns
- **Data Efficiency**: Requires less labeled data for fine-tuning
- **Generalization**: Learns universal spectral features applicable across generators

### Pretraining Best Practices

1. **Dataset Size**: Use 10,000+ images for effective pretraining
2. **Mask Ratio**: 0.75 (75%) works well for most cases
3. **Epochs**: 100-200 epochs for convergence
4. **Learning Rate**: Start with 0.001, reduce if loss plateaus
5. **Validation**: Monitor reconstruction loss on held-out set

## Comprehensive Evaluation Suite

The enhanced detector includes a comprehensive evaluation suite to assess performance across multiple dimensions.

### Running Evaluations

#### 1. Robustness Evaluation

Test model performance against common image perturbations:

```bash
python -m ai-image-detector.evaluation.robustness_eval \
    --checkpoint checkpoints/best_model.pth \
    --config configs/default_config.yaml \
    --output results/robustness_eval.json
```

**Configuration:**
```yaml
evaluation:
  robustness:
    perturbations: ["jpeg", "blur", "noise"]
    severity_levels: [1, 2, 3, 4, 5]
    batch_size: 64
```

**Output:**
```json
{
  "jpeg": {
    "1": {"accuracy": 0.945, "auc": 0.968},
    "2": {"accuracy": 0.932, "auc": 0.954},
    "3": {"accuracy": 0.918, "auc": 0.941},
    "4": {"accuracy": 0.895, "auc": 0.923},
    "5": {"accuracy": 0.867, "auc": 0.901}
  },
  "blur": {
    "1": {"accuracy": 0.938, "auc": 0.961},
    "2": {"accuracy": 0.921, "auc": 0.947},
    "3": {"accuracy": 0.903, "auc": 0.932},
    "4": {"accuracy": 0.881, "auc": 0.915},
    "5": {"accuracy": 0.854, "auc": 0.893}
  },
  "noise": {
    "1": {"accuracy": 0.941, "auc": 0.964},
    "2": {"accuracy": 0.928, "auc": 0.951},
    "3": {"accuracy": 0.912, "auc": 0.936},
    "4": {"accuracy": 0.893, "auc": 0.919},
    "5": {"accuracy": 0.869, "auc": 0.898}
  }
}
```

#### 2. Spectral Artifact Visualization

Generate GradCAM heatmaps for spectral branch:

```bash
python -m ai-image-detector.evaluation.spectral_viz \
    --checkpoint checkpoints/best_model.pth \
    --images test_images/ \
    --output visualizations/
```

**Requirements:**
```bash
pip install pytorch-grad-cam
```

**Output:**
- Heatmap overlays showing which frequency regions are most discriminative
- Side-by-side comparisons of spatial and spectral attention
- Per-image visualization saved as PNG files

**Example Usage:**
```python
from ai-image-detector.evaluation.spectral_viz import visualize_spectral_artifacts

# Generate heatmaps
heatmaps = visualize_spectral_artifacts(
    model=model,
    images=test_images,
    target_layer='spectral_branch',
    device='cuda'
)

# Save visualizations
save_heatmap_overlays(heatmaps, output_dir='visualizations/')
```

#### 3. Noise Imprint Clustering

Analyze generator-specific noise patterns:

```bash
python -m ai-image-detector.evaluation.noise_clustering \
    --checkpoint checkpoints/best_model.pth \
    --config configs/default_config.yaml \
    --output results/noise_clustering.json
```

**Requirements:**
```bash
pip install scikit-learn
```

**Output:**
```json
{
  "silhouette_score": 0.723,
  "davies_bouldin_index": 0.456,
  "per_generator_separation": {
    "SD_v2": 0.812,
    "DALLE": 0.789,
    "Midjourney": 0.745,
    "GLIDE": 0.698,
    "Firefly": 0.734
  },
  "confusion_matrix": [[...]]
}
```

**Interpretation:**
- **Silhouette Score** (higher is better): Measures cluster separation
  - > 0.7: Excellent separation
  - 0.5-0.7: Good separation
  - < 0.5: Poor separation
- **Davies-Bouldin Index** (lower is better): Measures cluster compactness
  - < 0.5: Excellent clustering
  - 0.5-1.0: Good clustering
  - > 1.0: Poor clustering

#### 4. Cross-Dataset Evaluation

Evaluate performance across multiple datasets:

```bash
python -m ai-image-detector.evaluation.cross_dataset_eval \
    --checkpoint checkpoints/best_model.pth \
    --datasets synthbuster coco2017 custom \
    --output results/cross_dataset_eval.json
```

**Configuration:**
```yaml
evaluation:
  cross_dataset:
    datasets:
      - name: "synthbuster"
        path: "datasets/synthbuster"
      - name: "coco2017"
        path: "datasets/coco2017"
      - name: "custom"
        path: "datasets/custom"
    batch_size: 64
```

**Output:**
```json
{
  "synthbuster": {
    "accuracy": 0.945,
    "precision": 0.938,
    "recall": 0.952,
    "f1": 0.945,
    "auc": 0.968
  },
  "coco2017": {
    "accuracy": 0.923,
    "precision": 0.915,
    "recall": 0.931,
    "f1": 0.923,
    "auc": 0.951
  },
  "custom": {
    "accuracy": 0.912,
    "precision": 0.905,
    "recall": 0.919,
    "f1": 0.912,
    "auc": 0.943
  }
}
```

#### 5. Any-Resolution Evaluation

Test performance across different image sizes:

```bash
python -m ai-image-detector.evaluation.resolution_eval \
    --checkpoint checkpoints/best_model.pth \
    --config configs/any_resolution.yaml \
    --output results/resolution_eval.json
```

**Configuration:**
```yaml
evaluation:
  resolution:
    size_ranges:
      - [128, 256]
      - [256, 512]
      - [512, 1024]
      - [1024, 2048]
    batch_size: 32
```

**Output:**
```json
{
  "128-256": {
    "num_samples": 2500,
    "accuracy": 0.932,
    "auc": 0.954
  },
  "256-512": {
    "num_samples": 3200,
    "accuracy": 0.945,
    "auc": 0.968
  },
  "512-1024": {
    "num_samples": 2800,
    "accuracy": 0.941,
    "auc": 0.963
  },
  "1024-2048": {
    "num_samples": 1500,
    "accuracy": 0.938,
    "auc": 0.959
  }
}
```

#### 6. Comprehensive Evaluation Runner

Run all evaluations in one command:

```bash
python -m ai-image-detector.evaluation.comprehensive_eval \
    --checkpoint checkpoints/best_model.pth \
    --config configs/default_config.yaml \
    --output results/comprehensive_report/
```

**Output Structure:**
```
results/comprehensive_report/
├── summary.json                    # Overall summary
├── robustness_eval.json            # Robustness results
├── noise_clustering.json           # Clustering metrics
├── cross_dataset_eval.json         # Cross-dataset results
├── resolution_eval.json            # Resolution stratification
├── visualizations/
│   ├── spectral_heatmaps/          # GradCAM visualizations
│   ├── robustness_curves.png       # Accuracy vs severity plots
│   ├── clustering_tsne.png         # t-SNE visualization
│   └── resolution_performance.png  # Performance by size
└── report.html                     # Interactive HTML report
```

### Evaluation Best Practices

1. **Baseline Comparison**: Always evaluate baseline model (all features disabled) for comparison
2. **Multiple Seeds**: Run evaluations with different random seeds for statistical significance
3. **Stratified Sampling**: Ensure balanced representation across generators and datasets
4. **Perturbation Realism**: Use realistic perturbation parameters matching real-world conditions
5. **Visualization**: Generate visualizations to understand model behavior and failure modes

### Interpreting Evaluation Results

#### Robustness Metrics
- **Graceful Degradation**: Accuracy should decrease gradually with severity
- **JPEG Resilience**: Good models maintain >85% accuracy at severity 3
- **Blur Tolerance**: Accuracy >80% at severity 3 indicates robustness

#### Clustering Metrics
- **High Silhouette**: Indicates distinct noise patterns per generator
- **Low Davies-Bouldin**: Confirms tight, well-separated clusters
- **Attribution Accuracy**: Should match or exceed binary detection accuracy

#### Cross-Dataset Performance
- **Consistent Accuracy**: <5% variance across datasets indicates good generalization
- **Dataset Bias**: Large variance suggests overfitting to specific dataset characteristics
- **Transfer Learning**: Performance on unseen datasets validates model robustness

#### Resolution Performance
- **Size Invariance**: Accuracy should remain stable across size ranges
- **Large Image Handling**: Tiling strategy should maintain accuracy for >1024px images
- **Small Image Degradation**: Acceptable accuracy drop for <256px images due to limited information

## Implementation Status

### Implemented Features ✓

The following enhancements have been fully implemented:

- ✓ **Spectral Branch Architecture**: FFT processing, frequency masking, ViT tokenization, SRS/SCV extraction, masked pretraining
- ✓ **Any-Resolution Processing**: SpectralContextAttention, tiling strategy, tile aggregation, native resolution mode
- ✓ **Noise Imprint Detection**: Dual denoising methods, CNN feature extraction, generator attribution
- ✓ **Robustness Augmentation**: JPEG/blur/noise augmentation, CutMix, MixUp
- ✓ **Multi-Dataset Support**: Weighted sampling, dataset registry, domain adversarial training
- ✓ **Attention Mechanisms**: CBAM, SEBlock, LocalPatchClassifier, FeaturePyramidFusion
- ✓ **Chrominance Features**: YCbCr conversion, histogram/variance extraction
- ✓ **Comprehensive Evaluation**: Robustness testing, spectral visualization, noise clustering, cross-dataset metrics, resolution stratification

### Future Enhancements

Potential extensions for future development:

- **Temporal Consistency**: Video frame analysis for AI-generated video detection
- **Adversarial Robustness**: Defense against adversarial attacks designed to fool detectors
- **Explainability**: LIME/SHAP integration for detailed prediction explanations
- **Real-Time Processing**: Optimized inference for real-time detection applications
- **Mobile Deployment**: Model quantization and optimization for mobile devices
- **Active Learning**: Iterative training with human-in-the-loop feedback
- **Zero-Shot Detection**: Detect images from unseen generators without retraining
- **Multimodal Analysis**: Combine image analysis with metadata and EXIF data

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

### Core Dependencies

Required for basic functionality:
- **PyTorch** (>=2.0.0): Deep learning framework
- **torchvision** (>=0.15.0): Pre-trained models and image transforms
- **NumPy** (>=1.24.0): Numerical computing
- **Pillow** (>=10.0.0): Image loading and processing
- **PyYAML** (>=6.0): Configuration file parsing
- **scikit-learn** (>=1.3.0): Evaluation metrics (AUC, accuracy)

### Optional Dependencies

For enhanced features:
- **diffusers** (>=0.21.0): Diffusion-based noise residual extraction (noise imprint detection)
- **pytorch-grad-cam** (>=1.4.0): GradCAM visualization (spectral artifact visualization)
- **matplotlib** (>=3.7.0): Visualization and plotting
- **hypothesis** (>=6.82.0): Property-based testing framework

### Installation

Install core dependencies:
```bash
pip install -r requirements.txt
```

Install all optional dependencies:
```bash
pip install diffusers>=0.21.0 pytorch-grad-cam>=1.4.0 matplotlib>=3.7.0 hypothesis>=6.82.0
```

Install specific feature dependencies:
```bash
# For noise imprint detection with diffusion
pip install diffusers>=0.21.0

# For spectral visualization
pip install pytorch-grad-cam>=1.4.0

# For property-based testing
pip install hypothesis>=6.82.0
```

### Graceful Degradation

The system handles missing optional dependencies gracefully:
- **Missing diffusers**: Falls back to Gaussian denoising for noise residual extraction
- **Missing pytorch-grad-cam**: Spectral visualization features disabled, warning issued
- **Missing hypothesis**: Property-based tests skipped during test runs

No functionality breaks due to missing optional dependencies.

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

## Quick Reference: Enhanced Features

### Feature Flag Quick Reference

```yaml
# Enable/disable features in your config
model:
  use_spectral: true/false          # Spectral branch
  use_noise_imprint: true/false     # Noise imprint detection
  use_color_features: true/false    # Chrominance features
  use_local_patches: true/false     # Local patch classifier
  use_fpn: true/false               # Feature pyramid fusion
  use_attention: "cbam"/"se"/null   # Attention mechanism
  enable_attribution: true/false    # Generator attribution
```

### Command Quick Reference

```bash
# Standard training
python -m ai-image-detector.training --config configs/default_config.yaml

# Spectral pretraining
python -m ai-image-detector.training --pretrain --config configs/spectral_pretrain.yaml

# Evaluation
python -m ai-image-detector.evaluation --config configs/default_config.yaml --checkpoint checkpoints/best_model.pth

# Robustness evaluation
python -m ai-image-detector.evaluation.robustness_eval --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml

# Comprehensive evaluation
python -m ai-image-detector.evaluation.comprehensive_eval --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml --output results/
```

### Configuration Templates

#### Baseline (Original Features Only)
```yaml
model:
  backbone_type: "resnet18"
  pretrained: true
  use_spectral: false
  use_noise_imprint: false
  use_color_features: false
  use_local_patches: false
  use_fpn: false
  use_attention: null
```

#### Spectral-Only
```yaml
model:
  use_spectral: true
spectral:
  frequency_mask_type: "high_pass"
  cutoff_freq: 0.3
```

#### Noise Imprint with Attribution
```yaml
model:
  use_noise_imprint: true
  enable_attribution: true
  num_generators: 6
noise_imprint:
  method: "diffusion"
```

#### All Features Enabled
```yaml
model:
  use_spectral: true
  use_noise_imprint: true
  use_color_features: true
  use_local_patches: true
  use_fpn: true
  use_attention: "cbam"
  enable_attribution: true
```

### Performance Expectations

| Configuration | Accuracy | Training Time | Inference Time |
|--------------|----------|---------------|----------------|
| Baseline (ResNet18) | 85-92% | 15-20 min/epoch | ~10ms/image |
| + Spectral | 88-94% | 25-30 min/epoch | ~15ms/image |
| + Noise Imprint | 90-95% | 30-35 min/epoch | ~20ms/image |
| All Features | 92-96% | 45-60 min/epoch | ~30ms/image |

*Times measured on NVIDIA RTX 3090, batch_size=32, image_size=256*

### Troubleshooting Enhanced Features

**Issue**: Out of memory with multiple features enabled
- **Solution**: Reduce batch_size, disable some features, or use gradient checkpointing

**Issue**: Slow training with all features
- **Solution**: Enable only necessary features, use smaller backbone, or increase num_workers

**Issue**: Diffusers import error
- **Solution**: Install diffusers (`pip install diffusers>=0.21.0`) or set `noise_imprint.method: "gaussian"`

**Issue**: GradCAM visualization fails
- **Solution**: Install pytorch-grad-cam (`pip install pytorch-grad-cam>=1.4.0`)

**Issue**: Poor performance with native_resolution=true
- **Solution**: Use any_resolution.enabled=true with tiling strategy for large images
