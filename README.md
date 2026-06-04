# TruthLens AI — Deep Learning Based Detection of AI Generated Images

> B.Tech Final Year Project · ResNet50 · Grad-CAM · FastAPI · React 19

TruthLens AI is a full-stack system that classifies images as real or AI-generated using a CNN-based deep learning model with Grad-CAM explainability. Upload any image through the web interface and receive a prediction, confidence score, and visual heatmap showing which regions influenced the decision.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Backend — AI Image Detector](#backend--ai-image-detector)
   - [Model Architecture](#model-architecture)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Single Image Classification (CLI)](#single-image-classification-cli)
   - [Configuration Reference](#configuration-reference)
5. [Frontend — TruthLens Web App](#frontend--truthlens-web-app)
   - [API Server](#api-server)
   - [Running the Frontend](#running-the-frontend)
6. [Dataset Setup](#dataset-setup)
7. [Device & Hardware](#device--hardware)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

| Component | Technology |
|---|---|
| Detection model | ResNet50 (pretrained ImageNet, fine-tuned) |
| Explainability | Grad-CAM on `layer4[-1].conv2` |
| Training framework | PyTorch 2.x, AdamW, mixed precision |
| API server | FastAPI + Uvicorn |
| Frontend | React 19 + TypeScript + Vite 8 + Tailwind CSS 4 |
| Animations | Framer Motion |
| Dataset (real) | COCO 2017 — 10,000 images |
| Dataset (AI) | Synthbuster — 9,800 images (SD v2, DALL-E, Midjourney, GLIDE, Firefly) |
| Inference device | Auto-detected: CUDA → MPS → CPU |

### How It Works

1. **Upload** — Any JPEG, PNG, or WebP image.
2. **Preprocess** — Resized to 256 × 256, normalized with ImageNet statistics.
3. **Feature extraction** — ResNet50 backbone produces multi-scale feature maps.
4. **Classification** — Global average pooling → Dense → Sigmoid → P(fake).
5. **Grad-CAM** — Gradient-weighted activation map on `layer4[-1].conv2`.
6. **Result** — Verdict (Real / AI Generated) + confidence + heatmap overlay.

---

## Project Structure

```
AR-MAD/
├── ai-image-detector/          # Core ML package
│   ├── classify_image.py       # CLI inference tool with Grad-CAM
│   ├── models/
│   │   ├── classifier.py       # BinaryClassifier wrapper
│   │   ├── backbones.py        # SimpleCNN / ResNet18 / ResNet50
│   │   ├── spectral/           # Spectral artifact branch (optional)
│   │   ├── noise/              # Noise imprint branch (optional)
│   │   ├── color/              # Chrominance branch (optional)
│   │   ├── attention/          # CBAM / SE attention (optional)
│   │   ├── fusion/             # Feature pyramid fusion (optional)
│   │   └── resolution/         # Any-resolution processing (optional)
│   ├── data/
│   │   ├── synthbuster_loader.py   # Synthbuster dataset loader
│   │   ├── coco_loader.py          # COCO 2017 loader (real images)
│   │   ├── combined_loader.py      # Balanced combined dataset
│   │   └── augmentation/           # CutMix, MixUp, robustness augmentation
│   ├── training/
│   │   └── __main__.py         # Training entry point
│   ├── evaluation/
│   │   └── __main__.py         # Evaluation entry point
│   ├── configs/
│   │   ├── default_config.yaml
│   │   ├── enhanced_config.yaml
│   │   ├── fast_training.yaml
│   │   └── validator.py
│   └── utils/
│       ├── config_loader.py
│       └── check_corrupted_images.py
├── frontend/
│   ├── api_server.py           # FastAPI server (wraps classify_image.py)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── HomePage.tsx    # Landing page + inline analyzer
│   │   │   └── AboutPage.tsx   # Mission, research, tech stack
│   │   └── components/
│   │       ├── layout/Navbar.tsx
│   │       └── ui/             # ImageSlider, CircularGauge
│   └── package.json
├── checkpoints/
│   └── checkpoint_epoch_25.pth # Trained model (442 MB)
├── datasets/                   # Not included in repo (download separately)
│   ├── coco2017/train2017/
│   └── synthbuster/
├── requirements.txt            # All Python dependencies (unified)
└── README.md                   # This file
```

---

## Quick Start

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the API server

```bash
source .venv/bin/activate
python frontend/api_server.py
# Model loads from checkpoints/checkpoint_epoch_25.pth
# Server starts at http://localhost:8000
```

> First startup takes ~30 seconds to load the 442 MB model.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### 4. Classify a single image (CLI)

```bash
source .venv/bin/activate
python ai-image-detector/classify_image.py \
  --model checkpoints/checkpoint_epoch_25.pth \
  --image path/to/image.jpg \
  --output results/analysis.png
```

---

## Backend — AI Image Detector

### Model Architecture

The detection model is a `BinaryClassifier` built around a ResNet50 backbone:

```
Input Image (256×256×3)
    ↓
ResNet50 Backbone (pretrained ImageNet)
    ↓
Feature Maps (multi-scale)
    ↓
Global Average Pooling
    ↓
Classification Head (Dense → Dropout(0.5) → Sigmoid)
    ↓
P(fake) ∈ [0, 1]       threshold 0.5 → Real / AI Generated
    ↓
Grad-CAM on layer4[-1].conv2
```

**Optional feature branches** (all disabled by default, controlled via config flags):

| Flag | Branch | Description |
|---|---|---|
| `use_spectral` | Spectral | FFT + ViT-style tokenization for frequency artifacts |
| `use_noise_imprint` | Noise | Generator-specific noise pattern extraction |
| `use_color_features` | Chrominance | YCbCr histogram and variance features |
| `use_attention` | Attention | CBAM or SE channel/spatial attention |
| `use_fpn` | FPN | Feature Pyramid Network for multi-scale fusion |
| `use_local_patches` | Local | Patch-level fine-grained classifier |
| `enable_attribution` | Attribution | Identifies which generative model created the image |

Architecture is auto-detected from the checkpoint's state dict when loading — no config needed for inference.

### Training

```bash
# Default (balanced COCO + Synthbuster dataset)
python -m ai-image-detector.training \
  --config ai-image-detector/configs/default_config.yaml

# Fast training for low-spec systems
python -m ai-image-detector.training \
  --config ai-image-detector/configs/fast_training.yaml

# Resume from checkpoint
python -m ai-image-detector.training \
  --config ai-image-detector/configs/default_config.yaml \
  --resume checkpoints/checkpoint_epoch_5.pth
```

**Training config summary** (see `configs/default_config.yaml` for full details):

| Parameter | Value |
|---|---|
| Backbone | ResNet50 (pretrained) |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Epochs | 60 (checkpoint at 25 used) |
| Mixed precision | Yes (FP16) |
| Augmentation | Horizontal flip, random crop, blur, JPEG |

Checkpoints are saved every 5 epochs to `checkpoints/`. Best model saved as `best_model.pth`.

### Evaluation

```bash
python -m ai-image-detector.evaluation \
  --config ai-image-detector/configs/default_config.yaml \
  --checkpoint checkpoints/checkpoint_epoch_25.pth
```

Reports overall accuracy, AUC-ROC, and per-generator breakdown (SD v2, DALL-E, Midjourney, GLIDE, Firefly).

### Single Image Classification (CLI)

```bash
python ai-image-detector/classify_image.py \
  --model checkpoints/checkpoint_epoch_25.pth \
  --image test_image.jpg \
  [--image-size 256] \
  [--device cuda|mps|cpu] \
  [--output results/analysis.png] \
  [--no-display]
```

**Output:** Console prediction + confidence + Grad-CAM visualization (three-panel: original, heatmap, overlay).

The script can also be imported programmatically:

```python
from ai-image-detector.classify_image import load_model, load_image, classify_image, GradCAM

device = torch.device('cuda')
model  = load_model('checkpoints/checkpoint_epoch_25.pth', device)
tensor, original, size = load_image('test.jpg')
prediction, confidence, probs = classify_image(model, tensor, device)
```

### Configuration Reference

Key settings in `ai-image-detector/configs/default_config.yaml`:

```yaml
dataset:
  mode: "combined"              # "combined" | "synthbuster"
  synthbuster_root: "datasets/synthbuster"
  coco_root: "datasets/coco2017"
  image_size: 256
  val_ratio: 0.2
  balance_mode: "equal"         # 1:1 real/fake ratio

model:
  backbone_type: "resnet50"     # "simple_cnn" | "resnet18" | "resnet50"
  pretrained: true
  # All enhanced feature flags default to false
  use_spectral: false
  use_noise_imprint: false
  use_color_features: false
  use_attention: null           # null | "cbam" | "se"
  use_fpn: false
  use_local_patches: false
  enable_attribution: false

training:
  batch_size: 8
  learning_rate: 0.00001
  num_epochs: 60
  optimizer: "adamw"
  weight_decay: 0.0001
  mixed_precision: true
  checkpoint_dir: "checkpoints"
  save_every: 5

evaluation:
  checkpoint_path: "checkpoints/best_model.pth"

device: "auto"                  # "auto" | "cuda" | "mps" | "cpu"
```

---

## Frontend — TruthLens Web App

### API Server

`frontend/api_server.py` wraps `classify_image.py` without modifying the core ML code.

```
POST /api/analyze   multipart/form-data { file: <image> }
GET  /api/health    → { status, model_loaded, device }
```

**Response from `/api/analyze`:**

```json
{
  "prediction": "REAL" | "AI_GENERATED",
  "confidence": 94.2,
  "probabilities": { "real": 94.2, "fake": 5.8 },
  "original_image": "<base64 PNG>",
  "gradcam_overlay": "<base64 PNG>",
  "processing_time": 1.23,
  "model_info": { "checkpoint": "...", "device": "mps", "image_size": 256 },
  "timestamp": "2026-06-04T14:30:00"
}
```

Start the server:

```bash
source .venv/bin/activate
python frontend/api_server.py
```

### Running the Frontend

```bash
cd frontend
npm install       # first time only
npm run dev       # development server → http://localhost:5173
npm run build     # production bundle → dist/
```

The Vite dev server proxies `/api/*` to `http://localhost:8000` automatically.

**Pages:**
- **Detect** (`/`) — Hero with inline analyzer. Upload → analyze → result with Grad-CAM slider.
- **About** (`/about`) — Mission, problem statement, research approach, tech stack, vision.

---

## Dataset Setup

The trained checkpoint at `checkpoints/checkpoint_epoch_25.pth` is already available. Datasets are only needed for retraining.

```
datasets/
├── coco2017/
│   └── train2017/          # 118,287 real images (download from cocodataset.org)
└── synthbuster/
    ├── RAISE/              # Real images (optional, COCO preferred)
    ├── SD_v2/              # Stable Diffusion v2
    ├── GLIDE/
    ├── Firefly/
    ├── DALLE/
    └── Midjourney/
```

Verify dataset balance before training:

```bash
python verify_balanced_dataset.py
# Expected: 9,000 real + 9,800 AI images → balanced to 9,000 : 9,000
```

**Corrupted images:** If training crashes with `OSError: decoder error -2`, run:

```bash
python -m ai-image-detector.utils.check_corrupted_images datasets/synthbuster
# Add --remove to delete corrupted files automatically
```

---

## Device & Hardware

The system auto-detects the best available device at startup:

```
CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
```

Override in `default_config.yaml`:

```yaml
device: "cuda"   # or "mps" or "cpu"
```

**GPU memory guidance (training):**

| GPU VRAM | Recommended batch size | Backbone |
|---|---|---|
| 4 GB | 8–16 | resnet18 |
| 8 GB | 16–32 | resnet50 |
| 16 GB+ | 32–64 | resnet50 + features |

For CPU training, use `fast_training.yaml` and expect ~3–5× longer training time.

---

## Troubleshooting

**Model doesn't load / timeout on startup**
The checkpoint is 442 MB. Allow 30–60 seconds. On CPU this may take longer.

**`CUDA out of memory`**
Reduce `batch_size` in config. For 4 GB VRAM use `batch_size: 8`.

**Frontend shows "Disconnected" in settings**
The API server is not running. Start it with `python frontend/api_server.py`.

**`OSError: decoder error -2` during training**
A corrupted TIFF file is in the dataset. Remove it:
```bash
python -m ai-image-detector.utils.check_corrupted_images datasets/synthbuster --remove
```

**PyTorch CUDA not available on Python 3.14**
PyTorch CUDA builds require Python ≤ 3.12. Create a new venv with Python 3.12:
```bash
py -3.12 -m venv .venv312
.venv312/Scripts/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**`strict=False` warning on model load**
Expected for checkpoints trained with different config flags. The model still loads correctly.

---

## Running Tests

```bash
# All tests
python -m pytest ai-image-detector/ -v

# Specific modules
python -m pytest ai-image-detector/data/test_synthbuster_loader.py -v
python -m pytest ai-image-detector/data/test_combined_loader.py -v
python -m pytest ai-image-detector/utils/test_config_loader.py -v
```

---

*TruthLens AI — Deep Learning Based Detection of AI Generated Images*
