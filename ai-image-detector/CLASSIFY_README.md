# Image Classification Tool with Grad-CAM

Single-file CLI tool for classifying images as real or AI-generated with visual explanations.

## Features

- Automatic model architecture detection from checkpoint
- Binary classification (Real vs Fake/AI-Generated)
- Grad-CAM visualization showing important regions
- Support for all model architectures (SimpleCNN, ResNet18, ResNet50)
- Support for enhanced features (spectral, noise, color, attention, etc.)

## Usage

Basic usage:
```bash
python classify_image.py --model <path_to_model.pth> --image <path_to_image.jpg>
```

### Examples

Classify with default settings:
```bash
python classify_image.py --model checkpoints/best_model.pth --image test_image.jpg
```

Specify image size and device:
```bash
python classify_image.py --model checkpoints/best_model.pth --image test_image.jpg --image-size 256 --device cuda
```

Save visualization to specific path:
```bash
python classify_image.py --model checkpoints/best_model.pth --image test_image.jpg --output results/analysis.png
```

Use CPU instead of GPU:
```bash
python classify_image.py --model checkpoints/best_model.pth --image test_image.jpg --device cpu
```

## Command-Line Arguments

- `--model` (required): Path to trained model checkpoint (.pth file)
- `--image` (required): Path to image to classify
- `--image-size` (optional): Image size for model input (default: 256)
- `--device` (optional): Device to use - cuda, cpu, or mps (default: cuda)
- `--output` (optional): Path to save visualization (auto-generated if not specified)

## Output

The tool provides:

1. **Classification Result**: REAL or FAKE (AI-Generated)
2. **Confidence Score**: Percentage confidence in the prediction
3. **Probability Breakdown**: Individual class probabilities
4. **Grad-CAM Visualization**: Three-panel image showing:
   - Original image
   - Heatmap of important regions
   - Overlay combining both

### Example Output

```
Using device: cuda
======================================================================
Loading model from checkpoints/best_model.pth...
Detected architecture: resnet50
Model loaded successfully!

Loading image: test_image.jpg
Original image size: (1024, 768)

Classifying image...
======================================================================
CLASSIFICATION RESULTS
======================================================================
Prediction: FAKE (AI-Generated)
Confidence: 95.23%

Probabilities:
  Real:  4.77%
  Fake:  95.23%

Generating Grad-CAM visualization...
Using layer: Conv2d
Visualization saved to test_image_gradcam.png
======================================================================
Done!
```

## Understanding Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which regions of the image were most important for the model's decision:

- **Red/Yellow regions**: High importance - these areas strongly influenced the classification
- **Blue/Purple regions**: Low importance - these areas had minimal impact
- **Green regions**: Medium importance

For AI-generated images, the model often focuses on:
- Texture inconsistencies
- Unnatural patterns or repetitions
- Artifacts from the generation process
- Frequency domain anomalies

For real images, the model typically focuses on:
- Natural textures and patterns
- Consistent lighting and shadows
- Realistic object boundaries

## Supported Model Types

The tool automatically detects and supports:

- **SimpleCNN**: Lightweight 4-layer CNN
- **ResNet18**: 18-layer residual network
- **ResNet50**: 50-layer residual network

Enhanced features (automatically detected):
- Spectral branch (frequency domain analysis)
- Noise imprint detection
- Chrominance features
- Attention mechanisms (CBAM, SE)
- Feature pyramid fusion
- Local patch classification

## Troubleshooting

### CUDA Out of Memory
If you encounter GPU memory errors, try:
```bash
python classify_image.py --model model.pth --image image.jpg --device cpu
```

### Model Loading Errors
Ensure the checkpoint file is valid and accessible:
```bash
# Check if file exists
ls checkpoints/best_model.pth

# Try with absolute path
python classify_image.py --model C:/full/path/to/model.pth --image image.jpg
```

### Image Loading Errors
Supported image formats: JPG, JPEG, PNG, BMP, TIFF

Ensure the image file is valid:
```bash
# Check if file exists
ls test_image.jpg
```

## Technical Details

### Architecture Detection
The tool automatically detects the model architecture by analyzing:
1. Classifier input dimensions (512 for ResNet18/SimpleCNN, 2048 for ResNet50)
2. Backbone layer structure
3. Feature branch presence

### Grad-CAM Implementation
- Uses the last convolutional layer of the backbone
- Computes gradients with respect to the predicted class
- Generates weighted activation maps
- Normalizes and overlays on original image

### Image Preprocessing
- Resizes to specified size (default: 256x256)
- Normalizes using ImageNet statistics
- Converts to RGB if needed

## Performance

Typical inference times (on NVIDIA GPU):
- SimpleCNN: ~10ms per image
- ResNet18: ~15ms per image
- ResNet50: ~25ms per image

With enhanced features, add ~5-10ms per enabled feature.

## Integration

The tool can be integrated into larger workflows:

```python
# Import and use programmatically
from classify_image import load_model, load_image, classify_image, GradCAM

device = torch.device('cuda')
model = load_model('checkpoints/best_model.pth', device)
image_tensor, image_original, _ = load_image('test.jpg')
prediction, confidence, probs = classify_image(model, image_tensor, device)

print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
print(f"Confidence: {confidence:.2%}")
```
