#!/usr/bin/env python
"""
Single-file CLI tool for AI-generated image detection with Grad-CAM visualization.

Usage:
    python classify_image.py --model <path_to_model.pth> --image <path_to_image.jpg>

Example:
    python classify_image.py --model checkpoints/best_model.pth --image test_image.jpg
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Grad-CAM implementation for visualizing important regions."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap."""
        device = input_image.device
        
        # Forward pass
        output = self.model(input_image)
        
        # Handle tuple output (with attribution)
        if isinstance(output, tuple):
            output = output[0]
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Reconstruct model architecture from state dict
    model = reconstruct_model_from_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def reconstruct_model_from_state_dict(state_dict):
    """Reconstruct model architecture from state dict keys."""
    # Import model classes
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.classifier import BinaryClassifier
    
    # Detect backbone type from state dict keys and classifier dimensions
    # Check classifier input dimension to distinguish between architectures
    classifier_weight_key = 'classifier.classifier.0.weight'
    if classifier_weight_key in state_dict:
        classifier_input_dim = state_dict[classifier_weight_key].shape[1]
        
        # SimpleCNN: 512, ResNet18: 512, ResNet50: 2048
        if classifier_input_dim == 2048:
            backbone_type = 'resnet50'
        elif any('backbone.7' in k for k in state_dict.keys()):
            # ResNet50 has layers 4-7 (layer1-layer4), ResNet18 has 4-7 too but different structure
            # Check for layer depth - ResNet50 has more blocks
            if any('backbone.6.5' in k for k in state_dict.keys()):
                backbone_type = 'resnet50'
            else:
                backbone_type = 'resnet18'
        elif any('backbone.layer4' in k for k in state_dict.keys()) or any('backbone.7' in k for k in state_dict.keys()):
            # ResNet architecture (numbered layers)
            backbone_type = 'resnet18'
        elif any('backbone.conv1' in k for k in state_dict.keys()):
            # SimpleCNN (named conv layers)
            backbone_type = 'simple_cnn'
        else:
            # Fallback: try to detect from structure
            backbone_type = 'resnet18'  # Default to resnet18
    else:
        # No classifier found, detect from backbone structure
        if any('backbone.conv1' in k for k in state_dict.keys()):
            backbone_type = 'simple_cnn'
        elif any('backbone.6.5' in k for k in state_dict.keys()):
            backbone_type = 'resnet50'
        else:
            backbone_type = 'resnet18'
    
    # Detect feature flags from state dict
    use_spectral = any('spectral_branch' in k for k in state_dict.keys())
    use_noise_imprint = any('noise_branch' in k for k in state_dict.keys())
    use_color_features = any('chrominance_branch' in k for k in state_dict.keys())
    use_local_patches = any('local_patch_classifier' in k for k in state_dict.keys())
    use_fpn = any('fpn' in k for k in state_dict.keys())
    
    # Detect attention type
    use_attention = None
    if any('cbam' in k for k in state_dict.keys()):
        use_attention = 'cbam'
    elif any('se_block' in k for k in state_dict.keys()):
        use_attention = 'se'
    
    # Detect attribution
    enable_attribution = any('attribution_head' in k for k in state_dict.keys())
    
    print(f"Detected architecture: {backbone_type}")
    if use_spectral:
        print("  - Spectral branch enabled")
    if use_noise_imprint:
        print("  - Noise imprint detection enabled")
    if use_color_features:
        print("  - Color features enabled")
    if use_attention:
        print(f"  - Attention mechanism: {use_attention}")
    
    # Create model with detected configuration
    model = BinaryClassifier(
        backbone_type=backbone_type,
        pretrained=False,
        use_spectral=use_spectral,
        use_noise_imprint=use_noise_imprint,
        use_color_features=use_color_features,
        use_local_patches=use_local_patches,
        use_fpn=use_fpn,
        use_attention=use_attention,
        enable_attribution=enable_attribution
    )
    
    return model


def load_image(image_path, image_size=256):
    """Load and preprocess image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0)
    
    # Keep original for visualization
    image_original = Image.open(image_path).convert('RGB')
    image_original = image_original.resize((image_size, image_size))
    
    return image_tensor, image_original, original_size


def classify_image(model, image_tensor, device):
    """Classify image as real or fake."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        print(f"Raw logit: {output.item():.4f}")
        print(f"Sigmoid prob: {torch.sigmoid(output).item():.4f}")
        
        # Handle both single output and tuple output (with attribution)
        if isinstance(output, tuple):
            output = output[0]  # Take classification output only
        
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    return prediction, confidence, probabilities[0]


def get_target_layer(model):
    """Get the target layer for Grad-CAM."""
    # Try to find the last convolutional layer in the backbone
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        
        # For ResNet
        if hasattr(backbone, 'layer4'):
            return backbone.layer4[-1].conv2
        
        # For SimpleCNN
        if hasattr(backbone, 'features'):
            # Find last conv layer
            for layer in reversed(list(backbone.features)):
                if isinstance(layer, nn.Conv2d):
                    return layer
    
    # Fallback: find any conv layer
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    
    raise ValueError("Could not find suitable convolutional layer for Grad-CAM")


def visualize_gradcam(image_original, cam, prediction, confidence, save_path=None, display=True):
    """Visualize Grad-CAM heatmap overlaid on original image."""
    # Resize CAM to match image size
    image_np = np.array(image_original)
    h, w = image_np.shape[:2]
    cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image_original)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add classification result
    label = "REAL" if prediction == 1 else "FAKE (AI-Generated)"
    color = 'green' if prediction == 1 else 'red'
    fig.suptitle(f'Classification: {label} (Confidence: {confidence:.2%})', 
                 fontsize=16, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if display:
        print("Displaying visualization...")
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Classify image as real or AI-generated with Grad-CAM visualization'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image to classify')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Image size for model input (default: 256)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda, cpu, or mps (default: cuda)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization (optional, if not provided visualization will only be displayed)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display the visualization window (only save if --output is provided)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print("=" * 70)
    
    try:
        # Load model
        model = load_model(args.model, device)
        
        # Load image
        print(f"\nLoading image: {args.image}")
        image_tensor, image_original, original_size = load_image(args.image, args.image_size)
        print(f"Original image size: {original_size}")
        
        # Classify image
        print("\nClassifying image...")
        prediction, confidence, probabilities = classify_image(model, image_tensor, device)
        
        # Print results
        print("=" * 70)
        print("CLASSIFICATION RESULTS")
        print("=" * 70)
        
        label = "REAL" if prediction == 1 else "FAKE (AI-Generated)"
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nProbabilities:")
        
        # Handle variable number of classes
        if len(probabilities) >= 2:
            print(f"  Real:  {probabilities[0]:.2%}")
            print(f"  Fake:  {probabilities[1]:.2%}")
        else:
            print(f"  Class {prediction}:  {probabilities[0]:.2%}")
        
        # Generate Grad-CAM
        print("\nGenerating Grad-CAM visualization...")
        target_layer = get_target_layer(model)
        print(f"Using layer: {target_layer.__class__.__name__}")
        
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(image_tensor.to(device), target_class=prediction)
        
        # Visualize
        output_path = args.output  # Only save if explicitly provided
        display = not args.no_display
        
        if not display and not output_path:
            print("Warning: --no-display specified without --output, skipping visualization")
        else:
            visualize_gradcam(image_original, cam, prediction, confidence, output_path, display)
        
        print("=" * 70)
        print("Done!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
