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
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap."""
        device = input_image.device
        
        # Forward pass
        output = self.model(input_image)
        
        # Handle tuple output (with attribution)
        if isinstance(output, tuple):
            output = output[0]
        
        # For single-logit output, target_class is irrelevant — use logit directly
        self.model.zero_grad()
        output.backward()
        
        # Generate CAM
        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured. The target layer may not be in the "
                "gradient path for this input. Try a different target layer."
            )
        gradients = self.gradients[0]   # [C, H, W]
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

    # The SpectralPatchTokenizer initializes pos_embedding lazily in forward(),
    # so it exists in the saved state_dict but not yet on the fresh model.
    # Pre-register it as an nn.Parameter so strict loading works.
    for name, param in state_dict.items():
        if name.endswith('.pos_embedding'):
            # Walk the module path and set the parameter
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            if getattr(module, 'pos_embedding', None) is None:
                module.pos_embedding = nn.Parameter(torch.empty_like(param))
                module.num_patches = param.shape[1]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def reconstruct_model_from_state_dict(state_dict):
    """Reconstruct model architecture from state dict keys."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.classifier import BinaryClassifier
    
    # Detect backbone type
    classifier_weight_key = 'classifier.classifier.0.weight'
    if classifier_weight_key in state_dict:
        classifier_input_dim = state_dict[classifier_weight_key].shape[1]
        if classifier_input_dim == 2048:
            backbone_type = 'resnet50'
        elif any('backbone.6.5' in k for k in state_dict.keys()):
            backbone_type = 'resnet50'
        elif any('backbone.7' in k for k in state_dict.keys()):
            backbone_type = 'resnet18'
        elif any('backbone.conv1' in k for k in state_dict.keys()):
            backbone_type = 'simple_cnn'
        else:
            backbone_type = 'resnet18'
    else:
        if any('backbone.conv1' in k for k in state_dict.keys()):
            backbone_type = 'simple_cnn'
        elif any('backbone.6.5' in k for k in state_dict.keys()):
            backbone_type = 'resnet50'
        else:
            backbone_type = 'resnet18'
    
    # Detect feature flags from state dict keys
    use_spectral      = any('spectral_branch' in k for k in state_dict.keys())
    use_noise_imprint = any('noise_branch' in k for k in state_dict.keys())
    use_color_features = any('chrominance_branch' in k for k in state_dict.keys())
    use_local_patches  = any('local_patch_classifier' in k for k in state_dict.keys())
    use_fpn            = any(k.startswith('fpn') for k in state_dict.keys())
    enable_attribution = any('attribution_head' in k for k in state_dict.keys())

    use_attention = None
    if any('attention_module' in k for k in state_dict.keys()):
        if any('cbam' in k.lower() or 'channel_attention' in k for k in state_dict.keys()):
            use_attention = 'cbam'
        else:
            use_attention = 'se'

    print(f"Detected architecture: {backbone_type}")
    if use_spectral:       print("  - Spectral branch enabled")
    if use_noise_imprint:  print("  - Noise imprint detection enabled")
    if use_color_features: print("  - Color features enabled")
    if use_fpn:            print("  - FPN enabled")
    if use_attention:      print(f"  - Attention mechanism: {use_attention}")

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
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    image_original = Image.open(image_path).convert('RGB').resize((image_size, image_size))
    
    return image_tensor, image_original, original_size


def classify_image(model, image_tensor, device):
    """
    Classify image as real or fake.

    Convention: label 0 = REAL, label 1 = FAKE.
    Model outputs a single logit (BCEWithLogitsLoss).
      logit > 0  →  FAKE
      logit <= 0 →  REAL
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)

        # Handle tuple output (with attribution head)
        if isinstance(output, tuple):
            output = output[0]

        logit = output.item()
        prob_fake = torch.sigmoid(output).item()   # P(FAKE)
        prob_real = 1.0 - prob_fake

        print(f"Raw logit:            {logit:.4f}")
        print(f"P(fake):              {prob_fake:.4f}")
        print(f"P(real):              {prob_real:.4f}")

        if logit > 0.01:
            prediction = 1          # FAKE
            confidence = prob_fake
        else:
            prediction = 0          # REAL
            confidence = prob_real

        # Two-element tensor for display: [P(real), P(fake)]
        probabilities = torch.tensor([prob_real, prob_fake])

    return prediction, confidence, probabilities


def get_target_layer(model):
    """Get the target layer for Grad-CAM."""
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        # Named attribute (standard torchvision ResNet)
        if hasattr(backbone, 'layer4'):
            return backbone.layer4[-1].conv2
        # Sequential-wrapped ResNet (backbone[7] = layer4)
        if isinstance(backbone, nn.Sequential):
            # ResNet50: indices 0-7 are conv1,bn1,relu,maxpool,layer1-4
            # layer4 is at index 7 for both resnet18 and resnet50
            try:
                layer4 = backbone[7]
                if isinstance(layer4, nn.Sequential):
                    last_block = layer4[-1]
                    # Bottleneck (resnet50) has conv3; BasicBlock (resnet18) has conv2
                    if hasattr(last_block, 'conv3'):
                        return last_block.conv3
                    if hasattr(last_block, 'conv2'):
                        return last_block.conv2
            except (IndexError, TypeError):
                pass
        if hasattr(backbone, 'features'):
            for layer in reversed(list(backbone.features)):
                if isinstance(layer, nn.Conv2d):
                    return layer
    raise ValueError("Could not find suitable convolutional layer for Grad-CAM")


def visualize_gradcam(image_original, cam, prediction, confidence, save_path=None, display=True):
    """Visualize Grad-CAM heatmap overlaid on original image."""
    image_np = np.array(image_original)
    h, w = image_np.shape[:2]
    cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(image_original)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    # prediction=1 → FAKE, prediction=0 → REAL
    label = "FAKE (AI-Generated)" if prediction == 1 else "REAL"
    color = 'red' if prediction == 1 else 'green'
    fig.suptitle(f'Classification: {label} (Confidence: {confidence:.1%})',
                 fontsize=16, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Classify image as real or AI-generated with Grad-CAM visualization'
    )
    parser.add_argument('--model',      type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--image',      type=str, required=True,
                        help='Path to image to classify')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size for model input (default: 256)')
    parser.add_argument('--device',     type=str, default='cuda',
                        help='Device: cuda, cpu, or mps (default: cuda)')
    parser.add_argument('--output',     type=str, default=None,
                        help='Path to save visualization (optional)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display the visualization window')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print("=" * 70)
    
    try:
        model = load_model(args.model, device)
        
        print(f"\nLoading image: {args.image}")
        image_tensor, image_original, original_size = load_image(args.image, args.image_size)
        print(f"Original image size: {original_size}")
        
        print("\nClassifying image...")
        prediction, confidence, probabilities = classify_image(model, image_tensor, device)
        
        print("=" * 70)
        print("CLASSIFICATION RESULTS")
        print("=" * 70)

        # Convention: prediction=1 → FAKE, prediction=0 → REAL
        label = "FAKE (AI-Generated)" if prediction == 1 else "REAL"
        print(f"Prediction:  {label}")
        print(f"Confidence:  {confidence:.1%}")
        print(f"\nProbabilities:")
        print(f"  Real:  {probabilities[0]:.1%}")
        print(f"  Fake:  {probabilities[1]:.1%}")
        
        print("\nGenerating Grad-CAM visualization...")
        target_layer = get_target_layer(model)
        print(f"Using layer: {target_layer.__class__.__name__}")
        
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(image_tensor.to(device), target_class=prediction)
        
        display = not args.no_display
        if not display and not args.output:
            print("Warning: --no-display set without --output, skipping visualization")
        else:
            visualize_gradcam(image_original, cam, prediction, confidence,
                              args.output, display)
        
        print("=" * 70)
        print("Done!")
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()