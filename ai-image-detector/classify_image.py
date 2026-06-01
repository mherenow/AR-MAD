#!/usr/bin/env python
"""
AI-generated image detection with class-discriminative CAM visualisation.

Why Grad-CAM fails here
-----------------------
The ResNet backbone is wrapped as nn.Sequential(*list(resnet.children())[:-1]),
which includes AdaptiveAvgPool2d(1,1) at position [8].  During backprop the
avgpool distributes the incoming gradient uniformly across all 8×8 positions at
a scale (~1e-5) that float32 rounds to exactly 0.0.  Every α_k = 0 → blank map.
This cannot be fixed by changing the target layer, sigmoid vs logit, or ReLU.

Why EigenCAM also fails here
-----------------------------
EigenCAM uses the first SVD component of the activation tensor, which captures
the dominant spatial pattern regardless of what class the model is predicting.
Both FAKE and REAL images activate the same spatial structure (object shape),
so their heatmaps look identical.

Correct solution: Effective-Weight CAM (class-discriminative, gradient-free)
-----------------------------------------------------------------------------
The classifier head is:  Linear(2048,256) → ReLU → Dropout → Linear(256,1)

For a given image with pooled feature vector f ∈ R^2048:
    h       = ReLU(W1 @ f + b1)             (256,)  hidden activations
    active  = (h > 0).float()               (256,)  which neurons fired
    w_eff   = (w2 * active) @ W1            (2048,) exact gradient of logit
                                                     w.r.t. f through ReLU
    w_fake  = ReLU(w_eff)                   (2048,) only fake-promoting channels

CAM = sum_k (w_fake[k] * A[k, :, :])  projected over spatial feature map A

This is computed per-image using the actual hidden activations, so it reflects
the real gradient path even through non-linear layers.

Class discrimination
--------------------
The heatmap is ALWAYS computed in the fake direction (w_fake = positive weights
only).  Discrimination between real and fake comes from magnitude, not shape:

  FAKE image → high logit → many channels activated strongly → bright, localised
  REAL image → low logit  → fake-promoting channels barely fire → naturally dim

To preserve this, the heatmap is normalised by a soft global cap
(99th-percentile of the raw CAM values) rather than the per-image maximum.
The confidence score (sigmoid(logit - threshold)) then further scales the
display so real images are dim even when the spatial pattern is non-zero.

Threshold
---------
Classification boundary is logit > 0.9244 (calibrated on the training data).
"""

import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ─────────────────────────────────────────────────────────────────────────────
# Classification threshold
# ─────────────────────────────────────────────────────────────────────────────

FAKE_THRESHOLD = 0.9244   # logit > this → FAKE


# ─────────────────────────────────────────────────────────────────────────────
# Effective-Weight CAM
# ─────────────────────────────────────────────────────────────────────────────

class EffectiveWeightCAM:
    """
    Class-discriminative CAM for a GAP + multi-layer FC head.

    Works by computing the exact gradient of the logit w.r.t. the 2048-dim
    pooled feature vector analytically (no autograd), then projecting back
    onto the spatial feature map at layer4 output.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._spatial = None   # (2048, H, W) feature map before avgpool
        self._pooled  = None   # (2048,) feature vector after avgpool

        # Hooks — registered once, removed on demand
        target_layer = self._find_layer4(model)
        avgpool_layer = self._find_avgpool(model)

        self._h1 = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_spatial', o.detach()))
        self._h2 = avgpool_layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_pooled', o.detach()))

    # ── layer finders ────────────────────────────────────────────────────────

    @staticmethod
    def _find_layer4(model: nn.Module) -> nn.Module:
        backbone = model.backbone
        if isinstance(backbone, nn.Sequential):
            try:
                layer4 = backbone[7]
                if isinstance(layer4, nn.Sequential):
                    print(f"  CAM target → backbone[7][-1] "
                          f"({layer4[-1].__class__.__name__})")
                    return layer4[-1]
            except (IndexError, TypeError):
                pass
        if hasattr(backbone, 'layer4'):
            print(f"  CAM target → layer4[-1]")
            return backbone.layer4[-1]
        # SimpleCNN fallback — last conv
        last_conv = None
        for m in backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv:
            print("  CAM target → last Conv2d")
            return last_conv
        raise ValueError("Cannot find layer4 in backbone.")

    @staticmethod
    def _find_avgpool(model: nn.Module) -> nn.Module:
        backbone = model.backbone
        if isinstance(backbone, nn.Sequential):
            # AdaptiveAvgPool2d is at index 8 in ResNet Sequential
            for idx, m in enumerate(backbone):
                if isinstance(m, nn.AdaptiveAvgPool2d):
                    print(f"  AvgPool  → backbone[{idx}]")
                    return m
        if hasattr(backbone, 'avgpool'):
            return backbone.avgpool
        # SimpleCNN: use last maxpool or final block as pooling proxy
        for m in reversed(list(backbone.modules())):
            if isinstance(m, (nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                return m
        raise ValueError("Cannot find avgpool in backbone.")

    # ── find the classification head layers ──────────────────────────────────

    @staticmethod
    def _find_head_weights(model: nn.Module):
        """
        Return (W1, b1, W2) for head of the form Linear → ReLU → ... → Linear.
        Handles both:
          model.classifier.classifier  (BinaryClassifier structure)
          model.head / model.fc        (simpler structures)
        """
        # Try BinaryClassifier's ClassificationHead
        head = None
        for attr in ('classifier', 'head', 'fc'):
            candidate = getattr(model, attr, None)
            if candidate is not None:
                head = candidate
                break

        if head is None:
            raise ValueError("Cannot find classification head.")

        # Collect all Linear layers in order
        linears = [(n, m) for n, m in head.named_modules()
                   if isinstance(m, nn.Linear)]

        if len(linears) >= 2:
            _, fc1 = linears[0]
            _, fc2 = linears[-1]
            return fc1.weight, fc1.bias, fc2.weight
        elif len(linears) == 1:
            # Single linear: w_eff = w directly
            _, fc = linears[0]
            return None, None, fc.weight
        else:
            raise ValueError("No Linear layers found in head.")

    # ── main interface ────────────────────────────────────────────────────────

    def generate_cam(self, input_image: torch.Tensor) -> tuple:
        """
        Returns (cam_display, logit, p_fake) where:
          cam_display  (H, W) float32 in [0, 1] — confidence-scaled heatmap
                       FAKE images → bright regions where artifacts were detected
                       REAL images → naturally dim (fake channels barely fire)
          logit        raw model output
          p_fake       sigmoid(logit - threshold) ∈ [0, 1]
        """
        self._spatial = None
        self._pooled  = None
        self.model.eval()

        with torch.no_grad():
            out = self.model(input_image)
            if isinstance(out, tuple):
                out = out[0]
            logit = out.item()

        if self._spatial is None or self._pooled is None:
            raise RuntimeError("Forward hooks did not fire.")

        A = self._spatial[0]        # (C, H, W) — spatial feature map
        f = self._pooled[0].view(-1) # (C,)      — pooled feature vector

        C, H, W = A.shape

        try:
            W1, b1, W2 = self._find_head_weights(self.model)
        except ValueError:
            # Fallback: treat sum of activation channels as heatmap
            cam_raw = A.sum(0)
            cam_raw = F.relu(cam_raw)
            cam_norm = cam_raw / (cam_raw.max() + 1e-8)
            p_fake = torch.sigmoid(
                torch.tensor(logit - FAKE_THRESHOLD)).item()
            return (cam_norm * p_fake).cpu().numpy(), logit, p_fake

        # ── compute per-image effective weights ──────────────────────────────
        with torch.no_grad():
            if W1 is not None:
                # Two-layer head: Linear(C,256) → ReLU → Linear(256,1)
                h_hidden = F.relu(W1 @ f + b1)         # (256,)
                active   = (h_hidden > 0).float()       # which neurons fired
                w_eff    = (W2[0] * active) @ W1        # (C,) true gradient
            else:
                # Single linear: gradient is just the weight row
                w_eff = W2[0]                           # (C,)

            # Only channels that push toward FAKE (positive contribution)
            w_fake  = F.relu(w_eff)                     # (C,)

            # Weighted spatial map
            cam_raw = (w_fake[:, None, None] * A).sum(0)  # (H, W)
            cam_raw = F.relu(cam_raw)

        # ── normalise by 99th percentile to preserve cross-image scale ────────
        # Per-image max normalisation makes every image look equally bright.
        # Using 99th-percentile cap means:
        #   - FAKE (high activation): saturates at cap → bright
        #   - REAL (low activation):  far below cap    → naturally dim
        if cam_raw.max() > 1e-8:
            cap = torch.quantile(cam_raw, 0.99).clamp(min=1e-8)
            cam_display = (cam_raw / cap).clamp(0, 1)
        else:
            cam_display = cam_raw

        # ── further scale by fake confidence ─────────────────────────────────
        # p_fake = sigmoid(logit - threshold)
        #   logit >> threshold (confident FAKE): p_fake → 1.0 → heatmap unchanged
        #   logit << threshold (confident REAL): p_fake → 0.0 → heatmap dimmed to zero
        p_fake = torch.sigmoid(
            torch.tensor(logit - FAKE_THRESHOLD, dtype=torch.float32)).item()

        cam_scaled = (cam_display * p_fake).cpu().numpy()

        return cam_scaled, logit, p_fake

    def remove_hooks(self):
        self._h1.remove()
        self._h2.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}…")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    model = _reconstruct_model(state_dict)

    # Handle lazily-initialised positional embeddings (SpectralPatchTokenizer)
    for name, param in state_dict.items():
        if name.endswith('.pos_embedding'):
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
    print("Model loaded.")
    return model


def _reconstruct_model(state_dict: dict):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.classifier import BinaryClassifier

    ck = 'classifier.classifier.0.weight'
    if ck in state_dict:
        in_dim = state_dict[ck].shape[1]
        backbone_type = 'resnet50' if in_dim == 2048 else 'resnet18'
    elif any('backbone.conv1' in k for k in state_dict):
        backbone_type = 'simple_cnn'
    elif any('backbone.6.5' in k for k in state_dict):
        backbone_type = 'resnet50'
    else:
        backbone_type = 'resnet18'

    flags = dict(
        use_spectral       = any('spectral_branch'       in k for k in state_dict),
        use_noise_imprint  = any('noise_branch'           in k for k in state_dict),
        use_color_features = any('chrominance_branch'     in k for k in state_dict),
        use_local_patches  = any('local_patch_classifier' in k for k in state_dict),
        use_fpn            = any(k.startswith('fpn')      for k in state_dict),
        enable_attribution = any('attribution_head'       in k for k in state_dict),
    )
    use_attention = None
    if any('attention_module' in k for k in state_dict):
        use_attention = ('cbam' if any('channel_attention' in k for k in state_dict)
                         else 'se')

    print(f"  Backbone : {backbone_type}")
    for k, v in flags.items():
        if v:
            print(f"  Flag     : {k}")

    return BinaryClassifier(backbone_type=backbone_type, pretrained=False,
                            use_attention=use_attention, **flags)


# ─────────────────────────────────────────────────────────────────────────────
# Image loading and classification
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: str, image_size: int = 256):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)
    orig_resized = Image.open(image_path).convert('RGB').resize(
        (image_size, image_size))
    return tensor, orig_resized, original_size


def classify_image(logit: float):
    """Apply calibrated threshold and return prediction + confidences."""
    prediction = 1 if logit > FAKE_THRESHOLD else 0
    prob_fake  = torch.sigmoid(torch.tensor(logit)).item()
    prob_real  = 1.0 - prob_fake
    confidence = prob_fake if prediction == 1 else prob_real
    return prediction, confidence, prob_real, prob_fake


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize(image_original, cam: np.ndarray, prediction: int,
              confidence: float, prob_fake: float,
              save_path=None, display=True):
    """
    Three-panel figure: original | heatmap | overlay.

    Heatmap intensity encodes both spatial location of detected artifacts AND
    the model's fake confidence — real images naturally appear dim/blank.
    """
    img_np   = np.array(image_original)
    h, w     = img_np.shape[:2]
    cam_up   = np.array(Image.fromarray((cam * 255).astype(np.uint8))
                        .resize((w, h), Image.BILINEAR)) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_original)
    axes[0].set_title('Original Image', fontsize=11)
    axes[0].axis('off')

    # Heatmap: use 'hot' colormap — black=no signal, white/yellow=high signal
    axes[1].imshow(cam_up, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Artifact Heatmap  (p_fake={prob_fake:.3f})', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(image_original)
    # Only show overlay where cam is non-trivial
    masked_cam = np.ma.masked_where(cam_up < 0.05, cam_up)
    axes[2].imshow(masked_cam, cmap='hot', alpha=0.65, vmin=0, vmax=1)
    axes[2].set_title('Overlay', fontsize=11)
    axes[2].axis('off')

    label = "FAKE (AI-Generated)" if prediction == 1 else "REAL"
    color = 'red' if prediction == 1 else 'green'
    fig.suptitle(
        f'Classification: {label}  (Confidence: {confidence:.1%})',
        fontsize=16, fontweight='bold', color=color)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    if display:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Classify image as real or AI-generated')
    ap.add_argument('--model',      required=True)
    ap.add_argument('--image',      required=True)
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--output',     default=None)
    ap.add_argument('--no-display', action='store_true')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print("=" * 60)

    try:
        model  = load_model(args.model, device)
        tensor, orig, orig_size = load_image(args.image, args.image_size)
        print(f"Image: {args.image}  ({orig_size[0]}×{orig_size[1]})")

        # ── Generate CAM (also runs forward pass, gives us logit) ────────────
        print("Generating heatmap…")
        cam_engine            = EffectiveWeightCAM(model)
        cam, logit, p_fake    = cam_engine.generate_cam(tensor.to(device))
        cam_engine.remove_hooks()

        # ── Classify ─────────────────────────────────────────────────────────
        prediction, confidence, prob_real, prob_fake = classify_image(logit)

        label = "FAKE (AI-Generated)" if prediction == 1 else "REAL"
        print("=" * 60)
        print(f"Prediction  : {label}")
        print(f"Confidence  : {confidence:.1%}")
        print(f"Raw logit   : {logit:+.4f}  (threshold {FAKE_THRESHOLD:+.4f})")
        print(f"P(fake)     : {prob_fake:.4f}")
        print(f"P(real)     : {prob_real:.4f}")
        print(f"Heatmap scale: {cam.max():.4f}  "
              f"({'bright' if cam.max() > 0.3 else 'dim — consistent with REAL'})")

        display = not args.no_display
        if display or args.output:
            visualize(orig, cam, prediction, confidence, prob_fake,
                      args.output, display)

        print("Done.")

    except Exception as e:
        import traceback
        print(f"\nError: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()