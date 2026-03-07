"""
Spectral artifact visualization module using GradCAM.

This module provides visualization tools for spectral branch features using
GradCAM (Gradient-weighted Class Activation Mapping) to highlight frequency
domain regions that contribute most to the model's predictions.

The module handles the optional pytorch-grad-cam dependency gracefully,
providing informative error messages when the library is unavailable.
"""

import warnings
from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
import numpy as np


# Check for optional pytorch-grad-cam dependency
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    warnings.warn(
        "pytorch-grad-cam not available. Spectral artifact visualization disabled. "
        "Install with: pip install pytorch-grad-cam>=1.4.0",
        ImportWarning
    )


class SpectralGradCAM:
    """
    GradCAM visualization for spectral branch features.
    
    This class wraps the pytorch-grad-cam library to provide easy-to-use
    visualization of spectral artifacts. It generates attention heatmaps
    showing which frequency domain regions are most important for the
    model's predictions.
    
    Args:
        model: Model with spectral branch
        target_layer: Target layer for GradCAM computation. Can be:
                     - Layer name string (e.g., 'spectral_branch.transformer_encoder.layers.3')
                     - nn.Module instance
                     If None, attempts to use the last transformer layer
        use_cuda: Whether to use CUDA if available (default: True)
    
    Raises:
        ImportError: If pytorch-grad-cam is not installed
        ValueError: If target_layer cannot be found in the model
    
    Example:
        >>> model = BinaryClassifier(use_spectral=True)
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> viz = SpectralGradCAM(model)
        >>> images = torch.randn(4, 3, 256, 256)
        >>> heatmaps = viz.generate_heatmaps(images)
        >>> heatmaps.shape
        torch.Size([4, 256, 256])
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[str, nn.Module]] = None,
        use_cuda: bool = True
    ):
        if not GRADCAM_AVAILABLE:
            raise ImportError(
                "pytorch-grad-cam is required for spectral visualization. "
                "Install with: pip install pytorch-grad-cam>=1.4.0"
            )
        
        self.model = model
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Resolve target layer
        if target_layer is None:
            # Try to find the last transformer layer in spectral branch
            target_layer = self._find_default_target_layer()
        elif isinstance(target_layer, str):
            target_layer = self._get_layer_by_name(target_layer)
        
        if target_layer is None:
            raise ValueError(
                "Could not find target layer. Please specify target_layer explicitly."
            )
        
        self.target_layer = target_layer
        
        # Initialize GradCAM
        self.cam = GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            use_cuda=use_cuda
        )
    
    def _find_default_target_layer(self) -> Optional[nn.Module]:
        """
        Find the default target layer (last transformer layer in spectral branch).
        
        Returns:
            Target layer module or None if not found
        """
        # Try to access spectral_branch.transformer_encoder.layers[-1]
        if hasattr(self.model, 'spectral_branch'):
            spectral_branch = self.model.spectral_branch
            if hasattr(spectral_branch, 'transformer_encoder'):
                transformer = spectral_branch.transformer_encoder
                if hasattr(transformer, 'layers') and len(transformer.layers) > 0:
                    return transformer.layers[-1]
        
        return None
    
    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """
        Get a layer from the model by its name.
        
        Args:
            layer_name: Dot-separated layer name (e.g., 'spectral_branch.transformer_encoder.layers.3')
        
        Returns:
            Layer module or None if not found
        """
        parts = layer_name.split('.')
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and isinstance(module, (nn.ModuleList, list)):
                idx = int(part)
                if 0 <= idx < len(module):
                    module = module[idx]
                else:
                    return None
            else:
                return None
        
        return module
    
    def generate_heatmaps(
        self,
        images: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate GradCAM heatmaps for input images.
        
        Args:
            images: Input images of shape (B, 3, H, W)
            target_class: Target class for GradCAM (default: None, uses predicted class)
                         For binary classification: 0 or 1
        
        Returns:
            heatmaps: GradCAM heatmaps of shape (B, H, W) with values in [0, 1]
        
        Example:
            >>> viz = SpectralGradCAM(model)
            >>> images = torch.randn(4, 3, 256, 256)
            >>> heatmaps = viz.generate_heatmaps(images)
            >>> # Visualize first heatmap
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow(heatmaps[0].cpu().numpy(), cmap='jet')
            >>> plt.colorbar()
            >>> plt.show()
        """
        images = images.to(self.device)
        
        # Set up targets for GradCAM
        if target_class is not None:
            targets = [BinaryClassifierOutputTarget(target_class)] * images.shape[0]
        else:
            targets = None
        
        # Generate CAM
        with torch.no_grad():
            # Get model predictions to determine target if not specified
            if targets is None:
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predictions = (outputs > 0.5).long().squeeze()
                targets = [BinaryClassifierOutputTarget(int(pred)) for pred in predictions]
        
        # Compute GradCAM
        grayscale_cam = self.cam(input_tensor=images, targets=targets)
        
        # Convert to tensor
        heatmaps = torch.from_numpy(grayscale_cam).float()
        
        return heatmaps
    
    def visualize_spectral_artifacts(
        self,
        images: torch.Tensor,
        overlay_alpha: float = 0.5,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Generate visualization overlaying heatmaps on original images.
        
        Args:
            images: Input images of shape (B, 3, H, W) in range [0, 1]
            overlay_alpha: Alpha blending factor for overlay (default: 0.5)
            colormap: Matplotlib colormap name (default: 'jet')
        
        Returns:
            overlays: Overlaid images of shape (B, H, W, 3) in range [0, 255]
        
        Example:
            >>> viz = SpectralGradCAM(model)
            >>> images = torch.randn(4, 3, 256, 256)
            >>> overlays = viz.visualize_spectral_artifacts(images)
            >>> # Save first overlay
            >>> from PIL import Image
            >>> img = Image.fromarray(overlays[0].astype(np.uint8))
            >>> img.save('spectral_overlay.png')
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # Generate heatmaps
        heatmaps = self.generate_heatmaps(images)
        
        # Convert images to numpy (B, H, W, 3)
        images_np = images.cpu().numpy()
        images_np = np.transpose(images_np, (0, 2, 3, 1))  # (B, 3, H, W) -> (B, H, W, 3)
        images_np = (images_np * 255).astype(np.uint8)
        
        # Apply colormap to heatmaps
        cmap = cm.get_cmap(colormap)
        heatmaps_np = heatmaps.cpu().numpy()
        
        overlays = []
        for img, heatmap in zip(images_np, heatmaps_np):
            # Apply colormap (returns RGBA)
            heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Blend with original image
            overlay = (overlay_alpha * heatmap_colored + (1 - overlay_alpha) * img).astype(np.uint8)
            overlays.append(overlay)
        
        return np.array(overlays)
    
    def __del__(self):
        """Cleanup GradCAM resources."""
        if hasattr(self, 'cam'):
            del self.cam


def visualize_spectral_artifacts(
    model: nn.Module,
    images: torch.Tensor,
    target_layer: Optional[Union[str, nn.Module]] = None,
    device: Optional[torch.device] = None,
    overlay_alpha: float = 0.5,
    colormap: str = 'jet'
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convenience function to visualize spectral artifacts using GradCAM.
    
    This function provides a simple interface for generating GradCAM
    visualizations without manually creating a SpectralGradCAM instance.
    
    Args:
        model: Model with spectral branch
        images: Input images of shape (B, 3, H, W)
        target_layer: Target layer for GradCAM (default: None, auto-detect)
        device: Device to run on (default: None, auto-detect)
        overlay_alpha: Alpha blending factor for overlay (default: 0.5)
        colormap: Matplotlib colormap name (default: 'jet')
    
    Returns:
        If GRADCAM_AVAILABLE:
            overlays: Overlaid images of shape (B, H, W, 3) in range [0, 255]
        Else:
            None (with warning message)
    
    Raises:
        ImportError: If pytorch-grad-cam is not installed
    
    Example:
        >>> model = BinaryClassifier(use_spectral=True)
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> images = torch.randn(4, 3, 256, 256)
        >>> overlays = visualize_spectral_artifacts(model, images)
        >>> if overlays is not None:
        ...     # Save visualizations
        ...     for i, overlay in enumerate(overlays):
        ...         img = Image.fromarray(overlay.astype(np.uint8))
        ...         img.save(f'spectral_viz_{i}.png')
    """
    if not GRADCAM_AVAILABLE:
        warnings.warn(
            "pytorch-grad-cam not available. Cannot generate spectral visualizations. "
            "Install with: pip install pytorch-grad-cam>=1.4.0"
        )
        return None
    
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create SpectralGradCAM instance
    use_cuda = device.type == 'cuda'
    viz = SpectralGradCAM(model, target_layer=target_layer, use_cuda=use_cuda)
    
    # Generate overlays
    overlays = viz.visualize_spectral_artifacts(
        images,
        overlay_alpha=overlay_alpha,
        colormap=colormap
    )
    
    return overlays


def get_available_target_layers(model: nn.Module) -> List[str]:
    """
    Get list of available target layers in the model for GradCAM.
    
    This function helps identify suitable layers for GradCAM visualization
    by listing all named modules in the model.
    
    Args:
        model: Model to inspect
    
    Returns:
        List of layer names suitable for GradCAM
    
    Example:
        >>> model = BinaryClassifier(use_spectral=True)
        >>> layers = get_available_target_layers(model)
        >>> print("Available layers:")
        >>> for layer in layers:
        ...     print(f"  - {layer}")
    """
    available_layers = []
    
    for name, module in model.named_modules():
        # Include layers that are commonly used for GradCAM
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
            available_layers.append(name)
        # Include transformer encoder layers
        elif 'transformer' in name.lower() and 'layer' in name.lower():
            available_layers.append(name)
    
    return available_layers


def check_gradcam_availability() -> bool:
    """
    Check if pytorch-grad-cam is available.
    
    Returns:
        True if pytorch-grad-cam is installed, False otherwise
    
    Example:
        >>> if check_gradcam_availability():
        ...     print("GradCAM visualization is available")
        ... else:
        ...     print("Install pytorch-grad-cam to enable visualization")
    """
    return GRADCAM_AVAILABLE
