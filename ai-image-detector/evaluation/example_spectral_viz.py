"""
Example usage of spectral artifact visualization module.

This script demonstrates how to use the SpectralGradCAM class to visualize
spectral artifacts in ML-generated images.
"""

import torch
import numpy as np
from pathlib import Path

from spectral_viz import (
    SpectralGradCAM,
    visualize_spectral_artifacts,
    get_available_target_layers,
    check_gradcam_availability
)


def example_basic_usage():
    """
    Example 1: Basic usage with automatic layer detection.
    
    This example shows the simplest way to generate spectral visualizations
    using the convenience function.
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Check if GradCAM is available
    if not check_gradcam_availability():
        print("ERROR: pytorch-grad-cam is not installed.")
        print("Install with: pip install pytorch-grad-cam>=1.4.0")
        return
    
    # Load your model (replace with actual model loading)
    print("\n1. Loading model...")
    from ai_image_detector.models.classifier import BinaryClassifier
    
    model = BinaryClassifier(
        backbone_type='resnet18',
        use_spectral=True
    )
    # model.load_state_dict(torch.load('checkpoint.pth'))
    model.eval()
    print("   Model loaded successfully")
    
    # Create sample images (replace with actual images)
    print("\n2. Creating sample images...")
    images = torch.randn(4, 3, 256, 256)
    print(f"   Images shape: {images.shape}")
    
    # Generate visualizations
    print("\n3. Generating spectral visualizations...")
    overlays = visualize_spectral_artifacts(
        model=model,
        images=images,
        overlay_alpha=0.5,
        colormap='jet'
    )
    
    if overlays is not None:
        print(f"   Overlays shape: {overlays.shape}")
        print(f"   Overlays dtype: {overlays.dtype}")
        print(f"   Value range: [{overlays.min()}, {overlays.max()}]")
        
        # Save visualizations
        print("\n4. Saving visualizations...")
        output_dir = Path('spectral_visualizations')
        output_dir.mkdir(exist_ok=True)
        
        from PIL import Image
        for i, overlay in enumerate(overlays):
            img = Image.fromarray(overlay.astype(np.uint8))
            output_path = output_dir / f'spectral_viz_{i}.png'
            img.save(output_path)
            print(f"   Saved: {output_path}")
    
    print("\n" + "=" * 60)


def example_advanced_usage():
    """
    Example 2: Advanced usage with custom target layer.
    
    This example shows how to use SpectralGradCAM class directly with
    a specific target layer for more control.
    """
    print("=" * 60)
    print("Example 2: Advanced Usage with Custom Layer")
    print("=" * 60)
    
    if not check_gradcam_availability():
        print("ERROR: pytorch-grad-cam is not installed.")
        return
    
    # Load model
    print("\n1. Loading model...")
    from ai_image_detector.models.classifier import BinaryClassifier
    
    model = BinaryClassifier(
        backbone_type='resnet18',
        use_spectral=True
    )
    model.eval()
    
    # List available target layers
    print("\n2. Available target layers:")
    layers = get_available_target_layers(model)
    for i, layer in enumerate(layers[:10]):  # Show first 10
        print(f"   {i+1}. {layer}")
    if len(layers) > 10:
        print(f"   ... and {len(layers) - 10} more")
    
    # Create SpectralGradCAM with specific layer
    print("\n3. Creating SpectralGradCAM instance...")
    target_layer = 'spectral_branch.transformer_encoder.layers.3'
    print(f"   Target layer: {target_layer}")
    
    viz = SpectralGradCAM(
        model=model,
        target_layer=target_layer,
        use_cuda=torch.cuda.is_available()
    )
    print("   SpectralGradCAM initialized")
    
    # Generate heatmaps
    print("\n4. Generating heatmaps...")
    images = torch.randn(2, 3, 256, 256)
    heatmaps = viz.generate_heatmaps(images)
    print(f"   Heatmaps shape: {heatmaps.shape}")
    print(f"   Value range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
    
    # Generate overlays
    print("\n5. Generating overlays...")
    overlays = viz.visualize_spectral_artifacts(
        images=images,
        overlay_alpha=0.6,
        colormap='hot'
    )
    print(f"   Overlays shape: {overlays.shape}")
    
    print("\n" + "=" * 60)


def example_batch_processing():
    """
    Example 3: Batch processing multiple images.
    
    This example shows how to process a batch of images and save
    visualizations with metadata.
    """
    print("=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    if not check_gradcam_availability():
        print("ERROR: pytorch-grad-cam is not installed.")
        return
    
    # Load model
    print("\n1. Loading model...")
    from ai_image_detector.models.classifier import BinaryClassifier
    
    model = BinaryClassifier(
        backbone_type='resnet18',
        use_spectral=True
    )
    model.eval()
    
    # Create SpectralGradCAM
    print("\n2. Creating SpectralGradCAM...")
    viz = SpectralGradCAM(model, use_cuda=False)
    
    # Process multiple batches
    print("\n3. Processing batches...")
    num_batches = 3
    batch_size = 4
    
    output_dir = Path('spectral_batch_results')
    output_dir.mkdir(exist_ok=True)
    
    for batch_idx in range(num_batches):
        print(f"\n   Batch {batch_idx + 1}/{num_batches}")
        
        # Generate sample images
        images = torch.randn(batch_size, 3, 256, 256)
        
        # Generate visualizations
        overlays = viz.visualize_spectral_artifacts(images)
        
        # Save each image in the batch
        for img_idx, overlay in enumerate(overlays):
            global_idx = batch_idx * batch_size + img_idx
            output_path = output_dir / f'batch_{batch_idx}_img_{img_idx}.png'
            
            from PIL import Image
            img = Image.fromarray(overlay.astype(np.uint8))
            img.save(output_path)
            
            print(f"      Saved image {global_idx}: {output_path}")
    
    print(f"\n   Total images processed: {num_batches * batch_size}")
    print(f"   Output directory: {output_dir}")
    
    print("\n" + "=" * 60)


def example_comparison():
    """
    Example 4: Compare real vs AI-generated images.
    
    This example shows how to visualize and compare spectral artifacts
    between real and AI-generated images.
    """
    print("=" * 60)
    print("Example 4: Real vs AI-Generated Comparison")
    print("=" * 60)
    
    if not check_gradcam_availability():
        print("ERROR: pytorch-grad-cam is not installed.")
        return
    
    # Load model
    print("\n1. Loading model...")
    from ai_image_detector.models.classifier import BinaryClassifier
    
    model = BinaryClassifier(
        backbone_type='resnet18',
        use_spectral=True
    )
    model.eval()
    
    # Create SpectralGradCAM
    print("\n2. Creating SpectralGradCAM...")
    viz = SpectralGradCAM(model, use_cuda=False)
    
    # Simulate real and AI-generated images
    print("\n3. Processing images...")
    real_images = torch.randn(2, 3, 256, 256)
    ai_images = torch.randn(2, 3, 256, 256)
    
    print("   Generating visualizations for real images...")
    real_overlays = viz.visualize_spectral_artifacts(real_images)
    
    print("   Generating visualizations for AI-generated images...")
    ai_overlays = viz.visualize_spectral_artifacts(ai_images)
    
    # Save comparison
    print("\n4. Saving comparison...")
    output_dir = Path('spectral_comparison')
    output_dir.mkdir(exist_ok=True)
    
    from PIL import Image
    for i in range(len(real_overlays)):
        # Save real image visualization
        real_img = Image.fromarray(real_overlays[i].astype(np.uint8))
        real_img.save(output_dir / f'real_{i}.png')
        
        # Save AI-generated image visualization
        ai_img = Image.fromarray(ai_overlays[i].astype(np.uint8))
        ai_img.save(output_dir / f'ai_generated_{i}.png')
        
        print(f"   Saved comparison pair {i}")
    
    print(f"\n   Output directory: {output_dir}")
    print("\n" + "=" * 60)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SPECTRAL ARTIFACT VISUALIZATION EXAMPLES")
    print("=" * 60 + "\n")
    
    # Check availability first
    if not check_gradcam_availability():
        print("ERROR: pytorch-grad-cam is not installed.")
        print("Install with: pip install pytorch-grad-cam>=1.4.0")
        print("\nExamples cannot run without this dependency.")
        return
    
    try:
        # Run examples
        example_basic_usage()
        print("\n")
        
        example_advanced_usage()
        print("\n")
        
        example_batch_processing()
        print("\n")
        
        example_comparison()
        print("\n")
        
        print("=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
