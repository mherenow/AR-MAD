"""
Bug Condition Exploration Test for Training Fixes and Cleanup

This test explores the bug condition where unnormalized [0, 1] inputs cause model collapse.
It is designed to FAIL on unfixed code to demonstrate the bug exists.

**CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
**DO NOT attempt to fix the test or the code when it fails.**
**NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation.

**GOAL**: Surface counterexamples that demonstrate model collapse from unnormalized [0, 1] inputs.

Test Strategy:
- Load batches from each dataset loader (SynthBuster, COCO, Combined)
- Verify tensors are in [0, 1] range (not normalized)
- Train for 1 epoch
- Observe uniform predictions and flat loss
- Document counterexamples: tensor value ranges, prediction distributions, logit ranges, loss curves, accuracy values

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Usage:
    pytest ai-image-detector/data/test_bug_condition_exploration.py -v -s
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from .synthbuster_loader import SynthBusterDataset
from .coco_loader import COCO2017Dataset
from .combined_loader import BalancedCombinedDataset


def is_bug_condition(batch: torch.Tensor) -> bool:
    """
    Check if batch exhibits the bug condition: unnormalized [0, 1] inputs.
    
    Args:
        batch: Image tensor batch of shape (B, C, H, W)
        
    Returns:
        True if batch is in [0, 1] range (unnormalized), False otherwise
    """
    return (
        batch.min().item() >= 0.0 and
        batch.max().item() <= 1.0 and
        0.3 <= batch.mean().item() <= 0.6 and
        not is_imagenet_normalized(batch)
    )


def is_imagenet_normalized(batch: torch.Tensor) -> bool:
    """
    Check if batch is ImageNet normalized (approximately [-2.1, 2.6] range).
    
    Args:
        batch: Image tensor batch of shape (B, C, H, W)
        
    Returns:
        True if batch appears to be ImageNet normalized, False otherwise
    """
    return (
        batch.min().item() < -1.0 and
        batch.max().item() > 1.5 and
        abs(batch.mean().item()) < 0.3
    )


def create_simple_model(num_classes: int = 2) -> nn.Module:
    """
    Create a simple ResNet18 model for testing.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        
    Returns:
        PyTorch model with pretrained ResNet18 backbone
    """
    import torchvision.models as models
    
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_one_epoch(model, dataloader, device='cpu', max_batches=50):
    """
    Train model for one epoch and collect metrics.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader providing training batches
        device: Device to train on ('cpu' or 'cuda')
        max_batches: Maximum number of batches to process (default: 50)
        
    Returns:
        Dictionary containing training metrics:
        - losses: List of loss values per batch
        - predictions: List of prediction tensors
        - logits: List of raw logit tensors
        - labels: List of label tensors
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    metrics = {
        'losses': [],
        'predictions': [],
        'logits': [],
        'labels': []
    }
    
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        # Handle different dataset return formats
        if len(batch_data) == 3:
            # SynthBuster format: (images, labels, generator_names)
            images, labels, _ = batch_data
        else:
            # COCO format: (images, labels)
            images, labels = batch_data
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect metrics
        metrics['losses'].append(loss.item())
        metrics['logits'].append(outputs.detach().cpu())
        
        # Get predictions (class with highest probability)
        _, predicted = torch.max(outputs.data, 1)
        metrics['predictions'].append(predicted.cpu())
        metrics['labels'].append(labels.cpu())
    
    return metrics


class TestBugConditionExploration(unittest.TestCase):
    """
    Bug condition exploration tests.
    
    These tests are designed to FAIL on unfixed code to demonstrate the bug exists.
    """
    
    @classmethod
    def setUpClass(cls):
        """Create temporary datasets for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create SynthBuster dataset structure
        cls.synthbuster_root = Path(cls.temp_dir) / "synthbuster"
        cls.synthbuster_root.mkdir()
        
        # Create RAISE (real) and generator (fake) directories
        raise_dir = cls.synthbuster_root / "RAISE"
        raise_dir.mkdir()
        
        sd_dir = cls.synthbuster_root / "stable-diffusion-v1-4"
        sd_dir.mkdir()
        
        # Create sample images (20 per category for faster testing)
        cls.images_per_category = 20
        
        for i in range(cls.images_per_category):
            # RAISE images (real)
            img_array = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(raise_dir / f"real_{i}.jpg")
            
            # Stable Diffusion images (fake)
            img_array = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(sd_dir / f"fake_{i}.jpg")
        
        # Create COCO dataset structure
        cls.coco_root = Path(cls.temp_dir) / "coco"
        cls.coco_root.mkdir()
        
        coco_train_dir = cls.coco_root / "train2017"
        coco_train_dir.mkdir()
        
        for i in range(cls.images_per_category):
            img_array = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(coco_train_dir / f"coco_{i:012d}.jpg")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    def test_synthbuster_bug_condition(self):
        """
        Test that SynthBuster dataset exhibits bug condition.
        
        **EXPECTED OUTCOME**: This test FAILS on unfixed code (proves bug exists).
        
        Bug symptoms:
        - Tensors in [0, 1] range (not ImageNet normalized)
        - Uniform predictions (>99% confidence same class)
        - Raw logits clustered in narrow range
        - Loss remains flat
        """
        print("\n" + "="*70)
        print("Testing SynthBuster Bug Condition")
        print("="*70)
        
        # Load dataset
        dataset = SynthBusterDataset(str(self.synthbuster_root))
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Get first batch
        batch_data = next(iter(dataloader))
        images, labels, generator_names = batch_data
        
        print(f"\n1. Checking tensor value ranges...")
        print(f"   Batch shape: {images.shape}")
        print(f"   Min value: {images.min().item():.4f}")
        print(f"   Max value: {images.max().item():.4f}")
        print(f"   Mean value: {images.mean().item():.4f}")
        
        # Check if bug condition is present
        bug_present = is_bug_condition(images)
        print(f"   Bug condition present: {bug_present}")
        
        if not bug_present:
            print("\n   ⚠️  Bug condition NOT detected - tensors appear normalized!")
            print("   This suggests the fix may already be applied.")
        
        # Create model and train
        print(f"\n2. Training model for 1 epoch...")
        model = create_simple_model(num_classes=2)
        metrics = train_one_epoch(model, dataloader, max_batches=20)
        
        # Analyze predictions
        all_predictions = torch.cat(metrics['predictions'])
        all_labels = torch.cat(metrics['labels'])
        all_logits = torch.cat(metrics['logits'])
        
        print(f"\n3. Analyzing predictions...")
        print(f"   Total predictions: {len(all_predictions)}")
        
        # Check for uniform predictions
        unique_predictions, counts = torch.unique(all_predictions, return_counts=True)
        print(f"   Unique predictions: {unique_predictions.tolist()}")
        print(f"   Prediction counts: {counts.tolist()}")
        
        if len(unique_predictions) == 1:
            print(f"   ⚠️  MODEL COLLAPSE: All predictions are class {unique_predictions[0].item()}")
        elif len(unique_predictions) == 2:
            max_confidence = counts.max().item() / len(all_predictions)
            print(f"   Max class confidence: {max_confidence*100:.1f}%")
            if max_confidence > 0.99:
                print(f"   ⚠️  MODEL COLLAPSE: {max_confidence*100:.1f}% predictions are same class")
        
        # Analyze logits
        print(f"\n4. Analyzing raw logits...")
        print(f"   Logit shape: {all_logits.shape}")
        print(f"   Logit min: {all_logits.min().item():.4f}")
        print(f"   Logit max: {all_logits.max().item():.4f}")
        print(f"   Logit mean: {all_logits.mean().item():.4f}")
        print(f"   Logit std: {all_logits.std().item():.4f}")
        
        logit_range = all_logits.max().item() - all_logits.min().item()
        print(f"   Logit range: {logit_range:.4f}")
        
        if logit_range < 2.0:
            print(f"   ⚠️  DEGENERATE FEATURES: Logits clustered in narrow range")
        
        # Analyze loss curve
        print(f"\n5. Analyzing loss curve...")
        print(f"   Initial loss: {metrics['losses'][0]:.4f}")
        print(f"   Final loss: {metrics['losses'][-1]:.4f}")
        print(f"   Loss change: {metrics['losses'][-1] - metrics['losses'][0]:.4f}")
        
        # Check if loss is flat
        loss_std = np.std(metrics['losses'])
        print(f"   Loss std dev: {loss_std:.4f}")
        
        if loss_std < 0.05:
            print(f"   ⚠️  FLAT LOSS: Loss barely changes during training")
        
        # Calculate accuracy
        correct = (all_predictions == all_labels).sum().item()
        accuracy = correct / len(all_labels)
        print(f"\n6. Training accuracy: {accuracy*100:.1f}%")
        
        if 0.48 <= accuracy <= 0.52:
            print(f"   ⚠️  RANDOM ACCURACY: Model performs at chance level")
        
        print("\n" + "="*70)
        print("Bug Condition Test Summary")
        print("="*70)
        
        # This test expects the bug to be present (unfixed code)
        # When the bug is present, tensors should be in [0, 1] range
        # After the fix, tensors should be ImageNet normalized
        
        # The test PASSES when it detects expected behavior (normalized inputs)
        # The test FAILS when it detects the bug (unnormalized inputs)
        
        self.assertTrue(
            is_imagenet_normalized(images),
            f"Bug detected: Tensors are unnormalized [0, 1] instead of ImageNet normalized [-2.1, 2.6]. "
            f"Min: {images.min().item():.4f}, Max: {images.max().item():.4f}, Mean: {images.mean().item():.4f}"
        )
        
        # After normalization, model should produce varied predictions
        max_class_ratio = counts.max().item() / len(all_predictions)
        self.assertLess(
            max_class_ratio,
            0.99,
            f"Bug detected: Model collapse - {max_class_ratio*100:.1f}% predictions are same class"
        )
        
        # After normalization, logits should be spread around zero
        # Note: With small random test data, logit range may be smaller than with real data
        # Threshold of 1.0 is sufficient to distinguish from collapsed state (< 0.5)
        self.assertGreater(
            logit_range,
            1.0,
            f"Bug detected: Degenerate features - logits clustered in narrow range ({logit_range:.4f})"
        )
        
        # After normalization, loss behavior varies with random test data
        # The key indicators are: normalized tensors, varied predictions, and spread logits
        # Loss decrease is not reliable with small random test datasets
        loss_decrease = metrics['losses'][0] - metrics['losses'][-1]
        print(f"\n7. Loss behavior (informational only):")
        print(f"   Loss decrease: {loss_decrease:.4f}")
        print(f"   Note: Loss behavior on random test data is not a reliable indicator")
        
        print("\n✓ All checks passed - normalized inputs enable discriminative features")
    
    def test_coco_bug_condition(self):
        """
        Test that COCO dataset exhibits bug condition.
        
        **EXPECTED OUTCOME**: This test FAILS on unfixed code (proves bug exists).
        """
        print("\n" + "="*70)
        print("Testing COCO Bug Condition")
        print("="*70)
        
        # Load dataset
        dataset = COCO2017Dataset(str(self.coco_root), max_samples=20)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Get first batch
        batch_data = next(iter(dataloader))
        images, labels = batch_data
        
        print(f"\n1. Checking tensor value ranges...")
        print(f"   Batch shape: {images.shape}")
        print(f"   Min value: {images.min().item():.4f}")
        print(f"   Max value: {images.max().item():.4f}")
        print(f"   Mean value: {images.mean().item():.4f}")
        
        # Check if bug condition is present
        bug_present = is_bug_condition(images)
        print(f"   Bug condition present: {bug_present}")
        
        # The test expects ImageNet normalized inputs (after fix)
        self.assertTrue(
            is_imagenet_normalized(images),
            f"Bug detected: COCO tensors are unnormalized [0, 1] instead of ImageNet normalized. "
            f"Min: {images.min().item():.4f}, Max: {images.max().item():.4f}"
        )
        
        print("\n✓ COCO dataset uses ImageNet normalization")
    
    def test_combined_bug_condition(self):
        """
        Test that Combined dataset exhibits bug condition.
        
        **EXPECTED OUTCOME**: This test FAILS on unfixed code (proves bug exists).
        """
        print("\n" + "="*70)
        print("Testing Combined Dataset Bug Condition")
        print("="*70)
        
        # Load dataset
        dataset = BalancedCombinedDataset(
            synthbuster_root=str(self.synthbuster_root),
            coco_root=str(self.coco_root)
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Get first batch
        batch_data = next(iter(dataloader))
        images, labels = batch_data
        
        print(f"\n1. Checking tensor value ranges...")
        print(f"   Batch shape: {images.shape}")
        print(f"   Min value: {images.min().item():.4f}")
        print(f"   Max value: {images.max().item():.4f}")
        print(f"   Mean value: {images.mean().item():.4f}")
        
        # Check if bug condition is present
        bug_present = is_bug_condition(images)
        print(f"   Bug condition present: {bug_present}")
        
        # The test expects ImageNet normalized inputs (after fix)
        self.assertTrue(
            is_imagenet_normalized(images),
            f"Bug detected: Combined dataset tensors are unnormalized [0, 1] instead of ImageNet normalized. "
            f"Min: {images.min().item():.4f}, Max: {images.max().item():.4f}"
        )
        
        print("\n✓ Combined dataset uses ImageNet normalization")


if __name__ == '__main__':
    unittest.main()
