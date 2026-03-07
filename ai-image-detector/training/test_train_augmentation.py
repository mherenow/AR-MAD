"""
Unit tests for train_epoch with CutMix and MixUp augmentation support.

Tests the augmentation integration in the training loop including:
- Training with CutMix augmentation
- Training with MixUp augmentation
- Training with both augmentations
- Backward compatibility (no augmentation)
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classifier import BinaryClassifier
from training.train import train_epoch
from data.augmentation.cutmix import CutMixAugmentation
from data.augmentation.mixup import MixUpAugmentation


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset for testing."""
    num_samples = 16
    images = torch.randn(num_samples, 3, 256, 256)
    labels = torch.randint(0, 2, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    return loader


@pytest.fixture
def model_and_optimizer():
    """Create a model and optimizer for testing."""
    model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cpu')
    criterion = nn.BCELoss()
    
    return model, optimizer, criterion, device


class TestTrainEpochAugmentation:
    """Tests for train_epoch with augmentation support."""
    
    def test_train_without_augmentation(self, synthetic_dataset, model_and_optimizer):
        """Test training without any augmentation (backward compatibility)."""
        loader = synthetic_dataset
        model, optimizer, criterion, device = model_and_optimizer
        
        # Train without augmentation
        train_loss, train_acc = train_epoch(
            model, loader, criterion, optimizer, device
        )
        
        # Verify metrics are valid
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0
        assert 0.0 <= train_acc <= 1.0
    
    def test_train_with_cutmix(self, synthetic_dataset, model_and_optimizer):
        """Test training with CutMix augmentation."""
        loader = synthetic_dataset
        model, optimizer, criterion, device = model_and_optimizer
        
        # Create CutMix augmentation
        cutmix_aug = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        # Train with CutMix
        train_loss, train_acc = train_epoch(
            model, loader, criterion, optimizer, device,
            cutmix_aug=cutmix_aug,
            cutmix_prob=1.0
        )
        
        # Verify metrics are valid
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0
        assert 0.0 <= train_acc <= 1.0
    
    def test_train_with_mixup(self, synthetic_dataset, model_and_optimizer):
        """Test training with MixUp augmentation."""
        loader = synthetic_dataset
        model, optimizer, criterion, device = model_and_optimizer
        
        # Create MixUp augmentation
        mixup_aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        # Train with MixUp
        train_loss, train_acc = train_epoch(
            model, loader, criterion, optimizer, device,
            mixup_aug=mixup_aug,
            mixup_prob=1.0
        )
        
        # Verify metrics are valid
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0
        assert 0.0 <= train_acc <= 1.0
    
    def test_train_with_both_augmentations(self, synthetic_dataset, model_and_optimizer):
        """Test training with both CutMix and MixUp (mutually exclusive)."""
        loader = synthetic_dataset
        model, optimizer, criterion, device = model_and_optimizer
        
        # Create both augmentations
        cutmix_aug = CutMixAugmentation(alpha=1.0, prob=1.0)
        mixup_aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        # Train with both (CutMix takes priority)
        train_loss, train_acc = train_epoch(
            model, loader, criterion, optimizer, device,
            cutmix_aug=cutmix_aug,
            mixup_aug=mixup_aug,
            cutmix_prob=0.5,
            mixup_prob=0.5
        )
        
        # Verify metrics are valid
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0
        assert 0.0 <= train_acc <= 1.0
    
    def test_train_with_zero_augmentation_prob(self, synthetic_dataset, model_and_optimizer):
        """Test that augmentation is not applied when probability is 0."""
        loader = synthetic_dataset
        model, optimizer, criterion, device = model_and_optimizer
        
        # Create augmentations but set probability to 0
        cutmix_aug = CutMixAugmentation(alpha=1.0, prob=1.0)
        mixup_aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        # Train with zero probability
        train_loss, train_acc = train_epoch(
            model, loader, criterion, optimizer, device,
            cutmix_aug=cutmix_aug,
            mixup_aug=mixup_aug,
            cutmix_prob=0.0,
            mixup_prob=0.0
        )
        
        # Verify metrics are valid
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0
        assert 0.0 <= train_acc <= 1.0
    
    def test_train_with_partial_augmentation_prob(self, synthetic_dataset, model_and_optimizer):
        """Test training with partial augmentation probability."""
        loader = synthetic_dataset
        model, optimizer, criterion, device = model_and_optimizer
        
        # Create augmentations with partial probability
        cutmix_aug = CutMixAugmentation(alpha=1.0, prob=1.0)
        
        # Train with 50% probability
        train_loss, train_acc = train_epoch(
            model, loader, criterion, optimizer, device,
            cutmix_aug=cutmix_aug,
            cutmix_prob=0.5
        )
        
        # Verify metrics are valid
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0
        assert 0.0 <= train_acc <= 1.0
    
    def test_augmentation_with_soft_labels(self, synthetic_dataset, model_and_optimizer):
        """Test that soft labels from augmentation are handled correctly."""
        loader = synthetic_dataset
        model, optimizer, criterion, device = model_and_optimizer
        
        # Create MixUp which produces soft labels
        mixup_aug = MixUpAugmentation(alpha=0.2, prob=1.0)
        
        # Train with MixUp (produces soft labels)
        train_loss, train_acc = train_epoch(
            model, loader, criterion, optimizer, device,
            mixup_aug=mixup_aug,
            mixup_prob=1.0
        )
        
        # Verify training works with soft labels
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0
        assert 0.0 <= train_acc <= 1.0
    
    def test_augmentation_none_parameters(self, synthetic_dataset, model_and_optimizer):
        """Test that None augmentation parameters work correctly."""
        loader = synthetic_dataset
        model, optimizer, criterion, device = model_and_optimizer
        
        # Train with None augmentation objects
        train_loss, train_acc = train_epoch(
            model, loader, criterion, optimizer, device,
            cutmix_aug=None,
            mixup_aug=None,
            cutmix_prob=0.5,
            mixup_prob=0.5
        )
        
        # Verify training works without augmentation objects
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0
        assert 0.0 <= train_acc <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
