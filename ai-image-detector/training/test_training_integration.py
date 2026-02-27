"""
Integration tests for the training module.

Tests the complete training pipeline including:
- Training on small synthetic datasets
- Checkpoint saving and loading
- Training resumption from checkpoints
- End-to-end training workflow
"""

import os
import tempfile
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classifier import BinaryClassifier
from training.train import (
    train_epoch,
    validate,
    save_checkpoint,
    load_checkpoint
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def synthetic_dataset():
    """
    Create a small synthetic dataset for testing.
    
    Returns:
        Tuple of (train_loader, val_loader) with synthetic image data
    """
    # Create synthetic images (batch_size=8, channels=3, height=256, width=256)
    num_train_samples = 16
    num_val_samples = 8
    
    # Generate random images
    train_images = torch.randn(num_train_samples, 3, 256, 256)
    train_labels = torch.randint(0, 2, (num_train_samples,))
    
    val_images = torch.randn(num_val_samples, 3, 256, 256)
    val_labels = torch.randint(0, 2, (num_val_samples,))
    
    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    return train_loader, val_loader


@pytest.fixture
def model_and_optimizer():
    """Create a model and optimizer for testing."""
    model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cpu')
    
    return model, optimizer, device


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_train_on_synthetic_dataset(self, synthetic_dataset, model_and_optimizer):
        """Test training on a small synthetic dataset."""
        train_loader, val_loader = synthetic_dataset
        model, optimizer, device = model_and_optimizer
        criterion = nn.BCELoss()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Verify metrics are valid
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_loss <= 10.0  # Loss should be reasonable
        assert 0.0 <= train_acc <= 1.0  # Accuracy should be between 0 and 1
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Verify validation metrics
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert 0.0 <= val_loss <= 10.0
        assert 0.0 <= val_acc <= 1.0
    
    def test_checkpoint_saving_and_loading(self, temp_dir, model_and_optimizer):
        """Test checkpoint saving and loading functionality."""
        model, optimizer, device = model_and_optimizer
        
        # Save initial state
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        epoch = 5
        loss = 0.123
        
        save_checkpoint(model, optimizer, epoch, loss, checkpoint_path)
        
        # Verify checkpoint file exists
        assert os.path.exists(checkpoint_path)
        
        # Create new model and optimizer
        new_model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        # Load checkpoint
        loaded_epoch = load_checkpoint(checkpoint_path, new_model, new_optimizer)
        
        # Verify epoch was loaded correctly
        assert loaded_epoch == epoch
        
        # Verify model states match
        for param1, param2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(param1, param2)
        
        # Verify optimizer states match
        assert len(optimizer.state_dict()['state']) == len(new_optimizer.state_dict()['state'])
    
    def test_checkpoint_loading_without_optimizer(self, temp_dir, model_and_optimizer):
        """Test loading checkpoint without optimizer state."""
        model, optimizer, device = model_and_optimizer
        
        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        save_checkpoint(model, optimizer, epoch=3, loss=0.456, filepath=checkpoint_path)
        
        # Create new model
        new_model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        
        # Load checkpoint without optimizer
        loaded_epoch = load_checkpoint(checkpoint_path, new_model, optimizer=None)
        
        # Verify epoch and model state
        assert loaded_epoch == 3
        for param1, param2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(param1, param2)
    
    def test_checkpoint_loading_nonexistent_file(self, model_and_optimizer):
        """Test error handling when loading non-existent checkpoint."""
        model, _, _ = model_and_optimizer
        
        with pytest.raises(FileNotFoundError):
            load_checkpoint('nonexistent_checkpoint.pth', model)
    
    def test_training_resumption_from_checkpoint(
        self, temp_dir, synthetic_dataset, model_and_optimizer
    ):
        """Test resuming training from a saved checkpoint."""
        train_loader, val_loader = synthetic_dataset
        model, optimizer, device = model_and_optimizer
        criterion = nn.BCELoss()
        
        # Train for 2 epochs
        for epoch in range(2):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
        
        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, 'resume_checkpoint.pth')
        save_checkpoint(model, optimizer, epoch=2, loss=train_loss, filepath=checkpoint_path)
        
        # Get model state after 2 epochs
        state_after_2_epochs = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Create new model and load checkpoint
        resumed_model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.001)
        
        loaded_epoch = load_checkpoint(checkpoint_path, resumed_model, resumed_optimizer)
        assert loaded_epoch == 2
        
        # Verify resumed model matches saved state
        for key in state_after_2_epochs:
            assert torch.allclose(
                state_after_2_epochs[key],
                resumed_model.state_dict()[key]
            )
        
        # Continue training for 1 more epoch
        train_loss_resumed, train_acc_resumed = train_epoch(
            resumed_model, train_loader, criterion, resumed_optimizer, device
        )
        
        # Verify training continues successfully
        assert isinstance(train_loss_resumed, float)
        assert isinstance(train_acc_resumed, float)
        assert 0.0 <= train_loss_resumed <= 10.0
        assert 0.0 <= train_acc_resumed <= 1.0
    
    def test_multi_epoch_training_convergence(self, synthetic_dataset, model_and_optimizer):
        """Test that model loss decreases over multiple epochs."""
        train_loader, val_loader = synthetic_dataset
        model, optimizer, device = model_and_optimizer
        criterion = nn.BCELoss()
        
        losses = []
        num_epochs = 5
        
        # Train for multiple epochs
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            losses.append(train_loss)
        
        # Verify we have all losses
        assert len(losses) == num_epochs
        
        # Check that loss generally decreases (allowing some fluctuation)
        # Compare first and last epoch
        assert losses[-1] <= losses[0] * 1.5  # Allow 50% tolerance
    
    def test_checkpoint_contains_all_required_fields(self, temp_dir, model_and_optimizer):
        """Test that saved checkpoint contains all required fields."""
        model, optimizer, device = model_and_optimizer
        
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        epoch = 10
        loss = 0.789
        
        save_checkpoint(model, optimizer, epoch, loss, checkpoint_path)
        
        # Load checkpoint manually to inspect contents
        checkpoint = torch.load(checkpoint_path)
        
        # Verify all required fields are present
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'loss' in checkpoint
        
        # Verify field values
        assert checkpoint['epoch'] == epoch
        assert checkpoint['loss'] == loss
        assert isinstance(checkpoint['model_state_dict'], dict)
        assert isinstance(checkpoint['optimizer_state_dict'], dict)
    
    def test_training_with_different_backbones(self, synthetic_dataset):
        """Test training with different backbone architectures."""
        train_loader, val_loader = synthetic_dataset
        criterion = nn.BCELoss()
        device = torch.device('cpu')
        
        backbones = ['simple_cnn', 'resnet18', 'resnet50']
        
        for backbone_type in backbones:
            # Create model with specific backbone
            model = BinaryClassifier(backbone_type=backbone_type, pretrained=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train for one epoch
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Verify training works for this backbone
            assert isinstance(train_loss, float)
            assert isinstance(train_acc, float)
            assert 0.0 <= train_loss <= 10.0
            assert 0.0 <= train_acc <= 1.0
    
    def test_checkpoint_compatibility_across_models(self, temp_dir):
        """Test that checkpoints can be loaded into models with same architecture."""
        device = torch.device('cpu')
        
        # Create and save checkpoint with first model
        model1 = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        
        checkpoint_path = os.path.join(temp_dir, 'compat_checkpoint.pth')
        save_checkpoint(model1, optimizer1, epoch=1, loss=0.5, filepath=checkpoint_path)
        
        # Create second model with same architecture
        model2 = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        
        # Load checkpoint into second model
        loaded_epoch = load_checkpoint(checkpoint_path, model2, optimizer2)
        
        # Verify models have identical weights
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(param1, param2)
        
        assert loaded_epoch == 1


class TestEndToEndTraining:
    """End-to-end integration tests for complete training workflow."""
    
    def test_complete_training_workflow(self, temp_dir, synthetic_dataset):
        """Test complete training workflow from initialization to final checkpoint."""
        train_loader, val_loader = synthetic_dataset
        device = torch.device('cpu')
        
        # Initialize model
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training configuration
        num_epochs = 3
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(
                checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
            save_checkpoint(model, optimizer, epoch+1, val_loss, checkpoint_path)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch+1, val_loss, best_checkpoint_path)
        
        # Verify all checkpoints were created
        assert os.path.exists(os.path.join(checkpoint_dir, 'checkpoint_epoch_1.pth'))
        assert os.path.exists(os.path.join(checkpoint_dir, 'checkpoint_epoch_2.pth'))
        assert os.path.exists(os.path.join(checkpoint_dir, 'checkpoint_epoch_3.pth'))
        assert os.path.exists(os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Load best model and verify
        final_model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        load_checkpoint(
            os.path.join(checkpoint_dir, 'best_model.pth'),
            final_model
        )
        
        # Verify final model can make predictions
        val_loss_final, val_acc_final = validate(
            final_model, val_loader, criterion, device
        )
        assert isinstance(val_loss_final, float)
        assert isinstance(val_acc_final, float)
    
    def test_training_interruption_and_resume(self, temp_dir, synthetic_dataset):
        """Test interrupting training and resuming from checkpoint."""
        train_loader, val_loader = synthetic_dataset
        device = torch.device('cpu')
        criterion = nn.BCELoss()
        
        # Phase 1: Initial training (simulate interruption after 2 epochs)
        model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(2):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
        
        # Save checkpoint before "interruption"
        interrupt_checkpoint = os.path.join(temp_dir, 'interrupt_checkpoint.pth')
        save_checkpoint(model, optimizer, epoch=2, loss=train_loss, filepath=interrupt_checkpoint)
        
        # Phase 2: Resume training (simulate restart)
        resumed_model = BinaryClassifier(backbone_type='simple_cnn', pretrained=False)
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.001)
        
        start_epoch = load_checkpoint(interrupt_checkpoint, resumed_model, resumed_optimizer)
        assert start_epoch == 2
        
        # Continue training for 2 more epochs
        for epoch in range(start_epoch, start_epoch + 2):
            train_loss, train_acc = train_epoch(
                resumed_model, train_loader, criterion, resumed_optimizer, device
            )
        
        # Save final checkpoint
        final_checkpoint = os.path.join(temp_dir, 'final_checkpoint.pth')
        save_checkpoint(
            resumed_model, resumed_optimizer, epoch=4, loss=train_loss, filepath=final_checkpoint
        )
        
        # Verify final checkpoint
        assert os.path.exists(final_checkpoint)
        final_epoch = load_checkpoint(final_checkpoint, resumed_model)
        assert final_epoch == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
