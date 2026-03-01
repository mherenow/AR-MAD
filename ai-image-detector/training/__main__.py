"""
Training script for AI Image Detector.

This module implements the main training loop for the binary classifier on the
SynthBuster dataset. It handles data loading, model initialization, training,
validation, and checkpoint saving.

Usage:1
    python -m ai-image-detector.training --config configs/default_config.yaml
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from data.synthbuster_loader import SynthBusterDataset, create_train_val_split
from data.combined_loader import BalancedCombinedDataset, create_train_val_split_combined
from models.classifier import BinaryClassifier


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    """
    Train the model for one epoch.
    
    Args:
        model: BinaryClassifier model
        dataloader: Training data loader
        criterion: Loss function (BCELoss)
        optimizer: Optimizer (Adam)
        device: Device to train on (cuda/cpu)
        epoch: Current epoch number
        num_epochs: Total number of epochs
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                leave=False, ncols=100)
    
    for batch_data in pbar:
        # Handle both (images, labels) and (images, labels, generator_name) formats
        if len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            images, labels = batch_data
            
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)  # (B,) -> (B, 1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        current_acc = correct / total if total > 0 else 0
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on validation set.
    
    Args:
        model: BinaryClassifier model
        dataloader: Validation data loader
        criterion: Loss function (BCELoss)
        device: Device to validate on (cuda/cpu)
        epoch: Current epoch number
        num_epochs: Total number of epochs
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]  ', 
                leave=False, ncols=100)
    
    with torch.no_grad():
        for batch_data in pbar:
            # Handle both (images, labels) and (images, labels, generator_name) formats
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data
                
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Metrics
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            current_acc = correct / total if total > 0 else 0
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train AI Image Detector')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Set device with automatic detection
    device_config = config.get('device', 'auto')
    
    if device_config == 'auto':
        # Automatically detect best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using device: {device} (auto-detected)")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using device: {device} (auto-detected)")
        else:
            device = torch.device('cpu')
            print(f"Using device: {device} (auto-detected)")
            print("WARNING: CUDA not available, training on CPU will be slower")
    else:
        # Use explicitly configured device
        if device_config == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        elif device_config == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')
            print(f"Using device: {device}")
            print("WARNING: CUDA requested but not available, falling back to CPU")
        else:
            device = torch.device(device_config)
            print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize dataset based on mode
    print("Initializing dataset...")
    dataset_mode = config['dataset'].get('mode', 'synthbuster')
    
    if dataset_mode == 'combined':
        # Use combined balanced dataset (SynthBuster + COCO2017)
        print("Using COMBINED dataset mode (SynthBuster + COCO2017)")
        synthbuster_root = config['dataset']['synthbuster_root']
        coco_root = config['dataset']['coco_root']
        val_ratio = config['dataset'].get('val_ratio', 0.2)
        
        # Create train/val split
        train_dataset, val_dataset = create_train_val_split_combined(
            synthbuster_root=synthbuster_root,
            coco_root=coco_root,
            val_ratio=val_ratio,
            seed=42
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
    else:
        # Use SynthBuster only (original mode)
        print("Using SYNTHBUSTER-ONLY dataset mode")
        root_dir = config['dataset'].get('root_dir') or config['dataset'].get('synthbuster_root')
        
        # Create train/val split
        val_ratio = config['dataset'].get('val_ratio', 0.2)
        train_paths, val_paths = create_train_val_split(root_dir, val_ratio=val_ratio)
        print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
        
        # Create full dataset
        full_dataset = SynthBusterDataset(root_dir=root_dir)
        
        # Create train/val subsets
        # Map paths to indices
        path_to_idx = {sample['path']: idx for idx, sample in enumerate(full_dataset.samples)}
        train_indices = [path_to_idx[path] for path in train_paths if path in path_to_idx]
        val_indices = [path_to_idx[path] for path in val_paths if path in path_to_idx]
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['dataset'].get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Initialize model
    print("Initializing model...")
    backbone_type = config['model']['backbone_type']
    pretrained = config['model'].get('pretrained', True)
    
    model = BinaryClassifier(
        backbone_type=backbone_type,
        pretrained=pretrained
    )
    model = model.to(device)
    
    print(f"Model: BinaryClassifier with {backbone_type} backbone")
    print(f"Pretrained: {pretrained}")
    
    # Initialize optimizer
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0001)
    
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch, num_epochs
        )
        
        # Log metrics
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'config': config
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, best_model_path)
            print(f"  ⭐ New best model saved! (val_acc: {val_acc:.4f})")
        
        # Show GPU memory usage if using CUDA
        if device.type == 'cuda':
            print(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB (current/peak)")
            torch.cuda.reset_peak_memory_stats()
        
        print("-" * 70)
    
    print("=" * 70)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    
    # TODO: Add data augmentation
    # Implement augmentation pipeline (random crops, flips, color jitter, etc.)
    # to improve model generalization and robustness. Consider using albumentations
    # or torchvision.transforms for augmentation strategies.
    
    # TODO: Add early stopping
    # Implement early stopping mechanism to prevent overfitting by monitoring
    # validation loss/accuracy. Stop training if validation performance doesn't
    # improve for N consecutive epochs (e.g., patience=5).


if __name__ == '__main__':
    main()
