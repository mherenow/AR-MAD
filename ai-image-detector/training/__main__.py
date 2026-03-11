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
# from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from data.synthbuster_loader import SynthBusterDataset, create_train_val_split
from data.combined_loader import BalancedCombinedDataset, create_train_val_split_combined
from models.classifier import BinaryClassifier


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs, 
                use_amp=False, scaler=None, grad_accum_steps=1):
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
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for mixed precision training
        grad_accum_steps: Number of gradient accumulation steps
        
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
    
    optimizer.zero_grad()
    
    for batch_idx, batch_data in enumerate(pbar):
        # Handle both (images, labels) and (images, labels, generator_name) formats
        if len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            images, labels = batch_data
            
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)  # (B,) -> (B, 1)
        
        # Forward pass with optional mixed precision
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Metrics (use unscaled loss for logging)
        total_loss += loss.item() * grad_accum_steps
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        current_acc = correct / total if total > 0 else 0
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}', 'acc': f'{current_acc:.4f}'})
    
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
    parser.add_argument(
        '--pretrain',
        action='store_true',
        help='Run spectral branch pretraining instead of classification training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Route to pretraining if --pretrain flag is set
    if args.pretrain:
        from training.pretrain_spectral import pretrain_spectral_branch
        
        # Set device
        device_config = config.get('device', 'auto')
        device = _get_device(device_config)
        
        # Initialize dataset and data loaders
        train_loader, val_loader, _, _ = _create_data_loaders(config, device)
        
        # Run pretraining
        pretrain_spectral_branch(config, train_loader, val_loader, device)
        return
    
    # Otherwise, run normal classification training
    _run_classification_training(config, args.resume)


def _get_device(device_config: str) -> torch.device:
    """
    Get device based on configuration.
    
    Args:
        device_config: Device configuration ('auto', 'cuda', 'cpu', 'mps')
    
    Returns:
        torch.device instance
    """
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
    
    return device


def _create_data_loaders(config: dict, device: torch.device):
    """
    Create train and validation data loaders.
    
    Args:
        config: Configuration dictionary
        device: Device for training
    
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
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
    
    # Performance optimizations
    use_cuda = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
        drop_last=True  # Drop incomplete batches for consistent performance
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def _run_classification_training(config: dict, resume_checkpoint: str = None):
    """
    Run normal classification training.
    
    Args:
        config: Configuration dictionary
        resume_checkpoint: Path to checkpoint file to resume from (optional)
    """
    # Set device with automatic detection
    device_config = config.get('device', 'auto')
    device = _get_device(device_config)
    
    # Create checkpoint directory
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, train_dataset, val_dataset = _create_data_loaders(config, device)
    
    # Initialize model
    print("Initializing model...")
    backbone_type = config['model']['backbone_type']
    pretrained = config['model'].get('pretrained', True)
    
    model = BinaryClassifier(
        backbone_type=backbone_type,
        pretrained=pretrained
    )
    model = model.to(device)
    
    # Compile model for PyTorch 2.0+ (10-30% speedup)
    if config['training'].get('compile_model', False):
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
            print("✓ Model compiled")
        else:
            print("⚠ torch.compile() not available (requires PyTorch 2.0+)")
    
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
    
    # Initialize augmentation modules if enabled
    cutmix_aug = None
    mixup_aug = None
    cutmix_prob = 0.0
    mixup_prob = 0.0
    
    augmentation_config = config.get('augmentation', {})
    
    if augmentation_config.get('cutmix', {}).get('enabled', False):
        from data.augmentation.cutmix import CutMixAugmentation
        cutmix_config = augmentation_config['cutmix']
        cutmix_aug = CutMixAugmentation(
            alpha=cutmix_config.get('alpha', 1.0),
            prob=cutmix_config.get('prob', 0.5)
        )
        cutmix_prob = cutmix_config.get('prob', 0.5)
        print(f"CutMix augmentation enabled (alpha={cutmix_config.get('alpha', 1.0)}, prob={cutmix_prob})")
    
    if augmentation_config.get('mixup', {}).get('enabled', False):
        from data.augmentation.mixup import MixUpAugmentation
        mixup_config = augmentation_config['mixup']
        mixup_aug = MixUpAugmentation(
            alpha=mixup_config.get('alpha', 0.2),
            prob=mixup_config.get('prob', 0.5)
        )
        mixup_prob = mixup_config.get('prob', 0.5)
        print(f"MixUp augmentation enabled (alpha={mixup_config.get('alpha', 0.2)}, prob={mixup_prob})")
    
    # Initialize domain adversarial training if enabled
    domain_discriminator = None
    domain_optimizer = None
    domain_lambda = 1.0
    dataset_to_domain = None
    
    domain_config = config.get('training', {}).get('domain_adversarial', {})
    if domain_config.get('enabled', False):
        from training.domain_adversarial import DomainDiscriminator
        
        # Determine feature dimension based on backbone
        if backbone_type == 'simple_cnn':
            feature_dim = 512
        elif backbone_type in ['resnet18', 'resnet34']:
            feature_dim = 512
        elif backbone_type in ['resnet50', 'resnet101', 'resnet152']:
            feature_dim = 2048
        else:
            feature_dim = 512  # Default
        
        # Get number of domains from dataset configuration
        dataset_config = config.get('data', {}).get('datasets', {})
        num_domains = len(dataset_config) if dataset_config else 2
        
        # Create domain discriminator
        hidden_dim = domain_config.get('hidden_dim', 256)
        domain_discriminator = DomainDiscriminator(
            feature_dim=feature_dim,
            num_domains=num_domains,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Create separate optimizer for domain discriminator
        domain_optimizer = Adam(
            domain_discriminator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        domain_lambda = domain_config.get('lambda', 1.0)
        
        # Create dataset to domain mapping
        if dataset_config:
            dataset_to_domain = {name: idx for idx, name in enumerate(dataset_config.keys())}
        else:
            # Default mapping for combined dataset mode
            dataset_to_domain = {'synthbuster': 0, 'coco2017': 1}
        
        print(f"Domain adversarial training enabled (lambda={domain_lambda}, num_domains={num_domains})")
        print(f"Dataset to domain mapping: {dataset_to_domain}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision training setup
    use_amp = config['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    if use_amp:
        print(f"✓ Mixed precision training enabled (AMP)")
    if grad_accum_steps > 1:
        print(f"✓ Gradient accumulation enabled ({grad_accum_steps} steps)")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {resume_checkpoint}")
        
        print(f"\nLoading checkpoint from {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model state loaded")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
        
        # Load domain discriminator state if present
        if domain_discriminator is not None and 'domain_discriminator_state_dict' in checkpoint:
            domain_discriminator.load_state_dict(checkpoint['domain_discriminator_state_dict'])
            print("✓ Domain discriminator state loaded")
        
        if domain_optimizer is not None and 'domain_optimizer_state_dict' in checkpoint:
            domain_optimizer.load_state_dict(checkpoint['domain_optimizer_state_dict'])
            print("✓ Domain optimizer state loaded")
        
        # Resume from next epoch
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('val_acc', 0.0)
        
        print(f"✓ Resuming from epoch {start_epoch} (best val_acc: {best_val_acc:.4f})")
    
    # Verify labels before training - print 10 samples
    print("\n" + "=" * 70)
    print("LABEL VERIFICATION - First 10 Training Samples")
    print("=" * 70)
    print("Checking for label flips (Label 0=REAL, Label 1=FAKE)...\n")
    
    # Get samples directly from dataset to show paths
    num_samples_to_show = min(10, len(train_dataset))
    
    for i in range(num_samples_to_show):
        try:
            sample = train_dataset[i]
            
            # Handle different return formats
            if len(sample) == 3:
                image, label, path = sample
            elif len(sample) == 2:
                image, label = sample
                path = None
            else:
                continue
            
            # Convert label to int
            label = label.item() if isinstance(label, torch.Tensor) else int(label)
            label_name = "REAL" if label == 0 else "FAKE"
            
            # Check pixel variance for FAKE images to detect blank/corrupted images
            if label == 1:
                variance = image.var().item()
                variance_status = "✓" if variance > 0.01 else "⚠ LOW"
            
            # Try to get path from underlying dataset
            if path is None:
                # For combined dataset, try to get source info
                if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'all_samples'):
                    # This is a Subset of BalancedCombinedDataset
                    actual_idx = train_dataset.indices[i]
                    source, source_idx, _ = train_dataset.dataset.all_samples[actual_idx]
                    
                    if source == 'coco':
                        path = f"coco2017/train/{source_idx}"
                    elif source == 'synthbuster_real':
                        if hasattr(train_dataset.dataset, 'synthbuster_real_samples'):
                            path = train_dataset.dataset.synthbuster_real_samples[source_idx].get('path', 'N/A')
                    elif source == 'synthbuster_fake':
                        if hasattr(train_dataset.dataset, 'synthbuster_fake_samples'):
                            path = train_dataset.dataset.synthbuster_fake_samples[source_idx].get('path', 'N/A')
                elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'samples'):
                    # This is a Subset of SynthBusterDataset
                    actual_idx = train_dataset.indices[i]
                    path = train_dataset.dataset.samples[actual_idx].get('path', 'N/A')
            
            # Format path for display
            if path and isinstance(path, str):
                path_parts = Path(path).parts
                short_path = "/".join(path_parts[-2:]) if len(path_parts) >= 2 else Path(path).name
            else:
                short_path = "N/A"
            
            # Print with variance info for FAKE images
            if label == 1:
                print(f"  Sample {i+1}: Label={label} ({label_name:4s}) | Variance={variance:.4f} {variance_status} | Path=.../{short_path}")
            else:
                print(f"  Sample {i+1}: Label={label} ({label_name:4s}) | Path=.../{short_path}")
            
        except Exception as e:
            print(f"  Sample {i+1}: Error reading sample - {e}")
    
    print("\n✓ Label verification complete")
    print("=" * 70)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(start_epoch, num_epochs):
        # Check if we need augmentation/domain adversarial training
        if cutmix_aug or mixup_aug or domain_discriminator:
            # Use enhanced training with augmentation
            from training.train import train_epoch as train_epoch_enhanced
            
            train_loss, train_acc = train_epoch_enhanced(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                cutmix_aug=cutmix_aug,
                mixup_aug=mixup_aug,
                cutmix_prob=cutmix_prob,
                mixup_prob=mixup_prob,
                domain_discriminator=domain_discriminator,
                domain_lambda=domain_lambda,
                dataset_to_domain=dataset_to_domain,
                epoch=epoch,
                num_epochs=num_epochs
            )
        else:
            # Use simple optimized training
            train_loss, train_acc = train_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=num_epochs,
                use_amp=use_amp,
                scaler=scaler,
                grad_accum_steps=grad_accum_steps
            )
        
        # Update domain discriminator optimizer if enabled
        if domain_optimizer is not None:
            domain_optimizer.step()
            domain_optimizer.zero_grad()
        
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
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'config': config
        }
        
        # Add domain discriminator state if enabled
        if domain_discriminator is not None:
            checkpoint_data['domain_discriminator_state_dict'] = domain_discriminator.state_dict()
            checkpoint_data['domain_optimizer_state_dict'] = domain_optimizer.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            best_checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }
            
            # Add domain discriminator state if enabled
            if domain_discriminator is not None:
                best_checkpoint_data['domain_discriminator_state_dict'] = domain_discriminator.state_dict()
                best_checkpoint_data['domain_optimizer_state_dict'] = domain_optimizer.state_dict()
            
            torch.save(best_checkpoint_data, best_model_path)
            print(f"New best model saved! (val_acc: {val_acc:.4f})")
        
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
