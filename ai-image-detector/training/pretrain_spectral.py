"""
Spectral branch pretraining script.

This module implements self-supervised pretraining for the spectral branch
using masked patch reconstruction. The pretrained weights can then be used
to initialize the spectral branch for downstream classification tasks.

Usage:
    python -m ai-image-detector.training --pretrain --config configs/enhanced_config.yaml
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spectral.spectral_branch import SpectralBranch
from models.spectral.pretraining import MaskedSpectralPretraining


def pretrain_epoch(
    model: MaskedSpectralPretraining,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> float:
    """
    Pretrain the spectral branch for one epoch.
    
    Args:
        model: MaskedSpectralPretraining model
        dataloader: Training data loader
        optimizer: Optimizer (AdamW)
        device: Device to train on (cuda/cpu)
        epoch: Current epoch number
        num_epochs: Total number of epochs
    
    Returns:
        Average reconstruction loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Create progress bar
    pbar = tqdm(
        dataloader,
        desc=f'Epoch {epoch+1}/{num_epochs} [Pretrain]',
        leave=False,
        ncols=100
    )
    
    for batch_data in pbar:
        # Handle both (images, labels) and (images, labels, generator_name) formats
        if len(batch_data) == 3:
            images, _, _ = batch_data
        else:
            images, _ = batch_data
        
        images = images.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        loss, pred, mask = model(images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_pretraining(
    model: MaskedSpectralPretraining,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> float:
    """
    Validate the pretraining model on validation set.
    
    Args:
        model: MaskedSpectralPretraining model
        dataloader: Validation data loader
        device: Device to validate on (cuda/cpu)
        epoch: Current epoch number
        num_epochs: Total number of epochs
    
    Returns:
        Average reconstruction loss for validation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Create progress bar
    pbar = tqdm(
        dataloader,
        desc=f'Epoch {epoch+1}/{num_epochs} [Val]    ',
        leave=False,
        ncols=100
    )
    
    with torch.no_grad():
        for batch_data in pbar:
            # Handle both (images, labels) and (images, labels, generator_name) formats
            if len(batch_data) == 3:
                images, _, _ = batch_data
            else:
                images, _ = batch_data
            
            images = images.to(device, non_blocking=True)
            
            # Forward pass
            loss, pred, mask = model(images)
            
            # Metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_spectral_checkpoint(
    spectral_branch: SpectralBranch,
    epoch: int,
    val_loss: float,
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """
    Save spectral branch checkpoint.
    
    Args:
        spectral_branch: SpectralBranch instance to save
        epoch: Current epoch number
        val_loss: Validation loss
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': spectral_branch.state_dict(),
        'val_loss': val_loss,
        'model_config': {
            'patch_size': spectral_branch.patch_size,
            'embed_dim': spectral_branch.embed_dim,
            'depth': spectral_branch.depth,
            'num_heads': spectral_branch.num_heads,
            'num_bands': spectral_branch.num_bands,
            'consistency_dim': spectral_branch.consistency_dim
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'spectral_pretrain_epoch_{epoch}.pth'
    )
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'spectral_pretrain_best.pth')
        torch.save(checkpoint, best_path)


def pretrain_spectral_branch(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> None:
    """
    Pretrain spectral branch using masked patch reconstruction.
    
    This function implements the main pretraining loop for the spectral branch.
    The pretrained weights can be loaded later for fine-tuning on downstream
    classification tasks.
    
    Args:
        config: Configuration dictionary containing:
            - spectral: Spectral branch configuration
            - pretraining: Pretraining configuration (decoder params, epochs, lr)
            - training: Training configuration (checkpoint_dir)
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cuda/cpu)
    
    Example:
        >>> config = load_config('configs/enhanced_config.yaml')
        >>> train_loader = DataLoader(train_dataset, batch_size=32)
        >>> val_loader = DataLoader(val_dataset, batch_size=32)
        >>> device = torch.device('cuda')
        >>> pretrain_spectral_branch(config, train_loader, val_loader, device)
    """
    print("\n" + "=" * 70)
    print("SPECTRAL BRANCH PRETRAINING")
    print("=" * 70)
    
    # Extract configurations
    spectral_config = config.get('spectral', {})
    pretraining_config = config.get('pretraining', {})
    training_config = config.get('training', {})
    
    # Initialize spectral branch
    print("\nInitializing spectral branch...")
    spectral_branch = SpectralBranch(
        patch_size=spectral_config.get('patch_size', 16),
        embed_dim=spectral_config.get('embed_dim', 256),
        depth=spectral_config.get('depth', 4),
        num_heads=spectral_config.get('num_heads', 8),
        mask_type=spectral_config.get('frequency_mask_type', 'high_pass'),
        cutoff_freq=spectral_config.get('cutoff_freq', 0.3),
        num_bands=spectral_config.get('num_bands', 4),
        consistency_dim=spectral_config.get('consistency_dim', 128)
    )
    
    print(f"  Patch size: {spectral_branch.patch_size}")
    print(f"  Embed dim: {spectral_branch.embed_dim}")
    print(f"  Depth: {spectral_branch.depth}")
    print(f"  Num heads: {spectral_branch.num_heads}")
    
    # Initialize pretraining model
    print("\nInitializing pretraining model...")
    pretraining_model = MaskedSpectralPretraining(
        spectral_branch=spectral_branch,
        decoder_embed_dim=pretraining_config.get('decoder_embed_dim', 128),
        decoder_depth=pretraining_config.get('decoder_depth', 2),
        mask_ratio=spectral_config.get('mask_ratio', 0.75),
        norm_pix_loss=pretraining_config.get('norm_pix_loss', True)
    )
    pretraining_model = pretraining_model.to(device)
    
    print(f"  Decoder embed dim: {pretraining_model.decoder_embed_dim}")
    print(f"  Decoder depth: {pretraining_model.decoder_depth}")
    print(f"  Mask ratio: {pretraining_model.mask_ratio}")
    
    # Count parameters
    total_params = sum(p.numel() for p in pretraining_model.parameters())
    trainable_params = sum(p.numel() for p in pretraining_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    learning_rate = pretraining_config.get('learning_rate', 0.001)
    weight_decay = pretraining_config.get('weight_decay', 0.05)
    
    optimizer = torch.optim.AdamW(
        pretraining_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )
    
    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    
    # Training loop
    num_epochs = pretraining_config.get('num_epochs', 100)
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    save_interval = pretraining_config.get('save_interval', 10)
    
    best_val_loss = float('inf')
    
    print(f"\nStarting pretraining for {num_epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        # Train
        train_loss = pretrain_epoch(
            pretraining_model,
            train_loader,
            optimizer,
            device,
            epoch,
            num_epochs
        )
        
        # Validate
        val_loss = validate_pretraining(
            pretraining_model,
            val_loader,
            device,
            epoch,
            num_epochs
        )
        
        # Log metrics
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save checkpoint at intervals
        if (epoch + 1) % save_interval == 0:
            save_spectral_checkpoint(
                spectral_branch,
                epoch + 1,
                val_loss,
                checkpoint_dir,
                is_best=False
            )
            print(f"  💾 Checkpoint saved at epoch {epoch+1}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_spectral_checkpoint(
                spectral_branch,
                epoch + 1,
                val_loss,
                checkpoint_dir,
                is_best=True
            )
            print(f"  ⭐ New best model saved! (val_loss: {val_loss:.4f})")
        
        # Show GPU memory usage if using CUDA
        if device.type == 'cuda':
            print(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / "
                  f"{torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB (current/peak)")
            torch.cuda.reset_peak_memory_stats()
        
        print("-" * 70)
    
    print("=" * 70)
    print(f"Pretraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'spectral_pretrain_best.pth')}")
    print("\nTo use pretrained weights in classification:")
    print("  1. Load checkpoint: checkpoint = torch.load('spectral_pretrain_best.pth')")
    print("  2. Initialize spectral branch with same config")
    print("  3. Load weights: spectral_branch.load_state_dict(checkpoint['model_state_dict'])")
    print("=" * 70)
