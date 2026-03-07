"""
Training utilities for AI image detection model.

This module provides core training functions including epoch training, validation,
and checkpoint management for the binary classifier.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List, Dict
import random
from tqdm import tqdm
from .domain_adversarial import compute_domain_adversarial_loss

# TODO: Add spectral/frequency-domain feature training
# Implement training procedures that incorporate frequency domain analysis.
# Consider adding auxiliary losses on FFT/DCT features to encourage the model
# to learn spectral artifacts (e.g., checkerboard patterns, upsampling artifacts).
# This could include spectral reconstruction losses or frequency-based regularization.

# TODO: Add noise-based imprint training
# Implement training with noise residual extraction to learn generator-specific
# fingerprints. Consider adding contrastive learning or metric learning objectives
# to cluster images by generator type based on noise patterns. This can improve
# both detection and attribution capabilities.

# TODO: Add multi-dataset training support
# Implement training procedures that handle multiple datasets simultaneously.
# Consider domain adaptation techniques (e.g., domain-adversarial training, MMD),
# multi-task learning with dataset-specific heads, or curriculum learning that
# progressively introduces datasets. This improves generalization across diverse
# image sources and generation methods.

# TODO: Add robustness training
# Implement training with data augmentation specifically designed for robustness:
# - JPEG compression at varying quality levels
# - Gaussian blur and noise injection
# - Random resizing and cropping
# - Color jittering and contrast adjustments
# This helps the model maintain performance under real-world image perturbations
# and makes it more resistant to adversarial attacks.

# TODO: Add any-resolution training support
# Implement training procedures that handle variable input resolutions.
# Consider random resolution sampling during training, multi-scale training,
# or progressive resolution training. This enables the model to process images
# at their native resolution without quality-degrading resize operations.


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.BCELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cutmix_aug: Optional['CutMixAugmentation'] = None,
    mixup_aug: Optional['MixUpAugmentation'] = None,
    cutmix_prob: float = 0.0,
    mixup_prob: float = 0.0,
    domain_discriminator: Optional[nn.Module] = None,
    domain_lambda: float = 1.0,
    dataset_to_domain: Optional[Dict[str, int]] = None,
    epoch: int = 0,
    num_epochs: int = 1
) -> Tuple[float, float]:
    """
    Train model for one epoch with optional CutMix, MixUp, and domain adversarial training.
    
    Args:
        model: Neural network model
        dataloader: Training data loader (can be MultiDatasetLoader or regular DataLoader)
        criterion: BCE loss function
        optimizer: Optimizer for backpropagation
        device: Device to run training on (cpu/cuda)
        cutmix_aug: Optional CutMixAugmentation instance
        mixup_aug: Optional MixUpAugmentation instance
        cutmix_prob: Probability of applying CutMix (default: 0.0)
        mixup_prob: Probability of applying MixUp (default: 0.0)
        domain_discriminator: Optional domain discriminator for adversarial training
        domain_lambda: Gradient reversal strength for domain adversarial loss (default: 1.0)
        dataset_to_domain: Optional mapping from dataset names to domain indices (required for
                          domain adversarial training with MultiDatasetLoader)
        epoch: Current epoch number (for progress bar display)
        num_epochs: Total number of epochs (for progress bar display)
    
    Returns:
        Tuple of (average_loss, accuracy) for the epoch
    
    Note:
        When both CutMix and MixUp are enabled, they are applied mutually exclusively
        (only one augmentation per batch). CutMix is checked first.
        
        Domain adversarial training requires domain_discriminator to be provided.
        For MultiDatasetLoader, dataset_to_domain mapping is also required to convert
        dataset names to domain indices. For regular DataLoader with domain labels,
        the dataloader should yield (images, labels, domain_labels) tuples.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                leave=False, ncols=100)
    
    # Convert dataloader to iterator for sampling pairs
    dataloader_iter = iter(dataloader)
    
    for batch_idx, batch_data in enumerate(pbar):
        # Handle both regular DataLoader (images, labels) and MultiDatasetLoader (images, labels, dataset_name)
        if len(batch_data) == 3:
            images, labels, dataset_names = batch_data
            # Convert dataset names to domain labels if mapping provided
            if domain_discriminator is not None and dataset_to_domain is not None:
                # Handle both single string and list of strings
                if isinstance(dataset_names, str):
                    # Single dataset name for entire batch
                    domain_labels = torch.tensor(
                        [dataset_to_domain[dataset_names]] * images.size(0),
                        dtype=torch.long,
                        device=device
                    )
                else:
                    # List of dataset names (one per sample)
                    domain_labels = torch.tensor(
                        [dataset_to_domain[name] for name in dataset_names],
                        dtype=torch.long,
                        device=device
                    )
            else:
                domain_labels = None
        else:
            images, labels = batch_data
            domain_labels = None
        
        # Move data to device
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1) if labels.dim() == 1 else labels.to(device).float()
        
        # Apply CutMix or MixUp augmentation
        augmentation_applied = False
        
        # Try CutMix first
        if cutmix_aug is not None and cutmix_prob > 0 and random.random() < cutmix_prob:
            try:
                # Get another batch for mixing
                images2, labels2 = next(dataloader_iter)
                images2 = images2.to(device)
                labels2 = labels2.to(device).float().unsqueeze(1) if labels2.dim() == 1 else labels2.to(device).float()
                
                # Apply CutMix
                images, labels = cutmix_aug(images, labels, images2, labels2)
                augmentation_applied = True
            except StopIteration:
                # If we run out of batches, restart the iterator
                dataloader_iter = iter(dataloader)
        
        # Try MixUp if CutMix wasn't applied
        if not augmentation_applied and mixup_aug is not None and mixup_prob > 0 and random.random() < mixup_prob:
            try:
                # Get another batch for mixing
                images2, labels2 = next(dataloader_iter)
                images2 = images2.to(device)
                labels2 = labels2.to(device).float().unsqueeze(1) if labels2.dim() == 1 else labels2.to(device).float()
                
                # Apply MixUp
                images, labels = mixup_aug(images, labels, images2, labels2)
                augmentation_applied = True
            except StopIteration:
                # If we run out of batches, restart the iterator
                dataloader_iter = iter(dataloader)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute BCE loss
        loss = criterion(outputs, labels)
        
        # Add domain adversarial loss if enabled
        if domain_discriminator is not None and domain_labels is not None:
            # Extract features from the model for domain discrimination
            # We need to get features before the final classification layer
            # This requires a partial forward pass through the model
            
            # Get spatial features from backbone
            spatial_features = model.backbone(images)
            
            # Apply attention if enabled
            if model.attention_module is not None:
                spatial_features = model.attention_module(spatial_features)
            
            # Global pool to get feature vector
            pooled_features = model.global_pool(spatial_features)
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            
            # Compute domain adversarial loss
            domain_loss = compute_domain_adversarial_loss(
                pooled_features,
                domain_labels,
                domain_discriminator,
                domain_lambda
            )
            
            # Add domain loss to total loss
            loss = loss + domain_loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # For accuracy calculation with soft labels (from augmentation)
        # Use 0.5 threshold for predictions
        predictions = (outputs >= 0.5).float()
        
        # For soft labels, count as correct if prediction matches the dominant class
        # (i.e., if label > 0.5 and prediction is 1, or label <= 0.5 and prediction is 0)
        hard_labels = (labels >= 0.5).float()
        correct += (predictions == hard_labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        current_acc = correct / total if total > 0 else 0
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.BCELoss,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on validation set without gradient updates.
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: BCE loss function
        device: Device to run validation on (cpu/cuda)
    
    Returns:
        Tuple of (average_loss, accuracy) for the validation set
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            # Forward pass
            outputs = model(images)
            
            # Compute BCE loss
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
) -> None:
    """
    Save model checkpoint to disk.
    
    Args:
        model: Neural network model
        optimizer: Optimizer with current state
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path where checkpoint will be saved
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> int:
    """
    Load model checkpoint from disk with error handling.
    
    Args:
        filepath: Path to checkpoint file
        model: Neural network model to load state into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Epoch number from checkpoint
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    try:
        checkpoint = torch.load(filepath)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        
        return epoch
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
