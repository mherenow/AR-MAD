"""
Combined Dataset Loader for Balanced Training

This module provides a combined dataset that balances real images from multiple sources
(COCO 2017 and SynthBuster RAISE) with AI-generated images from SynthBuster generators.

The balanced approach ensures equal representation of real and fake images during training,
which is critical for binary classification performance.
"""

import warnings
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms

from .coco_loader import COCO2017Dataset
from .synthbuster_loader import SynthBusterDataset


class BalancedCombinedDataset(Dataset):
    """
    Combined dataset that balances real and AI-generated images.
    
    This dataset combines:
    - Real images: COCO 2017 train set + SynthBuster RAISE folder
    - Fake images: SynthBuster AI generator folders
    
    The dataset automatically balances by limiting each category to have equal samples.
    
    Args:
        synthbuster_root: Path to SynthBuster dataset root
        coco_root: Path to COCO 2017 dataset root
        transform: Optional transform to apply to images
        balance_mode: How to balance datasets ('equal' or 'min')
            - 'equal': Sample equal amounts from real and fake
            - 'min': Use minimum of (real_count, fake_count) for both
        
    Returns:
        Tuple of (image_tensor, label) where:
        - image_tensor: torch.Tensor of shape (3, 256, 256)
        - label: int (0 for real, 1 for fake)
    """
    
    def __init__(
        self,
        synthbuster_root: str,
        coco_root: str,
        transform: Optional[transforms.Compose] = None,
        balance_mode: str = 'equal',
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize the combined balanced dataset.
        
        Args:
            synthbuster_root: Path to SynthBuster dataset root directory
            coco_root: Path to COCO 2017 dataset root directory
            transform: Optional custom transform
            balance_mode: 'equal' or 'min' for balancing strategy
            shuffle: Whether to shuffle the samples (default: True)
            seed: Random seed for shuffling (default: 42)
        """
        self.synthbuster_root = Path(synthbuster_root)
        self.coco_root = Path(coco_root)
        self.transform = transform
        self.balance_mode = balance_mode
        self.shuffle = shuffle
        self.seed = seed
        
        # Default transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Build the combined dataset
        self._build_combined_dataset()
        
    def _build_combined_dataset(self):
        """
        Build the combined dataset with balanced real and fake images.
        """
        print("\n" + "="*70)
        print("Building Balanced Combined Dataset")
        print("="*70)
        
        # First, load SynthBuster to count real (RAISE) and fake images
        print("\n1. Loading SynthBuster dataset...")
        synthbuster_full = SynthBusterDataset(
            root_dir=str(self.synthbuster_root),
            transform=self.transform
        )
        
        # Count real (RAISE) and fake images in SynthBuster
        synthbuster_real_count = sum(1 for s in synthbuster_full.samples if s['label'] == 0)
        synthbuster_fake_count = sum(1 for s in synthbuster_full.samples if s['label'] == 1)
        
        print(f"   SynthBuster RAISE (real): {synthbuster_real_count} images")
        print(f"   SynthBuster AI (fake): {synthbuster_fake_count} images")
        
        # Load COCO dataset
        print("\n2. Loading COCO 2017 dataset...")
        # Don't limit COCO yet, we'll balance after counting
        coco_full = COCO2017Dataset(
            root_dir=str(self.coco_root),
            transform=self.transform,
            max_samples=None  # Load all first to count
        )
        coco_count = len(coco_full.samples)
        print(f"   COCO 2017 (real): {coco_count} images available")
        
        # Calculate balanced counts
        total_real_available = synthbuster_real_count + coco_count
        total_fake_available = synthbuster_fake_count
        
        print(f"\n3. Calculating balanced split...")
        print(f"   Total real images available: {total_real_available}")
        print(f"   Total fake images available: {total_fake_available}")
        
        if self.balance_mode == 'equal':
            # Use the minimum to ensure equal representation
            target_per_class = min(total_real_available, total_fake_available)
        else:
            target_per_class = min(total_real_available, total_fake_available)
        
        print(f"   Target per class: {target_per_class}")
        
        # Now create balanced datasets
        # For real images: combine COCO + SynthBuster RAISE up to target
        real_needed = target_per_class
        
        # Use all SynthBuster RAISE first
        synthbuster_real_to_use = min(synthbuster_real_count, real_needed)
        real_needed -= synthbuster_real_to_use
        
        # Fill remaining with COCO
        coco_to_use = min(coco_count, real_needed)
        
        print(f"\n4. Creating balanced subsets...")
        print(f"   Using {synthbuster_real_to_use} from SynthBuster RAISE")
        print(f"   Using {coco_to_use} from COCO 2017")
        print(f"   Using {target_per_class} from SynthBuster AI generators")
        
        # Create COCO dataset with limit
        self.coco_dataset = COCO2017Dataset(
            root_dir=str(self.coco_root),
            transform=self.transform,
            max_samples=coco_to_use
        )
        
        # Create filtered SynthBuster datasets
        self.synthbuster_real_samples = [
            s for s in synthbuster_full.samples if s['label'] == 0
        ][:synthbuster_real_to_use]
        
        self.synthbuster_fake_samples = [
            s for s in synthbuster_full.samples if s['label'] == 1
        ][:target_per_class]
        
        # Combine all samples
        self.all_samples = []
        
        # Add COCO samples (label=0)
        for i in range(len(self.coco_dataset)):
            self.all_samples.append(('coco', i, 0))
        
        # Add SynthBuster real samples (label=0)
        for i, sample in enumerate(self.synthbuster_real_samples):
            self.all_samples.append(('synthbuster_real', i, 0))
        
        # Add SynthBuster fake samples (label=1)
        for i, sample in enumerate(self.synthbuster_fake_samples):
            self.all_samples.append(('synthbuster_fake', i, 1))
        
        # Count final distribution
        real_count = sum(1 for _, _, label in self.all_samples if label == 0)
        fake_count = sum(1 for _, _, label in self.all_samples if label == 1)
        
        # Shuffle samples if requested
        if self.shuffle:
            import random
            random.seed(self.seed)
            random.shuffle(self.all_samples)
        
        print(f"\n5. Final dataset composition:")
        print(f"   Real images (label=0): {real_count}")
        print(f"   Fake images (label=1): {fake_count}")
        print(f"   Total images: {len(self.all_samples)}")
        print(f"   Balance ratio: {real_count/fake_count:.2f}:1")
        print(f"   Shuffled: {self.shuffle}")
        print("="*70 + "\n")
        
        if abs(real_count - fake_count) > 100:
            warnings.warn(
                f"Dataset imbalance detected: {real_count} real vs {fake_count} fake. "
                f"Consider adjusting balance_mode or dataset sizes."
            )
    
    def __len__(self) -> int:
        """Return the total number of samples in the combined dataset."""
        return len(self.all_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a sample from the combined dataset.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Tuple of (image_tensor, label)
        """
        source, source_idx, label = self.all_samples[idx]
        
        if source == 'coco':
            image_tensor, _ = self.coco_dataset[source_idx]
            return image_tensor, label
        
        elif source == 'synthbuster_real':
            sample = self.synthbuster_real_samples[source_idx]
            from PIL import Image
            image = Image.open(sample['path']).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor, label
        
        elif source == 'synthbuster_fake':
            sample = self.synthbuster_fake_samples[source_idx]
            from PIL import Image
            image = Image.open(sample['path']).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor, label
        
        else:
            raise ValueError(f"Unknown source: {source}")


def create_train_val_split_combined(
    synthbuster_root: str,
    coco_root: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple['BalancedCombinedDataset', 'BalancedCombinedDataset']:
    """
    Create train/validation split for the combined dataset.
    
    Args:
        synthbuster_root: Path to SynthBuster dataset root
        coco_root: Path to COCO 2017 dataset root
        val_ratio: Ratio of validation samples (default: 0.2)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    import random
    
    # Create full combined dataset
    full_dataset = BalancedCombinedDataset(
        synthbuster_root=synthbuster_root,
        coco_root=coco_root
    )
    
    # Create indices for split
    indices = list(range(len(full_dataset)))
    random.seed(seed)
    random.shuffle(indices)
    
    # Split indices
    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Train/Val split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_dataset, val_dataset
