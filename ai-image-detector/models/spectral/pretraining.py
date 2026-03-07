"""Masked spectral pretraining module for self-supervised learning."""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .spectral_branch import SpectralBranch


class MaskedSpectralPretraining(nn.Module):
    """
    Masked autoencoder for spectral pretraining.
    
    This module implements self-supervised pretraining for the spectral branch
    using masked patch reconstruction. The approach is inspired by MAE (Masked
    Autoencoders) but applied to frequency domain features.
    
    Training procedure:
    1. Randomly mask a high ratio (default 75%) of spectral patches
    2. Encode visible patches using the SpectralBranch encoder
    3. Decode to reconstruct the masked patches
    4. Compute MSE loss between reconstructed and original masked patches
    
    Args:
        spectral_branch: SpectralBranch instance to pretrain
        decoder_embed_dim: Decoder embedding dimension (default: 128)
        decoder_depth: Number of decoder transformer layers (default: 2)
        mask_ratio: Ratio of patches to mask during pretraining (default: 0.75)
        norm_pix_loss: Whether to normalize pixel values in loss (default: True)
    
    Example:
        >>> spectral_branch = SpectralBranch(patch_size=16, embed_dim=256, depth=4)
        >>> pretraining_model = MaskedSpectralPretraining(spectral_branch)
        >>> images = torch.randn(4, 3, 256, 256)
        >>> loss, pred, mask = pretraining_model(images)
        >>> print(f"Loss: {loss.item():.4f}")
    """
    
    def __init__(
        self,
        spectral_branch: SpectralBranch,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True
    ):
        super().__init__()
        
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        
        self.spectral_branch = spectral_branch
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        # Get encoder parameters
        self.patch_size = spectral_branch.patch_size
        self.embed_dim = spectral_branch.embed_dim
        
        # Decoder: project from encoder embed_dim to decoder embed_dim
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embeddings (will be initialized dynamically)
        self.decoder_pos_embed = None
        
        # Decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_embed_dim,
            nhead=8,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder_blocks = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_depth
        )
        
        # Decoder norm
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Decoder prediction head: project to patch pixel values
        # Each patch has patch_size * patch_size * 3 (RGB) values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.patch_size * self.patch_size * 3
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights following MAE initialization."""
        # Initialize mask token with truncated normal
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Initialize decoder layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def _get_decoder_pos_embed(self, num_patches_h: int, num_patches_w: int, device: torch.device) -> torch.Tensor:
        """
        Get or create decoder positional embeddings.
        
        Args:
            num_patches_h: Number of patches in height dimension
            num_patches_w: Number of patches in width dimension
            device: Device to create embeddings on
        
        Returns:
            pos_embed: Positional embeddings (1, num_patches, decoder_embed_dim)
        """
        num_patches = num_patches_h * num_patches_w
        
        # Create if doesn't exist or size changed
        if self.decoder_pos_embed is None or self.decoder_pos_embed.shape[1] != num_patches:
            # Create 2D positional embeddings
            pos_embed = self._create_2d_pos_embed(
                num_patches_h, num_patches_w, self.decoder_embed_dim
            )
            self.decoder_pos_embed = pos_embed.to(device)
        
        return self.decoder_pos_embed
    
    def _create_2d_pos_embed(self, h: int, w: int, embed_dim: int) -> torch.Tensor:
        """
        Create 2D sinusoidal positional embeddings.
        
        Args:
            h: Height in patches
            w: Width in patches
            embed_dim: Embedding dimension
        
        Returns:
            pos_embed: Positional embeddings (1, h*w, embed_dim)
        """
        # Create grid
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)  # (2, h, w)
        
        grid = grid.reshape(2, -1).t()  # (h*w, 2)
        
        # Create sinusoidal embeddings
        pos_embed = self._get_sinusoidal_encoding(grid, embed_dim)
        pos_embed = pos_embed.unsqueeze(0)  # (1, h*w, embed_dim)
        
        return pos_embed
    
    def _get_sinusoidal_encoding(self, positions: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """
        Create sinusoidal positional encodings.
        
        Args:
            positions: Position coordinates (N, 2)
            embed_dim: Embedding dimension
        
        Returns:
            encodings: Sinusoidal encodings (N, embed_dim)
        """
        N = positions.shape[0]
        
        # Create frequency bands
        omega = torch.arange(embed_dim // 4, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))
        
        # Compute encodings for each dimension
        encodings = []
        for dim in range(2):
            pos = positions[:, dim].unsqueeze(1)  # (N, 1)
            pos_enc = pos * omega.unsqueeze(0)  # (N, embed_dim//4)
            pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=1)  # (N, embed_dim//2)
            encodings.append(pos_enc)
        
        encodings = torch.cat(encodings, dim=1)  # (N, embed_dim)
        return encodings
    
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking by shuffling patches.
        
        Args:
            x: Input tokens (B, num_patches, embed_dim)
            mask_ratio: Ratio of patches to mask
        
        Returns:
            x_masked: Visible tokens only (B, num_visible, embed_dim)
            mask: Binary mask (B, num_patches), 1 is keep, 0 is remove
            ids_restore: Indices to restore original order (B, num_patches)
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))
        
        # Generate random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep only the first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Generate binary mask: 1 is keep, 0 is remove
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for masked spectral pretraining.
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            loss: Reconstruction loss (scalar)
            pred: Predicted masked patches (B, num_patches, patch_size^2 * 3)
            mask: Binary mask indicating masked positions (B, num_patches)
        
        Example:
            >>> model = MaskedSpectralPretraining(spectral_branch)
            >>> images = torch.randn(4, 3, 256, 256)
            >>> loss, pred, mask = model(images)
            >>> print(f"Loss: {loss.item():.4f}, Mask ratio: {mask.mean():.2f}")
        """
        B, C, H, W = x.shape
        
        # Validate dimensions
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input height ({H}) and width ({W}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Get target patches (original spectral patches before masking)
        with torch.no_grad():
            # Convert to frequency domain
            magnitude_spectrum = self.spectral_branch.fft_processor(x)
            masked_spectrum = self.spectral_branch.frequency_masking(magnitude_spectrum)
            
            # Patchify: (B, 3, H, W) -> (B, num_patches, patch_size^2 * 3)
            target = self._patchify(masked_spectrum)
        
        # Encode with masking
        # Get tokens from spectral branch (before transformer)
        magnitude_spectrum = self.spectral_branch.fft_processor(x)
        masked_spectrum = self.spectral_branch.frequency_masking(magnitude_spectrum)
        tokens = self.spectral_branch.patch_tokenizer(masked_spectrum)
        
        # Apply random masking
        tokens_masked, mask, ids_restore = self.random_masking(tokens, self.mask_ratio)
        
        # Apply transformer encoder on visible tokens only
        encoded_tokens = self.spectral_branch.transformer_encoder(tokens_masked)
        
        # Decode
        pred = self._decode(encoded_tokens, ids_restore, num_patches_h, num_patches_w)
        
        # Compute loss only on masked patches
        loss = self._compute_loss(pred, target, mask)
        
        return loss, pred, mask
    
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches.
        
        Args:
            x: Image (B, 3, H, W)
        
        Returns:
            patches: Patches (B, num_patches, patch_size^2 * 3)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Reshape to patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        
        return x
    
    def _unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Convert patches back to image.
        
        Args:
            x: Patches (B, num_patches, patch_size^2 * 3)
            h: Height in patches
            w: Width in patches
        
        Returns:
            image: Image (B, 3, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        
        x = x.reshape(B, h, w, 3, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, 3, h, p, w, p)
        x = x.reshape(B, 3, h * p, w * p)
        
        return x
    
    def _decode(
        self,
        encoded_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
        num_patches_h: int,
        num_patches_w: int
    ) -> torch.Tensor:
        """
        Decode encoded tokens to reconstruct patches.
        
        Args:
            encoded_tokens: Encoded visible tokens (B, num_visible, embed_dim)
            ids_restore: Indices to restore original order (B, num_patches)
            num_patches_h: Number of patches in height
            num_patches_w: Number of patches in width
        
        Returns:
            pred: Predicted patches (B, num_patches, patch_size^2 * 3)
        """
        B = encoded_tokens.shape[0]
        num_patches = num_patches_h * num_patches_w
        
        # Project to decoder dimension
        x = self.decoder_embed(encoded_tokens)
        
        # Append mask tokens
        mask_tokens = self.mask_token.expand(B, num_patches - x.shape[1], -1)
        x = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to restore original order
        x = torch.gather(
            x, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
        )
        
        # Add positional embeddings
        pos_embed = self._get_decoder_pos_embed(num_patches_h, num_patches_w, x.device)
        x = x + pos_embed
        
        # Apply decoder transformer
        # For TransformerDecoder, we need memory (encoder output) and tgt (decoder input)
        # In MAE-style, we use self-attention, so we pass x as both
        memory = x  # Use same as memory
        x = self.decoder_blocks(x, memory)
        
        # Apply norm
        x = self.decoder_norm(x)
        
        # Predict pixel values
        pred = self.decoder_pred(x)
        
        return pred
    
    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss on masked patches.
        
        Args:
            pred: Predicted patches (B, num_patches, patch_size^2 * 3)
            target: Target patches (B, num_patches, patch_size^2 * 3)
            mask: Binary mask (B, num_patches), 1 is masked, 0 is visible
        
        Returns:
            loss: MSE loss on masked patches
        """
        if self.norm_pix_loss:
            # Normalize target patches
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        # Compute MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch pixels
        
        # Compute loss only on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
