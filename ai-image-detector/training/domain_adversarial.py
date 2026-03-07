"""
Domain adversarial training components for learning domain-invariant features.

This module implements gradient reversal layer and domain discriminator for
domain adversarial training, which helps the model learn features that are
invariant across different datasets/domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal layer for domain adversarial training.
    
    Forward pass: identity function (output = input)
    Backward pass: negates gradients (multiplies by -lambda)
    
    This layer is used to train a domain discriminator adversarially, where
    the feature extractor tries to fool the discriminator by learning
    domain-invariant features.
    
    Reference:
        Ganin & Lempitsky (2015). Unsupervised Domain Adaptation by
        Backpropagation. ICML.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass: identity function.
        
        Args:
            ctx: Context object for storing information for backward pass
            x: Input tensor (B, feature_dim)
            lambda_: Gradient reversal strength (typically 1.0)
        
        Returns:
            Output tensor (same as input)
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: negate gradients.
        
        Args:
            ctx: Context object with stored information
            grad_output: Gradient from downstream layers
        
        Returns:
            Negated gradient for input, None for lambda_
        """
        lambda_ = ctx.lambda_
        grad_input = -lambda_ * grad_output
        return grad_input, None


class DomainDiscriminator(nn.Module):
    """
    Domain classifier for adversarial training.
    
    This network predicts which dataset/domain a sample comes from based on
    its feature representation. During training, the feature extractor tries
    to fool this discriminator by learning domain-invariant features.
    
    Args:
        feature_dim: Input feature dimension
        num_domains: Number of domains/datasets to classify
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout probability (default: 0.5)
    
    Architecture:
        Input (feature_dim) -> FC -> ReLU -> Dropout -> FC -> Output (num_domains)
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_domains: int,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.hidden_dim = hidden_dim
        
        # Two-layer MLP with dropout
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_domains)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through domain discriminator.
        
        Args:
            x: Input features (B, feature_dim)
        
        Returns:
            Domain logits (B, num_domains)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def compute_domain_adversarial_loss(
    features: torch.Tensor,
    domain_labels: torch.Tensor,
    domain_discriminator: nn.Module,
    lambda_: float = 1.0
) -> torch.Tensor:
    """
    Compute domain adversarial loss with gradient reversal.
    
    This function applies gradient reversal to the features and then computes
    the cross-entropy loss for domain classification. During backpropagation,
    the negated gradients encourage the feature extractor to learn
    domain-invariant representations.
    
    Args:
        features: Feature vectors from the backbone (B, feature_dim)
        domain_labels: Domain indices for each sample (B,)
        domain_discriminator: Domain classifier network
        lambda_: Gradient reversal strength (default: 1.0)
                Higher values make domain adaptation stronger
    
    Returns:
        Domain adversarial loss (scalar tensor)
    
    Example:
        >>> features = backbone(images)  # (32, 512)
        >>> domain_labels = torch.tensor([0, 0, 1, 1, ...])  # (32,)
        >>> discriminator = DomainDiscriminator(512, 2)
        >>> loss = compute_domain_adversarial_loss(
        ...     features, domain_labels, discriminator, lambda_=1.0
        ... )
        >>> total_loss = classification_loss + 0.1 * loss
    """
    # Apply gradient reversal layer
    reversed_features = GradientReversalLayer.apply(features, lambda_)
    
    # Predict domain
    domain_pred = domain_discriminator(reversed_features)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(domain_pred, domain_labels)
    
    return loss
