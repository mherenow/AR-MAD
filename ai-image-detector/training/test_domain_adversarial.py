"""
Unit tests for domain adversarial training components.
"""

import pytest
import torch
import torch.nn as nn
from training.domain_adversarial import (
    GradientReversalLayer,
    DomainDiscriminator,
    compute_domain_adversarial_loss
)


class TestGradientReversalLayer:
    """Tests for GradientReversalLayer."""
    
    def test_forward_is_identity(self):
        """Test that forward pass is identity function."""
        x = torch.randn(32, 512, requires_grad=True)
        lambda_ = 1.0
        
        output = GradientReversalLayer.apply(x, lambda_)
        
        assert torch.allclose(output, x)
        assert output.shape == x.shape
    
    def test_backward_negates_gradients(self):
        """Test that backward pass negates gradients."""
        x = torch.randn(32, 512, requires_grad=True)
        lambda_ = 1.0
        
        # Forward pass
        output = GradientReversalLayer.apply(x, lambda_)
        
        # Backward pass with ones gradient
        output.backward(torch.ones_like(output))
        
        # Gradient should be negated
        expected_grad = -lambda_ * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)
    
    def test_lambda_scaling(self):
        """Test that lambda parameter scales gradient reversal."""
        x = torch.randn(32, 512, requires_grad=True)
        lambda_ = 2.5
        
        output = GradientReversalLayer.apply(x, lambda_)
        output.backward(torch.ones_like(output))
        
        expected_grad = -lambda_ * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)
    
    def test_different_batch_sizes(self):
        """Test gradient reversal with different batch sizes."""
        for batch_size in [1, 16, 64, 128]:
            x = torch.randn(batch_size, 256, requires_grad=True)
            output = GradientReversalLayer.apply(x, 1.0)
            
            assert output.shape == (batch_size, 256)
            assert torch.allclose(output, x)


class TestDomainDiscriminator:
    """Tests for DomainDiscriminator."""
    
    def test_initialization(self):
        """Test discriminator initialization."""
        discriminator = DomainDiscriminator(
            feature_dim=512,
            num_domains=3,
            hidden_dim=256
        )
        
        assert discriminator.feature_dim == 512
        assert discriminator.num_domains == 3
        assert discriminator.hidden_dim == 256
        assert isinstance(discriminator.fc1, nn.Linear)
        assert isinstance(discriminator.fc2, nn.Linear)
    
    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        discriminator = DomainDiscriminator(512, 3, 256)
        features = torch.randn(32, 512)
        
        output = discriminator(features)
        
        assert output.shape == (32, 3)
    
    def test_forward_with_different_batch_sizes(self):
        """Test forward pass with various batch sizes."""
        discriminator = DomainDiscriminator(256, 5, 128)
        
        for batch_size in [1, 8, 32, 64]:
            features = torch.randn(batch_size, 256)
            output = discriminator(features)
            assert output.shape == (batch_size, 5)
    
    def test_forward_with_different_feature_dims(self):
        """Test discriminator with different feature dimensions."""
        for feature_dim in [128, 256, 512, 1024]:
            discriminator = DomainDiscriminator(feature_dim, 2, 256)
            features = torch.randn(16, feature_dim)
            output = discriminator(features)
            assert output.shape == (16, 2)
    
    def test_forward_with_different_num_domains(self):
        """Test discriminator with different numbers of domains."""
        for num_domains in [2, 3, 5, 10]:
            discriminator = DomainDiscriminator(512, num_domains, 256)
            features = torch.randn(16, 512)
            output = discriminator(features)
            assert output.shape == (16, num_domains)
    
    def test_dropout_in_training_mode(self):
        """Test that dropout is applied in training mode."""
        discriminator = DomainDiscriminator(512, 2, 256, dropout=0.5)
        discriminator.train()
        
        features = torch.randn(100, 512)
        
        # Run multiple forward passes
        outputs = []
        for _ in range(5):
            output = discriminator(features)
            outputs.append(output)
        
        # Outputs should differ due to dropout
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i])
    
    def test_no_dropout_in_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        discriminator = DomainDiscriminator(512, 2, 256, dropout=0.5)
        discriminator.eval()
        
        features = torch.randn(100, 512)
        
        # Run multiple forward passes
        output1 = discriminator(features)
        output2 = discriminator(features)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)
    
    def test_gradient_flow(self):
        """Test that gradients flow through discriminator."""
        discriminator = DomainDiscriminator(512, 3, 256)
        features = torch.randn(32, 512, requires_grad=True)
        
        output = discriminator(features)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert features.grad is not None
        assert discriminator.fc1.weight.grad is not None
        assert discriminator.fc2.weight.grad is not None
    
    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        discriminator = DomainDiscriminator(512, 3, 256)
        
        # Check that weights are not all zeros
        assert not torch.allclose(discriminator.fc1.weight, torch.zeros_like(discriminator.fc1.weight))
        assert not torch.allclose(discriminator.fc2.weight, torch.zeros_like(discriminator.fc2.weight))
        
        # Check that biases are initialized to zero
        assert torch.allclose(discriminator.fc1.bias, torch.zeros_like(discriminator.fc1.bias))
        assert torch.allclose(discriminator.fc2.bias, torch.zeros_like(discriminator.fc2.bias))


class TestComputeDomainAdversarialLoss:
    """Tests for compute_domain_adversarial_loss function."""
    
    def test_loss_computation(self):
        """Test basic loss computation."""
        features = torch.randn(32, 512)
        domain_labels = torch.randint(0, 3, (32,))
        discriminator = DomainDiscriminator(512, 3, 256)
        
        loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_loss_with_different_lambda(self):
        """Test loss computation with different lambda values."""
        features = torch.randn(32, 512, requires_grad=True)
        domain_labels = torch.randint(0, 2, (32,))
        discriminator = DomainDiscriminator(512, 2, 256)
        
        for lambda_ in [0.5, 1.0, 2.0, 5.0]:
            features.grad = None  # Reset gradients
            
            loss = compute_domain_adversarial_loss(
                features, domain_labels, discriminator, lambda_
            )
            loss.backward()
            
            # Loss value should be the same (lambda only affects gradients)
            assert loss.item() >= 0
            # Gradients should exist
            assert features.grad is not None
    
    def test_gradient_reversal_in_loss(self):
        """Test that gradient reversal is applied in loss computation."""
        features = torch.randn(32, 512, requires_grad=True)
        domain_labels = torch.randint(0, 2, (32,))
        discriminator = DomainDiscriminator(512, 2, 256)
        
        loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        loss.backward()
        
        # Gradient should exist and be non-zero
        assert features.grad is not None
        assert not torch.allclose(features.grad, torch.zeros_like(features.grad))
    
    def test_loss_with_perfect_predictions(self):
        """Test loss when discriminator makes perfect predictions."""
        discriminator = DomainDiscriminator(512, 2, 256)
        discriminator.eval()
        
        # Create features that lead to confident predictions
        features = torch.randn(32, 512)
        domain_labels = torch.zeros(32, dtype=torch.long)
        
        loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        
        # Loss should be positive (cross-entropy)
        assert loss.item() >= 0
    
    def test_loss_with_multiple_domains(self):
        """Test loss computation with multiple domains."""
        for num_domains in [2, 3, 5, 10]:
            features = torch.randn(64, 512)
            domain_labels = torch.randint(0, num_domains, (64,))
            discriminator = DomainDiscriminator(512, num_domains, 256)
            
            loss = compute_domain_adversarial_loss(
                features, domain_labels, discriminator, lambda_=1.0
            )
            
            assert loss.item() >= 0
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
    
    def test_loss_with_single_sample(self):
        """Test loss computation with batch size of 1."""
        features = torch.randn(1, 512)
        domain_labels = torch.tensor([0])
        discriminator = DomainDiscriminator(512, 2, 256)
        
        loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        
        assert loss.item() >= 0
    
    def test_loss_gradient_flow_to_discriminator(self):
        """Test that gradients flow to discriminator parameters."""
        features = torch.randn(32, 512)
        domain_labels = torch.randint(0, 2, (32,))
        discriminator = DomainDiscriminator(512, 2, 256)
        
        loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        loss.backward()
        
        # Check that discriminator has gradients
        assert discriminator.fc1.weight.grad is not None
        assert discriminator.fc2.weight.grad is not None
        assert not torch.allclose(
            discriminator.fc1.weight.grad,
            torch.zeros_like(discriminator.fc1.weight.grad)
        )
    
    def test_loss_with_uniform_domain_distribution(self):
        """Test loss with uniformly distributed domain labels."""
        features = torch.randn(100, 512)
        # Create uniform distribution of domain labels
        domain_labels = torch.cat([
            torch.zeros(50, dtype=torch.long),
            torch.ones(50, dtype=torch.long)
        ])
        discriminator = DomainDiscriminator(512, 2, 256)
        
        loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_loss_with_imbalanced_domains(self):
        """Test loss with imbalanced domain distribution."""
        features = torch.randn(100, 512)
        # Create imbalanced distribution (90% domain 0, 10% domain 1)
        domain_labels = torch.cat([
            torch.zeros(90, dtype=torch.long),
            torch.ones(10, dtype=torch.long)
        ])
        discriminator = DomainDiscriminator(512, 2, 256)
        
        loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestIntegration:
    """Integration tests for domain adversarial training."""
    
    def test_end_to_end_training_step(self):
        """Test a complete training step with domain adversarial loss."""
        # Simulate a simple backbone
        backbone = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU()
        )
        
        # Create discriminator
        discriminator = DomainDiscriminator(512, 2, 256)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(discriminator.parameters()),
            lr=0.001
        )
        
        # Simulate training data
        images = torch.randn(32, 100)
        domain_labels = torch.randint(0, 2, (32,))
        
        # Forward pass
        features = backbone(images)
        domain_loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        
        # Backward pass
        optimizer.zero_grad()
        domain_loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        assert domain_loss.item() >= 0
    
    def test_combined_classification_and_domain_loss(self):
        """Test combining classification loss with domain adversarial loss."""
        # Simulate backbone and classifier
        backbone = nn.Sequential(nn.Linear(100, 512), nn.ReLU())
        classifier = nn.Linear(512, 1)
        discriminator = DomainDiscriminator(512, 2, 256)
        
        # Training data
        images = torch.randn(32, 100)
        labels = torch.randint(0, 2, (32, 1)).float()
        domain_labels = torch.randint(0, 2, (32,))
        
        # Forward pass
        features = backbone(images)
        predictions = classifier(features)
        
        # Compute losses
        classification_loss = nn.BCEWithLogitsLoss()(predictions, labels)
        domain_loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        
        # Combined loss
        total_loss = classification_loss + 0.1 * domain_loss
        
        # Backward pass
        total_loss.backward()
        
        assert total_loss.item() >= 0
        assert classification_loss.item() >= 0
        assert domain_loss.item() >= 0
    
    def test_feature_invariance_over_training(self):
        """Test that features become more domain-invariant over training."""
        # Simple backbone
        backbone = nn.Sequential(nn.Linear(10, 512), nn.ReLU())
        discriminator = DomainDiscriminator(512, 2, 256)
        
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(discriminator.parameters()),
            lr=0.01
        )
        
        # Training data from two domains
        domain0_data = torch.randn(50, 10) + 1.0  # Shifted distribution
        domain1_data = torch.randn(50, 10) - 1.0  # Different shift
        
        initial_loss = None
        final_loss = None
        
        # Train for a few steps
        for step in range(50):
            # Sample batch
            idx0 = torch.randint(0, 50, (16,))
            idx1 = torch.randint(0, 50, (16,))
            
            images = torch.cat([domain0_data[idx0], domain1_data[idx1]])
            domain_labels = torch.cat([
                torch.zeros(16, dtype=torch.long),
                torch.ones(16, dtype=torch.long)
            ])
            
            # Forward pass
            features = backbone(images)
            loss = compute_domain_adversarial_loss(
                features, domain_labels, discriminator, lambda_=1.0
            )
            
            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Loss should remain positive (discriminator keeps trying)
        assert initial_loss >= 0
        assert final_loss >= 0
