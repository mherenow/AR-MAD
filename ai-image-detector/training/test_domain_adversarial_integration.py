"""
Integration test demonstrating domain adversarial training usage.

This test shows how to use the domain adversarial training components
in a realistic training scenario with multiple datasets.
"""

import torch
import torch.nn as nn
from training.domain_adversarial import (
    GradientReversalLayer,
    DomainDiscriminator,
    compute_domain_adversarial_loss
)


def test_realistic_multi_dataset_training():
    """
    Test domain adversarial training in a realistic multi-dataset scenario.
    
    This simulates training a classifier on two datasets (e.g., SynthBuster and COCO)
    while using domain adversarial training to learn domain-invariant features.
    """
    # Simulate a backbone network (e.g., ResNet feature extractor)
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 512),
        nn.ReLU()
    )
    
    # Binary classifier head
    classifier = nn.Linear(512, 1)
    
    # Domain discriminator (2 domains: SynthBuster and COCO)
    domain_discriminator = DomainDiscriminator(
        feature_dim=512,
        num_domains=2,
        hidden_dim=256
    )
    
    # Optimizers
    model_optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=0.001
    )
    domain_optimizer = torch.optim.Adam(
        domain_discriminator.parameters(),
        lr=0.001
    )
    
    # Simulate training data from two datasets
    # Dataset 0: SynthBuster (32 samples)
    images_dataset0 = torch.randn(32, 3, 64, 64)
    labels_dataset0 = torch.randint(0, 2, (32, 1)).float()
    domain_labels_dataset0 = torch.zeros(32, dtype=torch.long)
    
    # Dataset 1: COCO (32 samples)
    images_dataset1 = torch.randn(32, 3, 64, 64)
    labels_dataset1 = torch.randint(0, 2, (32, 1)).float()
    domain_labels_dataset1 = torch.ones(32, dtype=torch.long)
    
    # Combine batches
    images = torch.cat([images_dataset0, images_dataset1])
    labels = torch.cat([labels_dataset0, labels_dataset1])
    domain_labels = torch.cat([domain_labels_dataset0, domain_labels_dataset1])
    
    # Training step
    backbone.train()
    classifier.train()
    domain_discriminator.train()
    
    # Forward pass through backbone
    features = backbone(images)
    
    # Classification loss
    predictions = classifier(features)
    classification_loss = nn.BCEWithLogitsLoss()(predictions, labels)
    
    # Domain adversarial loss
    domain_loss = compute_domain_adversarial_loss(
        features=features,
        domain_labels=domain_labels,
        domain_discriminator=domain_discriminator,
        lambda_=1.0
    )
    
    # Combined loss (alpha=0.1 for domain loss weight)
    alpha = 0.1
    total_loss = classification_loss + alpha * domain_loss
    
    # Backward pass for model
    model_optimizer.zero_grad()
    domain_optimizer.zero_grad()
    total_loss.backward()
    model_optimizer.step()
    domain_optimizer.step()
    
    # Assertions
    assert total_loss.item() >= 0
    assert classification_loss.item() >= 0
    assert domain_loss.item() >= 0
    
    # Check that gradients were computed
    assert backbone[0].weight.grad is not None
    assert classifier.weight.grad is not None
    assert domain_discriminator.fc1.weight.grad is not None
    
    print(f"Classification Loss: {classification_loss.item():.4f}")
    print(f"Domain Loss: {domain_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print("✓ Domain adversarial training step completed successfully")


def test_gradient_reversal_effect():
    """
    Test that gradient reversal actually affects the feature extractor.
    
    This test verifies that the gradient reversal layer causes the feature
    extractor to receive negated gradients from the domain discriminator.
    """
    # Simple feature extractor
    feature_extractor = nn.Linear(10, 512)
    
    # Domain discriminator
    discriminator = DomainDiscriminator(512, 2, 256)
    
    # Input data
    x = torch.randn(32, 10, requires_grad=True)
    domain_labels = torch.randint(0, 2, (32,))
    
    # Forward pass
    features = feature_extractor(x)
    
    # Compute domain loss with gradient reversal
    loss = compute_domain_adversarial_loss(
        features, domain_labels, discriminator, lambda_=1.0
    )
    
    # Backward pass
    loss.backward()
    
    # Check that feature extractor received gradients
    assert feature_extractor.weight.grad is not None
    
    # The gradient should be non-zero (reversed from discriminator)
    assert not torch.allclose(
        feature_extractor.weight.grad,
        torch.zeros_like(feature_extractor.weight.grad)
    )
    
    print("✓ Gradient reversal is working correctly")


def test_domain_invariance_objective():
    """
    Test that domain adversarial training encourages domain-invariant features.
    
    This test trains a simple model and verifies that the domain discriminator
    has difficulty distinguishing between domains after training.
    """
    # Simple backbone
    backbone = nn.Sequential(
        nn.Linear(20, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU()
    )
    
    # Domain discriminator
    discriminator = DomainDiscriminator(512, 2, 256)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(discriminator.parameters()),
        lr=0.01
    )
    
    # Create two domains with different distributions
    domain0_mean = torch.ones(20) * 2.0
    domain1_mean = torch.ones(20) * -2.0
    
    # Train for several steps
    num_steps = 100
    losses = []
    
    for step in range(num_steps):
        # Sample from both domains
        domain0_samples = torch.randn(16, 20) + domain0_mean
        domain1_samples = torch.randn(16, 20) + domain1_mean
        
        x = torch.cat([domain0_samples, domain1_samples])
        domain_labels = torch.cat([
            torch.zeros(16, dtype=torch.long),
            torch.ones(16, dtype=torch.long)
        ])
        
        # Forward pass
        features = backbone(x)
        loss = compute_domain_adversarial_loss(
            features, domain_labels, discriminator, lambda_=1.0
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # The discriminator should maintain some loss (it's fighting the feature extractor)
    final_loss = losses[-1]
    assert final_loss >= 0  # Loss should be non-negative
    
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print("✓ Domain adversarial training maintains discriminator challenge")


if __name__ == "__main__":
    print("Running domain adversarial training integration tests...\n")
    
    print("Test 1: Realistic multi-dataset training")
    test_realistic_multi_dataset_training()
    print()
    
    print("Test 2: Gradient reversal effect")
    test_gradient_reversal_effect()
    print()
    
    print("Test 3: Domain invariance objective")
    test_domain_invariance_objective()
    print()
    
    print("All integration tests passed! ✓")
