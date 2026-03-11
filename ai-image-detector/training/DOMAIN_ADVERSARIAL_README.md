# Domain Adversarial Training

This module implements domain adversarial training components for learning domain-invariant features across multiple datasets. Domain adversarial training helps the model generalize better by learning features that work well across different data distributions.

## Overview

Domain adversarial training uses a gradient reversal layer to train a domain discriminator that predicts which dataset an example comes from, while simultaneously training the feature extractor to fool the discriminator. This adversarial process encourages the model to learn features that are invariant across domains.

## Components

### 1. GradientReversalLayer

A custom autograd function that implements gradient reversal:
- **Forward pass**: Identity function (output = input)
- **Backward pass**: Negates gradients (multiplies by -lambda)

```python
from training.domain_adversarial import GradientReversalLayer

# Apply gradient reversal
reversed_features = GradientReversalLayer.apply(features, lambda_=1.0)
```

**Parameters:**
- `x`: Input tensor (B, feature_dim)
- `lambda_`: Gradient reversal strength (default: 1.0)

### 2. DomainDiscriminator

A neural network that predicts which dataset/domain a sample comes from:

```python
from training.domain_adversarial import DomainDiscriminator

# Create discriminator for 2 domains
discriminator = DomainDiscriminator(
    feature_dim=512,
    num_domains=2,
    hidden_dim=256,
    dropout=0.5
)

# Predict domain
domain_logits = discriminator(features)  # (B, num_domains)
```

**Architecture:**
- Input (feature_dim) → FC → ReLU → Dropout → FC → Output (num_domains)

**Parameters:**
- `feature_dim`: Input feature dimension
- `num_domains`: Number of domains/datasets to classify
- `hidden_dim`: Hidden layer dimension (default: 256)
- `dropout`: Dropout probability (default: 0.5)

### 3. compute_domain_adversarial_loss

Computes the domain adversarial loss with gradient reversal:

```python
from training.domain_adversarial import compute_domain_adversarial_loss

# Compute domain adversarial loss
domain_loss = compute_domain_adversarial_loss(
    features=features,
    domain_labels=domain_labels,
    domain_discriminator=discriminator,
    lambda_=1.0
)
```

**Parameters:**
- `features`: Feature vectors from backbone (B, feature_dim)
- `domain_labels`: Domain indices for each sample (B,)
- `domain_discriminator`: Domain classifier network
- `lambda_`: Gradient reversal strength (default: 1.0)

**Returns:**
- Domain adversarial loss (scalar tensor)

## Usage Example

### Basic Training Loop

```python
import torch
import torch.nn as nn
from training.domain_adversarial import (
    DomainDiscriminator,
    compute_domain_adversarial_loss
)

# Initialize models
backbone = YourBackbone()
classifier = YourClassifier()
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

# Training loop
for images, labels, domain_labels in multi_dataset_loader:
    # Forward pass
    features = backbone(images)
    predictions = classifier(features)
    
    # Classification loss
    cls_loss = nn.BCEWithLogitsLoss()(predictions, labels)
    
    # Domain adversarial loss
    domain_loss = compute_domain_adversarial_loss(
        features, domain_labels, domain_discriminator, lambda_=1.0
    )
    
    # Combined loss (alpha controls domain loss weight)
    alpha = 0.1
    total_loss = cls_loss + alpha * domain_loss
    
    # Backward pass
    model_optimizer.zero_grad()
    domain_optimizer.zero_grad()
    total_loss.backward()
    model_optimizer.step()
    domain_optimizer.step()
```

### Multi-Dataset Training

```python
from data.multi_dataset import MultiDatasetLoader

# Create multi-dataset loader
datasets = {
    'synthbuster': synthbuster_dataset,
    'coco': coco_dataset
}
weights = {
    'synthbuster': 1.0,
    'coco': 0.5
}

loader = MultiDatasetLoader(
    datasets=datasets,
    weights=weights,
    batch_size=32
)

# Training with domain adversarial loss
for images, labels, dataset_names in loader:
    # Convert dataset names to domain labels
    domain_labels = encode_dataset_names(dataset_names)
    
    # ... rest of training loop
```

## How It Works

1. **Feature Extraction**: The backbone network extracts features from input images
2. **Gradient Reversal**: Features pass through the gradient reversal layer
3. **Domain Classification**: The domain discriminator predicts which dataset the features came from
4. **Adversarial Training**: 
   - The discriminator tries to correctly classify domains
   - The feature extractor receives negated gradients, encouraging it to produce domain-invariant features
   - This adversarial process helps the model generalize across datasets

## Hyperparameters

### lambda_ (Gradient Reversal Strength)

Controls how strongly the feature extractor is encouraged to fool the discriminator:
- **lambda_ = 0.0**: No domain adaptation (discriminator has no effect on features)
- **lambda_ = 1.0**: Standard gradient reversal (recommended default)
- **lambda_ > 1.0**: Stronger domain adaptation (may hurt classification performance)

### alpha (Domain Loss Weight)

Controls the relative importance of domain adversarial loss vs classification loss:
- **alpha = 0.0**: No domain adaptation
- **alpha = 0.1**: Light domain adaptation (recommended starting point)
- **alpha = 1.0**: Equal weight to both losses
- **alpha > 1.0**: Prioritize domain invariance over classification

### hidden_dim (Discriminator Capacity)

Controls the capacity of the domain discriminator:
- **hidden_dim = 128**: Smaller discriminator (faster, less capacity)
- **hidden_dim = 256**: Standard discriminator (recommended default)
- **hidden_dim = 512**: Larger discriminator (more capacity, slower)

## Best Practices

1. **Start with small alpha**: Begin with alpha=0.1 and gradually increase if needed
2. **Monitor both losses**: Track both classification and domain losses during training
3. **Use separate optimizers**: Use separate optimizers for the model and discriminator
4. **Balance datasets**: Use weighted sampling to balance dataset representation
5. **Tune lambda dynamically**: Consider gradually increasing lambda during training

## Testing

Run the unit tests:
```bash
pytest ai-image-detector/training/test_domain_adversarial.py -v
```

Run the integration tests:
```bash
pytest ai-image-detector/training/test_domain_adversarial_integration.py -v
```

## References

- Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. ICML.
- Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. JMLR.

## Requirements

Validates Requirements 5.3: Domain adversarial training with gradient reversal layer for learning domain-invariant features.
