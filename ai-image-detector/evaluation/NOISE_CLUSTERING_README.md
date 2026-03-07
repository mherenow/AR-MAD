# Noise Imprint Clustering Analysis

This module provides tools for evaluating how well noise imprint features separate different AI image generators using clustering metrics.

## Overview

The noise imprint clustering analysis evaluates generator separation by:
1. Extracting noise residuals from images
2. Computing noise imprint features using the NoiseImprintBranch
3. Measuring cluster quality using silhouette score and Davies-Bouldin index

## Metrics

### Silhouette Score
- **Range**: [-1, 1]
- **Interpretation**:
  - > 0.7: Excellent separation
  - 0.5 - 0.7: Good separation
  - 0.25 - 0.5: Moderate separation
  - 0 - 0.25: Weak separation
  - < 0: Poor separation (overlapping clusters)

### Davies-Bouldin Index
- **Range**: [0, ∞)
- **Interpretation**:
  - < 0.5: Excellent separation
  - 0.5 - 1.0: Good separation
  - 1.0 - 1.5: Moderate separation
  - > 1.5: Poor separation

## Usage

### Basic Evaluation

```python
from evaluation.noise_clustering import evaluate_noise_imprint_clustering, print_clustering_report
from models.classifier import BinaryClassifier
import torch

# Load model with noise imprint branch
model = BinaryClassifier(use_noise_imprint=True)
model.load_state_dict(torch.load('checkpoint.pth'))
device = torch.device('cuda')

# Evaluate clustering
generator_labels = ['DALL-E', 'Midjourney', 'Stable Diffusion']
metrics = evaluate_noise_imprint_clustering(
    model=model,
    test_loader=test_loader,
    device=device,
    generator_labels=generator_labels
)

# Print report
print_clustering_report(metrics, verbose=True)
```

### Extract Features for Analysis

```python
from evaluation.noise_clustering import extract_noise_features

# Extract features from a batch of images
images = torch.randn(10, 3, 256, 256)
features = extract_noise_features(model, images, device)
# features.shape: (10, 256)
```

### Pairwise Generator Separability

```python
from evaluation.noise_clustering import compute_pairwise_separability
import numpy as np

# Compute pairwise silhouette scores
features = np.random.randn(100, 256)
labels = np.repeat([0, 1, 2], [30, 40, 30])
generator_labels = ['DALL-E', 'Midjourney', 'Stable Diffusion']

pairwise_scores = compute_pairwise_separability(
    features, labels, generator_labels
)

# Print pairwise scores
for (gen_i, gen_j), score in pairwise_scores.items():
    print(f"{generator_labels[gen_i]} vs {generator_labels[gen_j]}: {score:.4f}")
```

## Requirements

The model must have the following components:
- `noise_extractor`: NoiseResidualExtractor for extracting noise residuals
- `noise_branch`: NoiseImprintBranch for computing features

These are automatically included when creating a BinaryClassifier with `use_noise_imprint=True`.

## Data Format

The test_loader should return:
- `(images, labels)`: where labels are generator indices (0, 1, 2, ...)
- `(images, labels, generator_name)`: optional third element is ignored

Requirements:
- At least 2 different generators in the test set
- At least 2 samples per generator

## Example Output

```
========================================
NOISE IMPRINT CLUSTERING REPORT
========================================

Dataset Statistics:
  Samples:    250
  Generators: 5
  Labels:     DALL-E 3, Midjourney v6, Stable Diffusion XL, Firefly, Imagen

Clustering Metrics:
  Silhouette Score:      0.6234 (Good separation)
  Davies-Bouldin Index:  0.8123 (Good separation)

Interpretation:
  The silhouette score of 0.62 indicates that noise imprints
  from different generators are well-separated in feature space.
  The Davies-Bouldin index of 0.81 confirms good cluster quality.

========================================
```

## Running the Example

```bash
python -m evaluation.example_noise_clustering
```

This will:
1. Create a synthetic dataset with multiple generators
2. Evaluate clustering metrics
3. Compute pairwise separability
4. Print a detailed analysis report

## Testing

Run the test suite:

```bash
pytest ai-image-detector/evaluation/test_noise_clustering.py -v
```

## Implementation Details

### Clustering Metrics Computation

The module uses scikit-learn's implementations:
- `sklearn.metrics.silhouette_score`: Measures how similar samples are to their own cluster vs. other clusters
- `sklearn.metrics.davies_bouldin_score`: Measures the average similarity ratio of each cluster with its most similar cluster

### Feature Extraction

Features are extracted by:
1. Passing images through `noise_extractor` to get residuals
2. Processing residuals through `noise_branch` to get features
3. Handling both single output and tuple output (with attribution)

### Error Handling

The module validates:
- Model has required components (noise_extractor, noise_branch)
- At least 2 generators present in test set
- Each generator has at least 2 samples
- DataLoader returns proper format

## Related Modules

- `models/noise/residual_extractor.py`: Extracts noise residuals
- `models/noise/noise_branch.py`: Computes noise imprint features
- `evaluation/robustness_eval.py`: Robustness evaluation
- `evaluation/spectral_viz.py`: Spectral artifact visualization

## References

- Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
- Davies, D. L.; Bouldin, D. W. (1979). "A Cluster Separation Measure"
