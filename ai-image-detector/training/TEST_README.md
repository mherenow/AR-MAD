# Training Integration Tests

This directory contains comprehensive integration tests for the training module.

## Test File

- `test_training_integration.py` - Integration tests for the complete training pipeline

## Test Coverage

### TestTrainingIntegration Class

Tests core training functionality:

1. **test_train_on_synthetic_dataset** - Validates training on small synthetic dataset
2. **test_checkpoint_saving_and_loading** - Tests checkpoint save/load functionality
3. **test_checkpoint_loading_without_optimizer** - Tests loading model without optimizer state
4. **test_checkpoint_loading_nonexistent_file** - Tests error handling for missing checkpoints
5. **test_training_resumption_from_checkpoint** - Tests resuming training from saved checkpoint
6. **test_multi_epoch_training_convergence** - Validates loss decreases over multiple epochs
7. **test_checkpoint_contains_all_required_fields** - Verifies checkpoint structure
8. **test_training_with_different_backbones** - Tests training with simple_cnn, resnet18, resnet50
9. **test_checkpoint_compatibility_across_models** - Tests checkpoint compatibility

### TestEndToEndTraining Class

Tests complete training workflows:

1. **test_complete_training_workflow** - End-to-end training from initialization to final checkpoint
2. **test_training_interruption_and_resume** - Tests interrupting and resuming training

## Running Tests

Run all integration tests:
```bash
pytest ai-image-detector/training/test_training_integration.py -v
```

Run specific test class:
```bash
pytest ai-image-detector/training/test_training_integration.py::TestTrainingIntegration -v
```

Run specific test:
```bash
pytest ai-image-detector/training/test_training_integration.py::TestTrainingIntegration::test_train_on_synthetic_dataset -v
```

## Test Features

- Uses pytest fixtures for temporary directories and synthetic datasets
- Tests with synthetic data (no external dataset required)
- Validates checkpoint saving/loading with all backbone types
- Tests training resumption after interruption
- Verifies model convergence over multiple epochs
- Uses temporary directories for all test artifacts (auto-cleanup)

## Dependencies

- pytest
- torch
- torchvision (for ResNet backbones)

All dependencies are listed in `ai-image-detector/requirements.txt`.
