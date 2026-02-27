# SynthBuster Dataset Loader Tests

This directory contains comprehensive unit tests for the SynthBuster dataset loader.

## Test Coverage

The test suite (`test_synthbuster_loader.py`) includes:

### 1. TestSynthBusterDataset
- **test_dataset_initialization**: Verifies dataset loads correctly
- **test_image_shape_and_type**: Checks images are 256x256 RGB tensors normalized to [0,1]
- **test_label_assignment**: Validates RAISE=0 (real), others=1 (fake)
- **test_generator_name_returned**: Ensures generator names are correctly returned
- **test_corrupted_image_handling**: Tests graceful handling of corrupted images
- **test_index_out_of_range**: Validates IndexError for invalid indices
- **test_empty_directory_warning**: Checks warning for empty directories
- **test_nonexistent_directory**: Tests ValueError for missing directories
- **test_custom_transform**: Verifies custom transforms work correctly

### 2. TestTrainValSplit
- **test_split_ratio_default**: Tests default 80/20 split
- **test_split_ratio_custom**: Validates custom split ratios (10%, 30%, 50%)
- **test_split_no_overlap**: Ensures train/val sets don't overlap
- **test_split_reproducibility**: Checks same seed produces same split
- **test_split_different_seeds**: Verifies different seeds produce different splits
- **test_split_nonexistent_directory**: Tests error handling
- **test_split_empty_directory**: Validates empty directory handling

### 3. TestGeneratorSubsets
- **test_all_generators_found**: Verifies all generator directories are discovered
- **test_correct_image_counts**: Validates image counts per generator
- **test_paths_are_valid**: Checks returned paths are valid
- **test_unsupported_generator_warning**: Tests warning for unknown generators
- **test_empty_generator_directory**: Ensures empty dirs are excluded
- **test_nonexistent_directory**: Tests error handling
- **test_multiple_image_formats**: Validates support for .jpg, .png, .jpeg, .bmp
- **test_subset_filtering_by_generator**: Tests filtering by specific generator

### 4. TestIntegration
- **test_dataset_and_split_consistency**: Validates dataset and split agree on counts
- **test_dataset_and_subsets_consistency**: Ensures dataset and subsets match
- **test_dataloader_compatibility**: Tests PyTorch DataLoader integration

## Installation

Install required dependencies:

```bash
# Install from ai-image-detector requirements
pip install -r ai-image-detector/requirements.txt

# Or install individually
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.24.0 Pillow>=10.0.0 pytest
```

## Running Tests

### Using pytest (recommended)

```bash
# Run all tests with verbose output
pytest ai-image-detector/data/test_synthbuster_loader.py -v

# Run specific test class
pytest ai-image-detector/data/test_synthbuster_loader.py::TestSynthBusterDataset -v

# Run specific test method
pytest ai-image-detector/data/test_synthbuster_loader.py::TestSynthBusterDataset::test_image_shape_and_type -v

# Run with coverage
pytest ai-image-detector/data/test_synthbuster_loader.py --cov=ai-image-detector.data.synthbuster_loader --cov-report=html
```

### Using unittest

```bash
# Run all tests
python -m unittest ai-image-detector.data.test_synthbuster_loader

# Run specific test class
python -m unittest ai-image-detector.data.test_synthbuster_loader.TestSynthBusterDataset

# Run specific test method
python -m unittest ai-image-detector.data.test_synthbuster_loader.TestSynthBusterDataset.test_image_shape_and_type
```

### Direct execution

```bash
cd ai-image-detector/data
python test_synthbuster_loader.py
```

## Test Data

Tests automatically create temporary directories with synthetic test data:
- Multiple generator directories (RAISE, SD_v2, GLIDE, Midjourney, etc.)
- Random RGB images of various sizes
- Corrupted image files for error handling tests
- Different image formats (.jpg, .png, .jpeg, .bmp)

All test data is cleaned up automatically after tests complete.

## Expected Output

Successful test run should show:

```
============================== test session starts ==============================
collected 31 items

test_synthbuster_loader.py::TestSynthBusterDataset::test_dataset_initialization PASSED
test_synthbuster_loader.py::TestSynthBusterDataset::test_image_shape_and_type PASSED
test_synthbuster_loader.py::TestSynthBusterDataset::test_label_assignment PASSED
...
============================== 31 passed in X.XXs ===============================
```

## Troubleshooting

### ModuleNotFoundError: No module named 'torch'
Install PyTorch: `pip install torch torchvision`

### ModuleNotFoundError: No module named 'PIL'
Install Pillow: `pip install Pillow`

### Import errors
Ensure you're running from the project root directory or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Notes

- Tests use temporary directories and clean up automatically
- Each test class has its own isolated test data
- Tests are designed to be fast and deterministic
- All tests use fixed random seeds for reproducibility
