# Utility Scripts

> Analysis, evaluation, validation, and visualization utilities for Med3Tab-PFN

## Overview

This directory contains standalone scripts for data analysis, model evaluation, pipeline validation, and figure generation.

## Directory Structure

```
scripts/
├── analysis/           # Data analysis and statistics
├── evaluation/         # Model evaluation and testing
├── validation/         # Verification utilities
├── visualization/      # Figure generation
└── README.md           # This file
```

---

## Analysis Scripts

Scripts for analyzing datasets, lesion characteristics, and experimental results.

| Script | Description |
|--------|-------------|
| `analyze_labels.py` | Dataset structure and label distribution analysis |
| `analyze_lesion_sizes.py` | Lesion size distribution statistics |
| `summarize_lesion_shapes.py` | Lesion morphology summary |
| `create_dice_summary.py` | Dice score result summaries |
| `create_per_dataset_summary.py` | Per-dataset performance reports |
| `create_simple_summary.py` | Simplified experiment summaries |
| `trace_index_mapping.py` | Debug sample index tracking |

### Examples

```bash
# Analyze label distribution
python scripts/analysis/analyze_labels.py

# Generate lesion size statistics
python scripts/analysis/analyze_lesion_sizes.py

# Create Dice score summary
python scripts/analysis/create_dice_summary.py
```

---

## Evaluation Scripts

Scripts for evaluating model performance and running tests.

| Script | Description |
|--------|-------------|
| `test_sam_features.py` | Evaluate SAM-Med3D feature quality |
| `test_full_pipeline.py` | End-to-end pipeline validation |
| `compute_sam_dice_scores.py` | Compute segmentation Dice scores |
| `compute_dice_scores_roi.py` | ROI-specific Dice computation |

### Examples

```bash
# Test SAM feature quality
python scripts/evaluation/test_sam_features.py --dataset gist --epochs 10 --freeze

# Run full pipeline test
python scripts/evaluation/test_full_pipeline.py

# Compute Dice scores
python scripts/evaluation/compute_sam_dice_scores.py
```

---

## Validation Scripts

Verification utilities to ensure implementation correctness.

| Script | Description |
|--------|-------------|
| `verify_embeddings_fix.py` | Validate embedding extraction |
| `verify_pooling_logic.py` | Verify pooling strategy correctness |

### Examples

```bash
# Verify embedding extraction
python scripts/validation/verify_embeddings_fix.py

# Check pooling implementations
python scripts/validation/verify_pooling_logic.py
```

---

## Visualization Scripts

Scripts for generating figures and visualizations.

| Script | Description |
|--------|-------------|
| `generate_feature_extraction_pipeline_figure.py` | Pipeline diagram for publications |
| `visualize_sam_segmentations.py` | Segmentation overlay visualizations |

### Examples

```bash
# Generate pipeline figure
python scripts/visualization/generate_feature_extraction_pipeline_figure.py

# Visualize segmentations
python scripts/visualization/visualize_sam_segmentations.py
```

Output: Figures saved to `figures/` directory.

---

## Usage Guidelines

### Running Scripts

All scripts should be run from the project root:

```bash
cd Med3Tab-PFN/
python scripts/<category>/<script_name>.py
```

### Common Options

Many scripts accept these options:
- `--config`: Path to dataset configuration
- `--dataset`: Specific dataset to process
- `--output`: Output directory

---

## Related Documentation

- [Main README](../README.md)
- [Documentation Index](../docs/README.md)
- [Cluster Scripts](../cluster_scripts/README.md)

Scripts for evaluating model performance and running tests.

### `test_sam_features.py`
Evaluate SAM-Med3D feature quality with a classification head.

**Usage**:
```bash
python scripts/evaluation/test_sam_features.py --dataset gist --epochs 10 --freeze
```

**Key Options**:
- `--dataset`: Dataset to evaluate (e.g., gist, lipo, melanoma)
- `--epochs`: Number of training epochs
- `--freeze`: Freeze SAM-Med3D encoder (only train classification head)

**Output**: Classification accuracy and feature quality metrics.

---

### `test_full_pipeline.py`
End-to-end test of the complete Med3Tab-PFN pipeline.

**Usage**:
```bash
python scripts/evaluation/test_full_pipeline.py
```

**Purpose**: Validates preprocessing, feature extraction, and classification steps.

---

### `compute_sam_dice_scores.py`
Compute Dice scores for SAM-Med3D segmentations.

**Usage**:
```bash
python scripts/evaluation/compute_sam_dice_scores.py
```

**Output**: Dice coefficient statistics for segmentation quality.

---

### `compute_dice_scores_roi.py`
Compute Dice scores specifically for ROI-cropped volumes.

**Usage**:
```bash
python scripts/evaluation/compute_dice_scores_roi.py
```

---

## Validation Scripts (`validation/`)

Verification utilities to ensure correctness of implementations.

### `verify_embeddings_fix.py`
Verify that embedding extraction is producing correct outputs.

**Usage**:
```bash
python scripts/validation/verify_embeddings_fix.py
```

**Purpose**: Validates embedding dimensions and consistency.

---

### `verify_pooling_logic.py`
Verify correctness of pooling strategy implementations.

**Usage**:
```bash
python scripts/validation/verify_pooling_logic.py
```

**Purpose**: Ensures pooling operations (avg, multiscale, percentile) work correctly.

---

## Visualization Scripts (`visualization/`)

Scripts for generating figures and visualizations.

### `generate_feature_extraction_pipeline_figure.py`
Generate publication-quality pipeline diagram.

**Usage**:
```bash
python scripts/visualization/generate_feature_extraction_pipeline_figure.py
```

**Output**: Saves pipeline figure to `figures/` directory.

---

### `visualize_sam_segmentations.py`
Visualize SAM-Med3D segmentation results.

**Usage**:
```bash
python scripts/visualization/visualize_sam_segmentations.py
```

**Output**: Visual overlays of segmentations on medical images.

---

## Usage Guidelines

### Running Scripts

All scripts should be run from the project root directory:

```bash
# From project root
cd Med3Tab-PFN/
python scripts/<category>/<script_name>.py
```

### Adding New Scripts

When adding new scripts:
1. Place them in the appropriate category subdirectory
2. Add documentation to this README
3. Include usage examples and expected outputs
4. Use descriptive names that indicate purpose

### Script Categories

- **Analysis**: Data exploration, statistics, result summarization
- **Evaluation**: Model testing, performance metrics, benchmarking
- **Validation**: Code verification, correctness checks, unit tests
- **Visualization**: Figure generation, plotting, visual analysis

---

## See Also

- [Main README](../README.md)
- [Documentation](../docs/)
- [Cluster Scripts](../cluster_scripts/)

---

**Back to main**: [../README.md](../README.md)
