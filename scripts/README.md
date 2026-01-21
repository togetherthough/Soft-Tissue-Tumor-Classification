# Utility Scripts

Collection of utility and analysis scripts for the Med3Tab-PFN project.

---

## Directory Structure

```
scripts/
├── analysis/          # Data analysis and statistics
├── evaluation/        # Model evaluation and testing
├── validation/        # Verification and validation utilities
├── visualization/     # Visualization and figure generation
└── README.md         # This file
```

---

## Analysis Scripts (`analysis/`)

Scripts for analyzing datasets, lesion characteristics, and experimental results.

### `analyze_labels.py`
Analyze dataset structure and label distribution from `sheet.csv`.

**Usage**:
```bash
python scripts/analysis/analyze_labels.py
```

**Output**: Prints dataset statistics and label distributions.

---

### `analyze_lesion_sizes.py`
Analyze lesion size distributions across datasets.

**Usage**:
```bash
python scripts/analysis/analyze_lesion_sizes.py
```

**Output**: Statistical analysis of lesion dimensions and volumes.

---

### `summarize_lesion_shapes.py`
Generate summary statistics for lesion morphology.

**Usage**:
```bash
python scripts/analysis/summarize_lesion_shapes.py
```

---

### `create_dice_summary.py`
Create summary of Dice score results from experiments.

**Usage**:
```bash
python scripts/analysis/create_dice_summary.py
```

---

### `create_per_dataset_summary.py`
Generate per-dataset performance summaries.

**Usage**:
```bash
python scripts/analysis/create_per_dataset_summary.py
```

---

### `create_simple_summary.py`
Create simplified summary reports of experimental results.

**Usage**:
```bash
python scripts/analysis/create_simple_summary.py
```

---

### `trace_index_mapping.py`
Debug utility for tracing sample index mappings through preprocessing pipeline.

**Usage**:
```bash
python scripts/analysis/trace_index_mapping.py
```

---

## Evaluation Scripts (`evaluation/`)

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
