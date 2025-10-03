# Label Column Selection Guide

This guide explains how to select different label columns from your `sheet.csv` file for training HieraCascade models.

## Available Label Columns

Your dataset supports two label configurations:

1. **`Diagnosis`** - Multi-class classification (fine-grained tumor types)
   - melanoma, crlm, gist, lipo, desmoid, liver, etc.
   - Use for detailed tumor classification

2. **`Diagnosis_binary`** - Binary classification (malignant vs benign)
   - malignant, benign
   - Use for simplified clinical decision making

## Usage

### Command Line

#### Quick Start Script

```bash
# Multi-class classification (default)
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis \
    --output_dir outputs/multiclass \
    --fold 0

# Binary classification
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis_binary \
    --output_dir outputs/binary \
    --fold 0
```

#### Training Individual Stages

**Stage 1:**
```bash
# First, create the labels file with your chosen column
python -c "
from hieracascade.dataio import create_index_from_sheet
create_index_from_sheet(
    data_root='data',
    sheet_path='data/sheet.csv',
    label_column='Diagnosis',  # or 'Diagnosis_binary'
    output_csv='labels_multiclass.csv'
)
"

# Then train
python -m hieracascade.train_stage1 \
    --config hieracascade/configs/stage1.yaml \
    --data_root data \
    --labels_csv labels_multiclass.csv \
    --output_dir outputs/stage1/multiclass \
    --fold 0
```

**Stage 2:**
```bash
python -m hieracascade.train_stage2 \
    --config hieracascade/configs/stage2.yaml \
    --data_root data \
    --labels_csv labels_multiclass.csv \
    --stage1_ckpt outputs/stage1/multiclass/checkpoint_best.pt \
    --output_dir outputs/stage2/multiclass \
    --fold 0
```

### Jupyter Notebook

```python
from hieracascade.dataio import create_index_from_sheet, create_site_held_out_splits

# Option 1: Multi-class Classification
index_multiclass = create_index_from_sheet(
    data_root='data',
    sheet_path='data/sheet.csv',
    label_column='Diagnosis',
    output_csv='labels_multiclass.csv'
)

print(f"Multi-class: {len(index_multiclass)} studies loaded")
print(f"Classes found: {set([item['category'] for item in index_multiclass])}")

# Create stratified splits
splits = create_site_held_out_splits(index_multiclass, stratified=True)
train_idx, val_idx = splits[0]
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Option 2: Binary Classification
index_binary = create_index_from_sheet(
    data_root='data',
    sheet_path='data/sheet.csv',
    label_column='Diagnosis_binary',
    output_csv='labels_binary.csv'
)

print(f"Binary: {len(index_binary)} studies loaded")
print(f"Classes found: {set([item['category'] for item in index_binary])}")
```

### Python Script

```python
#!/usr/bin/env python
"""Example: Train models for both multi-class and binary classification"""

from pathlib import Path
from hieracascade.dataio import create_index_from_sheet
from hieracascade.train_stage1 import train_stage1
import yaml

# Configuration
DATA_ROOT = 'data'
SHEET_CSV = 'data/sheet.csv'
FOLD = 0

# Load config
with open('hieracascade/configs/stage1.yaml') as f:
    config = yaml.safe_load(f)

# 1. Train multi-class model
print("=" * 60)
print("Training Multi-class Model")
print("=" * 60)

index_multi = create_index_from_sheet(
    data_root=DATA_ROOT,
    sheet_path=SHEET_CSV,
    label_column='Diagnosis',
    output_csv='labels_multiclass.csv'
)

train_stage1(
    config=config,
    data_root=DATA_ROOT,
    labels_csv='labels_multiclass.csv',
    output_dir='outputs/multiclass/stage1',
    fold=FOLD
)

# 2. Train binary model
print("\n" + "=" * 60)
print("Training Binary Model")
print("=" * 60)

index_binary = create_index_from_sheet(
    data_root=DATA_ROOT,
    sheet_path=SHEET_CSV,
    label_column='Diagnosis_binary',
    output_csv='labels_binary.csv'
)

train_stage1(
    config=config,
    data_root=DATA_ROOT,
    labels_csv='labels_binary.csv',
    output_dir='outputs/binary/stage1',
    fold=FOLD
)

print("\n✓ Both models trained successfully!")
```

## Stratified Sampling Verification

The pipeline automatically uses stratified sampling for train/validation splits. You'll see output like:

```
Fold 0: Using stratified sampling within training sites
  Train size: 120, Val site: hospital_a (30 samples)
  Class distribution in train: {0: 25, 1: 35, 2: 20, 3: 40}

Found 4 unique categories: ['crlm', 'gist', 'lipo', 'melanoma']
```

This ensures:
- ✅ Class balance is maintained in training set
- ✅ Each class has sufficient representation
- ✅ Site-specific validation for domain generalization

## Class Hierarchy Mapping

When using multi-class labels, the hierarchy is automatically applied:

```python
# Fine-grained → Coarse mapping
CLASS_HIERARCHY = {
    'melanoma': 'malignant',
    'crlm': 'malignant',
    'gist': 'malignant',
    'lipo': 'benign',
    'desmoid': 'benign',
    'liver': 'other',
}
```

For binary classification with `Diagnosis_binary`, you typically have:
- `malignant` → malignant
- `benign` → benign

## Troubleshooting

### Wrong Column Name

**Error:**
```
ValueError: Column 'Target' not found in sheet.csv.
Available columns: ['case_id', 'Diagnosis', 'Diagnosis_binary', 'modality', 'site'].
Use --label_column to specify one of: 'Diagnosis', 'Diagnosis_binary', etc.
```

**Solution:** Use the exact column name as shown in the error message.

### Unknown Categories

**Warning:**
```
Warning: Unknown category 'sarcoma' for study Study-001, skipping
```

**Solution:** Make sure your category names match those defined in `hieracascade/dataio/utils.py`:
- Update `FINE_TO_IDX` to include new categories
- Update `CLASS_HIERARCHY` to map them to coarse families

### Insufficient Samples for Stratification

**Warning:**
```
Fold 0: Skipping stratification (insufficient samples per class)
  Train size: 15, Val site: hospital_c (5 samples)
```

**What it means:** Some classes have fewer than 2 samples, so stratified sampling is skipped.

**Solution:** This is just a warning. The split will still work, but stratification won't be applied.

## Best Practices

1. **Always specify the label column explicitly** when training to avoid confusion:
   ```bash
   --label_column Diagnosis
   ```

2. **Use separate output directories** for different label types:
   ```bash
   --output_dir outputs/multiclass  # for Diagnosis
   --output_dir outputs/binary      # for Diagnosis_binary
   ```

3. **Check the printed statistics** after loading to verify correct categories:
   ```
   Found 6 unique categories: ['crlm', 'desmoid', 'gist', 'lipo', 'liver', 'melanoma']
   ```

4. **Keep separate label CSV files** for reproducibility:
   ```python
   output_csv='labels_multiclass.csv'  # for Diagnosis
   output_csv='labels_binary.csv'      # for Diagnosis_binary
   ```

## Example Workflows

### Workflow 1: Compare Multi-class vs Binary

```bash
# Train both models and compare performance
for label in Diagnosis Diagnosis_binary; do
    python -m hieracascade.quick_start \
        --label_column $label \
        --output_dir outputs/$label \
        --fold 0
done

# Evaluate both
python -m hieracascade.evaluate \
    --checkpoint outputs/Diagnosis/stage2/fold0/checkpoint_best.pt \
    --stage stage2 \
    --labels_csv outputs/Diagnosis/labels.csv \
    --output_dir results/multiclass

python -m hieracascade.evaluate \
    --checkpoint outputs/Diagnosis_binary/stage2/fold0/checkpoint_best.pt \
    --stage stage2 \
    --labels_csv outputs/Diagnosis_binary/labels.csv \
    --output_dir results/binary
```

### Workflow 2: Cross-Validation with Specific Label

```bash
# Run 4-fold CV for multi-class classification
for fold in 0 1 2 3; do
    python -m hieracascade.quick_start \
        --label_column Diagnosis \
        --output_dir outputs/multiclass \
        --fold $fold \
        --device cuda
done
```

### Workflow 3: Notebook Exploration

```python
# Compare label distributions
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/sheet.csv')

# Multi-class distribution
print("Multi-class (Diagnosis):")
print(df['Diagnosis'].value_counts())

# Binary distribution
print("\nBinary (Diagnosis_binary):")
print(df['Diagnosis_binary'].value_counts())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Diagnosis'].value_counts().plot(kind='bar', ax=axes[0], title='Multi-class')
df['Diagnosis_binary'].value_counts().plot(kind='bar', ax=axes[1], title='Binary')
plt.tight_layout()
plt.show()
```

## Summary

- Use `--label_column Diagnosis` for **multi-class** tumor classification
- Use `--label_column Diagnosis_binary` for **binary** malignant/benign classification
- Stratified sampling is **enabled by default** and verified in console output
- Keep separate output directories for different label configurations
- Check printed statistics to verify correct categories are loaded

For questions or issues, refer to the main documentation in `hieracascade/README.md` and `hieracascade/TUTORIAL.md`.
