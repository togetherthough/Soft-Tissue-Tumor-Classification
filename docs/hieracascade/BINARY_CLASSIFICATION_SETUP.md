# Binary Classification Setup

**Date**: 2025-10-06  
**Task**: Binary classification of tumors (benign vs. malignant)  
**Status**: ✅ CONFIGURED AND READY

---

## Overview

The HieraCascade model has been simplified for **pure binary classification**:
- **2 classes**: Benign (0) vs. Malignant (1)
- **Target column**: `Diagnosis_binary` from `sheet.csv`
- **No tumor type hierarchy** - just benign/malignant distinction
- **Uses MIL** for crop aggregation in Stage-2

---

## Changes Made

### 1. Simplified Class Mappings

**File**: `hieracascade/dataio/utils.py`

```python
# Binary classification only
BINARY_TO_IDX = {
    'benign': 0,
    'malignant': 1,
}

IDX_TO_BINARY = {
    0: 'benign',
    1: 'malignant',
}
```

### 2. Updated Stage-1 Model

**File**: `hieracascade/models/stage1.py`

- **Input**: Full 3D volume
- **Output**: 
  - Binary logits (benign vs. malignant)
  - Saliency map for crop proposals
- **n_classes**: 2 (default)

### 3. Updated Stage-2 Model

**File**: `hieracascade/models/stage2.py`

- **Input**: K crops from saliency map
- **Pooling**: MIL aggregation (Set Transformer or Attention MIL)
- **Output**:
  - Binary logits (benign vs. malignant)
  - Study embedding
- **n_classes**: 2 (default)
- **Removed**: Coarse/fine hierarchy (single binary head only)

---

## Architecture

### Stage-1: Scout Network

```
Input: Full Volume (1, D, H, W)
   ↓
3D Swin Transformer (Tiny)
   ↓
   ├─→ Binary Classification Head → logits (2,)
   └─→ Saliency Head → saliency map (D, H, W)
```

**Purpose**:
- Quick binary prediction
- Generate saliency map to find lesion regions
- Propose K crops for Stage-2

### Stage-2: Expert Network

```
Input: K Crops (K, 1, S, S, S)
   ↓
For each crop:
   3D Swin Transformer (Base) → crop embedding
   ↓
MIL Pooling (Set Transformer or Attention)
   ↓
Study Embedding
   ↓
Binary Classification Head → logits (2,)
```

**Purpose**:
- Detailed analysis of crops
- MIL aggregation across multiple regions
- Final binary prediction (benign vs. malignant)

---

## Label Conversion

Your `Diagnosis_binary` column contains numeric values that are automatically converted:

### Original Values → Converted Labels

```python
0  → 'benign'    → class index 0
1  → 'malignant' → class index 1
-1 → 'unknown'   → class index 1 (treated as malignant)
```

**Note**: Unknown cases (-1) are treated as malignant for simplicity. You can modify this in `hieracascade/dataio/sheet_loader.py` if needed.

---

## Training Configuration

### Stage-1 Config

**File**: `hieracascade/configs/stage1.yaml`

```yaml
model:
  n_classes: 2  # Binary classification
  backbone: 'swin3d_t'
  embed_dim: 96

train:
  epochs: 20
  batch_size: 2
  lr: 1e-4
  loss:
    classification_weight: 1.0
    saliency_weight: 0.1
```

### Stage-2 Config

**File**: `hieracascade/configs/stage2.yaml`

```yaml
model:
  n_classes: 2  # Binary classification
  backbone: 'swin3d_b'
  embed_dim: 768
  pooling: 'set_transformer'

train:
  epochs: 40
  batch_size: 1
  lr: 5e-5

proposals:
  K: 8  # Number of crops per study
  crop_size: 96
  min_intensity: 0.1
```

---

## Data Loading

### Expected Structure

```
data/
├── sheet.csv  # Contains Diagnosis_binary column
├── crlm/
│   ├── CRLM-001_CT/
│   │   └── 1/NIFTI/image.nii.gz
│   └── ...
├── desmoid/
├── gist/
├── lipo/
├── liver/
└── melanoma/
```

### Label Semantics

The `Diagnosis_binary` column in your data has different meanings per dataset:
- **CRLM**: 0=responding, 1=not responding
- **GIST**: 0=mimic, 1=actual GIST
- **Desmoid**: 0=other sarcoma, 1=DTF
- **Lipo**: 0=benign, 1=malignant
- **Melanoma**: 0=benign, 1=malignant

**For binary classification**, we treat:
- `0` → **benign** (non-aggressive, responding, or benign mimic)
- `1` → **malignant** (aggressive, not responding, or malignant tumor)

---

## Training Commands

### Option 1: Quick Start (Both Stages)

```bash
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis_binary \
    --study_id_col Subject \
    --output_dir outputs/binary_classification \
    --fold 0 \
    --device cuda
```

### Option 2: Train Stages Separately

**Stage-1:**
```bash
python -m hieracascade.train_stage1 \
    --config hieracascade/configs/stage1.yaml \
    --data_root data \
    --labels_csv labels.csv \
    --output_dir outputs/stage1/fold0 \
    --fold 0
```

**Stage-2:**
```bash
python -m hieracascade.train_stage2 \
    --config hieracascade/configs/stage2.yaml \
    --data_root data \
    --labels_csv labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0 \
    --fold 0
```

**Evaluate:**
```bash
python -m hieracascade.evaluate \
    --checkpoint outputs/stage2/fold0/checkpoint_best.pt \
    --stage stage2 \
    --data_root data \
    --labels_csv labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0/eval \
    --fold 0
```

---

## Evaluation Metrics

For binary classification, the model will report:

### Classification Metrics
- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Average of per-class accuracy
- **Precision**: Positive predictive value
- **Recall** (Sensitivity): True positive rate
- **Specificity**: True negative rate
- **F1 Score**: Harmonic mean of precision and recall

### Probabilistic Metrics
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **Average Precision**: Summary of precision-recall curve

### Clinical Metrics
- **Sensitivity at 90% Specificity**: Operating point for clinical use
- **Sensitivity at 95% Specificity**: Conservative operating point
- **PPV** (Positive Predictive Value)
- **NPV** (Negative Predictive Value)

---

## Loss Functions

### Stage-1 Loss

```python
L1 = L_cls + λ_saliency * L_saliency

# Classification loss (CrossEntropy for 2 classes)
L_cls = CrossEntropyLoss(logits, labels)

# Saliency sparsity loss (encourage sparse attention)
L_saliency = mean(saliency_map)
```

### Stage-2 Loss

```python
L2 = CrossEntropyLoss(logits, labels)

# For binary classification (2 classes)
# Or use BCEWithLogitsLoss if treating as single sigmoid output
```

---

## MIL Strategies

Stage-2 uses Multiple Instance Learning (MIL) to aggregate crops:

### Available Pooling Methods

1. **Set Transformer** (default)
   - Attention-based permutation-invariant
   - Learns to weight important crops
   - Best for variable number of informative regions

2. **Attention MIL**
   - Simpler attention mechanism
   - Faster than Set Transformer
   - Good baseline

3. **Gated Attention MIL**
   - Adds gating mechanism
   - More robust to noisy crops

### Pooling Configuration

In `stage2.yaml`:
```yaml
model:
  pooling: 'set_transformer'  # or 'attention_mil' or 'gated_attention_mil'
  pooling_kwargs:
    num_heads: 4  # For Set Transformer
    num_inds: 8   # Number of inducing points
```

---

## Notebook Usage

**File**: `notebooks/hieracascade_final.ipynb`

The notebook is already configured for binary classification:

```python
# Configuration
LABEL_COLUMN = 'Diagnosis_binary'
SHEET_CSV = '../data/sheet.csv'
STUDY_ID_COL = 'Subject'

# Automatic conversion
# 0 → benign, 1 → malignant, -1 → unknown
```

---

## Expected Outputs

After training:

### Stage-1 Outputs
```
outputs/stage1/fold0/
├── checkpoint_best.pt         # Best model weights
├── checkpoint_last.pt          # Latest model weights
├── training_log.csv            # Loss curves
├── plots/
│   └── training_curves.png     # Loss/accuracy plots
└── visualizations/
    ├── study_001_saliency.png  # Saliency maps
    └── ...
```

### Stage-2 Outputs
```
outputs/stage2/fold0/
├── checkpoint_best.pt
├── checkpoint_last.pt
├── training_log.csv
├── plots/
│   └── training_curves.png
└── eval/
    ├── predictions.csv         # Per-study predictions
    ├── confusion_matrix.png    # Binary confusion matrix
    ├── roc_curve.png           # ROC curve
    ├── pr_curve.png            # Precision-recall curve
    └── metrics.json            # All metrics
```

---

## Class Imbalance Handling

Your dataset may have class imbalance. Options:

### 1. Class Weights (Recommended)

In training config:
```yaml
train:
  class_weights: [0.7, 1.3]  # Weight for [benign, malignant]
  # Or use 'balanced' for automatic calculation
```

### 2. Focal Loss

Replace CrossEntropy with Focal Loss:
```python
from .losses import FocalLoss

criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### 3. Resampling

Oversample minority class or undersample majority:
```python
from .dataio import create_balanced_sampler

sampler = create_balanced_sampler(dataset)
loader = DataLoader(dataset, sampler=sampler)
```

---

## Performance Expectations

### Dataset Size
- 930 studies total
- Expected ~50/50 benign/malignant split (check with analysis)

### Training Time (GPU)
- **Stage-1**: 10-20 minutes per epoch
- **Stage-2**: 30-60 minutes per epoch
- **Total**: 8-12 hours for complete training

### Expected Metrics
- **Accuracy**: 75-85% (baseline goal)
- **ROC-AUC**: 0.80-0.90 (good discrimination)
- **F1 Score**: 0.75-0.85

---

## Troubleshooting

### Issue: All predictions are same class

**Cause**: Severe class imbalance or learning rate too high

**Solution**:
1. Check class distribution in training data
2. Add class weights
3. Reduce learning rate
4. Use Focal Loss

### Issue: Low saliency quality

**Cause**: Saliency weight too low or too high

**Solution**:
1. Adjust `saliency_weight` in config (try 0.05-0.2 range)
2. Visualize saliency maps to inspect
3. Ensure lesions are visible in images

### Issue: Stage-2 worse than Stage-1

**Cause**: Poor crop proposals or overfitting

**Solution**:
1. Check saliency maps quality
2. Increase number of crops (K)
3. Add more dropout
4. Use stronger data augmentation

---

## Next Steps After Training

1. **Evaluate on validation set**
   ```bash
   python -m hieracascade.evaluate --checkpoint ... --fold 0
   ```

2. **Cross-validation**
   ```bash
   for fold in 0 1 2 3 4; do
       python -m hieracascade.quick_start --fold $fold
   done
   ```

3. **Error analysis**
   - Inspect false positives
   - Inspect false negatives
   - Look for patterns in errors

4. **Threshold tuning**
   - Find optimal operating point
   - Balance sensitivity/specificity for clinical use

5. **External validation** (future)
   - Test on new dataset
   - Assess generalization

---

## Summary

✅ **Model**: Binary classification (benign vs. malignant)  
✅ **Architecture**: 2-stage cascade with MIL  
✅ **Classes**: 2 (benign=0, malignant=1)  
✅ **Target**: `Diagnosis_binary` column  
✅ **Configuration**: Ready to train  
✅ **Notebook**: `hieracascade_final.ipynb`  

**You're ready to start training!** 🚀

```bash
# Start here:
python -m hieracascade.quick_start --data_root data --fold 0
```
