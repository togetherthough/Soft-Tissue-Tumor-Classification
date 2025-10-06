# Binary Classification Task - Complete Setup

**Date**: 2025-10-06  
**Status**: ✅ READY TO TRAIN  
**Task**: Binary tumor classification (benign vs. malignant)

---

## TL;DR

**Your current task**: Classify tumors as **benign** or **malignant** using `Diagnosis_binary` column.

**Model**: 2-stage cascade with binary classification heads  
**Classes**: 2 (benign=0, malignant=1)  
**Ready**: ✅ All code updated, ready to train

```bash
# Start training:
python -m hieracascade.quick_start --data_root data --fold 0
```

---

## Task Definition

### What You're Classifying

**Input**: 3D CT/MRI scan of a tumor  
**Output**: Binary prediction
- **Class 0**: Benign (non-aggressive, responding to therapy, or benign mimic)
- **Class 1**: Malignant (aggressive, not responding, or malignant tumor)

### Label Source

**Column**: `Diagnosis_binary` in `sheet.csv`

**Values**:
- `0` → benign
- `1` → malignant  
- `-1` → unknown (treated as malignant)

**Note**: The numeric label meaning varies by dataset:
- CRLM: therapy response
- GIST: GIST vs. mimic
- Desmoid: DTF vs. other sarcoma
- Lipo: benign vs. malignant

For this task, we **ignore** the tumor type and just use the binary label.

---

## Architecture

### Two-Stage Cascade

```
┌─────────────────────────────────────────────────────────┐
│ Stage-1: Scout Network                                  │
├─────────────────────────────────────────────────────────┤
│ Input: Full 3D Volume                                   │
│   ↓                                                      │
│ 3D Swin Transformer (Tiny)                             │
│   ↓                                                      │
│ ├─→ Binary Classification Head → (B, 2)                │
│ └─→ Saliency Head → Attention Map                      │
│                                                          │
│ Purpose: Quick prediction + crop proposals              │
└─────────────────────────────────────────────────────────┘

↓ (Top-K crops)

┌─────────────────────────────────────────────────────────┐
│ Stage-2: Expert Network                                 │
├─────────────────────────────────────────────────────────┤
│ Input: K Crops from saliency map                        │
│   ↓                                                      │
│ For each crop:                                           │
│   3D Swin Transformer (Base) → Crop Embedding          │
│   ↓                                                      │
│ MIL Pooling (Set Transformer / Attention)              │
│   ↓                                                      │
│ Study Embedding                                          │
│   ↓                                                      │
│ Binary Classification Head → (B, 2)                     │
│                                                          │
│ Purpose: Detailed analysis with MIL aggregation         │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Cascade Design
- **Stage-1**: Fast screening on full volume
- **Stage-2**: Detailed analysis on crops
- Benefits: Efficiency + accuracy

### 2. MIL (Multiple Instance Learning)
- Aggregates K crops per study
- No need for lesion annotations
- Learns which regions matter
- Pooling: Set Transformer, Attention MIL, or Gated Attention

### 3. Saliency-Guided Crops
- Stage-1 generates attention map
- Top-K salient regions selected
- Stage-2 focuses on informative areas

### 4. Modality-Aware
- Handles CT and MRI
- Modality embedding added
- Joint training across modalities

---

## Files Updated

### Models
- ✅ `hieracascade/models/stage1.py` - Binary classification (n_classes=2)
- ✅ `hieracascade/models/stage2.py` - Binary head + MIL

### Data
- ✅ `hieracascade/dataio/utils.py` - Binary mappings
- ✅ `hieracascade/dataio/sheet_loader.py` - Auto label conversion

### Documentation
- ✅ `BINARY_CLASSIFICATION_SETUP.md` - Complete guide
- ✅ `MIGRATION_TO_BINARY.md` - What changed
- ✅ `README_BINARY_TASK.md` (this file) - Quick reference

---

## Quick Start

### Option 1: Full Pipeline (Recommended)
```bash
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis_binary \
    --study_id_col Subject \
    --output_dir outputs/binary \
    --fold 0 \
    --device cuda
```

### Option 2: Jupyter Notebook
```bash
jupyter notebook notebooks/hieracascade_final.ipynb
```

### Option 3: Individual Stages
```bash
# Stage-1
python -m hieracascade.train_stage1 \
    --config hieracascade/configs/stage1.yaml \
    --data_root data \
    --labels_csv labels.csv \
    --fold 0

# Stage-2  
python -m hieracascade.train_stage2 \
    --config hieracascade/configs/stage2.yaml \
    --data_root data \
    --labels_csv labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --fold 0
```

---

## Training Configuration

### Stage-1 (Scout)
- **Backbone**: Swin3D-Tiny
- **Epochs**: 20-30
- **Batch size**: 2
- **Learning rate**: 1e-4
- **Loss**: Classification + saliency sparsity

### Stage-2 (Expert)
- **Backbone**: Swin3D-Base
- **Epochs**: 40-50
- **Batch size**: 1 (memory intensive)
- **Learning rate**: 5e-5
- **Crops**: K=8 per study
- **Crop size**: 96³ voxels

---

## Expected Results

### Dataset Statistics
- **Total**: 930 studies
- **Expected split**: ~50% benign, ~50% malignant (verify with analysis)

### Performance Targets
- **Accuracy**: 75-85%
- **ROC-AUC**: 0.80-0.90
- **Balanced Accuracy**: 70-80%
- **F1 Score**: 0.75-0.85

### Training Time (Single V100 GPU)
- **Stage-1**: 6-10 hours (20 epochs)
- **Stage-2**: 20-30 hours (40 epochs)
- **Total**: ~30-40 hours per fold

---

## Evaluation Metrics

The model will report:

### Classification
- Accuracy
- Balanced Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score

### Probabilistic
- ROC-AUC ⭐
- PR-AUC ⭐
- Average Precision

### Clinical
- Sensitivity @ 90% Specificity
- Sensitivity @ 95% Specificity
- PPV, NPV

### Visualization
- Confusion matrix (2x2)
- ROC curve
- Precision-Recall curve
- Per-study predictions

---

## Common Issues & Solutions

### 1. Class Imbalance

**Symptom**: Model predicts only one class

**Solution**:
```yaml
# In config
train:
  class_weights: [0.7, 1.3]  # Or 'balanced'
```

### 2. Poor Saliency Maps

**Symptom**: Crops miss lesions

**Solution**:
- Adjust saliency weight (0.05-0.2)
- Increase K (number of crops)
- Check data preprocessing

### 3. Overfitting

**Symptom**: Train accuracy >> validation accuracy

**Solution**:
- Add dropout
- Data augmentation
- Reduce model size
- Early stopping

### 4. Stage-2 Worse Than Stage-1

**Symptom**: Stage-2 accuracy lower

**Solution**:
- Check saliency quality
- Increase crops (K)
- Adjust pooling method
- Try different MIL strategy

---

## Data Requirements

### File Structure
```
data/
├── sheet.csv
├── <dataset>/
│   └── <Subject>_<Modality>/
│       └── 1/NIFTI/image.nii.gz
```

### Sheet.csv Columns Required
- `Subject`: Study ID
- `Diagnosis_binary`: 0 or 1 (or -1)
- `Dataset`: Tumor type (for stratification)

### Image Format
- NIfTI (.nii.gz)
- 3D volumes
- Preprocessed (normalized intensity)

---

## Next Steps

### 1. Verify Setup ✅
```bash
python -c "
from hieracascade.dataio import create_index_from_sheet
index = create_index_from_sheet(
    data_root='data',
    sheet_path='data/sheet.csv',
    label_column='Diagnosis_binary',
    study_id_col='Subject'
)
print(f'Loaded {len(index)} studies')
categories = [i['category'] for i in index]
print(f'Benign: {categories.count(\"benign\")}')
print(f'Malignant: {categories.count(\"malignant\")}')
"
```

### 2. Start Training ✅
```bash
python -m hieracascade.quick_start --data_root data --fold 0
```

### 3. Monitor Progress ✅
- Check `outputs/hieracascade/stage1/fold0/training_log.csv`
- Watch loss curves
- Inspect saliency visualizations

### 4. Evaluate ✅
```bash
python -m hieracascade.evaluate \
    --checkpoint outputs/stage2/fold0/checkpoint_best.pt \
    --stage stage2 \
    --fold 0
```

### 5. Cross-Validation (Optional) ✅
```bash
for fold in 0 1 2 3 4; do
    python -m hieracascade.quick_start --fold $fold
done
```

---

## Documentation Index

- **`BINARY_CLASSIFICATION_SETUP.md`** - Complete technical guide
- **`README_BINARY_TASK.md`** (this file) - Quick reference
- **`MIGRATION_TO_BINARY.md`** - What changed from multi-class
- **`HIERACASCADE.md`** - Main setup file
- **`notebooks/hieracascade_final.ipynb`** - Interactive notebook

---

## Summary

✅ **Task**: Binary classification (benign vs. malignant)  
✅ **Architecture**: 2-stage cascade with MIL  
✅ **Classes**: 2 (benign=0, malignant=1)  
✅ **Target**: `Diagnosis_binary` column  
✅ **Status**: Ready to train  

**Start training now:**
```bash
python -m hieracascade.quick_start --data_root data --fold 0
```

🚀 **Good luck with your binary tumor classification!**
