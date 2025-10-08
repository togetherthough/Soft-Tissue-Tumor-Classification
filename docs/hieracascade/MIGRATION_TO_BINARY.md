# Migration to Binary Classification

**Date**: 2025-10-06  
**Change**: Simplified from multi-class to binary classification

---

## What Changed

### Before (Multi-class)
- **Task**: Classify tumor types (CRLM, Desmoid, GIST, Lipo, Liver, Melanoma)
- **Classes**: 6+ tumor types
- **Architecture**: Hierarchical (coarse families + fine types)
- **Stage-2**: Dual heads (fine + coarse)

### After (Binary)
- **Task**: Classify benign vs. malignant
- **Classes**: 2 (benign, malignant)
- **Architecture**: Single binary classification
- **Stage-2**: Single binary head

---

## Files Modified

### 1. Class Mappings
**`hieracascade/dataio/utils.py`**
- Replaced multi-class mappings with binary
- Now: `BINARY_TO_IDX = {'benign': 0, 'malignant': 1}`

### 2. Stage-1 Model
**`hieracascade/models/stage1.py`**
- Default `n_classes=2`
- Docstring updated for binary task

### 3. Stage-2 Model
**`hieracascade/models/stage2.py`**
- Removed `n_fine` and `n_coarse` parameters
- Single parameter: `n_classes=2`
- Removed `head_fine` and `head_coarse`
- Single head: `head` for binary classification
- Forward returns: `(logits, study_embedding)` instead of `(logits_fine, logits_coarse, study_embedding)`

### 4. Builder Function
**`hieracascade/models/stage2.py:build_stage2_model()`**
- Changed signature: `build_stage2_model(n_classes=2, ...)`
- Removed `n_fine` and `n_coarse` parameters

---

## Configuration Updates Needed

### Update Stage-1 Config
**`hieracascade/configs/stage1.yaml`**

```yaml
model:
  n_classes: 2  # Changed from 6+ to 2
  backbone: 'swin3d_t'
  embed_dim: 96
```

### Update Stage-2 Config
**`hieracascade/configs/stage2.yaml`**

```yaml
model:
  n_classes: 2  # Changed from 6+ to 2
  backbone: 'swin3d_b'
  embed_dim: 768
  pooling: 'set_transformer'
```

---

## Training Code Updates Needed

### Old Training Code (Multi-class)
```python
# Stage-2 forward
logits_fine, logits_coarse, embeddings = model(crops, modality_id)

# Loss
loss_fine = CE(logits_fine, labels_fine)
loss_coarse = CE(logits_coarse, labels_coarse)
loss = loss_fine + 0.5 * loss_coarse
```

### New Training Code (Binary)
```python
# Stage-2 forward
logits, embeddings = model(crops, modality_id)

# Loss
loss = CE(logits, labels_binary)
```

---

## Evaluation Updates Needed

### Metrics to Report

**Remove** (multi-class specific):
- Per-class accuracy for 6+ classes
- Coarse family accuracy
- Fine-grained confusion matrix

**Keep/Add** (binary specific):
- Accuracy
- Balanced Accuracy
- Precision, Recall, F1
- **ROC-AUC** ⭐
- **PR-AUC** ⭐
- Sensitivity, Specificity
- PPV, NPV
- Confusion matrix (2x2)

---

## Quick Start Command

### No changes needed!
```bash
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis_binary \
    --study_id_col Subject \
    --fold 0
```

The model will automatically use binary classification since:
- `label_column=Diagnosis_binary` triggers binary mode
- Labels converted: 0→benign, 1→malignant
- Models default to `n_classes=2`

---

## Backward Compatibility

If you need multi-class mode in the future:

### Option 1: Restore Old Code
```bash
git checkout <commit> hieracascade/models/
```

### Option 2: Add Multi-class Flag
Can modify models to accept `task='binary'` or `task='multiclass'` parameter

### Option 3: Separate Branches
- `main` branch: Binary classification
- `multiclass` branch: Original hierarchical design

---

## Testing Checklist

- [ ] Stage-1 builds with `n_classes=2`
- [ ] Stage-2 builds with `n_classes=2`
- [ ] Forward pass returns correct shapes:
  - Stage-1: `(logits, saliency)` where `logits.shape = (B, 2)`
  - Stage-2: `(logits, embeddings)` where `logits.shape = (B, 2)`
- [ ] Labels load correctly (0=benign, 1=malignant)
- [ ] Loss computes without errors
- [ ] Training runs for 1 epoch
- [ ] Evaluation generates binary metrics

---

## Summary

✅ **Models**: Updated to binary (2 classes)  
✅ **Mappings**: Simplified to benign/malignant  
✅ **Architecture**: Single classification head  
✅ **MIL**: Kept for crop aggregation  
✅ **Notebook**: Ready to use  
✅ **Documentation**: `BINARY_CLASSIFICATION_SETUP.md`

**Ready to train!** 🚀
