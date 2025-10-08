# Two Cascade Pipelines Explained

**Date**: 2025-10-06  
**Status**: ✅ Both pipelines ready

---

## Overview

You now have **TWO cascade pipelines**, both using Stage-1 + Stage-2 + MIL, but for different tasks:

| Pipeline | Task | Label Column | Classes | Architecture |
|----------|------|--------------|---------|--------------|
| **HieraCascade** | Multi-class tumor types | `Dataset` | 6 fine + 3 coarse | Hierarchical (dual heads) |
| **BinaryCascade** | Benign vs. malignant | `Diagnosis_binary` | 2 binary | Simplified (single head) |

---

## Pipeline 1: HieraCascade (Multi-class with Hierarchy)

### Task
Classify **tumor types**: CRLM, Desmoid, GIST, Lipo, Liver, Melanoma

### Label Column
`Dataset` from `sheet.csv`

### Architecture

**Stage-1 (Scout)**:
```
Full Volume (1, D, H, W)
    ↓
3D Swin Transformer
    ↓
    ├─→ Coarse Classification (3 classes: malignant/benign/other)
    └─→ Saliency Map
```

**Stage-2 (Expert)**:
```
K Crops from saliency map
    ↓
For each crop:
    3D Swin Transformer → Crop Embedding
    ↓
MIL Pooling (Set Transformer)
    ↓
Study Embedding
    ↓
    ├─→ Fine Classification (6 tumor types)
    └─→ Coarse Classification (3 families) ← HIERARCHICAL
```

### Model Class
- `Stage2HierarchicalModel` in `stage2_hierarchical.py`
- Dual heads: `head_fine` (6 classes) + `head_coarse` (3 classes)

### Class Mappings
```python
# Fine-grained (tumor types)
FINE_TO_IDX = {
    'melanoma': 0,
    'crlm': 1,
    'gist': 2,
    'lipo': 3,
    'desmoid': 4,
    'liver': 5,
}

# Coarse (families)
COARSE_TO_IDX = {
    'malignant': 0,
    'benign': 1,
    'other': 2,
}

# Hierarchy
CLASS_HIERARCHY = {
    'melanoma': 'malignant',
    'crlm': 'malignant',
    'gist': 'malignant',
    'lipo': 'benign',
    'desmoid': 'benign',
    'liver': 'other',
}
```

### Training Command
```bash
python -m hieracascade.quick_start_hierarchical \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Dataset \
    --study_id_col Subject \
    --output_dir outputs/hieracascade \
    --fold 0
```

### Loss Function
```python
# Stage-1
L1 = CE(coarse_logits, coarse_labels) + λ_saliency * saliency_loss

# Stage-2 (hierarchical)
L2_fine = CE(fine_logits, fine_labels)
L2_coarse = CE(coarse_logits, coarse_labels)
L2_consistency = KL(coarse_from_fine, coarse_logits)  # Enforce consistency
L2 = L2_fine + α * L2_coarse + β * L2_consistency
```

### Use Case
- Research on tumor subtype classification
- When you have multiple tumor types to distinguish
- When tumor families matter (malignant vs. benign grouping)

---

## Pipeline 2: BinaryCascade (Binary Classification)

### Task
Classify **benign vs. malignant** tumors

### Label Column
`Diagnosis_binary` from `sheet.csv` (0=benign, 1=malignant)

### Architecture

**Stage-1 (Scout)**:
```
Full Volume (1, D, H, W)
    ↓
3D Swin Transformer
    ↓
    ├─→ Binary Classification (2 classes: benign/malignant)
    └─→ Saliency Map
```

**Stage-2 (Expert)**:
```
K Crops from saliency map
    ↓
For each crop:
    3D Swin Transformer → Crop Embedding
    ↓
MIL Pooling (Set Transformer)
    ↓
Study Embedding
    ↓
Binary Classification (2 classes) ← SIMPLIFIED
```

### Model Class
- `Stage2BinaryModel` in `stage2_binary.py`
- Single head: `head` (2 classes)

### Class Mappings
```python
# Binary only
BINARY_TO_IDX = {
    'benign': 0,
    'malignant': 1,
}
```

### Training Command
```bash
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis_binary \
    --study_id_col Subject \
    --output_dir outputs/binarycascade \
    --fold 0
```

### Loss Function
```python
# Stage-1
L1 = CE(binary_logits, binary_labels) + λ_saliency * saliency_loss

# Stage-2 (binary only)
L2 = CE(binary_logits, binary_labels)
```

### Use Case
- Clinical decision support (malignant vs. benign)
- Simpler, focused task
- Current goal until new dataset available

---

## Key Differences

| Aspect | HieraCascade | BinaryCascade |
|--------|--------------|---------------|
| **Task** | Multi-class (6 types) | Binary (2 classes) |
| **Labels** | `Dataset` column | `Diagnosis_binary` column |
| **Stage-2 heads** | Dual (fine + coarse) | Single (binary) |
| **Hierarchy** | Yes (fine → coarse) | No |
| **Complexity** | Higher | Lower |
| **Classes** | 6 fine, 3 coarse | 2 binary |
| **Loss** | 3 terms (fine, coarse, consistency) | 1 term (binary CE) |
| **Use case** | Research, subtypes | Clinical, screening |

---

## Files Structure

```
hieracascade/
├── models/
│   ├── stage1.py                    # Shared (works for both)
│   ├── stage2_binary.py             # BinaryCascade Stage-2 ✅
│   ├── stage2_hierarchical.py       # HieraCascade Stage-2 ✅
│   ├── __init__.py                  # Exports both ✅
│   └── ...
├── dataio/
│   ├── utils.py                     # Both mappings ✅
│   └── ...
├── quick_start.py                   # BinaryCascade (current default)
├── quick_start_hierarchical.py      # HieraCascade (to create)
└── ...
```

---

## Which Pipeline to Use?

### Use HieraCascade if:
- ✅ You want to classify tumor types (CRLM, GIST, Desmoid, etc.)
- ✅ You have labeled tumor types in your data
- ✅ You want to leverage hierarchical structure
- ✅ You have enough samples per tumor type (>20-30 per class)

### Use BinaryCascade if:
- ✅ You only care about benign vs. malignant
- ✅ You have binary labels (0/1)
- ✅ You want a simpler, faster model
- ✅ Clinical screening/triage is your goal (current task ✅)

---

## Training Both

You can train both pipelines on the same data:

### Step 1: Train HieraCascade (tumor types)
```bash
python -m hieracascade.quick_start_hierarchical \
    --label_column Dataset \
    --output_dir outputs/hieracascade \
    --fold 0
```

### Step 2: Train BinaryCascade (binary)
```bash
python -m hieracascade.quick_start \
    --label_column Diagnosis_binary \
    --output_dir outputs/binarycascade \
    --fold 0
```

### Step 3: Compare Results
- HieraCascade: Per-tumor-type accuracy
- BinaryCascade: Binary sensitivity/specificity

---

## Current Status

✅ **BinaryCascade**: Fully configured and ready
- Notebook: `hieracascade_final.ipynb` (needs minor updates)
- Model: `Stage2BinaryModel`
- Config: Using `Diagnosis_binary`

⚠️ **HieraCascade**: Model ready, needs:
- [ ] Training script: `quick_start_hierarchical.py`
- [ ] Notebook: `hieracascade_multiclass.ipynb`
- [ ] Config files for hierarchical mode

---

## Next Steps

### For BinaryCascade (Your Current Priority)
1. Update notebook terminology (minor fixes)
2. Start training with `quick_start.py`
3. Evaluate binary classification performance

### For HieraCascade (Future/Research)
1. Create `quick_start_hierarchical.py` script
2. Create separate training configs
3. Create `hieracascade_multiclass.ipynb` notebook
4. Train on `Dataset` labels

---

## Summary

You now have:
- ✅ **Two model architectures** (binary + hierarchical)
- ✅ **Two sets of class mappings** (binary + multi-class)
- ✅ **Clear separation** between pipelines
- ✅ **Flexibility** to train either one

**Current focus**: BinaryCascade for benign vs. malignant classification  
**Future option**: HieraCascade for tumor type research

Both use the same cascade + MIL framework, just different heads and labels! 🎉
