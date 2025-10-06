# Dataset Label Analysis

## ❌ The Binary MIL Simplification Does NOT Apply

### Your Data Structure

Based on analysis of `sheet.csv`:

**Total Studies**: 930

**Tumor Types** (Dataset column):
- CRLM: ~150 studies
- Desmoid: ~350 studies  
- GIST: ~130 studies
- Lipo: ~250 studies
- Liver: ~30 studies
- Melanoma: ~20 studies

**Binary Labels** (Diagnosis_binary column):
- Label 0: ~494 studies
- Label 1: ~436 studies

---

## Key Finding: This is NOT "Tumor vs. No-Tumor"

### ❌ What the Binary MIL Paper Assumes
- Binary classification: **"tumor present" vs. "no tumor"**
- All positive cases are the SAME thing (just "has tumor")
- Simple binary label applies to all datasets

### ✅ What Your Data Actually Has
- **ALL cases are tumors** (no healthy controls)
- Binary label meaning is **DATASET-SPECIFIC**:
  - **CRLM**: 0 = rHGP (responding to therapy), 1 = dHGP (not responding)
  - **Desmoid**: 0 = non-DTF (other sarcomas), 1 = DTF (desmoid fibromatosis)
  - **GIST**: 0 = non-GIST (GIST mimics), 1 = GIST (actual GIST)
  - **Lipo**: 0 = Lipoma (benign), 1 = WDLPS (well-differentiated liposarcoma)
  - **Melanoma**: 0 = benign, 1 = malignant
  - **Liver**: Unknown mapping

The "0" in CRLM means something completely different from "0" in GIST!

---

## Why the Binary MIL Simplification Doesn't Work

### Problem 1: Label Semantics Are Dataset-Specific
```
Patient A: CRLM, Binary=0 → "Responding to therapy"
Patient B: GIST, Binary=0 → "Not actually GIST, it's a mimic"
```

Training a single binary classifier would mix these incompatible concepts.

### Problem 2: You're Losing Valuable Structure
Your data has:
1. **Tumor families** (malignant sarcomas vs. benign tumors vs. metastases)
2. **Specific types** (CRLM, GIST, Desmoid, etc.)
3. **Within-type subtypes** (rHGP vs. dHGP, DTF vs. non-DTF)

A pure binary model throws away the tumor type information entirely.

### Problem 3: No "Negative" Cases
Binary MIL works when you have:
- Positive studies: contain lesion(s)
- Negative studies: completely clean, no lesions

Your data:
- All studies contain tumors
- The binary label differentiates BETWEEN tumor types/grades, not presence/absence

---

## What You Should Do Instead

### Option 1: Multi-Task Learning (Recommended)

Train a model with **two heads**:

1. **Primary head**: Multi-class tumor type classification (6 classes: CRLM, Desmoid, GIST, Lipo, Liver, Melanoma)
2. **Auxiliary head**: Dataset-specific binary classification

**Architecture**:
```
Backbone (3D ResNet) 
    ↓
    ├─→ Head 1: Tumor type (6 classes) → Cross-entropy loss
    └─→ Head 2: Binary label (1 class per dataset) → BCE loss with masking
```

**Loss**:
```python
# For each sample
loss_tumor_type = CE(pred_type, dataset_label)  # Always computed

# Binary loss only for relevant dataset
if dataset == 'CRLM':
    loss_binary_crlm = BCE(pred_binary, diagnosis_binary)
elif dataset == 'GIST':
    loss_binary_gist = BCE(pred_binary, diagnosis_binary)
...

total_loss = loss_tumor_type + λ * loss_binary_specific
```

### Option 2: Keep Hierarchical Design (Current Approach)

**Stage-1**:
- Coarse families: malignant vs. benign vs. metastasis
- This naturally groups similar tumors

**Stage-2**:
- Fine-grained: 6 tumor types
- Optional: Add per-dataset binary head

**Advantage**: Interpretable, aligns with clinical reasoning

### Option 3: Dataset-Specific Models

Train 6 separate binary classifiers:
- Model 1: CRLM only (rHGP vs. dHGP)
- Model 2: GIST only (GIST vs. non-GIST)
- ...

**Disadvantage**: Can't share knowledge across datasets, requires more data per model

---

## Recommendation

**Keep the hierarchical cascade**, but adapt it for your labels:

### Modified HieraCascade for Your Data

**Stage-1 (Scout Network)**:
- **Output 1**: Tumor family (malignant / benign / other)
  - CRLM, GIST, Melanoma (malignant) → malignant
  - Desmoid, Lipo → depends on binary label
- **Output 2**: Saliency map for crop proposals

**Stage-2 (Expert Network)**:
- **Output 1**: Tumor type (6 classes: CRLM, Desmoid, GIST, Lipo, Liver, Melanoma)
- **Output 2** (optional): Dataset-specific binary refinement

**Loss**:
```python
# Stage-1
L1 = CE(coarse_pred, family_label) + λ_s * saliency_loss

# Stage-2
L2_main = CE(fine_pred, dataset_label)  # Tumor type

# Optional: add binary auxiliary loss
L2_binary = BCE(binary_pred, diagnosis_binary)  # Only train on specific dataset

L2 = L2_main + λ_b * L2_binary
```

---

## Summary

| Feature | Binary MIL Paper | Your Dataset |
|---------|------------------|--------------|
| **Task** | Tumor vs. no tumor | Tumor type classification |
| **Negative cases** | Yes (healthy scans) | No (all are tumors) |
| **Binary label** | Same meaning for all | Different meaning per dataset |
| **Structure** | Flat binary | Multi-level hierarchy |
| **Recommendation** | ❌ Does not apply | ✅ Keep hierarchical OR multi-task |

---

## Action Items

1. **DON'T** simplify to pure binary classification - you'll lose important information
2. **DO** keep the hierarchical cascade design
3. **CONSIDER** adding dataset-specific binary heads as auxiliary tasks
4. **ALTERNATIVE**: Multi-task learning with shared backbone

---

## Questions for Clarification

Before making changes, please clarify:

1. **What is your PRIMARY goal?**
   - Classify tumor TYPES (CRLM vs. GIST vs. Desmoid)?
   - Predict the binary label within each dataset?
   - Both?

2. **How will you use the model clinically?**
   - "Tell me what tumor type this is" → Keep multi-class
   - "Is this CRLM responding to therapy?" → Dataset-specific binary
   - "Is this lesion malignant?" → Need different labels

3. **Do you want to train**:
   - One model for all datasets? (current approach)
   - Separate models per dataset?
   - One shared backbone, dataset-specific heads?

**Based on your current setup using `Diagnosis_binary`, the model is already configured correctly for this task.**
