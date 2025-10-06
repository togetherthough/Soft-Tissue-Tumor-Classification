# Sheet.csv Compatibility Analysis & Fixes

**Date**: 2025-10-06  
**Status**: ✅ FIXED

## Issues Found

### ❌ Issue 1: Wrong Label Column Selected - "Diagnosis"
**Your configuration**: Using `Diagnosis` column  
**Actual values in Diagnosis**: `rHGP`, `dHGP`, `DTF`, `non-DTF`, `WDLPS`, `Lipoma`, `BRAF`, etc.  
**Expected by model**: `melanoma`, `crlm`, `gist`, `lipo`, `desmoid`, `liver`

**Problem**: The `Diagnosis` column contains sub-type classifications within each tumor category, not the main tumor types that the model expects.

### ❌ Issue 2: Wrong Label Column Selected - "Diagnosis_binary"
**Your configuration**: Using `Diagnosis_binary` column  
**Actual values**: `0`, `1`, `-1` (numeric)  
**Expected by model**: `malignant`, `benign` (text strings)

**Problem**: The model's classification pipeline expects text labels, not numeric codes.

### ❌ Issue 3: Column Name Mismatch
**Your CSV structure**: Uses `Subject` for case IDs  
**Model default**: Expects `case_id` column  

**Problem**: Without specifying the correct column name, the loader will fail.

---

## ✅ Correct Configuration

Your **`Dataset`** column is the correct one to use! It contains:
- `CRLM` (Colorectal Liver Metastasis)
- `Desmoid`
- `GIST` (Gastrointestinal Stromal Tumor)
- `Lipo` (Lipoma/Liposarcoma)
- `Liver`
- `Melanoma`

These match exactly what the HieraCascade model expects (after automatic lowercase conversion).

---

## Fixes Applied

### 1. Updated `quick_start.py`
- ✅ Removed hardcoded `choices` restriction on `--label_column`
- ✅ Changed default from `Diagnosis` to `Dataset`
- ✅ Added `--study_id_col` parameter (default: `Subject`)
- ✅ Updated function call to pass `study_id_col` parameter

### 2. Updated `LABEL_SELECTION_GUIDE.md`
- ✅ Documented all three label column options
- ✅ Marked `Dataset` as RECOMMENDED
- ✅ Explained incompatibility of numeric `Diagnosis_binary` values
- ✅ Updated all example commands with correct parameters

---

## How to Run (Corrected Commands)

### Quick Start - Full Pipeline
```bash
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv sheet.csv \
    --label_column Dataset \
    --study_id_col Subject \
    --output_dir outputs/tumor_types \
    --fold 0
```

### Alternative: Manual Steps

**Step 1: Create labels CSV**
```bash
python -c "
from hieracascade.dataio import create_index_from_sheet
index = create_index_from_sheet(
    data_root='data',
    sheet_path='sheet.csv',
    label_column='Dataset',
    study_id_col='Subject',
    output_csv='labels_tumor_types.csv'
)
print(f'Loaded {len(index)} studies')
"
```

**Step 2: Train Stage 1**
```bash
python -m hieracascade.train_stage1 \
    --config hieracascade/configs/stage1.yaml \
    --data_root data \
    --labels_csv labels_tumor_types.csv \
    --output_dir outputs/stage1/fold0 \
    --fold 0
```

**Step 3: Train Stage 2**
```bash
python -m hieracascade.train_stage2 \
    --config hieracascade/configs/stage2.yaml \
    --data_root data \
    --labels_csv labels_tumor_types.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0 \
    --fold 0
```

---

## Expected Output

When you run with the correct configuration, you should see:

```
Loading labels from sheet.csv...
Using label column: 'Dataset'

Found 6 unique categories: ['crlm', 'desmoid', 'gist', 'lipo', 'liver', 'melanoma']

=== Dataset Statistics ===
Total studies: <N>

By category:
  crlm: <count>
  desmoid: <count>
  gist: <count>
  lipo: <count>
  liver: <count>
  melanoma: <count>
```

---

## Data Requirements

For the model to work, your data directory should have this structure:

```
data/
├── crlm/               # Lowercase tumor type
│   ├── CRLM-001_CT/    # <Subject>_<Modality>
│   │   └── 1/NIFTI/image.nii.gz
│   └── CRLM-002_CT/
│       └── 1/NIFTI/image.nii.gz
├── desmoid/
├── gist/
├── lipo/
├── liver/
└── melanoma/
```

**Note**: The directory names must be lowercase and match the tumor categories after conversion from your `Dataset` column.

---

## Column Mapping Reference

| CSV Column          | Purpose                       | Values                               | Compatible? |
|---------------------|-------------------------------|--------------------------------------|-------------|
| `Subject`           | Study/Case ID                 | CRLM-001, Desmoid-005, etc.          | ✅ Yes       |
| `Dataset`           | **Tumor Type (MAIN LABEL)**   | CRLM, Desmoid, GIST, Lipo, etc.      | ✅ Yes       |
| `Diagnosis`         | Sub-type classification       | rHGP, dHGP, DTF, WDLPS, etc.         | ⚠️ Advanced  |
| `Diagnosis_binary`  | Binary classification         | 0, 1, -1 (numeric)                   | ❌ No        |

---

## Testing the Fix

Run this test to verify your setup:

```bash
python -c "
from hieracascade.dataio import create_index_from_sheet

try:
    index = create_index_from_sheet(
        data_root='data',
        sheet_path='sheet.csv',
        label_column='Dataset',
        study_id_col='Subject'
    )
    print(f'✅ SUCCESS: Loaded {len(index)} studies')
    categories = set([item['category'] for item in index])
    print(f'✅ Categories found: {sorted(categories)}')
except Exception as e:
    print(f'❌ ERROR: {e}')
"
```

Expected output:
```
✅ SUCCESS: Loaded <N> studies
✅ Categories found: ['crlm', 'desmoid', 'gist', 'lipo', 'liver', 'melanoma']
```

---

## Summary

✅ **Fixed**: Script now accepts correct `Dataset` column  
✅ **Fixed**: Script now accepts correct `Subject` column for case IDs  
✅ **Updated**: Default parameters now match your CSV structure  
✅ **Documented**: All options explained in LABEL_SELECTION_GUIDE.md

**You can now run the training pipeline with the corrected command above!**
