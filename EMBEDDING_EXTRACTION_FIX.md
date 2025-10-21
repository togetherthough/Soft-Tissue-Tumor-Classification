# Automatic Embedding Extraction Fix

## Problem

When running multi-dataset experiments, if you processed GIST first, then tried to add LIPO later, the LIPO embeddings were never extracted because:

1. `skip_existing_embeddings=True` was the default
2. The pipeline didn't check **per-dataset** whether embeddings existed
3. Result: GIST embeddings existed, so the pipeline assumed all datasets were done

This meant LIPO would fail silently or produce errors because its embeddings were never created.

---

## Solution

**Modified:** `med3pipe/pipelines/multi_dataset.py`

Added **per-dataset embedding checks** that automatically force extraction for any dataset missing embeddings, regardless of the `skip_existing_embeddings` setting or what order datasets are processed.

### Changes Made

#### 1. Per-Dataset Embedding Check (Lines 208-227)

```python
# Check if embeddings exist for this specific dataset
feat_train_dir = sam3d_root / "features" / category / f"{ct_name}_train"
feat_val_dir = sam3d_root / "features" / category / ct_name

embeddings_exist = (
    feat_train_dir.exists() 
    and feat_val_dir.exists()
    and list(feat_train_dir.glob("*_embedding.pt"))
    and list(feat_val_dir.glob("*_embedding.pt"))
)

# Force extraction if embeddings don't exist for this dataset
force_extraction = not embeddings_exist
skip_for_this_dataset = skip_existing_embeddings and not force_extraction

if force_extraction:
    print(f"\n⚠️  {ds_key}: Embeddings missing. Forcing extraction...")
elif skip_existing_embeddings:
    print(f"\n✅ {ds_key}: Embeddings exist. Skipping extraction...")
```

#### 2. Per-Dataset Skip Flag (Line 244)

```python
skip_existing_embeddings=skip_for_this_dataset,  # Use dataset-specific flag
```

This ensures that:
- **GIST** with existing embeddings → skips extraction ✅
- **LIPO** with missing embeddings → forces extraction ⚠️

---

## How It Works

### Before the Fix ❌

```
User runs: run_multi_tabpfn(config, skip_existing_embeddings=True)

GIST:  Has embeddings → Skip ✅
LIPO:  No embeddings → Skip anyway (global setting) ❌ BUG!
```

### After the Fix ✅

```
User runs: run_multi_tabpfn(config, skip_existing_embeddings=True)

GIST:  Has embeddings → Skip ✅
LIPO:  No embeddings → FORCE EXTRACTION ⚠️ (automatic fix!)
```

---

## Benefits

### 1. **Order-Independent**
Run datasets in any order, and each gets processed correctly:
```python
# Run GIST first
run_multi_tabpfn(config, dataset_names=["gist"])

# Run LIPO later - embeddings automatically extracted!
run_multi_tabpfn(config, dataset_names=["lipo"])
```

### 2. **Works for Both Methods**
The fix applies to **both** TabPFN and LoCalPFN:
```python
# TabPFN - automatically extracts missing embeddings
run_multi_tabpfn(config)

# LoCalPFN - also automatically extracts missing embeddings
run_multi_localpfn(config)
```

### 3. **Smart Caching**
- Datasets with embeddings → skipped (fast)
- Datasets without embeddings → extracted (automatic)
- No manual intervention needed

### 4. **Clear Feedback**
```
⚠️  lipo: Embeddings missing. Forcing extraction...
✅ gist: Embeddings exist. Skipping extraction...
```

---

## Usage Examples

### Run All Datasets (Automatic Extraction)

```python
from med3pipe.pipelines import run_multi_tabpfn

# Automatically extracts embeddings for any dataset that needs them
result = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    skip_existing_embeddings=True,  # Still efficient - only missing datasets extracted
)
```

### CLI Usage

```bash
# TabPFN - automatically handles missing embeddings
python -m med3pipe multi-tabpfn --config configs/datasets.yaml

# LoCalPFN - also automatically handles missing embeddings
python -m med3pipe multi-localpfn --config configs/datasets.yaml --local-k 50
```

### Verify Before Running

```bash
# Check which datasets need embeddings
python scripts/verify_embeddings_fix.py
```

---

## Technical Details

### File Modified
- `med3pipe/pipelines/multi_dataset.py`
  - Function: `_run_multi_core()` (lines 208-244)
  - Functions using it: `run_multi_tabpfn()`, `run_multi_localpfn()`

### Logic Flow

```
For each dataset in config:
    1. Check if embeddings exist for THIS dataset
       ├─ Check train dir: features/{category}/{ct_name}_train/*.pt
       └─ Check val dir: features/{category}/{ct_name}/*.pt
    
    2. Determine skip flag for THIS dataset:
       ├─ If embeddings missing → force_extraction=True
       └─ If embeddings exist → use global skip_existing_embeddings
    
    3. Pass dataset-specific flag to run_single_dataset()
       └─ Ensures each dataset processed correctly
```

### Backward Compatibility

✅ **Fully backward compatible**
- Existing code works without changes
- Default behavior improved (smarter)
- No breaking changes to API

---

## Verification

### Check Current Status

```bash
python scripts/verify_embeddings_fix.py
```

Output example:
```
======================================================================
EMBEDDING STATUS CHECK
======================================================================

GIST       ✅ EXISTS
  Train:  29 embeddings in SAM-Med3D-main/SAM-Med3D-main/features/gist/ct_GIST_train
  Val:     8 embeddings in SAM-Med3D-main/SAM-Med3D-main/features/gist/ct_GIST

LIPO       ❌ MISSING
  Train:   0 embeddings in SAM-Med3D-main/SAM-Med3D-main/features/lipo/ct_LIPO_train
  Val:     0 embeddings in SAM-Med3D-main/SAM-Med3D-main/features/lipo/ct_LIPO

======================================================================
⚠️  Some embeddings are missing.

🔧 FIX: Run either command to automatically extract missing embeddings:
  python -m med3pipe multi-tabpfn --config configs/datasets.yaml
  python -m med3pipe multi-localpfn --config configs/datasets.yaml
```

### After Running the Pipeline

```
GIST       ✅ EXISTS
  Train:  29 embeddings
  Val:     8 embeddings

LIPO       ✅ EXISTS
  Train:  36 embeddings  ← Now extracted!
  Val:     9 embeddings  ← Now extracted!
```

---

## Summary

### What Was Fixed
- ✅ Per-dataset embedding checks
- ✅ Automatic extraction for missing datasets
- ✅ Order-independent processing
- ✅ Works for both TabPFN and LoCalPFN
- ✅ Clear user feedback

### How to Use
Just run your multi-dataset commands as normal. The pipeline now automatically detects and extracts missing embeddings:

```bash
# Just run this - it handles everything
python -m med3pipe multi-tabpfn --config configs/datasets.yaml
```

**No more manual extraction needed. No more order dependencies. It just works.** ✨

---

**Date:** October 21, 2025  
**Fixed by:** Automatic per-dataset embedding check logic  
**Affects:** `run_multi_tabpfn()`, `run_multi_localpfn()`, and all multi-dataset workflows
