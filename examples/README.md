# Examples

This directory contains minimal working examples for common tasks.

## Test SAM-Med3D Feature Quality

**File**: `test_sam_features_minimal.py`

**Purpose**: Quickly test if SAM-Med3D features are good for tumor classification.

**Usage**:
```bash
python examples/test_sam_features_minimal.py
```

**What it does**:
1. Prepares the GIST dataset
2. Loads labels from sheet.csv
3. Trains a simple classification head on SAM-Med3D features
4. Reports AUC and interpretation

**Expected output**:
```
===========================================================
RESULTS
===========================================================
Best Validation AUC: 0.8234
Final Accuracy:      0.8000

✅ GOOD: Features are discriminative!
   → Proceed with TabPFN/LoCalPFN pipeline
```

**Time**: ~10 minutes on GPU, ~1 hour on CPU

## More Examples

For comprehensive examples, see:
- `notebooks/Test-SAM-Features.ipynb` - Interactive notebook with visualizations
- `scripts/test_sam_features.py` - Full-featured command-line script
- `notebooks/Med3D-TabPFN.ipynb` - Complete Med3Pipe pipeline
