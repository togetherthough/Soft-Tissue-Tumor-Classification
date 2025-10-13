# Notebook Update Guide

This guide explains the changes needed in `notebooks/hieracascade_full_pipeline.ipynb` to use the correct column names from your `sheet.csv` file.

## Required Changes

### 1. Update Configuration Cell (Cell with LABEL_COLUMN)

**OLD CODE:**
```python
# Configuration parameters
DATA_ROOT = '../data'
SHEET_CSV = '../data/sheet.csv'
LABEL_COLUMN = 'Diagnosis'  # Change to 'Diagnosis_binary' for binary classification
OUTPUT_DIR = '../outputs/hieracascade_notebook'
FOLD = 0
```

**NEW CODE:**
```python
# Configuration parameters
DATA_ROOT = '../data'
SHEET_CSV = '../sheet.csv'  # Updated path
LABEL_COLUMN = 'Dataset'  # RECOMMENDED: Use 'Dataset' for tumor type classification
STUDY_ID_COL = 'Subject'  # Column name for case IDs
OUTPUT_DIR = '../outputs/hieracascade_notebook'
FOLD = 0

print(f"Configuration:")
print(f"  Data root: {DATA_ROOT}")
print(f"  Sheet CSV: {SHEET_CSV}")
print(f"  Label column: {LABEL_COLUMN}")
print(f"  Study ID column: {STUDY_ID_COL}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Fold: {FOLD}")
```

---

### 2. Update Markdown Cell (Configuration Section)

**OLD TEXT:**
```markdown
**Choose your label column:**
- `'Diagnosis'` - Multi-class classification (melanoma, crlm, gist, lipo, desmoid, liver)
- `'Diagnosis_binary'` - Binary classification (malignant, benign)
```

**NEW TEXT:**
```markdown
**Choose your label column:**
- `'Dataset'` - **RECOMMENDED** - Tumor type classification (CRLM, Desmoid, GIST, Lipo, Liver, Melanoma)
- `'Diagnosis'` - Sub-type classification (rHGP, dHGP, DTF, non-DTF, WDLPS, Lipoma, etc.)
- `'Diagnosis_binary'` - Binary classification (numeric: 0, 1, -1) - ⚠️ Currently incompatible

**Note:** Your CSV uses `Subject` for case IDs, not `case_id`.
```

---

### 3. Update Data Loading Cell

**OLD CODE:**
```python
# Load dataset index
labels_csv = output_dir / 'labels.csv'
index = create_index_from_sheet(
    data_root=DATA_ROOT,
    sheet_path=SHEET_CSV,
    label_column=LABEL_COLUMN,
    output_csv=str(labels_csv)
)
```

**NEW CODE:**
```python
# Load dataset index
labels_csv = output_dir / 'labels.csv'
index = create_index_from_sheet(
    data_root=DATA_ROOT,
    sheet_path=SHEET_CSV,
    label_column=LABEL_COLUMN,
    study_id_col=STUDY_ID_COL,  # Added parameter
    output_csv=str(labels_csv)
)

print(f"\n✓ Loaded {len(index)} studies")
print(f"✓ Saved labels to: {labels_csv}")
```

---

### 4. Update All Terminal Command Examples

**Search for these patterns and update them:**

**OLD:**
```bash
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis \
    --fold 0
```

**NEW:**
```bash
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv sheet.csv \
    --label_column Dataset \
    --study_id_col Subject \
    --fold 0
```

**Locations to update:**
1. First markdown cell (Terminal Alternative section)
2. Step 2 markdown cell (Stage-1 training)
3. Step 3 markdown cell (Stage-2 training)
4. Summary section (multiple command examples)

---

### 5. Update All Bash Script References

**OLD:**
```bash
run_hieracascade.bat 0 Diagnosis  # Windows
bash run_hieracascade.sh 0 Diagnosis  # Linux/Mac
```

**NEW:**
```bash
run_hieracascade.bat 0 Dataset  # Windows
bash run_hieracascade.sh 0 Dataset  # Linux/Mac
```

---

## Quick Find & Replace Guide

If editing the notebook JSON directly, search and replace:

1. **Default label column:**
   - Find: `'Diagnosis'  # Change to 'Diagnosis_binary'`
   - Replace: `'Dataset'  # Use 'Dataset' for tumor types, 'Diagnosis' for sub-types`

2. **Sheet CSV path:**
   - Find: `'../data/sheet.csv'`
   - Replace: `'../sheet.csv'`

3. **Command examples:**
   - Find: `--label_column Diagnosis`
   - Replace: `--label_column Dataset`

4. **Add study_id_col parameter:**
   - After each `--label_column Dataset`, add:
   - `--study_id_col Subject \`

---

## Manual Steps to Update the Notebook

### Option 1: Edit in Jupyter Notebook
1. Open `notebooks/hieracascade_full_pipeline.ipynb` in Jupyter
2. Go through each cell and apply the changes above
3. Save the notebook

### Option 2: Edit JSON Directly
1. Open the `.ipynb` file in a text editor (it's JSON)
2. Search for the strings listed above
3. Replace them carefully
4. Save the file

### Option 3: Use the Python Script Version
1. I can create a `.py` version with all corrections
2. You can convert it to notebook using:
   ```bash
   jupyter nbconvert --to notebook hieracascade_full_pipeline.py
   ```

---

## Testing the Updated Notebook

After making changes, test with this cell:

```python
# Test cell - run this first
from hieracascade.dataio import create_index_from_sheet

try:
    test_index = create_index_from_sheet(
        data_root='../data',
        sheet_path='../sheet.csv',
        label_column='Dataset',
        study_id_col='Subject'
    )
    print(f"✅ SUCCESS: Loaded {len(test_index)} studies")
    categories = set([item['category'] for item in test_index])
    print(f"✅ Categories: {sorted(categories)}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nCheck that:")
    print("  1. sheet.csv is in the correct location")
    print("  2. data/ directory exists with tumor subdirectories")
    print("  3. LABEL_COLUMN and STUDY_ID_COL are set correctly")
```

Expected output:
```
✅ SUCCESS: Loaded <N> studies
✅ Categories: ['crlm', 'desmoid', 'gist', 'lipo', 'liver', 'melanoma']
```

---

## Summary of Column Mapping

| Variable | Old Value | New Value | Reason |
|----------|-----------|-----------|---------|
| `SHEET_CSV` | `'../data/sheet.csv'` | `'../sheet.csv'` | File is at project root |
| `LABEL_COLUMN` | `'Diagnosis'` | `'Dataset'` | Correct tumor type column |
| `STUDY_ID_COL` | (not set) | `'Subject'` | Your CSV uses 'Subject' not 'case_id' |

---

## Need Help?

If you prefer, I can:
1. ✅ Create a corrected Python script version
2. ✅ Create a new notebook from scratch with corrections
3. ✅ Provide a sed/awk script to automate the changes

Just let me know which approach you prefer!
