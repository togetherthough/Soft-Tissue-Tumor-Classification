# HieraCascade Final Notebook

## ✅ Ready to Use!

The notebook `hieracascade_final.ipynb` is configured with your exact requirements:

### Configuration
- **Sheet Path**: `data/sheet.csv` ✅
- **Label Column**: `Diagnosis_binary` (binary classification) ✅
- **Study ID Column**: `Subject` ✅
- **Automatic Conversion**: 0→benign, 1→malignant, -1→unknown ✅

---

## Quick Start

### Open the Notebook
```bash
cd notebooks
jupyter notebook hieracascade_final.ipynb
```

### Run the Cells
1. **Setup cell** - Imports and device detection
2. **Configuration cell** - Already set with your parameters
3. **Load data cell** - Loads and converts labels automatically
4. **Visualize distributions** - See class balance
5. **Train Stage-1** - Uncomment to train
6. **Train Stage-2** - Uncomment to train
7. **Evaluate** - View results

---

## What's Different from Original

| Feature | Original | hieracascade_final |
|---------|----------|-------------------|
| **Sheet path** | `../data/sheet.csv` or `../sheet.csv` | `../data/sheet.csv` ✅ |
| **Label column** | `Diagnosis` or `Dataset` | `Diagnosis_binary` ✅ |
| **Study ID** | `case_id` (missing) | `Subject` ✅ |
| **Label conversion** | ❌ Not supported | ✅ Automatic (0/1/-1) |
| **Output dir** | `hieracascade_notebook` | `hieracascade_binary` |
| **Focus** | General | Binary classification |

---

## Test Configuration

Run this in the first code cell to verify setup:

```python
from hieracascade.dataio import create_index_from_sheet

# Test configuration
index = create_index_from_sheet(
    data_root='../data',
    sheet_path='../data/sheet.csv',
    label_column='Diagnosis_binary',
    study_id_col='Subject'
)

print(f"✅ Loaded {len(index)} studies")
categories = sorted(set([item['category'] for item in index]))
print(f"✅ Categories: {categories}")
```

**Expected output:**
```
✅ Loaded <N> studies
✅ Categories: ['benign', 'malignant'] or ['benign', 'malignant', 'unknown']
```

---

## Notebook Structure

### 32 Cells Total

1. **Header** (markdown) - Title and overview
2. **Setup** (markdown) - Import section
3. **Imports** (code) - All necessary imports
4. **Configuration** (markdown) - Explain settings
5. **Config** (code) - Set all parameters
6. **Load Data** (markdown) - Data loading section
7. **Load** (code) - Load and convert labels
8. **Visualize** (markdown) - Distribution plots
9. **Plots** (code) - Create visualizations
10. **Splits** (markdown) - CV split section
11. **Create splits** (code) - Stratified splitting
12. **Stage 1** (markdown) - Stage-1 overview
13. **Stage 1 config** (code) - Load config
14. **Train Stage 1** (markdown) - Training instructions
15. **Train** (code) - Training call (commented)
16. **Load checkpoint** (markdown) - Checkpoint loading
17. **Load** (code) - Load existing checkpoint
18. **Saliency** (markdown) - Visualization section
19. **Show saliency** (code) - Display saliency maps
20. **Stage 2** (markdown) - Stage-2 overview
21. **Stage 2 config** (code) - Load config
22. **Train Stage 2** (markdown) - Training instructions
23. **Train** (code) - Training call (commented)
24. **Evaluate** (markdown) - Evaluation section
25. **Eval commands** (code) - Print eval command
26. **Training curves** (markdown) - Curves section
27. **Show curves** (code) - Display curves
28. **Results** (markdown) - Results section
29. **Show results** (code) - Confusion matrix & metrics
30. **Summary** (markdown) - Complete summary
31. **Final** (code) - Print final status

---

## Training Options

### Option 1: Train in Notebook
Uncomment the training cells:
```python
# Change this:
# train_stage1(...)

# To this:
train_stage1(...)
```

### Option 2: Train in Terminal (Recommended)
```bash
python -m hieracascade.quick_start \
    --data_root data \
    --fold 0
```

### Option 3: Train Stages Separately
```bash
# Stage 1
python -m hieracascade.train_stage1 \
    --config hieracascade/configs/stage1.yaml \
    --data_root data \
    --labels_csv outputs/hieracascade_binary/labels.csv \
    --fold 0

# Stage 2
python -m hieracascade.train_stage2 \
    --config hieracascade/configs/stage2.yaml \
    --data_root data \
    --labels_csv outputs/hieracascade_binary/labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --fold 0
```

---

## Expected Outputs

### After Data Loading
- Label CSV: `outputs/hieracascade_binary/labels.csv`
- Console output showing benign/malignant counts

### After Stage-1 Training
- Checkpoint: `outputs/hieracascade_binary/stage1/fold0/checkpoint_best.pt`
- Saliency maps: `outputs/hieracascade_binary/stage1/fold0/visualizations/`
- Training curves: `outputs/hieracascade_binary/stage1/fold0/plots/`

### After Stage-2 Training
- Checkpoint: `outputs/hieracascade_binary/stage2/fold0/checkpoint_best.pt`
- Training curves: `outputs/hieracascade_binary/stage2/fold0/plots/`

### After Evaluation
- Predictions: `outputs/hieracascade_binary/stage2/fold0/eval/predictions_fine.csv`
- Confusion matrix: `outputs/hieracascade_binary/stage2/fold0/eval/confusion_matrix_fine.png`
- Metrics: Accuracy, F1-score, AUC

---

## Troubleshooting

### Issue: "No studies loaded"
**Check:**
1. `data/sheet.csv` exists
2. Data directories exist: `data/crlm/`, `data/desmoid/`, etc.
3. NIfTI files exist at expected paths

**Solution:**
```python
import pandas as pd
df = pd.read_csv('../data/sheet.csv')
print(df.columns)
print(df.head())
```

### Issue: "Column not found"
**Error:** `ValueError: Column 'Diagnosis_binary' not found`

**Solution:** Verify column name:
```python
import pandas as pd
df = pd.read_csv('../data/sheet.csv')
print("Columns:", df.columns.tolist())
```

### Issue: GPU Out of Memory
**Error:** CUDA out of memory

**Solution:** Reduce batch size in config cells:
```python
config_stage1['train']['batch_size'] = 1
config_stage2['train']['batch_size'] = 1
```

---

## Files in This Directory

- **`hieracascade_final.ipynb`** ✅ - Main notebook (ready to use)
- **`hieracascade_final.py`** - Python source (for reference)
- **`convert_hieracascade_final.py`** - Conversion script
- **`README_HIERACASCADE_FINAL.md`** (this file) - Usage guide
- `hieracascade_full_pipeline.ipynb` - Original version (outdated)
- `hieracascade_full_pipeline_CORRECTED.py` - Previous version

---

## Key Features

✅ **Binary classification** - Malignant vs benign  
✅ **Automatic label conversion** - No manual preprocessing  
✅ **Correct paths** - Uses `data/sheet.csv`  
✅ **Correct columns** - Uses `Subject` for IDs  
✅ **Stratified splits** - Maintains class balance  
✅ **Rich visualizations** - Distributions, saliency, curves  
✅ **Production ready** - Tested configuration  

---

## Documentation

See also:
- **`../FINAL_CONFIGURATION.md`** - Complete setup guide
- **`../CHANGES_SUMMARY.md`** - What changed
- **`../hieracascade/LABEL_SELECTION_GUIDE.md`** - Label options
- **`../hieracascade/COMPATIBILITY_FIX_SUMMARY.md`** - Original issues

---

## Summary

🎯 **Purpose**: Binary tumor classification (malignant vs benign)  
📊 **Data**: `data/sheet.csv` with `Diagnosis_binary` column  
🔧 **Configuration**: All set correctly by default  
🚀 **Ready**: Open and run!

**Open the notebook:**
```bash
jupyter notebook hieracascade_final.ipynb
```

**Or run from terminal:**
```bash
python -m hieracascade.quick_start --data_root data --fold 0
```
