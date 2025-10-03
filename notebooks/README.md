# HieraCascade Notebooks

This directory contains interactive notebooks and scripts for training and evaluating the HieraCascade model.

## Files

### `hieracascade_full_pipeline.py`
**Complete training pipeline that can be run in Jupyter or as a script.**

This notebook/script demonstrates the entire HieraCascade workflow:
1. Load data from sheet.csv with label selection
2. Create stratified train/validation splits
3. Train Stage-1 model (coarse predictions + saliency)
4. Train Stage-2 model (hierarchical classification)
5. Evaluate and visualize results

## Usage Options

### Option 1: Jupyter Notebook (Interactive)

1. **Convert to notebook:**
   ```bash
   # Install jupytext if needed
   pip install jupytext
   
   # Convert to notebook
   jupytext --to notebook hieracascade_full_pipeline.py
   ```

2. **Or open directly in VS Code/JupyterLab:**
   - Most modern editors support running `.py` files as notebooks
   - Look for `# %%` cell markers

3. **Run cells interactively:**
   - Configure your label column (`Diagnosis` or `Diagnosis_binary`)
   - Uncomment training cells to run training
   - View visualizations inline

**Pros:**
- ✅ Interactive exploration
- ✅ Immediate visualization
- ✅ Easy to experiment

**Cons:**
- ❌ Can be slow for long training
- ❌ Connection issues may interrupt training

---

### Option 2: Terminal (Recommended for Training)

For actual training, use the terminal for better stability and performance.

#### Quick Start (Both Stages)
```bash
# Multi-class classification
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis \
    --output_dir outputs/hieracascade \
    --fold 0 \
    --device cuda

# Binary classification
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --label_column Diagnosis_binary \
    --output_dir outputs/hieracascade_binary \
    --fold 0 \
    --device cuda
```

#### Or Use Convenience Scripts
```bash
# Windows
run_hieracascade.bat 0 Diagnosis
run_hieracascade.bat 0 Diagnosis_binary

# Linux/Mac
bash run_hieracascade.sh 0 Diagnosis
bash run_hieracascade.sh 0 Diagnosis_binary
```

**Pros:**
- ✅ Stable for long training runs
- ✅ Easy to run in background
- ✅ Better GPU utilization
- ✅ Automatic checkpointing

**Cons:**
- ❌ Less interactive
- ❌ Requires terminal access

---

### Option 3: Train Stages Separately

#### Step 1: Prepare Labels
```bash
# From notebook
python -c "
from hieracascade.dataio import create_index_from_sheet
create_index_from_sheet(
    data_root='data',
    sheet_path='data/sheet.csv',
    label_column='Diagnosis',
    output_csv='labels.csv'
)
"

# Or use prepare_data script
python -m hieracascade.prepare_data \
    --data_root data \
    --output_csv labels.csv
```

#### Step 2: Train Stage-1
```bash
python -m hieracascade.train_stage1 \
    --config hieracascade/configs/stage1.yaml \
    --data_root data \
    --labels_csv labels.csv \
    --output_dir outputs/stage1/fold0 \
    --fold 0 \
    --device cuda
```

**Expected time:** 3-5 hours on RTX 3090 (20 epochs, 50-100 studies)

#### Step 3: Train Stage-2
```bash
python -m hieracascade.train_stage2 \
    --config hieracascade/configs/stage2.yaml \
    --data_root data \
    --labels_csv labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0 \
    --fold 0 \
    --device cuda
```

**Expected time:** 5-8 hours on RTX 3090 (40 epochs)

#### Step 4: Evaluate
```bash
python -m hieracascade.evaluate \
    --checkpoint outputs/stage2/fold0/checkpoint_best.pt \
    --stage stage2 \
    --data_root data \
    --labels_csv labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0/eval \
    --fold 0 \
    --device cuda
```

**Expected time:** 15-30 minutes

---

## Workflow Recommendations

### For Exploration and Visualization
**Use Jupyter Notebook** (`hieracascade_full_pipeline.py`)
- Load data and inspect distributions
- Visualize saliency maps
- Explore trained model outputs
- Quick prototyping

### For Training
**Use Terminal** (quick_start or separate stages)
- More stable for multi-hour training
- Better resource management
- Easy to monitor with logs
- Can run in background/screen session

### For Production
**Use Separate Stage Training**
- Better checkpointing control
- Easier to resume if interrupted
- Can tune each stage independently
- Better for ablation studies

---

## Configuration

### Label Selection

Choose the appropriate label column for your task:

```python
# In notebook
LABEL_COLUMN = 'Diagnosis'  # Multi-class: melanoma, crlm, gist, lipo, desmoid, liver
# or
LABEL_COLUMN = 'Diagnosis_binary'  # Binary: malignant, benign
```

```bash
# In terminal
--label_column Diagnosis
# or
--label_column Diagnosis_binary
```

### GPU Memory

If you encounter out-of-memory errors:

1. **Reduce batch size:**
   ```yaml
   # In configs/stage1.yaml or stage2.yaml
   train:
     batch_size: 1  # Reduce from 2
   ```

2. **Reduce volume size:**
   ```yaml
   # In configs/stage1.yaml
   geom:
     size: [160, 160, 160]  # Reduce from [192, 192, 192]
   ```

3. **Reduce number of crops:**
   ```yaml
   # In configs/stage2.yaml
   proposals:
     K: 4  # Reduce from 8
   ```

---

## Expected Outputs

### During Training

**Stage-1:**
- `outputs/stage1/fold0/checkpoint_best.pt` - Best model
- `outputs/stage1/fold0/checkpoint_latest.pt` - Latest epoch
- `outputs/stage1/fold0/visualizations/` - Saliency overlays
- `outputs/stage1/fold0/plots/training_curves.png` - Loss curves
- `outputs/stage1/fold0/cache/` - Preprocessed volumes

**Stage-2:**
- `outputs/stage2/fold0/checkpoint_best.pt` - Best model
- `outputs/stage2/fold0/checkpoint_latest.pt` - Latest epoch
- `outputs/stage2/fold0/plots/training_curves.png` - Loss/metrics

### After Evaluation

- `outputs/stage2/fold0/eval/confusion_matrix_fine.png`
- `outputs/stage2/fold0/eval/confusion_matrix_coarse.png`
- `outputs/stage2/fold0/eval/predictions_fine.csv`
- `outputs/stage2/fold0/eval/predictions_coarse.csv`
- Console output with metrics (F1, Accuracy, AUC, etc.)

---

## Monitoring Training

### In Notebook
- Loss/metrics printed after each epoch
- Visualizations updated in real-time

### In Terminal

**Watch log file:**
```bash
# If you redirect output
python -m hieracascade.quick_start ... > training.log 2>&1
tail -f training.log
```

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check tensorboard (if you add it):**
```bash
tensorboard --logdir outputs/
```

---

## Troubleshooting

### Notebook Kernel Dies
- **Cause:** Out of memory
- **Solution:** Reduce batch size or use terminal

### Training Too Slow
- **Cause:** CPU mode or small GPU
- **Solution:** 
  - Check `device = 'cuda'` is set
  - Reduce batch size for better throughput
  - Use terminal instead of notebook

### Import Errors
- **Cause:** Path issues
- **Solution:**
  ```python
  import sys
  sys.path.append('..')  # Add parent directory
  ```

### Label Column Not Found
- **Cause:** Wrong column name
- **Solution:** Check `sheet.csv` columns and use exact name

---

## Example Session

### Complete Notebook Session
```python
# 1. Load and explore data
index = create_index_from_sheet(
    data_root='data',
    sheet_path='data/sheet.csv',
    label_column='Diagnosis'
)

# 2. Check distribution
import pandas as pd
pd.Series([i['category'] for i in index]).value_counts()

# 3. Create splits
splits = create_site_held_out_splits(index, stratified=True)
train, val = splits[0]

# 4. Train (in terminal for stability)
# See terminal commands above

# 5. Load results and visualize
import matplotlib.pyplot as plt
img = plt.imread('outputs/stage2/fold0/eval/confusion_matrix_fine.png')
plt.imshow(img)
plt.show()
```

---

## Documentation Links

- **Main README:** `../hieracascade/README.md`
- **Tutorial:** `../hieracascade/TUTORIAL.md`
- **Label Selection Guide:** `../hieracascade/LABEL_SELECTION_GUIDE.md`
- **Quick Reference:** `../hieracascade/QUICK_REFERENCE.md`

---

## Tips for Success

1. **Start small:** Use a subset of data for initial testing
2. **Monitor early:** Check first epoch outputs before long training
3. **Save often:** Checkpoints are automatic but monitor disk space
4. **Use stratification:** Enabled by default, maintains class balance
5. **Choose wisely:** Notebook for exploration, terminal for training
6. **Document results:** Save visualizations and metrics for your thesis

Good luck with your soft tissue tumor classification project! 🎓
