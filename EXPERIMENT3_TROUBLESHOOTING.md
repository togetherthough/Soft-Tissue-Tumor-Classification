# Experiment 3 Troubleshooting Guide

## 🔍 Files to Check

### 1. **Cluster Log Files**
```bash
# On the cluster, check these logs:
ls -lth /trinity/home/r112276/Med3Tab-PFN/logs/exp3_*.log
tail -n 100 /trinity/home/r112276/Med3Tab-PFN/logs/exp3_JOBID.log
tail -n 100 /trinity/home/r112276/Med3Tab-PFN/logs/exp3_error_JOBID.log
```

### 2. **SAM-Med3D Checkpoint**
```bash
# Verify checkpoint exists and is not corrupted
ls -lh SAM-Med3D-main/SAM-Med3D-main/ckpt/
file SAM-Med3D-main/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth

# Check file size (should be ~400MB+)
# If missing or wrong size, redownload:
cd SAM-Med3D-main/SAM-Med3D-main/ckpt/
rm sam_med3d_turbo.pth  # if corrupted
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
```

### 3. **Sheet.csv and Dataset Names**
```bash
# Check what dataset names are actually in sheet.csv
cat /data/scratch/r112276/sheet.csv | cut -d',' -f1 | sort | uniq
# Or on local:
cat sheet.csv | cut -d',' -f1 | sort | uniq

# Compare with config file dataset_name values
grep "dataset_name:" configs/datasets_cluster.yaml
```

### 4. **Embedding Files (if previously generated)**
```bash
# Check if any old/corrupted embeddings exist
find SAM-Med3D-main/SAM-Med3D-main/data/ -name "*_embedding.pt" -type f
# If corrupted, delete and regenerate
```

### 5. **Results Directory**
```bash
# Check what's in the results directory
ls -la results/classification_head/
# Look for error logs or partial results
```

---

## 🐛 Known Issues in Your Config

### **CRITICAL: Duplicate Datasets in datasets_cluster.yaml**
Lines 54-71 and 129-146: `gist` is defined twice
Lines 73-91 and 148-165: `lipo` is defined twice

This will cause the second definition to override the first, potentially with incorrect settings!

---

## ✅ How to Run Experiment 3 with Lower Epochs

### **Option 1: Edit SLURM Script (Recommended)**
```bash
nano cluster_scripts/slurm_experiment3.sh

# Change line 94 from:
#   --epochs 20 \
# To:
    --epochs 5 \
```

### **Option 2: Use Short Version Script**
```bash
# Pre-configured for quick testing
sbatch cluster_scripts/slurm_experiment3_short.sh
```

### **Option 3: Interactive Session (Best for Debugging)**
```bash
# Request GPU node
srun --partition=interactive --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash

# Activate environment
cd /trinity/home/r112276/Med3Tab-PFN
conda activate sammed3d

# Test on single dataset with 5 epochs
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist \
    --epochs 5 \
    --batch-size 4 \
    --freeze-encoder
```

### **Option 4: Run Locally (Windows)**
```powershell
# From Med3Tab-PFN directory
python cluster_scripts/run_experiment3_classification_head.py `
    --config configs/datasets.yaml `
    --datasets gist `
    --epochs 5 `
    --batch-size 2 `
    --freeze-encoder
```

---

## 🔧 Quick Fixes to Try

### **Fix 1: Fix Config Duplicates**
Remove duplicate entries from `datasets_cluster.yaml`

### **Fix 2: Verify Dataset Names Match sheet.csv**
Ensure exact case-sensitive match between config and CSV

### **Fix 3: Test with Single Dataset First**
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist \
    --epochs 3 \
    --freeze-encoder
```

### **Fix 4: Check and Redownload Checkpoint**
```bash
cd SAM-Med3D-main/SAM-Med3D-main/ckpt/
ls -lh sam_med3d_turbo.pth  # Should be ~400MB+
# If missing/small, redownload
```

---

## 🎯 Recommended Testing Sequence

1. **Fix config file** (remove duplicates)
2. **Verify checkpoint** exists and is valid
3. **Test single dataset** with 3 epochs interactively
4. **Check results** in `results/classification_head/gist/`
5. **If successful**, run all datasets with more epochs

---

## 📊 Expected Runtime with Different Epochs

| Epochs | Time per Dataset | All 6 Datasets |
|--------|------------------|----------------|
| 3      | ~10-15 min      | ~1 hour        |
| 5      | ~15-25 min      | ~1.5-2 hours   |
| 10     | ~30-45 min      | ~3-4 hours     |
| 20     | ~1-1.5 hours    | ~6-8 hours     |

With `--freeze-encoder` (recommended for initial testing)

---

## 🆘 If Errors Persist

Check these in order:
1. SLURM error log: `logs/exp3_error_JOBID.log`
2. Python environment: `conda list | grep torch`
3. GPU availability: `nvidia-smi`
4. Data paths: `ls -la /data/scratch/r112276/gist/`
5. Sheet.csv format and content
