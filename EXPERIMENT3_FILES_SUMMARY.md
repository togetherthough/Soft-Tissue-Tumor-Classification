# Experiment 3 - Complete File Reference

## 📚 Documentation Files Created

### **1. FIX_EXPERIMENT3_ERRORS.md** 🔧
**Purpose**: Step-by-step guide to fix YOUR specific errors
**Use when**: You have errors and need to troubleshoot

**Covers**:
- Fixing "invalid load key, 'E'." error (corrupted checkpoint)
- Fixing "No rows found after filtering" error (dataset name mismatch)
- Step-by-step commands to verify and fix issues
- Checklist before rerunning

**→ START HERE if you have errors!**

---

### **2. RUN_EXPERIMENT3_OPTIONS.md** 🚀
**Purpose**: All the different ways to run Experiment 3 with custom epochs
**Use when**: You want to run with 3, 5, 10, or 20 epochs

**Covers**:
- 5 different methods to run experiment 3
- How to change epoch count
- Interactive testing for debugging
- Performance tuning options
- Expected results by epoch count

**→ Use this to run with fewer epochs**

---

### **3. EXPERIMENT3_TROUBLESHOOTING.md** 🔍
**Purpose**: General troubleshooting and file checking guide
**Use when**: You need to verify your setup or investigate issues

**Covers**:
- Which files to check (logs, checkpoints, data)
- Known issues in config file
- Expected runtimes for different epoch counts
- Quick fix commands

**→ Use for general debugging**

---

## 🛠️ New Scripts Created

### **4. cluster_scripts/slurm_experiment3_quick.sh** ⚡
**Purpose**: Pre-configured script for fast 5-epoch runs
**Use when**: You want to quickly test all datasets

```bash
# Run all datasets with 5 epochs (~2 hours)
sbatch cluster_scripts/slurm_experiment3_quick.sh

# Run single dataset with 5 epochs (~20 min)
sbatch cluster_scripts/slurm_experiment3_quick.sh gist
```

**Features**:
- Uses short partition
- Only 5 epochs (vs 20 in standard script)
- Checks checkpoint before running
- Provides interpretation of results

---

### **5. cluster_scripts/diagnose_experiment3.sh** 🔎
**Purpose**: Automatic diagnostic checker for common issues
**Use when**: Before running experiment or when debugging

```bash
bash cluster_scripts/diagnose_experiment3.sh
```

**Checks**:
- ✅ Config file validity (no duplicates)
- ✅ sheet.csv existence and dataset names
- ✅ SAM-Med3D checkpoint file
- ✅ Dataset directories
- ✅ Python environment
- ✅ GPU availability
- ✅ Recent error logs

**→ Run this FIRST to identify problems!**

---

## 🔄 Modified Files

### **6. configs/datasets_cluster.yaml** ✓
**Status**: FIXED
**Change**: Removed duplicate `gist` and `lipo` entries

**Before**: Had entries for gist and lipo twice (lines 54-91 and 129-165)
**After**: Each dataset appears only once

---

## 📖 Existing Documentation (Cluster)

### **EXPERIMENT3_GUIDE.md** (already exists)
- General guide for Experiment 3
- Configuration options
- Expected results interpretation
- Output structure

---

## 🎯 Quick Start Guide

### **For First-Time Users:**

```bash
# 1. Run diagnostics
bash cluster_scripts/diagnose_experiment3.sh

# 2. Fix any issues found (see FIX_EXPERIMENT3_ERRORS.md)

# 3. Test with single dataset
python cluster_scripts/run_experiment3_classification_head.py \
    --datasets gist --epochs 3 --config configs/datasets_cluster.yaml

# 4. Run all datasets with 5 epochs
sbatch cluster_scripts/slurm_experiment3_quick.sh
```

### **For Debugging:**

```bash
# 1. Check error log
tail -n 100 logs/exp3_error_*.log

# 2. Run diagnostic
bash cluster_scripts/diagnose_experiment3.sh

# 3. Read specific guide
cat FIX_EXPERIMENT3_ERRORS.md

# 4. Test interactively
srun --partition=interactive --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
conda activate sammed3d
python cluster_scripts/run_experiment3_classification_head.py --datasets gist --epochs 3
```

---

## 🗺️ Decision Tree: Which File to Read?

```
Do you have errors?
├─ YES → Read FIX_EXPERIMENT3_ERRORS.md
│         (Step-by-step fixes for your specific errors)
│
└─ NO → Want to run with different epochs?
    ├─ YES → Read RUN_EXPERIMENT3_OPTIONS.md
    │         (5 methods to run with 3, 5, 10, or 20 epochs)
    │
    └─ NO → Need to verify setup?
        └─ YES → Run cluster_scripts/diagnose_experiment3.sh
                  (Automatic checks for common issues)
```

---

## 📋 Command Cheat Sheet

```bash
# Diagnostics
bash cluster_scripts/diagnose_experiment3.sh

# Quick run (5 epochs, all datasets)
sbatch cluster_scripts/slurm_experiment3_quick.sh

# Quick run (5 epochs, single dataset)
sbatch cluster_scripts/slurm_experiment3_quick.sh gist

# Interactive test (3 epochs)
python cluster_scripts/run_experiment3_classification_head.py --datasets gist --epochs 3

# Check logs
tail -f logs/exp3_*.log

# Check errors
tail -f logs/exp3_error_*.log

# Cancel job
scancel JOBID

# Check status
squeue -u $USER
```

---

## 📊 File Sizes & Locations

All files are in: `/trinity/home/r112276/Med3Tab-PFN/`

```
Med3Tab-PFN/
├── FIX_EXPERIMENT3_ERRORS.md           # Fixing your specific errors
├── RUN_EXPERIMENT3_OPTIONS.md          # How to run with custom epochs  
├── EXPERIMENT3_TROUBLESHOOTING.md      # General troubleshooting
├── EXPERIMENT3_FILES_SUMMARY.md        # This file
├── configs/
│   └── datasets_cluster.yaml           # Fixed (removed duplicates)
└── cluster_scripts/
    ├── diagnose_experiment3.sh         # Auto-diagnostic script
    ├── slurm_experiment3_quick.sh      # Quick 5-epoch script
    ├── slurm_experiment3.sh            # Standard 20-epoch script (existing)
    └── run_experiment3_classification_head.py  # Main Python script (existing)
```

---

## 🎓 Learning Path

1. **First time?** → Read `EXPERIMENT3_GUIDE.md` (existing)
2. **Have errors?** → Read `FIX_EXPERIMENT3_ERRORS.md` (new)
3. **Want fewer epochs?** → Read `RUN_EXPERIMENT3_OPTIONS.md` (new)
4. **Need to verify?** → Run `diagnose_experiment3.sh` (new)
5. **Quick test?** → Use `slurm_experiment3_quick.sh` (new)

---

## 💡 Pro Tips

1. **Always run diagnostics first**: `bash cluster_scripts/diagnose_experiment3.sh`
2. **Test with 3 epochs interactively** before submitting big jobs
3. **Start with frozen encoder** (`--freeze-encoder`) - it's much faster
4. **Use 5 epochs for initial experiments** - good balance of speed/quality
5. **Check checkpoint file size** - should be ~400-500 MB
6. **Verify dataset names** match sheet.csv exactly (case-sensitive!)

---

**Questions?**
- Check error logs: `logs/exp3_error_*.log`
- Run diagnostics: `bash cluster_scripts/diagnose_experiment3.sh`
- Read troubleshooting: `cat FIX_EXPERIMENT3_ERRORS.md`
