# How to Run Experiment 3 with Different Epoch Settings

## 🎯 Quick Reference

| Method | Epochs | Time | Best For |
|--------|--------|------|----------|
| **Quick Test** | 3-5 | ~1-2h | Debugging, testing setup |
| **Standard** | 10-15 | ~3-5h | Good balance |
| **Full** | 20-30 | ~6-10h | Best performance |

---

## 🚀 Method 1: Pre-made Quick Script (Easiest)

### **5 epochs - All datasets**
```bash
sbatch cluster_scripts/slurm_experiment3_quick.sh
```

### **5 epochs - Single dataset**
```bash
sbatch cluster_scripts/slurm_experiment3_quick.sh gist
```

---

## ⚙️ Method 2: Edit Existing Script

### Edit `slurm_experiment3.sh`:
```bash
nano cluster_scripts/slurm_experiment3.sh
```

### Change line 94:
```bash
# Original (20 epochs)
    --epochs 20 \

# For 3 epochs
    --epochs 3 \

# For 5 epochs  
    --epochs 5 \

# For 10 epochs
    --epochs 10 \
```

### Then submit:
```bash
sbatch cluster_scripts/slurm_experiment3.sh
```

---

## 🔬 Method 3: Interactive Testing (Best for Debugging)

### Step 1: Request GPU node
```bash
srun --partition=interactive --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
```

### Step 2: Activate environment
```bash
cd /trinity/home/r112276/Med3Tab-PFN
conda activate sammed3d
```

### Step 3: Run with custom epochs
```bash
# Test with 3 epochs on single dataset
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist \
    --epochs 3 \
    --batch-size 4 \
    --freeze-encoder

# Test with 5 epochs on multiple datasets
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist lipo crlm \
    --epochs 5 \
    --freeze-encoder
```

---

## 💻 Method 4: Direct Python Command (No SLURM)

### On compute node or local machine:
```bash
# 3 epochs - single dataset
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist \
    --epochs 3

# 5 epochs - all datasets
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --epochs 5

# 10 epochs with custom learning rate
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist lipo \
    --epochs 10 \
    --lr 5e-4 \
    --batch-size 8
```

---

## 🛠️ Method 5: Custom SLURM Script

Create `slurm_experiment3_custom.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=exp3_custom
#SBATCH --output=logs/exp3_custom_%j.log
#SBATCH --error=logs/exp3_custom_error_%j.log
#SBATCH --partition=short
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Activate environment
# conda activate sammed3d

# Custom configuration
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist lipo crlm \
    --epochs 7 \
    --batch-size 6 \
    --lr 2e-3 \
    --freeze-encoder
```

Then run:
```bash
sbatch slurm_experiment3_custom.sh
```

---

## 📊 Recommended Strategy

### **Phase 1: Quick Validation (3 epochs)**
Test if setup works and checkpoint loads:
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --datasets gist \
    --epochs 3 \
    --config configs/datasets_cluster.yaml
```
**Time**: ~10 minutes per dataset

### **Phase 2: All Datasets Fast (5 epochs)**
Check all datasets for major issues:
```bash
sbatch cluster_scripts/slurm_experiment3_quick.sh
```
**Time**: ~1.5-2 hours total

### **Phase 3: Full Run (15-20 epochs)**
Once everything works, do full training:
```bash
# Edit slurm_experiment3.sh to use --epochs 15
sbatch cluster_scripts/slurm_experiment3.sh
```
**Time**: ~5-7 hours total

---

## 🐛 Before Running - Diagnose Issues

### Run diagnostic script:
```bash
bash cluster_scripts/diagnose_experiment3.sh
```

This checks:
- ✅ Config file validity (no duplicates)
- ✅ sheet.csv existence and dataset names
- ✅ SAM-Med3D checkpoint
- ✅ Dataset directories
- ✅ Python environment
- ✅ GPU availability
- ✅ Recent error logs

---

## 🎓 Understanding the Arguments

### Core Arguments:
```bash
--config PATH                  # Config file (datasets.yaml or datasets_cluster.yaml)
--datasets [NAMES...]          # Which datasets to run (default: all)
--epochs N                     # Number of training epochs (default: 10)
--freeze-encoder               # Freeze SAM encoder (FAST, recommended first)
--fine-tune                    # Train full model (SLOW, only if frozen fails)
```

### Performance Tuning:
```bash
--batch-size N                 # Larger = faster but more memory (default: 4)
--lr FLOAT                     # Learning rate (default: 1e-3)
--weight-decay FLOAT           # Regularization (default: 1e-4)
--dropout FLOAT                # Dropout rate (default: 0.3)
--img-size N                   # Image size (default: 128)
--num-workers N                # Data loading threads (default: 2)
```

---

## 📈 Expected Results by Epoch Count

### **3 Epochs (Quick Test)**
- Enough to verify setup works
- Results may be suboptimal
- Use for debugging only

### **5 Epochs (Fast Evaluation)**
- Usually gives 80-90% of final performance
- Good for comparing datasets
- Recommended for initial experiments

### **10 Epochs (Standard)**
- Good performance for most datasets
- Default recommendation
- Usually sufficient with frozen encoder

### **20+ Epochs (Full Training)**
- Marginal improvements over 10 epochs
- Use only if you need absolute best results
- Watch for overfitting on small datasets

---

## 🔍 Monitoring Progress

### Check SLURM queue:
```bash
squeue -u $USER
```

### Watch live log:
```bash
tail -f logs/exp3_quick_*.log
```

### Check errors:
```bash
tail -f logs/exp3_quick_error_*.log
```

### Cancel job if needed:
```bash
scancel JOBID
```

---

## 📁 Where Results Are Saved

```
results/classification_head/
├── summary.csv              # All datasets summary
├── gist/
│   ├── best_model.pt       # Best model weights
│   ├── training_history.json
│   ├── final_metrics.json
│   └── predictions.npy
└── [other datasets...]
```

---

## ❓ FAQ

### Q: How many epochs should I use?
**A**: Start with 5 for testing, use 10-15 for real experiments.

### Q: Should I use --freeze-encoder or --fine-tune?
**A**: Always start with `--freeze-encoder` (much faster). Only use `--fine-tune` if frozen gives AUC < 0.65.

### Q: Can I run multiple datasets in parallel?
**A**: No, the script runs them sequentially. Submit separate jobs for parallel execution.

### Q: What if I run out of memory?
**A**: Reduce `--batch-size` (try 2 instead of 4) or `--img-size` (try 96 instead of 128).

---

## 🆘 Troubleshooting

### Error: "invalid load key, 'E'."
→ SAM-Med3D checkpoint is corrupted. Redownload it:
```bash
cd SAM-Med3D-main/SAM-Med3D-main/ckpt/
rm sam_med3d_turbo.pth
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
```

### Error: "No rows found after filtering by dataset"
→ `dataset_name` in config doesn't match sheet.csv. Check with:
```bash
cat /data/scratch/r112276/sheet.csv | cut -d',' -f1 | sort | uniq
```

### Error: "CUDA out of memory"
→ Reduce batch size:
```bash
--batch-size 2
```

---

**Need more help?** Check `EXPERIMENT3_TROUBLESHOOTING.md`
