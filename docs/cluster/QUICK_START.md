# ⚡ Quick Start: Running Experiment 1 on a Cluster

**Goal**: Train and test Med3-TabPFN, Med3-LoCalPFN, DenseNet121-3D, and ViT-3D on your datasets using SLURM.

## 🎯 Three Steps to Submit

### Step 1: Transfer to Cluster
```bash
# From your local machine
rsync -avz --progress Med3Tab-PFN/ user@cluster.edu:~/Med3Tab-PFN/
```

### Step 2: Edit Configuration
```bash
# SSH to cluster
ssh user@cluster.edu
cd ~/Med3Tab-PFN

# Edit the SLURM script
nano cluster_scripts/slurm_train_and_test.sh

# Required changes:
# 1. Line 12: Update your email
# 2. Lines 73-75: Uncomment and update your conda/virtualenv activation
```

**Example conda activation:**
```bash
# Uncomment these lines in slurm_train_and_test.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sammed3d
```

### Step 3: Submit Job
```bash
# Verify setup (optional)
bash cluster_scripts/check_setup.sh

# Submit the job
sbatch cluster_scripts/slurm_train_and_test.sh

# Monitor
squeue -u $USER
tail -f logs/exp1_*.log
```

## 📁 What Gets Created

The script will:
1. ✅ Extract SAM-Med3D features from your datasets
2. ✅ Train and test **Med3-TabPFN** (PFN classification head)
3. ✅ Train and test **Med3-LoCalPFN** (local context-aware PFN)
4. ✅ Train and test **DenseNet121-3D** (3D baseline)
5. ✅ Train and test **ViT-3D** (3D Vision Transformer baseline)
6. ✅ Generate combined results CSV with all metrics

## 📊 Results Location

```
results/experiment1/
└── combined_benchmarks_summary.csv    ← Your main results file
```

This CSV contains accuracy, F1, and ROC-AUC for all methods on all datasets.

## ⚙️ Customization

### Run Only PFN Methods (Skip Baselines)
Edit line 111 in `slurm_train_and_test.sh`:
```bash
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --skip-baselines
```

### Run Only Specific Datasets
```bash
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --datasets gist
```

### Use More Training Epochs
```bash
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs-3d 20
```

## 📖 More Information

- **Full guide**: See `cluster_scripts/README.md`
- **Detailed checklist**: See `cluster_scripts/SUBMIT_CHECKLIST.md`
- **Original notebook**: See `notebooks/Experiment1-Custom-Methods-vs-Baselines.ipynb`

## 💡 Pro Tips

1. **Test first**: Run with 1 epoch on 1 dataset to verify everything works
2. **Check logs early**: Monitor `logs/exp1_*.log` within first 5 minutes to catch early errors
3. **Expected runtime**: 4-8 hours for 2 datasets with 10 epochs per 3D model
4. **GPU required**: At least 1 GPU with 16GB+ VRAM

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "conda: command not found" | Update conda path in SLURM script |
| "CUDA out of memory" | Request more GPU memory: `#SBATCH --gres=gpu:a100:1` |
| "No checkpoint found" | Download SAM-Med3D checkpoint (see README) |
| Job fails immediately | Check error log: `tail logs/exp1_error_*.log` |

---

**Ready to submit?** Run: `sbatch cluster_scripts/slurm_train_and_test.sh`
