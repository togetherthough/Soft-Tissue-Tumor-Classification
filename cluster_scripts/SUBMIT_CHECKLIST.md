# 🚀 Quick Submit Checklist for Experiment 1

Use this checklist before submitting your SLURM job to train and test your methods.

## ✅ Pre-Submission Checklist

### 1. Environment Setup
- [ ] Code is transferred to the cluster
- [ ] Data is accessible on the cluster
- [ ] Python environment (conda/virtualenv) is set up
- [ ] Dependencies are installed (see `med3pipe/requirements.txt`)

### 2. SAM-Med3D Checkpoint
- [ ] Checkpoint directory exists: `SAM-Med3D-main/SAM-Med3D-main/ckpt/`
- [ ] Checkpoint file downloaded: `sam_med3d_turbo.pth` (~750MB)
  - Download from: https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth

### 3. Configuration Files
- [ ] `configs/datasets.yaml` exists and is properly configured
- [ ] Dataset paths in config match your cluster setup
- [ ] `sheet.csv` files exist for each dataset

### 4. SLURM Script Configuration
Edit `cluster_scripts/slurm_train_and_test.sh`:
- [ ] Update email address: `#SBATCH --mail-user=YOUR_EMAIL@example.com`
- [ ] Update partition if needed (default: `long`)
- [ ] Update time limit if needed (default: `2-00:00:00` = 2 days)
- [ ] Update GPU/CPU resources if needed
- [ ] Uncomment and update environment activation:
  ```bash
  # Example for conda:
  source /path/to/conda/etc/profile.d/conda.sh
  conda activate sammed3d
  ```

### 5. Verify Setup (Recommended)
Run the setup verification script:
```bash
cd /path/to/Med3Tab-PFN
bash cluster_scripts/check_setup.sh
```

## 📋 Submit Commands

### Standard Submission (All Methods, All Datasets)
```bash
# Navigate to your repository
cd /path/to/Med3Tab-PFN

# Submit the job
sbatch cluster_scripts/slurm_train_and_test.sh
```

### Monitor Your Job
```bash
# Check job status
squeue -u $USER

# View live output log
tail -f logs/exp1_<JOBID>.log

# View error log
tail -f logs/exp1_error_<JOBID>.log

# Cancel job if needed
scancel <JOBID>
```

## 🎯 Customization Options

### Run Only Specific Datasets
Edit the `srun` command in `slurm_train_and_test.sh`:
```bash
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --datasets gist lipo \
    --epochs-3d 10
```

### Skip Certain Methods
```bash
# Skip 3D baselines (only run PFN methods)
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --skip-baselines

# Skip TabPFN
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --skip-tabpfn

# Skip LoCalPFN
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --skip-localpfn
```

### Adjust Training Epochs for 3D Models
```bash
# Train 3D models for 20 epochs instead of default 10
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --epochs-3d 20
```

## 📊 Expected Results

After successful completion, you'll find:

```
results/experiment1/
├── combined_benchmarks_summary.csv    # ← Main results table
├── tabpfn_<dataset>/
│   ├── metrics.json
│   └── predictions.csv
├── localpfn_<dataset>/
│   ├── metrics.json
│   └── predictions.csv
└── [model checkpoints and outputs]
```

### Results Columns
The `combined_benchmarks_summary.csv` contains:
- **dataset**: Dataset name (gist, lipo, etc.)
- **method**: Method name (tabpfn, localpfn, densenet121_3d, vit3d)
- **accuracy**: Classification accuracy
- **macro_f1**: Macro-averaged F1 score
- **roc_auc**: Area under ROC curve

## ⏱️ Expected Runtime

| Configuration | Approximate Time |
|--------------|------------------|
| Both methods, 2 datasets, 10 epochs | 4-8 hours |
| Both methods, 2 datasets, 20 epochs | 8-12 hours |
| PFN methods only, 2 datasets | 1-2 hours |
| Single dataset, quick test (1 epoch) | 30-60 minutes |

*Times vary based on dataset size and GPU speed*

## 🔧 Common Issues

### "CUDA out of memory"
**Solution**: Request more GPU memory or use a GPU with more VRAM
```bash
#SBATCH --gres=gpu:a100:1  # Request A100 if available
#SBATCH --mem=128G          # Request more RAM
```

### "Command 'conda' not found"
**Solution**: Make sure you source conda properly:
```bash
# Find your conda installation
which conda

# Add to slurm script:
source /path/to/conda/etc/profile.d/conda.sh
conda activate sammed3d
```

### "No such file or directory: sheet.csv"
**Solution**: Check dataset paths in `configs/datasets.yaml` match your cluster structure

### Job terminates silently
**Solution**: Check error log
```bash
tail -n 100 logs/exp1_error_<JOBID>.log
```

## 📝 Quick Example Workflow

```bash
# 1. SSH to cluster
ssh user@cluster.edu

# 2. Navigate to repo
cd ~/Med3Tab-PFN

# 3. Verify setup (optional but recommended)
bash cluster_scripts/check_setup.sh

# 4. Edit SLURM script with your email
nano cluster_scripts/slurm_train_and_test.sh
# Update: #SBATCH --mail-user=YOUR_EMAIL@example.com
# Update: conda activate sammed3d (or your env)

# 5. Submit job
sbatch cluster_scripts/slurm_train_and_test.sh

# 6. Monitor
squeue -u $USER
tail -f logs/exp1_*.log

# 7. When complete, download results
# (On your local machine)
scp -r user@cluster.edu:~/Med3Tab-PFN/results/experiment1/ ./
```

## ✨ Tips

1. **Test first**: Run with `--epochs-3d 1` and one dataset to verify everything works
2. **Monitor GPU**: Check GPU usage with `nvidia-smi` on the compute node
3. **Save intermediate results**: The script saves results per method, so partial completion is useful
4. **Use screen/tmux**: For long monitoring sessions, use `screen` or `tmux`
5. **Check disk space**: Ensure adequate space for preprocessed data and results

---

**Need help?** See the full guide in `cluster_scripts/README.md`
