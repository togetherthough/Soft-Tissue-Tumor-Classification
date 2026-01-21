# Cluster Scripts

This directory contains SLURM scripts and Python runners for executing experiments on HPC clusters.

## 📋 Available Experiments

### Experiment 1: Head-to-Head Benchmarks
Trains and tests all methods on your datasets:

1. **Med3-TabPFN** - PFN-based classification head on SAM-Med3D features
2. **Med3-LoCalPFN** - Local context-aware PFN variant
3. **DenseNet121-3D** - 3D convolutional baseline
4. **ViT-3D** - 3D Vision Transformer baseline

**Scripts**: 
- `slurm_train_and_test.sh` → Original (no filtering)
- `slurm_train_and_test_filtered.sh` → With recommended filtering (NEW)
- `run_experiment1_benchmarks.py` → Python runner with filtering support

### Experiment 3: Classification Head on SAM-Med3D
Tests if SAM-Med3D features are discriminative by training a simple classification head:

- Trains linear classifier on frozen SAM-Med3D features
- Evaluates feature quality before running full PFN pipeline
- Option to fine-tune encoder
- Supports lesion size filtering

**Scripts**: 
- `slurm_experiment3.sh` → Original (no filtering)
- `slurm_experiment3_filtered.sh` → With recommended filtering (NEW)
- `run_experiment3_classification_head.py` → Python runner with filtering support

## 🚀 Quick Start

### 1. Prepare Your Environment

Before running on the cluster, ensure:

- [ ] Your data is accessible on the cluster at `/data/scratch/r112276/`
- [ ] Python environment with dependencies is set up
- [ ] SAM-Med3D checkpoint is downloaded
- [ ] Cluster config file (`configs/datasets_cluster.yaml`) has correct paths
  - **Note**: The cluster uses `datasets_cluster.yaml` with absolute paths to `/data/scratch/`
  - The local config `datasets.yaml` uses relative paths for local development
  - See `configs/README.md` for details on the two-config system

### 2. Transfer Files to Cluster

```bash
# Example using rsync
rsync -avz --progress \
    /path/to/Med3Tab-PFN/ \
    username@cluster.edu:/path/to/Med3Tab-PFN/

# Make sure to include:
# - Code (med3pipe/)
# - Data (or have it accessible)
# - SAM-Med3D checkpoint (sam-med3d/ckpt/)
# - Configs (configs/datasets.yaml)
```

### 3. Edit SLURM Script

Edit `slurm_train_and_test.sh` and update:

```bash
# Set your email for notifications
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Activate your conda environment (uncomment and adjust)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate sammed3d
```

### 4. Submit Job

```bash
# SSH to your cluster
ssh username@cluster.edu

# Navigate to the repository
cd /path/to/Med3Tab-PFN

# Submit the job
sbatch cluster_scripts/slurm_train_and_test.sh
```

### 5. Monitor Job

```bash
# Check job status
squeue -u $USER

# View live log (replace JOBID with your job ID)
tail -f logs/exp1_JOBID.log

# View error log
tail -f logs/exp1_error_JOBID.log
```

## 🔬 Lesion Size Filtering (NEW)

Train models on high-quality lesion subsets by filtering based on size metrics. Filtered and unfiltered results are saved in **separate folders** for easy comparison.

### Quick Start with Filtering

```bash
# Run Experiment 3 with recommended filtering
sbatch cluster_scripts/slurm_experiment3_filtered.sh

# Results saved to separate folders:
#   results/classification_head/gist_filtered_v500_d5_ρ0.30/
#   results/classification_head/lipo_filtered_v500_d5_ρ0.30/
```

### Filter Presets

```bash
# Recommended (good balance) - ~34% of cases
--filter-preset recommended  # voxels>=500, dimension>=5, density>=0.3

# Conservative (strict quality) - ~20% of cases  
--filter-preset conservative # voxels>=1000, dimension>=10, density>=0.3

# Lenient (more inclusive) - ~60% of cases
--filter-preset lenient      # voxels>=200, dimension>=3
```

### Custom Filtering

```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets.yaml \
    --min-voxels 500 \
    --min-dimension 5 \
    --min-density 0.3
```

### Documentation

- **Filtering Guide**: [../docs/FILTERING.md](../docs/FILTERING.md)
- **Cluster Quick Start**: [../docs/cluster/QUICK_START.md](../docs/cluster/QUICK_START.md)
- **Cluster Overview**: [../docs/cluster/README.md](../docs/cluster/README.md)
- **Experiment 1**: [../docs/cluster/experiment1/README.md](../docs/cluster/experiment1/README.md)
- **Experiment 3**: [../docs/cluster/experiment3/README.md](../docs/cluster/experiment3/README.md)
- **Submit Checklist**: [../docs/cluster/SUBMIT_CHECKLIST.md](../docs/cluster/SUBMIT_CHECKLIST.md)

## 📁 Files in This Directory

### Experiment Scripts

| File | Description |
|------|-------------|
| `slurm_train_and_test.sh` | **Experiment 1** - Benchmarks (no filtering) |
| `slurm_train_and_test_filtered.sh` | **Experiment 1** - Benchmarks with filtering (NEW) |
| `run_experiment1_benchmarks.py` | Python runner with filtering support |
| `slurm_experiment3.sh` | **Experiment 3** - Classification head (no filtering) |
| `slurm_experiment3_filtered.sh` | **Experiment 3** - With lesion filtering (NEW) |
| `run_experiment3_classification_head.py` | Python runner with filtering support |

### Helper Scripts

| File | Description |
|------|-------------|
| `check_setup.sh` | Verify environment setup |
| `verify_config.sh` | Check config file validity |
| `diagnose_experiment3.sh` | Troubleshoot Experiment 3 issues |

## ⚙️ Configuration Options

The `run_experiment1_benchmarks.py` script accepts several options:

```bash
# Run only specific datasets
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --datasets gist lipo \
    --config configs/datasets.yaml \
    --output-dir results/experiment1

# Skip certain methods
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --skip-tabpfn \
    --skip-localpfn \
    --skip-baselines

# Adjust 3D model training epochs
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --epochs-3d 20
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to datasets config YAML | `configs/datasets.yaml` |
| `--output-dir` | Output directory for results | `results/experiment1` |
| `--datasets` | Specific datasets to run (space-separated) | All in config |
| `--skip-tabpfn` | Skip TabPFN method | False |
| `--skip-localpfn` | Skip LoCalPFN method | False |
| `--skip-baselines` | Skip 3D baselines | False |
| `--epochs-3d` | Number of epochs for 3D models | 4 |

## 📊 Output Files

After successful completion, results will be in `results/experiment1/`:

```
results/experiment1/
├── combined_benchmarks_summary.csv    # Main results table
├── tabpfn_gist/                       # TabPFN outputs per dataset
│   ├── metrics.json
│   └── predictions.csv
├── localpfn_gist/                     # LoCalPFN outputs per dataset
│   ├── metrics.json
│   └── predictions.csv
├── densenet121_3d_gist_*.pth         # Trained model checkpoints
└── vit3d_gist_*.pth
```

The main results file `combined_benchmarks_summary.csv` contains:

- `dataset` - Dataset name
- `method` - Method name (tabpfn, localpfn, densenet121_3d, vit3d)
- `accuracy` - Classification accuracy
- `macro_f1` - Macro F1 score
- `roc_auc` - ROC AUC score

## 🔧 Troubleshooting

### "CUDA out of memory"

Reduce batch size or image size in the configs. Edit SLURM script to request more GPU memory:

```bash
#SBATCH --gres=gpu:a100:1  # Request specific GPU type
#SBATCH --mem=128G          # Request more RAM
```

### "Module not found: med3pipe"

The script should auto-detect the repo root. If it fails, manually add:

```python
import sys
sys.path.insert(0, '/absolute/path/to/Med3Tab-PFN')
```

### "No checkpoint found"

Download SAM-Med3D checkpoint:

```bash
# On the cluster
cd Med3Tab-PFN/sam-med3d/ckpt/
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
```

### Job fails silently

Check SLURM error log:

```bash
tail -n 100 logs/exp1_error_JOBID.log
```

### Environment not activated

Make sure you uncomment and adjust the conda/virtualenv activation in the SLURM script:

```bash
# In slurm_train_and_test.sh, uncomment and adjust:
source /path/to/conda/etc/profile.d/conda.sh
conda activate sammed3d
```

## 🎯 Example Workflows

### Experiment 1: Run All Methods on All Datasets

```bash
sbatch cluster_scripts/slurm_train_and_test.sh
```

### Experiment 3: Test Feature Quality

Run classification head experiment to test if SAM-Med3D features are good:

```bash
# Submit job
sbatch cluster_scripts/slurm_experiment3.sh

# Or run directly with custom options
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --epochs 20 \
    --batch-size 8 \
    --freeze-encoder

# Fine-tune encoder instead of freezing
python cluster_scripts/run_experiment3_classification_head.py \
    --fine-tune \
    --epochs 50 \
    --lr 1e-4
```

**Available options**:
- `--freeze-encoder`: Freeze SAM-Med3D encoder (default: True)
- `--fine-tune`: Fine-tune encoder (overrides freeze)
- `--epochs N`: Number of training epochs (default: 10)
- `--batch-size N`: Batch size (default: 4)
- `--lr FLOAT`: Learning rate (default: 1e-3)
- `--dropout FLOAT`: Dropout rate (default: 0.3)
- `--datasets gist lipo`: Run on specific datasets only

### Run Only PFN Methods

Edit the srun command in `slurm_train_and_test.sh`:

```bash
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --skip-baselines
```

### Run Only on GIST Dataset

```bash
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --datasets gist \
    --epochs-3d 20
```

### Test Run (Quick)

For testing, reduce epochs and run on one dataset:

```bash
srun python cluster_scripts/run_experiment1_benchmarks.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --datasets gist \
    --epochs-3d 1
```

## 📝 Notes

- **Runtime**: Full experiment with all methods on 2 datasets takes ~4-12 hours depending on:
  - Dataset sizes
  - Number of epochs
  - GPU speed
  
- **GPU Requirements**: At least 1 GPU with 16GB+ VRAM recommended

- **Storage**: Make sure you have enough space for:
  - Preprocessed data (~1-5GB per dataset)
  - Feature extractions (~100MB-1GB per dataset)
  - Model checkpoints (~200MB per 3D model)

## 🔗 Related Documentation

- [Main README](../README.md) - Project overview
- [Experiment1 Notebook](../notebooks/Experiment1-Custom-Methods-vs-Baselines.ipynb) - Interactive version
- [Dataset Config Guide](../docs/QUICK_REFERENCE.md) - How to configure datasets

## 💡 Tips

1. **Start Small**: Test on one dataset with few epochs before running the full experiment
2. **Monitor GPU**: Use `nvidia-smi` in a separate terminal to monitor GPU usage
3. **Save Logs**: Keep logs for debugging and comparing runs
4. **Use Screen/Tmux**: For long-running jobs, use `screen` or `tmux` when monitoring
5. **Check Quotas**: Ensure you have sufficient disk quota and compute allocation

---

For questions or issues, please open an issue on the repository.
