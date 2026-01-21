# Cluster Quick Start

> Get Med3Tab-PFN experiments running on SLURM in minutes

## Prerequisites

- Access to HPC cluster with SLURM
- GPU allocation (16GB+ VRAM recommended)
- Python environment with dependencies installed

---

## Step 1: Transfer Code to Cluster

```bash
# From local machine
rsync -avz --progress Med3Tab-PFN/ user@cluster.edu:~/Med3Tab-PFN/
```

---

## Step 2: Set Up Environment

```bash
ssh user@cluster.edu
cd ~/Med3Tab-PFN

# Create conda environment
conda create -n med3tabpfn python=3.9 -y
conda activate med3tabpfn

# Install dependencies
pip install -r med3pipe/requirements.txt
```

---

## Step 3: Configure SLURM Script

Edit the SLURM script:
```bash
nano cluster_scripts/slurm/slurm_exp1.sh
```

Update these lines:
```bash
#SBATCH --mail-user=your@email.edu    # Line 12: Your email

# Lines 50-52: Conda activation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate med3tabpfn
```

---

## Step 4: Verify Setup

```bash
# Check environment
bash cluster_scripts/utils/check_setup.sh

# Verify SAM checkpoint
bash cluster_scripts/utils/quick_weight_check.sh
```

---

## Step 5: Submit Job

```bash
# Submit experiment 1 (method comparison)
sbatch cluster_scripts/slurm/slurm_exp1.sh

# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/exp1_*.log
```

---

## Available Experiments

| Experiment | Command | Description |
|------------|---------|-------------|
| **Exp 1: Benchmarks** | `sbatch slurm/slurm_exp1.sh` | Compare all methods |
| **Exp 1: Filtered** | `sbatch slurm/slurm_exp1_filtered.sh` | With lesion filtering |
| **Exp 1: ROI** | `sbatch slurm/slurm_exp1_roi.sh` | With ROI cropping |
| **Exp 3: Clf Head** | `sbatch slurm/slurm_exp3.sh` | Classification head |
| **Preprocessing** | `sbatch slurm/slurm_compare_classifier.sh` | Three-way comparison |

---

## Results Location

```
results/
├── experiment1/
│   └── combined_benchmarks_summary.csv   # Main results
├── classification_head/
│   ├── gist/
│   │   ├── best_model.pth
│   │   └── metrics.txt
│   └── lipo/
└── three_way_comparison/
    └── preprocessing_comparison.csv
```

---

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "conda: command not found" | Update conda path in SLURM script |
| "CUDA out of memory" | Request larger GPU: `--gres=gpu:a100:1` |
| "No checkpoint found" | Verify `sam-med3d/ckpt/sam_med3d_turbo.pth` |
| Job fails immediately | Check: `tail logs/exp1_error_*.log` |

---

## Expected Runtime

| Experiment | Datasets | Estimated Time |
|------------|----------|----------------|
| Exp 1 (all methods) | 2 | 4-8 hours |
| Exp 3 (frozen) | 2 | 1-2 hours |
| Exp 3 (fine-tune) | 2 | 6-12 hours |
| Preprocessing comparison | 2 | 12-18 hours |

---

## Pro Tips

1. **Test first**: Run with 1 epoch on 1 dataset to verify
2. **Check early**: Monitor logs within first 5 minutes
3. **Save resources**: Use `--freeze-encoder` for quick tests
4. **Batch submissions**: Submit multiple jobs with different configs

---

## Next Steps

- [Full Cluster Guide](README.md) — Detailed cluster setup
- [Scripts Reference](SCRIPTS_REFERENCE.md) — All available scripts
- [Submit Checklist](SUBMIT_CHECKLIST.md) — Pre-submission verification
- [Experiment 1 Details](experiment1/README.md) — Benchmark experiment

---

**Ready?** Run: `sbatch cluster_scripts/slurm/slurm_exp1.sh`
