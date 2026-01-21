# Cluster Scripts

> HPC experiment scripts for running Med3Tab-PFN experiments on SLURM clusters

## Overview

This directory contains Python experiment scripts and SLURM job submission scripts for running large-scale benchmarking experiments on high-performance computing clusters.

## Directory Structure

```
cluster_scripts/
├── experiments/         # Python experiment scripts
│   ├── exp1_benchmarks.py                    # Method comparison
│   ├── exp3_classifier.py                    # Classification head
│   ├── compare_tabpfn.py                     # TabPFN three-way comparison
│   ├── compare_localpfn.py                   # LoCalPFN three-way comparison
│   ├── compare_classifier_preprocessing.py  # Preprocessing comparison
│   ├── ablate_roi.py                         # ROI margin ablation
│   └── run_all_comparisons.sh                # Run all comparisons
├── slurm/               # SLURM job scripts
│   ├── slurm_exp1.sh                         # Experiment 1 (all methods)
│   ├── slurm_exp1_filtered.sh                # Exp 1 with filtering
│   ├── slurm_exp1_roi.sh                     # Exp 1 with ROI cropping
│   ├── slurm_exp3.sh                         # Experiment 3 (clf head)
│   ├── slurm_exp3_filtered.sh                # Exp 3 with filtering
│   └── slurm_compare_classifier.sh           # Preprocessing comparison
└── utils/               # Utility scripts
    ├── check_setup.sh                        # Verify environment
    ├── verify_config.sh                      # Validate YAML config
    └── quick_weight_check.sh                 # Check SAM checkpoint
```

## Quick Start

### 1. Submit a Job

```bash
# Navigate to project root
cd ~/Med3Tab-PFN

# Submit experiment 1 (method comparison)
sbatch cluster_scripts/slurm/slurm_exp1.sh

# Monitor job status
squeue -u $USER
tail -f logs/exp1_*.log
```

### 2. Run Locally (Testing)

```bash
# Test with one dataset
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets.yaml \
    --datasets gist
```

## Experiment Scripts

### Experiment 1: Method Comparison
Compare TabPFN, LoCalPFN, DenseNet121-3D, and ViT-3D.

```bash
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets.yaml \
    --n-splits 5            # 5-fold CV
```

### Experiment 3: Classification Head
Evaluate SAM-Med3D feature quality with a linear classifier.

```bash
python cluster_scripts/experiments/exp3_classifier.py \
    --config configs/datasets.yaml \
    --freeze-encoder        # Frozen encoder (fast)
    --epochs 20
```

### Preprocessing Comparison
Compare baseline, filtered, and ROI-cropped preprocessing.

```bash
python cluster_scripts/experiments/compare_classifier_preprocessing.py \
    --config configs/datasets.yaml
```

## Key Options

### ROI Cropping
Extract tumor-centered volumes:
```bash
--use-roi-crop --roi-margin 10 --roi-target-size 128
```

### Lesion Filtering
Remove low-quality samples:
```bash
--filter-preset recommended
# Or custom: --min-voxels 500 --min-dimension 5 --min-density 0.3
```

### MedIM Integration
Use MedIM for model loading:
```bash
--use-medim    # Enable (default)
--no-medim     # Disable
```

## SLURM Scripts

| Script | Description | Time | Memory |
|--------|-------------|------|--------|
| `slurm_exp1.sh` | Full method comparison | 8h | 32GB |
| `slurm_exp1_filtered.sh` | With lesion filtering | 6h | 32GB |
| `slurm_exp1_roi.sh` | With ROI cropping | 8h | 32GB |
| `slurm_exp3.sh` | Classification head | 4h | 16GB |
| `slurm_compare_classifier.sh` | Preprocessing comparison | 48h | 32GB |

### Customizing SLURM Scripts

Edit the script header:
```bash
#SBATCH --time=8:00:00           # Time limit
#SBATCH --mem=32G                # Memory
#SBATCH --gres=gpu:1             # GPU allocation
#SBATCH --mail-user=your@email   # Notifications
```

## Output Structure

```
results/
├── experiment1/
│   └── combined_benchmarks_summary.csv
├── classification_head/
│   ├── gist/
│   │   ├── best_model.pth
│   │   ├── training_log.csv
│   │   └── metrics.txt
│   └── lipo/
└── three_way_comparison/
    └── preprocessing_comparison.csv
```

## Documentation

- [Cluster Quick Start](../docs/cluster/QUICK_START.md)
- [Scripts Reference](../docs/cluster/SCRIPTS_REFERENCE.md)
- [Submit Checklist](../docs/cluster/SUBMIT_CHECKLIST.md)
- [ROI Experiments](../docs/cluster/ROI_EXPERIMENTS.md)

### Example: Verify setup
```bash
bash cluster_scripts/utils/check_setup.sh
```
