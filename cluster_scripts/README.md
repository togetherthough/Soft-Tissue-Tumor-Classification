# Cluster Scripts

This directory contains scripts for running experiments on HPC clusters.

## Quick Reference

### Main Experiments
- **`exp1_benchmarks.py`** - Experiment 1: Method comparison (TabPFN, LoCalPFN, DenseNet, ViT)
- **`exp3_classifier.py`** - Experiment 3: Classification head training
- **`compare_tabpfn.py`** - Three-way comparison using TabPFN (baseline, filtered, ROI)
- **`compare_localpfn.py`** - Three-way comparison using LoCalPFN
- **`ablate_roi.py`** - ROI margin ablation study

### SLURM Scripts
- **`slurm_exp1*.sh`** - SLURM jobs for Experiment 1
- **`slurm_exp3*.sh`** - SLURM jobs for Experiment 3
- **`run_all_comparisons.sh`** - Run all three-way comparison experiments

### Utilities
- **`check_setup.sh`** - Verify environment and dependencies
- **`verify_config.sh`** - Validate YAML configuration
- **`verify_weights.py`** - Check SAM-Med3D checkpoint validity
- **`compare_weights.py`** - Compare checkpoint files

## Documentation

For detailed documentation, see:
- **[Scripts Reference](../docs/cluster/SCRIPTS_REFERENCE.md)** - Complete guide to all scripts
- **[ROI Experiments](../docs/cluster/ROI_EXPERIMENTS.md)** - ROI cropping experiments
- **[Quick Start](../docs/cluster/QUICK_START.md)** - Getting started on cluster
- **[Submit Checklist](../docs/cluster/SUBMIT_CHECKLIST.md)** - Pre-submission checklist

## Usage

All scripts use k-fold stratified cross-validation (default: 5 folds).

### Example: Run benchmarks
```bash
python cluster_scripts/exp1_benchmarks.py --config configs/datasets.yaml
```

### Example: Submit to SLURM
```bash
sbatch cluster_scripts/slurm_exp1.sh
```

### Example: Change number of folds
```bash
python cluster_scripts/exp1_benchmarks.py --n-splits 10
```
