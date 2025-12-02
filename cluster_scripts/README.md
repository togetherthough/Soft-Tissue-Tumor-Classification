# Cluster Scripts

This directory contains scripts for running experiments on HPC clusters.

## Directory Structure

```
cluster_scripts/
├── experiments/          # Main experiment scripts
├── slurm/               # SLURM job submission scripts
├── utils/               # Utility and diagnostic scripts
└── README.md
```

## Quick Reference

### 📁 experiments/
Main experiment scripts using k-fold cross-validation:
- **`exp1_benchmarks.py`** - Experiment 1: Method comparison (TabPFN, LoCalPFN, DenseNet, ViT)
- **`exp3_classifier.py`** - Experiment 3: Classification head training
- **`compare_tabpfn.py`** - Three-way comparison using TabPFN (baseline, filtered, ROI)
- **`compare_localpfn.py`** - Three-way comparison using LoCalPFN
- **`compare_classifier_preprocessing.py`** - Three-way preprocessing comparison for classification head
- **`ablate_roi.py`** - ROI margin ablation study
- **`run_all_comparisons.sh`** - Run all three-way comparison experiments

### 📁 slurm/
SLURM job submission scripts:
- **`slurm_exp1.sh`**, `slurm_exp1_filtered.sh`, `slurm_exp1_quick.sh` - Experiment 1 variants
- **`slurm_exp3.sh`**, `slurm_exp3_filtered.sh`, `slurm_exp3_quick.sh` - Experiment 3 variants
- **`slurm_compare_classifier.sh`** - Three-way preprocessing comparison for classification head

### 📁 utils/
Utility and diagnostic scripts:
- **`check_setup.sh`** - Verify environment and dependencies
- **`verify_config.sh`** - Validate YAML configuration
- **`verify_weights.py`** - Check SAM-Med3D checkpoint validity
- **`compare_weights.py`** - Compare checkpoint files
- **`diagnose_experiment3.sh`** - Debug Experiment 3 issues
- **`fix_dependencies.sh`** - Fix dependency issues
- **`list_datasets.sh`** - List available datasets
- **`quick_weight_check.sh`** - Quick checkpoint verification

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
python cluster_scripts/experiments/exp1_benchmarks.py --config configs/datasets.yaml
```

### Example: Submit to SLURM
```bash
sbatch cluster_scripts/slurm/slurm_exp1.sh
```

### Example: Change number of folds
```bash
python cluster_scripts/experiments/exp1_benchmarks.py --n-splits 10
```

### Example: Verify setup
```bash
bash cluster_scripts/utils/check_setup.sh
```
