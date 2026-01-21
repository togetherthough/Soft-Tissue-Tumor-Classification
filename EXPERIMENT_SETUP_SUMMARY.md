# Experiment Setup Guide

> Complete guide for setting up and running classification experiments

## Overview

This guide covers the three-way preprocessing comparison experiment, which evaluates different preprocessing strategies for the SAM-Med3D classification head.

## Experiment Structure

| Approach | Description | Purpose |
|----------|-------------|---------|
| **Baseline** | Full CT volume, no filtering | Establish baseline performance |
| **Filtered** | Full volume + quality filtering | Test if filtering improves signal |
| **ROI-Cropped** | Tumor-centered crop/pad | Focus on lesion-specific features |

---

## Quick Start

### On Cluster (Recommended)

```bash
# Navigate to project root
cd ~/Med3Tab-PFN

# Submit job
sbatch cluster_scripts/slurm/slurm_compare_classifier.sh

# Monitor
squeue -u $USER
tail -f logs/clf_compare_*.log
```

### Locally (Testing)

```bash
cd ~/Med3Tab-PFN
python cluster_scripts/experiments/compare_classifier_preprocessing.py
```

---

## Configuration

### Training Parameters

Located in `compare_classifier_preprocessing.py`:

```python
shared_params = {
    'freeze_encoder': True,
    'num_epochs': 20,
    'batch_size': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.3,
}

lesion_filter_params = {
    'min_voxels': 100,
    'min_dimension': 3,
    'min_density': 0.1,
}

roi_crop_params = {
    'roi_margin': 10,
    'roi_target_size': 128,
}
```

### SLURM Resources

Edit `slurm_compare_classifier.sh`:
```bash
#SBATCH --time=2-00:00:00    # Time limit
#SBATCH --mem=32G            # Memory
#SBATCH --cpus-per-task=4    # CPU cores
#SBATCH --gres=gpu:1         # GPU allocation
```

---

## Output Structure

```
results/classification_head_comparison/
├── all_results.csv              # Per-experiment results
├── preprocessing_comparison.csv # Side-by-side comparison
├── baseline/
│   └── <dataset>/
│       ├── best_model.pth
│       ├── training_log.csv
│       └── metrics.txt
├── filtered_baseline/
│   └── ...
└── roi_cropped/
    └── ...
```

### Comparison CSV Format

```csv
dataset,accuracy_baseline,accuracy_filtered,accuracy_roi,delta_filtered,delta_roi,...
gist,0.789,0.812,0.823,+0.023,+0.034,...
lipo,0.845,0.856,0.867,+0.011,+0.022,...
```

---

## Runtime Expectations

| Dataset Count | Per-Dataset | Total |
|---------------|-------------|-------|
| 1 dataset | 2-3 hours | 2-3 hours |
| 2 datasets | 2-3 hours | 4-6 hours |
| 6 datasets | 2-3 hours | 12-18 hours |

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
'batch_size': 2,
```

### Job Takes Too Long
```python
# Test with fewer datasets
datasets_to_run = ['gist', 'lipo']

# Reduce epochs
'num_epochs': 10,
```

### Missing Checkpoint
```bash
bash cluster_scripts/utils/quick_weight_check.sh
# Expected: sam-med3d/ckpt/sam_med3d_turbo.pth
```

---

## Analysis

After completion:

```bash
# View comparison
cat results/classification_head_comparison/preprocessing_comparison.csv

# Key questions to answer:
# 1. Which preprocessing works best overall?
# 2. Are there dataset-specific trends?
# 3. Do filtered/ROI approaches improve performance?
```

---

## Related Documentation

- [Preprocessing Comparison](docs/cluster/CLASSIFIER_PREPROCESSING_COMPARISON.md)
- [Cluster Quick Start](docs/cluster/QUICK_START.md)
- [Lesion Filtering](docs/FILTERING.md)
- [ROI Cropping](docs/roi_cropping.md)

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch size in the script
'batch_size': 2,  # Change from 4 to 2
```

**2. Job Takes Too Long**
```python
# Test with fewer datasets first
datasets_to_run = ['gist', 'lipo']  # Instead of all datasets

# Or reduce epochs
'num_epochs': 10,  # Change from 20 to 10
```

**3. Missing SAM-Med3D Checkpoint**
```bash
# Verify checkpoint exists
bash cluster_scripts/utils/quick_weight_check.sh

# Expected location:
# sam-med3d/ckpt/sam_med3d_turbo.pth
```

**4. Wrong Config File**
The script automatically tries:
1. `configs/datasets_cluster.yaml` (cluster-specific paths)
2. `configs/datasets.yaml` (fallback)

Make sure at least one exists with correct dataset paths.

## Next Steps

### After Submission

1. **Monitor the job:**
   ```bash
   # Check status
   squeue -u $USER
   
   # Watch logs in real-time
   tail -f logs/clf_compare_<job_id>.log
   ```

2. **Check for errors:**
   ```bash
   # If job fails, check error log
   cat logs/clf_compare_error_<job_id>.log
   ```

### After Completion

1. **Review results:**
   ```bash
   # View comparison table
   cat results/classification_head_comparison/preprocessing_comparison.csv
   
   # View all results
   cat results/classification_head_comparison/all_results.csv
   ```

2. **Analyze findings:**
   - Which preprocessing works best overall?
   - Are there dataset-specific trends?
   - Do filtered or ROI approaches improve performance?

3. **Share results:**
   - Results are in CSV format for easy analysis
   - Can be loaded into Python/R for visualization
   - Individual model checkpoints saved for further analysis

## File Summary

### Created Files
- ✅ `cluster_scripts/experiments/compare_classifier_preprocessing.py` - Main experiment script
- ✅ `cluster_scripts/slurm/slurm_compare_classifier.sh` - SLURM job script
- ✅ `docs/cluster/CLASSIFIER_PREPROCESSING_COMPARISON.md` - Detailed documentation
- ✅ `cluster_scripts/experiments/PREPROCESSING_COMPARISON_README.md` - Quick reference
- ✅ `EXPERIMENT_SETUP_SUMMARY.md` - This file

### Updated Files
- ✅ `cluster_scripts/README.md` - Added new scripts to quick reference

## Support

For detailed information, see:
- **Full documentation:** `docs/cluster/CLASSIFIER_PREPROCESSING_COMPARISON.md`
- **Quick reference:** `cluster_scripts/experiments/PREPROCESSING_COMPARISON_README.md`
- **Cluster guides:** `docs/cluster/QUICK_START.md`

For debugging:
```bash
# Verify environment setup
bash cluster_scripts/utils/check_setup.sh

# List available datasets
bash cluster_scripts/utils/list_datasets.sh
```

---

## Ready to Run! 🚀

Everything is set up. When you're ready, just run:
```bash
sbatch cluster_scripts/slurm/slurm_compare_classifier.sh
```

The experiment will compare all three preprocessing approaches across your datasets and provide comprehensive comparison results.
