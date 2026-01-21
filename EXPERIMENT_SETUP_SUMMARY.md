# Classification Head Preprocessing Comparison - Setup Complete ✅

## What Was Created

I've prepared a complete experiment to compare three preprocessing approaches for the classification head. Here's what was created:

### 1. Main Experiment Script
**File:** `cluster_scripts/experiments/compare_classifier_preprocessing.py`

This script runs three experiments per dataset:
- **Baseline:** Full-volume, no filtering
- **Filtered Baseline:** Full-volume + lesion filtering (min_voxels=100, min_dimension=3, min_density=0.1)
- **ROI-Cropped:** Adaptive crop/pad (roi_margin=10, roi_target_size=128)

### 2. SLURM Job Script
**File:** `cluster_scripts/slurm/slurm_compare_classifier.sh`

Ready-to-submit cluster job with:
- 2-day time limit
- GPU allocation
- 32GB memory
- Automatic environment setup

### 3. Documentation
- **`docs/cluster/CLASSIFIER_PREPROCESSING_COMPARISON.md`** - Detailed documentation
- **`cluster_scripts/experiments/PREPROCESSING_COMPARISON_README.md`** - Quick reference
- **`cluster_scripts/README.md`** - Updated with new scripts

## Quick Start

### Option 1: Run on Cluster (Recommended)
```bash
# Navigate to project root
cd ~/Med3Tab-PFN  # or wherever your project is

# Submit the job
sbatch cluster_scripts/slurm/slurm_compare_classifier.sh

# Check job status
squeue -u $USER

# Monitor progress
tail -f logs/clf_compare_<job_id>.log
```

### Option 2: Run Locally (For Testing)
```bash
# Navigate to project root
cd ~/Med3Tab-PFN

# Run the experiment
python cluster_scripts/experiments/compare_classifier_preprocessing.py
```

## What the Experiment Does

For each dataset in your config (`datasets_cluster.yaml` or `datasets.yaml`), it runs 3 experiments:

| Experiment | Preprocessing | Purpose |
|------------|---------------|---------|
| **Baseline** | Full CT volume, no filtering | Establish baseline performance |
| **Filtered Baseline** | Full volume + quality filtering | Test if filtering improves signal |
| **ROI-Cropped** | Tumor-centered crop/pad | Focus on lesion-specific features |

## Expected Output

```
results/classification_head_comparison/
├── all_results.csv                      # Individual results for each dataset × experiment
├── preprocessing_comparison.csv         # Side-by-side comparison with delta metrics
├── baseline/
│   ├── gist/
│   │   ├── best_model.pth              # Best model checkpoint
│   │   ├── training_log.csv            # Training history
│   │   └── metrics.txt                 # Final metrics
│   ├── lipo/
│   └── ... (all datasets)
├── filtered_baseline/
│   └── ... (same structure)
└── roi_cropped/
    └── ... (same structure)
```

### Key Output Files

#### `preprocessing_comparison.csv`
Side-by-side comparison showing:
- Performance metrics for all 3 approaches
- **Delta metrics** (improvement vs. baseline)
- Per-dataset results

Example:
```csv
dataset,accuracy_baseline,accuracy_filtered,accuracy_roi,accuracy_delta_filtered,accuracy_delta_roi,auc_baseline,auc_filtered,auc_roi,...
gist,0.7891,0.8123,0.8234,+0.0232,+0.0343,0.8456,0.8678,0.8789,...
lipo,0.8456,0.8567,0.8678,+0.0111,+0.0222,0.8901,0.9012,0.9123,...
```

## Runtime Expectations

- **Per dataset:** ~2-3 hours (depends on dataset size)
- **Total (6 datasets):** ~12-18 hours
- **SLURM allocation:** 2 days (includes buffer)

Each dataset runs 3 experiments sequentially:
1. Baseline → 2. Filtered Baseline → 3. ROI-Cropped

## Configuration

### Experiment Parameters

Edit `compare_classifier_preprocessing.py` if you want to change settings:

```python
# Line ~431 - Training parameters
shared_params = {
    'freeze_encoder': True,
    'num_epochs': 20,        # Training epochs
    'batch_size': 4,         # Batch size (reduce if OOM)
    'learning_rate': 1e-3,   # Learning rate
    'weight_decay': 1e-4,    # Weight decay
    'dropout': 0.3,          # Dropout rate
}

# Line ~443 - Lesion filtering (for Filtered Baseline)
lesion_filter_params = {
    'min_voxels': 100,       # Minimum voxels
    'min_dimension': 3,      # Minimum dimension
    'min_density': 0.1,      # Minimum density
}

# Line ~450 - ROI cropping (for ROI-Cropped)
roi_crop_params = {
    'roi_margin': 10,        # ROI margin in voxels
    'roi_target_size': 128,  # Target size after crop
}
```

### SLURM Settings

Edit `slurm_compare_classifier.sh` to adjust cluster resources:

```bash
#SBATCH --time=2-00:00:00              # Time limit
#SBATCH --mem=32G                      # Memory
#SBATCH --cpus-per-task=4              # CPU cores
#SBATCH --mail-user=YOUR_EMAIL         # Your email for notifications
```

## Troubleshooting

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
