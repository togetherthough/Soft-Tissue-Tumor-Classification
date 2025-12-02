# Classification Head Preprocessing Comparison Experiment

## Quick Start

### Run on Cluster
```bash
sbatch cluster_scripts/slurm/slurm_compare_classifier.sh
```

### Run Locally (for testing)
```bash
python cluster_scripts/experiments/compare_classifier_preprocessing.py
```

## What This Experiment Does

Compares **3 preprocessing approaches** for the classification head:

| Approach | Description | Key Parameters |
|----------|-------------|----------------|
| **Baseline** | Full CT volume, no filtering | Standard preprocessing only |
| **Filtered Baseline** | Full volume + lesion filtering | `min_voxels=100`<br>`min_dimension=3`<br>`min_density=0.1` |
| **ROI-Cropped** | Tumor-centered crop/pad | `roi_margin=10`<br>`roi_target_size=128` |

## Expected Output

```
results/classification_head_comparison/
├── all_results.csv                    # All individual results
├── preprocessing_comparison.csv       # Side-by-side comparison
├── baseline/[dataset]/               # Baseline results per dataset
├── filtered_baseline/[dataset]/      # Filtered results per dataset
└── roi_cropped/[dataset]/            # ROI results per dataset
```

## Key Metrics

Each experiment reports:
- **Accuracy** - Overall classification accuracy
- **ROC AUC** - Area under ROC curve (main metric)
- **F1 Score** - Harmonic mean of precision/recall
- **Precision** - Positive predictive value
- **Recall** - Sensitivity

Plus **delta metrics** showing improvement vs. baseline:
- `accuracy_delta_filtered` - Improvement from filtering
- `accuracy_delta_roi` - Improvement from ROI cropping
- Similar deltas for AUC and F1

## Example Results

```csv
dataset,accuracy_baseline,accuracy_filtered_baseline,accuracy_roi_cropped,accuracy_delta_filtered,accuracy_delta_roi
gist,0.7891,0.8123,0.8234,+0.0232,+0.0343
lipo,0.8456,0.8567,0.8678,+0.0111,+0.0222
```

## Configuration

### Modify Experiment Parameters

Edit `compare_classifier_preprocessing.py`:

```python
# Line ~431 - Shared parameters
shared_params = {
    'freeze_encoder': True,
    'num_epochs': 20,        # Change training epochs
    'batch_size': 4,         # Change batch size (reduce if OOM)
    'learning_rate': 1e-3,   # Change learning rate
    ...
}

# Line ~443 - Lesion filtering
lesion_filter_params = {
    'min_voxels': 100,       # Change minimum voxels
    'min_dimension': 3,      # Change minimum dimension
    'min_density': 0.1,      # Change minimum density
}

# Line ~450 - ROI cropping
roi_crop_params = {
    'roi_margin': 10,        # Change ROI margin
    'roi_target_size': 128,  # Change target size
}
```

### Run Specific Datasets Only

```python
# Line ~425 - Filter datasets
datasets_to_run = list(cfg['datasets'].keys())
# Change to:
datasets_to_run = ['gist', 'lipo']  # Run only these datasets
```

## Cluster Job Configuration

Edit `slurm_compare_classifier.sh` for cluster settings:

```bash
#SBATCH --time=2-00:00:00    # Adjust time limit
#SBATCH --mem=32G            # Adjust memory
#SBATCH --cpus-per-task=4    # Adjust CPU cores
#SBATCH --mail-user=YOUR_EMAIL@example.com  # Set your email
```

## Troubleshooting

### Out of Memory
Reduce batch size in the script:
```python
'batch_size': 2,  # Reduce from 4
```

### Job Takes Too Long
Reduce epochs or run fewer datasets:
```python
'num_epochs': 10,  # Reduce from 20
datasets_to_run = ['gist']  # Test with one dataset
```

### Missing Checkpoint
Verify SAM-Med3D weights exist:
```bash
bash cluster_scripts/utils/quick_weight_check.sh
```

## Related Files

- **Main script:** `cluster_scripts/experiments/compare_classifier_preprocessing.py`
- **SLURM job:** `cluster_scripts/slurm/slurm_compare_classifier.sh`
- **Documentation:** `docs/cluster/CLASSIFIER_PREPROCESSING_COMPARISON.md`
- **Config:** `configs/datasets_cluster.yaml` (or `datasets.yaml`)

## Support

For issues:
1. Check error logs: `logs/clf_compare_error_<job_id>.log`
2. Verify setup: `bash cluster_scripts/utils/check_setup.sh`
3. Review full documentation: `docs/cluster/CLASSIFIER_PREPROCESSING_COMPARISON.md`
