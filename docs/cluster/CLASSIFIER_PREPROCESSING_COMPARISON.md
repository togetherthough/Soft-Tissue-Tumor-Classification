# Classification Head Preprocessing Comparison

This experiment compares the performance of three different preprocessing approaches for the classification head on SAM-Med3D features.

## Overview

**Script:** `cluster_scripts/experiments/compare_classifier_preprocessing.py`  
**SLURM Job:** `cluster_scripts/slurm/slurm_compare_classifier.sh`  
**Output:** `results/classification_head_comparison/`

## Preprocessing Approaches

### 1. Baseline (Full-Volume, No Filtering)
- Uses the complete CT volume
- No lesion filtering applied
- Standard preprocessing only (normalization, resizing)
- **Purpose:** Establish baseline performance

### 2. Filtered Baseline (Full-Volume + Lesion Filtering)
- Uses the complete CT volume
- Applies lesion size filtering to remove low-quality cases
- **Filtering criteria:**
  - `min_voxels=100`: Minimum 100 voxels after preprocessing
  - `min_dimension=3`: At least 3 voxels in each dimension
  - `min_density=0.1`: At least 10% of bounding box is lesion
- **Purpose:** Test if filtering improves signal quality

### 3. ROI-Cropped (Adaptive Crop/Pad)
- Crops volumes to tumor-centered regions
- Adapts to lesion size with configurable margin
- **ROI parameters:**
  - `roi_margin=10`: 10 voxels of context around lesion
  - `roi_target_size=128`: Final volume size after crop/pad
- **Purpose:** Focus on lesion-specific features, reduce noise

## Experimental Setup

### Common Parameters
- **Model:** SAM-Med3D ViT-B encoder (frozen)
- **Classification head:** Single linear layer with dropout
- **Epochs:** 20
- **Batch size:** 4
- **Learning rate:** 1e-3
- **Weight decay:** 1e-4
- **Dropout:** 0.3
- **Image size:** 128³

### Datasets
Runs on all datasets defined in `configs/datasets_cluster.yaml` (or `configs/datasets.yaml`).

## Usage

### Local Execution
```bash
# From project root
python cluster_scripts/experiments/compare_classifier_preprocessing.py
```

### Cluster Submission
```bash
# Submit SLURM job
sbatch cluster_scripts/slurm/slurm_compare_classifier.sh

# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/clf_compare_<job_id>.log
```

### Expected Runtime
- **Per dataset:** ~2-3 hours (depends on dataset size)
- **Total (6 datasets × 3 experiments):** ~12-18 hours
- **Recommended SLURM time:** 2 days (includes buffer)

## Output Structure

```
results/classification_head_comparison/
├── all_results.csv                      # All experiment results
├── preprocessing_comparison.csv         # Side-by-side comparison
├── baseline/
│   ├── gist/
│   │   ├── best_model.pth
│   │   ├── training_log.csv
│   │   └── metrics.txt
│   └── lipo/
│       └── ...
├── filtered_baseline/
│   └── ...
└── roi_cropped/
    └── ...
```

### Output Files

#### `all_results.csv`
Individual results for each dataset × experiment combination:
```csv
dataset,experiment,best_epoch,best_auc,accuracy,auc,precision,recall,f1,output_dir
gist,baseline,15,0.8234,0.7891,0.8234,0.7654,0.8123,0.7882,results/.../baseline/gist
gist,filtered_baseline,12,0.8456,0.8123,0.8456,0.7891,0.8345,0.8112,results/.../filtered_baseline/gist
...
```

#### `preprocessing_comparison.csv`
Side-by-side comparison with delta metrics:
```csv
dataset,accuracy_baseline,accuracy_filtered_baseline,accuracy_roi_cropped,accuracy_delta_filtered,accuracy_delta_roi,...
gist,0.7891,0.8123,0.8234,+0.0232,+0.0343,...
lipo,0.8456,0.8567,0.8678,+0.0111,+0.0222,...
...
```

Includes metrics for:
- **Accuracy**
- **ROC AUC**
- **F1 Score**
- **Delta metrics** (improvement vs. baseline)

## Analysis

### Key Questions
1. **Does lesion filtering improve performance?**
   - Compare `filtered_baseline` vs `baseline`
   - Check `accuracy_delta_filtered`, `auc_delta_filtered`

2. **Does ROI cropping help?**
   - Compare `roi_cropped` vs `baseline`
   - Check `accuracy_delta_roi`, `auc_delta_roi`

3. **Which approach is best per dataset?**
   - Examine per-dataset results in comparison table
   - Consider dataset-specific characteristics

4. **What is the average improvement?**
   - Review summary statistics in output
   - Compare mean delta metrics

### Interpretation Guidelines

**Positive deltas (+)** indicate improvement over baseline:
- `+0.05` or more: Substantial improvement
- `+0.02` to `+0.05`: Moderate improvement
- `+0.01` to `+0.02`: Minor improvement
- Below `+0.01`: Negligible improvement

**Negative deltas (-)** indicate degradation:
- May suggest overfitting or information loss
- Could indicate approach is not suitable for specific dataset

## Troubleshooting

### Common Issues

**1. Out of memory (OOM) errors**
```bash
# Reduce batch size in the script
# Edit compare_classifier_preprocessing.py, line with 'batch_size'
batch_size: int = 2  # Reduce from 4 to 2
```

**2. Missing datasets**
```bash
# Check dataset configuration
cat configs/datasets_cluster.yaml

# Verify dataset paths exist
bash cluster_scripts/utils/list_datasets.sh
```

**3. Checkpoint not found**
```bash
# Verify SAM-Med3D checkpoint
bash cluster_scripts/utils/quick_weight_check.sh

# Expected location:
# SAM-Med3D-main/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth
```

**4. Job fails for specific dataset**
- Check individual error logs in job output
- Experiment continues with other datasets even if one fails
- Failed datasets will have `error` field in `all_results.csv`

### Debug Mode

To run a single dataset for testing:
```python
# Edit compare_classifier_preprocessing.py
# Change line:
datasets_to_run = list(cfg['datasets'].keys())
# To:
datasets_to_run = ['gist']  # Test with single dataset
```

## Related Experiments

- **`exp3_classifier.py`** - Single preprocessing configuration for classification head
- **`compare_tabpfn.py`** - Similar comparison for TabPFN method
- **`compare_localpfn.py`** - Similar comparison for LoCalPFN method
- **`ablate_roi.py`** - ROI margin ablation study

## References

See also:
- [Classification Head Training](../technical/classification-head.md) (if exists)
- [ROI Cropping Documentation](ROI_EXPERIMENTS.md)
- [Experiment 3 Documentation](experiment3/README.md) (if exists)
- [Scripts Reference](SCRIPTS_REFERENCE.md)
