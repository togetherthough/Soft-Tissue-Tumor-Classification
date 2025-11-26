# Running Experiments with ROI-Cropping

## Quick Start

The ROI-cropping approach is now fully integrated into the experiment pipeline. You can enable it with a single flag.

## Single Dataset Experiments

### Example 1: TabPFN with ROI Cropping
```python
from med3pipe.pipelines import run_single_dataset

result = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    category="gist",
    ct_name="ct_GIST_roi",  # Different name to distinguish from full-volume
    
    # Enable ROI cropping
    use_roi_crop=True,
    roi_margin=10,           # 10 voxels around tumor bounding box
    roi_target_size=128,     # Final size after crop/resize/pad
    
    # Standard parameters
    split_ratio=0.8,
    seed=2025,
    device="cuda",
)

print(f"Accuracy: {result.tabpfn['metrics']['accuracy']:.3f}")
print(f"ROC AUC: {result.tabpfn['metrics']['roc_auc']:.3f}")
```

### Example 2: LoCalPFN with ROI Cropping
```python
from med3pipe.pipelines import run_single_dataset

result = run_single_dataset(
    method="localpfn",
    dataset_root="gist",
    category="gist",
    ct_name="ct_GIST_roi",
    
    # Enable ROI cropping
    use_roi_crop=True,
    roi_margin=15,           # Larger margin for more context
    roi_target_size=128,
    
    # LoCalPFN specific
    local_k=5,
    device="cuda",
)
```

## Multi-Dataset Experiments

### Example 3: Run Multiple Datasets with ROI Cropping (TabPFN)
```python
from med3pipe.pipelines import run_multi_tabpfn

results = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    
    # Enable ROI cropping for ALL datasets
    use_roi_crop=True,
    roi_margin=10,
    roi_target_size=128,
    
    # Other parameters
    device="cuda",
    save_summary=True,
    summary_path="results/roi_cropped_tabpfn_summary.csv",
)

# Results saved to CSV with per-dataset metrics
```

### Example 4: Run Multiple Datasets with ROI Cropping (LoCalPFN)
```python
from med3pipe.pipelines import run_multi_localpfn

results = run_multi_localpfn(
    config_path="configs/datasets.yaml",
    dataset_names=["gist", "lipo", "desmoid"],  # Select specific datasets
    
    # Enable ROI cropping
    use_roi_crop=True,
    roi_margin=10,
    roi_target_size=128,
    
    # LoCalPFN configuration
    local_k=5,
    local_fit_adapter=True,
    device="cuda",
    save_summary=True,
)
```

## Comparing ROI-Cropped vs Full-Volume

### Experiment Setup
Run experiments twice: once with ROI cropping, once without.

```python
from med3pipe.pipelines import run_single_dataset

# Experiment 1: Full-volume (baseline)
result_full = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    category="gist",
    ct_name="ct_GIST",           # Standard name
    use_roi_crop=False,          # Default: full-volume
    device="cuda",
)

# Experiment 2: ROI-cropped (proposed)
result_roi = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    category="gist",
    ct_name="ct_GIST_roi",       # Different name
    use_roi_crop=True,           # ROI cropping enabled
    roi_margin=10,
    roi_target_size=128,
    device="cuda",
)

# Compare
print("Full-volume AUC:", result_full.tabpfn['metrics']['roc_auc'])
print("ROI-cropped AUC:", result_roi.tabpfn['metrics']['roc_auc'])
```

## Parameter Tuning

### ROI Margin
Controls how much context around the tumor is included:

```python
# Small margin: Tighter crop, less background
result_tight = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    ct_name="ct_GIST_roi_margin5",
    use_roi_crop=True,
    roi_margin=5,    # Minimal context
)

# Medium margin: Balanced
result_medium = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    ct_name="ct_GIST_roi_margin10",
    use_roi_crop=True,
    roi_margin=10,   # Standard (default)
)

# Large margin: More surrounding anatomy
result_loose = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    ct_name="ct_GIST_roi_margin20",
    use_roi_crop=True,
    roi_margin=20,   # More context
)
```

### ROI Target Size
Controls final volume size (affects resolution vs. computation):

```python
# Smaller volume: Faster, less detail
result_96 = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    ct_name="ct_GIST_roi_size96",
    use_roi_crop=True,
    roi_target_size=96,   # Faster processing
)

# Standard volume: Balanced
result_128 = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    ct_name="ct_GIST_roi_size128",
    use_roi_crop=True,
    roi_target_size=128,  # Default
)

# Larger volume: More detail, slower
result_160 = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    ct_name="ct_GIST_roi_size160",
    use_roi_crop=True,
    roi_target_size=160,  # Higher resolution
)
```

## Important Notes

### 1. Data Separation
Use different `ct_name` values to keep ROI-cropped and full-volume data separate:
- Full-volume: `ct_GIST`, `ct_LIPO`, etc.
- ROI-cropped: `ct_GIST_roi`, `ct_LIPO_roi`, etc.

This prevents conflicts and allows fair comparison.

### 2. Embedding Re-extraction
When switching between full-volume and ROI-cropped:
- Set `skip_existing_embeddings=False` to force re-extraction
- Or use different `ct_name` values (embeddings stored separately)

```python
# Force re-extraction for ROI-cropped data
result = run_single_dataset(
    method="tabpfn",
    dataset_root="gist",
    ct_name="ct_GIST_roi",
    use_roi_crop=True,
    skip_existing_embeddings=False,  # Re-extract even if embeddings exist
)
```

### 3. Expected Improvements
ROI-cropping is expected to improve performance for:
- **Small tumors** (< 5cm): Resolution preserved
- **GAP-based features**: Tumor signal no longer diluted
- **Homogeneous lesions**: Better texture capture

May not help or could hurt for:
- **Very large tumors**: Already dominant in full volume
- **Multi-focal disease**: Cropping may miss context
- **Invasive tumors**: Boundary information may be lost

## Example: Full Ablation Study

```python
from med3pipe.pipelines import run_multi_tabpfn

# Experiment 1: Baseline (full-volume)
results_baseline = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    use_roi_crop=False,  # Full-volume
    device="cuda",
    summary_path="results/experiment_baseline.csv",
)

# Experiment 2: ROI-cropped with standard margin
results_roi_std = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    use_roi_crop=True,
    roi_margin=10,
    device="cuda",
    summary_path="results/experiment_roi_margin10.csv",
)

# Experiment 3: ROI-cropped with large margin
results_roi_large = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    use_roi_crop=True,
    roi_margin=20,
    device="cuda",
    summary_path="results/experiment_roi_margin20.csv",
)

# Compare results
import pandas as pd
df_baseline = pd.read_csv("results/experiment_baseline.csv")
df_roi_std = pd.read_csv("results/experiment_roi_margin10.csv")
df_roi_large = pd.read_csv("results/experiment_roi_margin20.csv")

comparison = pd.DataFrame({
    'dataset': df_baseline['dataset'],
    'baseline_auc': df_baseline['roc_auc'],
    'roi_margin10_auc': df_roi_std['roc_auc'],
    'roi_margin20_auc': df_roi_large['roc_auc'],
})
print(comparison)
```

## Troubleshooting

### Issue: Empty masks
**Error**: `Skipped N cases due to empty masks`

**Solution**: Some cases may have invalid/empty segmentation files. This is expected. The pipeline automatically skips them.

### Issue: Embeddings not updating
**Symptom**: Results identical to previous run despite changing `use_roi_crop`

**Solution**: Either:
1. Use different `ct_name` values, OR
2. Set `skip_existing_embeddings=False`

### Issue: Out of memory
**Symptom**: CUDA out of memory errors

**Solution**: Reduce `roi_target_size` (e.g., from 128 to 96)
