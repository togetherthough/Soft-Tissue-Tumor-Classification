# Preprocessing Comparison Experiments

This directory contains GPU-ready experiment scripts for comprehensive preprocessing comparisons.

## Quick Start

### Three-Way Comparison (Recommended)
Compares **three** preprocessing approaches:
1. **Baseline**: Full-volume, no filtering
2. **Filtered Baseline**: Full-volume with lesion size filtering
3. **ROI-Cropped**: Adaptive crop/pad (tumor-centered volumes)

```powershell
cd cluster_scripts
.\run_all_three_way_experiments.ps1
```

## Individual Experiments

### 1. TabPFN Three-Way Comparison
```bash
python run_experiment_tabpfn_roi_comparison.py
```
Compares all three preprocessing approaches using TabPFN classifier.

**Output**: `results/three_way_comparison/`
- `baseline_summary.csv` - Full-volume, no filtering
- `filtered_baseline_summary.csv` - Full-volume with lesion filtering
- `roi_cropped_summary.csv` - ROI-cropped results
- `three_way_comparison.csv` - Side-by-side with deltas

### 2. LoCalPFN Three-Way Comparison
```bash
python run_experiment_localpfn_three_way.py
```
Compares all three preprocessing approaches using LoCalPFN classifier.

**Output**: `results/three_way_comparison_localpfn/`
- `baseline_summary.csv` - Full-volume, no filtering
- `filtered_baseline_summary.csv` - Full-volume with lesion filtering
- `roi_cropped_summary.csv` - ROI-cropped results
- `three_way_comparison.csv` - Side-by-side with deltas

### 3. ROI Margin Ablation
```bash
python run_experiment_roi_ablation.py
```
Tests different ROI margins (5, 10, 15, 20 voxels) to find optimal context size.

**Output**: `results/roi_ablation/`
- `margin_5_summary.csv`, `margin_10_summary.csv`, etc.
- `ablation_comparison.csv` - Performance comparison across margins

## Experiment Details

### Three Approaches Compared

#### 1. Baseline (Full-Volume, No Filtering)
- **Preprocessing**: Entire CT scan resized to 128³
- **Filtering**: None - all cases included
- **Feature Extraction**: Global Average Pooling over entire volume
- **Issues**:
  - Small tumors lose resolution (downsampling artifacts)
  - GAP includes >90% background (liver, air, bone)
  - Noisy small lesions included

#### 2. Filtered Baseline (Full-Volume + Lesion Filtering)
- **Preprocessing**: Entire CT scan resized to 128³ (same as baseline)
- **Filtering**: Remove cases with very small/noisy lesions:
  - Minimum 100 voxels total
  - Minimum 3 voxels in each dimension
  - Minimum 10% density (lesion vs. bounding box)
- **Feature Extraction**: Global Average Pooling over entire volume
- **Purpose**: Isolate the effect of lesion quality filtering

#### 3. ROI-Cropped (Adaptive Crop/Pad)
- **Preprocessing**: Crop around tumor with margin, resize/pad to 128³
- **Filtering**: None (handled by cropping logic)
- **Feature Extraction**: Global Average Pooling now mostly tumor
- **Benefits**:
  - Small tumors preserve resolution
  - Tumor centered and dominant in frame
  - GAP signal-to-noise ratio greatly improved

### Why Three-Way Comparison?

This design isolates the contributions of:
- **Filtering alone**: Baseline vs. Filtered Baseline
- **ROI-cropping alone**: Baseline vs. ROI-Cropped
- **Combined effect**: Baseline vs. Filtered Baseline vs. ROI-Cropped

Expected hypotheses:
- Filtered Baseline > Baseline (removing noise helps)
- ROI-Cropped > Baseline (better resolution + focus)
- ROI-Cropped vs. Filtered Baseline (tests if cropping is better than filtering)

### Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (tested on RTX 3090, A100)
- **VRAM**: 8GB+ for baseline experiments, 16GB+ for ablation
- **CPU**: Multi-core recommended for data loading
- **RAM**: 32GB+ recommended

### Runtime Estimates

Per dataset (e.g., GIST, LIPO, Desmoid):
- **Preparation**: 5-10 min (first time only)
- **Feature extraction**: 10-20 min (GPU) or 2-4 hours (CPU)
- **TabPFN training**: 1-5 min
- **LoCalPFN training**: 5-15 min (with adapter)

Full suite (all 3 experiments, all datasets):
- **First run**: 4-6 hours (includes data preparation)
- **Subsequent runs**: 1-2 hours (reuses embeddings)

### GPU Memory Usage

- **Baseline**: ~4-6 GB VRAM
- **ROI-cropped**: ~3-5 GB VRAM (slightly lower due to focused crops)
- **Ablation study**: ~4-6 GB VRAM (per margin setting)

To reduce memory:
- Lower `roi_target_size` (e.g., 96 instead of 128)
- Use smaller batch sizes in data loading

## Viewing Results

### Quick Look
```bash
# TabPFN comparison
cat results/roi_comparison/comparison_summary.csv

# LoCalPFN comparison
cat results/roi_comparison_localpfn/comparison_summary.csv

# Margin ablation
cat results/roi_ablation/ablation_comparison.csv
```

### Python Analysis
```python
import pandas as pd

# Load comparison
df = pd.read_csv('results/roi_comparison/comparison_summary.csv')

# View improvements
print(df[['dataset', 'accuracy_delta', 'auc_delta']])

# Average improvement
print(f"Average AUC improvement: {df['auc_delta'].mean():.4f}")
```

### Jupyter Notebook
Open `notebooks/experiments/ROI_Comparison_Analysis.ipynb` (if available) for interactive visualization.

## Customization

### Modify Experiment Parameters

Edit the Python scripts directly:

```python
# In run_experiment_tabpfn_roi_comparison.py

# Change ROI margin
roi_margin=15,  # Default is 10

# Change ROI size
roi_target_size=96,  # Default is 128 (faster but lower resolution)

# Change PCA components
n_components_max=300,  # Default is 500

# Select specific datasets
dataset_names=["gist", "lipo"],  # Default is all datasets in config
```

### Add New Experiments

Create a new script based on templates:

```python
#!/usr/bin/env python3
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med3pipe.pipelines import run_multi_tabpfn

# Your custom experiment here
results = run_multi_tabpfn(
    config_path=project_root / "configs" / "datasets.yaml",
    use_roi_crop=True,
    roi_margin=25,  # Custom margin
    # ... other parameters
)
```

## Troubleshooting

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce `roi_target_size` to 96 or 64
2. Close other GPU applications
3. Use CPU: set `device="cpu"` in scripts

### Datasets Not Found
```
FileNotFoundError: Could not resolve dataset_root
```

**Solutions**:
1. Check `configs/datasets.yaml` has correct paths
2. Ensure datasets are downloaded and extracted
3. Verify `dataset_root` points to correct location

### Empty Masks Error
```
Skipped N cases due to empty masks
```

**Explanation**: Some cases have invalid segmentation files. This is expected and handled gracefully. The pipeline will skip them.

### Slow on CPU
Feature extraction on CPU can take 2-4 hours per dataset.

**Solutions**:
1. Use GPU if available
2. Reduce number of datasets
3. Set `skip_existing_embeddings=True` to reuse previous embeddings

## Results Interpretation

### Comparison Summary Fields

- `accuracy_baseline` / `accuracy_roi`: Classification accuracy (0-1)
- `macro_f1_baseline` / `macro_f1_roi`: F1 score (0-1)
- `roc_auc_baseline` / `roc_auc_roi`: Area under ROC curve (0-1)
- `accuracy_delta`: Improvement in accuracy (positive = better)
- `f1_delta`: Improvement in F1 score
- `auc_delta`: Improvement in AUC

### Expected Results

**ROI-cropping should help when**:
- Small tumors (< 5cm): Resolution preserved
- Homogeneous tumors: Texture features enhanced
- Clear boundaries: Cropping removes irrelevant anatomy

**ROI-cropping may not help when**:
- Very large tumors: Already dominant in full volume
- Multi-focal disease: May crop out important lesions
- Infiltrative tumors: Boundary context important

## Citation

If you use these experiments in your research, please cite:
```
[Your paper citation here]
```

## Support

For issues or questions:
1. Check existing issues on GitHub
2. Open a new issue with experiment logs
3. Include GPU specs and dataset info
