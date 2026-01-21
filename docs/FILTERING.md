# Lesion Size Filtering

> Quality-based sample filtering to remove noisy or low-quality lesions from training data

## Overview

Lesion size filtering removes samples with lesions that are too small, poorly shaped, or have low density to provide meaningful signal for classification. This improves model robustness by focusing on high-quality samples.

## Quick Start

### Prerequisites

The repository includes a tracked `data/lesion_size_analysis.csv` file, so filtering works immediately after cloning.

### Basic Usage

```python
from med3pipe import run_single_dataset

result = run_single_dataset(
    method="tabpfn",
    dataset_root="data/gist",
    category="gist",
    ct_name="ct_GIST",
    min_voxels=500,       # Minimum voxel count
    min_dimension=5,      # Minimum bounding box dimension
    min_density=0.3,      # Minimum lesion density
)
```

### Command Line

```bash
# Using preset
python cluster_scripts/experiments/exp1_benchmarks.py \
    --filter-preset recommended

# Custom thresholds
python cluster_scripts/experiments/exp1_benchmarks.py \
    --min-voxels 500 \
    --min-dimension 5 \
    --min-density 0.3
```

---

## Filter Parameters

| Parameter | Description | Range | Example |
|-----------|-------------|-------|---------|
| `min_voxels` | Minimum lesion voxel count | Integer ≥ 0 | 500 |
| `min_dimension` | Minimum bounding box dimension | Integer ≥ 0 | 5 |
| `min_density` | Minimum lesion density (voxels/bbox_volume) | 0.0 - 1.0 | 0.3 |

## Recommended Thresholds

| Preset | min_voxels | min_dimension | min_density | Retention |
|--------|------------|---------------|-------------|-----------|
| **Recommended** | 500 | 5 | 0.3 | ~34% |
| **Conservative** | 1000 | 10 | 0.3 | ~20% |
| **Lenient** | 200 | 3 | - | ~60% |

```bash
# Use presets
--filter-preset recommended
--filter-preset conservative
--filter-preset lenient
```

---

## Python API

### LesionSizeFilter Class

```python
from med3pipe import LesionSizeFilter

# Create filter
filter_obj = LesionSizeFilter(
    min_voxels=500,
    min_dimension=5,
    min_density=0.3,
    lesion_csv_path=None,  # Auto-detected
)

# Check if active
if filter_obj.is_enabled():
    print(f"Filtering active: {len(filter_obj.get_valid_cases())} cases")

# Print summary
filter_obj.print_filter_summary()
```

### Pipeline Integration

```python
from med3pipe import run_single_dataset, run_classification_head_experiment

# Method 1: Pass filter object
result = run_single_dataset(..., lesion_filter=filter_obj)

# Method 2: Pass parameters directly
result = run_single_dataset(
    ...,
    min_voxels=500,
    min_dimension=5,
    min_density=0.3,
)

# Classification head
result = run_classification_head_experiment(
    ...,
    lesion_filter=filter_obj,
)
```

---

## Output Directories

Filtered results are saved in **separate folders** with descriptive names:

```
results/classification_head/
├── gist/                            # Unfiltered (all data)
├── gist_filtered_v500_d5_ρ0.30/     # Filtered (quality subset)
└── ...
```

This ensures:
- ✅ No overwriting of unfiltered results
- ✅ Clear identification of filter settings
- ✅ Easy comparison between approaches

---

## Regenerating the Analysis File

If you add new datasets, regenerate the lesion analysis:

```bash
# Generate analysis
python scripts/analysis/analyze_lesion_sizes.py --config configs/datasets.yaml

# Copy to tracked location
cp results/lesion_preprocessing/lesion_size_analysis.csv data/lesion_size_analysis.csv

# Commit update
git add data/lesion_size_analysis.csv
git commit -m "Update lesion size analysis"
```

---

## Output Example

When filtering is active, you'll see:

```
======================================================================
LESION SIZE FILTERING
======================================================================
Filtering criteria:
  • Preprocessed voxels >= 500
  • Minimum dimension >= 5
  • Lesion density >= 0.30

Results:
  • Total cases: 930
  • Viable cases: 317 (34.1%)
  • Filtered out: 613

Per-dataset breakdown:
  • gist: 102/246 (41.5%)
  • lipo: 81/115 (70.4%)
======================================================================
```

---

## Implementation Details

### How It Works

1. **Load CSV**: Reads `data/lesion_size_analysis.csv`
2. **Apply Filters**: Identifies cases meeting all criteria
3. **Filter Data**: Removes filtered cases before train/val split
4. **Save Results**: Outputs to descriptive folder names

### Auto-Detection Paths

The filter searches for the analysis file in:
1. `data/lesion_size_analysis.csv` (tracked, preferred)
2. `results/lesion_preprocessing/lesion_size_analysis.csv`
3. `results/lesion_size_analysis.csv`

### CSV Columns

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name |
| `case` | Case ID |
| `raw_voxels` | Original voxel count |
| `preproc_voxels` | Voxels after preprocessing |
| `bbox_x`, `bbox_y`, `bbox_z` | Bounding box dimensions |
| `min_dimension` | Smallest bbox dimension |
| `density` | preproc_voxels / bbox_volume |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "CSV not found" | Run `python scripts/analysis/analyze_lesion_sizes.py` |
| "No cases remaining" | Lower thresholds (try `min_voxels=200`) |
| Want to disable | Don't pass any filter parameters |

---

## Related Documentation

- [Quick Reference](QUICK_REFERENCE.md) — Command cheat sheet
- [Preprocessing](technical/preprocessing.md) — Preprocessing pipeline
- [Cluster Scripts](../cluster_scripts/README.md) — HPC experiment scripts

## What You'll See

```
======================================================================
LESION SIZE FILTERING
======================================================================
Filtering criteria:
  • Preprocessed voxels >= 500
  • Minimum dimension >= 5
  • Lesion density >= 0.30

Results:
  • Total cases: 930
  • Viable cases: 317 (34.1%)
  • Filtered out: 613

Per-dataset breakdown:
  • gist: 102/246 (41.5%)
  • lipo: 81/115 (70.4%)
======================================================================
```

## Python API Reference

### `LesionSizeFilter`

```python
from med3pipe import LesionSizeFilter

filter_obj = LesionSizeFilter(
    min_voxels=500,          # Minimum voxel count
    min_dimension=5,         # Minimum bbox dimension
    min_density=0.3,         # Minimum density (0-1)
    lesion_csv_path=None,    # Auto-detected if None
)

# Check if filtering is enabled
if filter_obj.is_enabled():
    print("Filtering active")

# Get valid cases
valid_cases = filter_obj.get_valid_cases()

# Print statistics
filter_obj.print_filter_summary()
```

### Pipeline Functions

All these functions accept filtering parameters:

```python
from med3pipe import run_single_dataset, run_from_prepared
from med3pipe import run_classification_head_experiment

# Method 1: Pass filter object
result = run_single_dataset(..., lesion_filter=filter_obj)

# Method 2: Pass parameters directly
result = run_single_dataset(
    ...,
    min_voxels=500,
    min_dimension=5,
    min_density=0.3,
)

# Classification head training
result = run_classification_head_experiment(
    ...,
    lesion_filter=filter_obj,
)
```

## Cluster Scripts Reference

### Command Line Arguments

```bash
--min-voxels INT        # Minimum voxel count (e.g., 500)
--min-dimension INT     # Minimum bbox dimension (e.g., 5)
--min-density FLOAT     # Minimum density (e.g., 0.3)
--filter-preset PRESET  # Use preset: recommended, conservative, lenient
```

### Examples

```bash
# Use preset
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets.yaml \
    --filter-preset recommended

# Custom thresholds
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets.yaml \
    --min-voxels 1000 \
    --min-dimension 10

# Specific datasets
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets.yaml \
    --datasets gist lipo \
    --filter-preset recommended
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "CSV not found" | Run `python scripts/analyze_lesion_sizes.py` |
| "No cases remaining" | Lower thresholds (try min_voxels=200) |
| Want to disable | Don't pass any filter parameters |

## Implementation Details

### How It Works

1. **Load CSV**: Reads `results/lesion_preprocessing/lesion_size_analysis.csv`
2. **Apply Filters**: Identifies cases meeting all criteria
3. **Filter Data**: Removes filtered cases before train/val split
4. **Save Results**: Outputs to separate folder with descriptive name

### Modified Files

- `med3pipe/tabular/lesion_filter.py` - Core filtering logic
- `med3pipe/tabular/stratify.py` - Applies filtering during split
- `med3pipe/pipelines/end_to_end.py` - Pipeline integration
- `med3pipe/training/classification_head.py` - Classification head support
- `cluster_scripts/run_experiment1_benchmarks.py` - Experiment 1 (benchmarks) support
- `cluster_scripts/run_experiment3_classification_head.py` - Experiment 3 (classification head) support

### Key Points

- Filtering is **opt-in** (disabled by default)
- Results saved in **separate folders** (no overwrites)
- **Backward compatible** (existing code works unchanged)
- Works with **all methods** (TabPFN, LoCalPFN, classification head, 3D baselines)
- **All experiments** support filtering (Experiment 1 benchmarks, Experiment 3 classification head)
- **Consistent filtering**: When enabled, ALL methods in an experiment use the SAME filtered dataset for fair comparison

---

## Cluster Scripts Usage

### Quick Start on HPC

**With filtering:**
```bash
sbatch cluster_scripts/slurm_experiment3_filtered.sh
```

**Without filtering:**
```bash
sbatch cluster_scripts/slurm_experiment3.sh
```

**Compare both:**
```bash
sbatch cluster_scripts/slurm_experiment3.sh          # Baseline
sbatch cluster_scripts/slurm_experiment3_filtered.sh # Filtered
```

### Cluster Command Examples

**Experiment 1 (Benchmarks) with filtering:**
```bash
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets_cluster.yaml \
    --filter-preset recommended
```

**Experiment 3 (Classification Head) with filtering:**
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --filter-preset recommended
```

**Custom thresholds:**
```bash
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets_cluster.yaml \
    --min-voxels 500 \
    --min-dimension 5 \
    --min-density 0.3
```

**Specific datasets:**
```bash
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist lipo \
    --filter-preset recommended
```

---

**More examples**: `examples/example_lesion_filtering.py`
