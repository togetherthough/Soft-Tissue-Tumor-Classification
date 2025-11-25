# Lesion Size Filtering

Complete guide to filtering training/testing data by lesion size.

## Quick Answer: What is "pct"?

**"pct" = percentage (0-100%)** - You cannot filter by "500 pct"

Use these instead:
- `min_voxels=500` - Minimum voxel count
- `min_dimension=5` - Minimum dimension
- `min_density=0.3` - Minimum density (0-1)

## Quick Start

### Prerequisites (run once)
```bash
python scripts/analyze_lesion_sizes.py
```

### Python API
```python
from med3pipe import run_single_dataset

# Simple filtering
result = run_single_dataset(
    method="tabpfn",
    dataset_root="path/to/gist",
    category="gist",
    ct_name="ct_GIST",
    min_voxels=500,  # Filter to >= 500 voxels
)

# Recommended filtering
from med3pipe import LesionSizeFilter

filter_obj = LesionSizeFilter(
    min_voxels=500,
    min_dimension=5,
    min_density=0.3,
)

result = run_single_dataset(..., lesion_filter=filter_obj)
```

### Cluster Scripts
```bash
# With filtering (saves to separate _filtered folders)
sbatch cluster_scripts/slurm_experiment3_filtered.sh

# Without filtering
sbatch cluster_scripts/slurm_experiment3.sh

# Custom filtering
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets.yaml \
    --min-voxels 500 \
    --min-dimension 5
```

## Output Directories

Filtered results are saved in **separate folders**:

```
results/classification_head/
├── gist/                            ← Unfiltered (all data)
├── gist_filtered_v500_d5_ρ0.30/   ← Filtered (quality subset)
```

The folder name shows which filters were applied!

## Recommended Thresholds

| Use Case | min_voxels | min_dimension | min_density | Result |
|----------|------------|---------------|-------------|--------|
| **Start here** | 500 | 5 | 0.3 | ~34% of cases |
| **Strict** | 1000 | 10 | 0.3 | ~20% of cases |
| **Lenient** | 200 | 3 | - | ~60% of cases |
| **Voxels only** | 500 | - | - | ~50% of cases |

## Filter Presets

```bash
# Recommended
--filter-preset recommended  # voxels>=500, dim>=5, density>=0.3

# Conservative (strict)
--filter-preset conservative # voxels>=1000, dim>=10, density>=0.3

# Lenient (more inclusive)
--filter-preset lenient      # voxels>=200, dim>=3
```

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
