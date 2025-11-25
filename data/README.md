# Data Directory

## Tracked Files

### `lesion_size_analysis.csv`

**Purpose**: This file contains preprocessing metrics for all cases in the datasets and is required for lesion size filtering.

**Columns**:
- `dataset`: Dataset name (e.g., gist, lipo, crlm)
- `case`: Case ID (e.g., GIST-001_CT)
- `raw_voxels`: Original voxel count before preprocessing
- `preproc_voxels`: Voxel count after preprocessing (downsampled to 128³)
- `downsampling_ratio`: Factor by which data was downsampled
- `bbox_x`, `bbox_y`, `bbox_z`: Bounding box dimensions
- `bbox_volume`: Total bounding box volume
- `min_dimension`: Smallest of the 3 bbox dimensions
- `density`: Lesion density (preproc_voxels / bbox_volume)

**Why tracked?**
- Filtering depends on this file to determine which cases meet quality thresholds
- Without it, filtering features won't work
- Tracked so filtering works out of the box after cloning

**When to regenerate:**
1. When adding new datasets
2. When preprocessing parameters change (e.g., different target resolution)

**How to regenerate:**
```bash
# From repository root
python scripts/analyze_lesion_sizes.py --config configs/datasets.yaml

# Copy to tracked location
cp results/lesion_preprocessing/lesion_size_analysis.csv data/lesion_size_analysis.csv

# Commit the update
git add data/lesion_size_analysis.csv
git commit -m "Update lesion size analysis with new datasets"
```

**Auto-detection:**
The `LesionSizeFilter` class automatically searches for this file in:
1. `data/lesion_size_analysis.csv` (tracked, preferred)
2. `results/lesion_preprocessing/lesion_size_analysis.csv` (gitignored)
3. `results/lesion_size_analysis.csv` (gitignored)

## Other Data Files

The `/data/` directory itself is gitignored to prevent accidentally committing large datasets. Only specific files like `lesion_size_analysis.csv` are tracked via `.gitignore` exceptions.
