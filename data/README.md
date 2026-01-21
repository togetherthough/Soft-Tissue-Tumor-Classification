# Data Directory

> Dataset storage and tracked analysis files

## Overview

This directory contains medical imaging datasets and tracked metadata files. The datasets themselves are gitignored to prevent accidentally committing large files.

## Directory Structure

```
data/
├── lesion_size_analysis.csv    # Tracked: lesion metrics for filtering
├── sheet.csv                   # Tracked: labels and metadata
├── README.md                   # This file
├── gist/                       # Dataset (gitignored)
├── lipo/                       # Dataset (gitignored)
├── crlm/                       # Dataset (gitignored)
├── melanoma/                   # Dataset (gitignored)
└── ...
```

## Tracked Files

### `lesion_size_analysis.csv`

**Purpose**: Contains preprocessing metrics for all cases, required for lesion size filtering.

**Columns**:
| Column | Description |
|--------|-------------|
| `dataset` | Dataset name (e.g., gist, lipo) |
| `case` | Case ID (e.g., GIST-001_CT) |
| `raw_voxels` | Original voxel count |
| `preproc_voxels` | Voxels after preprocessing |
| `bbox_x`, `bbox_y`, `bbox_z` | Bounding box dimensions |
| `min_dimension` | Smallest bbox dimension |
| `density` | Lesion density (preproc_voxels / bbox_volume) |

**Regenerating** (when adding new datasets):
```bash
python scripts/analysis/analyze_lesion_sizes.py --config configs/datasets.yaml
cp results/lesion_preprocessing/lesion_size_analysis.csv data/lesion_size_analysis.csv
git add data/lesion_size_analysis.csv
```

### `sheet.csv`

**Purpose**: Master labels file with patient/case metadata and classification labels.

**Expected columns**:
| Column | Description |
|--------|-------------|
| `Subject` | Case identifier |
| `Diagnosis_binary` | Binary classification label (0/1) |
| `Dataset` | Dataset name (optional) |

## Dataset Structure

Each dataset should follow this structure:

```
data/<dataset_name>/
├── CASE-001_CT/
│   └── 1/NIFTI/
│       ├── image.nii.gz           # CT/MRI volume
│       └── segmentation.nii.gz    # Lesion mask
├── CASE-002_CT/
│   └── 1/NIFTI/
│       ├── image.nii.gz
│       └── segmentation.nii.gz
└── sheet.csv                       # Dataset-specific labels (optional)
```

## Adding a New Dataset

1. **Place data** in `data/<dataset_name>/`
2. **Verify structure** matches expected format above
3. **Add configuration** to `configs/datasets.yaml`:
   ```yaml
   new_dataset:
     dataset_root: data/new_dataset
     category: new_dataset
     ct_name: ct_NEWDATA
     labels:
       sheet_csv: sheet.csv
       subject_col: Subject
       label_col: Diagnosis_binary
   ```
4. **Regenerate lesion analysis**:
   ```bash
   python scripts/analysis/analyze_lesion_sizes.py --config configs/datasets.yaml
   cp results/lesion_preprocessing/lesion_size_analysis.csv data/lesion_size_analysis.csv
   ```
5. **Commit tracked files**:
   ```bash
   git add data/lesion_size_analysis.csv
   git commit -m "Add new_dataset lesion metrics"
   ```

## Gitignore Rules

The `.gitignore` is configured to:
- ✅ Track `lesion_size_analysis.csv`
- ✅ Track `sheet.csv`
- ❌ Ignore dataset directories (large NIfTI files)
- ❌ Ignore intermediate outputs

## Related Documentation

- [Lesion Filtering](../docs/FILTERING.md) — Quality-based sample filtering
- [Configuration](../configs/README.md) — Dataset YAML configuration
- [Preprocessing](../docs/technical/preprocessing.md) — Data preprocessing pipeline
