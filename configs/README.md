# Configuration Files

> Dataset configuration files for local and cluster environments

## Overview

This directory contains YAML configuration files that define dataset paths, preprocessing parameters, and experiment settings. Two configurations are provided:

| File | Environment | Paths |
|------|-------------|-------|
| `datasets.yaml` | Local machine | Relative paths |
| `datasets_cluster.yaml` | HPC cluster | Absolute paths |

## Configuration Format

```yaml
datasets:
  gist:                              # Dataset identifier
    dataset_root: data/gist          # Path to dataset directory
    category: gist                   # Category for output organization
    ct_name: ct_GIST                 # SAM-Med3D folder name
    labels:
      sheet_csv: sheet.csv           # CSV file with labels
      dataset_name: GIST             # Filter value (optional)
      subject_col: Subject           # Column for case IDs
      label_col: Diagnosis_binary    # Column for labels
      case_suffix: _CT               # Suffix in case IDs
    prepare:
      case_glob: null                # Glob pattern for discovery
      max_cases: null                # Limit cases (for testing)
    split:
      ratio: 0.8                     # Train/validation ratio
      seed: 2025                     # Random seed
    extraction:
      img_size: 128                  # Volume size for SAM-Med3D
```

## Usage

### Local Development

```bash
# Use datasets.yaml (default)
python -m med3pipe multi-tabpfn --config configs/datasets.yaml
```

### Cluster Execution

```bash
# Use cluster config with absolute paths
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets_cluster.yaml
```

### Programmatic Access

```python
import yaml

with open("configs/datasets.yaml") as f:
    config = yaml.safe_load(f)

for name, params in config["datasets"].items():
    print(f"Dataset: {name}, Root: {params['dataset_root']}")
```

## Adding a New Dataset

1. **Create dataset directory** with the expected structure:
   ```
   data/new_dataset/
   ├── CASE-001_CT/
   │   └── 1/NIFTI/
   │       ├── image.nii.gz
   │       └── segmentation.nii.gz
   └── sheet.csv
   ```

2. **Add to `datasets.yaml`**:
   ```yaml
   new_dataset:
     dataset_root: data/new_dataset
     category: new_dataset
     ct_name: ct_NEWDATA
     labels:
       sheet_csv: sheet.csv
       subject_col: Subject
       label_col: Diagnosis_binary
     split:
       ratio: 0.8
       seed: 2025
     extraction:
       img_size: 128
   ```

3. **Add to `datasets_cluster.yaml`** with absolute paths:
   ```yaml
   new_dataset:
     dataset_root: /data/scratch/user/new_dataset
     # ... same parameters ...
   ```

## Configuration Parameters

### Labels

| Parameter | Description | Example |
|-----------|-------------|---------|
| `sheet_csv` | Path to CSV file | `sheet.csv` |
| `dataset_name` | Filter column value | `GIST` |
| `subject_col` | Case ID column | `Subject` |
| `label_col` | Binary label column | `Diagnosis_binary` |
| `case_suffix` | Suffix to match | `_CT` |

### Preparation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `case_glob` | Discovery pattern | Auto-detect |
| `max_cases` | Limit for testing | None (all) |

### Split

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ratio` | Train fraction | 0.8 |
| `seed` | Random seed | 2025 |

### Extraction

| Parameter | Description | Default |
|-----------|-------------|---------|
| `img_size` | Volume dimension | 128 |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "dataset_root not found" | Check path exists and matches environment |
| "sheet.csv not found" | Verify `sheet_csv` path is correct |
| "No cases discovered" | Add custom `case_glob` pattern |
| Different results local/cluster | Ensure `seed` values match |

## Related Documentation

- [Main README](../README.md) — Project overview
- [Cluster Scripts](../cluster_scripts/README.md) — HPC experiment scripts
- [Quick Reference](../docs/QUICK_REFERENCE.md) — Command cheat sheet
- ✅ Testing locally before cluster submission

### Edit `datasets_cluster.yaml` when:
- ✅ Dataset locations on cluster have changed
- ✅ New datasets are added to `/data/scratch/`
- ✅ You need different parameters for cluster runs
- ✅ Your user ID or scratch space path changes

## 🔧 Adding a New Dataset

To add a new dataset, add an entry to both files:

**In `datasets.yaml` (local)**:
```yaml
  new_dataset:
    dataset_root: data/new_dataset      # Relative path
    category: new_dataset
    ct_name: ct_NEW_DATASET
    labels:
      sheet_csv: ../sheet.csv           # Or data/new_dataset/sheet.csv
      dataset_name: NewDataset
      subject_col: Subject
      label_col: Diagnosis_binary
      case_suffix: _CT
    prepare:
      case_glob: null
      max_cases: null
    split:
      ratio: 0.8
      seed: 2025
    extraction:
      img_size: 128
```

**In `datasets_cluster.yaml` (cluster)**:
```yaml
  new_dataset:
    dataset_root: /data/scratch/r112276/new_dataset  # Absolute path
    category: new_dataset
    ct_name: ct_NEW_DATASET
    labels:
      sheet_csv: sheet.csv              # Located at /data/scratch/r112276/new_dataset/sheet.csv
      dataset_name: NewDataset
      subject_col: Subject
      label_col: Diagnosis_binary
      case_suffix: _CT
    prepare:
      case_glob: null
      max_cases: null
    split:
      ratio: 0.8
      seed: 2025
    extraction:
      img_size: 128
```

## 💡 Tips

1. **Keep configs in sync**: Parameter changes (split ratios, seeds, etc.) should be identical in both files
2. **Test locally first**: Validate your config works with `datasets.yaml` before submitting to cluster
3. **Absolute vs Relative**: Cluster config uses absolute paths, local uses relative
4. **sheet.csv location**: 
   - Local: Can be at repo root with `../sheet.csv`
   - Cluster: Should be in each dataset directory with `sheet.csv`

## 🆘 Troubleshooting

### "FileNotFoundError: Could not resolve dataset_root"
- **Cause**: Wrong config file is being used for your environment
- **Solution**: Check which config is loaded, ensure paths match your filesystem

### "sheet.csv not found"
- **Cause**: `sheet_csv` path is incorrect
- **Solution**: 
  - Local: Use `../sheet.csv` if at repo root, or `sheet.csv` if in data folder
  - Cluster: Use `sheet.csv` (must be in dataset directory)

### Different results between local and cluster
- **Cause**: Configs may have different parameters (seeds, ratios)
- **Solution**: Verify `split`, `seed`, and `extraction` parameters match

---

**For more information**, see:
- Main README: `../README.md`
- Cluster guide: `../cluster_scripts/README.md`
- Quick reference: `../docs/QUICK_REFERENCE.md`
