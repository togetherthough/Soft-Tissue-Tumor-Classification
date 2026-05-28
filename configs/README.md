# Dataset Configuration Files

This directory contains dataset configuration files for different execution environments.

## 📁 Configuration Files

### `datasets.yaml` - Local Machine Configuration
**Use when**: Running experiments on your local machine or laptop

**Data location**: Relative paths within the repository
- `data/gist/`
- `data/lipo/`
- `sheet.csv` files can be at repo root

**Example structure**:
```
Med3Tab-PFN/
├── data/
│   ├── gist/
│   │   └── *.nii.gz files
│   └── lipo/
│       └── *.nii.gz files
├── sheet.csv
└── configs/
    └── datasets.yaml  ← Uses relative paths
```

### `datasets_cluster.yaml` - GPU Cluster Configuration
**Use when**: Running experiments on the GPU cluster via SLURM

**Data location**: Absolute paths to cluster storage
- `/data/scratch/r112276/gist/`
- `/data/scratch/r112276/lipo/`
- `sheet.csv` files are within each dataset directory

**Example structure**:
```
# Cluster filesystem layout:
/trinity/home/r112276/Med3Tab-PFN/     # Code repository
/data/scratch/r112276/
├── gist/
│   ├── sheet.csv
│   └── *.nii.gz files
└── lipo/
    ├── sheet.csv
    └── *.nii.gz files
```

## 🔄 Switching Between Modes

### Automatic (Recommended)
The SLURM script automatically uses the correct config:
- `cluster_scripts/slurm_train_and_test.sh` → uses `datasets_cluster.yaml`
- Notebooks and local scripts → use `datasets.yaml`

### Manual Override
You can specify which config to use via command line:

```bash
# Use cluster config
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets_cluster.yaml

# Use local config
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets.yaml
```

## 📝 Configuration Format

Both files use the same YAML structure:

```yaml
datasets:
  dataset_name:
    dataset_root: /path/to/dataset      # Local: relative, Cluster: absolute
    category: dataset_category           # Used for organizing outputs
    ct_name: ct_DATASET_NAME            # SAM-Med3D folder name
    labels:
      sheet_csv: sheet.csv               # Label file (relative to dataset_root)
      dataset_name: DATASET               # Optional: filter in sheet CSV
      subject_col: Subject                # Column name for subject IDs
      label_col: Diagnosis_binary         # Column name for labels
      case_suffix: _CT                    # Suffix to match in sheet (e.g., "001_CT")
    prepare:
      case_glob: null                     # Optional: glob pattern for cases
      max_cases: null                     # Optional: limit number of cases
    split:
      ratio: 0.8                          # Train/val split ratio
      seed: 2025                          # Random seed for reproducibility
    extraction:
      img_size: 128                       # Image size for SAM-Med3D
```

## 🎯 When to Edit Each File

### Edit `datasets.yaml` when:
- ✅ Adding a new dataset for local development
- ✅ Changing split ratios or seeds
- ✅ Modifying preprocessing parameters
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
