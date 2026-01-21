# med3pipe

> Core Python package for Med3Tab-PFN: SAM-Med3D feature extraction and tabular classification

## Overview

`med3pipe` provides utilities for converting 3D medical imaging datasets into the format required by SAM-Med3D, extracting embeddings, and performing classification with TabPFN/LoCalPFN.

### Key Features

- **Dataset Preparation**: Convert NIfTI volumes to SAM-Med3D format
- **Validation Splitting**: Create stratified train/validation splits
- **Feature Extraction**: Extract SAM-Med3D image encoder embeddings
- **ROI Pooling**: Convert 3D feature maps to fixed-length vectors
- **Tabular Classification**: TabPFN and LoCalPFN wrappers
- **3D Baselines**: DenseNet121-3D and Swin Transformer-3D

## Installation

```bash
pip install -r med3pipe/requirements.txt
```

### Requirements
- Python 3.9+
- PyTorch 2.0+
- SimpleITK
- TorchIO
- NumPy, Pandas
- scikit-learn

## Quick Start

### Command Line Interface

```bash
# Show available commands
python -m med3pipe --help

# Prepare dataset for SAM-Med3D
python -m med3pipe prepare \
    --dataset-root data/gist \
    --category gist \
    --ct-name ct_GIST

# Create train/validation split
python -m med3pipe split \
    --category gist \
    --ct-name ct_GIST \
    --split-ratio 0.8

# Run TabPFN on multiple datasets
python -m med3pipe multi-tabpfn --config configs/datasets.yaml
```

### Python API

```python
from pathlib import Path
from med3pipe import (
    prepare_for_sam3d,
    split_validation,
    build_sam3d_model,
    extract_embeddings_train_val,
    load_pooled_features,
    load_labels_from_sheet,
    tabpfn_pipeline,
)

# Step 1-2: Prepare dataset
n_cases, paths = prepare_for_sam3d(
    dataset_root=Path("data/gist"),
    sam3d_root=Path("sam-med3d"),
    category="gist",
    ct_name="ct_GIST",
)

# Step 3: Create validation split
train_n, val_n = split_validation(paths, split_ratio=0.8, seed=2025)

# Step 4: Load SAM-Med3D model
model = build_sam3d_model(
    sam3d_root=Path("sam-med3d"),
    checkpoint=Path("sam-med3d/ckpt/sam_med3d_turbo.pth"),
)

# Step 5: Extract embeddings
extract_embeddings_train_val(paths, model, sam3d_root=Path("sam-med3d"))

# Step 6: Pool features
X_train, ids_train = load_pooled_features(feat_train_dir, paths.labels_tr)
X_val, ids_val = load_pooled_features(feat_val_dir, paths.labels_val)

# Step 7: Load and align labels
df, label_map = load_labels_from_sheet(Path("data/gist/sheet.csv"))
y_train, _ = build_y(ids_train, label_map)
y_val, _ = build_y(ids_val, label_map)

# Step 8: Train and evaluate TabPFN
result = tabpfn_pipeline(
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    ids_val=ids_val,
    category="gist", ct_name="ct_GIST",
)
print(f"Accuracy: {result['metrics']['accuracy']:.3f}")
```

## Module Structure

```
med3pipe/
├── __init__.py           # Package exports
├── __main__.py           # CLI entry point
├── cli.py                # Command-line interface
├── data/                 # Data loading utilities
├── pipelines/            # End-to-end pipelines
│   ├── end_to_end.py     # Single dataset pipelines
│   └── multi_dataset.py  # Multi-dataset workflows
├── sam/                  # SAM-Med3D integration
│   ├── core.py           # Model building and extraction
│   └── transforms.py     # Preprocessing transforms
├── tabular/              # Tabular classification
│   ├── pfn.py            # TabPFN wrapper
│   ├── local_pfn.py      # LoCalPFN wrapper
│   ├── lesion_filter.py  # Quality filtering
│   └── stratify.py       # Stratified splitting
├── training/             # Training utilities
│   └── classification_head.py
└── vision/               # 3D baseline models
    └── v3d.py            # DenseNet, Swin Transformer
```

## CLI usage

Run from your project root, e.g., `C:/Users/<you>/Desktop/Med3Tab-PFN`:

```bash
# Show help
python -m med3pipe --help

# Step 1–2: Prepare imagesTr/labelsTr (GIST defaults)
python -m med3pipe prepare \
  --dataset-root gist \
  --category gist \
  --ct-name ct_GIST

# Step 3: Create validation split (copy, non destructive)
python -m med3pipe split \
  --category gist \
  --ct-name ct_GIST \
  --split-ratio 0.8 \
  --seed 2025

# Steps 1–3 together (prepare then split)
python -m med3pipe prepare-split \
  --dataset-root gist \
  --category gist \
  --ct-name ct_GIST \
  --split-ratio 0.8 \
  --seed 2025
```

Examples for a dataset folder named `name/` with a custom pattern:

```bash
# Suppose cases live under: name/<CASE_ID>/1/NIFTI
python -m med3pipe prepare \
  --dataset-root name \
  --case-glob "*/1/NIFTI" \
  --category myds \
  --ct-name ct_MYDS

python -m med3pipe split \
  --category myds \
  --ct-name ct_MYDS \
  --split-ratio 0.75
```

By default, the CLI auto-detects the SAM-Med3D repo root as `./sam-med3d`.
You can override with `--sam3d-root <path>`.

## Python API

```python
from pathlib import Path
from med3pipe import (
    prepare_for_sam3d, split_validation, Sam3DPaths,
    build_sam3d_model, extract_embeddings_train_val, default_feature_dirs,
    load_pooled_features, load_labels_from_sheet, build_y,
    find_default_sam3d_root,
)

# Prepare
n, paths = prepare_for_sam3d(
    dataset_root=Path("gist"),
    sam3d_root=Path("sam-med3d"),
    category="gist",
    ct_name="ct_GIST",
)

# Split (copy validation subset)
train_n, val_n = split_validation(paths, split_ratio=0.8, seed=2025, copy=True)

# Build SAM-Med3D model (optionally load a checkpoint)
SAM3D_ROOT = find_default_sam3d_root()
ckpt = SAM3D_ROOT / "ckpt" / "sam_med3d_turbo.pth"  # optional
model = build_sam3d_model(sam3d_root=SAM3D_ROOT, checkpoint=ckpt, model_type="vit_b_ori")

# Extract embeddings for TRAIN and VAL
feat_dirs = default_feature_dirs(SAM3D_ROOT, category=paths.category, ct_name=paths.ct_name)
extract_embeddings_train_val(paths, model, sam3d_root=SAM3D_ROOT)

# Pool to per-case vectors using Global Average Pooling
X_train, ids_train = load_pooled_features(feat_dirs.train_dir, paths.labels_tr)
X_val,   ids_val   = load_pooled_features(feat_dirs.val_dir,   paths.labels_val)

# Load labels and align to case IDs
df_gist, lab_map = load_labels_from_sheet(Path("gist") / "sheet.csv")
y_train, missing_tr = build_y(ids_train, lab_map)
y_val,   missing_va = build_y(ids_val,   lab_map)
print("Missing labels train/val:", len(missing_tr), len(missing_va))
```

### Steps 7–8: Standardize + PCA, then TabPFN (saved by default)

You can run preprocessing and TabPFN training/evaluation in one call with default saving:

```python
from med3pipe import tabpfn_pipeline

res = tabpfn_pipeline(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    ids_val=ids_val,
    category=paths.category,
    ct_name=paths.ct_name,
)

print("Saved outputs under:", res["out_dir"])            # e.g., tabpfn_runs/gist_ct_GIST_<timestamp>
print("Preproc dir:", res["preproc_dir"])                # scaler/pca and transformed arrays
print("Predictions:", res["pred_path"])                 # tabpfn_val_predictions.csv
print("Metrics:", res["metrics_path"])                  # tabpfn_metrics.json
print("Report:", res["report_path"])                    # classification_report.txt
```

By default, artifacts are saved to:

```
tabpfn_runs/<category>_<ct_name>_<timestamp>/
├── preproc/
│   ├── scaler.joblib
│   ├── pca.joblib
│   ├── X_train_p.npy
│   └── X_val_p.npy
├── tabpfn_config.json
├── tabpfn_val_predictions.csv
├── tabpfn_metrics.json
└── tabpfn_classification_report.txt
```

If you prefer to run steps separately, you can use:

```python
from med3pipe import standardize_pca, train_eval_tabpfn, default_tabpfn_out_dir
from pathlib import Path

out_dir = default_tabpfn_out_dir(paths.category, paths.ct_name)
X_train_p, X_val_p, scaler, pca = standardize_pca(
    X_train, X_val, n_components_max=500, random_state=42, save_dir=out_dir / "preproc"
)
res = train_eval_tabpfn(
    X_train=X_train_p,
    y_train=y_train,
    X_val=X_val_p,
    y_val=y_val,
    ids_val=ids_val,
    out_dir=out_dir,
)
```

## Output layout

```
data/
└── train/
    └── <category>/<ct_name>/
        ├── imagesTr/
        │   ├── <CASE_ID>.nii.gz
        │   └── ...
        └── labelsTr/
            ├── <CASE_ID>.nii.gz   # binary mask
            └── ...
validation/
    └── <category>/<ct_name>/
        ├── imagesVal/
        │   ├── <CASE_ID>.nii.gz
        │   └── ...
        └── labelsVal/
            ├── <CASE_ID>.nii.gz
            └── ...
```
```

## Design notes

- Order of operations follows the notebook: prepare to `imagesTr/labelsTr` first, then create
  `imagesVal/labelsVal` by copying the validation subset. This ensures that later steps (feature
  extraction) can access both TRAIN and VAL folders simultaneously.
- Masks are binarized (`>0`) to match SAM-Med3D training expectations.
- Geometry alignment: images are resampled to the label geometry if size/spacing/origin/direction
  do not match.
- Multiple lesion support: merges `image_lesion_*.nii.gz` and `segmentation_lesion_*.nii.gz`.

### Fine-tuning helper (optional)

You can launch SAM-Med3D fine-tuning from Python with a thin wrapper that patches
`utils/data_paths.py` and spawns `train.py` as a subprocess:

```python
from med3pipe import finetune_sam3d

proc = finetune_sam3d(
    paths,
    model_type="vit_b_ori",
    checkpoint=ckpt,            # optional
    device="cuda",             # or "cpu"
    num_epochs=2,               # small smoke test
    batch_size=4,
)
proc.wait()  # wait for training to finish
```

## Troubleshooting

- If discovery finds 0 cases, pass a custom `--case-glob`, e.g. `"*/1/NIFTI"` or use the full
  GIST pattern `"GIST-*_CT/1/NIFTI"`.
- If `SimpleITK` is missing, install with `pip install SimpleITK`.
- If the SAM-Med3D repo lives in a different location, pass `--sam3d-root <path>`.
