# med3pipe: Prepare datasets for SAM-Med3D (steps 1–3)

Utilities and a CLI to convert raw 3D medical datasets (e.g., the GIST dataset) into the folder
layout expected by SAM-Med3D, and to create a train/validation split.

This library implements the first three steps from `notebooks/gist_tabpfn_end_to_end_corrected.ipynb`,
and provides methods for steps 4–6 as reusable functions:

1. Discover raw cases in a dataset directory (e.g., `gist/`).
2. Prepare nnU-Net-style folders under the SAM-Med3D repository:
   - `data/train/<category>/<ct_name>/{imagesTr, labelsTr}`
   - Labels are binary (foreground > 0).
   - Multiple lesion files per case are merged (images by max, masks by union).
   - Image geometry is aligned to mask geometry when mismatched.
3. Create a validation split by copying (or moving) a subset into
   `data/validation/<category>/<ct_name>/{imagesVal, labelsVal}`.
4. Build a SAM-Med3D model and extract image-encoder embeddings.
5. ROI-pool embeddings with lesion masks to get per-case feature vectors.
6. Load labels from CSV (e.g., `gist/sheet.csv`) and align to case IDs.

The implementation mirrors the notebook behavior and incorporates the following fixes:

- Multiple lesions per case are detected and merged: supports `image_lesion_*.nii.gz` and
  `segmentation_lesion_*.nii.gz` (and falls back to `segmentation_*.nii.gz`).
- Output filenames are `<case_id>.nii.gz` to ensure 1:1 correspondence between images and labels.

## Install

- Python 3.9+
- Dependencies (runtime):
  - `numpy`
  - `SimpleITK`
  - `torch`
  - `torchio`
  - `pandas` (for loading labels)

You can install them with pip:

```bash
pip install -r med3pipe/requirements.txt
```

## 3D Classification: DenseNet121 (3D) and !!!!Vision Transformer!!!!! (3D)

For native 3D volumetric classification, use the MONAI-based APIs in `med3pipe.vision.v3d`. These operate directly on NIfTI volumes prepared by steps 1–3 (SAM-Med3D layout), loading from:

- `data/train/<category>/<ct_name>/imagesTr/*.nii.gz`
- `data/validation/<category>/<ct_name>/imagesVal/*.nii.gz`

Labels are read from your dataset sheet via `load_labels_from_sheet`, ensuring case IDs match the prepared filenames (consistent with multiple-lesion merging and `.nii`-suffix handling).

### Train DenseNet121 (3D)

```python
from pathlib import Path
from med3pipe import Sam3DPaths, Train3DConfig, train_eval_densenet121_3d

SAM3D_ROOT = Path("SAM-Med3D-main/SAM-Med3D-main")
paths = Sam3DPaths(sam3d_root=SAM3D_ROOT, category="gist", ct_name="ct_GIST")

res = train_eval_densenet121_3d(
    paths=paths,
    sheet_csv=Path("gist") / "sheet.csv",
    num_classes=2,
    cfg=Train3DConfig(epochs=30, img_size=96, batch_size=2, device="cuda"),  # set device to "cuda" for GPU
)
print("Saved to:", res["out_dir"])  # baselines/densenet121_3d_<timestamp>
```

### Train Swin Transformer (3D)

```python
from med3pipe import train_eval_swin_transformer_3d, Train3DConfig, Sam3DPaths
from pathlib import Path

SAM3D_ROOT = Path("SAM-Med3D-main/SAM-Med3D-main")
paths = Sam3DPaths(sam3d_root=SAM3D_ROOT, category="gist", ct_name="ct_GIST")

res = train_eval_swin_transformer_3d(
    paths=paths,
    sheet_csv=Path("gist") / "sheet.csv",
    num_classes=2,
    cfg=Train3DConfig(epochs=30, img_size=96, batch_size=2, device="cuda"),
)
print("Saved to:", res["out_dir"])  # baselines/swin3d_<timestamp>
```

Both training and validation inference use the selected device (`cfg.device`), so GPU is used end-to-end when you set `device="cuda"`.

Note: The CLI defers importing heavy dependencies so `python -m med3pipe --help` works even if
`SimpleITK` is not installed yet.

## Folder assumptions

- SAM-Med3D repo is present at `./SAM-Med3D-main/SAM-Med3D-main` from your working directory.
  If your path differs, pass `--sam3d-root`.
- For GIST, each case is under `gist/GIST-XXX_CT/1/NIFTI/` with `image.nii.gz` and one or more
  `segmentation*.nii.gz` files.
- For other datasets with a different root name (e.g., `name/` instead of `gist/`), point the CLI
  to that directory with `--dataset-root` and, if needed, a custom discovery pattern via `--case-glob`.

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

By default, the CLI auto-detects the SAM-Med3D repo root as `./SAM-Med3D-main/SAM-Med3D-main`.
You can override with `--sam3d-root <path>`.

## Python API

```python
from pathlib import Path
from med3pipe import (
    prepare_for_sam3d, split_validation, Sam3DPaths,
    build_sam3d_model, extract_embeddings_train_val, default_feature_dirs,
    load_roi_features, load_labels_from_sheet, build_y,
    find_default_sam3d_root,
)

# Prepare
n, paths = prepare_for_sam3d(
    dataset_root=Path("gist"),
    sam3d_root=Path("SAM-Med3D-main/SAM-Med3D-main"),
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

# ROI-pool to per-case vectors
X_train, ids_train = load_roi_features(feat_dirs.train_dir, paths.labels_tr)
X_val,   ids_val   = load_roi_features(feat_dirs.val_dir,   paths.labels_val)

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
SAM-Med3D-main/SAM-Med3D-main/
└── data/
    ├── train/
    │   └── <category>/<ct_name>/
    │       ├── imagesTr/
    │       │   ├── <CASE_ID>.nii.gz
    │       │   └── ...
    │       └── labelsTr/
    │           ├── <CASE_ID>.nii.gz   # binary mask
    │           └── ...
    └── validation/
        └── <category>/<ct_name>/
            ├── imagesVal/
            │   ├── <CASE_ID>.nii.gz
            │   └── ...
            └── labelsVal/
                ├── <CASE_ID>.nii.gz
                └── ...
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
