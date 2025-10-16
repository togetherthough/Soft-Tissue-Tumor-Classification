# Soft-Tissue-Tumor-Classification

**Master Thesis**: Classification of soft tissue tumors from 3D medical imaging  
**Data**: 930 studies (CT/MRI) from `sheet.csv`

## Three Approaches

### 1. BinaryCascade ⭐ (Deep Learning - Binary)
End-to-end deep learning for benign vs. malignant classification
```bash
python -m hieracascade.quick_start --data_root data --fold 0
```

### 2. HieraCascade (Deep Learning - Multi-class)
End-to-end deep learning for tumor type classification (CRLM, GIST, Desmoid, Lipo, Liver, Melanoma)
```bash
python -m hieracascade.quick_start_hierarchical --data_root data --fold 0
```

### 3. Med3Pipe (Transfer Learning)
SAM-Med3D feature extraction + TabPFN/LoCalPFN for binary classification (baseline)

**New**: Test SAM-Med3D feature quality before running full pipeline:
```bash
python scripts/test_sam_features.py --dataset gist --epochs 10 --freeze
```

## What's included
- `notebooks/Med3D-TabPFN.ipynb`: step-by-step single-dataset pipeline (GIST example)
- `notebooks/Med3D-LoCalPFN-fast.ipynb`: LoCalPFN experiment (single dataset)
- `notebooks/MultiDataset-PFNs-sequential.ipynb`: run multiple datasets and both methods (TabPFN then LoCalPFN) sequentially via YAML
- `notebooks/Test-SAM-Features.ipynb`: evaluate SAM-Med3D feature quality with classification head
- `configs/datasets.yaml`: register datasets and their metadata once
- `docs/MULTI_DATASET.md`: documentation for running multiple datasets and adding new ones
- `docs/SAM_FEATURE_EVALUATION.md`: guide for testing SAM-Med3D feature quality

## Quick start (multi-dataset)
1. Ensure your datasets are placed as either:
   - `project_root/<dataset_name>/` (e.g., `project_root/gist/`) or
   - `project_root/data/<dataset_name>/` (e.g., `project_root/data/lipo/`)
2. Verify each dataset has a `sheet.csv` (or update the YAML to point to the correct path). Default columns used:
   - `Subject` (case ID without suffix)
   - `Diagnosis_binary` (0/1 label)
3. Edit `configs/datasets.yaml` to ensure each dataset block is correct (category, ct_name, labels, etc.).
4. Choose one of the following ways to run:
   - Notebook: open and run `notebooks/MultiDataset-PFNs-clean.ipynb`.
   - CLI:
    ```bash
    # TabPFN across datasets (evaluation uses stratified feature-level split)
    python -m med3pipe multi-tabpfn --config configs/datasets.yaml \
      --outputs-base notebooks

    # LoCalPFN across datasets (stratified evaluation; optional ablations)
    python -m med3pipe multi-localpfn --config configs/datasets.yaml \
      --outputs-base notebooks \
      --local-k 50 --local-fit-adapter --local-adapter-epochs 8
    ```
   - Programmatic API:
     ```python
     from med3pipe.pipelines import run_multi_tabpfn
     res = run_multi_tabpfn("configs/datasets.yaml")
     print(res["summary_df"].to_string())
     ```
   - It will:
     - Prepare the dataset in SAM-Med3D format (creates a folder-level validation subset for caching)
     - Build/load SAM-Med3D, extract embeddings, ROI-pool to vectors
     - Perform a feature-level STRATIFIED train/validation split from the union of features
     - Train/evaluate TabPFN
     - Save artifacts under `notebooks/tabpfn_runs/`
     - Write a summary table to `notebooks/multi_results_summary.csv`

If you need more detail, see `docs/MULTI_DATASET.md`.

## Requirements
Install dependencies (prefer a fresh environment). Minimal set is under `med3pipe/requirements.txt`:

```bash
pip install -r med3pipe/requirements.txt
```

Note: TabPFN is imported dynamically. If you have the TabPFN source checked out under `TabPFN-main/TabPFN-main/src` at the project root, it will be auto-discovered. Otherwise, install `tabpfn` via pip or point the code to your local path.

## Datasets
- Example datasets registered in `configs/datasets.yaml`:
  - `gist` (ct_GIST)
  - `lipo` (ct_LIPO)
- The pipeline assumes binary classification by default (see `label_col`).

## Methods
- Method 1: Med3D embeddings → TabPFN (`med3pipe.pipelines.end_to_end.run_end_to_end`)
- Method 2: Med3D embeddings → LoCalPFN (`med3pipe.pipelines.end_to_end.local_end_to_end`)

Both methods share Steps 1–7 (prepare, folder split for caching, build SAM-Med3D, extract embeddings, ROI pooling, label alignment, feature-level STRATIFIED split, standardize+PCA) and differ in the final inference/training step.

## Notes
- GPU is used if available for SAM-Med3D and TabPFN; otherwise CPU is used.
- Fine-tuning of SAM-Med3D is available via `med3pipe.finetune_sam3d` but is disabled in the multi-dataset notebook for speed.
- Outputs per run are timestamped and organized by dataset/method.

---

## Documentation
Start here: `docs/README.md`

### Essential
- **`docs/MULTI_DATASET.md`** - Med3Pipe multi-dataset workflows
- **`docs/SAM_FEATURE_EVALUATION.md`** - SAM-Med3D feature quality evaluation
- **`med3pipe/README.md`** - Med3Pipe API reference
- **`docs/TROUBLESHOOTING.md`** - Troubleshooting fine-tuning & PFN workflows

### Additional
- `scripts/` - Utility scripts
  - `analyze_labels.py` - Dataset analysis
  - `run_hieracascade.sh` / `.bat` - Training scripts

### Project Structure
```
├── med3pipe/              # Med3Pipe (transfer learning)
├── notebooks/             # Jupyter notebooks
├── configs/               # Configuration files
├── docs/                  # Documentation (index + guides)
├── scripts/               # Utility scripts
└── data/                  # Data directory
    └── sheet.csv          # Labels (930 studies)
```
