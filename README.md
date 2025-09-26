# Soft-Tissue-Tumor-Classification
This is my master thesis work. It centers around the classification of multiple types of STS from 3D scans. The proposed technique uses SAM-Med3D as an encoder of tabular features that are then used for the classification task by TabPFN. In the study two subsequent improvements of the model are proposed.

## What's included
- `notebooks/Med3D-TabPFN.ipynb`: step-by-step single-dataset pipeline (GIST example)
- `notebooks/Med3D-LoCalPFN-fast.ipynb`: LoCalPFN experiment (single dataset)
- `notebooks/MultiDataset-PFNs-clean.ipynb`: run multiple datasets and both methods (TabPFN and LoCalPFN) end-to-end, YAML-driven
- `configs/datasets.yaml`: register datasets and their metadata once
- `docs/MULTI_DATASET.md`: documentation for running multiple datasets and adding new ones

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
     python -m med3pipe multi --config configs/datasets.yaml \
       --methods tabpfn,localpfn \
       --outputs-base notebooks
     ```
   - Programmatic API:
     ```python
     from med3pipe.pipelines import run_multi_dataset
     res = run_multi_dataset("configs/datasets.yaml", methods=("tabpfn","localpfn"))
     print(res["summary_df"].to_string())
     ```
   - It will:
     - Prepare the dataset in SAM-Med3D format (train/val split)
     - Build/load SAM-Med3D, extract embeddings, ROI-pool to vectors
     - Train/evaluate TabPFN and LoCalPFN
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

Both methods share Steps 1–7 (prepare, split, build SAM-Med3D, extract embeddings, ROI pooling, label alignment, standardize+PCA) and differ in the final inference/training step.

## Notes
- GPU is used if available for SAM-Med3D and TabPFN; otherwise CPU is used.
- Fine-tuning of SAM-Med3D is available via `med3pipe.finetune_sam3d` but is disabled in the multi-dataset notebook for speed.
- Outputs per run are timestamped and organized by dataset/method.
