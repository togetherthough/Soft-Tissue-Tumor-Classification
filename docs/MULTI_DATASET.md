# Multi-dataset pipeline (YAML-driven)

This repo supports running the full SAM-Med3D ➜ TabPFN/LoCalPFN pipeline across arbitrarily many datasets using a single YAML configuration.

- Primary config: `configs/datasets.yaml`
- Notebook to run multiple datasets/methods: `notebooks/MultiDataset-PFNs.ipynb`

## Dataset registration (configs/datasets.yaml)

Each dataset is registered under the `datasets:` key. Example:

```yaml
datasets:
  gist:
    dataset_root: data/gist            # can be relative or absolute; see path resolution below
    category: gist                     # used to build SAM-Med3D paths (data/train/<category>/<ct_name>)
    ct_name: ct_GIST                   # nnU-Net task name used in SAM-Med3D layout
    labels:
      sheet_csv: sheet.csv             # path relative to dataset_root
      dataset_name: GIST
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

  lipo:
    dataset_root: data/lipo
    category: lipo
    ct_name: ct_LIPO
    labels:
      sheet_csv: sheet.csv
      dataset_name: LIPO
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

### Path resolution
The notebook resolves `dataset_root` with the following strategy:
1) If `dataset_root` is absolute and exists, use it.
2) Else try `<PROJECT_ROOT>/<dataset_root>`.
3) Else try `<PROJECT_ROOT>/<category>`.
4) Else try `<PROJECT_ROOT>/data/<category>`.

This allows flexibility if some datasets live at `project_root/gist/` and others under `project_root/data/<name>/`.

## Running multiple datasets
Open and execute `notebooks/MultiDataset-PFNs.ipynb`. The notebook:
- Loads `configs/datasets.yaml`
- Iterates all datasets listed under `datasets:`
- For each dataset, runs two methods end-to-end:
  - Method 1: Med3D embeddings ➜ TabPFN (`med3pipe.pipelines.end_to_end.run_end_to_end`)
  - Method 2: Med3D embeddings ➜ LoCalPFN (`med3pipe.pipelines.end_to_end.local_end_to_end`)
- Aggregates metrics (accuracy, macro-F1, AUROC, confusion matrix) into a table and saves it at `notebooks/multi_results_summary.csv`.
- Writes full per-run artifacts (preprocessing objects, predictions, metrics, classification report) into `notebooks/tabpfn_runs/...` directories.

Notes:
- The notebook attempts to use CUDA if available (fallback to CPU otherwise).
- Fine-tuning of SAM-Med3D is not included in the multi-dataset loop by default to keep runs quick. You can fine-tune separately via `med3pipe.finetune_sam3d` if desired.

## Adding a new dataset
1) Place the dataset folder under either `project_root/<name>/` or `project_root/data/<name>/`.
2) Ensure a `sheet.csv` exists (or update the YAML to point to a custom name/location). Typical columns:
   - `Subject` (case ID without suffix)
   - `Diagnosis_binary` (0/1 label)
   - The notebook/app assumes binary labels; extend as needed for multi-class.
3) Add a block to `configs/datasets.yaml` with the fields described above.
4) Run `notebooks/MultiDataset-PFNs.ipynb`.

## Outputs
- TabPFN runs: `notebooks/tabpfn_runs/<category>_<ct_name>_<timestamp>/`
- LoCalPFN runs: `notebooks/tabpfn_runs/local_<category>_<ct_name>_<timestamp>/`
- Summary table: `notebooks/multi_results_summary.csv`

Each run directory includes:
- Preprocessing artifacts (scaler, PCA) under `preproc/`
- Predictions CSV and metrics JSON
- A classification report (text)

## Troubleshooting
- If `sheet.csv` is not found, pass it explicitly in YAML under `labels.sheet_csv`, or verify the dataset root is resolved correctly.
- If extraction is slow on CPU, consider enabling GPU for SAM-Med3D and TabPFN (device auto-detection is used by default).
- For LoCalPFN, you can enable the lightweight adapter (`fit_adapter: true`) by editing `LocalPFNConfig` in the notebook cell where it is used.
