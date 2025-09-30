# Multi-dataset pipeline (YAML-driven)

This repo supports running the full SAM-Med3D ➜ TabPFN/LoCalPFN pipeline across arbitrarily many datasets using either a YAML configuration or automatic folder discovery.

- Primary config: `configs/datasets.yaml`
- Notebook (YAML-driven): `notebooks/MultiDataset-PFNs.ipynb`
- Notebook (folder discovery): `notebooks/MultiDataset-FromDataFolder.ipynb`

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

You now have four interchangeable ways to run the multi-dataset pipeline:

1) Notebook (YAML): open and execute `notebooks/MultiDataset-PFNs.ipynb` (or the clean variant). The notebook:
- Loads `configs/datasets.yaml`
- Iterates all datasets listed under `datasets:`
- For each dataset, runs two methods end-to-end:
  - Method 1: Med3D embeddings ➜ TabPFN (`med3pipe.pipelines.end_to_end.run_end_to_end`)
  - Method 2: Med3D embeddings ➜ LoCalPFN (`med3pipe.pipelines.end_to_end.local_end_to_end`)
- Uses a feature-level STRATIFIED train/validation split from the union of per-case features for evaluation.
- Aggregates metrics (accuracy, macro-F1, AUROC, confusion matrix) into a table and saves it at `notebooks/multi_results_summary.csv`.
- Writes full per-run artifacts (preprocessing objects, predictions, metrics, classification report) into `notebooks/tabpfn_runs/...` directories.

2) Notebook (folder discovery): open and execute `notebooks/MultiDataset-FromDataFolder.ipynb`. It discovers datasets under `data/` and runs the same two methods per dataset without needing a YAML file.

3) Programmatic API (YAML): call the method-specific entrypoints from Python.

```python
from med3pipe.pipelines import run_multi_tabpfn, run_multi_localpfn

# TabPFN only
res = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    outputs_base_dir="notebooks",
)
print(res["summary_df"].to_string())

# LoCalPFN only (with ablations)
res = run_multi_localpfn(
    config_path="configs/datasets.yaml",
    outputs_base_dir="notebooks",
    local_k=50,
    local_fit_adapter=True,
    local_adapter_epochs=8,
)
print(res["summary_df"].to_string())
```

4) Programmatic API (folder discovery): call method-specific folder runners from Python.

```python
from med3pipe.pipelines import run_multi_tabpfn_from_folder, run_multi_localpfn_from_folder

# TabPFN only
res = run_multi_tabpfn_from_folder(
    datasets_dir="data",
    outputs_base_dir="notebooks",
)
print(res["summary_df"].to_string())

# LoCalPFN only (with ablations)
res = run_multi_localpfn_from_folder(
    datasets_dir="data",
    outputs_base_dir="notebooks",
    local_k=50,
)
print(res["summary_df"].to_string())
```

Single‑method convenience wrappers are also available:

- TabPFN only (YAML): `run_multi_tabpfn(config_path, ...)`
- LoCalPFN only (YAML): `run_multi_localpfn(config_path, ...)`
- TabPFN only (folder discovery): `run_multi_tabpfn_from_folder(datasets_dir, ...)`
- LoCalPFN only (folder discovery): `run_multi_localpfn_from_folder(datasets_dir, ...)`

5) CLI: use the packaged command to run from the terminal.

```bash
# TabPFN only
python -m med3pipe multi --config configs/datasets.yaml \
  --method tabpfn \
  --datasets gist,lipo \   # optional filter
  --outputs-base notebooks  # optional base directory for run folders

# LoCalPFN only (with ablations)
python -m med3pipe multi --config configs/datasets.yaml \
  --method localpfn \
  --outputs-base notebooks \
  --local-k 50 --local-fit-adapter --local-adapter-epochs 8
```

CLI flags include shared overrides like `--sam3d-root`, `--model-type`, `--checkpoint`,
`--device`, `--n-components-max`, `--random-state`, and LoCalPFN-specific ones such as
`--local-k`, `--local-metric`, `--local-fit-adapter`, `--local-adapter-epochs`,
`--local-adapter-lr`, `--local-adapter-weight-decay`, `--local-adapter-num-queries`.

Notes:
- Folder-level validation directories (`imagesVal/labelsVal`) are created to enable cached extraction and consistent directory structure. Evaluation uses a feature-level STRATIFIED split from the union of features (train_ratio) to ensure label balance.
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

## Label handling (binarization quick fix)

Some datasets encode targets with three values (e.g., `-1`, `0`, `1`) where `-1` may mean "unknown" or a second negative category. When pooling multiple datasets, this can make evaluation (especially ROC AUC) ambiguous per dataset.

To provide a consistent binary target across datasets by default, the label loader `med3pipe.sam.core.load_labels_from_sheet(...)` now applies a small, pragmatic fix:

- Maps `-1 -> 0`, then clamps labels to `{0, 1}`.
- Controlled by the parameter `binarize_neg1_to0` (default: `True`).

This is intentionally a quick fix to make pooled TabPFN experiments easy to run and to ensure per‑dataset ROC AUC is well‑defined. If you require strict multi‑class behavior, disable this by passing `binarize_neg1_to0=False` wherever you call `load_labels_from_sheet` (e.g., in your notebook):

```python
df, lab_map = load_labels_from_sheet(
    sheet_csv=sheet_csv,
    dataset_col=ds_col,
    dataset_name=ds_name,
    subject_col=labs.get('subject_col','Subject'),
    label_col=labs.get('label_col','Diagnosis_binary'),
    case_suffix=labs.get('case_suffix','_CT'),
    binarize_neg1_to0=False,   # disable the quick-fix binarization
)
```

Notes:
- Keeping `binarize_neg1_to0=True` is recommended when pooling heterogeneous datasets to avoid label‑space mismatch.
- If you see unexpectedly low metrics after switching to multi‑class, consider training per‑dataset models or adding a dataset indicator feature to the pooled tabular inputs.

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
