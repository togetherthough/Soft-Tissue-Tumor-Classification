# Documentation Index

This repository focuses on Med3Pipe (SAM-Med3D features + TabPFN/LoCalPFN). Start here and follow links.

- **Med3Pipe (Transfer Learning)**
  - API: `med3pipe/README.md`
  - Multi-dataset how-to: `docs/MULTI_DATASET.md`
  - SAM feature quality (+ TL;DR quick test): `docs/SAM_FEATURE_EVALUATION.md`
  - Experiments overview: `notebooks/README_Experiments.md`
- **Troubleshooting**
  - `docs/TROUBLESHOOTING.md` (CPU checkpoints, OpenMP, workers)

---

## Quick Start

- **Environment**
  - Create a fresh env. Minimal packages: `pip install -r med3pipe/requirements.txt`
- **Datasets**
  - Place datasets under `project_root/<dataset>/` or `project_root/data/<dataset>/`.
  - Verify each has a `sheet.csv` with labels and IDs.
- **Run PFNs on multiple datasets**
  - Notebook: `notebooks/MultiDataset-PFNs-sequential.ipynb`
  - CLI: see root `README.md` under Quick start (multi-dataset)
- **Evaluate SAM features**
  - `docs/SAM_FEATURE_EVALUATION.md`
<!-- Optional: cascaded DL pipelines docs removed for minimal set -->

---

## Fine-tuning SAM-Med3D (summary)

- Use `med3pipe.training.finetune.finetune_sam3d()` to spawn SAM-Med3D's `train.py` with your prepared dataset.
- Recommended on CPU-only:
  - Convert or load a CPU checkpoint (map GPU checkpoints with `map_location='cpu'`).
  - Reduce workers: `--num_workers 4..8`.
  - If you see `OpenMP` duplicate runtime error, set `KMP_DUPLICATE_LIB_OK=TRUE`.
- After fine-tuning, point PFN experiments to the fine-tuned checkpoint.

See `docs/TROUBLESHOOTING.md` for concrete fixes.

---

## Notebooks & Experiments

- Catalog: `notebooks/README_Experiments.md`
- Feature extraction and PFN baselines: `notebooks/Med3D-TabPFN.ipynb`, `notebooks/Med3D-LoCalPFN.ipynb`
- SAM feature quick test: `notebooks/Test-SAM-Features.ipynb`

---

## Project Structure (high level)

```
├── hieracascade/          # BinaryCascade & HieraCascade pipelines
├── med3pipe/              # Med3Pipe (transfer learning APIs/CLI)
├── SAM-Med3D-main/        # Upstream SAM-Med3D repo (training/extraction)
├── configs/               # Datasets config (YAML)
├── docs/                  # This folder (index + guides)
├── notebooks/             # Experiments & analyses
├── scripts/               # Utility CLI/scripts
└── data/                  # Your datasets (outside git)
```
