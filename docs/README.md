# Documentation

All project documentation organized by topic.

## Structure

```
docs/
├── FILTERING.md         # Lesion size filtering (complete guide)
├── QUICK_REFERENCE.md   # Quick commands cheat sheet
├── TROUBLESHOOTING.md   # General troubleshooting
├── CHANGES_SUMMARY.md   # Major project changes
├── cluster/             # HPC cluster experiments
│   ├── README.md                  # Cluster overview
│   ├── QUICK_START.md             # Quick start for HPC
│   ├── SUBMIT_CHECKLIST.md        # Pre-submission checklist
│   ├── experiment1/               # Benchmarks (TabPFN, LoCalPFN, baselines)
│   │   └── README.md
│   └── experiment3/               # Classification head experiments
│       ├── README.md
│       └── troubleshooting.md
├── technical/           # Technical implementation details
│   ├── embeddings.md              # Embedding extraction and reuse
│   ├── preprocessing.md           # Preprocessing pipeline details
│   ├── multi-dataset.md           # Multi-dataset experiments
│   └── sam-dice-scores.md         # SAM Dice score evaluation
├── setup/              # Environment and cluster setup
│   └── cluster-setup.md           # Cluster environment configuration
└── analysis/           # Data analysis and visualization
    ├── visualization-notes.md     # Spacing, padding explanations
    └── sam-feature-evaluation.md  # SAM feature analysis
```

## Quick Links

### Getting Started
- **Lesion Filtering**: [FILTERING.md](FILTERING.md) ⭐ (consolidated guide)
- **Cluster Quick Start**: [cluster/QUICK_START.md](cluster/QUICK_START.md)
- **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

### Cluster Experiments
- **Cluster Overview**: [cluster/README.md](cluster/README.md)
- **Experiment 1 (Benchmarks)**: [cluster/experiment1/README.md](cluster/experiment1/README.md)
- **Experiment 3 (Classification Head)**: [cluster/experiment3/README.md](cluster/experiment3/README.md)
- **Submit Checklist**: [cluster/SUBMIT_CHECKLIST.md](cluster/SUBMIT_CHECKLIST.md)
- **Cluster Setup**: [setup/cluster-setup.md](setup/cluster-setup.md)

### Technical Details
- **Embeddings**: [technical/embeddings.md](technical/embeddings.md)
- **Preprocessing**: [technical/preprocessing.md](technical/preprocessing.md)
- **SAM Dice Scores**: [technical/sam-dice-scores.md](technical/sam-dice-scores.md)

### Troubleshooting
- **Experiment 3 Issues**: [cluster/experiment3/troubleshooting.md](cluster/experiment3/troubleshooting.md)
- **General Issues**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Analysis
- **Visualization Notes**: [analysis/visualization-notes.md](analysis/visualization-notes.md)
- **SAM Features**: [analysis/sam-feature-evaluation.md](analysis/sam-feature-evaluation.md)

## Additional Resources

- **Cluster Scripts**: See `cluster_scripts/` directory for batch job scripts and additional guides
- **Notebooks**: See `notebooks/` directory for interactive examples
- **Config Files**: See `configs/` directory for dataset configurations (CPU checkpoints, OpenMP, workers)

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
├── med3pipe/              # Med3Pipe (transfer learning APIs/CLI)
├── SAM-Med3D-main/        # Upstream SAM-Med3D repo (training/extraction)
├── configs/               # Datasets config (YAML)
├── docs/                  # This folder (index + guides)
├── notebooks/             # Experiments & analyses
├── scripts/               # Utility CLI/scripts
└── data/                  # Your datasets (outside git)
```
