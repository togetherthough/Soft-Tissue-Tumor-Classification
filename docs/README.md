# Documentation

Welcome to the Med3Tab-PFN documentation. This guide covers installation, usage, experiments, and technical implementation details.

## 📚 Documentation Overview

| Section | Description |
|---------|-------------|
| [Getting Started](#getting-started) | Installation, quick start, and configuration |
| [Technical Guides](#technical-guides) | Preprocessing, embeddings, and implementation details |
| [Experiments](#experiments) | HPC cluster experiments and benchmarking |
| [Troubleshooting](#troubleshooting) | Common issues and solutions |
| [Analysis](#analysis-tools) | Visualization and feature analysis |

---

## Getting Started

### Quick Links
| Document | Description |
|----------|-------------|
| [Quick Reference](QUICK_REFERENCE.md) | Command cheat sheet for common operations |
| [Lesion Filtering](FILTERING.md) | Filter samples by lesion quality metrics |
| [Dataset Configuration](../configs/README.md) | YAML configuration for datasets |

### First Steps

1. **Install dependencies**: `pip install -r med3pipe/requirements.txt`
2. **Configure datasets**: Edit `configs/datasets.yaml` with your data paths
3. **Run a quick test**: `python -m med3pipe multi-tabpfn --config configs/datasets.yaml`

---

## Technical Guides

Detailed documentation on the pipeline implementation.

| Document | Description |
|----------|-------------|
| [Preprocessing Pipeline](technical/preprocessing.md) | Resize-then-pad approach for 3D volumes |
| [Embedding Extraction](technical/embeddings.md) | SAM-Med3D feature extraction and caching |
| [Multi-Dataset Workflows](technical/multi-dataset.md) | Processing multiple datasets |
| [SAM Dice Scores](technical/sam-dice-scores.md) | Segmentation quality evaluation |
| [Pipeline Testing](technical/pipeline-testing.md) | End-to-end testing procedures |

### Preprocessing Improvements
- [Changes Summary](CHANGES_SUMMARY.md) — Overview of resize-then-pad upgrade
- [ROI Cropping](roi_cropping.md) — Tumor-centered volume extraction

---

## Experiments

### HPC Cluster Deployment

| Document | Description |
|----------|-------------|
| [Cluster Quick Start](cluster/QUICK_START.md) | Get running on SLURM in minutes |
| [Cluster Overview](cluster/README.md) | Full cluster setup and configuration |
| [Submit Checklist](cluster/SUBMIT_CHECKLIST.md) | Pre-submission verification |
| [Cluster Setup](setup/cluster-setup.md) | Environment configuration |

### Experiment Guides

| Experiment | Document | Description |
|------------|----------|-------------|
| **Experiment 1** | [Benchmarks](cluster/experiment1/README.md) | TabPFN, LoCalPFN, DenseNet, ViT comparison |
| **Experiment 3** | [Classification Head](cluster/experiment3/README.md) | SAM-Med3D feature evaluation |
| **Preprocessing** | [Comparison](cluster/CLASSIFIER_PREPROCESSING_COMPARISON.md) | Baseline vs. Filtered vs. ROI |
| **ROI Ablation** | [ROI Experiments](cluster/ROI_EXPERIMENTS.md) | ROI margin analysis |

### Script References
- [Scripts Reference](cluster/SCRIPTS_REFERENCE.md) — Complete guide to all cluster scripts
- [Cluster Scripts README](../cluster_scripts/README.md) — Quick reference for scripts

---

## Troubleshooting

| Document | Description |
|----------|-------------|
| [General Troubleshooting](TROUBLESHOOTING.md) | Common issues with CPU/GPU, checkpoints, data loaders |
| [Experiment 3 Issues](cluster/experiment3/troubleshooting.md) | Classification head specific problems |

### Common Issues

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch size or use `--freeze-encoder` |
| CPU checkpoint on GPU | Load with `map_location='cpu'` |
| OpenMP duplicate runtime | Set `KMP_DUPLICATE_LIB_OK=TRUE` |
| Missing embeddings | Run extraction with `skip_existing_embeddings=True` |

---

## Analysis Tools

### Visualization & Analysis
| Document | Description |
|----------|-------------|
| [Visualization Notes](analysis/visualization-notes.md) | Understanding spacing and padding |
| [SAM Feature Evaluation](analysis/sam-feature-evaluation.md) | Analyzing feature quality |

### Scripts
See [scripts/README.md](../scripts/README.md) for utility scripts including:
- Lesion size analysis
- Label distribution analysis
- Dice score summaries
- Per-dataset performance reports

---

## Directory Structure

```
docs/
├── README.md                  # This file
├── FILTERING.md               # Lesion size filtering guide
├── QUICK_REFERENCE.md         # Command cheat sheet
├── TROUBLESHOOTING.md         # General troubleshooting
├── CHANGES_SUMMARY.md         # Project changelog
├── roi_cropping.md            # ROI cropping documentation
├── roi_cropping_experiments.md # ROI experiment details
├── LESION_SIZE_FILTERING.md   # Legacy filtering docs
│
├── cluster/                   # HPC cluster documentation
│   ├── README.md              # Cluster overview
│   ├── QUICK_START.md         # Quick start guide
│   ├── SUBMIT_CHECKLIST.md    # Pre-submission checklist
│   ├── SCRIPTS_REFERENCE.md   # Script documentation
│   ├── ROI_EXPERIMENTS.md     # ROI experiments
│   ├── CLASSIFIER_PREPROCESSING_COMPARISON.md
│   ├── experiment1/           # Benchmark experiments
│   └── experiment3/           # Classification head
│
├── technical/                 # Implementation details
│   ├── embeddings.md          # Embedding extraction
│   ├── preprocessing.md       # Preprocessing pipeline
│   ├── multi-dataset.md       # Multi-dataset workflows
│   ├── sam-dice-scores.md     # SAM evaluation
│   └── pipeline-testing.md    # Testing procedures
│
├── setup/                     # Environment setup
│   └── cluster-setup.md       # Cluster configuration
│
└── analysis/                  # Analysis guides
    ├── visualization-notes.md # Visualization
    └── sam-feature-evaluation.md # Feature analysis
```

---

## Quick Commands

```bash
# Run TabPFN on all datasets
python -m med3pipe multi-tabpfn --config configs/datasets.yaml

# Run with lesion filtering
python -m med3pipe multi-tabpfn --config configs/datasets.yaml \
    --min-voxels 500 --min-dimension 5

# Run with ROI cropping
python cluster_scripts/experiments/exp1_benchmarks.py \
    --use-roi-crop --roi-margin 10

# Test SAM features
python scripts/test_sam_features.py --dataset gist --epochs 10

# Submit cluster job
sbatch cluster_scripts/slurm/slurm_exp1.sh
```

---

## Related Documentation

- **[Main README](../README.md)** — Project overview and quick start
- **[med3pipe Package](../med3pipe/README.md)** — Core library documentation
- **[Configuration Guide](../configs/README.md)** — Dataset configuration
- **[Cluster Scripts](../cluster_scripts/README.md)** — HPC scripts reference
- Feature extraction and PFN baselines: `notebooks/Med3D-TabPFN.ipynb`, `notebooks/Med3D-LoCalPFN.ipynb`
- SAM feature quick test: `notebooks/Test-SAM-Features.ipynb`

---

## Project Structure (high level)

```
├── med3pipe/              # Med3Pipe (transfer learning APIs/CLI)
├── sam-med3d/             # SAM-Med3D checkpoints and features
├── configs/               # Datasets config (YAML)
├── docs/                  # This folder (index + guides)
├── notebooks/             # Experiments & analyses
├── scripts/               # Utility CLI/scripts
└── data/                  # Your datasets (outside git)
```
