# Med3Tab-PFN

> **Transfer Learning for 3D Medical Image Classification using SAM-Med3D and Tabular Foundation Models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Med3Tab-PFN is a novel **transfer learning framework** for binary classification of 3D medical imaging data (CT/MRI). It combines:

- **[SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)**: A foundation model pretrained on 131K 3D medical images for feature extraction
- **[TabPFN](https://github.com/automl/TabPFN) / [LoCalPFN](https://github.com/automl/LoCalPFN)**: Prior-data fitted networks for tabular classification

**Key Innovation**: Instead of training end-to-end deep learning models from scratch, we extract rich semantic embeddings from SAM-Med3D and apply tabular foundation models for classification—achieving competitive performance with significantly reduced computational cost.

### Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            Med3Tab-PFN Pipeline                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐               │
│   │  3D Volume  │───▶│     ROI      │───▶│    SAM-Med3D     │               │
│   │  (CT/MRI)   │    │   Cropping   │    │   Image Encoder  │               │
│   └─────────────┘    └──────────────┘    └─────────┬────────┘               │
│                                                     │                         │
│                                                     ▼                         │
│                                         ┌──────────────────────────┐          │
│                                         │   Feature Embeddings     │          │
│                                         │   (384-dim vectors)      │          │
│                                         └─────────┬────────────────┘          │
│                                                   │                           │
│                                                   ▼                           │
│   ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐    │
│   │   Labels    │───▶│   TabPFN or      │◀───│   Average Pooling +      │    │
│   │  (Binary)   │    │   LoCalPFN       │    │   Standardization + PCA  │    │
│   └─────────────┘    └────────┬─────────┘    └──────────────────────────┘    │
│                               │                                               │
│                               ▼                                               │
│                      ┌──────────────────┐                                     │
│                      │   Classification │                                     │
│                      │   (Accuracy, AUC)│                                     │
│                      └──────────────────┘                                     │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **SAM-Med3D Embeddings** | Extract semantically rich 384-dimensional feature vectors from 3D volumes |
| 🧠 **TabPFN/LoCalPFN** | State-of-the-art tabular classification with minimal hyperparameter tuning |
| 📊 **Multi-Dataset** | Process multiple datasets via unified YAML configuration |
| 🎯 **ROI Cropping** | Tumor-centered preprocessing for focused feature extraction |
| 🔧 **Lesion Filtering** | Quality-based sample filtering to remove noisy data |
| 📈 **Benchmarking** | Built-in comparison with DenseNet121-3D and ViT-3D baselines |
| 🖥️ **HPC Ready** | SLURM scripts for cluster deployment |

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Med3Tab-PFN.git
cd Med3Tab-PFN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r med3pipe/requirements.txt
```

**Note on TabPFN**: If using TabPFN v2.5+, you need HuggingFace authentication:
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login with your HuggingFace token
huggingface-cli login

# Accept model terms at: https://huggingface.co/Prior-Labs/tabpfn_2_5
```

*Alternatively, use TabPFN v1.x (no auth required)*: `pip install "tabpfn<2.0"`

### 2. Download SAM-Med3D Checkpoint

```bash
mkdir -p sam-med3d/ckpt
# Download from: https://github.com/uni-medical/SAM-Med3D
# Place at: sam-med3d/ckpt/sam_med3d_turbo.pth
```

### 3. Run Experiments

**Option A: Jupyter Notebook** *(Recommended for exploration)*
```bash
jupyter notebook notebooks/walkthrough/Med3D-TabPFN.ipynb
```

**Option B: Command Line**
```bash
# TabPFN classification
python -m med3pipe multi-tabpfn --config configs/datasets.yaml

# LoCalPFN with custom parameters
python -m med3pipe multi-localpfn --config configs/datasets.yaml \
    --local-k 50 --local-fit-adapter
```

**Option C: Python API**
```python
from med3pipe.pipelines import run_multi_tabpfn

results = run_multi_tabpfn("configs/datasets.yaml")
print(results["summary_df"].to_string())
```

---

## Dataset Configuration

### Data Structure

```
data/
├── gist/
│   ├── GIST-001_CT/
│   │   └── 1/NIFTI/
│   │       ├── image.nii.gz
│   │       └── segmentation.nii.gz
│   ├── GIST-002_CT/
│   │   └── ...
│   └── sheet.csv
└── lipo/
    └── ...
```

### Configuration File (`configs/datasets.yaml`)

```yaml
datasets:
  gist:
    dataset_root: data/gist
    category: gist
    ct_name: ct_GIST
    labels:
      sheet_csv: sheet.csv
      subject_col: Subject
      label_col: Diagnosis_binary
    split:
      ratio: 0.8
      seed: 2025
```

See [configs/README.md](configs/README.md) for detailed configuration options.

---

## Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **Med3-TabPFN** | SAM-Med3D → TabPFN | Fast, minimal tuning required |
| **Med3-LoCalPFN** | SAM-Med3D → LoCalPFN | Better for heterogeneous data |
| **DenseNet121-3D** | End-to-end 3D CNN | Baseline comparison |
| **ViT-3D (Swin)** | 3D Vision Transformer | Baseline comparison |

---

## Experiments

| Experiment | Description | Script |
|------------|-------------|--------|
| **Exp 1: Benchmarks** | Compare all methods | `cluster_scripts/experiments/exp1_benchmarks.py` |
| **Exp 3: Classification Head** | Evaluate SAM-Med3D features | `cluster_scripts/experiments/exp3_classifier.py` |
| **Preprocessing Comparison** | Baseline vs Filtered vs ROI | `cluster_scripts/experiments/compare_classifier_preprocessing.py` |

### Running on HPC

```bash
# Submit to SLURM
sbatch cluster_scripts/slurm/slurm_exp1.sh

# Monitor
squeue -u $USER
tail -f logs/exp1_*.log
```

---

## Project Structure

```
Med3Tab-PFN/
├── med3pipe/              # Core Python package
│   ├── pipelines/         # End-to-end pipelines
│   ├── sam/               # SAM-Med3D integration
│   ├── tabular/           # TabPFN/LoCalPFN wrappers
│   ├── vision/            # 3D CNN baselines
│   └── training/          # Training utilities
├── notebooks/             # Jupyter notebooks
│   ├── walkthrough/       # Step-by-step tutorials
│   └── experiments/       # Experiment notebooks
├── cluster_scripts/       # HPC experiment scripts
│   ├── experiments/       # Python scripts
│   └── slurm/             # SLURM job scripts
├── configs/               # Dataset configurations
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
├── data/                  # Datasets (gitignored)
├── results/               # Experiment outputs
└── sam-med3d/             # SAM-Med3D resources
```

---

## Documentation

📚 **[Full Documentation Index](docs/README.md)**

### Getting Started
- [Quick Reference](docs/QUICK_REFERENCE.md) — Common commands cheat sheet
- [Dataset Configuration](configs/README.md) — YAML setup guide
- [Troubleshooting](docs/TROUBLESHOOTING.md) — Common issues and fixes

### Technical Guides
- [Preprocessing Pipeline](docs/technical/preprocessing.md) — Resize-then-pad approach
- [Embedding Extraction](docs/technical/embeddings.md) — Reusing cached embeddings
- [Lesion Filtering](docs/FILTERING.md) — Quality-based sample selection
- [Multi-Dataset Workflows](docs/technical/multi-dataset.md)

### HPC & Experiments
- [Cluster Quick Start](docs/cluster/QUICK_START.md)
- [Experiment 1: Benchmarks](docs/cluster/experiment1/README.md)
- [Experiment 3: Classification Head](docs/cluster/experiment3/README.md)
- [Scripts Reference](docs/cluster/SCRIPTS_REFERENCE.md)

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.0+ (recommended)
- 16GB+ RAM (32GB recommended)

See [med3pipe/requirements.txt](med3pipe/requirements.txt) for dependencies.

---

## Citation

```bibtex
@mastersthesis{med3tabpfn2026,
  title={Transfer Learning for 3D Medical Image Classification 
         using SAM-Med3D and Tabular Foundation Models},
  author={Your Name},
  school={Your Institution},
  year={2026}
}
```

## Acknowledgments

- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) — 3D medical image foundation model
- [TabPFN](https://github.com/automl/TabPFN) — Tabular prior-data fitted networks
- [LoCalPFN](https://github.com/automl/LoCalPFN) — Local context-aware TabPFN
- [MONAI](https://monai.io/) — Medical Open Network for AI

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Questions?</b> Open an <a href="https://github.com/yourusername/Med3Tab-PFN/issues">issue</a> · 
  Check the <a href="docs/TROUBLESHOOTING.md">troubleshooting guide</a>
</p>
