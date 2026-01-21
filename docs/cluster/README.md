# HPC Cluster Experiments

> Guide for running Med3Tab-PFN experiments on HPC clusters with SLURM

## Overview

This directory contains documentation for running experiments on high-performance computing clusters. The experiments evaluate SAM-Med3D features and compare classification methods.

## Available Experiments

| Experiment | Description | Script | Estimated Time |
|------------|-------------|--------|----------------|
| **Experiment 1** | Method comparison (TabPFN, LoCalPFN, baselines) | `exp1_benchmarks.py` | 4-8 hours |
| **Experiment 3** | SAM-Med3D classification head evaluation | `exp3_classifier.py` | 1-6 hours |
| **Preprocessing** | Three-way comparison (baseline, filtered, ROI) | `compare_classifier_preprocessing.py` | 12-18 hours |

---

## Quick Start

```bash
# 1. Configure environment in SLURM script
nano cluster_scripts/slurm/slurm_exp1.sh

# 2. Submit job
sbatch cluster_scripts/slurm/slurm_exp1.sh

# 3. Monitor
squeue -u $USER
tail -f logs/exp1_*.log
```

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions.

---

## Experiment Details

### Experiment 1: Method Comparison

Compare classification performance across methods:
- **Med3-TabPFN**: SAM-Med3D embeddings + TabPFN
- **Med3-LoCalPFN**: SAM-Med3D embeddings + LoCalPFN
- **DenseNet121-3D**: End-to-end 3D CNN
- **ViT-3D (Swin)**: 3D Vision Transformer

**Documentation**: [experiment1/README.md](experiment1/README.md)

```bash
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets_cluster.yaml \
    --n-splits 5
```

### Experiment 3: Classification Head

Evaluate SAM-Med3D feature quality before running the full pipeline:
- Trains linear classifier on frozen SAM-Med3D features
- AUC > 0.70 indicates good feature quality
- AUC < 0.60 suggests features need improvement

**Documentation**: [experiment3/README.md](experiment3/README.md)

```bash
python cluster_scripts/experiments/exp3_classifier.py \
    --config configs/datasets_cluster.yaml \
    --freeze-encoder \
    --epochs 20
```

### Preprocessing Comparison

Compare preprocessing strategies:
- **Baseline**: Full CT volume, no filtering
- **Filtered**: With lesion quality filtering
- **ROI-Cropped**: Tumor-centered extraction

**Documentation**: [CLASSIFIER_PREPROCESSING_COMPARISON.md](CLASSIFIER_PREPROCESSING_COMPARISON.md)

```bash
python cluster_scripts/experiments/compare_classifier_preprocessing.py \
    --config configs/datasets_cluster.yaml
```

---

## SLURM Configuration

### Standard Job Template

```bash
#!/bin/bash
#SBATCH --job-name=med3tabpfn
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_error_%j.log
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.edu
```

### Available Scripts

| Script | Purpose |
|--------|---------|
| `slurm_exp1.sh` | Experiment 1 (all methods) |
| `slurm_exp1_filtered.sh` | Experiment 1 with filtering |
| `slurm_exp1_roi.sh` | Experiment 1 with ROI cropping |
| `slurm_exp3.sh` | Experiment 3 (classification head) |
| `slurm_compare_classifier.sh` | Preprocessing comparison |

---

## Results Interpretation

### Experiment 3 (Feature Quality)

| AUC Score | Interpretation | Action |
|-----------|----------------|--------|
| > 0.70 | Good features | Proceed to Experiment 1 |
| 0.60 - 0.70 | Moderate features | Consider fine-tuning |
| < 0.60 | Poor features | Check data quality |

### Experiment 1 (Benchmarks)

Expected metrics in `results/experiment1/combined_benchmarks_summary.csv`:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

---

## Output Structure

```
results/
├── experiment1/
│   └── combined_benchmarks_summary.csv
├── classification_head/
│   ├── gist/
│   │   ├── best_model.pth
│   │   ├── training_log.csv
│   │   └── metrics.txt
│   └── lipo/
└── three_way_comparison/
    └── preprocessing_comparison.csv
```

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [QUICK_START.md](QUICK_START.md) | Getting started guide |
| [SUBMIT_CHECKLIST.md](SUBMIT_CHECKLIST.md) | Pre-submission verification |
| [SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md) | Complete script documentation |
| [ROI_EXPERIMENTS.md](ROI_EXPERIMENTS.md) | ROI cropping experiments |
| [experiment1/README.md](experiment1/README.md) | Benchmark experiment details |
| [experiment3/README.md](experiment3/README.md) | Classification head details |
| [experiment3/troubleshooting.md](experiment3/troubleshooting.md) | Experiment 3 troubleshooting |

---

## Related Documentation

- [Main README](../../README.md) — Project overview
- [Cluster Scripts](../../cluster_scripts/README.md) — Script reference
- [Troubleshooting](../TROUBLESHOOTING.md) — General troubleshooting

### 3. Run on All Datasets
```bash
# Once satisfied, run on all datasets
sbatch cluster_scripts/slurm_experiment3.sh
```

## 💡 Tips

### Frozen Encoder (Recommended First)
**Pros**:
- ⚡ Fast (~30 min per dataset)
- 💾 Low memory usage
- 🎯 Tests pretrained feature quality

**Cons**:
- May not reach optimal performance
- Features fixed from pretraining

**When to use**: Always start here!

### Fine-Tuning
**Pros**:
- 📈 Can improve performance
- 🎨 Adapts features to your data

**Cons**:
- ⏰ Much slower (~3-6 hours per dataset)
- 💾 Higher memory usage
- ⚠️ Risk of overfitting on small datasets

**When to use**: Only if frozen encoder gives poor results (AUC < 0.65)

## 🆘 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
--batch-size 2

# Or reduce image size
--img-size 96
```

### "No checkpoint found"
Download SAM-Med3D checkpoint:
```bash
cd sam-med3d/ckpt/
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
```

### Poor results (AUC < 0.60) even with fine-tuning
1. **Check data quality**: Verify labels are correct
2. **Check class balance**: Ensure both classes are represented
3. **Try different hyperparameters**: Adjust learning rate, dropout
4. **Check preprocessing**: Ensure images are normalized properly

### Training is slow
- Frozen encoder: Should take ~30 min per dataset
- Fine-tuning: Can take 3-6 hours per dataset
- Use `nvidia-smi` to check GPU utilization

## 📊 Example Results

### Good Features (Proceed to Full Pipeline)
```
Dataset: gist
Best Epoch: 7
Best Val AUC: 0.7456  ← Good! Features are discriminative
Final Accuracy: 0.7200
```

### Moderate Features (Try Fine-Tuning)
```
Dataset: lipo
Best Epoch: 9
Best Val AUC: 0.6543  ← Moderate, consider fine-tuning
Final Accuracy: 0.6087
```

### Poor Features (Investigate)
```
Dataset: example
Best Epoch: 5
Best Val AUC: 0.5234  ← Poor, check data or try different approach
Final Accuracy: 0.5100
```

## 🔗 Related Files

- **Notebook version**: `notebooks/Experiment3-Classification-Head-on-SAM.ipynb`
- **Python script**: `cluster_scripts/run_experiment3_classification_head.py`
- **SLURM script**: `cluster_scripts/slurm_experiment3.sh`
- **Full benchmarks**: See Experiment 1 for complete comparison

---

**Next step after Experiment 3**: If features are good (AUC > 0.70), run Experiment 1 with `slurm_train_and_test.sh`
