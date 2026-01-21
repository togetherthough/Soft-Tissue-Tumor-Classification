# Quick Reference

> Command cheat sheet for common Med3Tab-PFN operations

## Pipeline Commands

### Run TabPFN Classification
```bash
# All datasets
python -m med3pipe multi-tabpfn --config configs/datasets.yaml

# Single dataset
python -m med3pipe multi-tabpfn --config configs/datasets.yaml --datasets gist
```

### Run LoCalPFN Classification
```bash
python -m med3pipe multi-localpfn --config configs/datasets.yaml \
    --local-k 50 --local-fit-adapter
```

### Run with Filtering
```bash
python -m med3pipe multi-tabpfn --config configs/datasets.yaml \
    --min-voxels 500 --min-dimension 5 --min-density 0.3
```

---

## Experiment Commands

### Experiment 1: Method Comparison
```bash
# Standard run
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets.yaml

# With ROI cropping
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets.yaml \
    --use-roi-crop --roi-margin 10

# With filtering
python cluster_scripts/experiments/exp1_benchmarks.py \
    --config configs/datasets.yaml \
    --filter-preset recommended
```

### Experiment 3: Classification Head
```bash
# Frozen encoder (fast)
python cluster_scripts/experiments/exp3_classifier.py \
    --config configs/datasets.yaml \
    --freeze-encoder --epochs 20

# Fine-tuning
python cluster_scripts/experiments/exp3_classifier.py \
    --config configs/datasets.yaml \
    --fine-tune --epochs 50
```

---

## SLURM Submission

```bash
# Experiment 1
sbatch cluster_scripts/slurm/slurm_exp1.sh

# Experiment 3
sbatch cluster_scripts/slurm/slurm_exp3.sh

# With filtering
sbatch cluster_scripts/slurm/slurm_exp1_filtered.sh

# Monitor job
squeue -u $USER
tail -f logs/exp1_*.log
```

---

## Data Preparation

### Prepare Dataset
```bash
python -m med3pipe prepare \
    --dataset-root data/gist \
    --category gist \
    --ct-name ct_GIST
```

### Create Train/Val Split
```bash
python -m med3pipe split \
    --category gist \
    --ct-name ct_GIST \
    --split-ratio 0.8
```

---

## Preprocessing

### Resize-Then-Pad Transform
```python
from med3pipe.sam.transforms import load_volume_resize_pad, ResizeLargestTo

# Load with preprocessing
image = load_volume_resize_pad("path/to/image.nii.gz", img_size=128)

# Use transform directly
import torchio as tio
transform = tio.Compose([
    tio.ToCanonical(),
    ResizeLargestTo(target_size=128),
    tio.CropOrPad(target_shape=(128, 128, 128)),
])
```

---

## Python API

### End-to-End Pipeline
```python
from med3pipe.pipelines import run_multi_tabpfn

results = run_multi_tabpfn("configs/datasets.yaml")
print(results["summary_df"])
```

### Single Dataset
```python
from med3pipe import run_single_dataset

result = run_single_dataset(
    method="tabpfn",
    dataset_root="data/gist",
    category="gist",
    ct_name="ct_GIST",
)
```

### 3D Baselines
```python
from med3pipe import train_eval_densenet121_3d, Sam3DPaths, Train3DConfig

paths = Sam3DPaths(sam3d_root="sam-med3d", category="gist", ct_name="ct_GIST")
cfg = Train3DConfig(epochs=30, img_size=96, batch_size=2, device="cuda")

result = train_eval_densenet121_3d(paths, "data/gist/sheet.csv", num_classes=2, cfg=cfg)
```

---

## Utility Commands

### Check Environment
```bash
bash cluster_scripts/utils/check_setup.sh
```

### Verify SAM Checkpoint
```bash
bash cluster_scripts/utils/quick_weight_check.sh
```

### Analyze Lesion Sizes
```bash
python scripts/analysis/analyze_lesion_sizes.py --config configs/datasets.yaml
```

### Test SAM Features
```bash
python scripts/evaluation/test_sam_features.py --dataset gist --epochs 10 --freeze
```

---

## Filter Presets

| Preset | Command | Retention |
|--------|---------|-----------|
| Recommended | `--filter-preset recommended` | ~34% |
| Conservative | `--filter-preset conservative` | ~20% |
| Lenient | `--filter-preset lenient` | ~60% |

---

## Key File Locations

| File | Purpose |
|------|---------|
| `configs/datasets.yaml` | Dataset configuration |
| `sam-med3d/ckpt/sam_med3d_turbo.pth` | SAM-Med3D weights |
| `data/lesion_size_analysis.csv` | Lesion metrics for filtering |
| `results/experiment1/` | Benchmark results |
| `results/classification_head/` | Classification head results |

---

## Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| CUDA OOM | `--batch-size 2` or `--freeze-encoder` |
| CPU checkpoint on GPU | `torch.load(..., map_location='cpu')` |
| OpenMP conflict | `export KMP_DUPLICATE_LIB_OK=TRUE` |
| Slow DataLoader | `--num-workers 4` |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.
