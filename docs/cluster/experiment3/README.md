# Experiment 3: SAM-Med3D Classification Head Training

Complete guide for running Experiment 3 on both local and cluster environments.

## Overview

Experiment 3 trains a classification head on top of frozen SAM-Med3D embeddings. This approach:
- Uses pre-trained SAM-Med3D image encoder (frozen)
- Extracts 3D embeddings from medical images
- Trains a lightweight classification head (Linear → ReLU → Dropout → Linear)
- Achieves good performance with minimal training time

## Quick Start

### Local Testing (Windows)
```powershell
# Test with single dataset, 3 epochs (~10 minutes)
python cluster_scripts/run_experiment3_classification_head.py `
    --config configs/datasets.yaml `
    --datasets gist `
    --epochs 3 `
    --batch-size 2 `
    --freeze-encoder
```

### Cluster - Quick Test (5 epochs, ~2 hours)
```bash
# Pre-configured for fast testing
sbatch cluster_scripts/slurm_experiment3_quick.sh

# Single dataset
sbatch cluster_scripts/slurm_experiment3_quick.sh gist
```

### Cluster - Full Run (20 epochs, ~6 hours)
```bash
sbatch cluster_scripts/slurm_experiment3.sh
```

## Running with Different Epochs

### Method 1: Use Quick Script (Recommended)
```bash
# Pre-configured for 5 epochs
sbatch cluster_scripts/slurm_experiment3_quick.sh
```

### Method 2: Edit Existing Script
```bash
nano cluster_scripts/slurm_experiment3.sh
# Change line 94: --epochs 20 → --epochs 5
sbatch cluster_scripts/slurm_experiment3.sh
```

### Method 3: Interactive Testing (Best for Debugging)
```bash
# Request GPU node
srun --partition=interactive --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash

# Activate environment
cd /trinity/home/r112276/Med3Tab-PFN
conda activate sammed3d

# Test with custom settings
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist \
    --epochs 3 \
    --batch-size 4 \
    --freeze-encoder
```

### Method 4: Direct Python Command
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist lipo crlm \
    --epochs 5 \
    --lr 5e-4 \
    --batch-size 8
```

## Configuration

### Core Arguments
```bash
--config PATH                  # Config file (datasets.yaml or datasets_cluster.yaml)
--datasets [NAMES...]          # Which datasets to run (default: all)
--epochs N                     # Number of training epochs (default: 10)
--freeze-encoder               # Freeze SAM encoder (FAST, recommended first)
--fine-tune                    # Train full model (SLOW, only if frozen fails)
```

### Performance Tuning
```bash
--batch-size N                 # Larger = faster but more memory (default: 4)
--lr FLOAT                     # Learning rate (default: 1e-3)
--weight-decay FLOAT           # Regularization (default: 1e-4)
--dropout FLOAT                # Dropout rate (default: 0.3)
--img-size N                   # Image size (default: 128)
--num-workers N                # Data loading threads (default: 2)
```

## Expected Results by Epoch Count

| Epochs | Time per Dataset | All 6 Datasets | Performance | Use Case |
|--------|------------------|----------------|-------------|----------|
| 3      | ~10-15 min      | ~1 hour        | ~70-80%     | Debugging, testing setup |
| 5      | ~15-25 min      | ~1.5-2 hours   | ~80-90%     | Fast evaluation, initial experiments |
| 10     | ~30-45 min      | ~3-4 hours     | ~85-95%     | Standard training (recommended) |
| 20     | ~1-1.5 hours    | ~6-8 hours     | ~90-95%     | Best performance (diminishing returns) |

*Times with `--freeze-encoder` (recommended for initial testing)*

## Recommended Testing Strategy

### Phase 1: Verify Setup (3 epochs)
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --datasets gist \
    --epochs 3 \
    --config configs/datasets_cluster.yaml
```
**Time**: ~10 minutes per dataset  
**Purpose**: Test if setup works and checkpoint loads

### Phase 2: Fast Screening (5 epochs)
```bash
sbatch cluster_scripts/slurm_experiment3_quick.sh
```
**Time**: ~1.5-2 hours total  
**Purpose**: Check all datasets for major issues

### Phase 3: Full Training (15-20 epochs)
```bash
# Edit slurm_experiment3.sh to use --epochs 15
sbatch cluster_scripts/slurm_experiment3.sh
```
**Time**: ~5-7 hours total  
**Purpose**: Final results with best performance

## Diagnostic Tools

### Run Diagnostics Before Starting
```bash
bash cluster_scripts/diagnose_experiment3.sh
```

This checks:
- ✅ Config file validity (no duplicates)
- ✅ sheet.csv existence and dataset names
- ✅ SAM-Med3D checkpoint file
- ✅ Dataset directories
- ✅ Python environment
- ✅ GPU availability
- ✅ Recent error logs

### Monitor Progress
```bash
# Check SLURM queue
squeue -u $USER

# Watch live log
tail -f logs/exp3_*.log

# Check errors
tail -f logs/exp3_error_*.log

# Cancel job if needed
scancel JOBID
```

## Output Structure

Results are saved to `results/classification_head/`:

```
results/classification_head/
├── summary.csv              # All datasets summary
├── gist/
│   ├── best_model.pt       # Best model weights
│   ├── training_history.json
│   ├── final_metrics.json
│   └── predictions.npy
└── [other datasets...]
```

## Success Indicators

You'll know it's working when you see:

```
[1/4] Preparing dataset...
✓ Prepared 123 cases

[2/4] Creating validation split...
✓ Split created (ratio=0.8, seed=2025)

[3/4] Loading labels...
✓ Loaded 123 labels
  Class distribution: {0: 62, 1: 61}

[4/4] Training classification head...
  Freeze encoder: True
  Epochs: 5
  Batch size: 4
  Learning rate: 0.001

Epoch 1/5: 100%|████████| 25/25 [00:15<00:00]
Train Loss: 0.6234, Train Acc: 0.6800
Val Loss: 0.5891, Val Acc: 0.7200, Val AUC: 0.7456
...
```

## See Also

- **Troubleshooting Guide**: [troubleshooting.md](./troubleshooting.md)
- **Main Experiment Guide**: `cluster_scripts/EXPERIMENT3_GUIDE.md`
- **Cluster Scripts README**: `cluster_scripts/README.md`
