# Cluster Setup Guide

## Virtual Environment Activation

All SLURM scripts use **module + venv** instead of conda.

### Fixed Scripts
1. ✅ `slurm_train_and_test_quick.sh`
2. ✅ `slurm_train_and_test.sh`
3. ✅ `slurm_experiment3.sh`
4. ✅ `slurm_experiment3_quick.sh`
5. ✅ `slurm_experiment3_gpu.sh`

### Standard Activation Pattern
```bash
# Load modules
module load Python/3.10

# Activate virtual environment
source /trinity/home/r112276/Med3Tab-PFN/venv/bin/activate
```

## Verify Environment

Before submitting jobs, verify the environment works:

```bash
# Interactive session
srun --partition=interactive --gres=gpu:1 --mem=32G --time=00:30:00 --pty bash

# Load module and activate venv
module load Python/3.10
source /trinity/home/r112276/Med3Tab-PFN/venv/bin/activate

# Verify Python
python --version
which python

# Verify PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Verify med3pipe
python -c "import med3pipe; print('med3pipe OK')"

# Exit interactive session
exit
```

## Common Issues

### Exit Code 127: "Command not found"

**Cause**: Scripts trying to use `python` without activating environment

**Solution**: Verify module name and venv path

```bash
# List available Python modules
module avail Python

# You might need exact name, e.g.:
# module load Python/3.10.8-GCCcore-12.2.0

# Check if venv exists
ls -la /trinity/home/r112276/Med3Tab-PFN/venv/bin/activate
```

### Wrong Environment Name

Based on your prompt `(thesis_peron)`, you might be using a different environment. Update the venv path in all scripts if needed.

## SLURM Scripts Overview

| Script | Purpose | Epochs | Time |
|--------|---------|--------|------|
| **slurm_train_and_test_quick.sh** | Experiment 1 quick test | 3 | ~1-2h |
| **slurm_train_and_test.sh** | Experiment 1 full run | 20 | ~8-12h |
| **slurm_experiment3_quick.sh** | Experiment 3 quick test | 5 | ~2h |
| **slurm_experiment3.sh** | Experiment 3 standard | 20 | ~6h |
| **slurm_experiment3_gpu.sh** | Experiment 3 GPU version | 20 | ~6h |

## Submitting Jobs

### Quick Test
```bash
cd /trinity/home/r112276/Med3Tab-PFN
sbatch cluster_scripts/slurm_train_and_test_quick.sh
```

### Monitor Jobs
```bash
# Check status
squeue -u r112276

# Watch log (replace JOBID)
tail -f logs/exp1_test_JOBID.log

# Check error log
tail -f logs/exp1_test_error_JOBID.log
```

## Expected Output on Success

```
==========================================
EXPERIMENT 1 - QUICK TEST (3 EPOCHS)
==========================================
SLURM Job ID: 793XXX
Job Name: exp1_test3ep
Node: gpu-XXX
Start Time: Fri Oct 31 13:55:00 CET 2025
==========================================

Code directory: /trinity/home/r112276/Med3Tab-PFN
Data directory: /data/scratch/r112276
Config file: /trinity/home/r112276/Med3Tab-PFN/configs/datasets_cluster.yaml
Results directory: /trinity/home/r112276/Med3Tab-PFN/results/experiment1_test

==========================================
Setting up environment...
==========================================
SLURM assigned GPU(s): 0

Python version:
Python 3.10.x

PyTorch and CUDA info:
PyTorch: 2.x.x
CUDA available: True
CUDA version: 11.x
GPU count: 1

==========================================
Running Experiment 1 - TEST MODE (3 epochs)
==========================================
...
```

## Troubleshooting Failed Jobs

### Collect Debug Info
```bash
# Get job details
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed

# Get full error log
cat logs/exp1_test_error_JOBID.log

# Check environment
module list
echo $VIRTUAL_ENV
which python
```

## Local vs Cluster Configs

Use different config files for local and cluster:
- **Local**: `configs/datasets.yaml`
- **Cluster**: `configs/datasets_cluster.yaml`

The cluster config has paths adjusted for `/data/scratch/r112276/` data directory.
