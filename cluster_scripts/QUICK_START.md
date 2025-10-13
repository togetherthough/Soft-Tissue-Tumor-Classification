# Quick Start Guide: Running Experiment 1 on GPU Cluster

This is a condensed guide for experienced users. For detailed instructions, see `CLUSTER_EXPERIMENT_GUIDE.md`.

## Prerequisites Checklist

- ✅ Code copied to `/trinity/home/r112276/Med3Tab-PFN`
- ✅ Data in `/data/scratch/r112276/data/{gist,lipo}/`
- ✅ Python environment with PyTorch + CUDA
- ✅ Required packages installed

## Quick Setup (5 steps)

### 1. SSH to cluster

```bash
ssh r112276@gpu-login.erasmusmc.nl
cd /trinity/home/r112276/Med3Tab-PFN/cluster_scripts
```

### 2. Verify environment (optional but recommended)

```bash
bash check_setup.sh
```

### 3. Edit SLURM script

```bash
nano slurm_experiment1.sh
```

**Required changes:**
- Line 10: Update email address
- Line 68-72: Uncomment and fix environment activation (conda/virtualenv)

**Optional changes:**
- Line 5-9: Adjust resources (time, CPUs, GPU, memory)
- Line 103-140: Verify data paths match your setup

### 4. Make script executable

```bash
chmod +x slurm_experiment1.sh check_setup.sh
```

### 5. Submit job

```bash
sbatch slurm_experiment1.sh
```

## Monitor Job

```bash
# Check status
squeue -u r112276

# Watch output log (replace 123456 with your job ID)
tail -f logs/experiment1_benchmarks_123456.log

# Check errors
tail -f logs/experiment1_benchmarks_error_123456.log
```

## View Results

```bash
# Summary CSV
cat results/experiment1/combined_benchmarks_summary.csv

# All results
ls -lh results/experiment1/
```

## Cancel Job

```bash
# Replace 123456 with your job ID
scancel 123456
```

## Common Issues

### 1. Job fails immediately
```bash
# Check error log
cat logs/experiment1_benchmarks_error_*.log
```

**Solutions:**
- Verify environment activation in SLURM script
- Check data paths exist
- Ensure GPU requested: `#SBATCH --gres=gpu:1`

### 2. Import errors
```bash
# Activate environment and test
conda activate sammed3d  # or your env name
python -c "import torch, med3pipe; print('OK')"
```

### 3. CUDA not available
```bash
# On compute node (not login node):
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Note:** CUDA may not be available on login nodes - this is normal. The compute node will have GPU access.

### 4. Out of memory
Increase memory in SLURM script:
```bash
#SBATCH --mem=128G  # was 64G
```

## Customization

### Run specific datasets only

Edit line 171 in `slurm_experiment1.sh`:
```bash
srun python ${SCRIPT_DIR}/run_experiment1_benchmarks.py \
    --datasets gist \  # Only run GIST
    ...
```

### Skip certain methods

Add flags to the python command:
```bash
srun python ${SCRIPT_DIR}/run_experiment1_benchmarks.py \
    --skip-baselines \    # Skip DenseNet & ViT
    --epochs-3d 20 \      # More epochs for remaining models
    ...
```

### Adjust training epochs

```bash
--epochs-3d 20  # Default is 4
```

## Resource Guidelines

### Quick test (1-2 hours)
```bash
#SBATCH --partition=short
#SBATCH --time=0-02:00:00
#SBATCH --mem=32G
```

### Full benchmark (1-2 days)
```bash
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
```

### Extensive run (several days)
```bash
#SBATCH --partition=long
#SBATCH --time=5-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
```

## Files Overview

| File | Purpose |
|------|---------|
| `slurm_experiment1.sh` | Main SLURM batch script |
| `run_experiment1_benchmarks.py` | Python experiment script |
| `check_setup.sh` | Environment verification |
| `EXPERIMENT_GUIDE.md` | Detailed documentation |
| `QUICK_START.md` | This file |
| `COMMANDS.txt` | Command reference card |
| `README.md` | Folder overview |

## Expected Output Files

```
results/experiment1/
├── combined_benchmarks_summary.csv    # Main results table
├── tabpfn_runs/                        # TabPFN detailed outputs
├── localpfn_runs/                      # LoCalPFN detailed outputs
└── baselines/                          # DenseNet & ViT outputs
```

## One-Liner Commands

```bash
# Setup, verify, and submit in one go
cd /trinity/home/r112276/Med3Tab-PFN/cluster_scripts && \
chmod +x slurm_experiment1.sh check_setup.sh && \
bash check_setup.sh && \
sbatch slurm_experiment1.sh

# Download results to local machine (run from local terminal)
scp -r r112276@gpu-login.erasmusmc.nl:/trinity/home/r112276/Med3Tab-PFN/results/experiment1 ./

# Quick status check
squeue -u r112276 | grep exp1

# View latest log
tail -50 logs/experiment1_benchmarks_$(ls -t logs/ | grep experiment1_benchmarks_ | head -1 | sed 's/experiment1_benchmarks_//' | sed 's/.log//')
```

## Getting Help

1. **Check logs** - Most issues are explained in error logs
2. **Run setup check** - `bash check_setup.sh`
3. **Test interactively** - Use `srun --pty bash` for debugging
4. **Read full guide** - `EXPERIMENT_GUIDE.md`

---

**Ready to run?**

```bash
cd /trinity/home/r112276/Med3Tab-PFN/cluster_scripts
sbatch slurm_experiment1.sh
```
