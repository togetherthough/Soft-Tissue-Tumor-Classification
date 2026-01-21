# Experiment 3 Troubleshooting Guide

## Common Errors and Solutions

### Error: "invalid load key, 'E'."

**Cause**: Corrupted or incorrectly downloaded SAM-Med3D checkpoint file

**Solution**:
```bash
# On cluster
cd /trinity/home/r112276/Med3Tab-PFN/sam-med3d/ckpt/

# Check file exists and size (should be ~400-500MB)
ls -lh sam_med3d_turbo.pth

# If corrupted or missing, redownload:
rm sam_med3d_turbo.pth
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth

# Verify the download
ls -lh sam_med3d_turbo.pth
file sam_med3d_turbo.pth  # Should show: data (not ASCII text or HTML)
```

### Error: "No rows found after filtering by dataset"

**Cause**: Dataset names in config don't match sheet.csv (case-sensitive!)

**Solution**:

#### Step 1: Check what's in your sheet.csv
```bash
# On cluster
cut -d',' -f1 /data/scratch/r112276/sheet.csv | tail -n +2 | sort | uniq

# Example output:
# CRLM
# Desmoid
# GIST
# Lipo
# Liver
# Melanoma
```

#### Step 2: Compare with config file
```bash
grep "dataset_name:" /trinity/home/r112276/Med3Tab-PFN/configs/datasets_cluster.yaml
```

#### Step 3: Fix mismatches
Common issues:
- **Case sensitivity**: `CRLM` vs `crlm` vs `Crlm`
- **Spelling**: `Desmoid` vs `desmoid`
- **Spaces**: `Liver ` (with space) vs `Liver`

Edit `configs/datasets_cluster.yaml` to match EXACTLY:
```yaml
desmoid:
  labels:
    dataset_name: Desmoid  # Must match sheet.csv EXACTLY
```

#### Automated Check
```bash
# Create and run verification script
cat > /tmp/check_datasets.py << 'EOF'
import pandas as pd
import yaml

df = pd.read_csv('/data/scratch/r112276/sheet.csv')
csv_datasets = sorted(df['Dataset'].unique())
print("Datasets in sheet.csv:")
for ds in csv_datasets:
    count = len(df[df['Dataset'] == ds])
    print(f"  - {ds!r} ({count} samples)")

print("\nDatasets in config:")
with open('/trinity/home/r112276/Med3Tab-PFN/configs/datasets_cluster.yaml') as f:
    cfg = yaml.safe_load(f)
    for key, val in cfg['datasets'].items():
        name = val['labels']['dataset_name']
        print(f"  - {key}: dataset_name={name!r}")
        
print("\n✅ = Match, ❌ = Mismatch")
with open('/trinity/home/r112276/Med3Tab-PFN/configs/datasets_cluster.yaml') as f:
    cfg = yaml.safe_load(f)
    for key, val in cfg['datasets'].items():
        name = val['labels']['dataset_name']
        if name in csv_datasets:
            print(f"  ✅ {key}: '{name}' found in sheet.csv")
        else:
            print(f"  ❌ {key}: '{name}' NOT FOUND in sheet.csv")
            close_matches = [ds for ds in csv_datasets if ds.lower() == name.lower()]
            if close_matches:
                print(f"     Did you mean: {close_matches[0]!r}?")
EOF

python /tmp/check_datasets.py
```

### Error: "CUDA out of memory"

**Solution**: Reduce batch size or image size
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --batch-size 2 \
    --img-size 96 \
    --datasets gist
```

### Error: Config file has duplicate entries

**Solution**: Check and remove duplicates from config
```bash
# Find duplicates
grep -n "^[a-z]*:" configs/datasets_cluster.yaml | sort -t: -k2

# Edit to remove duplicates
nano configs/datasets_cluster.yaml
```

## Pre-Flight Checklist

Before running Experiment 3, verify:

- [ ] Config file has no duplicate entries
- [ ] SAM-Med3D checkpoint exists and is ~400-500MB
- [ ] Checkpoint file is not corrupted (check with `file` command)
- [ ] Dataset names in config EXACTLY match sheet.csv (case-sensitive!)
- [ ] All dataset directories exist in `/data/scratch/r112276/`
- [ ] Conda environment `sammed3d` is activated
- [ ] PyTorch and CUDA are working: `python -c "import torch; print(torch.cuda.is_available())"`

## Diagnostic Commands

### Check Environment
```bash
# Verify Python and packages
conda activate sammed3d
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Check GPU
nvidia-smi
```

### Check Files and Paths
```bash
# Verify checkpoint
ls -lh sam-med3d/ckpt/sam_med3d_turbo.pth

# Verify dataset directories
ls -la /data/scratch/r112276/

# Check sheet.csv
head /data/scratch/r112276/sheet.csv
```

### Check Recent Logs
```bash
# View latest error log
ls -t logs/exp3_error_*.log | head -1 | xargs tail -n 100

# View latest output log
ls -t logs/exp3_*.log | head -1 | xargs tail -n 50
```

## Step-by-Step Debugging Sequence

### 1. Run Diagnostics
```bash
bash cluster_scripts/diagnose_experiment3.sh
```

### 2. Fix Issues Found
Address any errors reported by the diagnostic script.

### 3. Test with Single Dataset (3 epochs)
```bash
# Interactive session
srun --partition=interactive --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
conda activate sammed3d

# Test run
python cluster_scripts/run_experiment3_classification_head.py \
    --datasets gist \
    --epochs 3 \
    --config configs/datasets_cluster.yaml
```

### 4. Check Results
```bash
ls -la results/classification_head/gist/
cat results/classification_head/gist/final_metrics.json
```

### 5. If Successful, Run All Datasets
```bash
exit  # Exit interactive session
sbatch cluster_scripts/slurm_experiment3_quick.sh
```

## Files to Check When Debugging

### 1. Cluster Log Files
```bash
ls -lth /trinity/home/r112276/Med3Tab-PFN/logs/exp3_*.log
tail -n 100 /trinity/home/r112276/Med3Tab-PFN/logs/exp3_JOBID.log
tail -n 100 /trinity/home/r112276/Med3Tab-PFN/logs/exp3_error_JOBID.log
```

### 2. Config Files
```bash
# Check for syntax errors or duplicates
python -c "import yaml; yaml.safe_load(open('configs/datasets_cluster.yaml'))"
```

### 3. Data Files
```bash
# Check embeddings if previously generated
find sam-med3d/features/ -name "*_embedding.pt" -type f

# If corrupted, delete and regenerate
find sam-med3d/features/ -name "*_embedding.pt" -delete
```

### 4. Results Directory
```bash
ls -la results/classification_head/
# Look for error logs or partial results
```

## Performance Optimization

### Speed up Training
```bash
# Use frozen encoder (recommended)
--freeze-encoder

# Increase batch size (if GPU memory allows)
--batch-size 8

# Reduce image size slightly
--img-size 96

# Use fewer workers if CPU is bottleneck
--num-workers 1
```

### Improve Accuracy
```bash
# More epochs
--epochs 15

# Fine-tune full model (slower!)
--fine-tune

# Adjust learning rate
--lr 5e-4

# Reduce regularization
--weight-decay 1e-5
--dropout 0.2
```

## FAQ

**Q: How many epochs should I use?**  
A: Start with 5 for testing, use 10-15 for real experiments.

**Q: Should I use --freeze-encoder or --fine-tune?**  
A: Always start with `--freeze-encoder` (much faster). Only use `--fine-tune` if frozen gives AUC < 0.65.

**Q: Can I run multiple datasets in parallel?**  
A: No, the script runs them sequentially. Submit separate jobs for parallel execution.

**Q: What if I run out of memory?**  
A: Reduce `--batch-size` (try 2 instead of 4) or `--img-size` (try 96 instead of 128).

## Getting Help

If errors persist after trying these solutions:

### Collect Debug Information
```bash
# Run full diagnostic
bash cluster_scripts/diagnose_experiment3.sh > diagnosis.txt 2>&1

# Check recent error logs
tail -n 100 logs/exp3_error_*.log > error_details.txt

# Download to local machine
scp r112276@gpu-login.erasmusmc.nl:/trinity/home/r112276/Med3Tab-PFN/diagnosis.txt ./
scp r112276@gpu-login.erasmusmc.nl:/trinity/home/r112276/Med3Tab-PFN/error_details.txt ./
```

Share these files for further debugging.

## Quick Command Reference

```bash
# Diagnostics
bash cluster_scripts/diagnose_experiment3.sh

# Quick run (5 epochs, all datasets)
sbatch cluster_scripts/slurm_experiment3_quick.sh

# Quick run (5 epochs, single dataset)
sbatch cluster_scripts/slurm_experiment3_quick.sh gist

# Interactive test (3 epochs)
python cluster_scripts/run_experiment3_classification_head.py --datasets gist --epochs 3

# Check logs
tail -f logs/exp3_*.log

# Check errors
tail -f logs/exp3_error_*.log

# Cancel job
scancel JOBID

# Check status
squeue -u $USER
```
