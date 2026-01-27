# Troubleshooting Guide

> Solutions for common issues when running Med3Tab-PFN experiments

## Quick Fixes

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch-size` or use `--freeze-encoder` |
| CPU checkpoint on GPU | Load with `map_location='cpu'` |
| OpenMP duplicate runtime | Set `KMP_DUPLICATE_LIB_OK=TRUE` |
| No checkpoint saved | Check target type (`gt3D.float()`) |
| Slow DataLoader | Reduce `--num-workers` to 4 |

---

## GPU/CUDA Issues

### "Attempting to deserialize object on CUDA device but cuda is not available"

**Cause**: Checkpoint saved on GPU, loading on CPU.

**Solution**:
```python
import torch

# Load with CPU mapping
ckpt = torch.load("sam_med3d_turbo.pth", map_location="cpu")

# Save as CPU checkpoint
torch.save(ckpt, "sam_med3d_turbo_cpu.pth")
```

### CUDA Out of Memory

**Solution 1**: Reduce batch size
```bash
python cluster_scripts/experiments/exp3_classifier.py --batch-size 2
```

**Solution 2**: Use frozen encoder
```bash
python cluster_scripts/experiments/exp3_classifier.py --freeze-encoder
```

**Solution 3**: Reduce image size
```bash
python cluster_scripts/experiments/exp1_benchmarks.py --img-size 96
```

---

## OpenMP / Library Conflicts

### "libomp.dll" and "libiomp5md.dll" conflict

**Cause**: Multiple OpenMP runtime libraries loaded.

**Solution**: Set environment variable before running:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

Or in shell:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac
set KMP_DUPLICATE_LIB_OK=TRUE     # Windows
```

---

## DataLoader Issues

### "This DataLoader will create 24 worker processes..."

**Symptom**: Slow or stalled training on CPU.

**Solution**: Reduce workers
```bash
python cluster_scripts/experiments/exp3_classifier.py --num-workers 4
```

### Workers timing out

**Solution**: Increase timeout or reduce workers:
```python
DataLoader(..., num_workers=2, timeout=60)
```

---

## TabPFN / HuggingFace Issues

### "Failed to download TabPFN v2.5 model" or "HuggingFace authentication error"

**Cause**: TabPFN v2.5+ is a gated model on HuggingFace requiring authentication.

**Solution 1**: Authenticate with HuggingFace
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login (you'll be prompted for your token)
huggingface-cli login
# Or: hf auth login
```

**Get your HuggingFace token**:
1. Visit https://huggingface.co/settings/tokens
2. Click "New token" → Select "Read" access
3. Copy the token and paste when prompted

**Accept model terms**:
1. Visit https://huggingface.co/Prior-Labs/tabpfn_2_5
2. Click "Agree and access repository"

**Solution 2**: Use TabPFN v1.x (no authentication needed)
```bash
# Downgrade to v1.x
pip install "tabpfn<2.0"
```

Update `med3pipe/requirements.txt`:
```
tabpfn<2.0  # Use v1.x to avoid gated model authentication
```

**Solution 3**: Set HF_TOKEN environment variable
```bash
# Export token before running
export HF_TOKEN="hf_your_token_here"

# Or add to SLURM script before Python command
```

---

## Checkpoint Issues

### No checkpoint saved / crash during epoch

**Common causes**:
1. Loss expects float targets
2. Checkpoint device mismatch
3. Too many workers

**Fixes**:

1. Ensure float masks:
```python
loss = criterion(outputs, gt3D.float())
```

2. Load checkpoint correctly:
```python
ckpt = torch.load(path, map_location=device)
```

3. Reduce workers:
```bash
--num-workers 4
```

### Checkpoint not found

**Expected location**:
```
sam-med3d/ckpt/sam_med3d_turbo.pth
```

**Verify**:
```bash
bash cluster_scripts/utils/quick_weight_check.sh
```

---

## Path Issues

### SAM-Med3D root not found

**Expected structure**:
```
project_root/
├── sam-med3d/
│   ├── ckpt/
│   │   └── sam_med3d_turbo.pth
│   ├── data/
│   └── features/
└── med3pipe/
```

**Override path**:
```bash
--sam3d-root /path/to/sam-med3d
```

### Dataset not found

**Check config**:
```bash
python -c "import yaml; print(yaml.safe_load(open('configs/datasets.yaml')))"
```

**Common fixes**:
- Use absolute paths for cluster
- Verify `dataset_root` exists
- Check `sheet_csv` location

---

## Autocast Warning

### "GradScaler is enabled but CUDA is not available"

**Message**: Harmless warning, training continues.

**Meaning**: Autocast disabled automatically on CPU.

---

## Where Are Logs?

### Training logs

```
logs/
├── exp1_*.log        # Experiment 1 logs
├── exp3_*.log        # Experiment 3 logs
└── exp*_error_*.log  # Error logs
```

### Model outputs

```
results/
├── experiment1/
├── classification_head/
│   └── <dataset>/
│       ├── best_model.pth
│       ├── training_log.csv
│       └── metrics.txt
└── three_way_comparison/
```

### SAM-Med3D work directories

```
notebooks/finetuned_pfn_results/<dataset>_finetune_workdir/
├── sam_model_latest.pth
├── sam_model_dice_best.pth
├── Loss.png
└── Dice.png
```

---

## Cluster-Specific Issues

### Job fails immediately

**Check error log**:
```bash
cat logs/exp1_error_*.log
```

### conda not found

**Fix in SLURM script**:
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sammed3d
```

### Module not found

**Check environment activation**:
```bash
which python
pip list | grep torch
```

---

## Still Stuck?

1. Check the [documentation](README.md)
2. Run diagnostics:
   ```bash
   bash cluster_scripts/utils/check_setup.sh
   ```
3. Check experiment-specific docs:
   - [Experiment 3 Troubleshooting](cluster/experiment3/troubleshooting.md)
   - [Cluster Quick Start](cluster/QUICK_START.md)
