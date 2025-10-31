# ✅ Fixed: Virtual Environment Activation in SLURM Scripts

## 🔴 Problem
Your job failed with **exit code 127** (command not found) because the SLURM scripts were trying to use `python` without activating your virtual environment.

## ✅ Solution Applied

All SLURM scripts have been updated to use **module + venv** instead of conda:

### **Scripts Fixed:**
1. ✅ `slurm_train_and_test_quick.sh`
2. ✅ `slurm_train_and_test.sh`
3. ✅ `slurm_experiment3.sh`
4. ✅ `slurm_experiment3_quick.sh`
5. ✅ `slurm_experiment3_gpu.sh`

### **Added Lines:**
```bash
# Load modules
module load Python/3.10

# Activate virtual environment
source /trinity/home/r112276/Med3Tab-PFN/venv/bin/activate
```

---

## 🚀 Resubmit Your Job

### **For Experiment 1 (Quick Test with 3 epochs):**
```bash
cd /trinity/home/r112276/Med3Tab-PFN
sbatch cluster_scripts/slurm_train_and_test_quick.sh
```

### **Monitor the Job:**
```bash
# Check status
squeue -u r112276

# Watch log (replace JOBID with actual job ID)
tail -f logs/exp1_test_JOBID.log

# Check error log if it fails
tail -f logs/exp1_test_error_JOBID.log
```

---

## 📋 What Each Script Does

| Script | Purpose | Epochs | Time |
|--------|---------|--------|------|
| **slurm_train_and_test_quick.sh** | Experiment 1 quick test | 3 | ~1-2h |
| **slurm_train_and_test.sh** | Experiment 1 full run | 20 | ~8-12h |
| **slurm_experiment3_quick.sh** | Experiment 3 quick test | 5 | ~2h |
| **slurm_experiment3.sh** | Experiment 3 standard | 20 | ~6h |
| **slurm_experiment3_gpu.sh** | Experiment 3 GPU version | 20 | ~6h |

---

## 🔍 Verify Environment First (Optional)

Before submitting, you can verify the environment works:

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

---

## ❓ Still Getting Exit Code 127?

If you still get "command not found" errors, check:

### **1. Module Name Might Be Different**
```bash
# List available Python modules
module avail Python

# You might need to use exact name, e.g.:
# module load Python/3.10.8-GCCcore-12.2.0
```

### **2. Venv Path Might Be Different**
```bash
# Check if venv exists
ls -la /trinity/home/r112276/Med3Tab-PFN/venv/bin/activate

# If in different location, update scripts
```

### **3. Check Your Environment Name**
Based on your prompt `(thesis_peron)`, you might be using a different environment. If so, update the venv path in all scripts.

---

## 🎯 Next Steps

1. **Resubmit the job:**
   ```bash
   sbatch cluster_scripts/slurm_train_and_test_quick.sh
   ```

2. **Check it starts properly:**
   ```bash
   squeue -u r112276
   # Should show RUNNING status
   ```

3. **Watch the log:**
   ```bash
   tail -f logs/exp1_test_JOBID.log
   ```

4. **Look for success indicators:**
   - "Python version: 3.10.x"
   - "PyTorch: x.x.x"
   - "CUDA available: True"
   - "Dataset: gist" (or other datasets)
   - "[1/4] Preparing dataset..."

---

## 📊 Expected Output

Successful start will show:
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

---

## 🆘 If It Still Fails

Collect debug info and share:
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

---

**All scripts are now ready with venv activation! 🎉**
