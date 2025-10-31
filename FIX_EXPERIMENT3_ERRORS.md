# Fixing Your Experiment 3 Errors

## 🔴 Your Current Errors

```
crlm,"invalid load key, 'E'."
desmoid,No rows found after filtering by dataset (if applied)
gist,"invalid load key, 'E'."
lipo,"invalid load key, 'E'."
liver,No rows found after filtering by dataset (if applied)
melanoma,No rows found after filtering by dataset (if applied)
```

---

## ✅ Issues Fixed Already

### **1. Config File Duplicates** ✓
**Status**: FIXED
- Removed duplicate `gist` and `lipo` entries from `configs/datasets_cluster.yaml`
- This was causing the config to be invalid

---

## 🔧 Issues You Need to Fix

### **2. SAM-Med3D Checkpoint Problem** ❌

**Error**: `"invalid load key, 'E'."`

**Cause**: Corrupted or incorrectly downloaded checkpoint file

**Fix**:
```bash
# On cluster, run these commands:
cd /trinity/home/r112276/Med3Tab-PFN/SAM-Med3D-main/SAM-Med3D-main/ckpt/

# Check if file exists and size
ls -lh sam_med3d_turbo.pth

# If file is missing, small (<400MB), or corrupted, redownload:
rm sam_med3d_turbo.pth  # Remove corrupted file
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth

# Verify the download (should be ~400-500 MB)
ls -lh sam_med3d_turbo.pth
```

### **3. Dataset Name Mismatch** ❌

**Error**: `"No rows found after filtering by dataset"`

**Cause**: Dataset names in `configs/datasets_cluster.yaml` don't match `sheet.csv`

**Fix**:

#### Step 1: Check what's in your sheet.csv
```bash
# On cluster:
cut -d',' -f1 /data/scratch/r112276/sheet.csv | tail -n +2 | sort | uniq

# This will show you the EXACT dataset names, for example:
# CRLM
# Desmoid
# GIST
# Lipo
# Liver
# Melanoma
```

#### Step 2: Compare with config file
```bash
# Show what's in your config:
grep "dataset_name:" /trinity/home/r112276/Med3Tab-PFN/configs/datasets_cluster.yaml
```

#### Step 3: Fix mismatches
Common issues:
- **Case sensitivity**: `CRLM` vs `crlm` vs `Crlm`
- **Spelling**: `Desmoid` vs `desmoid`
- **Spaces**: `Liver ` (with space) vs `Liver`

Edit `configs/datasets_cluster.yaml` to match EXACTLY what's in sheet.csv.

Example - if your sheet.csv has `Desmoid` (capital D), make sure config has:
```yaml
desmoid:
  labels:
    dataset_name: Desmoid  # Must match sheet.csv EXACTLY
```

---

## 🚀 Step-by-Step Fix Instructions

### **On the Cluster:**

```bash
# 1. SSH to cluster
ssh r112276@gpu-login.erasmusmc.nl

# 2. Go to project directory
cd /trinity/home/r112276/Med3Tab-PFN

# 3. Run diagnostic script
bash cluster_scripts/diagnose_experiment3.sh

# 4. Check SAM-Med3D checkpoint
cd SAM-Med3D-main/SAM-Med3D-main/ckpt/
ls -lh sam_med3d_turbo.pth

# If corrupted or missing (file size < 400MB):
rm sam_med3d_turbo.pth
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth

# 5. Verify dataset names in sheet.csv
cd /trinity/home/r112276/Med3Tab-PFN
echo "=== Dataset names in sheet.csv ==="
cut -d',' -f1 /data/scratch/r112276/sheet.csv | tail -n +2 | sort | uniq
echo ""
echo "=== Dataset names in config ==="
grep "dataset_name:" configs/datasets_cluster.yaml

# 6. If names don't match EXACTLY, edit config:
nano configs/datasets_cluster.yaml
# Fix the dataset_name values to match sheet.csv

# 7. Test with single dataset (3 epochs, ~10 minutes)
srun --partition=interactive --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
conda activate sammed3d
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --datasets gist \
    --epochs 3

# 8. If test succeeds, run all datasets with 5 epochs
exit  # Exit interactive session
sbatch cluster_scripts/slurm_experiment3_quick.sh
```

---

## 📋 Quick Checklist

Before rerunning Experiment 3:

- [ ] Config file has no duplicate entries ✓ (FIXED)
- [ ] SAM-Med3D checkpoint exists and is ~400-500MB
- [ ] Checkpoint file is not corrupted (redownload if needed)
- [ ] Dataset names in config EXACTLY match sheet.csv (case-sensitive!)
- [ ] All dataset directories exist in `/data/scratch/r112276/`
- [ ] Conda environment `sammed3d` is activated
- [ ] PyTorch and CUDA are working (`python -c "import torch; print(torch.cuda.is_available())"`)

---

## 🎯 Recommended Testing Sequence

### **Test 1: Single dataset, 3 epochs** (~10 min)
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --datasets gist \
    --epochs 3 \
    --config configs/datasets_cluster.yaml
```

**If this fails**: Debug checkpoint and dataset name issues

### **Test 2: Three datasets, 5 epochs** (~40 min)
```bash
sbatch cluster_scripts/slurm_experiment3_quick.sh gist lipo crlm
```

**If this succeeds**: Your setup is correct!

### **Test 3: All datasets, 10 epochs** (~4 hours)
```bash
# Edit slurm_experiment3.sh to use --epochs 10
nano cluster_scripts/slurm_experiment3.sh  # Change line 94
sbatch cluster_scripts/slurm_experiment3.sh
```

---

## 🔍 How to Check If Issues Are Fixed

### **Check 1: Checkpoint is valid**
```bash
cd /trinity/home/r112276/Med3Tab-PFN/SAM-Med3D-main/SAM-Med3D-main/ckpt/
ls -lh sam_med3d_turbo.pth
# Should show: ~400-500 MB

# Check file type
file sam_med3d_turbo.pth
# Should show: data (not ASCII text or HTML)
```

### **Check 2: Dataset names match**
```bash
# Create temporary check script
cat > /tmp/check_datasets.py << 'EOF'
import pandas as pd
import yaml

# Load sheet.csv
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

---

## 💻 If Working Locally (Windows)

Pull updated config:
```powershell
# From your local machine
cd C:\Users\cahel\Desktop\Med3Tab-PFN

# If you have the fixed config, upload it to cluster:
scp configs/datasets_cluster.yaml r112276@gpu-login.erasmusmc.nl:/trinity/home/r112276/Med3Tab-PFN/configs/
```

---

## 📞 Still Having Issues?

### **Collect debug information:**
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

---

## ✅ Success Indicators

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

---

**Quick commands summary:**
```bash
# 1. Fix checkpoint
cd SAM-Med3D-main/SAM-Med3D-main/ckpt/
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth

# 2. Check dataset names match
cut -d',' -f1 /data/scratch/r112276/sheet.csv | tail -n +2 | sort | uniq
grep "dataset_name:" configs/datasets_cluster.yaml

# 3. Test
python cluster_scripts/run_experiment3_classification_head.py --datasets gist --epochs 3

# 4. Run full
sbatch cluster_scripts/slurm_experiment3_quick.sh
```
