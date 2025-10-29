# Cluster Scripts - Files Overview

Quick reference for all cluster-related files and their purposes.

## 📋 Experiment Scripts

### Experiment 1: Full Benchmarks
| File | Type | Purpose |
|------|------|---------|
| `slurm_train_and_test.sh` | SLURM | Submit Experiment 1 to cluster |
| `run_experiment1_benchmarks.py` | Python | Runs all 4 methods (TabPFN, LoCalPFN, DenseNet, ViT) |

**Submit**: `sbatch cluster_scripts/slurm_train_and_test.sh`  
**Runtime**: 4-12 hours for 2 datasets with all methods

### Experiment 3: Feature Quality Test
| File | Type | Purpose |
|------|------|---------|
| `slurm_experiment3.sh` | SLURM | Submit Experiment 3 to cluster |
| `run_experiment3_classification_head.py` | Python | Tests SAM-Med3D feature quality with classification head |

**Submit**: `sbatch cluster_scripts/slurm_experiment3.sh`  
**Runtime**: 30 min - 1 hour per dataset (frozen encoder)

## 📚 Documentation

| File | Purpose |
|------|---------|
| `README.md` | Main guide for all experiments |
| `QUICK_START.md` | 3-step quick start for Experiment 1 |
| `SUBMIT_CHECKLIST.md` | Pre-submission checklist |
| `EXPERIMENT3_GUIDE.md` | Detailed guide for Experiment 3 |
| `FILES_OVERVIEW.md` | This file - overview of all scripts |

## ⚙️ Configuration

| File | Purpose |
|------|---------|
| `../configs/datasets.yaml` | Local machine config (relative paths) |
| `../configs/datasets_cluster.yaml` | Cluster config (absolute paths to `/data/scratch/`) |
| `../configs/README.md` | Explains the two-config system |

## 🔧 Utilities

| File | Purpose |
|------|---------|
| `check_setup.sh` | Verifies cluster environment is ready |
| `COMMANDS.txt` | Collection of useful commands |

## 🎯 Quick Command Reference

### Check Environment
```bash
bash cluster_scripts/check_setup.sh
```

### Submit Jobs
```bash
# Experiment 1: Full benchmarks
sbatch cluster_scripts/slurm_train_and_test.sh

# Experiment 3: Feature quality test
sbatch cluster_scripts/slurm_experiment3.sh
```

### Monitor Jobs
```bash
# Check queue
squeue -u $USER

# View logs
tail -f logs/exp1_*.log
tail -f logs/exp3_*.log

# Check job details
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed
```

### Run Directly (Without SLURM)
```bash
# Experiment 1
python cluster_scripts/run_experiment1_benchmarks.py \
    --config configs/datasets_cluster.yaml \
    --output-dir results/experiment1 \
    --epochs-3d 10

# Experiment 3
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --output-dir results/classification_head \
    --epochs 10 \
    --freeze-encoder
```

## 📂 Results Location

After running experiments:

```
results/
├── experiment1/                           # From Experiment 1
│   └── combined_benchmarks_summary.csv   # Main results: all methods
│
└── classification_head/                   # From Experiment 3
    └── summary.csv                        # Feature quality test results
```

## 🔄 Typical Workflow

### First Time Setup
1. ✅ Transfer code to cluster
2. ✅ Set up data at `/data/scratch/r112276/`
3. ✅ Download SAM-Med3D checkpoint
4. ✅ Edit configs and SLURM scripts (email, conda env)
5. ✅ Run `check_setup.sh` to verify

### Running Experiments

**Option A: Quick Feature Test First (Recommended)**
```bash
# Test if features are good
sbatch cluster_scripts/slurm_experiment3.sh

# Check results (wait ~30 min)
cat results/classification_head/summary.csv

# If AUC > 0.70, run full benchmarks
sbatch cluster_scripts/slurm_train_and_test.sh
```

**Option B: Run Full Benchmarks Directly**
```bash
sbatch cluster_scripts/slurm_train_and_test.sh
```

## 📝 File Relationships

```
User submits SLURM script
        ↓
slurm_train_and_test.sh ──→ run_experiment1_benchmarks.py ──→ med3pipe/pipelines/
        │                            │
        │                            ├─→ run_multi_tabpfn()
        │                            ├─→ run_multi_localpfn()
        │                            ├─→ train_eval_densenet121_3d()
        │                            └─→ train_eval_vit_3d()
        │
        └─→ Uses: configs/datasets_cluster.yaml


User submits SLURM script
        ↓
slurm_experiment3.sh ──→ run_experiment3_classification_head.py ──→ med3pipe/training/
        │                            │
        │                            └─→ run_classification_head_experiment()
        │
        └─→ Uses: configs/datasets_cluster.yaml
```

## 💡 Which Experiment Should I Run?

### Run Experiment 3 If:
- ✅ First time using SAM-Med3D features
- ✅ Want quick validation (~30 min)
- ✅ Unsure if features will work for your data
- ✅ Need a baseline before full pipeline

### Run Experiment 1 If:
- ✅ Already know features are good (from Experiment 3)
- ✅ Want to compare multiple methods
- ✅ Ready for full benchmarking (4-12 hours)
- ✅ Need publication-ready results

### Run Both If:
- ✅ Doing thorough evaluation
- ✅ Writing a paper
- ✅ Want to show feature quality + method comparison

## 🆘 Common Issues

| Issue | File to Edit | What to Change |
|-------|-------------|----------------|
| Wrong data paths | `configs/datasets_cluster.yaml` | Update `dataset_root` to actual paths |
| Email not set | `slurm_*.sh` | Line 12: update email address |
| Conda not activating | `slurm_*.sh` | Lines 54-55: uncomment and fix path |
| Job fails immediately | Check error log | `tail logs/exp*_error_*.log` |
| Can't find checkpoint | Download it | See `check_setup.sh` output |

## 📊 Expected Results

### Experiment 3 Output
```csv
dataset,category,best_epoch,best_auc,final_accuracy,final_auc
gist,gist,7,0.7456,0.7200,0.7456
lipo,lipo,9,0.6543,0.6087,0.6543
```

### Experiment 1 Output
```csv
dataset,method,accuracy,macro_f1,roc_auc
gist,tabpfn,0.6600,0.6599,0.7440
gist,localpfn,0.6600,0.6599,0.6896
gist,densenet121_3d,0.6400,0.6352,0.6821
gist,vit3d,0.6200,0.6143,0.6654
```

## 🔗 External Dependencies

- **SAM-Med3D checkpoint**: Download from HuggingFace
  - URL: https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
  - Save to: `SAM-Med3D-main/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth`
  - Size: ~750MB

- **Python packages**: See `med3pipe/requirements.txt`

---

**Quick Links**:
- [Full README](README.md) - Detailed documentation
- [Experiment 3 Guide](EXPERIMENT3_GUIDE.md) - Feature quality testing
- [Submit Checklist](SUBMIT_CHECKLIST.md) - Pre-flight checks
- [Quick Start](QUICK_START.md) - Get running in 3 steps
