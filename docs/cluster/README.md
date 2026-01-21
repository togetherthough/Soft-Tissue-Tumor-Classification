# Experiment 3: Classification Head on SAM-Med3D

Quick guide for running Experiment 3 to test if SAM-Med3D features are discriminative for your classification task.

## 🎯 Purpose

Before running the full TabPFN/LoCalPFN pipeline, this experiment:
- ✅ Tests if SAM-Med3D features capture useful tumor information
- ✅ Trains a simple linear classifier on frozen encoder features
- ✅ Provides baseline performance metrics (accuracy, AUC)
- ✅ Helps decide if features are good enough to proceed

## 📊 Expected Results Interpretation

| AUC Score | Interpretation | Next Steps |
|-----------|---------------|------------|
| **> 0.70** | 🟢 **Good features** | Proceed with TabPFN/LoCalPFN pipeline |
| **0.60 - 0.70** | 🟡 **Moderate features** | Try fine-tuning or proceed with caution |
| **< 0.60** | 🔴 **Poor features** | Consider different pretrained weights or architecture |

## 🚀 Quick Submit (Cluster)

### Step 1: Edit SLURM Script
```bash
nano cluster_scripts/slurm_experiment3.sh

# Update line 12 with your email
# Update lines 54-55 to activate your conda environment
```

### Step 2: Submit Job
```bash
sbatch cluster_scripts/slurm_experiment3.sh
```

### Step 3: Monitor
```bash
# Check status
squeue -u $USER

# Watch log
tail -f logs/exp3_*.log
```

## ⚙️ Configuration Options

### Basic Usage (Frozen Encoder - Fast)
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --freeze-encoder \
    --epochs 10
```

**Runtime**: ~30 minutes per dataset with frozen encoder

### Fine-Tuning (Full Training - Slow)
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --config configs/datasets_cluster.yaml \
    --fine-tune \
    --epochs 50 \
    --lr 1e-4
```

**Runtime**: ~3-6 hours per dataset with fine-tuning

### Run on Specific Datasets
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --datasets gist \
    --epochs 20
```

### Custom Hyperparameters
```bash
python cluster_scripts/run_experiment3_classification_head.py \
    --epochs 30 \
    --batch-size 8 \
    --lr 5e-4 \
    --weight-decay 1e-5 \
    --dropout 0.5
```

## 📁 Output Structure

After completion, results are saved to `results/classification_head/`:

```
results/classification_head/
├── summary.csv                    # Summary of all datasets
├── gist/
│   ├── best_model.pt             # Best model checkpoint
│   ├── training_history.json     # Loss/accuracy curves
│   ├── final_metrics.json        # Detailed metrics
│   └── predictions.npy           # Model predictions
└── lipo/
    ├── best_model.pt
    ├── training_history.json
    ├── final_metrics.json
    └── predictions.npy
```

### Summary CSV Format
```csv
dataset,category,best_epoch,best_auc,final_accuracy,final_auc,output_dir
gist,gist,5,0.7234,0.6800,0.7234,results/classification_head/gist
lipo,lipo,8,0.6543,0.6521,0.6543,results/classification_head/lipo
```

## 🔧 All Command-Line Arguments

```
--config PATH              Config file (default: configs/datasets.yaml)
--output-dir PATH          Output directory (default: results/classification_head)
--datasets [DS ...]        Specific datasets to run (default: all)
--freeze-encoder           Freeze SAM-Med3D encoder (default: True)
--fine-tune                Fine-tune encoder (overrides --freeze-encoder)
--epochs N                 Number of epochs (default: 10)
--batch-size N             Batch size (default: 4)
--lr FLOAT                 Learning rate (default: 0.001)
--weight-decay FLOAT       Weight decay (default: 0.0001)
--dropout FLOAT            Dropout rate (default: 0.3)
--img-size N               Image size (default: 128)
--num-workers N            Data loader workers (default: 2)
```

## 📝 Recommended Workflow

### 1. Test with Frozen Encoder (Quick Check)
```bash
# Fast test on one dataset (~15-30 min)
python cluster_scripts/run_experiment3_classification_head.py \
    --datasets gist \
    --freeze-encoder \
    --epochs 10
```

**Check results**:
- If AUC > 0.70: ✅ Features are good, proceed to Experiment 1
- If AUC < 0.60: ❌ Try fine-tuning or check data quality

### 2. Fine-Tune If Needed (If AUC 0.60-0.70)
```bash
# Fine-tune to potentially improve
python cluster_scripts/run_experiment3_classification_head.py \
    --datasets gist \
    --fine-tune \
    --epochs 50 \
    --lr 1e-4
```

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
